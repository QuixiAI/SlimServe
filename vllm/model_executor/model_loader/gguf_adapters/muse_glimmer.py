# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight adapter for the Muse-Glimmer-30B GGUF set.

`muse-glimmer` is not in gguf-py's MODEL_ARCH table and the config is built
from GGUF metadata, so the default AutoModel-derived mapping cannot apply;
this adapter carries the map explicitly.

Text tensors stream from the backbone GGUF still quantized (Q4_K/Q5_K/Q6_K)
into the GGUF quant-method linears. Vision tensors stream from the mmproj
GGUF dequantized to torch dtypes: the tower runs as ordinary unquantized
modules, the same call the GLM-5.2 adapter makes for its mmproj.

The drafter file (`dflash` arch, Muse schema) is handled by
`MuseGlimmerDFlashGGUFAdapter` below with the same per-layer map minus the
gate/sandwich tensors, plus the fc/enc fusion tensors.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import regex as re
import torch

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import gguf_reader

from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import ModelConfig

logger = init_logger(__name__)

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
_V_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.(.+?)\.(weight|bias)$")

# GGUF per-layer tensor -> HF-style module path under model.layers.N.
_TEXT_BLK_RENAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_gate.weight": "self_attn.gate_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_TEXT_TOP_RENAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

_VISION_BLK_RENAMES = {
    "ln1": "layer_norm1",
    "ln2": "layer_norm2",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_out": "self_attn.out_proj",
    "ffn_up": "mlp.fc1",
    "ffn_down": "mlp.fc2",
}

_VISION_TOP_RENAMES = {
    "v.patch_embd.weight": "vision_tower.patch_embed.weight",
    # position_embed is a bare nn.Parameter, not a module with a .weight.
    "v.position_embd.weight": "vision_tower.position_embed",
    "v.pre_ln.weight": "vision_tower.pre_layernorm.weight",
    "v.pre_ln.bias": "vision_tower.pre_layernorm.bias",
    "v.post_ln.weight": "vision_tower.post_layernorm.weight",
    "v.post_ln.bias": "vision_tower.post_layernorm.bias",
    "mm.0.weight": "projector.linear_1.weight",
    "mm.1.weight": "projector.linear_2.weight",
    "mm.2.weight": "projector.linear_3.weight",
}


class MuseGlimmerGGUFAdapter(GGUFWeightsAdapter):
    """Backbone (text) plus mmproj (vision) for Muse-Glimmer."""

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        # The config was built from this same GGUF; nothing to patch.
        return hf_config

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        multimodal = getattr(config, "vision_config", None) is not None
        prefix = "language_model." if multimodal else ""
        name_map: dict[str, str] = {}
        for gguf_name, hf_name in _TEXT_TOP_RENAMES.items():
            name_map[gguf_name] = prefix + hf_name
        for idx in range(text_config.num_hidden_layers):
            for gguf_part, hf_part in _TEXT_BLK_RENAMES.items():
                name_map[f"blk.{idx}.{gguf_part}"] = (
                    f"{prefix}model.layers.{idx}.{hf_part}"
                )
        return name_map

    @staticmethod
    def get_unquantized_modules(weight_type_map: dict[str, str]) -> list[str]:
        # The embedding table stays fp16: Metal has no generic GGUF dequant
        # kernel yet (the qgemv/qgemm paths read blocks directly), so the
        # embedding gather dequantizes at load instead. ~2.7 GiB for the
        # 202k x 6656 table; binding the vendored dequant_gather shader is
        # the follow-up that would reclaim it.
        modules = GGUFWeightsAdapter.get_unquantized_modules(weight_type_map)
        for name in weight_type_map:
            if name.endswith("embed_tokens.weight") and name not in modules:
                modules.append(name.removesuffix(".weight"))
        return modules

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in super().prepare_weights(model_config):
            if name.endswith("embed_tokens.qweight_type"):
                continue
            if name.endswith("embed_tokens.qweight"):
                yield (
                    name.removesuffix(".qweight") + ".weight",
                    (self._dequantized_embed_table()),
                )
                continue
            yield name, weight
        if getattr(model_config.hf_config, "vision_config", None) is not None:
            yield from self._vision_weights()

    def _dequantized_embed_table(self) -> torch.Tensor:
        assert self.load_spec is not None
        for tensor in gguf_reader(self.load_spec.weights_source[0]).tensors:
            if tensor.name == "token_embd.weight":
                data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                return torch.from_numpy(data).to(torch.float16)
        raise RuntimeError("token_embd.weight missing from the backbone GGUF")

    def _vision_weights(self) -> Iterable[tuple[str, torch.Tensor]]:
        from vllm.transformers_utils.gguf_native import find_mmproj

        assert self.load_spec is not None
        mmproj = find_mmproj(self.load_spec.weights_source[0])
        if mmproj is None:
            raise RuntimeError(
                "vision_config is set but no mmproj*.gguf was found beside "
                f"{self.load_spec.weights_source[0]}"
            )
        logger.info("Loading Muse-Glimmer vision tower from %s", mmproj.name)
        count = 0
        for tensor in gguf_reader(str(mmproj)).tensors:
            match = _V_BLK_RE.match(tensor.name)
            if match:
                idx, part, suffix = match.groups()
                if part not in _VISION_BLK_RENAMES:
                    raise RuntimeError(f"unmapped mmproj tensor {tensor.name}")
                hf = f"vision_tower.layers.{idx}.{_VISION_BLK_RENAMES[part]}.{suffix}"
            elif tensor.name in _VISION_TOP_RENAMES:
                hf = _VISION_TOP_RENAMES[tensor.name]
            else:
                raise RuntimeError(f"unmapped mmproj tensor {tensor.name}")
            value = torch.from_numpy(
                gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            )
            count += 1
            yield hf, value
        logger.info("Loaded %d vision tensors from mmproj", count)


# Drafter: 5 layers, no gate / sandwich norms, fc + enc fusion tensors.
# `enc.output_norm` normalizes the fc-fused target context; that role is
# DFlashQwen3Model's `hidden_norm`.
_DFLASH_TOP_RENAMES = {
    "fc.weight": "model.fc.weight",
    "enc.output_norm.weight": "model.hidden_norm.weight",
    "output_norm.weight": "model.norm.weight",
}

_DFLASH_BLK_KEYS = (
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)


class MuseGlimmerDFlashGGUFAdapter(GGUFWeightsAdapter):
    """The Muse-Glimmer DFlash drafter GGUF (dense-GQA `dflash` schema)."""

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        return hf_config

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        name_map = dict(_DFLASH_TOP_RENAMES)
        for idx in range(config.num_hidden_layers):
            for key in _DFLASH_BLK_KEYS:
                hf_part = _TEXT_BLK_RENAMES[key]
                if key == "ffn_norm.weight":
                    # The drafter runs on DFlashQwen3DecoderLayer, whose
                    # pre-MLP norm keeps Qwen naming.
                    hf_part = "post_attention_layernorm.weight"
                name_map[f"blk.{idx}.{key}"] = f"model.layers.{idx}.{hf_part}"
        return name_map
