# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weight adapter for the qwen35 hybrid target (Qwen3.8-27B).

Layout facts (verified against the unsloth UD-Q2_K_XL file, see
perf/qwen38_metal_design.md):

- Full-attention layers ship SPLIT projections; `attn_q` carries the
  attention output gate fused inside (width = heads * head_dim * 2). The
  vendored Qwen3NextAttention declares its qkv q-shard at exactly that
  doubled width and splits gate from q at runtime, so the tensor passes
  through untouched.
- Linear-attention (gated-deltanet) layers ship a fused `attn_qkv`
  (q|k|v) plus a separate `attn_gate` (z) -- a 1:1 match for the vendored
  model's `in_proj_qkv` / `in_proj_z` stacking -- and the ssm_* set.
- `ssm_a` stores the "no-scan" form -exp(A_log) (llama.cpp multiplies it
  in directly); the vLLM GDN kernels compute -exp(A_log) themselves, so
  the adapter converts back: A_log = log(-ssm_a).
- `ssm_conv1d` is (conv_dim, kernel); the GDN module holds
  (conv_dim, 1, kernel).
- `blk.<n_layers>.*` is the MTP/NextN block -- intentionally unmapped
  (DFlash 2 replaces MTP; the weight iterator skips unmapped tensors).
- The UD quant mix includes formats without any Metal kernel (IQ1_M,
  IQ2_S, IQ3_S -- see the design doc's kernel audit); those are
  dequantized to fp16 at load (~+12 GiB, bring-up only) along with the
  Q2_K embedding table (Metal has no generic GGUF dequant-gather
  kernel).
"""

from collections.abc import Iterable

import gguf
import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import gguf_reader

from .base import GGUFLoadSpec
from .default import GGUFWeightsAdapter

logger = init_logger(__name__)

# Formats with no Metal kernel at all (neither qgemv nor the qgemm tile).
# Dequantized to fp16 at load until native decode lands per format. IQ1_S,
# IQ2_XS, IQ3_XXS and IQ4_XS stay quantized: qgemm.metal carries tiles for
# them, routed via METAL_MMQ_QUANT_TYPES.
_DEQUANT_TYPES = frozenset({"IQ1_M", "IQ2_S", "IQ3_S"})

_COMMON_BLK_RENAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_FULL_ATTN_BLK_RENAMES = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}

_LINEAR_ATTN_BLK_RENAMES = {
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}

_TOP_RENAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


class Qwen35GGUFAdapter(GGUFWeightsAdapter):
    """The qwen35 hybrid target GGUF (dense Qwen3.5 family)."""

    def patch_hf_config(self, model_path: str, hf_config):
        # The config was built from this same GGUF; nothing to patch.
        return hf_config

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        text_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        name_map = dict(_TOP_RENAMES)
        for idx, layer_type in enumerate(text_config.layer_types):
            per_layer = dict(_COMMON_BLK_RENAMES)
            if layer_type == "full_attention":
                per_layer.update(_FULL_ATTN_BLK_RENAMES)
            else:
                per_layer.update(_LINEAR_ATTN_BLK_RENAMES)
            for gguf_part, hf_part in per_layer.items():
                name_map[f"blk.{idx}.{gguf_part}"] = f"model.layers.{idx}.{hf_part}"
        # blk.{num_hidden_layers}.* (the MTP/NextN block) stays unmapped on
        # purpose: the weight iterator skips tensors absent from the map.
        return name_map

    def prepare_loading(
        self,
        model_path: str,
        model_config: ModelConfig,
    ) -> GGUFLoadSpec:
        spec = super().prepare_loading(model_path, model_config)
        weight_type_map = self.get_weight_type_map(
            model_path, spec.gguf_to_hf_name_map or {}
        )
        self._dequant_stems = {
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if name.endswith(".weight")
            and (weight_type in _DEQUANT_TYPES or name.endswith("embed_tokens.weight"))
        }
        # The GGUF quant method requires every shard of a fused module to be
        # uniformly quantized or uniformly skipped (is_layer_skipped_gguf).
        # Dequantizing one shard therefore pulls its siblings along.
        fused_groups = (
            ("in_proj_qkv", "in_proj_z"),
            ("in_proj_b", "in_proj_a"),
            ("q_proj", "k_proj", "v_proj"),
            ("gate_proj", "up_proj"),
        )
        sibling_map = {shard: group for group in fused_groups for shard in group}
        for stem in list(self._dequant_stems):
            leaf = stem.rsplit(".", 1)[-1]
            group = sibling_map.get(leaf)
            if group is None:
                continue
            base = stem.removesuffix(leaf)
            self._dequant_stems.update(base + sibling for sibling in group)
        for stem in sorted(self._dequant_stems):
            if stem not in spec.unquantized_modules:
                spec.unquantized_modules.append(stem)
        logger.info(
            "qwen35 GGUF: dequantizing %d modules at load "
            "(no-Metal-kernel formats %s + the embedding table)",
            len(self._dequant_stems),
            sorted(_DEQUANT_TYPES),
        )
        return spec

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        assert self.load_spec is not None
        reverse = {
            hf: g for g, hf in (self.load_spec.gguf_to_hf_name_map or {}).items()
        }
        text_config = model_config.hf_config.get_text_config()
        key_dim = text_config.linear_num_key_heads * text_config.linear_key_head_dim
        value_dim = (
            text_config.linear_num_value_heads * text_config.linear_value_head_dim
        )

        def split_qkv(name: str, weight: torch.Tensor):
            """The fused GDN in_proj_qkv splits only on row boundaries.

            GGUF quantizes each output row independently and the reader hands
            quantized data as (rows, bytes_per_row), so slicing rows is exact
            for quantized and dequantized forms alike. The per-shard names
            land on scalar shard ids in the model's stacked mapper (the fused
            tuple-shard path cannot split quantized bytes).
            """
            stem, leaf = name.rsplit(".", 1)
            if leaf == "qweight_type":
                for shard in ("q", "k", "v"):
                    yield (
                        f"{stem[: -len('in_proj_qkv')]}in_proj_{shard}_shard.{leaf}",
                        weight,
                    )
                return
            assert weight.shape[0] == 2 * key_dim + value_dim, (name, weight.shape)
            splits = torch.split(weight, [key_dim, key_dim, value_dim], dim=0)
            for shard, part in zip(("q", "k", "v"), splits):
                yield (
                    f"{stem[: -len('in_proj_qkv')]}in_proj_{shard}_shard.{leaf}",
                    part,
                )

        for name, weight in super().prepare_weights(model_config):
            stem = None
            if name.endswith(".qweight"):
                stem = name.removesuffix(".qweight")
            elif name.endswith(".qweight_type"):
                stem = name.removesuffix(".qweight_type")
            if stem is not None and stem in self._dequant_stems:
                if name.endswith(".qweight_type"):
                    continue
                gguf_name = reverse[stem + ".weight"]
                name = stem + ".weight"
                weight = self.transform_weight(
                    name, self._dequantized_tensor(gguf_name)
                )
            if name.rsplit(".", 1)[0].endswith("linear_attn.in_proj_qkv"):
                yield from split_qkv(name, weight)
                continue
            yield name, weight

    def transform_weight(self, hf_name: str, weight: torch.Tensor) -> torch.Tensor:
        if hf_name.endswith("linear_attn.conv1d.weight") and weight.dim() == 2:
            # GGUF (conv_dim, kernel) -> module (conv_dim, 1, kernel).
            return weight.unsqueeze(1)
        if hf_name.endswith("linear_attn.A_log"):
            # GGUF stores the no-scan form -exp(A_log); the GDN kernels
            # apply -exp() themselves.
            return torch.log(-weight.to(torch.float32))
        return weight

    def _dequantized_tensor(self, gguf_name: str) -> torch.Tensor:
        assert self.load_spec is not None
        for tensor in gguf_reader(self.load_spec.weights_source[0]).tensors:
            if tensor.name == gguf_name:
                data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                return torch.from_numpy(data).to(torch.float16)
        raise RuntimeError(f"{gguf_name} missing from the target GGUF")
