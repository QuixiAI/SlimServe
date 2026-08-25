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
import regex as re
import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import gguf_reader

from .base import GGUFLoadSpec
from .default import GGUFWeightsAdapter

logger = init_logger(__name__)

# Formats with no Metal kernel at all (neither qgemv nor the qgemm tile),
# dequantized to fp16 at load. Empty since IQ2_S/IQ3_S/IQ1_M landed native
# Metal decode (dequant.metal + qgemv/qgemm instantiations, routed via
# MMVQ_QUANT_TYPES / METAL_MMQ_QUANT_TYPES); every GGUF format in this model
# now stays quantized. Only embed_tokens still dequantizes (see below).
_DEQUANT_TYPES: frozenset[str] = frozenset()

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

# --- mmproj (vision tower, qwen3vl_merger) -------------------------------
# GGUF names verified against the real mmproj-F16 dump (334 tensors, see
# perf/qwen38_metal_design.md). Module names land under the
# Qwen3_5ForConditionalGeneration tree (models/qwen3_5.py -> qwen3_vl.py),
# the same tree the safetensors checkpoint loads into.

_V_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.(.+?)\.(weight|bias)$")

_VISION_BLK_RENAMES = {
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
}

_VISION_TOP_RENAMES = {
    "v.patch_embd.bias": "visual.patch_embed.proj.bias",
    "v.position_embd.weight": "visual.pos_embed.weight",
    # llama.cpp's conversion maps `visual.merger.norm` to V_POST_NORM: the
    # GGUF's post_ln IS the merger's pre-shuffle LayerNorm.
    "v.post_ln.weight": "visual.merger.norm.weight",
    "v.post_ln.bias": "visual.merger.norm.bias",
    "mm.0.weight": "visual.merger.linear_fc1.weight",
    "mm.0.bias": "visual.merger.linear_fc1.bias",
    "mm.2.weight": "visual.merger.linear_fc2.weight",
    "mm.2.bias": "visual.merger.linear_fc2.bias",
}

# The two temporal conv taps (still images duplicate the frame, so the
# taps are summed by construction); reassembled into the [out, C, T, P, P]
# Conv3d layout the HF checkpoint ships.
_PATCH_EMBD_TAPS = ("v.patch_embd.weight", "v.patch_embd.weight.1")


def _vision_linear_modules(depth: int) -> list[str]:
    """Every linear in the mmproj tower, by its vLLM module path.

    Listed exactly rather than by prefix: `is_layer_skipped_gguf` compares
    a fused module's shard names against this list with the containment test
    reversed (`shard_prefix in module_name`), so only an exact entry matches
    for `attn.qkv`. All of these come from the F16 mmproj, never from the
    quantized text GGUF.
    """
    names = [
        "visual.patch_embed.proj",
        "visual.merger.linear_fc1",
        "visual.merger.linear_fc2",
    ]
    for idx in range(depth):
        names += [
            f"visual.blocks.{idx}.attn.qkv",
            f"visual.blocks.{idx}.attn.proj",
            f"visual.blocks.{idx}.mlp.linear_fc1",
            f"visual.blocks.{idx}.mlp.linear_fc2",
        ]
    return names


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
        # When the mmproj was found the config is the composite
        # Qwen3_5ForConditionalGeneration; the text half lives under
        # `language_model.` (Muse-Glimmer pattern).
        multimodal = getattr(config, "vision_config", None) is not None
        prefix = "language_model." if multimodal else ""
        name_map: dict[str, str] = {}
        for gguf_name, hf_name in _TOP_RENAMES.items():
            name_map[gguf_name] = prefix + hf_name
        for idx, layer_type in enumerate(text_config.layer_types):
            per_layer = dict(_COMMON_BLK_RENAMES)
            if layer_type == "full_attention":
                per_layer.update(_FULL_ATTN_BLK_RENAMES)
            else:
                per_layer.update(_LINEAR_ATTN_BLK_RENAMES)
            for gguf_part, hf_part in per_layer.items():
                name_map[f"blk.{idx}.{gguf_part}"] = (
                    f"{prefix}model.layers.{idx}.{hf_part}"
                )
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
        # The mmproj tower is entirely F16/F32 and is streamed by
        # `_vision_weights` rather than through `gguf_to_hf_name_map`, so the
        # base class's weight-type scan never sees it. Mark it unquantized
        # explicitly: the shared vision modules take a quant_config (unlike a
        # hand-rolled plain-Linear tower) and would otherwise be built
        # expecting GGUF-packed weights.
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        if vision_config is not None:
            for name in _vision_linear_modules(int(vision_config.depth)):
                if name not in spec.unquantized_modules:
                    spec.unquantized_modules.append(name)
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

        if getattr(model_config.hf_config, "vision_config", None) is not None:
            yield from self._vision_weights()

    def _vision_weights(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Stream the mmproj tower dequantized to torch dtypes.

        The file is all F16/F32; the tower runs as plain unquantized
        modules (the Muse-Glimmer `_vision_weights` pattern). Every tensor
        must map; unmapped names hard-raise.
        """
        from vllm.transformers_utils.gguf_native import find_mmproj

        assert self.load_spec is not None
        mmproj = find_mmproj(self.load_spec.weights_source[0])
        if mmproj is None:
            raise RuntimeError(
                "vision_config is set but no mmproj*.gguf was found beside "
                f"{self.load_spec.weights_source[0]}"
            )
        logger.info("Loading Qwen3.5 vision tower from %s", mmproj.name)
        taps: dict[str, torch.Tensor] = {}
        count = 0
        for tensor in gguf_reader(str(mmproj)).tensors:
            value = torch.from_numpy(
                gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            )
            count += 1
            if tensor.name in _PATCH_EMBD_TAPS:
                taps[tensor.name] = value
                continue
            match = _V_BLK_RE.match(tensor.name)
            if match:
                idx, part, suffix = match.groups()
                if part not in _VISION_BLK_RENAMES:
                    raise RuntimeError(f"unmapped mmproj tensor {tensor.name}")
                hf = f"visual.blocks.{idx}.{_VISION_BLK_RENAMES[part]}.{suffix}"
            elif tensor.name in _VISION_TOP_RENAMES:
                hf = _VISION_TOP_RENAMES[tensor.name]
            else:
                raise RuntimeError(f"unmapped mmproj tensor {tensor.name}")
            yield hf, value
        if set(taps) != set(_PATCH_EMBD_TAPS):
            raise RuntimeError(
                f"mmproj patch-embed taps incomplete: found {sorted(taps)}"
            )
        # [out, C, P, P] per tap -> [out, C, T, P, P], which is exactly the
        # HF Conv3d weight llama.cpp's conversion/qwen3vl.py split at dim 2.
        # Restoring the 5-D shape (rather than flattening) is what lets the
        # GGUF artifact load into the same Qwen3_VisionPatchEmbed the
        # safetensors checkpoint uses; Conv3dLayer folds it back to a matmul
        # at runtime because kernel_size == stride.
        fused = torch.stack([taps[name] for name in _PATCH_EMBD_TAPS], dim=2)
        yield "visual.patch_embed.proj.weight", fused
        logger.info("Loaded %d vision tensors from mmproj", count)

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
