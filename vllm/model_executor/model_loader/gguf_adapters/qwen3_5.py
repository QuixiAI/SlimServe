# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight adapter for the Qwen3.8-27B ``qwen35`` GGUF layout.

llama.cpp's converter reorders the 48 Gated DeltaNet value heads from HF's
GROUPED order (v-head h reads k-head h // 3) into a TILED order
(``h_tiled = r * 16 + k``) so ggml's modulo broadcast (``vhead % 16``) works.
Everything this fork owns — the FLA Triton kernels, the torch MPS fallbacks,
and the dormant Metal GDN kernel — hard-codes the grouped division broadcast,
so this adapter un-permutes back to grouped order at load: grouped slot
``h`` takes tiled slot ``(h % 3) * 16 + h // 3``. That touches the V rows of
``attn_qkv``, all of ``attn_gate`` (the z gate), ``ssm_beta``/``ssm_alpha``
rows, ``ssm_a``/``ssm_dt.bias`` elements, the V channels of ``ssm_conv1d``,
and the COLUMNS of ``ssm_out``. Row moves are lossless for any row-major
quant; the ssm_out column move shuffles whole quant blocks and is guarded to
formats whose block width divides the 128-wide head dim.

Other conversion inverses applied here:
- ``ssm_a`` stores ``-exp(A_log)``; invert to ``A_log = log(-ssm_a)``.
- Every RMS norm EXCEPT ``ssm_norm`` is stored with +1 pre-added (ggml uses
  plain RMS norm; the model uses zero-centered GemmaRMSNorm), so subtract 1.
  ``ssm_norm`` feeds RMSNormGated, which uses the plain weight: pass raw.
- ``attn_qkv`` + ``attn_gate`` are emitted pre-fused as ``in_proj_qkvz``:
  the model's merged projection loads GGUF shards by single index only, and
  a whole fused tensor splits row-wise (rows are quant blocks) at load.
- ``ssm_beta``/``ssm_alpha`` are Q8_0 at [5120 -> 48]; 48-wide quantized
  GEMMs trip the Metal prefill shape gates, so they are dequantized at load
  and ``in_proj_ba`` is built unquantized (~0.5 MB each).
- ``token_embd`` dequantizes to fp16 (no dequant-gather kernel on Metal;
  same as Muse-Glimmer).

Full-attention ``attn_q`` keeps llama.cpp's layout untouched: it is the HF
q_proj, per-head interleaved [q(256) | gate(256)], exactly what
Qwen3NextAttention's fused-gate split expects.
"""

from __future__ import annotations

from collections.abc import Iterable

import gguf
import torch
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import gguf_architecture

from .default import GGUFWeightsAdapter

logger = init_logger(__name__)

# GGUF per-layer tensor -> HF-style module path under model.layers.N.
_GDN_BLK_RENAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_FULL_ATTN_BLK_RENAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_TOP_RENAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# GGUF norms stored with +1 pre-added (all EXCEPT linear_attn.norm).
_MINUS_ONE_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


class Qwen35GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for the text backbone of the Qwen3.8-27B GGUF release."""

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        return getattr(config, "model_type", None) == "qwen3_5_text"

    @staticmethod
    def matches_gguf(gguf_path: str) -> bool:
        return gguf_architecture(gguf_path) == "qwen35"

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        # The config was already built from this GGUF by GGUFConfigParser.
        del model_path
        return hf_config

    def update_tie_word_embeddings(self, model_path, hf_config, gguf_to_hf_name_map):
        # The release ships a separate Q6_K lm_head.
        hf_config.update({"tie_word_embeddings": False})

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _gdn_geometry(self) -> tuple[int, int, int, int]:
        config = self.config
        num_k_heads = int(config.linear_num_key_heads)
        num_v_heads = int(config.linear_num_value_heads)
        head_v_dim = int(config.linear_value_head_dim)
        key_dim = num_k_heads * int(config.linear_key_head_dim)
        return num_k_heads, num_v_heads, head_v_dim, key_dim

    def _v_head_perm(self) -> torch.Tensor:
        """grouped slot h <- tiled slot (h % ratio) * num_k_heads + h // ratio."""
        num_k_heads, num_v_heads, _, _ = self._gdn_geometry()
        ratio = num_v_heads // num_k_heads
        return torch.tensor(
            [(h % ratio) * num_k_heads + h // ratio for h in range(num_v_heads)],
            dtype=torch.long,
        )

    def _permute_v_rows(self, weight: torch.Tensor, row_offset: int) -> torch.Tensor:
        """Un-permute 48 v-head row groups of head_v_dim rows each, starting
        at row_offset. Rows are independent for any row-major layout, so this
        is valid on raw qweight bytes and on plain tensors alike."""
        _, num_v_heads, head_v_dim, _ = self._gdn_geometry()
        index = torch.arange(weight.shape[0], dtype=torch.long)
        src = row_offset + (
            self._v_head_perm()[:, None] * head_v_dim
            + torch.arange(head_v_dim)[None, :]
        )
        index[row_offset : row_offset + num_v_heads * head_v_dim] = src.reshape(-1)
        return weight[index]

    def _permute_out_proj_columns(
        self, qweight: torch.Tensor, qweight_type: int
    ) -> torch.Tensor:
        """Un-permute the 48 v-head COLUMN groups of ssm_out inside each
        quantized row. A 128-wide head group must cover whole quant blocks."""
        _, num_v_heads, head_v_dim, _ = self._gdn_geometry()
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        if head_v_dim % block_size != 0:
            raise RuntimeError(
                f"ssm_out quant block width {block_size} does not divide the "
                f"{head_v_dim}-wide value head; this quantization of ssm_out "
                "cannot be column-permuted losslessly. Re-quantize ssm_out "
                "to Q8_0/F16/BF16/F32 (the shipped releases override it to "
                "q8_0 for exactly this reason)."
            )
        group_bytes = (head_v_dim // block_size) * type_size
        rows, row_bytes = qweight.shape
        if row_bytes != num_v_heads * group_bytes:
            raise RuntimeError(
                f"ssm_out row is {row_bytes} bytes, expected "
                f"{num_v_heads} x {group_bytes}"
            )
        return (
            qweight.reshape(rows, num_v_heads, group_bytes)
            .index_select(1, self._v_head_perm())
            .reshape(rows, row_bytes)
        )

    # ------------------------------------------------------------------
    # Name map
    # ------------------------------------------------------------------

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config.get_text_config()
        name_map: dict[str, str] = dict(_TOP_RENAMES)
        for idx, layer_type in enumerate(config.layer_types):
            renames = (
                _GDN_BLK_RENAMES
                if layer_type == "linear_attention"
                else _FULL_ATTN_BLK_RENAMES
            )
            for gguf_part, hf_part in renames.items():
                name_map[f"blk.{idx}.{gguf_part}"] = f"model.layers.{idx}.{hf_part}"
        return name_map

    def prepare_loading(self, model_path: str, model_config: ModelConfig):
        spec = super().prepare_loading(model_path, model_config)
        # Modules whose GGUF source is quantized but which must be built
        # unquantized: the embedding gather (dequantized at load, muse
        # precedent) and the 48-wide b/a projection (dequantized here).
        config = model_config.hf_config.get_text_config()
        spec.unquantized_modules.append("model.embed_tokens")
        for idx, layer_type in enumerate(config.layer_types):
            if layer_type == "linear_attention":
                # is_layer_skipped_gguf expands the fused in_proj_ba module
                # into its shard prefixes and requires each to substring-match
                # an entry, so list the shards, not the parent.
                spec.unquantized_modules.append(
                    f"model.layers.{idx}.linear_attn.in_proj_b"
                )
                spec.unquantized_modules.append(
                    f"model.layers.{idx}.linear_attn.in_proj_a"
                )
        return spec

    # ------------------------------------------------------------------
    # Weight stream
    # ------------------------------------------------------------------

    def map_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        num_k_heads, num_v_heads, head_v_dim, key_dim = self._gdn_geometry()
        v_row_offset = 2 * key_dim  # rows [q | k | v] in attn_qkv / conv dim

        # name (without .qweight suffix) -> GGML type, from the type tags the
        # iterator emits ahead of each quantized tensor.
        qtypes: dict[str, int] = {}
        # layer fusion buffers: "...linear_attn" prefix -> {"qkv": t, "z": t}
        pending_qkvz: dict[str, dict[str, torch.Tensor]] = {}

        def fused_ready(prefix: str) -> Iterable[tuple[str, torch.Tensor]]:
            parts = pending_qkvz.pop(prefix)
            qkv_type = qtypes[f"{prefix}.in_proj_qkv"]
            z_type = qtypes[f"{prefix}.in_proj_z"]
            if qkv_type != z_type:
                raise RuntimeError(
                    f"{prefix}: attn_qkv ({qkv_type}) and attn_gate "
                    f"({z_type}) quant types differ; cannot fuse in_proj_qkvz"
                )
            yield f"{prefix}.in_proj_qkvz.qweight_type", torch.tensor(qkv_type)
            yield (
                f"{prefix}.in_proj_qkvz.qweight",
                torch.cat([parts["qkv"], parts["z"]], dim=0),
            )

        for name, weight in weights:
            if name.endswith(".qweight_type"):
                base = name.removesuffix(".qweight_type")
                qtypes[base] = int(weight.item())
                # Type tags for tensors this adapter rewrites are re-emitted
                # (or dropped) alongside the rewritten tensor.
                if (
                    base.endswith(
                        (".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a")
                    )
                    or base == "model.embed_tokens"
                ):
                    continue
                yield name, weight
                continue

            if name.endswith(".in_proj_qkv.qweight"):
                prefix = name.removesuffix(".in_proj_qkv.qweight")
                pending_qkvz.setdefault(prefix, {})["qkv"] = self._permute_v_rows(
                    weight, v_row_offset
                )
                if len(pending_qkvz[prefix]) == 2:
                    yield from fused_ready(prefix)
                continue

            if name.endswith(".in_proj_z.qweight"):
                prefix = name.removesuffix(".in_proj_z.qweight")
                pending_qkvz.setdefault(prefix, {})["z"] = self._permute_v_rows(
                    weight, 0
                )
                if len(pending_qkvz[prefix]) == 2:
                    yield from fused_ready(prefix)
                continue

            if name.endswith((".in_proj_b.qweight", ".in_proj_a.qweight")):
                base = name.removesuffix(".qweight")
                data = gguf.quants.dequantize(weight.numpy(), qtypes[base])
                value = torch.from_numpy(data)[self._v_head_perm()]
                yield f"{base}.weight", value
                continue

            if name.endswith(".linear_attn.A_log"):
                yield name, torch.log(-weight.float())[self._v_head_perm()]
                continue

            if name.endswith(".linear_attn.dt_bias"):
                yield name, weight[self._v_head_perm()]
                continue

            if name.endswith(".linear_attn.conv1d.weight"):
                # [conv_dim, width] F32; permute V channels, restore the
                # [conv_dim, 1, width] depthwise shape the loader expects.
                yield name, self._permute_v_rows(weight, v_row_offset).unsqueeze(1)
                continue

            if name.endswith(".linear_attn.out_proj.qweight"):
                base = name.removesuffix(".qweight")
                yield name, self._permute_out_proj_columns(weight, qtypes[base])
                continue

            if name == "model.embed_tokens.qweight":
                data = gguf.quants.dequantize(
                    weight.numpy(), qtypes["model.embed_tokens"]
                )
                yield (
                    "model.embed_tokens.weight",
                    torch.from_numpy(data).to(torch.float16),
                )
                continue

            # Unquantized (.weight) variants of the permuted GDN tensors, for
            # requants that store them as F16/BF16/F32. The fused-shard
            # constraint above is GGUF-qweight-specific, so these keep their
            # separate names and load through the tuple-shard path.
            if name.endswith(".in_proj_qkv.weight"):
                yield name, self._permute_v_rows(weight, v_row_offset)
                continue
            if name.endswith(".in_proj_z.weight"):
                yield name, self._permute_v_rows(weight, 0)
                continue
            if name.endswith((".in_proj_b.weight", ".in_proj_a.weight")):
                yield name, weight[self._v_head_perm()]
                continue
            if name.endswith(".linear_attn.out_proj.weight"):
                rows = weight.shape[0]
                value = (
                    weight.reshape(rows, num_v_heads, head_v_dim)
                    .index_select(1, self._v_head_perm())
                    .reshape(rows, -1)
                )
                yield name, value
                continue

            if name.endswith(_MINUS_ONE_SUFFIXES) or name == "model.norm.weight":
                # Stored +1 by the converter; the model's GemmaRMSNorm adds
                # the 1 itself. linear_attn.norm (RMSNormGated) is stored raw
                # and falls through untouched.
                yield name, weight.float() - 1.0
                continue

            yield name, weight

        if pending_qkvz:
            raise RuntimeError(
                f"GGUF stream ended with unpaired in_proj tensors: "
                f"{sorted(pending_qkvz)}"
            )
