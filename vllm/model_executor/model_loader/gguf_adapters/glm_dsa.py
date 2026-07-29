# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weights adapter for the ``glm-dsa`` architecture (GLM-5.2 / GLM-5.2-Vision).

The default adapter builds its GGUF->HF name map by instantiating
``AutoModelForCausalLM.from_config`` on the meta device and walking the
resulting state dict through ``gguf.get_tensor_name_map``. Neither half works
here: transformers has no ``glm_moe_dsa`` entry, and ``gguf`` only grew the
``glm-dsa`` arch recently. So this adapter carries the map explicitly.

Three tensor groups need more than a rename:

* **MLA** — GGUF ships the *absorbed* per-head ``attn_k_b`` (H, kv_lora, qk_nope)
  and ``attn_v_b`` (H, v_head, kv_lora). vLLM wants a single ``kv_b_proj`` of
  ``[H*(qk_nope + v_head), kv_lora]``. Getting there transposes ``k_b``, which
  is impossible on quantized blocks, so both are dequantized and the result is
  emitted as BF16. It is ~1B params for GLM-5.2 (~2 GB), which is acceptable.
* **DSA indexer** — vLLM fuses ``wk`` and ``weights_proj`` into one
  ``MergedColumnParallelLinear`` built with ``quant_config=None``, so both
  sources must arrive dequantized as shards 0 and 1.
* **MTP** — the GGUF carries one more block than ``num_hidden_layers`` (the
  nextn/MTP layer). It is dropped unless a speculator asks for it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import gguf.quants
import torch
from vllm.logger import init_logger

from vllm.model_executor.model_loader.gguf_weight_utils import gguf_quant_weights_iterator_multi
from .default import GGUFWeightsAdapter
from vllm.transformers_utils.gguf_utils import gguf_reader

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

# Per-layer tensors that are a plain rename.
_LAYER_RENAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    # dense (first_k_dense_replace) layers
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    # MoE layers
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
    # 3D expert stacks; map_weights unbinds these into per-expert rows.
    "ffn_gate_exps.weight": "mlp.experts.0.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.0.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.0.down_proj.weight",
}

_GLOBAL_RENAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# Indexer wk + weights_proj. vLLM fuses these into one
# ``MergedColumnParallelLinear`` via stacked_params_mapping
# [("wk_weights_proj", "wk", 0), ("wk_weights_proj", "weights_proj", 1)], doing
# ``name.replace(weight_name, param_name)`` itself -- so they must be emitted
# SEPARATELY under the pre-fusion names. Emitting an already-fused
# ``wk_weights_proj`` name gets mangled by that replace into
# ``wk_weights_proj_weights_proj``. Both still have to be dequantized: the fused
# layer is built with ``quant_config=None``.
_INDEXER_HALVES = {
    "indexer.attn_k.weight": "self_attn.indexer.wk.weight",
    "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
}

# MTP / nextn block. vLLM's DeepSeekMTP loader takes names in the *original*
# model's layout and inserts ``.mtp_block.`` itself for anything that is not one
# of enorm / hnorm / eh_proj / shared_head / embed_tokens, so the ordinary
# per-layer map above applies unchanged to the MTP layer and only these extras
# need naming.
_MTP_RENAMES = {
    "nextn.enorm.weight": "enorm.weight",
    "nextn.hnorm.weight": "hnorm.weight",
    "nextn.eh_proj.weight": "eh_proj.weight",
    "nextn.shared_head_norm.weight": "shared_head.norm.weight",
}

# Modules this adapter emits dequantized, so the GGUF linear method must not
# claim them. Both the pre-fusion names and the fused destination are listed
# because the consumer does a substring match against the layer prefix.
_UNQUANTIZED_SUFFIXES = (
    "kv_b_proj",
    "indexer.wk",
    "indexer.weights_proj",
    "indexer.wk_weights_proj",
)

# Vision half. The mmproj GGUF is the preferred source: it keeps the deployment
# to the two GGUF files and needs no other checkpoint present at serve time.
# GGUF `ne` is reverse-ordered relative to torch, and the reader already hands
# back torch-ordered arrays, so these are pure renames -- no transposes
# (v.patch_embd ne=[14,14,3,1152] -> (1152,3,14,14), etc).
#
# EXCEPTION: attn_qkv is NOT a pure rename. llama.cpp's converter permutes Q and
# K into "split" 2D-RoPE format (see tools/mtmd/models/kimik25.cpp: "Kimi-K2.5
# uses interleaved 2D RoPE pattern natively, but Q / K are permuted during
# conversion to use split format"). clip's build_rope_2d then treats each head
# as [x half | y half]: dims 0..D/2-1 carry the width component, D/2..D-1 the
# height component, each rotated as adjacent pairs.
#
# vLLM's MoonViT keeps the native interleaved form: Rope2DPosEmbRepeated packs
# freqs as [x0,y0,x1,y1,...] and apply_rope pairs adjacent elements, so dims
# (4j, 4j+1) are x_j and (4j+2, 4j+3) are y_j.
#
# Loading the GGUF tensor unconverted leaves the tower doing correct math on
# wrongly arranged weights: coarse image structure survives, text and fine
# detail are destroyed. See _qk_split_to_interleaved.
_VISION_BLK_RENAMES = {
    "attn_qkv": "wqkv",
    "attn_out": "wo",
    "ffn_up": "mlp.fc0",
    "ffn_down": "mlp.fc1",
    "ln1": "norm0",
    "ln2": "norm1",
}
_VISION_RENAMES = {
    "v.patch_embd": "vision_tower.patch_embed.proj",
    "v.position_embd": "vision_tower.patch_embed.pos_emb",
    "v.post_ln": "vision_tower.encoder.final_layernorm",
    "mm.input_norm": "mm_projector.pre_norm",
    "mm.1": "mm_projector.linear_1",
    "mm.2": "mm_projector.linear_2",
}
_V_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.([a-z0-9_]+)\.(weight|bias)$")

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


class GlmDsaGGUFAdapter(GGUFWeightsAdapter):
    """Adapter for GLM-5.2's ``glm-dsa`` GGUF layout."""

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        text_config = config.get_text_config()
        model_type = getattr(text_config, "model_type", None)
        # "deepseek_mtp" is what SpeculativeConfig.hf_config_override rewrites
        # glm_moe_dsa to when building the MTP draft; the GGUF nextn layout is
        # the same for both, and the default adapter cannot handle either.
        return model_type in ("glm_moe_dsa", "deepseek_mtp")

    @staticmethod
    def _is_mtp_model(model_config: ModelConfig) -> bool:
        text_config = model_config.hf_config.get_text_config()
        return getattr(text_config, "model_type", None) == "deepseek_mtp"

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        # The GGUF metadata is a subset of the HF config we already resolve via
        # --hf-config-path, and the default patcher assumes a gguf arch entry
        # that does not describe the MLA/indexer tensors. Leave it alone.
        del model_path
        return hf_config

    def _text_config(self, model_config: ModelConfig):
        return model_config.hf_config.get_text_config()

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        if self._is_mtp_model(model_config):
            return self._build_mtp_name_map(model_config)
        text_config = self._text_config(model_config)
        num_layers = text_config.num_hidden_layers
        prefix = "language_model." if self._is_multimodal(model_config) else ""

        name_map: dict[str, str] = {
            gguf: prefix + hf for gguf, hf in _GLOBAL_RENAMES.items()
        }
        for idx in range(num_layers):
            base = f"{prefix}model.layers.{idx}."
            for suffix, hf_suffix in _LAYER_RENAMES.items():
                name_map[f"blk.{idx}.{suffix}"] = base + hf_suffix
            # kv_b_proj is assembled from two GGUF tensors; both map to the same
            # destination and are combined in prepare_weights.
            for suffix in ("attn_k_b.weight", "attn_v_b.weight"):
                name_map[f"blk.{idx}.{suffix}"] = base + "self_attn.kv_b_proj.weight"
            for suffix, hf_suffix in _INDEXER_HALVES.items():
                name_map[f"blk.{idx}.{suffix}"] = base + hf_suffix
        return name_map

    def _build_mtp_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        """Name map for the MTP draft built from the same GGUF.

        The draft is one block, indexed at ``num_hidden_layers`` (78 for
        GLM-5.2), and shares the target's embedding and output head.
        """
        text_config = self._text_config(model_config)
        layer = text_config.num_hidden_layers
        base = f"model.layers.{layer}."
        name_map = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output.weight": base + "shared_head.head.weight",
        }
        for suffix, hf_suffix in _MTP_RENAMES.items():
            name_map[f"blk.{layer}.{suffix}"] = base + hf_suffix
        for suffix, hf_suffix in _LAYER_RENAMES.items():
            name_map[f"blk.{layer}.{suffix}"] = base + hf_suffix
        for suffix in ("attn_k_b.weight", "attn_v_b.weight"):
            name_map[f"blk.{layer}.{suffix}"] = base + "self_attn.kv_b_proj.weight"
        for suffix, hf_suffix in _INDEXER_HALVES.items():
            name_map[f"blk.{layer}.{suffix}"] = base + hf_suffix
        return name_map

    @staticmethod
    def _is_multimodal(model_config: ModelConfig) -> bool:
        config = model_config.hf_config
        return getattr(config, "vision_config", None) is not None

    def prepare_loading(self, model_path: str, model_config: ModelConfig):
        # model_config.model is the resolved config source (--hf-config-path):
        # the safetensors checkpoint the vision half is read from.
        self._hf_path = model_config.model
        # Needed to convert the mmproj QKV layout (see _qk_split_to_interleaved).
        self._hf_config = model_config.hf_config
        spec = super().prepare_loading(model_path, model_config)
        # kv_b_proj and the fused indexer projection are emitted dequantized, so
        # they must not be treated as GGUF-quantized modules.
        text_config = self._text_config(model_config)
        if self._is_mtp_model(model_config):
            base = f"model.layers.{text_config.num_hidden_layers}.self_attn."
            spec.unquantized_modules.extend(
                [base + n for n in _UNQUANTIZED_SUFFIXES]
            )
            return spec
        prefix = "language_model." if self._is_multimodal(model_config) else ""
        for idx in range(text_config.num_hidden_layers):
            base = f"{prefix}model.layers.{idx}.self_attn."
            spec.unquantized_modules.extend(
                [base + n for n in _UNQUANTIZED_SUFFIXES]
            )
        return spec

    def update_tie_word_embeddings(self, model_path, hf_config, gguf_to_hf_name_map):
        # GLM-5.2 always ships a separate output.weight.
        hf_config.update({"tie_word_embeddings": False})

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        text_config = self._text_config(model_config)
        is_mtp = self._is_mtp_model(model_config)
        # For the target, blocks >= num_hidden_layers are the MTP block and are
        # skipped; for the draft it is the only block that matters.
        num_layers = text_config.num_hidden_layers
        qk_nope = text_config.qk_nope_head_dim
        v_head = text_config.v_head_dim
        kv_lora = text_config.kv_lora_rank

        # Per layer: buffers for the two halves of kv_b_proj and of the fused
        # indexer projection, flushed as soon as both halves have arrived.
        kv_parts: dict[int, dict[str, torch.Tensor]] = {}
        name_map = self.load_spec.gguf_to_hf_name_map

        # The special tensors feed layers vLLM builds unquantized, so they have
        # to be dequantized here rather than handed over as packed blocks. The
        # shared iterator only yields packed data, so walk the readers directly
        # and delegate everything ordinary back to it.
        special = set(_INDEXER_HALVES) | {
            "attn_k_b.weight",
            "attn_v_b.weight",
            "nextn.eh_proj.weight",
        }
        deferred: set[str] = set()
        for gguf_file in self.load_spec.weights_source:
            reader = gguf_reader(gguf_file)
            for tensor in reader.tensors:
                match = _BLK_RE.match(tensor.name)
                if match is None:
                    continue
                layer, suffix = int(match.group(1)), match.group(2)
                if suffix not in special:
                    continue
                deferred.add(tensor.name)
                if (layer >= num_layers) != is_mtp:
                    continue  # target skips the MTP block; the draft keeps only it
                hf_name = name_map[tensor.name]
                value = torch.from_numpy(
                    gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                )
                if suffix in _INDEXER_HALVES:
                    # vLLM's loader fuses these two into wk_weights_proj itself.
                    yield hf_name, value.to(torch.bfloat16)
                elif suffix in ("attn_k_b.weight", "attn_v_b.weight"):
                    slot = kv_parts.setdefault(layer, {})
                    slot[suffix] = value
                    if len(slot) == 2:
                        yield (
                            hf_name,
                            self._assemble_kv_b(
                                slot.pop("attn_k_b.weight"),
                                slot.pop("attn_v_b.weight"),
                                qk_nope,
                                v_head,
                                kv_lora,
                            ),
                        )
                        kv_parts.pop(layer, None)
                else:
                    yield hf_name, value.to(torch.bfloat16)

        if self._is_multimodal(model_config) and not is_mtp:
            yield from self._vision_weights()

        ordinary = {
            gguf_name: hf_name
            for gguf_name, hf_name in name_map.items()
            if gguf_name not in deferred
            and self._is_mtp(gguf_name, num_layers) == is_mtp
        }
        yield from self.map_weights(
            gguf_quant_weights_iterator_multi(
                self.load_spec.weights_source, ordinary
            )
        )

        for layer, slot in kv_parts.items():
            raise RuntimeError(
                f"layer {layer}: incomplete MLA kv_b_proj, got {sorted(slot)}"
            )

    def _vision_weights(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield the tower + projector from the mmproj GGUF.

        The mmproj carries all 335 vision tensors (329 tower + 6 projector)
        and gguf-py hands back torch-order arrays, so this is a pure rename.
        There is no safetensors fallback: the mmproj is where the vision half
        lives, and the unquantized checkpoint is not a dependency of this repo.
        """
        yield from self._vision_weights_gguf(self._find_mmproj())

    def _find_mmproj(self) -> str:
        """Locate the mmproj GGUF beside the text GGUF or one directory up.

        Unsloth-style layouts put the quant shards in a subdirectory and the
        shared mmproj at the repo root. ``VLLM_GGUF_MMPROJ`` overrides.
        """
        import glob
        import os

        override = os.environ.get("VLLM_GGUF_MMPROJ")
        if override:
            if not os.path.isfile(override):
                raise RuntimeError(f"VLLM_GGUF_MMPROJ does not exist: {override}")
            return override
        text_dir = os.path.dirname(os.path.abspath(self.load_spec.weights_source[0]))
        for d in (text_dir, os.path.dirname(text_dir)):
            hits = sorted(glob.glob(os.path.join(d, "mmproj*.gguf")))
            if hits:
                return hits[0]
        raise RuntimeError(
            f"no mmproj*.gguf beside {text_dir} or its parent; the vision half "
            "lives there and there is no other source"
        )

    @staticmethod
    def _qk_split_to_interleaved(
        value: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """Convert a fused QKV tensor from llama.cpp split 2D-RoPE to interleaved.

        Applies to Q and K only; V carries no rotary and is left alone. Per head
        of size D (D % 4 == 0), with H = D // 2:

            interleaved[4j + 0] <- split[2j]          (x, real)
            interleaved[4j + 1] <- split[2j + 1]      (x, imag)
            interleaved[4j + 2] <- split[H + 2j]      (y, real)
            interleaved[4j + 3] <- split[H + 2j + 1]  (y, imag)

        Verified against ``vision_tower.safetensors`` for blocks 0/5/13/26:
        max abs diff 2.98e-08 after conversion vs up to 8.3e-01 before, and the
        bias converts exactly (0.0).
        """
        rows = value.shape[0]
        if rows % 3 != 0:
            raise RuntimeError(f"fused qkv first dim {rows} is not divisible by 3")
        part = rows // 3
        head_dim = part // num_heads
        if head_dim % 4 != 0:
            raise RuntimeError(
                f"head_dim {head_dim} must be divisible by 4 for 2D RoPE"
            )
        half = head_dim // 2
        idx = torch.empty(head_dim, dtype=torch.long)
        for j in range(half // 2):
            idx[4 * j + 0] = 2 * j
            idx[4 * j + 1] = 2 * j + 1
            idx[4 * j + 2] = half + 2 * j
            idx[4 * j + 3] = half + 2 * j + 1

        def convert(block: torch.Tensor) -> torch.Tensor:
            tail = block.shape[1:]
            return block.view(num_heads, head_dim, *tail)[:, idx].reshape(
                part, *tail
            )

        q, k, v = value[:part], value[part : 2 * part], value[2 * part :]
        return torch.cat([convert(q), convert(k), v])

    def _vision_num_heads(self) -> int:
        vision_config = getattr(self._hf_config, "vision_config", None)
        num_heads = getattr(vision_config, "num_attention_heads", None)
        if not num_heads:
            raise RuntimeError(
                "vision_config.num_attention_heads is required to convert the "
                "mmproj QKV layout"
            )
        return int(num_heads)

    def _vision_weights_gguf(self, path: str) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield the tower + projector from the mmproj GGUF."""
        logger.info("Loading vision tower from mmproj %s", path)
        num_heads = self._vision_num_heads()
        count = 0
        for tensor in gguf_reader(path).tensors:
            stem, _, suffix = tensor.name.rpartition(".")
            match = _V_BLK_RE.match(tensor.name)
            if match:
                idx, part, suffix = match.groups()
                if part not in _VISION_BLK_RENAMES:
                    raise RuntimeError(f"unmapped mmproj tensor {tensor.name}")
                hf = (
                    f"vision_tower.encoder.blocks.{idx}."
                    f"{_VISION_BLK_RENAMES[part]}.{suffix}"
                )
            elif stem in _VISION_RENAMES:
                hf = f"{_VISION_RENAMES[stem]}.{suffix}"
            else:
                raise RuntimeError(f"unmapped mmproj tensor {tensor.name}")
            value = torch.from_numpy(
                gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            )
            if match and match.group(2) == "attn_qkv":
                value = self._qk_split_to_interleaved(value, num_heads)
            count += 1
            yield hf, value
        logger.info("Loaded %d vision tensors from mmproj", count)

    @staticmethod
    def _is_mtp(gguf_name: str, num_layers: int) -> bool:
        match = _BLK_RE.match(gguf_name)
        return match is not None and int(match.group(1)) >= num_layers

    @staticmethod
    def _assemble_kv_b(
        k_b: torch.Tensor,
        v_b: torch.Tensor,
        qk_nope: int,
        v_head: int,
        kv_lora: int,
    ) -> torch.Tensor:
        """Rebuild ``kv_b_proj`` from the absorbed per-head MLA weights.

        GGUF stores ``attn_k_b`` as ``(H, kv_lora, qk_nope)`` and ``attn_v_b`` as
        ``(H, v_head, kv_lora)``. vLLM's ``kv_b_proj`` is
        ``[H*(qk_nope + v_head), kv_lora]`` with the two halves interleaved per
        head. ``k_b`` therefore needs a transpose, which is why this runs on
        dequantized values and returns BF16.
        """
        k_b = k_b.to(torch.float32)
        v_b = v_b.to(torch.float32)
        heads = k_b.shape[0]
        if k_b.shape[1:] != (kv_lora, qk_nope):
            k_b = k_b.reshape(heads, kv_lora, qk_nope)
        if v_b.shape[1:] != (v_head, kv_lora):
            v_b = v_b.reshape(heads, v_head, kv_lora)
        k_b = k_b.transpose(1, 2).contiguous()  # (H, qk_nope, kv_lora)
        fused = torch.cat([k_b, v_b], dim=1)  # (H, qk_nope + v_head, kv_lora)
        return fused.reshape(heads * (qk_nope + v_head), kv_lora).to(torch.bfloat16)
