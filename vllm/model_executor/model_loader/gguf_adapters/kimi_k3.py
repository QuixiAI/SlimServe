# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for Kimi K3's ``kimi-k3`` GGUF layout.

This file is unusual among the adapters here because its converter emitted two
naming conventions at once. Everything except the routed experts already
carries HF names under a ``language_model.`` prefix -- the multimodal wrapper's
namespace -- so those map to themselves. Only the three expert stacks use
llama.cpp's ``blk.N.ffn_*_exps.weight`` form, and only those need renaming.

The expert stacks are 3D (``[..., ..., num_experts]``); as in the DeepSeek-V4
adapter they are mapped onto the ``experts.0.wN`` slot and the shared
``map_weights`` unbinds them into per-expert rows. Kimi's checkpoint names for
the three projections are ``w1``/``w2``/``w3`` (gate/down/up), which is what
``KimiLinearForCausalLM.load_weights`` asks
``fused_moe_make_expert_params_mapping`` for.
"""

from __future__ import annotations

from collections.abc import Iterable

import regex as re
import torch
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import gguf_architecture, gguf_reader

from .default import GGUFWeightsAdapter

logger = init_logger(__name__)

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# gate -> w1, down -> w2, up -> w3.
_EXPERT_RENAMES = {
    "ffn_gate_exps.weight": "block_sparse_moe.experts.0.w1.weight",
    "ffn_down_exps.weight": "block_sparse_moe.experts.0.w2.weight",
    "ffn_up_exps.weight": "block_sparse_moe.experts.0.w3.weight",
}

_LM_PREFIX = "language_model."

# Vision half, from the sibling mmproj. The tower is the (config-driven)
# Kimi-K2.5 MoonViT, so the module names are that implementation's; K3 only
# differs in what the config asks it for -- rmsnorm, an asymmetric 1536-wide
# QKV and an output-side projector norm.
_VISION_BLOCK_RENAMES = {
    "ln1.weight": "norm0.weight",
    "ln2.weight": "norm1.weight",
    "attn_qkv.weight": "wqkv.weight",
    "attn_out.weight": "wo.weight",
    "ffn_up.weight": "mlp.fc0.weight",
    "ffn_down.weight": "mlp.fc1.weight",
}

_VISION_GLOBAL_RENAMES = {
    "v.patch_embd.weight": "vision_tower.patch_embed.proj.weight",
    "v.position_embd.weight": "vision_tower.patch_embed.pos_emb.weight",
    "v.post_ln.weight": "vision_tower.encoder.final_layernorm.weight",
    # patchmergerv2: two projections and a norm on the output side.
    "mm.1.weight": "mm_projector.linear_1.weight",
    "mm.2.weight": "mm_projector.linear_2.weight",
    "mm.post_norm.weight": "mm_projector.post_norm.weight",
}

_VISION_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.(.+)$")

# lm_head is BF16 in the file and feeds a module built unquantized, so it must
# be dequantized rather than renamed to `.qweight`.
_UNQUANTIZED_MODULES = ("lm_head",)


class KimiK3GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for the Kimi K3 mixed IQ2_XXS/Q2_K release."""

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        return getattr(config, "model_type", None) == "kimi_k3"

    @staticmethod
    def matches_gguf(gguf_path: str) -> bool:
        return gguf_architecture(gguf_path) == "kimi-k3"

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        # The config was already built from this GGUF by GGUFConfigParser; the
        # default patcher assumes a gguf arch entry that does not exist for
        # kimi-k3 and would overwrite good values with defaults.
        del model_path
        return hf_config

    def update_tie_word_embeddings(self, model_path, hf_config, gguf_to_hf_name_map):
        # The release ships a separate lm_head.
        hf_config.update({"tie_word_embeddings": False})

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        num_layers = model_config.hf_config.get_text_config().num_hidden_layers

        present: set[str] = set()
        for gguf_file in self._get_all_gguf_files(model_config.model):
            present.update(t.name for t in gguf_reader(gguf_file).tensors)

        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(present):
            if name.startswith(_LM_PREFIX):
                # Already HF-named; the model's own WeightsMapper takes it
                # from here.
                name_map[name] = name
                continue
            match = _BLK_RE.match(name)
            if match is None:
                unmapped.append(name)
                continue
            layer, suffix = int(match.group(1)), match.group(2)
            hf_suffix = _EXPERT_RENAMES.get(suffix)
            if hf_suffix is None:
                unmapped.append(name)
                continue
            if layer >= num_layers:
                continue
            name_map[name] = f"{_LM_PREFIX}model.layers.{layer}.{hf_suffix}"

        if unmapped:
            logger.warning(
                "kimi-k3 GGUF: %d tensors with no mapping, e.g. %s",
                len(unmapped),
                unmapped[:3],
            )
        logger.info(
            "kimi-k3 GGUF: mapped %d tensors (%d expert stacks)",
            len(name_map),
            sum(1 for v in name_map.values() if ".experts.0." in v),
        )
        return name_map

    def prepare_loading(self, model_path: str, model_config: ModelConfig):
        spec = super().prepare_loading(model_path, model_config)
        spec.unquantized_modules.extend(_UNQUANTIZED_MODULES)
        weight_type_map = self.get_weight_type_map(
            model_path, spec.gguf_to_hf_name_map or {}
        )
        fused_parents = self._unquantized_fused_parents(weight_type_map)
        logger.info(
            "kimi-k3 GGUF: %d fused parents marked unquantized (e.g. %s)",
            len(fused_parents),
            fused_parents[:2],
        )
        spec.unquantized_modules.extend(fused_parents)
        return spec

    @staticmethod
    def _unquantized_fused_parents(weight_type_map: dict[str, str]) -> list[str]:
        """Mark parents whose every child tensor is unquantized.

        `get_unquantized_modules` works off checkpoint names, but the model
        fuses projections the GGUF stores separately -- the KDA block merges
        six (q/k/v/g/f_a/b) into one `in_proj_qkvgfab`. That fused name is in
        no checkpoint, and `is_layer_skipped_gguf` only expands fused names it
        finds in a `packed_modules_mapping`, which Kimi K3 does not declare.
        So the fused layer was built quantized and blew up reaching for
        `.weight` on a module that only had `.qweight`.

        Every attention tensor in this release is BF16 or F32 -- 1158 of them,
        nothing quantized -- so marking the parent is right here. It is
        derived rather than hardcoded: a future build that quantizes attention
        stops matching and the layer goes back to the quantized path.
        """
        quantized: set[str] = set()
        unquantized: set[str] = set()
        for hf_name, weight_type in weight_type_map.items():
            # Strip `.weight`/`.bias`, then the projection name, leaving the
            # block that owns it -- `...layers.5.self_attn` for `q_proj`.
            parent = hf_name.rpartition(".")[0].rpartition(".")[0]
            if not parent:
                continue
            target = unquantized if weight_type in ("F32", "F16", "BF16") else quantized
            target.add(parent)

        # Drop any candidate that is an ancestor of a quantized block.
        # `is_layer_skipped_gguf` matches by substring, so a shallow prefix
        # like `language_model` -- unquantized only because lm_head sits
        # directly under it -- would match every layer in the model and
        # silently unquantize the whole MoE.
        return sorted(
            parent
            for parent in unquantized - quantized
            if not any(q.startswith(f"{parent}.") for q in quantized)
        )

    @staticmethod
    def _find_mmproj(text_gguf: str) -> str:
        """Locate the vision projector beside the text GGUF, or one level up.

        ``VLLM_GGUF_MMPROJ`` overrides. Otherwise prefer the BF16 build: the
        released vision weights are BF16, so it is the identity encoding,
        while the F16 build is a conversion away from them that gains nothing
        (its extra mantissa bits are zero-filled from a BF16 source) and puts
        ~0.2% of weights below F16's smallest normal.
        """
        import glob
        import os

        override = os.environ.get("VLLM_GGUF_MMPROJ")
        if override:
            if not os.path.isfile(override):
                raise RuntimeError(f"VLLM_GGUF_MMPROJ does not exist: {override}")
            return override

        text_dir = os.path.dirname(os.path.abspath(text_gguf))
        for directory in (text_dir, os.path.dirname(text_dir)):
            hits = sorted(glob.glob(os.path.join(directory, "mmproj*.gguf")))
            if not hits:
                continue
            preferred = [h for h in hits if "bf16" in os.path.basename(h).lower()]
            return (preferred or hits)[0]
        raise RuntimeError(
            f"no mmproj*.gguf beside {text_dir} or its parent; Kimi K3 keeps "
            "its vision half there and there is no other source"
        )

    def _vision_name_map(self, mmproj: str) -> dict[str, str]:
        name_map = dict(_VISION_GLOBAL_RENAMES)
        unmapped: list[str] = []
        for tensor in gguf_reader(mmproj).tensors:
            name = tensor.name
            if name in _VISION_GLOBAL_RENAMES:
                continue
            match = _VISION_BLK_RE.match(name)
            suffix = _VISION_BLOCK_RENAMES.get(match.group(2)) if match else None
            if suffix is None:
                unmapped.append(name)
                continue
            block = int(match.group(1))
            name_map[name] = f"vision_tower.encoder.blocks.{block}.{suffix}"
        if unmapped:
            raise RuntimeError(
                f"kimi-k3 mmproj {mmproj}: {len(unmapped)} unmapped tensors, "
                f"e.g. {unmapped[:3]}"
            )
        return name_map

    @staticmethod
    def _qk_split_to_interleaved(value: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Restore K3's native interleaved 2D-RoPE Q/K layout.

        llama.cpp stores the Q and K rows in split ``[x | y]`` order, while
        the MoonViT implementation applies RoPE to native interleaved
        ``[x0, x1, y0, y1, ...]`` groups. V is not rotary and stays unchanged.
        """
        rows = value.shape[0]
        if rows % 3 != 0:
            raise RuntimeError(f"fused qkv first dim {rows} is not divisible by 3")
        part = rows // 3
        head_dim = part // num_heads
        if part % num_heads != 0 or head_dim % 4 != 0:
            raise RuntimeError(
                f"invalid fused qkv layout: part={part}, heads={num_heads}"
            )

        half = head_dim // 2
        index = torch.empty(head_dim, dtype=torch.long)
        for offset in range(half // 2):
            index[4 * offset] = 2 * offset
            index[4 * offset + 1] = 2 * offset + 1
            index[4 * offset + 2] = half + 2 * offset
            index[4 * offset + 3] = half + 2 * offset + 1

        def convert(block: torch.Tensor) -> torch.Tensor:
            tail = block.shape[1:]
            return block.view(num_heads, head_dim, *tail)[:, index].reshape(part, *tail)

        query, key, value_rows = value.split(part, dim=0)
        return torch.cat((convert(query), convert(key), value_rows))

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Text weights from the backbone, vision weights from the mmproj.

        The tower is built unquantized, so its tensors are dequantized here
        rather than handed to the loader under `.qweight` names.
        """
        import gguf

        yield from super().prepare_weights(model_config)

        load_spec = self.load_spec
        assert load_spec is not None
        mmproj = self._find_mmproj(load_spec.weights_source[0])
        name_map = self._vision_name_map(mmproj)
        vision_config = model_config.hf_config.vision_config
        num_heads = int(vision_config.num_attention_heads)
        logger.info("kimi-k3: loading %d vision tensors from %s", len(name_map), mmproj)
        for tensor in gguf_reader(mmproj).tensors:
            hf_name = name_map.get(tensor.name)
            if hf_name is None:
                continue
            value = torch.from_numpy(
                gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            )
            if tensor.name.endswith(".attn_qkv.weight"):
                value = self._qk_split_to_interleaved(value, num_heads)
            yield hf_name, value.to(torch.bfloat16)
