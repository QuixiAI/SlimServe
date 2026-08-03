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
        return spec

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return super().prepare_weights(model_config)
