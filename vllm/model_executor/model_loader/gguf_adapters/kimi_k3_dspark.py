# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weight mapping for the Kimi K3 Q8_0 DSpark draft."""

from __future__ import annotations

from typing import TYPE_CHECKING

import regex as re

from vllm.model_executor.model_loader.gguf_adapters.default import (
    GGUFWeightsAdapter,
)
from vllm.transformers_utils.gguf_utils import gguf_reader

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import ModelConfig

_BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")
_LAYER_NAMES = {
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
}
_GLOBAL_NAMES = {
    "dflash.dspark.markov.w1": "markov_head.markov_w1.weight",
    "dflash.dspark.markov.w2": "markov_head.markov_w2.weight",
    "dflash.fc.weight": "fc.weight",
    "dflash.hidden_norm.weight": "hidden_norm.weight",
    "output_norm.weight": "norm.weight",
}


class KimiK3DSparkGGUFAdapter(GGUFWeightsAdapter):
    """Map only the standalone Kimi K3 DSpark GGUF into its draft model."""

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        return "Qwen3DSparkModel" in (getattr(config, "architectures", None) or ())

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        del model_path
        return hf_config

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        present = {
            tensor.name
            for path in self._get_all_gguf_files(model_config.model)
            for tensor in gguf_reader(path).tensors
        }
        names = {key: value for key, value in _GLOBAL_NAMES.items() if key in present}
        for name in present:
            match = _BLOCK.match(name)
            if match and (suffix := _LAYER_NAMES.get(match.group(2))):
                names[name] = f"layers.{match.group(1)}.{suffix}"
        return names

    def update_tie_word_embeddings(self, model_path, hf_config, gguf_to_hf_name_map):
        del model_path, hf_config, gguf_to_hf_name_map
