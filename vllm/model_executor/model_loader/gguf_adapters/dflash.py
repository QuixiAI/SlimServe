# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF adapter for the standardized DeepSeek-V4 DSpark dflash artifact."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf.quants
import regex as re
import torch

from vllm.model_executor.model_loader.gguf_weight_utils import (
    gguf_quant_weights_iterator_multi,
)
from vllm.transformers_utils.gguf_dflash import (
    BLOCK_COUNT,
    dflash_tensor_specs,
    validate_dflash_reader,
)
from vllm.transformers_utils.gguf_utils import gguf_architecture, gguf_reader

from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import ModelConfig

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_LAYER_RENAMES = {
    "attn_sinks.weight": "attn.attn_sink",
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "ffn_gate_exps.weight": "ffn.experts.0.w1.weight",
    "ffn_up_exps.weight": "ffn.experts.0.w3.weight",
    "ffn_down_exps.weight": "ffn.experts.0.w2.weight",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_scale.weight": "hc_ffn_scale",
}

_GLOBAL_RENAMES = {
    "fc.weight": "mtp.0.main_proj.weight",
    "enc.output_norm.weight": "mtp.0.main_norm.weight",
    "output_norm.weight": "mtp.2.norm.weight",
    "markov_w1.weight": "mtp.2.markov_head.markov_w1.weight",
    "markov_w2.weight": "mtp.2.markov_head.markov_w2.weight",
    "output_hc_fn.weight": "mtp.2.hc_head_fn",
    "output_hc_base.weight": "mtp.2.hc_head_base",
    "output_hc_scale.weight": "mtp.2.hc_head_scale",
    "conf_proj.weight": "mtp.2.confidence_head.proj.weight",
}


class DFlashGGUFAdapter(GGUFWeightsAdapter):
    """Restore the DSpark runtime's ``mtp.*`` checkpoint naming contract."""

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        return "DSparkDraftModel" in getattr(config, "architectures", ())

    @staticmethod
    def matches_gguf(gguf_path: str) -> bool:
        return gguf_architecture(gguf_path) == "dflash"

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        del model_path
        return hf_config

    def update_tie_word_embeddings(self, model_path, hf_config, gguf_to_hf_name_map):
        del model_path, gguf_to_hf_name_map
        hf_config.update({"tie_word_embeddings": False})

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        reader = gguf_reader(model_config.model)
        validate_dflash_reader(reader)

        name_map = dict(_GLOBAL_RENAMES)
        for name in dflash_tensor_specs():
            match = _BLK_RE.match(name)
            if match is None:
                continue
            layer = int(match.group(1))
            suffix = match.group(2)
            name_map[name] = f"mtp.{layer}.{_LAYER_RENAMES[suffix]}"
        return name_map

    def prepare_loading(self, model_path: str, model_config: ModelConfig):
        spec = super().prepare_loading(model_path, model_config)
        unquantized = {
            "model.markov_head.markov_w1",
            "model.markov_head.markov_w2",
            "model.confidence_head.proj",
        }
        first_draft_layer = model_config.hf_config.num_hidden_layers
        for layer in range(BLOCK_COUNT):
            runtime_layer = first_draft_layer + layer
            unquantized.add(f"model.layers.{runtime_layer}.attn.wo_a")
        spec.unquantized_modules.extend(sorted(unquantized))
        return spec

    def transform_weight(self, hf_name: str, weight: torch.Tensor) -> torch.Tensor:
        # The converter intentionally collapses the confidence projection's
        # singleton output dimension in GGUF. ReplicatedLinear keeps it.
        if hf_name == "mtp.2.confidence_head.proj.weight":
            return weight.view(1, -1)
        return weight

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        name_map = self.load_spec.gguf_to_hf_name_map
        assert name_map is not None
        dequant_names = {
            name for name in name_map if name.endswith(".attn_output_a.weight")
        }

        for gguf_file in self.load_spec.weights_source:
            for tensor in gguf_reader(gguf_file).tensors:
                if tensor.name not in dequant_names:
                    continue
                value = torch.from_numpy(
                    gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                )
                yield name_map[tensor.name], value.to(torch.bfloat16)

        ordinary = {
            name: hf for name, hf in name_map.items() if name not in dequant_names
        }
        yield from self.map_weights(
            gguf_quant_weights_iterator_multi(self.load_spec.weights_source, ordinary)
        )
