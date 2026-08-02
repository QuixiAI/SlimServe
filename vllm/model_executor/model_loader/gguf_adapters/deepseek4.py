# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weights adapter for the ``deepseek4`` architecture (DeepSeek-V4-Flash).

Same reason as `glm_dsa` for carrying the map explicitly: the default adapter
derives GGUF->HF names by instantiating `AutoModelForCausalLM.from_config` and
walking the state dict through `gguf.get_tensor_name_map`, and transformers has
no `deepseek_v4` entry.

Unlike the GLM adapter, nothing here needs assembling or dequantizing. Every
GGUF tensor is a pure rename onto one model parameter, because the shapes
already match: `gguf-py` hands back torch-ordered arrays, and each fused module
in the model is fed through the pre-fusion shard names that
`stacked_params_mapping` expects rather than as an assembled tensor.

Names are emitted in the ORIGINAL DeepSeek checkpoint layout (`layers.N.*`,
`embed.weight`, `norm.weight`, `head.weight`) rather than vLLM's, because
`DeepseekV4ForCausalLM.load_weights` runs its own `hf_to_vllm_mapper` over
whatever it is given. Emitting `model.layers.*` here would send everything
through that mapper twice.

The three things that are not uniform across layers, all verified against the
0731 release rather than assumed:

* **Hash layers** — layers 0..`num_hash_layers`-1 carry `ffn_gate_tid2eid` and
  no `exp_probs_b`, matching the model setting `gate.e_score_correction_bias`
  to None and allocating `gate.tid2eid` for exactly those layers.
* **Compressor** — absent on layers 0 and 1.
* **Indexer** — present only on even layers from 2 up.

Rather than encode those rules, the map is built from the tensors the file
actually contains: a rule that drifts from the file produces a confusing load
failure, while a missing name is simply never emitted.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf.quants
import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.gguf_weight_utils import (
    gguf_quant_weights_iterator_multi,
)
from vllm.transformers_utils.gguf_utils import gguf_architecture, gguf_reader

from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import ModelConfig

logger = init_logger(__name__)

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Per-layer tensors that are a plain rename. The attention and compressor
# projections are named for the pre-fusion shards in stacked_params_mapping
# (attn.wq_a/attn.wkv -> attn.fused_wqa_wkv, compressor.wkv/compressor.wgate ->
# compressor.fused_wkv_wgate); emitting an already-fused name would be mangled
# by the loader's own `name.replace(weight_name, param_name)`.
_LAYER_RENAMES = {
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    # MLA
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    # Attention sinks: a bare Parameter, and the loader narrows it per TP rank.
    "attn_sinks.weight": "attn.attn_sink",
    # Compressor (layers 2..N-1)
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "attn_compressor_ape.weight": "attn.compressor.ape",
    # DSA indexer (even layers 2..N-1) and its own compressor
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "indexer_compressor_kv.weight": "attn.indexer.compressor.wkv.weight",
    "indexer_compressor_gate.weight": "attn.indexer.compressor.wgate.weight",
    "indexer_compressor_norm.weight": "attn.indexer.compressor.norm.weight",
    "indexer_compressor_ape.weight": "attn.indexer.compressor.ape",
    # MoE routing. `.ffn.gate.bias` is what the model's mapper rewrites to
    # e_score_correction_bias; `tid2eid` is the hash-layer routing table.
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "ffn_gate_tid2eid.weight": "ffn.gate.tid2eid",
    # Shared expert. w1/w3 fuse into gate_up_proj, w2 maps to down_proj.
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    # 3D routed-expert stacks, yielded whole under the expert-0 name: the
    # weight loader's full_load path TP-slices the stack in one strided copy.
    # Unbinding 256 experts per tensor costs tens of thousands of H2D copies.
    "ffn_gate_exps.weight": "ffn.experts.0.w1.weight",
    "ffn_up_exps.weight": "ffn.experts.0.w3.weight",
    "ffn_down_exps.weight": "ffn.experts.0.w2.weight",
    # Hyper-connections: bare Parameters, shapes already torch-ordered.
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_scale.weight": "hc_ffn_scale",
}

_GLOBAL_RENAMES = {
    "token_embd.weight": "embed.weight",
    "output_norm.weight": "norm.weight",
    "output.weight": "head.weight",
    "output_hc_fn.weight": "hc_head_fn",
    "output_hc_base.weight": "hc_head_base",
    "output_hc_scale.weight": "hc_head_scale",
}


# Modules this adapter emits dequantized, so the GGUF linear method must not
# claim them. Matching is a substring test against the module prefix, so the
# bare suffix covers every layer.
#
#  - `attn.wo_a`: the ROCm inv-rope path reads `wo_a.weight` directly to build
#    a per-group einsum operand, so a packed `qweight` is not usable there.
#  - `lm_head`: built unquantized, and the model's `head.weight ->
#    lm_head.weight` suffix rule stops matching once a name becomes `.qweight`.
_UNQUANTIZED_MODULES = ("attn.wo_a", "lm_head")


class Deepseek4GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for DeepSeek-V4-Flash's ``deepseek4`` GGUF layout."""

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        model_type = getattr(config.get_text_config(), "model_type", None)
        return model_type == "deepseek_v4"

    @staticmethod
    def matches_gguf(gguf_path: str) -> bool:
        return gguf_architecture(gguf_path) == "deepseek4"

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        # The config already came from this GGUF via GGUFConfigParser; the
        # default patcher assumes a gguf arch entry that does not describe the
        # MLA, compressor or indexer tensors.
        del model_path
        return hf_config

    def update_tie_word_embeddings(self, model_path, hf_config, gguf_to_hf_name_map):
        # DeepSeek-V4-Flash always ships a separate output.weight.
        hf_config.update({"tie_word_embeddings": False})

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        num_layers = model_config.hf_config.get_text_config().num_hidden_layers

        name_map = dict(_GLOBAL_RENAMES)
        present: set[str] = set()
        for gguf_file in self._get_all_gguf_files(model_config.model):
            present.update(t.name for t in gguf_reader(gguf_file).tensors)

        skipped = 0
        for name in present:
            match = _BLK_RE.match(name)
            if match is None:
                continue
            layer, suffix = int(match.group(1)), match.group(2)
            if layer >= num_layers:
                # Blocks past the configured depth are the nextn/MTP head,
                # which this release does not ship in the backbone file.
                skipped += 1
                continue
            hf_suffix = _LAYER_RENAMES.get(suffix)
            if hf_suffix is None:
                logger.warning("deepseek4 GGUF: no mapping for %s", name)
                continue
            name_map[name] = f"layers.{layer}.{hf_suffix}"

        unmapped = sorted(set(_GLOBAL_RENAMES) - present)
        if unmapped:
            for name in unmapped:
                name_map.pop(name, None)
            logger.warning("deepseek4 GGUF: absent global tensors %s", unmapped)
        if skipped:
            logger.info(
                "deepseek4 GGUF: skipped %d tensors past layer %d",
                skipped,
                num_layers - 1,
            )
        return name_map

    def prepare_loading(self, model_path: str, model_config: ModelConfig):
        spec = super().prepare_loading(model_path, model_config)
        spec.unquantized_modules.extend(_UNQUANTIZED_MODULES)
        return spec

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield every tensor, dequantizing the two that cannot stay packed.

        Both are Q8_0 in the file and both feed modules built unquantized, so
        the ordinary iterator would rename them to `.qweight` and hand the
        loader a parameter that does not exist. Together they are ~3.9 GB in
        bf16 before TP sharding.
        """
        name_map = self.load_spec.gguf_to_hf_name_map
        dequant_names = {
            name
            for name in name_map
            if name == "output.weight" or name.endswith(".attn_output_a.weight")
        }

        for gguf_file in self.load_spec.weights_source:
            for tensor in gguf_reader(gguf_file).tensors:
                if tensor.name not in dequant_names:
                    continue
                value = torch.from_numpy(
                    gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                )
                yield name_map[tensor.name], value.to(torch.bfloat16)

        ordinary = {g: hf for g, hf in name_map.items() if g not in dequant_names}
        yield from self.map_weights(
            gguf_quant_weights_iterator_multi(self.load_spec.weights_source, ordinary)
        )
