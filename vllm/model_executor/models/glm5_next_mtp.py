# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GLM-5.3-Flash checkpoint MTP adapter.

The checkpoint's next-N layer is an ordinary residual sparse-MLA/MoE block,
not a main-model mHC/KDA block. Reuse the MTP recurrence, shared head and
expert loader, but retain GLM's NoPE pooled indexer and its weight names.
Serving profiles remain non-speculative until end-to-end acceptance.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    SharedHead,
)
from .deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2MoE
from .glm5_next import Glm5NextMLAAttention
from .utils import get_spec_layer_idx_from_weight_name, maybe_prefix


def _draft_config(vllm_config):
    # The proposer passes the target VllmConfig to the draft constructor;
    # model_config= at the loader selects the class, not this config field.
    return vllm_config.speculative_config.draft_model_config.hf_config


class Glm5NextMTPBlock(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config, prefix, topk_indices_buffer):
        nn.Module.__init__(self)
        config = _draft_config(vllm_config)
        self.use_sequence_parallel_moe = False
        self.self_attn = Glm5NextMLAAttention(
            config,
            vllm_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )
        self.mlp = DeepseekV2MoE(
            config=config,
            parallel_config=vllm_config.parallel_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions, hidden_states, residual=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Glm5NextMTPLayer(DeepSeekMultiTokenPredictorLayer):
    def __init__(self, vllm_config, prefix, topk_indices_buffer):
        nn.Module.__init__(self)
        self.config = config = _draft_config(vllm_config)
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config, prefix, vllm_config.quant_config)
        self.mtp_block = Glm5NextMTPBlock(vllm_config, prefix, topk_indices_buffer)


class Glm5NextMultiTokenPredictor(DeepSeekMultiTokenPredictor):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = _draft_config(vllm_config)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers != 1:
            raise ValueError("GLM-5.3-Flash MTP currently supports its one-layer head")
        # Exactly match the target's expanded-pool + tail width, not the
        # ordinary DeepSeek index_topk width. The proposer shares this buffer.
        width = (config.index_topk + config.index_kpool - 1 + 31) // 32 * 32
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            width,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        idx = self.mtp_start_layer_idx
        self.layers = nn.ModuleDict(
            {
                str(idx): Glm5NextMTPLayer(
                    vllm_config, f"{prefix}.layers.{idx}", self.topk_indices_buffer
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)


@support_torch_compile
class Glm5NextMTP(DeepSeekMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = _draft_config(vllm_config)
        self.quant_config = vllm_config.quant_config
        self.model = Glm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded_indexer = set()

        def normalized_weights():
            for name, weight in weights:
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model.") :]
                spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
                if spec_layer is None:
                    continue
                if ".self_attn.indexer." in name:
                    # DeepSeek's loader unconditionally maps wk/weights_proj
                    # to its fused indexer projection. GLM keeps these separate
                    # and has learned pool APE/gate parameters of its own.
                    target = self._rewrite_spec_layer_name(spec_layer, name)
                    param = params[target]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, weight)
                    loaded_indexer.add(target)
                else:
                    yield name, weight

        loaded = super().load_weights(normalized_weights())
        return loaded | loaded_indexer
