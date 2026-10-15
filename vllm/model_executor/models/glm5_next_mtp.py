# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GLM-5.3-Flash checkpoint MTP adapter.

The checkpoint's next-N layer is an ordinary residual sparse-MLA/MoE block,
not a main-model mHC/KDA block. Reuse the MTP recurrence, shared head and
expert loader, but retain GLM's NoPE pooled indexer and its weight names.
Serving profiles remain non-speculative until end-to-end acceptance.
"""

import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    SharedHead,
)
from .deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2MoE
from .glm5_next import Glm5NextMLAAttention
from .utils import get_spec_layer_idx_from_weight_name, maybe_prefix

logger = init_logger(__name__)


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

    def forward(self, positions, hidden_states, residual=None, output_rows=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        if output_rows is not None:
            # Attention ran every row (it owns the layer's latent and pool
            # caches); only the selected rows feed the experts and the head.
            hidden_states = hidden_states.index_select(0, output_rows)
            residual = residual.index_select(0, output_rows)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
        output_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DeepSeek's layer zeroes the embedding at absolute position 0; GLM's
        # head was trained without that mask (the reference implementation
        # concatenates enorm(embeds) and hnorm(previous) unconditionally).
        assert inputs_embeds is not None
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                dim=-1,
            )
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
            output_rows=output_rows,
        )
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
        # Recycle the post-final-norm hidden into the next draft step;
        # compute_logits applies shared_head (== final norm) to the pre-norm
        # element, so logits and the recycle each get exactly one final norm.
        return hidden_states, self.shared_head(hidden_states)


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
        # A draft-only NVFP4 lm_head (slimserve.nvfp4_swapset --draft-lm-head,
        # SLIMSERVE_NVFP4_DRAFT_LMHEAD): the drafter's logits come from it
        # instead of the target's shared fp8 head; the verifier's head and
        # the output distribution are untouched, only the acceptance rate can
        # move.
        from slimserve.nvfp4_swapset import (
            DRAFT_LMHEAD_ENV,
            DRAFT_LMHEAD_MODULE,
            load_draft_lmhead_manifest,
        )

        self.draft_lm_head: ParallelLMHead | None = None
        model_path = vllm_config.speculative_config.draft_model_config.model
        self._draft_lmhead_manifest = load_draft_lmhead_manifest(model_path)
        if self._draft_lmhead_manifest is None and os.environ.get(DRAFT_LMHEAD_ENV, "0") not in ("", "0"):
            logger.warning(
                "glm5_next_mtp: %s=%s selects a draft lm_head sidecar but %s has none; the drafter "
                "shares the target's head (build it with python -m slimserve.nvfp4_swapset --draft-lm-head)",
                DRAFT_LMHEAD_ENV, os.environ.get(DRAFT_LMHEAD_ENV), model_path,
            )
        if self._draft_lmhead_manifest is not None:
            self.draft_lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, DRAFT_LMHEAD_MODULE),
            )

    def _head(self, mtp_layer):
        return self.draft_lm_head if self.draft_lm_head is not None else mtp_layer.shared_head.head

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor(self._head(mtp_layer), mtp_layer.shared_head(hidden_states))

    def compute_local_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, int]:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_local_logits(
            self._head(mtp_layer), mtp_layer.shared_head(hidden_states)
        )

    def load_draft_lm_head(self) -> set[str]:
        """The sidecar's packed tensors into the draft head's parameters."""
        manifest = self._draft_lmhead_manifest
        if manifest is None or self.draft_lm_head is None:
            return set()
        import os

        from safetensors.torch import load_file

        file = os.path.join(os.path.dirname(manifest["path"]), manifest["file"])
        tensors = load_file(file)
        params = dict(self.draft_lm_head.named_parameters())
        loaded: set[str] = set()
        for name in manifest["tensors"]:
            pname = name.split(".", 1)[1]
            param = params[pname]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensors[name])
            loaded.add(name)
        logger.info(
            "glm5_next_mtp: NVFP4 draft lm_head from %s (the target keeps its own head)", file
        )
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        output_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
            output_rows,
        )

    def set_skip_topk(self, skip: bool) -> None:
        """index_share_for_mtp_iteration: steps 1+ reuse step 0's top-k rows."""
        for layer in self.layers.values():
            layer.mtp_block.self_attn.mla_attn.skip_topk = skip

    def compact_topk_indices(self, row_ids: torch.Tensor) -> None:
        """Gather the top-k rows at ``row_ids`` to the front of the buffer."""
        num_rows = row_ids.numel()
        self.topk_indices_buffer[:num_rows] = self.topk_indices_buffer[row_ids]


def _with_mtp_experts_sidecar(
    weights: Iterable[tuple[str, torch.Tensor]], model_path: str | None
) -> Iterable[tuple[str, torch.Tensor]]:
    """The NVFP4 sidecar of the MTP layer's experts (slimserve.nvfp4_swapset
    --mtp-experts) when SLIMSERVE_NVFP4_MTP_SWAPSET selects one: the
    checkpoint's FP8 expert tensors are consumed and the sidecar's packed
    NVFP4 tensors take their place."""
    import os

    from slimserve.nvfp4_swapset import MTP_ENV, load_mtp_manifest
    from vllm.model_executor.models.glm5_next import iter_with_overrides

    manifest = load_mtp_manifest(model_path)
    if manifest is None:
        if os.environ.get(MTP_ENV, "0") not in ("", "0") and model_path:
            logger.warning(
                "glm5_next_mtp: %s=%s selects an NVFP4 experts sidecar but %s has none; "
                "serving the checkpoint's FP8 experts (build it with "
                "python -m slimserve.nvfp4_swapset --mtp-experts)",
                MTP_ENV, os.environ.get(MTP_ENV), model_path,
            )
        return weights
    from safetensors.torch import load_file

    file = os.path.join(os.path.dirname(manifest["path"]), manifest["file"])
    tensors = load_file(file)
    wanted = set(manifest["tensors"])
    missing = wanted - set(tensors)
    if missing:
        raise ValueError(f"{file}: {len(missing)} manifest tensors missing, e.g. {sorted(missing)[0]}")
    extras = {k: tensors[k] for k in wanted}
    consume: dict[str, torch.Tensor | None] = {name: None for name in manifest["consume"]}
    logger.info(
        "glm5_next_mtp: NVFP4 experts sidecar from %s: layer %d, %d experts",
        file, manifest["layer"], manifest["experts"],
    )
    return iter_with_overrides(weights, consume, extras)


@support_torch_compile
class Glm5NextMTP(DeepSeekMTP):
    # The speculator passes `output_rows` on every forward (the request
    # tails on the draft's prefill step, the identity on decode steps), so
    # the AOT-compiled forward has one specialization.
    selects_output_rows = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = _draft_config(vllm_config)
        self.quant_config = vllm_config.quant_config
        self._model_path = vllm_config.speculative_config.draft_model_config.model
        self.model = Glm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        output_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
            output_rows,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded_indexer = set()
        weights = _with_mtp_experts_sidecar(weights, self._model_path)

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
        loaded |= {"model." + n for n in self.model.load_draft_lm_head()}
        return loaded | loaded_indexer
