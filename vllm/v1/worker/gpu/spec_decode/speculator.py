# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.quixicore.ops import quixicore_ops
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample import topk_sample
from vllm.v1.worker.gpu.sample.gumbel import (
    DRAFT_NOISE_SALT,
    apply_temperature,
    gumbel_sample,
)
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def mask_draft_logits(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    native: bool,
) -> torch.Tensor:
    """The draft's logits tempered and cut to the request's top-k / top-p the
    way the verifier cuts the target's: a fresh FP32 row set, -inf outside the
    cutoff. `top_k` / `top_p` are the per-request states ([max_num_reqs]), or
    None where no request in the batch sets them; `native` says every row's
    top-k fits the native cutoff kernel. Padded rows (index -1) borrow row 0's
    state; their draw is discarded downstream."""
    rows = idx_mapping.clamp_min(0)
    logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    apply_temperature(logits, rows, temperature)
    gather = rows.to(torch.int64)
    k = None if top_k is None else top_k[gather]
    p = None if top_p is None else top_p[gather]
    if (
        native
        and k is not None
        and topk_sample.mask_enabled()
        and logits.shape[1] >= topk_sample.MIN_VOCAB
    ):
        topk_sample.mask(logits, k, p)
        return logits
    return apply_top_k_top_p(logits, k, p)


class BaseSpeculator(ABC):
    # The verifier's per-request sampling states; a drafter that draws under
    # the request's top-k / top-p reads them here (bound by the runner).
    sampling_states: SamplingStates | None = None

    def bind_sampling_states(self, states: SamplingStates) -> None:
        self.sampling_states = states

    @abstractmethod
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    @abstractmethod
    def capture(self) -> None:
        pass

    @abstractmethod
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        draft_grammar=None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        pass


class DraftModelSpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_max_seq_len = self.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        # Widen for HC-multiplexed residuals (e.g. DeepSeek V4 feeds the MTP
        # draft the target's pre-hc_head (T, hc_mult * hidden_size) residual).
        # Non-HC models default to hc_mult=1 and are unaffected.
        hc_mult = getattr(self.draft_model_config.hf_config, "hc_mult", 1)
        self.hidden_size = self.hidden_size * hc_mult
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel
        self.use_local_argmax_reduction = (
            self.speculative_config.use_local_argmax_reduction
        )

        # DP configuration. Replicated-MoE DP has no cross-replica collective:
        # the drafter dispatches its batch descriptor locally too, otherwise
        # the active replica blocks in the drafter's DP all-reduce while the
        # idle replica (which no longer dummy-steps) never joins it.
        parallel_config = vllm_config.parallel_config
        self.dp_size = (
            1
            if parallel_config.data_parallel_replicate_moe
            else parallel_config.data_parallel_size
        )
        self.dp_rank = parallel_config.data_parallel_rank

        self.eplb_state: EplbState | None = None

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        # draft_top_k_top_p: the batch's cutoff state for the current propose
        # (native cutoff usable, any top-k, any top-p); None when the draw is
        # temperature-only.
        self.draft_top_k_top_p = self.speculative_config.draft_top_k_top_p
        self._draft_mask: tuple[bool, bool, bool] | None = None
        # F5a: the candidate draft sampler. When every request's top-k fits
        # the native cutoff, the draft token is drawn from the top-K of each
        # rank's vocab shard gathered across TP ([T, tp x K] instead of the
        # [T, vocab] logits all-gather) with the same cut, processed row and
        # (seed, pos, token)-keyed noise as the full path. Opt-in
        # (QC_DRAFT_CANDIDATES=1): measured level on the rtx6000 record
        # (2026-09-19, notebook "Phase 9 / F5a") - the shard top-k and the two
        # small gathers cost what the one logits gather and the vocab-wide
        # cut saved.
        self.use_candidate_draft = (
            os.environ.get("QC_DRAFT_CANDIDATES", "0") == "1"
            and quixicore_ops.has_v2_candidate_draft()
        )
        self._candidate_draft_announced = False
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

        self.draft_logits: torch.Tensor | None = None
        # Bound per propose() call from its draft_grammar parameter; internal
        # plumbing between propose, _generate_draft (eager gating), and the
        # DSpark sequential sampler. Never mutated from outside.
        self.draft_grammar = None
        if self.speculative_config.draft_sample_method == "probabilistic":
            # Pre-temperature logits, cached from the previous decode step.
            dtype, fill = self.draft_logits_spec(vllm_config)
            self.draft_logits = torch.full(
                (
                    self.max_num_reqs,
                    self.num_speculative_steps,
                    self.vocab_size,
                ),
                fill,
                dtype=dtype,
                device=device,
            )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        """Dtype and initial value for the cached proposal distribution.

        A speculator that writes every column each step can start from zero.
        One that writes a subset -- DFlash2 caches only its K candidates --
        overrides this, since the columns it never touches have to read as
        impossible.
        """
        return torch.float32, 0.0

    @abstractmethod
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        pass

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        self._validate_local_argmax_reduction()

        all_attn_layers = set[str](
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.draft_attn_layer_names = all_attn_layers - target_attn_layer_names

    def set_eplb_state(self, eplb_state: EplbState) -> None:
        """Inject EPLB state after construction."""
        self.eplb_state = eplb_state

    def _prepare_eplb_forward(self, num_unpadded_tokens: int) -> None:
        """Call EPLB prepare_forward if EPLB is active for the draft model."""
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(
                self.speculative_config.draft_model_config,
                num_unpadded_tokens,
            )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        """Config for the draft's attention metadata builders. Overridden by
        speculators whose attention mode differs from the target's."""
        return self.vllm_config

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.attn_groups, self.attn_cg_support, _ = init_attn_backend(
            kv_cache_config,
            self.attn_vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables
        # The target model runner's buffers and attention groups. Draft
        # prefill reuses the target model's attention metadata, so its
        # cudagraph capture must build dummy metadata through the same
        # builders and buffers.
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int = 1,
        causal: bool | Mapping[int, bool] = True,
    ) -> dict[str, Any] | None:
        # Uniform query: query_start_loc[i] = min(i, num_reqs) * num_query_per_req.
        # Clamp keeps the series non-decreasing past num_reqs, which some
        # attention backends require.
        query_start_loc_cpu = (
            torch.clamp(self.arange[: num_reqs_padded + 1], max=num_reqs)
            * num_query_per_req
        )
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        draft_seq_lens_cpu_upper_bound = torch.zeros(
            num_reqs_padded, dtype=torch.int32, device="cpu"
        )
        torch.add(
            seq_lens_cpu_upper_bound[:num_reqs],
            step,
            out=draft_seq_lens_cpu_upper_bound[:num_reqs],
        )
        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
            causal=causal,
            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
        )
        return attn_metadata

    def _validate_local_argmax_reduction(self) -> None:
        if not self.use_local_argmax_reduction:
            return
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "use_local_argmax_reduction is not compatible with "
                "draft_sample_method='probabilistic'."
            )
        if not hasattr(self.model, "get_top_tokens"):
            raise ValueError(
                "use_local_argmax_reduction is enabled but draft model "
                f"{self.model.__class__.__name__} does not implement "
                "get_top_tokens()."
            )
        logger.info(
            "Using local argmax reduction for draft token generation "
            "(communication: O(2*tp_size) vs O(vocab_size))."
        )

    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)  # type: ignore[operator]
        logits = self.model.compute_logits(hidden_states)  # type: ignore[operator]
        return logits.argmax(dim=-1)

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if draft_logits is not None:
            if (
                self._draft_mask is not None
                and self._draft_mask[0]
                and self._draft_mask[1]
                and self.use_candidate_draft
                and self.sampling_states is not None
                and hasattr(self.model, "compute_local_logits")
            ):
                return self._candidate_sample_draft(
                    hidden_states, positions, idx_mapping, temperature, seeds, draft_step, draft_logits
                )
            logits = self.model.compute_logits(hidden_states)  # type: ignore[operator]
            apply_temp = True
            if self._draft_mask is not None:
                # Under the request's cutoffs the temperature is applied here,
                # before the cut (top-p reads the tempered distribution).
                logits = self._mask_draft_logits(logits, idx_mapping, temperature)
                apply_temp = False
            # The drafted token's position (positions + 1) keys its draw; the
            # drafting salt keeps that stream disjoint from the target's
            # verification and resample draws at the same position.
            return gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                positions + 1,
                apply_temperature=apply_temp,
                output_processed_logits=draft_logits,
                output_processed_logits_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
                is_drafting=True,
            )
        return self._greedy_sample_draft(hidden_states)

    def _candidate_sample_draft(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> torch.Tensor:
        """The full path's draw from the ranks' top-K candidates: every token
        a request's top-k (<= K) can keep is among them, so the cut, the
        processed row and the noise per token id are the same."""
        from vllm.distributed import tensor_model_parallel_all_gather

        assert self._draft_mask is not None and self.sampling_states is not None
        states = self.sampling_states
        local, vocab_start = self.model.compute_local_logits(hidden_states)  # type: ignore[operator]
        k = min(topk_sample.MAX_TOP_K, local.shape[-1])
        vals, ids = local.float().topk(k, dim=-1)
        ids = (ids + vocab_start).to(torch.int32)
        if local.shape[-1] < self.vocab_size:
            vals = tensor_model_parallel_all_gather(vals, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
        if not self._candidate_draft_announced:
            self._candidate_draft_announced = True
            logger.info(
                "Drafting from %d gathered top-%d candidates per token instead of the "
                "[T, %d] logits all-gather (QC_DRAFT_CANDIDATES=1)",
                vals.shape[-1], k, self.vocab_size,
            )
        return quixicore_ops.v2_candidate_draft(
            vals.contiguous(),
            ids.contiguous(),
            states.top_k.gpu,
            states.top_p.gpu if self._draft_mask[2] else None,
            idx_mapping,
            seeds,
            positions + (1 + DRAFT_NOISE_SALT),
            temperature,
            self.vocab_size,
            draft_logits,
            draft_step,
            self.use_fp64_gumbel,
        )

    def _mask_draft_logits(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        assert self._draft_mask is not None and self.sampling_states is not None
        native, do_top_k, do_top_p = self._draft_mask
        states = self.sampling_states
        return mask_draft_logits(
            logits,
            idx_mapping,
            temperature,
            states.top_k.gpu if do_top_k else None,
            states.top_p.gpu if do_top_p else None,
            native,
        )

    def _copy_request_inputs(
        self,
        num_reqs: int,
        # [num_reqs]
        idx_mapping: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        # [num_reqs], the CPU copy of idx_mapping
        idx_mapping_np: np.ndarray | None = None,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        # (draft_top_k_top_p opts back in: the batch's cutoffs are noted here
        # and applied to the draft's logits in sample_draft.)
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.idx_mapping[:num_reqs].copy_(idx_mapping)
        if self.draft_logits is not None:
            # idx_mapping for CG padded requests points to -1, which is ignored
            # during sampling to prevent writing stale values to draft logits.
            self.idx_mapping[num_reqs:].fill_(-1)
        self._draft_mask = None
        states = self.sampling_states
        if (
            self.draft_top_k_top_p
            and self.draft_logits is not None
            and states is not None
            and idx_mapping_np is not None
        ):
            top_k = states.top_k.np[idx_mapping_np]
            do_top_k = bool(np.any(top_k != states.vocab_size))
            do_top_p = bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))
            if do_top_k or do_top_p:
                native = bool(np.all((top_k >= 1) & (top_k <= topk_sample.MAX_TOP_K)))
                self._draft_mask = (native, do_top_k, do_top_p)
