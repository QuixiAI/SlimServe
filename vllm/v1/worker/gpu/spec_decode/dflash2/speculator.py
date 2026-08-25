# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        # Load at the width the argmax will reduce in. Loading fp32 and letting
        # the noise promote to fp64 gives the two arms of that branch different
        # types, which Triton rejects on ROCm.
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # The candidate token ids key the noise, so a token drawn at this slot
        # gets the same noise the target's own sampling would give it.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )
        if self.draft_logits is not None:
            # Restated rather than trusted: a construction path that skips the
            # base allocation would otherwise hand the walk uninitialized memory
            # where it needs every unwritten column to be impossible.
            self.draft_logits.fill_(-float("inf"))

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32, not the head dtype. Rounding real selector scores to bf16 moves
        # the argmax of a candidate row 0.81% of the time and reverses the order
        # of 0.68% of candidate pairs, so the walk and the rejection that checks
        # it would no longer read the same distribution. The fill is -inf because
        # the cache kernel writes only the K candidates, and every column it
        # never touches has to read as impossible.
        return torch.float32, -float("inf")

    def _sample_path_mps(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        # Torch-native walk: num_speculative_steps sequential [R, K] gathers
        # and argmaxes -- microseconds against the draft forward. As in
        # gumbel_sample's MPS branch, MPS has no stateless Philox, so the
        # probabilistic arm draws torch Gumbel noise: the sampling distribution
        # is preserved (rejection stays distribution-correct), but the
        # draw-for-draw coupling with verification -- and the acceptance-rate
        # bonus it buys -- remains native-CUDA/HIP-only. Greedy (temp 0) rows
        # are exact argmax, matching the Triton kernel token for token.
        steps = self.num_speculative_steps
        rows = torch.arange(num_reqs, device=candidate_ids.device)
        probabilistic = self.draft_logits is not None
        if probabilistic:
            req_state = (
                self.sample_idx_mapping[: num_reqs * steps]
                .view(num_reqs, steps)[:, 0]
                .to(torch.int64)
            )
            temps = self.temperature[req_state]
        previous = torch.zeros(num_reqs, dtype=torch.int64, device=rows.device)
        for step in range(steps):
            row_scores = scores[rows, step, previous]  # [R, K]
            self._selector_scores[:num_reqs, step] = row_scores
            if probabilistic:
                noise = -torch.empty_like(row_scores).exponential_().log()
                divisors = torch.where(temps == 0, 1, temps).unsqueeze(1)
                noised = row_scores / divisors + noise
                index = torch.where(
                    temps == 0, row_scores.argmax(-1), noised.argmax(-1)
                )
            else:
                index = row_scores.argmax(-1)
            self.draft_tokens[:num_reqs, step] = candidate_ids[rows, step, index]
            previous = index

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        if candidate_ids.device.type == "mps":
            self._sample_path_mps(candidate_ids, scores, num_reqs)
            return
        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        if candidate_ids.device.type == "mps":
            steps = self.num_speculative_steps
            top_k = self.selector_top_k
            req_state = self.sample_idx_mapping[:num_sample].to(torch.int64)
            # Boolean-mask indexing below is data-dependent-shape (the
            # pattern the serving hot paths avoid on MPS); acceptable here
            # because this runs once per drafter step on tiny tensors, off
            # the target-model critical path.
            valid = req_state >= 0
            batch_row = torch.arange(num_sample, device=candidate_ids.device)
            step = batch_row % steps
            # The cache is keyed by persistent request state; the walk's
            # realized scores are keyed by batch row (flat = row * steps + step,
            # as in the Triton kernel).
            flat_row = (req_state * steps + step)[valid]
            new_ids = candidate_ids.reshape(-1, top_k)[:num_sample][valid]
            walk_scores = self._selector_scores.view(-1, top_k)[batch_row[valid]]
            cached = self._cached_candidate_ids.view(-1, top_k)
            logits_rows = draft_logits.view(-1, draft_logits.shape[-1])
            row_exp = flat_row.unsqueeze(1).expand(-1, top_k)
            # Same order as the Triton kernel: retire the previous step's K
            # candidates to -inf, then write this step's scores.
            logits_rows[row_exp, cached[flat_row]] = -float("inf")
            logits_rows[row_exp, new_ids] = walk_scores
            cached[flat_row] = new_ids
            return
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
