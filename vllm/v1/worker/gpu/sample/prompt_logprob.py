# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import numpy as np
import torch

from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int, logprobs_mode: LogprobsMode = "raw_logprobs"):
        self.max_num_reqs = max_num_reqs
        self.logprobs_mode = logprobs_mode

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: SamplingParams):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        includes_prompt = computed_prefill < prompt_lens
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # get the maximum number in this batch
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
                self.logprobs_mode,
            )
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        query_start_loc_np = input_batch.query_start_loc_np
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = (
                prompt_logprobs.shape[1]
                if req_num_prompt_logprobs == -1
                else req_num_prompt_logprobs + 1
            )
            # no logprobs if start_idx >= end_idx
            logprobs = (
                None
                if start_idx >= end_idx
                else LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )
            )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)
            if req_is_prompt_chunked:
                # Prompt is chunked. Do not return the logprobs yet.
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                logprobs = LogprobsTensors.cat(prompt_logprobs_list)
                prompt_logprobs_list.clear()

            if logprobs is None:
                continue

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    from vllm.v1.worker.gpu.sample.gumbel import _use_native_sample_kernels

    if _use_native_sample_kernels():
        from vllm.quixicore import quixicore_ops

        quixicore_ops.v2_prompt_logprobs_token_ids(
            token_ids,
            query_start_loc,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        return token_ids

    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


_FIRST_CHUNK = 8
# 64 tokens: <= 64 MiB of gathered fp32 logits per chunk plus temporaries.
# Larger chunks (the old fixed 1024) fragmented the caching allocator so a
# later prefill could not get a 128 MiB block at 0.975 utilization.
_MAX_CHUNK = 64
# Per token: the gathered fp32 logits, the local shard, and the top-k /
# log-softmax temporaries of compute_topk_scores (~4x the logits row).
_BYTES_PER_TOKEN_PER_VOCAB = 4 * 4


def _chunk_for_budget(vocab_size: int, device: torch.device) -> int:
    """Prompt-logprob chunk (tokens) that fits half of the memory currently
    free on the device (device free + the caching allocator's idle
    reserve)."""
    if device.type != "cuda":
        return _MAX_CHUNK
    free_device, _ = torch.cuda.mem_get_info(device)
    idle_reserved = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    budget = (free_device + max(idle_reserved, 0)) // 2
    per_token = max(vocab_size, 1) * _BYTES_PER_TOKEN_PER_VOCAB
    chunk = int(max(_FIRST_CHUNK, min(_MAX_CHUNK, budget // per_token)))
    # The logits all-gather needs every TP rank to use the SAME chunk: a
    # per-rank budget misaligns the gathered rows (garbage logprobs for the
    # rest of the prompt) or deadlocks when the chunk counts differ
    # (2026-09-07: both seen on the 4x3090 record). Agree on the minimum.
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size > 1:
        t = torch.tensor([chunk], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=tp.device_group)
        chunk = int(t.item())
    return chunk


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
    logprobs_mode: LogprobsMode = "raw_logprobs",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks. The chunk is sized from the memory actually
    # free on the device: with a 0.975 utilization the slack past the KV
    # pools is tens of MB, and a fixed 1024-token chunk (1 GB of fp32
    # full-vocab logits after the TP gather) OOM-killed the production
    # worker on 2026-09-07. The first chunk is small so the vocab width is
    # known before the budget is applied.
    token_ids = []
    scores = []
    ranks = []
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    num_tokens = prompt_token_ids.shape[0]
    start_idx = 0
    chunk_size = _FIRST_CHUNK
    while start_idx < num_tokens:
        end_idx = min(start_idx + chunk_size, num_tokens)
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        chunk_size = _chunk_for_budget(prompt_logits.shape[-1], prompt_logits.device)
        requested_num = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        result = compute_topk_scores(
            prompt_logits,
            requested_num,
            prompt_token_ids[start_idx:end_idx],
            logits_mode=logits_mode,
        )
        token_ids.append(result.logprob_token_ids)
        scores.append(result.logprobs)
        ranks.append(result.selected_token_ranks)
        start_idx = end_idx

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    scores = torch.cat(scores, dim=0) if len(scores) > 1 else scores[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, scores, ranks
