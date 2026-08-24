# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils import random_uuid
from vllm.utils.math_utils import next_power_of_2


def _quixicore_disabled() -> bool:
    """VLLM_QC_DISABLE_NATIVE=1 forces the Triton path.

    Exists so the native and Triton routes can be compared on an otherwise
    identical server; the kernels are bitwise-equal, so greedy output must
    match token for token.
    """
    import os

    return os.environ.get("VLLM_QC_DISABLE_NATIVE") == "1"


@cache
def _use_native(op_name: str) -> bool:
    """Prefer the native batch-prep kernel over the Triton one."""
    if _quixicore_disabled():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return False
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.is_available() and quixicore_ops.has(op_name)


class InputBuffers:
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # DCP: per-request local seq_lens buffer
        self.dcp_local_seq_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    # sum(num_scheduled_tokens)
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests.
    num_draft_tokens: int
    # [num_reqs] number of draft tokens scheduled for each request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs]
    num_computed_tokens_np: np.ndarray
    # [num_reqs]
    prefill_len_np: np.ndarray
    # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray
    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).
    is_prefilling_np: np.ndarray

    # [num_reqs] only populated when pipeline parallelism is enabled.
    max_seq_len_np: np.ndarray | None

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor
    # [num_tokens_after_padding]
    is_padding: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    # [num_reqs] per-request prompt length, only populated for R-SWA.
    prompt_lens: torch.Tensor | None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        ).zero_()

        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[:num_reqs] = num_tokens // num_reqs
        input_buffers.seq_lens[num_reqs - 1] += num_tokens % num_reqs
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        input_buffers.is_padding[:num_tokens].fill_(True)
        is_padding = input_buffers.is_padding[:num_tokens]

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Dummy: seq_len == query_len (fresh-prefill shape).
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            max_seq_len_np=None,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
        )


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    next_pos = num_computed + query_len
    if next_pos < prefill_len:
        next_token = tl.load(request_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)


def mps_segment_ids(cu: torch.Tensor, num_segments: int, total: int) -> torch.Tensor:
    """Row id per element for CSR-style int64 boundaries, with no host syncs.

    Marks each interior segment start with scatter_add and prefix-sums the
    markers. The hard invariant is that no interior boundary equals
    ``total`` (a trailing empty segment would scatter out of bounds);
    interior empty segments are handled (two markers on one index bump the
    cumsum by 2). Scheduled requests satisfy this: each has at least one
    token and one logit.
    """
    marker = torch.zeros(total, dtype=torch.int64, device=cu.device)
    if num_segments > 1:
        marker.scatter_add_(
            0,
            cu[1:num_segments],
            torch.ones(num_segments - 1, dtype=torch.int64, device=cu.device),
        )
    return marker.cumsum(0)


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    num_tokens: int | None = None,
) -> None:
    num_reqs = idx_mapping.shape[0]
    if input_ids.device.type == "mps":
        # Tensorized: shape control comes from CPU-known ints, content stays
        # on the GPU. Writes that must be skipped are routed to a scratch
        # cell one past the real buffer instead (duplicate scratch writes
        # race benignly — the cell is discarded; never make them
        # accumulate).
        device = input_ids.device
        idx = idx_mapping.to(torch.int64)
        qsl = query_start_loc[: num_reqs + 1].to(torch.int64)
        if num_tokens is None:
            num_tokens = int(qsl[num_reqs].cpu())
        if not num_tokens:
            return
        computed = num_computed_tokens.index_select(0, idx).to(torch.int64)
        plen = prefill_len.index_select(0, idx).to(torch.int64)
        is_prefill = computed < plen

        seg = mps_segment_ids(qsl, num_reqs, num_tokens)
        token_arange = torch.arange(num_tokens, dtype=torch.int64, device=device)
        local = token_arange - qsl[:-1].index_select(0, seg)
        row_state = idx.index_select(0, seg)
        row_capacity = all_token_ids.shape[1]
        src_col = (computed.index_select(0, seg) + local).clamp(max=row_capacity - 1)
        tokens = all_token_ids.view(-1).index_select(
            0, row_state * row_capacity + src_col
        )
        num_ids = input_ids.shape[0]
        dst = torch.where(
            is_prefill.index_select(0, seg),
            token_arange,
            torch.full_like(token_arange, num_ids),
        )
        padded = torch.cat([input_ids, input_ids.new_zeros(1)])
        padded[dst] = tokens.to(padded.dtype)
        input_ids.copy_(padded[:num_ids])

        qlens = qsl[1:] - qsl[:-1]
        next_pos = computed + qlens
        has_next = is_prefill & (next_pos < plen)
        next_tokens = all_token_ids.view(-1).index_select(
            0, idx * row_capacity + next_pos.clamp(max=row_capacity - 1)
        )
        num_states = next_prefill_tokens.shape[0]
        next_dst = torch.where(has_next, idx, torch.full_like(idx, num_states))
        next_padded = torch.cat([next_prefill_tokens, next_prefill_tokens.new_zeros(1)])
        next_padded[next_dst] = next_tokens.to(next_padded.dtype)
        next_prefill_tokens.copy_(next_padded[:num_states])
        return
    if _use_native("prepare_prefill_inputs"):
        from vllm.quixicore import quixicore_ops

        quixicore_ops.prepare_prefill_inputs(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            prefill_len,
            num_computed_tokens,
        )
        return
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
    num_tokens: int | None = None,
) -> None:
    num_reqs = idx_mapping.shape[0]
    if pos.device.type == "mps":
        device = pos.device
        idx = idx_mapping.to(torch.int64)
        qsl = query_start_loc[: num_reqs + 1].to(torch.int64)
        computed = num_computed_tokens.index_select(0, idx).to(torch.int64)
        qlens = qsl[1:] - qsl[:-1]
        seq_lens[:num_reqs].copy_((computed + qlens).to(seq_lens.dtype))
        seq_lens[num_reqs:].zero_()
        if num_tokens is None:
            num_tokens = int(qsl[num_reqs].cpu())
        if num_tokens:
            seg = mps_segment_ids(qsl, num_reqs, num_tokens)
            local = torch.arange(num_tokens, dtype=torch.int64, device=device) - qsl[
                :-1
            ].index_select(0, seg)
            pos[:num_tokens].copy_(
                (computed.index_select(0, seg) + local).to(pos.dtype)
            )
        return
    if _use_native("prepare_pos_seq_lens"):
        from vllm.quixicore import quixicore_ops

        quixicore_ops.prepare_pos_seq_lens(
            pos,
            seq_lens,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            seq_lens.shape[0],
        )
        return
    # NOTE(woosuk): We do +1 because the last thread block is used
    # to pad unused seq_lens as 0 for full CUDA graphs.
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
) -> torch.Tensor:
    assert num_new_sampled_tokens in (0, 1), (
        f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    )
    # use idx_mapping.shape[0] for actual request count
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )
    if input_ids.device.type == "mps":
        device = input_ids.device
        idx = idx_mapping.to(torch.int64)
        cu = cu_num_logits.to(torch.int64)
        qsl = query_start_loc[: num_reqs + 1].to(torch.int64)

        seg = mps_segment_ids(cu, num_reqs, num_logits)
        local = torch.arange(num_logits, dtype=torch.int64, device=device) - cu[
            :-1
        ].index_select(0, seg)
        nl_req = cu[1:] - cu[:-1]
        input_start_req = qsl[1:] - nl_req
        torch.add(input_start_req.index_select(0, seg), local, out=logits_indices)

        # Masked writes route to a scratch cell one past the buffer, so no
        # host-side loop control or boolean indexing (both sync) is needed.
        num_ids = input_ids.shape[0]
        padded = torch.cat([input_ids, input_ids.new_zeros(1)])
        if num_new_sampled_tokens:
            seq = seq_lens[:num_reqs].to(torch.int64)
            first_logit_pos = seq - nl_req
            plen = prefill_len.index_select(0, idx).to(torch.int64)
            do_write = first_logit_pos >= plen
            dst0 = torch.where(
                do_write, input_start_req, torch.full_like(input_start_req, num_ids)
            )
            padded[dst0] = (
                last_sampled_tokens.reshape(-1).index_select(0, idx).to(padded.dtype)
            )
        if num_speculative_steps > 0:
            nd = (nl_req - num_new_sampled_tokens).clamp(min=0)
            cols = torch.arange(
                num_speculative_steps, dtype=torch.int64, device=device
            ).unsqueeze(0)
            write_mask = cols < nd.unsqueeze(1)
            dst = (qsl[1:] - nd).unsqueeze(1) + cols
            dst = torch.where(write_mask, dst, torch.full_like(dst, num_ids))
            src = draft_tokens.index_select(0, idx)[:, :num_speculative_steps]
            padded[dst.view(-1)] = src.reshape(-1).to(padded.dtype)
        input_ids.copy_(padded[:num_ids])
        return logits_indices
    if _use_native("combine_sampled_and_draft_tokens"):
        from vllm.quixicore import quixicore_ops

        quixicore_ops.combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu_num_logits,
            logits_indices,
            num_new_sampled_tokens,
        )
        return logits_indices
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
        # last sampled token in addition to all draft tokens.
        BLOCK_SIZE=next_power_of_2(num_speculative_steps + num_new_sampled_tokens),
    )
    return logits_indices


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    if num_sampled.device.type == "mps":
        mapped_prefill_len = prefill_len[idx_mapping.to(torch.int64)]
        is_chunked_prefill = seq_lens[:num_reqs] < mapped_prefill_len
        num_sampled.masked_fill_(is_chunked_prefill, 0)
        num_logits = cu_num_logits[1 : num_reqs + 1] - cu_num_logits[:num_reqs]
        num_rejected.copy_(torch.where(is_chunked_prefill, 0, num_logits - num_sampled))
        return num_sampled, num_rejected
    if _use_native("get_num_sampled_and_rejected"):
        from vllm.quixicore import quixicore_ops

        quixicore_ops.get_num_sampled_and_rejected(
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        return num_sampled, num_rejected
    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return num_sampled, num_rejected


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    if req_state_idx < 0:
        # Filter rows with negative index entries.
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    if query_start_loc_ptr is None:
        query_len = 0
    else:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


def post_update(
    # [num_reqs] batch_idx -> req_state_idx; negative index means skip.
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor | None,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor | None,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    if idx_mapping.device.type == "mps":
        # Fully tensorized: num_sampled/num_rejected are verify outputs that
        # only exist on the GPU, so loop control on them would be a sync.
        # Masked writes route to scratch cells; zero-valued index_add entries
        # implement the masked accumulations.
        device = idx_mapping.device
        max_reqs, row_capacity = all_token_ids.shape
        slots = sampled_tokens.shape[1]
        idx = idx_mapping.to(torch.int64)
        valid_req = idx >= 0
        safe_idx = torch.where(valid_req, idx, torch.zeros_like(idx))
        counts = num_sampled.to(torch.int64)
        has_sampled = (counts > 0) & valid_req

        cols = torch.arange(slots, dtype=torch.int64, device=device).unsqueeze(0)
        write_mask = (cols < counts.unsqueeze(1)) & valid_req.unsqueeze(1)
        old_total = total_len.index_select(0, safe_idx).to(torch.int64)

        flat = all_token_ids.view(-1)
        flat_dst = safe_idx.unsqueeze(1) * row_capacity + old_total.unsqueeze(1) + cols
        # In-place masked scatter. Masked lanes are redirected to flat cell 0
        # and write back that cell's current value, so the write is a no-op
        # there and no full-matrix scratch copy is needed. Cell 0 holds
        # request 0's first prompt token, which no valid write can target
        # (old_total >= 1 for any live request), and every masked lane
        # writes the identical gathered value, so the duplicate writes are
        # benign.
        flat_dst = torch.where(write_mask, flat_dst, torch.zeros_like(flat_dst))
        cur_vals = flat.index_select(0, flat_dst.view(-1))
        new_vals = torch.where(
            write_mask.view(-1),
            sampled_tokens.to(flat.dtype).view(-1),
            cur_vals,
        )
        flat[flat_dst.view(-1)] = new_vals

        gather_pos = (counts - 1).clamp(min=0).unsqueeze(1)
        last_tok = sampled_tokens.gather(1, gather_pos).squeeze(1)
        # last_sampled_tokens is [max_num_reqs, 1]; work on the flat view.
        last_flat = last_sampled_tokens.reshape(-1)
        last_padded = torch.cat([last_flat, last_flat.new_zeros(1)])
        last_dst = torch.where(
            has_sampled, safe_idx, torch.full_like(safe_idx, max_reqs)
        )
        last_padded[last_dst] = last_tok.to(last_padded.dtype)
        last_flat.copy_(last_padded[:max_reqs])

        total_len.index_add_(
            0,
            safe_idx,
            (counts * has_sampled.to(torch.int64)).to(total_len.dtype),
        )

        if output_bin_counts is not None:
            vocab_size = output_bin_counts.shape[1]
            token_grid = sampled_tokens.to(torch.int64).clamp(min=0)
            bin_idx = safe_idx.unsqueeze(1) * vocab_size + token_grid
            output_bin_counts.view(-1).index_add_(
                0,
                bin_idx.view(-1),
                write_mask.view(-1).to(output_bin_counts.dtype),
            )

        if query_start_loc is not None:
            qsl = query_start_loc[: num_reqs + 1].to(torch.int64)
            query_lens = qsl[1:] - qsl[:-1]
        else:
            query_lens = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        delta = (query_lens - num_rejected.to(torch.int64)) * valid_req.to(torch.int64)
        num_computed_tokens.index_add_(0, safe_idx, delta.to(num_computed_tokens.dtype))
        return
    if _use_native("post_update"):
        from vllm.quixicore import quixicore_ops

        quixicore_ops.post_update(
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            output_bin_counts.stride(0) if output_bin_counts is not None else 0,
            sampled_tokens,
            sampled_tokens.stride(0),
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
        )
        return
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        num_warps=1,
    )


@triton.jit
def _post_update_num_computed_tokens_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


def post_update_num_computed_tokens(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    if idx_mapping.device.type == "mps":
        query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        num_computed_tokens.index_add_(
            0, idx_mapping.to(torch.int64), query_lens.to(num_computed_tokens.dtype)
        )
        return
    if _use_native("post_update_num_computed_tokens"):
        from vllm.quixicore import quixicore_ops

        quixicore_ops.post_update_num_computed_tokens(
            idx_mapping,
            num_computed_tokens,
            query_start_loc,
        )
        return
    _post_update_num_computed_tokens_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    if idx_mapping.device.type == "mps":
        cu = cu_num_logits.to(torch.int64)
        seg = mps_segment_ids(cu, num_reqs, total_num_logits)
        expanded_idx_mapping = idx_mapping.index_select(0, seg)
        expanded_local_pos = (
            torch.arange(total_num_logits, dtype=torch.int64, device=idx_mapping.device)
            - cu[:-1].index_select(0, seg)
        ).to(torch.int32)
        return expanded_idx_mapping, expanded_local_pos
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    if _use_native("expand_idx_mapping"):
        from vllm.quixicore import quixicore_ops

        quixicore_ops.expand_idx_mapping(
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            cu_num_logits,
        )
        return expanded_idx_mapping, expanded_local_pos
    _expand_idx_mapping_kernel[(num_reqs,)](
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits,
        BLOCK_SIZE=next_power_of_2(max_expand_len),
    )
    return expanded_idx_mapping, expanded_local_pos
