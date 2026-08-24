# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


def save_partial_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
) -> None:
    """Write packed [kv, score+ape] partial states into the compressor cache.

    One program per token; pads (slot_id == -1) are skipped.
    """
    num_actual = slot_mapping.shape[0]
    head_size = kv.shape[-1]
    from vllm.platforms import current_platform

    if current_platform.is_metal():
        from vllm.quixicore.ops import quixicore_ops

        # kv/score are the two halves of the fused kv_score GEMM output;
        # the host op binds the row-strided views directly (bit-identical
        # values, no per-call .contiguous() copies). fp16 halves pass
        # straight through — the kernel rounds to bf16 in-register, a
        # single RNE rounding bit-identical to an eager
        # .float()+.to(bfloat16) chain. Other dtypes convert here.
        kv_v = kv[:num_actual]
        score_v = score[:num_actual]
        if kv_v.dtype not in (torch.bfloat16, torch.float16):
            kv_v = kv_v.to(torch.bfloat16)
        if score_v.dtype != kv_v.dtype:
            score_v = score_v.to(kv_v.dtype)
        if ape.dtype != torch.bfloat16 or not ape.is_contiguous():
            # One-time conversions only: the compressor passes a memoized
            # bf16 ape on Metal.
            ape = ape.to(torch.bfloat16).contiguous()
        quixicore_ops.deepseek_v4_save_partial_states(
            kv_v,
            score_v,
            ape,
            positions,
            state_cache,
            slot_mapping,
            block_size,
            state_width,
            compress_ratio,
        )
        return

    _save_partial_states_kernel[(num_actual,)](
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        ape.stride(0),
        positions,
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        slot_mapping,
        block_size,
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        **(pdl_kwargs or {}),
    )


@triton.jit
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
    # by vLLM).  During CUDA graph replay the batch may contain padding
    # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
    # illegal memory access.
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    # Fused: score += ape[position % compress_ratio]
    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(
        base_ptr + STATE_WIDTH + block,
        score + ape,
        mask=mask,
    )
