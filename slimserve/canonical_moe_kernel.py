# SPDX-License-Identifier: Apache-2.0
"""Quarantined ordering diagnostic; loaded only by an enabled intervention."""

import triton
import triton.language as tl


@triton.jit
def _sort_expert_assignments(
    Sorted,
    Experts,
    Padded,
    NUMEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    SCAN_SIZE: tl.constexpr,
    SORT_SIZE: tl.constexpr,
):
    expert = tl.program_id(0)
    block = tl.arange(0, SCAN_SIZE)
    used_blocks = tl.load(Padded) // BLOCK_SIZE
    owners = tl.load(Experts + block, block < MAX_BLOCKS, other=-1)
    owned = (block < used_blocks) & (owners == expert)
    start = tl.min(tl.where(owned, block, MAX_BLOCKS)) * BLOCK_SIZE
    length = tl.sum(owned.to(tl.int32)) * BLOCK_SIZE
    # Each CTA exclusively owns one existing expert-contiguous range. There is
    # no inter-CTA dependency and no read/write of uninitialized capacity tails.
    if length > 0:
        offset = tl.arange(0, SORT_SIZE)
        values = tl.load(Sorted + start + offset, offset < length, other=NUMEL)
        ordered = tl.sort(values, descending=False)
        tl.store(Sorted + start + offset, ordered, offset < length)


def sort_expert_assignments(
    sorted_ids, expert_ids, padded_count, numel, block_size, max_blocks, max_expert_rows
):
    _sort_expert_assignments[(288,)](
        sorted_ids,
        expert_ids,
        padded_count,
        numel,
        block_size,
        max_blocks,
        triton.next_power_of_2(max_blocks),
        triton.next_power_of_2(max_expert_rows),
        num_warps=4 if max_expert_rows <= 1024 else 8,
    )
