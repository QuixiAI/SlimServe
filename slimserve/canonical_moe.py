# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM53 ordering intervention, NOT a production performance path.

Sort existing expert-contiguous alignment in place. This preserves routing,
expert block counts, padding, and GEMM arithmetic, but adds one sorting launch.
Requires the bounded model journal so a diagnostic cannot silently become a
serving baseline. A final implementation should construct stable alignment at
the source rather than sorting an already constructed alignment.
"""

import os


def enabled():
    flag = os.environ.get("SLIMSERVE_GLM53_CANONICAL_MOE", "0")
    if flag not in ("0", "1"):
        raise ValueError("SLIMSERVE_GLM53_CANONICAL_MOE must be 0 or 1")
    if flag == "1" and os.environ.get("SLIMSERVE_GLM53_MODEL_JOURNAL") != "1":
        raise ValueError("canonical MoE diagnostic requires the bounded model journal")
    return flag == "1"


def stable_route_enabled():
    flag = os.environ.get("SLIMSERVE_GLM53_STABLE_ROUTE", "0")
    if flag not in ("0", "1"):
        raise ValueError("stable route flag must be 0 or 1")
    if flag == "1" and not enabled():
        raise ValueError("stable route diagnostic requires canonical MoE")
    return flag == "1"


def geometry(tokens, topk, experts, block_size):
    # The scoped router selects DISTINCT top8 experts for every token, so one
    # expert owns at most `tokens` assignments, even for maximally skewed input.
    if not (
        type(tokens) is int
        and 1 <= tokens <= 8192
        and (topk, experts) == (8, 288)
        and block_size in (8, 16, 32, 48, 64)
    ):
        raise ValueError("canonical MoE requires bounded GLM53 top8 routing")
    max_padded = tokens * topk + experts * (block_size - 1)
    if tokens * topk < experts:
        max_padded = min(tokens * topk * block_size, max_padded)
    return (
        max_padded,
        (max_padded + block_size - 1) // block_size,
        ((tokens + block_size - 1) // block_size) * block_size,
    )


def canonicalize(sorted_ids, expert_ids, padded_count, *, tokens, block_size):
    import torch

    from slimserve.canonical_moe_kernel import sort_expert_assignments

    capacity, max_blocks, max_expert_rows = geometry(tokens, 8, 288, block_size)
    tensors = (sorted_ids, expert_ids, padded_count)
    if (
        any(t.dtype != torch.int32 or not t.is_contiguous() for t in tensors)
        or not sorted_ids.is_cuda
        or any(t.device != sorted_ids.device for t in tensors)
        or sorted_ids.shape != (capacity,)
        or expert_ids.shape != (max_blocks,)
        or padded_count.shape != (1,)
    ):
        raise ValueError("canonical MoE alignment tensor contract changed")
    sort_expert_assignments(
        sorted_ids,
        expert_ids,
        padded_count,
        tokens * 8,
        block_size,
        max_blocks,
        max_expert_rows,
    )
