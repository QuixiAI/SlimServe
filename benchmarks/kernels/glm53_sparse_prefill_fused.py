# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rejected sparse-prefill diagnostics; never imported by serving.

Accumulator fusion, query reload, and value splitting were closed on 2026-09-10
without a useful speed gain. Kept solely to reproduce the recorded experiments.
See perf/optimization_status.md, "Close sparse-prefill local variants".
"""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["bt_stride", "max_topk"])
def kernel(
    q_ptr,
    cache_ptr,
    bt_ptr,
    idx_ptr,
    tlen_ptr,
    out_ptr,
    max_topk,
    bt_stride,
    page_stride,
    block_size,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RELOAD_Q: tl.constexpr = False,
    FUSE_ACC: tl.constexpr = True,
    VALUE_TILE: tl.constexpr = 0,
):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    DV: tl.constexpr = VALUE_TILE if VALUE_TILE else D
    dv = tl.program_id(1) * DV + tl.arange(0, DV)
    if not RELOAD_Q:
        q = tl.load(q_ptr + (b.to(tl.int64) * H + h)[:, None] * D + d[None, :])
    tlen = tl.minimum(tl.maximum(tl.load(tlen_ptr + b), 0), max_topk)
    m_i = tl.full((H,), -1.0e30, tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((H, DV), tl.float32)
    for j0 in range(0, tlen, BLOCK_N):
        j = j0 + tl.arange(0, BLOCK_N)
        t = tl.load(idx_ptr + b.to(tl.int64) * max_topk + j, mask=j < tlen, other=-1)
        col = t // block_size
        ok = (t >= 0) & (col < bt_stride)
        blk = tl.load(bt_ptr + b.to(tl.int64) * bt_stride + col, mask=ok, other=-1)
        ok = ok & (blk >= 0)
        slot = t - col * block_size
        rows = blk.to(tl.int64) * page_stride + slot.to(tl.int64) * D
        k = tl.load(cache_ptr + rows[:, None] + d[None, :], mask=ok[:, None], other=0.0)
        if RELOAD_Q:
            # Cacheable reload shortens Q's live range across the value product.
            # Volatile prevents loop hoisting.
            q = tl.load(
                q_ptr + (b.to(tl.int64) * H + h)[:, None] * D + d[None, :],
                volatile=True,
            )
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if VALUE_TILE:
            v = tl.load(
                cache_ptr + rows[:, None] + dv[None, :], mask=ok[:, None], other=0.0
            ).to(tl.float16)
        else:
            v = k.to(tl.float16)
        # Keep the running output in the tensor-core accumulator instead of
        # materializing a second HxD FP32 product and adding it afterwards.
        # This changes FP32 summation order, not operand/accumulator precision.
        if FUSE_ACC:
            acc = tl.dot(p.to(tl.float16), v, acc * alpha[:, None])
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(
        out_ptr + (b.to(tl.int64) * H + h)[:, None] * D + dv[None, :],
        out.to(tl.bfloat16),
    )
