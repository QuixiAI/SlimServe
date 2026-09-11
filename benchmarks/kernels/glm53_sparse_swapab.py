# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 4.3/4.5 diagnostic: candidate-major scores and transposed PV.

Mechanism reference: FlashInfer #4751, merged through #4802. This adapts ONLY
operand ownership, not that implementation's FP8 quantization or dispatch.
Uses SlimServe's original BF16 Q/KV and FP16 P/V arithmetic and tile extents.
"""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["bt_stride", "max_topk"])
def sparse_swapab(
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
):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    # Q stays the reusable B operand; candidates own M. O is [D,H], so
    # normalization runs over candidates within each head, not across heads.
    q = tl.load(q_ptr + (b.to(tl.int64) * H + h)[None, :] * D + d[:, None])
    tlen = tl.minimum(tl.maximum(tl.load(tlen_ptr + b), 0), max_topk)
    m_i = tl.full((H,), -1.0e30, tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((D, H), tl.float32)
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
        s = tl.dot(k, q) * scale
        s = tl.where(ok[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[None, :])
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha[None, :] + tl.dot(
            tl.trans(k.to(tl.float16)), p.to(tl.float16)
        )
        m_i = m_new
    out = tl.where(l_i[None, :] > 0.0, acc / l_i[None, :], 0.0)
    tl.store(
        out_ptr + (b.to(tl.int64) * H + h)[None, :] * D + d[:, None],
        out.to(tl.bfloat16),
    )
