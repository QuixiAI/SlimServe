# SPDX-License-Identifier: Apache-2.0
"""Owned graph-safe compact-pool scorer candidate for SM80 decode.

Device-visible row lengths select tile16/32/64 and active CTA count under
a fixed graph launch. The explicit two16-head reduction preserves the
established numerical gate when Triton changes the dot-product layout.
The larger BP128 variant remains excluded after strict replay failures.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tile(
    Q,
    W,
    CACHE,
    BT,
    ROW_REQ,
    VISIBLE,
    OUT,
    MAX_POOLS,
    BT_STRIDE,
    PAGE_STRIDE,
    SCALE,
    BS: tl.constexpr,
    H: tl.constexpr,
    BP: tl.constexpr,
    ACTIVE: tl.constexpr,
):
    r, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(VISIBLE + r) // 4
    tiles = tl.cdiv(count, BP)
    if (block < tiles) & (block < ACTIVE):
        req = tl.load(ROW_REQ + r)
        h, d = tl.arange(0, H), tl.arange(0, 128)
        q = tl.load(Q + (r.to(tl.int64) * H + h[:, None]) * 128 + d[None, :])
        weights = tl.load(W + r * H + h)
        for tile in range(block, tiles, ACTIVE):
            p = tile * BP + tl.arange(0, BP)
            valid = p < count
            page = tl.load(BT + req.to(tl.int64) * BT_STRIDE + p * 4 // BS, valid, 0)
            base = page.to(tl.int64) * PAGE_STRIDE + (p % (BS // 4)) * 128
            pooled = tl.load(CACHE + base[:, None] + d[None, :], valid[:, None], 0)
            scores = tl.maximum(tl.dot(pooled, tl.trans(q)) * SCALE, 0.0)
            # Preserve the baseline's two 16-head partial sums even when
            # a wider M tile makes Triton assign fewer warps along N.
            weighted = (scores * weights[None, :]).reshape(BP, 2, 16)
            logits = tl.sum(tl.sum(weighted, axis=2), axis=1)
            tl.store(
                OUT + r.to(tl.int64) * MAX_POOLS + p,
                tl.where(valid, logits, float("-inf")),
                p < MAX_POOLS,
            )


@triton.jit
def _adaptive_pool_logits(
    Q,
    W,
    CACHE,
    BT,
    ROW_REQ,
    VISIBLE,
    OUT,
    MAX_POOLS,
    BT_STRIDE,
    PAGE_STRIDE,
    SCALE,
    BS: tl.constexpr,
    H: tl.constexpr,
    SHORT_BP: tl.constexpr,
    SHORT_PROGRAMS: tl.constexpr,
    LONG_PROGRAMS: tl.constexpr,
):
    count = tl.load(VISIBLE + tl.program_id(0)) // 4
    if count <= 512:
        _tile(
            Q,
            W,
            CACHE,
            BT,
            ROW_REQ,
            VISIBLE,
            OUT,
            MAX_POOLS,
            BT_STRIDE,
            PAGE_STRIDE,
            SCALE,
            BS,
            H,
            SHORT_BP,
            SHORT_PROGRAMS,
        )
    elif count <= 8192:
        _tile(
            Q,
            W,
            CACHE,
            BT,
            ROW_REQ,
            VISIBLE,
            OUT,
            MAX_POOLS,
            BT_STRIDE,
            PAGE_STRIDE,
            SCALE,
            BS,
            H,
            64,
            SHORT_PROGRAMS,
        )
    else:
        # BP128 still exceeds the strict replay tolerance; retain BP64
        # until its accumulation layout is separately corrected.
        _tile(
            Q,
            W,
            CACHE,
            BT,
            ROW_REQ,
            VISIBLE,
            OUT,
            MAX_POOLS,
            BT_STRIDE,
            PAGE_STRIDE,
            SCALE,
            BS,
            H,
            64,
            LONG_PROGRAMS,
        )


def adaptive_pool_logits(q, weights, cache, block_table, row_req, visible, out):
    assert q.shape[1:] == (32, 128)
    assert q.dtype == cache.dtype == torch.bfloat16
    assert weights.dtype == out.dtype == torch.float32
    assert cache.shape[2] == 64 and cache.stride()[1:] == (64, 1)
    assert block_table.stride(1) == 1
    assert all(t.is_contiguous() for t in (q, weights, row_req, visible, out))
    rows = q.shape[0]
    if not rows:
        return
    short_bp = 16 if rows < 16 else 32 if rows < 32 else 64
    short_programs = 128 if rows == 1 else 32
    long_programs = 64 if rows >= 32 else 128
    _adaptive_pool_logits[(rows, max(short_programs, long_programs))](
        q,
        weights,
        cache,
        block_table,
        row_req,
        visible,
        out,
        out.shape[1],
        block_table.stride(0),
        cache.stride(0),
        128**-0.5,
        cache.shape[1],
        q.shape[1],
        short_bp,
        short_programs,
        long_programs,
        num_warps=4,
    )
