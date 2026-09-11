# SPDX-License-Identifier: Apache-2.0
"""Experimental compact GLM indexer pages; opt-in through additional_config.

Each logical page stores BF16 completed pool keys followed by an eight-row
raw key/gate ring. A logical row budgets 64 BF16 elements versus the old
256. At the minimum 64-token page, pool keys and ring exactly fill the page.
Larger pages retain padding so the allocator keeps power-of-two page ratios.
The ring supports up to five speculative tokens; do not silently exceed it.

Like Qwen QSA compression, pools read current input rows or saved raw state.
Unlike its separate state owner, this ring travels with its physical page,
so prefix sharing and host/disk restore do not require an untracked cache.
"""

import torch

from vllm.triton_utils import tl, triton

POOL_CACHE_HEAD_DIM = 64
RAW_RING_ROWS = 8


@triton.jit
def _complete_pools(
    SRC,
    CACHE,
    SLOTS,
    APE,
    N,
    PAGE_STRIDE,
    BS: tl.constexpr,
):
    # Preserve the scorer's three-dimensional pool/member/channel tile.
    # A single-pool reduction changes FP32 summation and can cross BF16
    # rounding boundaries; cached keys must use the existing arithmetic.
    t = tl.program_id(0) * 16 + tl.arange(0, 16)
    slot = tl.load(SLOTS + t, t < N, -1)
    valid = (slot >= 0) & (slot % 4 == 3)
    page = slot // BS
    offset = slot % BS
    member = tl.arange(0, 4)
    d = tl.arange(0, 128)
    src_row = t[:, None] - 3 + member[None, :]
    wanted_slot = slot[:, None] - 3 + member[None, :]
    candidate_slot = tl.load(SLOTS + src_row, valid[:, None] & (src_row >= 0), -1)
    in_batch = (src_row >= 0) & (candidate_slot == wanted_slot)
    old_base = CACHE + page[:, None, None].to(tl.int64) * PAGE_STRIDE + (BS // 4) * 128
    old_base = old_base + (wanted_slot % 8)[:, :, None] * 256
    new_base = SRC + src_row[:, :, None].to(tl.int64) * 256
    new_mask = valid[:, None, None] & in_batch[:, :, None]
    old_mask = valid[:, None, None] & ~in_batch[:, :, None]
    k_new = tl.load(new_base + d[None, None, :], new_mask, 0).to(tl.float32)
    g_new = tl.load(new_base + 128 + d[None, None, :], new_mask, 0).to(tl.float32)
    k_old = tl.load(old_base + d[None, None, :], old_mask, 0).to(tl.float32)
    g_old = tl.load(old_base + 128 + d[None, None, :], old_mask, 0).to(tl.float32)
    k = tl.where(in_batch[:, :, None], k_new, k_old)
    g = tl.where(in_batch[:, :, None], g_new, g_old)
    ape = tl.load(APE + member[:, None] * 128 + d[None, :])
    g += ape[None, :, :]
    e = tl.exp(g - tl.max(g, axis=1)[:, None, :])
    p = e / tl.sum(e, axis=1)[:, None, :]
    pooled = tl.sum(p * k, axis=1).to(tl.bfloat16)
    dest = CACHE + page.to(tl.int64) * PAGE_STRIDE + (offset // 4) * 128
    tl.store(dest[:, None] + d[None, :], pooled, valid[:, None])


@triton.jit
def _singleton_pool_update(SRC, CACHE, SLOTS, APE, PAGE_STRIDE, BS: tl.constexpr):
    slot = tl.load(SLOTS)
    if slot < 0:
        return
    if slot % 4 == 3:
        _complete_pools(SRC, CACHE, SLOTS, APE, 1, PAGE_STRIDE, BS)
        # All old ring reads finish within this sole CTA before overwriting
        # the ring. This is not safe to generalize to multiple CTAs as-is.
        tl.debug_barrier()
    d = tl.arange(0, 256)
    value = tl.load(SRC + d)
    dest = CACHE + (slot // BS).to(tl.int64) * PAGE_STRIDE + (BS // 4) * 128
    tl.store(dest + (slot % 8) * 256 + d, value)


@triton.jit
def _save_raw_tail(SRC, CACHE, SLOTS, N, PAGE_STRIDE, BS: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(SLOTS + t)
    # Input rows for one request are contiguous and its physical page slots
    # are consecutive. Only the last writer to each ring row may store.
    later = tl.load(SLOTS + t + 8, t + 8 < N, -1)
    superseded = (later == slot + 8) & (later // BS == slot // BS)
    if slot < 0 or superseded:
        return
    d = tl.arange(0, 256)
    value = tl.load(SRC + t.to(tl.int64) * 256 + d)
    dest = CACHE + (slot // BS).to(tl.int64) * PAGE_STRIDE + (BS // 4) * 128
    tl.store(dest + (slot % 8) * 256 + d, value)


def update_pool_cache(packed, slots, ape, cache, singleton_fused=False):
    """Insert contiguous per-request input rows; caller owns rejection masks."""
    assert packed.ndim == 2 and packed.shape[1] == 256 and packed.is_contiguous()
    assert packed.dtype == cache.dtype == torch.bfloat16
    assert slots.shape == (packed.shape[0],) and slots.is_contiguous()
    assert ape.shape == (4, 128) and ape.dtype == torch.float32
    assert ape.is_contiguous() and cache.ndim == 3
    assert cache.shape[2] == POOL_CACHE_HEAD_DIM
    bs = cache.shape[1]
    assert bs >= 64 and bs % 64 == 0
    assert cache.stride()[1:] == (POOL_CACHE_HEAD_DIM, 1)
    assert packed.is_cuda and all(
        t.device == packed.device for t in (slots, ape, cache)
    )
    n = packed.shape[0]
    if not n:
        return
    if singleton_fused and n == 1:
        _singleton_pool_update[(1,)](
            packed, cache, slots, ape, cache.stride(0), bs, num_warps=4
        )
        return
    # Separate launches are essential: ring stores must not race a partial
    # pool's reads of committed rows from the previous step.
    _complete_pools[(triton.cdiv(n, 16),)](
        packed,
        cache,
        slots,
        ape,
        n,
        cache.stride(0),
        bs,
        num_warps=4,
    )
    _save_raw_tail[(n,)](packed, cache, slots, n, cache.stride(0), bs, num_warps=4)


@triton.jit
def _cached_pool_logits(
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
):
    r, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(VISIBLE + r) // 4
    tiles = tl.cdiv(count, BP)
    if block >= tiles:
        return
    req = tl.load(ROW_REQ + r)
    h, d = tl.arange(0, H), tl.arange(0, 128)
    q = tl.load(Q + (r.to(tl.int64) * H + h[:, None]) * 128 + d[None, :])
    weights = tl.load(W + r * H + h)
    for tile in range(block, tiles, tl.num_programs(1)):
        p = tile * BP + tl.arange(0, BP)
        valid = p < count
        page = tl.load(BT + req.to(tl.int64) * BT_STRIDE + p * 4 // BS, valid, 0)
        base = page.to(tl.int64) * PAGE_STRIDE + (p % (BS // 4)) * 128
        pooled = tl.load(CACHE + base[:, None] + d[None, :], valid[:, None], 0)
        scores = tl.maximum(tl.dot(pooled, tl.trans(q)) * SCALE, 0.0)
        logits = tl.sum(scores * weights[None, :], axis=1)
        tl.store(
            OUT + r.to(tl.int64) * MAX_POOLS + p,
            tl.where(valid, logits, float("-inf")),
            p < MAX_POOLS,
        )


def cached_pool_logits(
    q, weights, cache, block_table, row_req, visible, out, programs=128
):
    assert q.is_contiguous() and weights.is_contiguous()
    assert q.dtype == cache.dtype == torch.bfloat16 and weights.dtype == torch.float32
    assert q.shape[2] == 128 and cache.shape[2] == POOL_CACHE_HEAD_DIM
    if not q.shape[0]:
        return
    _cached_pool_logits[(q.shape[0], min(programs, triton.cdiv(out.shape[1], 16)))](
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
        16,
        num_warps=4,
    )


@triton.jit
def _cached_pool_logits_grouped(
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
    G,
    BS: tl.constexpr,
    H: tl.constexpr,
    BP: tl.constexpr,
    GP: tl.constexpr,
):
    """Score one request's G consecutive rows per pooled-cache read.

    The per-row kernel streams every pool of the request once per row; a
    speculative verify batch has k+1 rows per request that share the same
    history, so this variant loads each pool tile once and dots it against
    all G x H query heads. GP (a power of two >= G) pads the head tile;
    padded rows are masked on load and store. Rows of a request must be
    contiguous, which is the decode layout (row_req is checked on the host).
    """
    grp, block = tl.program_id(0), tl.program_id(1)
    r0 = grp * G
    g = tl.arange(0, GP)
    rows = r0 + g
    row_ok = g < G
    counts = tl.load(VISIBLE + rows, row_ok, 0) // 4
    count_max = tl.max(counts, axis=0)
    tiles = tl.cdiv(count_max, BP)
    if block >= tiles:
        return
    req = tl.load(ROW_REQ + r0)
    gh = tl.arange(0, GP * H)
    d = tl.arange(0, 128)
    gh_row = r0 + gh // H
    gh_ok = (gh // H) < G
    q = tl.load(
        Q + (gh_row.to(tl.int64) * H + gh % H)[:, None] * 128 + d[None, :],
        gh_ok[:, None],
        0,
    )
    weights = tl.load(W + gh_row * H + gh % H, gh_ok, 0.0)
    for tile in range(block, tiles, tl.num_programs(1)):
        p = tile * BP + tl.arange(0, BP)
        valid = p < count_max
        page = tl.load(BT + req.to(tl.int64) * BT_STRIDE + p * 4 // BS, valid, 0)
        base = page.to(tl.int64) * PAGE_STRIDE + (p % (BS // 4)) * 128
        pooled = tl.load(CACHE + base[:, None] + d[None, :], valid[:, None], 0)
        scores = tl.maximum(tl.dot(pooled, tl.trans(q)) * SCALE, 0.0)
        weighted = (scores * weights[None, :]).reshape(BP, GP, H)
        logits = tl.sum(weighted, axis=2)
        ok = (p[:, None] < counts[None, :]) & row_ok[None, :]
        tl.store(
            OUT + rows[None, :].to(tl.int64) * MAX_POOLS + p[:, None],
            tl.where(ok, logits, float("-inf")),
            (p[:, None] < MAX_POOLS) & row_ok[None, :],
        )


def cached_pool_logits_grouped(
    q, weights, cache, block_table, row_req, visible, out, group, programs=64
):
    """Grouped scorer; requires rows in contiguous groups of `group` per
    request (checked). Same outputs as cached_pool_logits."""
    assert 2 <= group <= 8
    rows = q.shape[0]
    assert rows % group == 0 and q.shape[1:] == (32, 128)
    assert q.dtype == cache.dtype == torch.bfloat16
    assert weights.dtype == out.dtype == torch.float32
    assert cache.shape[2] == POOL_CACHE_HEAD_DIM
    assert all(t.is_contiguous() for t in (q, weights, row_req, visible, out))
    if not rows:
        return
    gp = 2 if group <= 2 else 4 if group <= 4 else 8
    _cached_pool_logits_grouped[(rows // group, programs)](
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
        group,
        cache.shape[1],
        q.shape[1],
        16,
        gp,
        num_warps=4,
    )


def rows_form_request_groups(row_req: torch.Tensor, group: int) -> bool:
    """True when every consecutive run of `group` rows shares one request."""
    if group <= 1 or row_req.shape[0] % group:
        return False
    view = row_req.view(-1, group)
    return bool((view == view[:, :1]).all().item())
