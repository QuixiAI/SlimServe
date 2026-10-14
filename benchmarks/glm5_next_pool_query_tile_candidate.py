# SPDX-License-Identifier: Apache-2.0
"""Quarantined compact-pool scorer: share keys across adjacent query rows.

Homogeneous request groups use a wider tensor-core N tile. Groups crossing a
request boundary fall back to independent rows. No changes to serving code.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _independent_row(
    Q, W, CACHE, BT, ROW_REQ, VISIBLE, OUT, R, MAX_POOLS,
    BT_STRIDE, PAGE_STRIDE, SCALE, BS: tl.constexpr,
):
    block = tl.program_id(1)
    count = tl.load(VISIBLE + R) // 4
    tiles = tl.cdiv(count, 16)
    if block < tiles:
        req = tl.load(ROW_REQ + R)
        h, d = tl.arange(0, 32), tl.arange(0, 128)
        q = tl.load(Q + (R.to(tl.int64) * 32 + h[:, None]) * 128 + d[None, :])
        weights = tl.load(W + R * 32 + h)
        for tile in range(block, tiles, tl.num_programs(1)):
            p = tile * 16 + tl.arange(0, 16)
            valid = p < count
            page = tl.load(BT + req.to(tl.int64) * BT_STRIDE + p * 4 // BS,
                           valid, 0)
            base = page.to(tl.int64) * PAGE_STRIDE + (p % (BS // 4)) * 128
            pooled = tl.load(CACHE + base[:, None] + d[None, :], valid[:, None], 0)
            scores = tl.maximum(tl.dot(pooled, tl.trans(q)) * SCALE, 0.0)
            logits = tl.sum(scores * weights[None, :], axis=1)
            tl.store(OUT + R.to(tl.int64) * MAX_POOLS + p,
                     tl.where(valid, logits, float("-inf")), p < MAX_POOLS)


@triton.jit
def _query_tile(
    Q, W, CACHE, BT, ROW_REQ, VISIBLE, OUT, ROWS, MAX_POOLS,
    BT_STRIDE, PAGE_STRIDE, SCALE, BS: tl.constexpr, BQ: tl.constexpr,
    FALLBACK: tl.constexpr = True,
):
    row0 = tl.program_id(0) * BQ
    rows = row0 + tl.arange(0, BQ)
    live = rows < ROWS
    requests = tl.load(ROW_REQ + rows, live, -1)
    req0 = tl.load(ROW_REQ + row0)
    same = tl.sum(((requests == req0) | ~live).to(tl.int32)) == BQ
    if same:
        counts = tl.load(VISIBLE + rows, live, 0) // 4
        count = tl.max(counts)
        block = tl.program_id(1)
        tiles = tl.cdiv(count, 16)
        if block < tiles:
            n, d = tl.arange(0, BQ * 32), tl.arange(0, 128)
            query_rows = row0 + n // 32
            q = tl.load(Q + (query_rows[:, None].to(tl.int64) * 32
                            + n[:, None] % 32) * 128 + d[None, :],
                        (query_rows < ROWS)[:, None], 0)
            weights = tl.load(W + query_rows * 32 + n % 32, query_rows < ROWS, 0)
            for tile in range(block, tiles, tl.num_programs(1)):
                p = tile * 16 + tl.arange(0, 16)
                valid = p < count
                page = tl.load(BT + req0.to(tl.int64) * BT_STRIDE + p * 4 // BS,
                               valid, 0)
                base = page.to(tl.int64) * PAGE_STRIDE + (p % (BS // 4)) * 128
                pooled = tl.load(CACHE + base[:, None] + d[None, :],
                                 valid[:, None], 0)
                scores = tl.maximum(tl.dot(pooled, tl.trans(q)) * SCALE, 0.0)
                # Keep the query boundary explicit while testing the native
                # 32-head reduction against the wider dot's register layout.
                weighted = (scores * weights[None, :]).reshape(16, BQ, 32)
                logits = tl.sum(weighted, axis=2)
                row_valid = p[:, None] < counts[None, :]
                written = p[:, None] < tl.cdiv(counts[None, :], 16) * 16
                tl.store(OUT + rows[None, :].to(tl.int64) * MAX_POOLS + p[:, None],
                         tl.where(row_valid, logits, float("-inf")),
                         live[None, :] & written & (p < MAX_POOLS)[:, None])
    elif FALLBACK:
        for offset in tl.static_range(BQ):
            r = row0 + offset
            if r < ROWS:
                _independent_row(Q, W, CACHE, BT, ROW_REQ, VISIBLE, OUT, r,
                                 MAX_POOLS, BT_STRIDE, PAGE_STRIDE, SCALE, BS)


def validate_query_tile_inputs(q, weights, cache, block_table, row_req, visible, out,
                               query_tile, programs):
    assert query_tile in (2, 4) and programs > 0
    assert q.shape[1:] == (32, 128) and cache.shape[2] == 64
    assert q.dtype == cache.dtype == torch.bfloat16
    assert weights.dtype == out.dtype == torch.float32
    assert weights.shape == q.shape[:2]
    assert row_req.shape == visible.shape == (q.shape[0],)
    assert row_req.dtype == visible.dtype == block_table.dtype == torch.int32
    assert cache.shape[1] % 4 == 0 and cache.stride()[1:] == (64, 1)
    assert block_table.stride(1) == 1
    assert out.shape[0] == q.shape[0] and out.shape[1] > 0
    assert all(t.is_contiguous() for t in (q, weights, row_req, visible, out))


def query_tiled_pool_logits(q, weights, cache, block_table, row_req, visible, out,
                           query_tile=4, programs=128):
    validate_query_tile_inputs(q, weights, cache, block_table, row_req, visible, out,
                               query_tile, programs)
    if q.shape[0] == 0:
        return
    _query_tile[(triton.cdiv(q.shape[0], query_tile),
                 min(programs, triton.cdiv(out.shape[1], 16)))](
        q, weights, cache, block_table, row_req, visible, out, q.shape[0],
        out.shape[1], block_table.stride(0), cache.stride(0), 128**-0.5,
        cache.shape[1], query_tile, num_warps=4,
    )


def compile_only(block_size=4608):
    import json

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = dict(Q="*bf16", W="*fp32", CACHE="*bf16", BT="*i32",
                     ROW_REQ="*i32", VISIBLE="*i32", OUT="*fp32", ROWS="i32",
                     MAX_POOLS="i32", BT_STRIDE="i32", PAGE_STRIDE="i32",
                     SCALE="fp32")
    for tile in (2, 4):
        kernel = triton.compile(
            ASTSource(_query_tile, signature,
                      constexprs=dict(BS=block_size, BQ=tile, FALLBACK=True)),
            target=GPUTarget("cuda", 80, 32), options=dict(num_warps=4),
        )
        print(json.dumps(dict(query_tile=tile, block_size=block_size,
                              shared_bytes=kernel.metadata.shared,
                              scope="Offline SM80 compile only; no GPU validation")),
              flush=True)


if __name__ == "__main__":
    compile_only()
