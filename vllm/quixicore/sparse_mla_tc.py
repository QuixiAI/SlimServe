# SPDX-License-Identifier: Apache-2.0
"""Owned opt-in NoPE sparse MLA tensor-core implementation for SM80.

Unchanged BF16 query/cache, global logical indices and strided physical pages.
Each partition shares gathered KV across heads. Softmax remains FP32; two
BF16 probability components feed two tensor-core value products rather than
rounding the whole probability to one BF16 value. Numerical equivalence and
serving performance require separate correctness gates and profile A/B.
"""

import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sparse_tc_part(
    Q,
    CACHE,
    BT,
    INDICES,
    TLEN,
    PART_OUT,
    PART_M,
    PART_L,
    BT_STRIDE,
    PAGE_STRIDE,
    CACHE_PAGES,
    SCALE,
    KV_SCALE,
    H: tl.constexpr,
    BS: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    PARTS: tl.constexpr,
    SPLIT: tl.constexpr,
    FP8: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    h, d = tl.arange(0, 16), tl.arange(0, 512)
    j = part * SPLIT + tl.arange(0, SPLIT)
    count = tl.minimum(tl.load(TLEN + row), MAX_TOPK)
    logical = tl.load(
        INDICES + row.to(tl.int64) * MAX_TOPK + j,
        (j < count) & (j < MAX_TOPK),
        -1,
    )
    valid = (j < count) & (logical >= 0) & (logical < BT_STRIDE * BS)
    page = tl.load(BT + row.to(tl.int64) * BT_STRIDE + logical // BS, valid, -1)
    valid = valid & (page >= 0) & (page < CACHE_PAGES)
    base = (row.to(tl.int64) * PARTS + part) * H + h
    # Expanded short-context lists have large -1 gaps before the local
    # tail. Do not run tensor-core products for an entirely empty tile.
    if tl.sum(valid.to(tl.int32), axis=0) == 0:
        tl.store(PART_OUT + base[:, None] * 512 + d[None, :], 0.0, h[:, None] < H)
        tl.store(PART_M + base, float("-inf"), h < H)
        tl.store(PART_L + base, 0.0, h < H)
        return
    if FP8:
        # e4m3 bytes -> fp32 by bit assembly (sm80 has no fp8 hardware):
        # normal: sign | (exp - 7 + 127) << 23 | man << 20;
        # subnormal (exp == 0): man * 2^-9; exp 15 & man 7 is NaN -> 0.
        # (A 16-bit assembly to fp16 measured 4.6x slower in Triton on
        # sm80, 2026-09-12; int32 is the fast form here.) kv_scale is folded
        # applied per element: folding it into q or the accumulator instead
        # measured 632 us vs 361 us at B=32 (Triton codegen), so it stays.
        raw = tl.load(
            CACHE
            + page[:, None].to(tl.int64) * PAGE_STRIDE
            + (logical[:, None] % BS) * 512
            + d[None, :],
            valid[:, None],
            0,
        ).to(tl.int32)
        sign = (raw >> 7) & 1
        exp = (raw >> 3) & 15
        man = raw & 7
        normal_bits = (sign << 31) | ((exp + 120) << 23) | (man << 20)
        normal = normal_bits.to(tl.float32, bitcast=True)
        sub = tl.where(sign == 1, -1.0, 1.0) * man.to(tl.float32) * 0.001953125
        val = tl.where(exp == 0, sub, normal)
        val = tl.where((exp == 15) & (man == 7), 0.0, val)
        kv = (val * KV_SCALE).to(tl.bfloat16)
    else:
        kv = tl.load(
            CACHE
            + page[:, None].to(tl.int64) * PAGE_STRIDE
            + (logical[:, None] % BS) * 512
            + d[None, :],
            valid[:, None],
            0,
        )
    q = tl.load(
        Q + (row.to(tl.int64) * H + h[:, None]) * 512 + d[None, :],
        h[:, None] < H,
        0,
    )
    scores = tl.dot(q, tl.trans(kv)) * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    maximum = tl.max(scores, axis=1)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    probabilities = tl.exp(scores - safe_max[:, None])
    denominator = tl.sum(probabilities, axis=1)
    high = probabilities.to(tl.bfloat16)
    low = (probabilities - high.to(tl.float32)).to(tl.bfloat16)
    accumulator = tl.dot(high, kv)
    accumulator = tl.dot(low, kv, accumulator)
    tl.store(PART_OUT + base[:, None] * 512 + d[None, :], accumulator, h[:, None] < H)
    tl.store(PART_M + base, maximum, h < H)
    tl.store(PART_L + base, denominator, h < H)


@triton.jit
def _sparse_tc_reduce(
    PART_OUT,
    PART_M,
    PART_L,
    OUT,
    H: tl.constexpr,
    PARTS: tl.constexpr,
    PAD_PARTS: tl.constexpr,
):
    row, head, feature_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    part = tl.arange(0, PAD_PARTS)
    d = feature_tile * 64 + tl.arange(0, 64)
    base = (row.to(tl.int64) * PARTS + part) * H + head
    maximum = tl.load(PART_M + base, part < PARTS, float("-inf"))
    denominator = tl.load(PART_L + base, part < PARTS, 0)
    global_max = tl.max(maximum, axis=0)
    global_max = tl.where(global_max == float("-inf"), 0.0, global_max)
    factors = tl.exp(maximum - global_max)
    normalizer = tl.sum(factors * denominator, axis=0)
    values = tl.load(
        PART_OUT + base[:, None] * 512 + d[None, :], (part < PARTS)[:, None], 0
    )
    numerator = tl.sum(values * factors[:, None], axis=0)
    result = tl.where(normalizer > 0, numerator / normalizer, 0.0)
    tl.store(OUT + (row.to(tl.int64) * H + head) * 512 + d, result)


def sparse_tc_nope(
    q, cache, block_table, indices, topk_length, scale, *, split=32, kv_scale=1.0
):
    """`cache` is the bf16 latent [pages, BS, 512] or its fp8 (e4m3) storage
    viewed as uint8 [pages, BS, 512]; fp8 is decoded in-kernel and scaled by
    kv_scale (the per-tensor fp8 MLA scale)."""
    if split not in (32, 64, 128):
        raise ValueError("split must be32,64 or128")
    assert q.ndim == 3 and q.shape[1] in (8, 16) and q.shape[2] == 512
    assert cache.ndim == 3 and cache.shape[2] == 512
    fp8 = cache.dtype == torch.uint8
    assert q.dtype == torch.bfloat16 and (fp8 or cache.dtype == torch.bfloat16)
    if fp8 and split > 64 and os.getenv("QC_SPARSE_TC_FP8_SPLIT128", "0") != "1":
        # The 16-bit e4m3 decode halves the tile footprint of the int32 one;
        # SPLIT=128 is re-tested behind this switch before it becomes default.
        split = 64
    assert cache.stride()[1:] == (512, 1) and cache.shape[1] > 0
    assert indices.shape[0] == block_table.shape[0] == q.shape[0]
    assert topk_length.shape == (q.shape[0],)
    assert all(t.dtype == torch.int32 for t in (block_table, indices, topk_length))
    assert all(t.is_contiguous() for t in (q, block_table, indices, topk_length))
    assert q.is_cuda and all(
        t.device == q.device for t in (cache, block_table, indices, topk_length)
    )
    rows, heads, _ = q.shape
    out = torch.empty_like(q)
    if rows == 0:
        return out
    assert indices.shape[1] > 0
    parts = triton.cdiv(indices.shape[1], split)
    partial = torch.empty(rows, parts, heads, 512, device=q.device, dtype=torch.float32)
    maxima = torch.empty(rows, parts, heads, device=q.device, dtype=torch.float32)
    denominators = torch.empty_like(maxima)
    _sparse_tc_part[(rows, parts)](
        q,
        cache,
        block_table,
        indices,
        topk_length,
        partial,
        maxima,
        denominators,
        block_table.stride(0),
        cache.stride(0),
        cache.shape[0],
        scale,
        float(kv_scale),
        heads,
        cache.shape[1],
        indices.shape[1],
        parts,
        split,
        fp8,
        num_warps=4,
        num_stages=1,
    )
    _sparse_tc_reduce[(rows, heads, 8)](
        partial,
        maxima,
        denominators,
        out,
        heads,
        parts,
        triton.next_power_of_2(parts),
        num_warps=4,
    )
    return out


@triton.jit
def _sparse_tc_rows(
    Q,
    CACHE,
    BT,
    INDICES,
    TLEN,
    OUT,
    BT_STRIDE,
    PAGE_STRIDE,
    CACHE_PAGES,
    SCALE,
    H: tl.constexpr,
    BS: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    TILE: tl.constexpr,
):
    """One program per query token, every head at once, online softmax over
    the token's index list in TILE-wide gathered tiles: no partition scratch,
    so it scales to prefill chunks of thousands of rows. Each gathered tile is
    read once for all heads (the SIMT kernel walks the list once per head).
    bf16 latents only: an fp8 main KV takes the native fp8 prefill kernel."""
    row = tl.program_id(0)
    h, d = tl.arange(0, 16), tl.arange(0, 512)
    count = tl.minimum(tl.load(TLEN + row), MAX_TOPK)
    # The query is held transposed ([512, 16]) so the gathered tile is the A
    # operand of both products and never needs a transposed shared-memory
    # copy (sm_120 has 99 KB of shared memory per block).
    q_t = tl.load(
        Q + (row.to(tl.int64) * H + h[None, :]) * 512 + d[:, None],
        h[None, :] < H,
        0,
    )
    running_max = tl.full([16], float("-inf"), tl.float32)
    running_sum = tl.zeros([16], tl.float32)
    accumulator = tl.zeros([16, 512], tl.float32)
    for start in range(0, count, TILE):
        j = start + tl.arange(0, TILE)
        logical = tl.load(
            INDICES + row.to(tl.int64) * MAX_TOPK + j,
            (j < count) & (j < MAX_TOPK),
            -1,
        )
        valid = (j < count) & (logical >= 0) & (logical < BT_STRIDE * BS)
        page = tl.load(BT + row.to(tl.int64) * BT_STRIDE + logical // BS, valid, -1)
        valid = valid & (page >= 0) & (page < CACHE_PAGES)
        kv = tl.load(
            CACHE
            + page[:, None].to(tl.int64) * PAGE_STRIDE
            + (logical[:, None] % BS) * 512
            + d[None, :],
            valid[:, None],
            0,
        )
        # scores_t[j, h] for the tile's keys j and the heads h.
        scores_t = tl.dot(kv, q_t) * SCALE
        scores_t = tl.where(valid[:, None], scores_t, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores_t, axis=0))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        rescale = tl.exp(running_max - safe_max)
        probabilities_t = tl.exp(scores_t - safe_max[None, :])
        running_sum = running_sum * rescale + tl.sum(probabilities_t, axis=0)
        accumulator = accumulator * rescale[:, None]
        high = probabilities_t.to(tl.bfloat16)
        low = (probabilities_t - high.to(tl.float32)).to(tl.bfloat16)
        accumulator = tl.dot(tl.trans(high), kv, accumulator)
        accumulator = tl.dot(tl.trans(low), kv, accumulator)
        running_max = new_max
    result = tl.where(running_sum[:, None] > 0, accumulator / running_sum[:, None], 0.0)
    tl.store(
        OUT + (row.to(tl.int64) * H + h[:, None]) * 512 + d[None, :],
        result.to(tl.bfloat16),
        h[:, None] < H,
    )


def sparse_tc_nope_rows(q, cache, block_table, indices, topk_length, scale, *, tile=64):
    """Prefill-shaped counterpart of `sparse_tc_nope` for bf16 latents: same
    numerics (fp32 softmax, split bf16 probabilities), one program per row
    over the whole index list, no partition scratch. `tile` is the gathered
    tile width (32 or 64 rows of the latent)."""
    if tile not in (32, 64):
        raise ValueError("tile must be 32 or 64")
    assert q.ndim == 3 and q.shape[1] in (8, 16) and q.shape[2] == 512
    assert cache.ndim == 3 and cache.shape[2] == 512
    assert q.dtype == torch.bfloat16 and cache.dtype == torch.bfloat16
    assert cache.stride()[1:] == (512, 1) and cache.shape[1] > 0
    assert indices.shape[0] == block_table.shape[0] == q.shape[0]
    assert topk_length.shape == (q.shape[0],)
    assert all(t.dtype == torch.int32 for t in (block_table, indices, topk_length))
    assert all(t.is_contiguous() for t in (q, block_table, indices, topk_length))
    assert q.is_cuda and all(
        t.device == q.device for t in (cache, block_table, indices, topk_length)
    )
    rows, heads, _ = q.shape
    out = torch.empty_like(q)
    if rows == 0:
        return out
    assert indices.shape[1] > 0
    _sparse_tc_rows[(rows,)](
        q,
        cache,
        block_table,
        indices,
        topk_length,
        out,
        block_table.stride(0),
        cache.stride(0),
        cache.shape[0],
        scale,
        heads,
        cache.shape[1],
        indices.shape[1],
        tile,
        num_warps=4,
        num_stages=1,
    )
    return out


def compile_only():
    import json

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = {
        "Q": "*bf16",
        "CACHE": "*bf16",
        "BT": "*i32",
        "INDICES": "*i32",
        "TLEN": "*i32",
        "PART_OUT": "*fp32",
        "PART_M": "*fp32",
        "PART_L": "*fp32",
        "BT_STRIDE": "i32",
        "PAGE_STRIDE": "i32",
        "CACHE_PAGES": "i32",
        "SCALE": "fp32",
        "KV_SCALE": "fp32",
    }
    for heads in (8, 16):
        for split in (32, 64, 128):
            parts = triton.cdiv(2080, split)
            constants = dict(
                H=heads, BS=576, MAX_TOPK=2080, PARTS=parts, SPLIT=split, FP8=False
            )
            kernel = triton.compile(
                ASTSource(_sparse_tc_part, signature, constexprs=constants),
                target=GPUTarget("cuda", 80, 32),
                options={"num_warps": 4, "num_stages": 1},
            )
            reduction = triton.compile(
                ASTSource(
                    _sparse_tc_reduce,
                    dict(PART_OUT="*fp32", PART_M="*fp32", PART_L="*fp32", OUT="*bf16"),
                    constexprs=dict(
                        H=heads, PARTS=parts, PAD_PARTS=triton.next_power_of_2(parts)
                    ),
                ),
                target=GPUTarget("cuda", 80, 32),
                options={"num_warps": 4},
            )
            print(
                json.dumps(
                    dict(
                        heads=heads,
                        split=split,
                        part_shared=kernel.metadata.shared,
                        reduce_shared=reduction.metadata.shared,
                        part_hash=kernel.hash,
                        reduce_hash=reduction.hash,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    compile_only()
