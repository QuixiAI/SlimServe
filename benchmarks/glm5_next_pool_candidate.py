"""Diagnostic short-context pooled-score schedule; not a serving dispatch.

One program owns one pool at a time, distributing short contexts across SMs
instead of packing 16 pools into each of only 16 active CTAs at 1000 tokens.
The serving tensor-core schedule remains the long-context reference.
"""

import triton
import triton.language as tl


@triton.jit
def pooled_logits_one_pool(
    q_ptr,
    w_ptr,
    ape_ptr,
    cache_ptr,
    bt_ptr,
    row_req_ptr,
    vis_ptr,
    out_ptr,
    max_pools,
    bt_stride,
    page_stride,
    softmax_scale,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KP: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    programs = tl.num_programs(1)
    n_pools = tl.load(vis_ptr + r) // KP
    if pb < n_pools:
        request = tl.load(row_req_ptr + r).to(tl.int64)
        member = tl.arange(0, KP)
        channel = tl.arange(0, D)
        head = tl.arange(0, H)
        ape = tl.load(ape_ptr + member[:, None] * D + channel[None, :])
        q = tl.load(q_ptr + (r.to(tl.int64) * H + head[:, None]) * D + channel[None, :])
        w = tl.load(w_ptr + r * H + head)
        for pool in range(pb, n_pools, programs):
            token = pool * KP + member
            block = tl.load(bt_ptr + request * bt_stride + token // BLOCK_SIZE)
            address = block.to(tl.int64) * page_stride + token % BLOCK_SIZE * ROW
            k = tl.load(cache_ptr + address[:, None] + channel[None, :]).to(tl.float32)
            g = tl.load(cache_ptr + address[:, None] + D + channel[None, :]).to(
                tl.float32
            )
            gates = g + ape
            exp = tl.exp(gates - tl.max(gates, axis=0)[None, :])
            probs = exp / tl.sum(exp, axis=0)[None, :]
            key = tl.sum(probs * k, axis=0).to(tl.bfloat16).to(tl.float32)
            score = tl.sum(q.to(tl.float32) * key[None, :], axis=1)
            logit = tl.sum(tl.maximum(score * softmax_scale, 0.0) * w, axis=0)
            tl.store(out_ptr + r.to(tl.int64) * max_pools + pool, logit)
