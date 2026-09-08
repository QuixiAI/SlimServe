# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head-batched sparse MLA attention for prefill chunks (NoPE bf16 latents).

`mla_decode_fp8_v<true, false, 512, 512, 0, 0>` walks a query token's
top-k list with one warp per (head, token): every head re-reads the same
gathered 1 KB rows and does its dot products on CUDA cores, which is the
right shape for a decode step of a few tokens and the wrong one for a
7000-token prefill chunk (11 x 5.75 ms of the c8 step). Here one program
owns one query token and all of its heads: the 32 heads are the M
dimension of two tensor-core products per 32-key tile (S = Q K^T, then
O += P V with V the same gathered rows), with an online softmax per head.
Indices are request-local token positions resolved through the token's
block-table row exactly as the decode kernel resolves them; -1 entries
and out-of-table entries are skipped; a row with no valid entry is 0.

Numerics against the decode walk: the score products are the same bf16
pairs summed in a different order, and the probabilities go through the
P V product in fp16 (2^-11) instead of fp32; the outputs agree to one
bf16 rounding of the result (max 1.6e-2 at values near 4, mean 1e-5).
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

# Kill switch for A/B runs only (read here, so not a torch-compile cache
# factor): VLLM_MLA_SPARSE_PREFILL_TC=0 keeps prefill chunks on the decode
# walk.
ENABLED = os.getenv("VLLM_MLA_SPARSE_PREFILL_TC", "1") != "0"

LATENT = 512
_BLOCK_N = 32  # keys per tile: 64 does not fit the 100 KB shared budget
_NUM_WARPS = 4
_NUM_STAGES = 2


@triton.jit(do_not_specialize=["bt_stride", "max_topk"])
def _sparse_mla_prefill_kernel(
    q_ptr,  # [B, H, D] bf16
    cache_ptr,  # bf16 pages: block * page_stride + slot * D
    bt_ptr,  # [B, bt_stride] int32: the token's request's block table
    idx_ptr,  # [B, max_topk] int32 request-local positions, -1 = skip
    tlen_ptr,  # [B] int32 entries to walk (last valid + 1)
    out_ptr,  # [B, H, D] bf16
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
    q = tl.load(q_ptr + (b.to(tl.int64) * H + h)[:, None] * D + d[None, :])
    tlen = tl.load(tlen_ptr + b)
    tlen = tl.minimum(tl.maximum(tlen, 0), max_topk)
    # a finite floor instead of -inf so an all-skipped tile keeps the
    # running max finite (exp(m_i - m_new) stays 1, not nan)
    m_i = tl.full((H,), -1.0e30, tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((H, D), tl.float32)
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
        s = tl.dot(q, tl.trans(k)) * scale  # [H, BLOCK_N] fp32
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), k.to(tl.float16))
        m_i = m_new
    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(
        out_ptr + (b.to(tl.int64) * H + h)[:, None] * D + d[None, :],
        out.to(tl.bfloat16),
    )


def supports(q: torch.Tensor) -> bool:
    """The kernel takes the heads as a tensor-core M dimension: a power of
    two from 16 up (TP <= 8 for GLM-5.3-Flash's 128 heads)."""
    H = q.shape[1]
    return q.shape[-1] == LATENT and H >= 16 and (H & (H - 1)) == 0


def sparse_mla_prefill_nope(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    block_size: int,
    scale: float,
    page_stride_bytes: int = 0,
) -> torch.Tensor:
    """Same contract as quixicore_ops.mla_decode_bf16_sparse_nope
    (unpartitioned): returns [tokens, heads, 512] bf16."""
    B, H, D = q.shape
    assert supports(q), q.shape
    assert q.is_contiguous() and indices.is_contiguous()
    assert block_table.is_contiguous() and kv_cache.stride(-1) == 1
    page_stride = (
        page_stride_bytes // kv_cache.element_size()
        if page_stride_bytes
        else block_size * D
    )
    out = torch.empty_like(q)
    if B == 0:
        return out
    _sparse_mla_prefill_kernel[(B,)](
        q,
        kv_cache,
        block_table,
        indices,
        topk_length,
        out,
        indices.shape[1],
        block_table.shape[1],
        page_stride,
        block_size,
        scale,
        H=H,
        D=D,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out
