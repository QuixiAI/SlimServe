# SPDX-License-Identifier: Apache-2.0
"""Small-batch GLM mHC: split FP32 projections, fused Sinkhorn/pre/RMSNorm.

One token and 256 input channels per partial program avoids the native
cooperative-grid barrier and the register-heavy padded tensor-core tile.
Checkpoint weights stay FP32. Both BF16 rounding boundaries are preserved.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mhc_partials(X, RES, POST, COMB, FN, RES_OUT, PART, FUSED_POST: tl.constexpr):
    split, token = tl.program_id(0), tl.program_id(1)
    k = split * 256 + tl.arange(0, 256)
    n = tl.arange(0, 32)
    if FUSED_POST:
        stream, dim = k // 4096, k % 4096
        x = tl.load(X + token * 4096 + dim).to(tl.float32)
        post = tl.load(POST + token * 4 + stream)
        value = tl.full((256,), 0, tl.float32)
        for source in tl.static_range(4):
            mix = tl.load(COMB + token * 16 + source * 4 + stream)
            r = tl.load(RES + token * 16384 + source * 4096 + dim).to(tl.float32)
            # Match the native kernel's first multiply/FMA rounding order;
            # reversing these terms changes occasional BF16 boundary values.
            value = tl.fma(post, x, mix * r) if source == 0 else tl.fma(mix, r, value)
        rounded = value.to(tl.bfloat16)
        tl.store(RES_OUT + token * 16384 + k, rounded)
        a = rounded.to(tl.float32)
    else:
        a = tl.load(RES + token * 16384 + k).to(tl.float32)
    w = tl.load(FN + n[:, None] * 16384 + k[None, :], n[:, None] < 24, 0)
    projection = tl.sum(w * a[None, :], axis=1)
    tl.store(PART + token * 64 * 32 + split * 32 + n, projection, n < 24)
    squares = tl.sum(a * a, axis=0)
    tl.store(PART + token * 64 * 32 + split * 32 + 24, squares)


@triton.jit
def _mhc_finalize(
    PART,
    RES,
    SCALE,
    BASE,
    POST,
    COMB,
    OUT,
    NORM,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    ITERATIONS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    DO_NORM: tl.constexpr,
):
    t = tl.program_id(0)
    s, n = tl.arange(0, 64), tl.arange(0, 32)
    values = tl.load(
        PART + t * 64 * 32 + s[:, None] * 32 + n[None, :], n[None, :] < 25, 0
    )
    mixes = tl.sum(values, axis=0)
    sq = tl.sum(tl.where(n == 24, mixes, 0), axis=0)
    inv = tl.rsqrt(sq / 16384 + RMS_EPS)
    h = tl.arange(0, 4)
    scale0, scale1, scale2 = tl.load(SCALE), tl.load(SCALE + 1), tl.load(SCALE + 2)
    pre = tl.sigmoid(tl.gather(mixes, h, 0) * inv * scale0 + tl.load(BASE + h)) + HC_EPS
    post = POST_MULT * tl.sigmoid(
        tl.gather(mixes, h + 4, 0) * inv * scale1 + tl.load(BASE + h + 4)
    )
    tl.store(POST + t * 4 + h, post)
    i = tl.arange(0, 16)
    matrix = (
        tl.gather(mixes, i + 8, 0) * inv * scale2 + tl.load(BASE + i + 8)
    ).reshape(4, 4)
    matrix = tl.exp(matrix - tl.max(matrix, axis=1)[:, None])
    matrix = matrix / tl.sum(matrix, axis=1)[:, None] + HC_EPS
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)
    for _ in range(ITERATIONS - 1):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + HC_EPS)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)
    tl.store(COMB + t * 16 + i, matrix.reshape(16))
    d = tl.arange(0, 4096)
    layer = tl.full((4096,), 0, tl.float32)
    for stream in tl.static_range(4):
        coeff = tl.sum(tl.where(h == stream, pre, 0), axis=0)
        residual = tl.load(RES + t * 16384 + stream * 4096 + d).to(tl.float32)
        layer = tl.fma(coeff, residual, layer)
    rounded = layer.to(tl.bfloat16)
    if DO_NORM:
        v = rounded.to(tl.float32)
        norm = tl.rsqrt(tl.sum(v * v, axis=0) / 4096 + NORM_EPS)
        rounded = (v * norm * tl.load(NORM + d).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + t * 4096 + d, rounded)


def mhc_transition(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    post: torch.Tensor | None,
    comb: torch.Tensor | None,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.shape[0]
    assert residual.shape == (m, 4, 4096) and fn.shape == (24, 16384)
    assert 0 < m <= 64 and sinkhorn_iters >= 1
    assert residual.dtype == torch.bfloat16 and residual.is_cuda
    assert fn.dtype == scale.dtype == base.dtype == torch.float32
    assert scale.shape == (3,) and base.shape == (24,)
    assert all(
        t.is_contiguous() and t.device == residual.device
        for t in (residual, fn, scale, base)
    )
    if x is not None:
        assert post is not None and comb is not None
        assert x.shape == (m, 4096) and x.dtype == residual.dtype
        assert post.numel() == m * 4 and comb.numel() == m * 16
        assert post.dtype == comb.dtype == torch.float32
        assert all(
            t.is_contiguous() and t.device == residual.device for t in (x, post, comb)
        )
    if norm_weight is not None:
        assert norm_weight.shape == (4096,) and norm_weight.dtype == residual.dtype
        assert norm_weight.is_contiguous() and norm_weight.device == residual.device
    res_out = torch.empty_like(residual) if x is not None else residual
    partial = torch.empty((m, 64, 32), device=residual.device, dtype=torch.float32)
    post_out = torch.empty((m, 4), device=residual.device, dtype=torch.float32)
    comb_out = torch.empty((m, 4, 4), device=residual.device, dtype=torch.float32)
    out = torch.empty((m, 4096), device=residual.device, dtype=residual.dtype)
    _mhc_partials[(64, m)](
        x, residual, post, comb, fn, res_out, partial, x is not None, num_warps=4
    )
    _mhc_finalize[(m,)](
        partial,
        res_out,
        scale,
        base,
        post_out,
        comb_out,
        out,
        norm_weight,
        rms_eps,
        hc_eps,
        post_mult,
        sinkhorn_iters,
        norm_eps,
        norm_weight is not None,
        num_warps=4,
    )
    return res_out, post_out, comb_out, out
