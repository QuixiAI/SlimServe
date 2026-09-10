# SPDX-License-Identifier: Apache-2.0
"""mHC schedule diagnostics: SIMT and rejected TF32x3 tiles, never serving."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partials(
    X,
    RES,
    POST,
    COMB,
    FN,
    RES_OUT,
    PART,
    M: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    FUSED_POST: tl.constexpr,
):
    split, tile = tl.program_id(0), tl.program_id(1)
    m = tile * BM + tl.arange(0, BM)
    k = split * BK + tl.arange(0, BK)
    n = tl.arange(0, 32)
    if FUSED_POST:
        stream, dim = k // 4096, k % 4096
        x = tl.load(X + m[:, None] * 4096 + dim[None, :], m[:, None] < M, 0).to(
            tl.float32
        )
        post = tl.load(POST + m[:, None] * 4 + stream[None, :], m[:, None] < M, 0)
        value = x * post
        for source in tl.static_range(4):
            mix = tl.load(
                COMB + m[:, None] * 16 + source * 4 + stream[None, :], m[:, None] < M, 0
            )
            r = tl.load(
                RES + m[:, None] * 16384 + source * 4096 + dim[None, :],
                m[:, None] < M,
                0,
            ).to(tl.float32)
            value = tl.fma(post, x, mix * r) if source == 0 else tl.fma(mix, r, value)
        rounded = value.to(tl.bfloat16)
        tl.store(RES_OUT + m[:, None] * 16384 + k[None, :], rounded, m[:, None] < M)
        a = rounded.to(tl.float32)
    else:
        a = tl.load(RES + m[:, None] * 16384 + k[None, :], m[:, None] < M, 0).to(
            tl.float32
        )
    w = tl.load(FN + n[None, :] * 16384 + k[:, None], n[None, :] < 24, 0)
    if BM == 1:
        projection = tl.sum(tl.trans(w) * a, axis=1)[None, :]
    else:
        projection = tl.dot(a, w, input_precision="tf32x3")
    splits: tl.constexpr = 16384 // BK
    ptr = PART + m[:, None] * splits * 32 + split * 32 + n[None, :]
    tl.store(ptr, projection, (m[:, None] < M) & (n[None, :] < 24))
    squares = tl.sum(a * a, axis=1)
    tl.store(PART + m * splits * 32 + split * 32 + 24, squares, m < M)


@triton.jit
def _finalize(
    PART,
    RES,
    SCALE,
    BASE,
    POST,
    COMB,
    OUT,
    NORM,
    SPLITS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERATIONS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    DO_NORM: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.arange(0, SPLITS)
    n = tl.arange(0, 32)
    values = tl.load(
        PART + t * SPLITS * 32 + s[:, None] * 32 + n[None, :], n[None, :] < 25, 0
    )
    mixes = tl.sum(values, axis=0)
    sq = tl.sum(tl.where(n == 24, mixes, 0), axis=0)
    inv = tl.rsqrt(sq / 16384 + RMS_EPS)
    h = tl.arange(0, 4)
    scale0, scale1, scale2 = tl.load(SCALE), tl.load(SCALE + 1), tl.load(SCALE + 2)
    pre = tl.sigmoid(tl.gather(mixes, h, 0) * inv * scale0 + tl.load(BASE + h)) + HC_EPS
    post = 2 * tl.sigmoid(
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


def transition(
    x,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    weight=None,
    *,
    bk=512,
    bm=1,
    diagnostics=None,
):
    m = residual.shape[0]
    assert residual.shape == (m, 4, 4096) and fn.shape == (24, 16384)
    assert fn.dtype == torch.float32 and residual.dtype == torch.bfloat16
    assert bk in (64, 128, 256, 512, 1024)
    res_out = torch.empty_like(residual) if x is not None else residual
    partial = torch.empty(
        m, 16384 // bk, 32, device=residual.device, dtype=torch.float32
    )
    post_out = torch.empty(m, 4, device=residual.device, dtype=torch.float32)
    comb_out = torch.empty(m, 4, 4, device=residual.device, dtype=torch.float32)
    out = torch.empty(m, 4096, device=residual.device, dtype=torch.bfloat16)
    partial_kernel = _partials[(16384 // bk, triton.cdiv(m, bm))](
        x,
        residual,
        post,
        comb,
        fn,
        res_out,
        partial,
        m,
        bk,
        bm,
        x is not None,
        num_warps=4,
        num_stages=1,
    )
    finalize_kernel = _finalize[(m,)](
        partial,
        res_out,
        scale,
        base,
        post_out,
        comb_out,
        out,
        weight,
        16384 // bk,
        1e-5,
        1e-6,
        20,
        1e-5,
        weight is not None,
        num_warps=4,
    )
    if diagnostics is not None:
        for name, kernel in (
            ("partial", partial_kernel),
            ("finalize", finalize_kernel),
        ):
            diagnostics[name] = {
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared": kernel.metadata.shared,
            }
    return res_out, post_out, comb_out, out
