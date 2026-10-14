# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: overlap deferred mHC coefficients with their consumer.

Not imported by serving. The pre-mix/RMSNorm output feeds a projection now;
post and Sinkhorn coefficients are consumed after that attention/MLP site.
DSV4's urgent/deferred split is the precedent, but this experiment neither
fuses all-reduce nor changes channel ownership. Persistent test buffers and
an explicit stream join keep the prototype's graph/lifetime contract local.
"""

import torch

from vllm.model_executor.layers.glm5_next_mhc_triton import _mhc_partials
from vllm.triton_utils import tl, triton


@triton.jit
def _finalize_phase(
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
    DEFERRED: tl.constexpr,
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
    if DEFERRED:
        scale1, scale2 = tl.load(SCALE + 1), tl.load(SCALE + 2)
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
    else:
        pre = (
            tl.sigmoid(
                tl.gather(mixes, h, 0) * inv * tl.load(SCALE) + tl.load(BASE + h)
            )
            + HC_EPS
        )
        d = tl.arange(0, 4096)
        layer = tl.full((4096,), 0, tl.float32)
        for stream in tl.static_range(4):
            coeff = tl.sum(tl.where(h == stream, pre, 0), axis=0)
            residual = tl.load(RES + t * 16384 + stream * 4096 + d).to(tl.float32)
            layer = tl.fma(coeff, residual, layer)
        rounded = layer.to(tl.bfloat16)
        v = rounded.to(tl.float32)
        norm = tl.rsqrt(tl.sum(v * v, axis=0) / 4096 + NORM_EPS)
        rounded = (v * norm * tl.load(NORM + d).to(tl.float32)).to(tl.bfloat16)
        tl.store(OUT + t * 4096 + d, rounded)


class DeferredTransition:
    """Fixed-shape diagnostic; the caller must not invoke it concurrently."""

    def __init__(
        self,
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        norm,
        weight,
        *,
        projection_fp32=False,
    ):
        self.x, self.residual, self.post, self.comb = x, residual, post, comb
        self.fn, self.scale, self.base, self.norm = fn, scale, base, norm
        self.weight = weight
        self.m = x.shape[0]
        assert residual.shape == (self.m, 4, 4096)
        assert weight.ndim == 2 and weight.shape[1] == 4096
        assert all(
            t.is_contiguous()
            for t in (x, residual, post, comb, fn, scale, base, norm, weight)
        )
        self.res_out = torch.empty_like(residual)
        self.partial = torch.empty(self.m, 64, 32, device=x.device, dtype=torch.float32)
        self.post_out = torch.empty(self.m, 4, device=x.device, dtype=torch.float32)
        self.comb_out = torch.empty(self.m, 4, 4, device=x.device, dtype=torch.float32)
        self.out = torch.empty_like(x)
        self.projection_fp32 = projection_fp32
        self.projected = torch.empty(
            self.m,
            weight.shape[0],
            device=x.device,
            dtype=torch.float32 if projection_fp32 else x.dtype,
        )
        self.side = torch.cuda.Stream(device=x.device)

    def phase(self, deferred):
        _finalize_phase[(self.m,)](
            self.partial,
            self.res_out,
            self.scale,
            self.base,
            self.post_out,
            self.comb_out,
            self.out,
            self.norm,
            1e-5,
            1e-6,
            2.0,
            20,
            1e-5,
            deferred,
            num_warps=4,
        )

    def run(self, *, overlap):
        _mhc_partials[(64, self.m)](
            self.x,
            self.residual,
            self.post,
            self.comb,
            self.fn,
            self.res_out,
            self.partial,
            True,
            num_warps=4,
        )
        main = torch.cuda.current_stream(self.x.device)
        if overlap:
            # Producer -> deferred branch. The urgent branch and its GEMM
            # stay on main; join only after GEMM, before returning any result.
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                self.phase(True)
        self.phase(False)
        if not overlap:
            self.phase(True)
        if self.projection_fp32:
            torch.mm(
                self.out, self.weight.T, out_dtype=torch.float32, out=self.projected
            )
        else:
            torch.mm(self.out, self.weight.T, out=self.projected)
        if overlap:
            main.wait_stream(self.side)
        return self.res_out, self.post_out, self.comb_out, self.out, self.projected
