# SPDX-License-Identifier: Apache-2.0
"""Quarantined c1 dense BF16 projection experiment; not serving-imported.

The current TP8 KDA input GEMM reads ~27MB in22us plus split-K reduction.
The owned DSV4 reference uses one CTA/output row and vector loads. This
candidate groups several rows per CTA, reuses input lanes, and accumulates
over the entire4096-wide K without a second reduction launch. No repack,
new precision format or weight cache. Both numerical and HBM-churn timing
gates are required; this is not presumed faster than cuBLAS.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _bf16_gemv(X, WEIGHT, OUT, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for start in range(0, 4096, BK):
        x = tl.load(X + start + k).to(tl.float32)
        w = tl.load(
            WEIGHT + row[:, None] * 4096 + start + k[None, :], row[:, None] < N, 0
        ).to(tl.float32)
        acc = tl.fma(w, x[None, :], acc)
    value = tl.sum(acc, axis=1)
    tl.store(OUT + row, value, row < N)


def gemv(x, weight, out, *, bn=4, bk=512, warps=4):
    assert x.shape == (1, 4096) and weight.shape[1] == 4096
    assert out.shape == (1, weight.shape[0])
    assert x.dtype == weight.dtype == out.dtype == torch.bfloat16
    assert all(t.is_contiguous() for t in (x, weight, out))
    _bf16_gemv[(triton.cdiv(weight.shape[0], bn),)](
        x, weight, out, weight.shape[0], bn, bk, num_warps=warps
    )


def compile_only():
    import json

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    for bn in (2, 4, 8):
        for bk in (256, 512, 1024):
            kernel = triton.compile(
                ASTSource(
                    _bf16_gemv,
                    dict(X="*bf16", WEIGHT="*bf16", OUT="*bf16"),
                    constexprs=dict(N=3336, BN=bn, BK=bk),
                ),
                target=GPUTarget("cuda", 80, 32),
                options={"num_warps": 4},
            )
            print(
                json.dumps(
                    dict(bn=bn, bk=bk, shared=kernel.metadata.shared, hash=kernel.hash)
                ),
                flush=True,
            )


if __name__ == "__main__":
    compile_only()
