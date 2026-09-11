# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planar CT GEMM (qgemm_nvfp4_planar / qgemm_fp8ch): float64 oracle.

The M > 8 band of the compressed-tensors Metal path: weights decode
in-kernel from the checkpoint's planar buffers (no bf16 materialization,
no per-call dequant copy). The oracle dequantizes in float64 on the CPU
and compares at serving-relevant shapes, including K from the real
Qwen3.8-27B projections (5120, 17408) and prefill-width M.
"""

import pytest
import torch

from vllm.quixicore import quixicore_ops

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal required"
)

E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float64,
)


def _nvfp4_case(n, k, m, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    wq = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    # Group scales as genuine e4m3 bytes of plausible magnitudes.
    ws_f = torch.rand(n, k // 16, generator=g) * 0.5 + 0.01
    ws = ws_f.to(torch.float8_e4m3fn).view(torch.uint8)
    gs = torch.tensor([0.731], dtype=torch.float32)
    x = (torch.randn(m, k, generator=g) * 0.5).to(dtype)

    lo = E2M1[(wq & 0x0F).long()]
    hi = E2M1[(wq >> 4).long()]
    codes = torch.stack([lo, hi], dim=-1).reshape(n, k)
    scales = ws.view(torch.float8_e4m3fn).to(torch.float64)
    w_ref = codes * scales.repeat_interleave(16, dim=1) * float(gs)
    ref = x.to(torch.float64) @ w_ref.T
    return wq, ws, gs, x, ref


def _fp8ch_case(n, k, m, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_f = torch.randn(n, k, generator=g) * 0.05
    wq = w_f.to(torch.float8_e4m3fn).view(torch.uint8)
    ws = (torch.rand(n, generator=g) * 0.4 + 0.005).to(torch.float32)
    x = (torch.randn(m, k, generator=g) * 0.5).to(dtype)
    w_ref = wq.view(torch.float8_e4m3fn).to(torch.float64) * ws.to(
        torch.float64
    ).unsqueeze(1)
    ref = x.to(torch.float64) @ w_ref.T
    return wq, ws, x, ref


def _check(out, ref, dtype):
    got = out.cpu().to(torch.float64)
    # Element tolerance: half-staged operands, fp32 accumulate, T output
    # cast. bf16 output rounding dominates its bound.
    tol = 2.5e-2 if dtype == torch.bfloat16 else 8e-3
    denom = ref.abs().clamp_min(ref.abs().mean() + 1e-6)
    maxrel = ((got - ref).abs() / denom).max().item()
    assert maxrel < tol, f"maxrel {maxrel:.4e} over tol {tol}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "n,k,m",
    [
        (128, 512, 9),  # first GEMM-band M
        (128, 512, 33),  # M-tile boundary + 1
        (256, 5120, 16),  # real qkv-class K
        (256, 5120, 64),
        (192, 5120, 257),  # N and M tails together
        (64, 17408, 32),  # real down_proj K
        (4096, 5120, 12),  # lm_head-class N slice
    ],
)
def test_nvfp4_gemm_matches_float64_oracle(dtype, n, k, m):
    wq, ws, gs, x, ref = _nvfp4_case(n, k, m, dtype)
    out = quixicore_ops.nvfp4_mul_mat_vec(
        wq.to("mps"), x.to("mps"), ws.to("mps"), gs.to("mps")
    )
    _check(out, ref, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "n,k,m",
    [
        (128, 512, 9),
        (128, 512, 33),
        (256, 5120, 16),
        (192, 5120, 257),
        (64, 17408, 32),
        (4096, 5120, 12),
    ],
)
def test_fp8ch_gemm_matches_float64_oracle(dtype, n, k, m):
    wq, ws, x, ref = _fp8ch_case(n, k, m, dtype)
    out = quixicore_ops.fp8ch_mul_mat_vec(wq.to("mps"), x.to("mps"), ws.to("mps"))
    _check(out, ref, dtype)


def test_prefill_width_m2176():
    """Full chunked-prefill width against a float32 reference."""
    dtype = torch.bfloat16
    wq, ws, gs, x, ref = _nvfp4_case(512, 5120, 2176, dtype)
    out = quixicore_ops.nvfp4_mul_mat_vec(
        wq.to("mps"), x.to("mps"), ws.to("mps"), gs.to("mps")
    )
    _check(out, ref, dtype)


def test_repeat_call_bit_stable():
    wq, ws, gs, x, _ = _nvfp4_case(256, 5120, 33, torch.bfloat16)
    args = (wq.to("mps"), x.to("mps"), ws.to("mps"), gs.to("mps"))
    a = quixicore_ops.nvfp4_mul_mat_vec(*args).cpu().clone()
    b = quixicore_ops.nvfp4_mul_mat_vec(*args).cpu()
    assert torch.equal(a, b)


def test_gemv_band_still_routes_and_agrees():
    """M <= 8 stays on the GEMV twins; the GEMM band must agree with the
    GEMV band on the shared rows of a split batch (not bit-identical -
    different accumulation trees - but within the same tolerance)."""
    dtype = torch.float16
    wq, ws, gs, x, ref = _nvfp4_case(128, 2048, 12, dtype)
    args_w = (wq.to("mps"), ws.to("mps"), gs.to("mps"))
    full = quixicore_ops.nvfp4_mul_mat_vec(args_w[0], x.to("mps"), args_w[1], args_w[2])
    head = quixicore_ops.nvfp4_mul_mat_vec(
        args_w[0], x[:8].contiguous().to("mps"), args_w[1], args_w[2]
    )
    _check(full, ref, dtype)
    _check(head, ref[:8], dtype)
