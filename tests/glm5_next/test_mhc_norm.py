# SPDX-License-Identifier: Apache-2.0
"""GLM's mHC normalization seam, including the BF16 intermediate rounding."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.quixicore.ops import quixicore_ops as qc


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 64, 257])
@pytest.mark.parametrize("fused_post", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_mhc_norm(tokens, fused_post, zero):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(37)
    residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
    if zero:
        residual.zero_()
        x.zero_()
    fn = torch.randn(24, 16384, device="cuda") * 0.01
    scale = torch.tensor([0.2, 0.2, 0.2], device="cuda")
    base = torch.randn(24, device="cuda") * 0.01
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    # Actual GLM checkpoint values, not the DSV4 benchmark's four iterations.
    args = (residual, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20)
    call = qc.dsv4_mhc_pre
    if fused_post:
        post, comb, _ = qc.dsv4_mhc_pre(*args)
        args = (x, residual, post, comb, *args[1:])
        call = qc.dsv4_mhc_fused_post_pre
    original_x, original_residual = x.clone(), residual.clone()
    separate = call(*args)
    expected = torch.empty_like(separate[-1])
    ops.rms_norm(expected, separate[-1], weight, 1e-5)

    def check(result):
        for actual, reference in zip(result[:-1], separate[:-1]):
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result[-1], expected, atol=1e-5, rtol=0.008)
        assert torch.isfinite(result[-1]).all()
        torch.testing.assert_close(x, original_x, atol=0, rtol=0)
        torch.testing.assert_close(residual, original_residual, atol=0, rtol=0)

    check(call(*args, weight, 1e-5))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call(*args, weight, 1e-5)
    for _ in range(3):
        graph.replay()
        check(result)
