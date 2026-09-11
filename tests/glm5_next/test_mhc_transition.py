# SPDX-License-Identifier: Apache-2.0
"""New mHC against the owned native implementation, including state parity."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.glm5_next_mhc_triton import mhc_transition
from vllm.quixicore.ops import quixicore_ops as qc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 16, 24, 32, 40, 48, 56, 64])
@pytest.mark.parametrize("fused_post", [False, True])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("seed", [0, 21, 327])
@torch.no_grad()
def test_mhc_transition(tokens, fused_post, norm, seed):
    torch.manual_seed(seed)
    residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
    x = (
        torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
        if fused_post
        else None
    )
    fn = torch.randn(24, 16384, device="cuda") * 0.01
    scale = torch.tensor([0.2, 0.2, 0.2], device="cuda")
    base = torch.randn(24, device="cuda") * 0.01
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16) if norm else None
    common = (fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20)
    post, comb, _ = qc.dsv4_mhc_pre(residual, *common)

    def reference():
        if fused_post:
            result = qc.dsv4_mhc_fused_post_pre(x, residual, post, comb, *common)
        else:
            result = (residual, *qc.dsv4_mhc_pre(residual, *common))
        if weight is not None:
            normalized = torch.empty_like(result[-1])
            ops.rms_norm(normalized, result[-1], weight, 1e-5)
            result = (*result[:-1], normalized)
        return result

    def check(result):
        expected = reference()
        torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0)
        for actual, ref in zip(result[1:3], expected[1:3]):
            torch.testing.assert_close(actual, ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result[-1], expected[-1], atol=0.008, rtol=0.008)
        assert all(t.isfinite().all() for t in result)

    args = (x, residual, post, comb, fn, scale, base, 1e-5, 1e-6, 2.0, 20, weight, 1e-5)
    check(mhc_transition(*args))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = mhc_transition(*args)
    for _ in range(2):
        residual.mul_(0.75)
        if x is not None:
            x.mul_(-0.5)
        graph.replay()
        check(result)
    residual.zero_()
    if x is not None:
        x.zero_()
    graph.replay()
    check(result)


@pytest.mark.parametrize("tokens", [1, 64, 65, 257])
def test_mhc_transition_fake_schema(tokens):
    from torch._subclasses.fake_tensor import FakeTensorMode

    import vllm.model_executor.layers.glm5_next_mhc_ops  # noqa: F401

    with FakeTensorMode():
        residual = torch.empty(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
        fn = torch.empty(24, 16384, device="cuda")
        scale = torch.empty(3, device="cuda")
        base = torch.empty(24, device="cuda")
        weight = torch.empty(4096, device="cuda", dtype=torch.bfloat16)
        post, comb, x = torch.ops.vllm.glm5_mhc_pre(
            residual,
            fn,
            scale,
            base,
            1e-5,
            1e-6,
            2.0,
            20,
            weight,
            1e-5,
        )
        result = torch.ops.vllm.glm5_mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            1e-5,
            1e-6,
            2.0,
            20,
            weight,
            1e-5,
        )
        assert [t.shape for t in result] == [
            (tokens, 4, 4096),
            (tokens, 4, 1),
            (tokens, 4, 4),
            (tokens, 4096),
        ]
