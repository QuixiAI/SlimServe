# SPDX-License-Identifier: Apache-2.0
"""Opaque allocation/stream-lifetime gate, separate from serving integration."""

import pytest
import torch

from tests.glm5_next.test_mhc_runtime_stream import context
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers import glm5_next_mhc_project  # noqa: F401
from vllm.model_executor.layers.glm5_next_mhc_ops import glm5_mhc_fused_post_pre


def inputs(m, width, device):
    def tensor(*shape, dtype=torch.float32):
        return torch.randn(*shape, device=device, dtype=dtype)

    x = tensor(m, 4096, dtype=torch.bfloat16)
    residual = tensor(m, 4, 4096, dtype=torch.bfloat16)
    post = tensor(m, 4, 1) * 0.1
    comb = torch.eye(4, device=device).expand(m, -1, -1).contiguous()
    fn = tensor(24, 16384) * 0.01
    scale = torch.full((3,), 0.2, device=device)
    base = tensor(24) * 0.01
    norm = tensor(4096, dtype=torch.bfloat16)
    weight = tensor(width, 4096, dtype=torch.bfloat16) * 0.01
    return x, residual, post, comb, fn, scale, base, norm, weight


@pytest.mark.parametrize("m", [0, 1, 32, 65])
@pytest.mark.parametrize("fp32,width", [(False, 3336), (True, 288)])
@pytest.mark.parametrize("runtime", [False, True])
def test_opaque_meta_schema(m, fp32, width, runtime):
    args = inputs(m, width, "meta")
    op = (
        torch.ops.vllm.glm5_mhc_project_runtime
        if runtime
        else torch.ops.vllm.glm5_mhc_project
    )
    actual = op(*args, "model.mhc_stream" if runtime else 1, fp32)
    assert [t.shape for t in actual] == [
        (m, 4, 4096),
        (m, 4, 1),
        (m, 4, 4),
        (m, 4096),
        (m, width),
    ]
    assert actual[-1].dtype == (torch.float32 if fp32 else torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 65])
@pytest.mark.parametrize("fp32,width", [(False, 3336), (True, 288)])
@torch.no_grad()
def test_chained_opaque_allocations_and_changing_graph_replay(m, fp32, width):
    torch.manual_seed(9010 + m)
    args = inputs(m, width, "cuda")
    side = torch.cuda.Stream()

    def chain(candidate):
        x, residual, post, comb, *params = args
        for _ in range(3):
            if candidate:
                with override_forward_context(context(side)):
                    result = torch.ops.vllm.glm5_mhc_project_runtime(
                        x, residual, post, comb, *params, "model.mhc_stream", fp32
                    )
            else:
                fn, scale, base, norm, weight = params
                result = glm5_mhc_fused_post_pre(
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
                    norm,
                    1e-5,
                )
                if fp32:
                    projected = torch.mm(result[-1], weight.T, out_dtype=torch.float32)
                else:
                    projected = torch.mm(result[-1], weight.T)
                result = *result, projected
            residual, post, comb, layer, projected = result
            # Consume both branches, release earlier intermediates, and
            # exercise graph-pool reuse like sequential model layers.
            x = layer + projected.sum(-1, keepdim=True).to(layer.dtype) * 0.001
        return result

    def compare(actual, expected):
        for index, (a, b) in enumerate(zip(actual, expected)):
            tolerance = 1e-6 if index in (1, 2) else 0
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)

    compare(chain(True), chain(False))
    for _ in range(3):
        chain(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = chain(True)
    for _ in range(8):
        args[0].normal_()
        args[1].normal_()
        graph.replay()
        compare(actual, chain(False))
