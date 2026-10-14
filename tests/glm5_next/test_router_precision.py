# SPDX-License-Identifier: Apache-2.0
"""GLM Ampere logits must not be rounded to BF16 before expert selection."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="GLM router dispatch is specific to SM80",
)


def make_gate(*, deferred=False, experts=288, bias=False):
    with ExitStack() as stack, torch.device("cuda"):
        for module in ("layers.linear", "parameter"):
            for name, value in (("rank", 0), ("world_size", 1)):
                stack.enter_context(
                    patch(
                        f"vllm.model_executor.{module}.get_tensor_model_parallel_{name}",
                        return_value=value,
                    )
                )
        gate = GateLinear(
            4096,
            experts,
            bias=bias,
            params_dtype=torch.bfloat16,
            out_dtype=None if deferred else torch.float32,
        )
        if deferred:
            gate.set_out_dtype(torch.float32)
        return gate


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64, 257, 2048])
@pytest.mark.parametrize("deferred", [False, True])
@torch.no_grad()
def test_fp32_logits_and_graph_replay(tokens, deferred):
    torch.manual_seed(91)
    gate = make_gate(deferred=deferred)
    assert gate.allow_cublas_router_gemm
    gate.weight.normal_(std=0.01)
    x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)

    def check(output):
        reference = F.linear(x.float(), gate.weight.float())
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, reference, atol=2e-5, rtol=1e-5)

    result, bias = gate(x)
    assert bias is None
    check(result)
    rounded = F.linear(x, gate.weight).float()
    assert (result - rounded).abs().max() > 1e-4
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, _ = gate(x)
    for _ in range(3):
        x.normal_()
        graph.replay()
        check(result)
    x.zero_()
    graph.replay()
    check(result)
    assert not result.count_nonzero()


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("experts,bias", [(256, False), (289, False), (288, True)])
def test_dispatch_stays_narrow(deferred, experts, bias):
    gate = make_gate(deferred=deferred, experts=experts, bias=bias)
    assert not gate.allow_cublas_router_gemm
    assert gate.allow_dsv4_ampere_router_gemm == (experts == 256 and not bias)
