# SPDX-License-Identifier: Apache-2.0
"""Eager/compiled custom-op and changed-input CUDA graph qualification."""

import pytest
import torch

from slimserve.canonical_moe import canonicalize
from tests.kernels.test_glm53_stable_route import assert_bits, check_result
from vllm.model_executor.layers.fused_moe.router import glm_route_align as route
from vllm.quixicore.ops import quixicore_ops as qc

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("tokens", (1, 2, 8, 13, 16))
@pytest.mark.parametrize("stable", (False, True))
@pytest.mark.parametrize("compiled", (False, True))
def test_custom_op_graph(monkeypatch, tokens, stable, compiled):
    monkeypatch.setattr(route, "_STABLE_ALIGNMENT_DIAGNOSTIC", stable)
    capacity, blocks = route.alignment_geometry(tokens, 8, 288, 8)
    generator = torch.Generator().manual_seed(53500 + tokens)
    random_logits = torch.randn(tokens, 288, generator=generator)
    random_bias = torch.randn(288, generator=generator)
    cases = [
        (random_logits, random_bias),
        (torch.zeros_like(random_logits), torch.zeros_like(random_bias)),
        (torch.full_like(random_logits, float("nan")), random_bias),
        (-random_logits, -random_bias),
    ]
    logits, bias = random_logits.cuda(), random_bias.cuda()

    def call(logits, bias):
        return torch.ops.vllm.glm_route_align(
            logits, bias, 8, 0, True, 2.5, 8, capacity, blocks
        )

    fn = torch.compile(call, fullgraph=True) if compiled else call

    def check(result):
        control = qc.glm_route_align(logits, bias, 8, 0, True, 2.5, 8, capacity, blocks)
        if not stable:
            canonicalize(*result[2:], tokens=tokens, block_size=8)
        check_result(result, control, tokens, 8)

    for x, b in cases:
        logits.copy_(x)
        bias.copy_(b)
        check(fn(logits, bias))
        assert_bits(logits, x)
        assert_bits(bias, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(logits, bias)
    for x, b in cases + cases[::-1]:
        logits.copy_(x)
        bias.copy_(b)
        for tensor in output:
            tensor.fill_(-123)
        graph.replay()
        check(output)
        assert_bits(logits, x)
        assert_bits(bias, b)
