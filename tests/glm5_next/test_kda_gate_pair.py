# SPDX-License-Identifier: Apache-2.0
"""Real TP4/TP8 gate shapes, ragged batches, stride and graph replay."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.kda_gate_projection import kda_gate_pair


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [0, 1, 2, 3, 4, 8, 16, 24, 32, 40, 48, 56, 64, 65, 257])
@pytest.mark.parametrize("n", [1024, 2048])
@torch.no_grad()
def test_gate_pair(tokens, n):
    torch.manual_seed(21)
    fa_backing = torch.randn(
        tokens, 3 * n + n // 128 + 256, device="cuda", dtype=torch.bfloat16
    )
    ga_backing = torch.randn(tokens, 144, device="cuda", dtype=torch.bfloat16)
    fa, ga = fa_backing[:, -256:-128], ga_backing[:, 8:136]
    wf = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    wg = torch.randn_like(wf) * 0.1
    wf_before, wg_before = wf.clone(), wg.clone()

    def check(result):
        expected = (fa @ wf.T, ga @ wg.T)
        for actual, ref in zip(result, expected):
            torch.testing.assert_close(actual, ref, atol=0.008, rtol=0.008)
            assert actual.is_contiguous()
            assert torch.isfinite(actual).all()
        if tokens:
            assert (
                result[0].untyped_storage().data_ptr()
                != result[1].untyped_storage().data_ptr()
            )
        torch.testing.assert_close(wf, wf_before, atol=0, rtol=0)
        torch.testing.assert_close(wg, wg_before, atol=0, rtol=0)

    check(kda_gate_pair(fa, ga, wf, wg))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = torch.ops.vllm.kda_gate_pair(fa, ga, wf, wg)
    for _ in range(3):
        # Different inputs on the same captured allocations each replay.
        fa.mul_(0.5)
        ga.mul_(-0.25)
        graph.replay()
        check(result)
    fa.zero_()
    ga.zero_()
    graph.replay()
    check(result)
    assert not result[0].count_nonzero() and not result[1].count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gate_pair_op_contract():
    fa = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    ga = torch.randn_like(fa)
    # Also exercise masked output columns, although serving dimensions align.
    wf = torch.randn(257, 128, device="cuda", dtype=torch.bfloat16)
    wg = torch.randn_like(wf)
    result = kda_gate_pair(fa, ga, wf, wg)
    for actual, expected in zip(result, (fa @ wf.T, ga @ wg.T)):
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
    torch.library.opcheck(torch.ops.vllm.kda_gate_pair, (fa, ga, wf, wg))
