# SPDX-License-Identifier: Apache-2.0
"""KDA's opaque attention body must use a fused, stride-safe output norm."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _apply_kda_output_norm,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated


@pytest.fixture
def default_vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 8, 16, 37, 257])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("strided_gate", [False, True])
@torch.no_grad()
def test_kda_cuda_output_norm(tokens, heads, strided_gate, default_vllm_config):
    torch.manual_seed(42)
    dim = 128
    norm = FusedRMSNormGated(
        dim, activation="sigmoid", device="cuda", dtype=torch.bfloat16
    )
    norm.weight.copy_(torch.randn_like(norm.weight))
    x = torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    backing = torch.randn(
        1,
        tokens,
        heads * (2 if strided_gate else 1),
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = backing[:, :, :heads]
    gate_before = gate.clone()
    expected = norm.forward_native(x, gate)
    original = x.clone()
    actual = norm.forward_cuda(x, gate)
    # The serving buffer is contiguous: the CUDA path must not allocate a
    # replacement output or require a copy back into kda_attention's buffer.
    assert actual.data_ptr() == x.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
    torch.testing.assert_close(gate, gate_before, rtol=0, atol=0)
    # Capture/replay on the same input allocation, restoring the input each
    # time because this implementation intentionally mutates it.
    x.copy_(original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _apply_kda_output_norm(norm, x, gate)
    for _ in range(3):
        x.copy_(original)
        graph.replay()
        torch.testing.assert_close(x, expected, atol=0.008, rtol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strided_output", [False, True])
@pytest.mark.parametrize("zero_input", [False, True])
@torch.no_grad()
def test_kda_output_norm_edge_values(strided_output, zero_input, default_vllm_config):
    torch.manual_seed(17)
    norm = FusedRMSNormGated(
        128, activation="sigmoid", device="cuda", dtype=torch.bfloat16
    )
    norm.weight.copy_(torch.randn_like(norm.weight))
    backing = torch.full(
        (1, 3, 8, 256 if strided_output else 128),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = backing[..., ::2] if strided_output else backing
    output.copy_(torch.zeros_like(output) if zero_input else torch.randn_like(output))
    gate_values = torch.tensor(
        [-100, -10, 0, 10, 100], device="cuda", dtype=torch.bfloat16
    )
    gate = gate_values[torch.arange(output.numel(), device="cuda") % 5].view_as(output)
    expected = norm.forward_native(output, gate)
    _apply_kda_output_norm(norm, output, gate)
    assert output.isfinite().all()
    torch.testing.assert_close(output, expected, atol=0.008, rtol=0.008)
    if strided_output:
        assert backing[..., 1::2].isnan().all(), "output overwrote stride padding"
