# SPDX-License-Identifier: Apache-2.0
"""No-copy packed KDA: exact output/state and changed-input graph replay."""

import pytest
import torch

from benchmarks.glm5_next_kda_direct_output import direct_output
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    fused_recurrent_kda_packed_decode,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@torch.no_grad()
def test_direct_output_exact_state_and_graph(rows, heads, lower_bound):
    torch.manual_seed(9400 + rows + heads)
    d = 128
    # Padded producer rows and physical state slots exercise serving strides.
    packed = torch.randn(
        rows, 3 * heads * d + 128, device="cuda", dtype=torch.bfloat16
    )[:, : 3 * heads * d]
    gate = torch.randn(1, rows, heads + 1, d, device="cuda", dtype=torch.bfloat16)[
        :, :, :heads
    ]
    beta = torch.randn(1, rows, heads + 1, device="cuda", dtype=torch.bfloat16)[
        :, :, :heads
    ]
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1
    bias = torch.randn(heads * d, device="cuda", dtype=torch.float32) * 0.1
    initial = torch.randn(
        rows + 3, heads * d * d + 128, device="cuda", dtype=torch.float32
    )
    reference_slab, direct_slab = initial.clone(), initial.clone()
    state_ref = reference_slab[:, : heads * d * d].view(rows + 3, heads, d, d)
    state_direct = direct_slab[:, : heads * d * d].view(rows + 3, heads, d, d)
    indices = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
    output_slab = torch.full(
        (1, rows + 2, heads, d), 7.0, device="cuda", dtype=torch.bfloat16
    )
    out = output_slab[:, 1 : rows + 1]

    def launch():
        return direct_output(
            packed, gate, beta, a_log, bias, lower_bound, state_direct, indices, out
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = launch()
    assert result.data_ptr() == out.data_ptr()
    for replay in range(8):
        initial.normal_()
        reference_slab.copy_(initial)
        direct_slab.copy_(initial)
        packed.normal_()
        gate.normal_()
        beta.normal_()
        indices.copy_(torch.randperm(rows, device="cuda") + 1)
        if replay % 3 == 1:
            indices[0] = 0
        elif replay % 3 == 2:
            indices[0] = -1
        expected, _ = fused_recurrent_kda_packed_decode(
            packed, gate, beta, a_log, bias, lower_bound, state_ref, indices
        )
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(direct_slab, reference_slab, rtol=0, atol=0)
        assert (output_slab[:, 0] == 7).all()
        assert (output_slab[:, -1] == 7).all()
