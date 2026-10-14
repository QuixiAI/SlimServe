# SPDX-License-Identifier: Apache-2.0
"""Both low-rank KDA inputs must be replicated, while Q/K/V/beta shard."""

from unittest.mock import patch

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KimiGDNMergedColumnParallelLinear,
)


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("rank_kind", ["first", "last"])
@pytest.mark.parametrize("loader_version", [1, 2])
@torch.no_grad()
def test_replicated_gate_rows(tp, rank_kind, loader_version):
    rank = 0 if rank_kind == "first" else tp - 1
    sizes = [8192, 8192, 8192, 64, 128, 128]
    # Small input width suffices to check the exact production row layout.
    with (
        set_current_vllm_config(VllmConfig()),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=rank,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=tp,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=rank,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=tp,
        ),
    ):
        layer = _KimiGDNMergedColumnParallelLinear(
            16,
            sizes,
            replicated_shard_id=(4, 5),
            tp_size=tp,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        original_param_rank = getattr(layer.weight, "tp_rank", None)
        expected = []
        for shard_id, size in enumerate(sizes):
            weight = (
                torch.arange(size * 16).reshape(size, 16) % 127 + shard_id * 128
            ).to(torch.bfloat16)
            loader = (
                layer.weight_loader if loader_version == 1 else layer.weight_loader_v2
            )
            loader(layer.weight, weight, shard_id)
            expected.append(weight if shard_id in (4, 5) else weight.chunk(tp)[rank])
            assert layer.tp_rank == rank
            assert getattr(layer.weight, "tp_rank", None) == original_param_rank
        torch.testing.assert_close(layer.weight, torch.cat(expected), atol=0, rtol=0)
        assert sizes == [8192, 8192, 8192, 64, 128, 128], (
            "constructor mutated input sizes"
        )


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("tokens", [1, 8, 16, 257])
@torch.no_grad()
def test_merged_gate_projection_parity(tp, tokens):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    width = 3 * 8192 // tp + 64 // tp + 128
    x = torch.randn(tokens, 4096, dtype=torch.bfloat16, device="cuda")
    old_weight = torch.randn(width, 4096, dtype=torch.bfloat16, device="cuda") * 0.01
    gate_weight = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda") * 0.01
    merged_weight = torch.cat([old_weight, gate_weight])
    expected = torch.cat([x @ old_weight.T, x @ gate_weight.T], dim=-1)
    actual = x @ merged_weight.T
    # A GEMM's changed N can select another accumulation schedule; compare
    # to BF16 precision, not sampled completion hashes.
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = x @ merged_weight.T
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(replayed, actual, atol=0, rtol=0)
