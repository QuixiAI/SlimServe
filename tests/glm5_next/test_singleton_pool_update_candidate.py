# SPDX-License-Identifier: Apache-2.0
"""Byte-exact state and changed-slot replay gate for c1 fused update."""

import pytest
import torch

from vllm.model_executor.layers.glm5_next_pool_cache import update_pool_cache


def singleton_update(source, slots, ape, cache):
    update_pool_cache(source, slots, ape, cache, singleton_fused=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bs", [64, 4608])
@torch.no_grad()
def test_all_phases_page_boundaries_and_invalid_slot_in_one_graph(bs):
    torch.manual_seed(9300 + bs)
    original = torch.randn(3, 2, bs, 64, device="cuda", dtype=torch.bfloat16)
    candidate = original.clone()
    expected, actual = original[:, 0], candidate[:, 0]
    source = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([3], device="cuda", dtype=torch.int64)
    ape = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    singleton_update(source, slots, ape, actual)
    update_pool_cache(source, slots, ape, expected)
    assert torch.equal(candidate, original)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        singleton_update(source, slots, ape, actual)
    # Repeat all four phases, ring wrap, logical-page boundary and padded
    # slots. Compare the other slab and unused padding as well as cache data.
    for slot in [-1, *range(16), *range(bs - 8, bs + 8), -1, 2 * bs + 3]:
        slots.fill_(slot)
        source.normal_()
        graph.replay()
        update_pool_cache(source, slots, ape, expected)
        assert torch.equal(candidate, original), f"cache differs at slot {slot}"
