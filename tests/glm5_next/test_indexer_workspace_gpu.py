# SPDX-License-Identifier: Apache-2.0
"""Shared scratch must not alias selected outputs across layers/replays."""

import pytest
import torch

from vllm.model_executor.layers.glm5_next_indexer import _pooled_select
from vllm.model_executor.layers.glm5_next_indexer_workspace import (
    Glm5NextIndexerWorkspace,
)
from vllm.model_executor.layers.glm5_next_pool_cache import update_pool_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("compact", [False, True])
@torch.no_grad()
def test_sequential_layers_and_changing_graph_replay(rows, compact):
    torch.manual_seed(7200 + rows)
    length, bs = 3077, 64
    blocks = (length + bs - 1) // bs
    pools = (length + 3) // 4
    table = torch.arange(blocks, device="cuda", dtype=torch.int32).view(1, -1)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
    visible = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    owner = Glm5NextIndexerWorkspace()
    shared = owner.get_decode_logits(
        rows, pools, torch.device("cuda", torch.cuda.current_device())
    )
    states = []
    snapshots = []
    for _ in range(2):
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, 32, device="cuda") * 32**-0.5
        ape = torch.randn(4, 128, device="cuda")
        cache = torch.randn(blocks, bs, 256, device="cuda", dtype=torch.bfloat16)
        if compact:
            packed = cache.view(-1, 256)[:length].clone()
            cache = torch.empty(blocks, bs, 64, device="cuda", dtype=torch.bfloat16)
            slots = torch.arange(length, device="cuda", dtype=torch.int64)
            update_pool_cache(packed, slots, ape, cache)
        reference = torch.empty_like(shared)
        expected = torch.empty(rows, 96, device="cuda", dtype=torch.int32)
        actual = torch.empty_like(expected)
        states.append((q, weights, ape, cache, reference, expected, actual))
        snapshots.append(torch.empty_like(shared))

    def select(state, workspace, output):
        q, weights, ape, cache, *_ = state
        _pooled_select(
            q,
            weights,
            ape,
            cache,
            table,
            requests,
            visible,
            workspace,
            pools,
            bs,
            128**-0.5,
            16,
            output,
            4,
        )

    def shared_step():
        for state, snapshot in zip(states, snapshots):
            select(state, shared, state[-1])
            # Observe each layer before the next layer overwrites scratch.
            snapshot.copy_(shared)

    shared_step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        shared_step()
    private_order_changes = 0
    for iteration in range(8):
        visible.fill_(length - iteration % 4)
        for state in states:
            state[0].normal_()
            state[1].normal_()
        graph.replay()
        valid = torch.arange(pools, device="cuda")[None, :] < visible[:, None] // 4
        for state, snapshot in zip(states, snapshots):
            select(state, state[-3], state[-2])
            torch.testing.assert_close(
                snapshot[valid], state[-3][valid], atol=0, rtol=0
            )
            # Native top-k emits histogram winners using atomicAdd; its
            # contract is the selected set, not a stable index order.
            # Verify that contract with a private-vs-private control too.
            repeated = torch.empty_like(state[-2])
            select(state, state[-3], repeated)
            private_order_changes += int(not torch.equal(repeated, state[-2]))
            assert torch.equal(repeated.sort().values, state[-2].sort().values)
            assert torch.equal(state[-1].sort().values, state[-2].sort().values)
    print(f"private-vs-private order changes: {private_order_changes}/16")
    assert states[0][-1].data_ptr() != states[1][-1].data_ptr()
