# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from benchmarks.glm5_next_sparse_scratch_candidate import (
    SparseTCWorkspace,
    sparse_tc_tile_capacity,
)


def test_all_decode_shapes_share_fixed_nonoverlapping_views():
    owner = SparseTCWorkspace(32, 2080, 8, "cpu")
    assert owner.tiles == 544
    assert owner._storage.numel() * 4 == 8_947_712
    pointers = None
    for rows in range(1, 33):
        split = 32 if rows < 8 else 128
        parts = (2080 + split - 1) // split
        views = owner.get(rows, 2080, split, 8, "cpu")
        assert [v.shape for v in views] == [
            (rows, parts, 8, 512), (rows, parts, 8), (rows, parts, 8)
        ]
        assert all(v.is_contiguous() and v.dtype == torch.float32 for v in views)
        addresses = [v.data_ptr() for v in views]
        pointers = addresses if pointers is None else pointers
        assert addresses == pointers
        for i in range(2):
            assert addresses[i] + views[i].numel() * 4 <= addresses[i + 1]
        for i, view in enumerate(views):
            view.fill_(i + rows)
        assert all((view == i + rows).all() for i, view in enumerate(views))
    other = SparseTCWorkspace(32, 2080, 8, "cpu")
    assert other._storage.data_ptr() != owner._storage.data_ptr()


@pytest.mark.parametrize("rows,width", [(0, 2080), (33, 2080), (32, 0)])
def test_invalid_capacity(rows, width):
    with pytest.raises(ValueError):
        sparse_tc_tile_capacity(rows, width)


@pytest.mark.parametrize("args", [
    (33, 2080, 128, 8, "cpu"), (1, 2081, 32, 8, "cpu"),
    (1, 2080, 128, 8, "cpu"), (8, 2080, 32, 8, "cpu"),
    (8, 2080, 128, 16, "cpu"), (8, 2080, 128, 8, "cuda:0"),
])
def test_reject_growth_or_policy_change(args):
    owner = SparseTCWorkspace(32, 2080, 8, "cpu")
    with pytest.raises(ValueError):
        owner.get(*args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 7, 8, 16, 32])
@torch.no_grad()
def test_shared_scratch_sequential_layers_changed_graph(rows):
    from benchmarks.glm5_next_sparse_scratch_candidate import (
        shared_sparse_tc_nope as sparse_tc_nope,
    )

    torch.manual_seed(58000 + rows)
    device = torch.device("cuda", torch.cuda.current_device())
    owner = SparseTCWorkspace(32, 2080, 8, device)
    split = 32 if rows < 8 else 128
    states = []
    for layer in range(3):
        q = torch.randn(rows, 8, 512, dtype=torch.bfloat16, device=device) * 0.2
        cache = torch.randn(8, 2, 576, 512, dtype=torch.bfloat16, device=device)[:, 1]
        table = torch.arange(8, dtype=torch.int32, device=device).repeat(rows, 1)
        indices = torch.arange(2080, dtype=torch.int32, device=device).repeat(rows, 1)
        lengths = torch.full((rows,), 2080 - layer, dtype=torch.int32, device=device)
        states.append((q, cache, table, indices, lengths))

    def run(workspace):
        return [sparse_tc_nope(*state, 1 / 16, split=split, workspace=workspace)
                for state in states]

    run(owner)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(owner)
    assert len({output.data_ptr() for output in actual}) == len(states)
    for replay in range(5):
        for layer, (q, cache, table, indices, lengths) in enumerate(states):
            q.normal_(std=0.2)
            cache.normal_()
            table.copy_(torch.randperm(8, device=device).int().repeat(rows, 1))
            lengths.fill_((0, 1, 37, 2048, 2080)[(replay + layer) % 5])
            indices[:, 5::7] = -1
        owner._storage.fill_(float("nan"))
        for output in actual:
            output.fill_(float("nan"))
        graph.replay()
        reference = run(None)
        for output, expected in zip(actual, reference):
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
