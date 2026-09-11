# SPDX-License-Identifier: Apache-2.0
"""Strict candidate gates; not wired into the serving selector."""

import pytest
import torch

from benchmarks import glm5_next_pool_query_split_candidate as split_candidate
from benchmarks import glm5_next_pool_query_tile_candidate as candidate
from vllm.model_executor.layers.glm5_next_pool_cache import cached_pool_logits


@pytest.mark.parametrize("rows,tile", [(0, 2), (1, 2), (5, 4), (256, 4)])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("programs", [1, 7, 128])
@pytest.mark.parametrize("bs", [576, 4608])
def test_launch_geometry_and_strided_pages(
    monkeypatch, rows, tile, split, programs, bs,
):
    q = torch.zeros(rows, 32, 128, dtype=torch.bfloat16)
    weights = torch.zeros(rows, 32)
    cache = torch.empty(2, 11, bs, 64, dtype=torch.bfloat16)[:, 1]
    table = torch.zeros(1, 2, dtype=torch.int32)
    req = torch.zeros(rows, dtype=torch.int32)
    visible = torch.zeros(rows, dtype=torch.int32)
    out = torch.empty(rows, 513)
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return launch

    module = split_candidate if split else candidate
    monkeypatch.setattr(module, "_query_tile", Kernel())
    if split:
        monkeypatch.setattr(module, "_mixed_rows", Kernel())
    function = (split_candidate.split_query_pool_logits if split
                else candidate.query_tiled_pool_logits)
    function(q, weights, cache, table, req, visible, out, query_tile=tile,
             programs=programs)
    if rows == 0:
        assert not launches
    else:
        grid, args, kwargs = launches[0]
        assert grid == ((rows + tile - 1) // tile, min(programs, 33))
        assert args[10] == cache.stride(0) == 11 * bs * 64
        assert args[-1] == tile
        assert kwargs == ({"num_warps": 4, "FALLBACK": False} if split
                          else {"num_warps": 4})
        assert len(launches) == (2 if split else 1)
        if split:
            mixed_grid, mixed_args, mixed_kwargs = launches[1]
            assert mixed_grid == (rows, min(programs, 33))
            assert all(a is b or a == b for a, b in zip(args, mixed_args))
            assert mixed_kwargs == {"num_warps": 4}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tile", [2, 4])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("rows,context,bs", [
    (5, 2052, 64), (64, 131076, 576), (256, 8196, 576),
    (256, 131076, 576), (2048, 131076, 576),
    (5, 9219, 4608), (256, 8196, 4608), (2048, 131076, 4608),
])
@torch.no_grad()
def test_changed_graph_homogeneous_and_cross_request_groups(tile, rows, context, bs,
                                                          split):
    torch.manual_seed(81200 + rows)
    pages = (context + bs - 1) // bs
    cache = torch.randn(3 * pages, 11, bs, 64, device="cuda",
                        dtype=torch.bfloat16)[:, 1]
    table = torch.randperm(3 * pages, device="cuda").int().view(3, pages)
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device="cuda") * 32**-0.5
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    visible = torch.full((rows,), context, device="cuda", dtype=torch.int32)
    columns = context // 4 + 1
    backing = torch.full((rows * columns + 64,), -77., device="cuda")
    actual = backing[32:-32].view(rows, columns)
    expected = torch.empty_like(actual)

    def run():
        function = (split_candidate.split_query_pool_logits if split
                    else candidate.query_tiled_pool_logits)
        function(q, weights, cache, table, req, visible, actual, query_tile=tile)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    choices = torch.tensor([0, 3, 4, 65, 1001, context - 3, context],
                           device="cuda", dtype=torch.int32)
    row_ids = torch.arange(rows, device="cuda")
    for replay in range(7):
        q.normal_()
        weights.normal_()
        table.copy_(table.roll(1, dims=1))
        if replay == 0:
            req.zero_()  # Wide dot across adjacent queries of one request.
        elif replay == 1:
            req.copy_((row_ids // (tile * 3)) % 3)
        else:
            req.copy_((row_ids + replay) % 3)  # Cross-request fallback.
        visible.copy_(choices[(row_ids + replay) % len(choices)])
        if replay == 4:
            visible.zero_()
        if replay == 5:
            req.zero_()
            visible.fill_(context)
        if replay == 6:
            # In the same launch, exercise homogeneous groups, mixed groups,
            # and (for rows=5) a partially populated final group.
            req.copy_(torch.where(row_ids < rows // 2, 0, row_ids % 3))
        actual.fill_(float("-inf"))
        expected.fill_(float("-inf"))
        graph.replay()
        cached_pool_logits(q, weights, cache, table, req, visible, expected)
        valid = torch.arange(columns, device="cuda")[None, :] < visible[:, None] // 4
        # Same strict gate as the existing pool-score candidate. Do not relax
        # this if the wider dot/reduction changes rounding near cancellation.
        torch.testing.assert_close(
            actual.masked_fill(~valid, 0), expected.masked_fill(~valid, 0),
            atol=1e-6, rtol=1e-6,
        )
        long = visible // 4 >= 512
        if long.any():
            a = actual.masked_fill(~valid, float("-inf"))[long]
            b = expected.masked_fill(~valid, float("-inf"))[long]
            assert torch.equal(a.topk(512).indices.sort().values,
                               b.topk(512).indices.sort().values)
        assert (backing[:32] == -77).all() and (backing[-32:] == -77).all()
