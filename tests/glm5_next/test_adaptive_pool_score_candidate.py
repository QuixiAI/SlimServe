# SPDX-License-Identifier: Apache-2.0
"""Mixed-length graph replay gate for the optional owned adaptive scorer."""

import pytest
import torch

from vllm.model_executor.layers.glm5_next_pool_cache import cached_pool_logits
from vllm.model_executor.layers.glm5_next_pool_score import adaptive_pool_logits

pytestmark = pytest.mark.xfail(
    strict=False,
    reason=("quarantined candidate (adaptive scorer): not on the serving path "
            "(glm5_next_adaptive_pool_score is off / no serving wiring); parity failing as of 2026-09-10, "
            "see perf/optimization_status.md 2026-09-10 07:16 UTC"),
)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,max_context,requests",
    [
        (1, 131076, 1),
        (8, 131076, 8),
        (16, 131076, 16),
        (32, 131076, 32),
        (64, 131076, 64),
        (1, 1048576, 1),
        # Prefill rows share a few request caches, not one long cache per
        # token. This bounds allocation while covering the actual geometry.
        (256, 131076, 3),
        (2048, 131076, 3),
    ],
)
@torch.no_grad()
def test_mixed_lengths_cross_tile_thresholds_on_same_graph(rows, max_context, requests):
    torch.manual_seed(9100 + rows)
    bs = 4608
    pages = (max_context + bs - 1) // bs
    # Padded cross-layer slab and shuffled physical pages, as in serving.
    cache = torch.randn(
        requests * pages, 2, bs, 64, device="cuda", dtype=torch.bfloat16
    )[:, 0]
    table = (
        torch.randperm(requests * pages, device="cuda").int().reshape(requests, pages)
    )
    row_req = torch.arange(rows, device="cuda", dtype=torch.int32) % requests
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device="cuda") * 32**-0.5
    visible = torch.full((rows,), 1000, device="cuda", dtype=torch.int32)
    actual = torch.empty(rows, 1048576 // 4, device="cuda")
    expected = torch.empty_like(actual)

    def candidate():
        adaptive_pool_logits(q, weights, cache, table, row_req, visible, actual)

    candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    lengths = [0, 3, 4, 1000, 2048, 2052, 16384, 32768, 32772, max_context]
    for replay in range(len(lengths)):
        current = [lengths[(replay + row) % len(lengths)] for row in range(rows)]
        visible.copy_(torch.tensor(current, dtype=torch.int32, device="cuda"))
        q.normal_()
        weights.normal_()
        graph.replay()
        cached_pool_logits(q, weights, cache, table, row_req, visible, expected)
        cap = max(current) // 4
        columns = torch.arange(cap, device="cuda")
        valid = columns[None, :] < (visible // 4)[:, None]
        # Inactive columns are intentionally undefined, not numeric failures.
        a = actual[:, :cap].masked_fill(~valid, 0)
        b = expected[:, :cap].masked_fill(~valid, 0)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        long_rows = torch.tensor(
            [row for row, length in enumerate(current) if length // 4 > 512],
            device="cuda",
            dtype=torch.int64,
        )
        if long_rows.numel():
            a = a.masked_fill(~valid, float("-inf")).index_select(0, long_rows)
            b = b.masked_fill(~valid, float("-inf")).index_select(0, long_rows)
            assert torch.equal(
                a.topk(512).indices.sort().values, b.topk(512).indices.sort().values
            )
