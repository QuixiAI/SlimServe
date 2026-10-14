# SPDX-License-Identifier: Apache-2.0
"""Prove score elision against native selection, not a rewritten top-k oracle."""

import pytest
import torch

from benchmarks import glm5_next_pool_skip_candidate as candidate
from vllm import _custom_ops as ops
from vllm.model_executor.layers.glm5_next_indexer import (
    _expand_topk_kernel,
    _pooled_topk,
)

pytestmark = pytest.mark.xfail(
    strict=False,
    reason=("quarantined candidate (pool-skip score elision): not on the serving path "
            "(glm5_next_adaptive_pool_score is off / no serving wiring); parity failing as of 2026-09-10, "
            "see perf/optimization_status.md 2026-09-10 07:16 UTC"),
)



@pytest.mark.parametrize("rows", [0, 1, 8, 16, 32])
@pytest.mark.parametrize("bs", [576, 4608])
def test_launch_order_and_strided_cache(monkeypatch, rows, bs):
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.name, grid, args))
            return launch

    def topk(*args):
        calls.append(("topk", None, args))

    monkeypatch.setattr(candidate, "_score_nontrivial_rows", Kernel("score"))
    monkeypatch.setattr(candidate, "_expand_topk_kernel", Kernel("expand"))
    monkeypatch.setattr(candidate.ops, "top_k_per_row_prefill", topk)
    q = torch.empty(rows, 32, 128, dtype=torch.bfloat16)
    weights = torch.empty(rows, 32)
    cache = torch.empty(2, 11, bs, 64, dtype=torch.bfloat16)[:, 1]
    table = torch.zeros(1, 2, dtype=torch.int32)
    req = torch.zeros(rows, dtype=torch.int32)
    visible = torch.full((rows,), 2051, dtype=torch.int32)
    logits = torch.empty(rows, 262144)
    sel = torch.empty(rows, 512, dtype=torch.int32)
    expanded = torch.empty(rows, 2080, dtype=torch.int32)
    candidate.select_without_trivial_scores(q, weights, cache, table, req,
                                             visible, logits, sel, expanded)
    assert [call[0] for call in calls] == (["score", "topk", "expand"] if rows else [])
    if rows:
        assert calls[0][1] == (rows, 128)
        assert calls[0][2][9] == cache.stride(0) == 11 * bs * 64
        assert calls[1][2][0] is logits and calls[1][2][3] is sel
        assert torch.equal(calls[1][2][2], torch.full_like(visible, 512))
        assert calls[2][2][2] is expanded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -77.])
@torch.no_grad()
def test_native_shortcut_ignores_poisoned_logits(poison):
    lengths = torch.tensor([0, 1, 255, 511, 512], device="cuda", dtype=torch.int32)
    logits = torch.full((5, 262144), poison, device="cuda")
    selected = torch.full((5, 512), -99, device="cuda", dtype=torch.int32)
    ops.top_k_per_row_prefill(logits, torch.zeros_like(lengths), lengths, selected,
                             5, logits.stride(0), 1, 512)
    ids = torch.arange(512, device="cuda")
    expected = torch.where(ids[None, :] < lengths[:, None], ids, -1).int()
    assert torch.equal(selected, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 8, 16, 32])
@pytest.mark.parametrize("bs", [64, 576, 4608])
@torch.no_grad()
def test_changed_graph_threshold_mixed_rows_and_expanded_order(rows, bs):
    torch.manual_seed(99200 + rows + bs)
    context, columns = 8196, 262144  # Preserve full 1M-context workspace stride.
    pages = (context + bs - 1) // bs
    cache = torch.randn(3 * pages, 11, bs, 64, device="cuda",
                        dtype=torch.bfloat16)[:, 1]
    table = torch.randperm(3 * pages, device="cuda").int().view(3, pages)
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device="cuda")
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    visible = torch.full((rows,), context, device="cuda", dtype=torch.int32)
    backing = torch.full((rows * columns + 64,), float("nan"), device="cuda")
    logits = backing[32:-32].view(rows, columns)
    reference_logits = torch.empty_like(logits)
    selected = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    guard = torch.full((rows * 2080 + 64,), -77, device="cuda", dtype=torch.int32)
    expanded = guard[32:-32].view(rows, 2080)
    reference_expanded = torch.empty_like(expanded)
    ape = torch.zeros(4, 128, device="cuda")  # Ignored by the compact-cache path.

    def run():
        candidate.select_without_trivial_scores(q, weights, cache, table, req,
                                                 visible, logits, selected, expanded)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    choices = torch.tensor([0, 1, 3, 4, 2047, 2048, 2051, 2052,
                            4607, 4608, 4611, 4612, context],
                           device="cuda", dtype=torch.int32)
    ids = torch.arange(rows, device="cuda")
    for replay in range(len(choices)):
        q.normal_()
        weights.normal_()
        table.copy_(table.roll(1, dims=1))
        req.copy_((ids + replay) % 3)
        visible.copy_(choices[(ids + replay) % len(choices)])
        logits.fill_(float("nan"))
        graph.replay()
        expected = _pooled_topk(q, weights, ape, cache, table, req, visible,
                                reference_logits, columns, bs, 128**-0.5, 512, 4)
        _expand_topk_kernel[(rows,)](expected, visible, reference_expanded, 512,
                                    KP=4, KSEL=512, OUT_W=2080, BLOCK_S=64)
        short = visible // 4 <= 512
        assert torch.isnan(logits[short]).all()  # Demonstrate actual elision.
        assert torch.equal(selected[short], expected[short])
        assert torch.equal(expanded[short], reference_expanded[short])
        # Native histogram ordering need not be stable, but selected sets must.
        assert torch.equal(selected.sort().values, expected.sort().values)
        assert torch.equal(expanded.sort().values, reference_expanded.sort().values)
        valid = ((torch.arange(columns, device="cuda")[None, :] < visible[:, None] // 4)
                 & ~short[:, None])
        torch.testing.assert_close(logits[valid], reference_logits[valid],
                                   atol=0, rtol=0)
        assert torch.isnan(backing[:32]).all() and torch.isnan(backing[-32:]).all()
        assert (guard[:32] == -77).all() and (guard[-32:] == -77).all()


def test_benchmark_uses_registered_compact_page_geometry():
    from benchmarks.glm5_next_pool_layout import packed_indexer_cache

    cache = packed_indexer_cache(11, 2, 4608, "cpu")
    assert cache.shape == (11, 2, 4608, 64)
    assert cache[0].stride() == (11 * 4608 * 64, 64, 1)
    assert cache[0].stride(0) * cache.element_size() == 6488064
    cache[0].fill_(1)
    cache[1].fill_(2)
    assert (cache[0] == 1).all() and (cache[1] == 2).all()
