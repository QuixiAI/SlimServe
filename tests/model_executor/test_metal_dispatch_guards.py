# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.activation import _metal_swiglu
from vllm.model_executor.layers.quantization.gguf.fused_moe import (
    _metal_q2k_sum_rows_supported,
)
from vllm.models.deepseek_v4.metal_indexer import (
    _score_and_select,
    _score_and_select_streaming,
)


def test_oai_swiglu_nondefault_scalars_do_not_dispatch_native() -> None:
    x = torch.randn(3, 16, dtype=torch.float16)
    out = torch.empty(3, 8, dtype=torch.float16)

    # The guard must fire before importing or invoking the MPS extension, so
    # this is intentionally a CPU-only dispatch-policy test.
    assert not _metal_swiglu(out, x, 7.0, oai_form=True, alpha=1.702, beta=1.0)
    assert not _metal_swiglu(out, x, 7.0, oai_form=True, alpha=1.0, beta=0.25)


def test_q2k_sum_folded_row_alignment_guard() -> None:
    assert _metal_q2k_sum_rows_supported(64, torch.float16)
    assert not _metal_q2k_sum_rows_supported(48, torch.float16)
    assert _metal_q2k_sum_rows_supported(24, torch.bfloat16)
    assert not _metal_q2k_sum_rows_supported(20, torch.bfloat16)
    assert not _metal_q2k_sum_rows_supported(64, torch.float32)


def test_streaming_indexer_fallback_matches_full_selection() -> None:
    torch.manual_seed(5)
    rows, heads, dim, keys, topk = 3, 2, 4, 2051, 17
    q = torch.randn(rows, heads, dim, dtype=torch.float16)
    weights = torch.rand(rows, heads, dtype=torch.float32)
    k_vals = torch.randn(keys, dim, dtype=torch.float32)
    k_scale = torch.rand(keys, dtype=torch.float32)
    lo = torch.tensor([0, 233, 1027])
    hi = torch.tensor([keys, 1900, keys])

    full = _score_and_select(q, weights, k_vals, k_scale, lo, hi, topk)
    streamed = _score_and_select_streaming(q, weights, k_vals, k_scale, lo, hi, topk)
    assert torch.equal(streamed, full)


def test_streaming_indexer_fallback_tie_order() -> None:
    """Ties are reachable in production (relu zeroes any candidate whose
    head dots are all <= 0; stale e4m3 slots decode to 0), and torch.topk's
    tie order is unspecified — both fallbacks must break ties the way the
    native kernel documents: (logit desc, index asc)."""
    rows, heads, dim, keys, topk = 3, 2, 4, 3000, 17
    # All-zero queries make every candidate logit exactly 0.0: a full-width
    # tie. The deterministic order selects the first topk valid columns.
    q = torch.zeros(rows, heads, dim, dtype=torch.float16)
    weights = torch.ones(rows, heads, dtype=torch.float32)
    k_vals = torch.randn(keys, dim, dtype=torch.float32)
    k_scale = torch.ones(keys, dtype=torch.float32)
    lo = torch.tensor([0, 233, 1027])
    hi = torch.tensor([keys, 1900, keys])

    full = _score_and_select(q, weights, k_vals, k_scale, lo, hi, topk)
    streamed = _score_and_select_streaming(q, weights, k_vals, k_scale, lo, hi, topk)
    expected = torch.stack(
        [torch.arange(int(start), int(start) + topk) for start in lo]
    )
    assert torch.equal(full, expected)
    assert torch.equal(streamed, expected)

    # Partial ties: a handful of strict winners above a tied plateau.
    q2 = torch.randn(rows, heads, dim, dtype=torch.float16)
    k_flat = torch.zeros(keys, dim, dtype=torch.float32)
    k_flat[100] = 1.0
    k_flat[2500] = 1.0
    full2 = _score_and_select(q2.abs(), weights, k_flat, k_scale, lo, hi, topk)
    streamed2 = _score_and_select_streaming(
        q2.abs(), weights, k_flat, k_scale, lo, hi, topk
    )
    assert torch.equal(streamed2, full2)


def test_streaming_indexer_fallback_per_row_layout() -> None:
    """The decode fallback supplies per-row K ([rows, n_k, 128] with a
    [rows, n_k] scale) — both pre-gathered and via the k_provider tile
    fetch — and must match the full path exactly, tie plateaus included."""
    torch.manual_seed(7)
    rows, heads, dim, keys, topk = 4, 2, 4, 2500, 17
    q = torch.randn(rows, heads, dim, dtype=torch.float16)
    weights = torch.rand(rows, heads, dtype=torch.float32)
    k_vals = torch.randn(rows, keys, dim, dtype=torch.float32)
    k_vals[:, 1200:] = 0.0  # tie plateau spanning multiple tiles
    k_scale = torch.rand(rows, keys, dtype=torch.float32)
    hi = torch.tensor([keys, 1500, 700, keys])

    full = _score_and_select(q, weights, k_vals, k_scale, None, hi, topk)
    streamed = _score_and_select_streaming(q, weights, k_vals, k_scale, None, hi, topk)
    assert torch.equal(streamed, full)

    def provider(k0: int, k1: int):
        return k_vals[:, k0:k1], k_scale[:, k0:k1]

    provided = _score_and_select_streaming(
        q, weights, None, None, None, hi, topk, n_k=keys, k_provider=provider
    )
    assert torch.equal(provided, full)
