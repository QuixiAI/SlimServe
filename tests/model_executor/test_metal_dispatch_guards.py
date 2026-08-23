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
