# SPDX-License-Identifier: Apache-2.0
"""Closed-form CPU checks of the independent high-precision oracle."""

import torch

from benchmarks.glm5_next_pool_oracle import (
    error_record,
    pool_keys_fp64,
    scores_fp64,
)


def test_uniform_gates_average_only_completed_groups():
    rows = torch.zeros(9, 256, dtype=torch.bfloat16)
    rows[:, :128] = torch.arange(9)[:, None]
    result = pool_keys_fp64(rows, torch.zeros(4, 128))
    expected = torch.tensor([1.5, 5.5], dtype=torch.float64)[:, None].expand(2, 128)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_pool_softmax_is_stable_under_large_common_offset():
    rows = torch.randn(8, 256, dtype=torch.bfloat16)
    offsets = torch.arange(4, dtype=torch.float64)[:, None].expand(4, 128)
    reference = pool_keys_fp64(rows, offsets)
    shifted = pool_keys_fp64(rows, offsets + 1000)
    torch.testing.assert_close(shifted, reference, rtol=1e-12, atol=1e-12)


def test_scores_clamp_each_head_before_weighted_sum():
    pools = torch.ones(2, 128, dtype=torch.bfloat16)
    query = torch.ones(32, 128, dtype=torch.bfloat16)
    query[0] = -1
    weights = torch.zeros(32)
    weights[0], weights[1] = 100, -2
    expected = torch.full((2,), -2 * 128**0.5, dtype=torch.float64)
    torch.testing.assert_close(scores_fp64(pools, query, weights), expected)


def test_error_record_fails_nonfinite_values():
    record = error_record(
        torch.tensor([float("nan"), 1.0]), torch.ones(2), atol=0.002, rtol=0.002
    )
    assert record["failing_values"] == 1
