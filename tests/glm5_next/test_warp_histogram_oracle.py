# SPDX-License-Identifier: Apache-2.0
"""CPU oracle tests; native helper execution is a separate GPU gate."""

import pytest
import torch

from benchmarks.benchmark_glm5_next_warp_histogram import (
    edge_case_inputs,
    histogram_oracle,
)


@pytest.mark.parametrize("length", [0, 1, 3, 4, 511, 512, 513, 32768])
def test_visible_counts_and_poisoned_padding(length):
    logits = torch.full((2, length + 4), float("nan"))
    logits[0, :length] = 1.0
    logits[1, :length] = -1.0
    result = histogram_oracle(logits, torch.tensor([length, length]))
    assert result.shape == (2, 2048)
    assert result[0, 543] == length  # ~fp16(1.0) >> 5
    assert result[1, 1504] == length  # fp16(-1.0) >> 5
    assert result.sum().item() == 2 * length


def test_signed_zero_and_infinities_follow_sampler_bin_order():
    logits = torch.tensor([[0.0, -0.0, float("inf"), -float("inf")]])
    result = histogram_oracle(logits, torch.tensor([4]))
    assert result[0].nonzero().flatten().tolist() == [31, 1023, 1024, 2016]
    assert result.max() == 1


@pytest.mark.parametrize("length", [-1, 5])
def test_invalid_visible_range_is_rejected(length):
    with pytest.raises(AssertionError):
        histogram_oracle(torch.ones(1, 4), torch.tensor([length]))


def test_native_edge_fixture_covers_partial_warps_and_vector_tails():
    logits, lengths = edge_case_inputs()
    assert logits.shape == (32, 4096)
    assert {0, 1, 2, 3} == {int(length) % 4 for length in lengths}
    assert {127, 128, 129, 2047, 2048, 2049}.issubset(set(lengths.tolist()))
    result = histogram_oracle(logits, lengths)
    torch.testing.assert_close(result.sum(dim=1), lengths.long(), atol=0, rtol=0)
    for row, length in enumerate(lengths.tolist()):
        assert not logits[row, :length].isnan().any()
        assert logits[row, length:].isnan().all()
