# SPDX-License-Identifier: Apache-2.0
"""Test the diagnostic gate without CUDA or any serving changes."""

import pytest
import torch

from benchmarks.benchmark_glm5_next_pool_topk import validate_selection


def test_values_ties_empty_and_padding():
    logits = torch.tensor(
        [[4.0, 4.0, 2.0, 1.0], [3.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    lengths = torch.tensor([4, 1, 0])
    selected = torch.tensor([[1, 0], [0, -1], [-1, -1]])
    validate_selection(logits, lengths, selected)


@pytest.mark.parametrize("indices", [[0, 0], [0, 4], [0, 2], [0, -2], [0, -1]])
def test_rejects_duplicate_oob_wrong_rank_bad_padding_and_missing(indices):
    with pytest.raises(AssertionError):
        validate_selection(
            torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
            torch.tensor([4]),
            torch.tensor([indices]),
        )


def test_padding_cannot_be_a_different_negative_value():
    with pytest.raises(AssertionError):
        validate_selection(torch.ones(1, 4), torch.tensor([1]), torch.tensor([[0, -2]]))
