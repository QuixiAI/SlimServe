# SPDX-License-Identifier: Apache-2.0
import torch

from benchmarks.kernels.check_glm53_indexer_order import selection_validity


def test_membership_checker_accepts_order_and_ties_but_not_bad_selection():
    scores = torch.tensor([[1.0, 3.0, 3.0, 0.0]])
    assert selection_validity(scores, torch.tensor([[2, 1]]), 2)
    assert selection_validity(scores, torch.tensor([[1]]), 1)
    assert selection_validity(scores, torch.tensor([[2]]), 1)
    for ids in ([[1, 1]], [[0, 1]], [[1, -1]], [[1, 4]]):
        assert not selection_validity(scores, torch.tensor(ids), 2)


def test_membership_checker_checks_short_row_padding():
    scores = torch.tensor([[1.0, 3.0]])
    assert selection_validity(scores, torch.tensor([[1, 0, -1, -1]]), 4)
    assert not selection_validity(scores, torch.tensor([[1, 0, -1, 99]]), 4)
