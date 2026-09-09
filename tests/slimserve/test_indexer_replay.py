# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from benchmarks.kernels.replay_glm53_indexer import verified_selection


def fixture():
    logits = torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 2.0], [2.0, 0.0, 0.0]])
    starts = torch.zeros(3, dtype=torch.int32)
    ends = torch.tensor([0, 3, 1], dtype=torch.int32)
    indices = torch.tensor([[-1, -1], [2, 1], [0, -1]], dtype=torch.int32)
    return logits, starts, ends, indices


def test_ragged_selection_oracle_and_ties():
    data = fixture()
    ordered, ties = verified_selection(*data, k=2)
    assert ties == 0 and torch.equal(ordered, data[-1].sort(dim=1).values)
    data[0][1] = 0
    _, ties = verified_selection(*data, k=2)
    assert ties == 1


@pytest.mark.parametrize(
    "bad", ["range", "duplicate", "score", "tail", "nan", "padding"]
)
def test_bad_capture_rejected(bad):
    logits, starts, ends, indices = fixture()
    if bad == "range":
        indices[1, 0] = 3
    elif bad == "duplicate":
        indices[1, 0] = 1
    elif bad == "score":
        indices[1, 0] = 0
    elif bad == "tail":
        logits[0, 0] = 1
    elif bad == "nan":
        logits[1, 0] = float("nan")
    elif bad == "padding":
        indices[0, 0] = 0
    with pytest.raises(AssertionError):
        verified_selection(logits, starts, ends, indices, k=2)
