# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from benchmarks.kernels.replay_glm53_indexer_ties import canonical_selection


def test_score_then_pool_id_without_score_perturbation():
    logits = torch.tensor([[0.0, -0.0, 0.0, -0.0], [-8.0, -7.0, -8.0, -9.0]])
    before = logits.view(torch.uint8).clone()
    result = canonical_selection(
        logits,
        torch.zeros(2, dtype=torch.int32),
        torch.full((2,), 4, dtype=torch.int32),
        k=2,
    )
    assert torch.equal(result, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))
    assert torch.equal(before, logits.view(torch.uint8))


def test_neighboring_float_scores_are_not_ties():
    value = torch.tensor(-8.358503341674805, dtype=torch.float32)
    larger = torch.nextafter(value, torch.tensor(torch.inf))
    logits = torch.stack([value, larger, value])[None, :]
    result = canonical_selection(
        logits,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        k=1,
    )
    assert result.item() == 1


def test_empty_short_rows_and_undefined_nan_tails():
    logits = torch.tensor([[float("nan")] * 4, [-3.0, 7.0, float("nan"), float("nan")]])
    result = canonical_selection(
        logits,
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        k=3,
    )
    assert result.tolist() == [[-1, -1, -1], [0, 1, -1]]


@pytest.mark.parametrize("bad", ["start", "end", "nan"])
def test_invalid_oracle_input(bad):
    logits = torch.zeros(1, 3)
    starts, ends = (
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    if bad == "start":
        starts[0] = 1
    elif bad == "end":
        ends[0] = 4
    else:
        logits[0, 1] = float("nan")
    with pytest.raises(AssertionError):
        canonical_selection(logits, starts, ends, k=2)
