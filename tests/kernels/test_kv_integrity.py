# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The opt-in KV slot-mapping integrity check.

Paged attention is bookkeeping: a slot mapping that addresses the wrong page
returns another request's tokens while the model stays fluent, so the failure
reads as a quality problem rather than a bug. These pin that the check catches
the two shapes it claims to and stays quiet on a healthy batch.
"""

import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import _check_slot_mappings


def _mapping(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int64)


def test_healthy_batch_passes():
    """Distinct slots per token, padding where there is no token."""
    mapping = _mapping([[0, 1, 2, 3, PAD_SLOT_ID], [10, 11, 12, 13, PAD_SLOT_ID]])
    _check_slot_mappings(mapping, mapping.shape[1])


def test_padding_may_repeat():
    """Only real slots must be unique; the pad sentinel repeats by design."""
    mapping = _mapping([[7, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID]])
    _check_slot_mappings(mapping, mapping.shape[1])


def test_two_tokens_sharing_one_slot_is_caught():
    """The aliasing case: one token's KV silently overwrites another's."""
    mapping = _mapping([[4, 5, 4, 6]])
    with pytest.raises(RuntimeError, match="claimed 2 times"):
        _check_slot_mappings(mapping, mapping.shape[1])


def test_negative_slot_that_is_not_the_sentinel_is_caught():
    """A garbage index that survived as a small negative, not a pad marker."""
    mapping = _mapping([[1, -3, 2]])
    with pytest.raises(RuntimeError, match="negative slots"):
        _check_slot_mappings(mapping, mapping.shape[1])


def test_only_the_live_prefix_is_examined():
    """Slots past num_tokens belong to the previous step and are not ours."""
    mapping = _mapping([[1, 2, 9, 9]])
    _check_slot_mappings(mapping, 2)
    with pytest.raises(RuntimeError):
        _check_slot_mappings(mapping, 4)


def test_each_group_is_checked_independently():
    """Hybrid models have several caches; the same slot id in two of them is
    two different pages, not a collision."""
    mapping = _mapping([[3, 4], [3, 4]])
    _check_slot_mappings(mapping, 2)
