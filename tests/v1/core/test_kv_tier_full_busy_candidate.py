# SPDX-License-Identifier: Apache-2.0
"""Full-busy rejection must stop immediately and recover on real ACKs."""

from unittest.mock import Mock

import pytest

from vllm.v1.core.kv_tier_index import HostKVTierIndex


@pytest.mark.parametrize("disk", [False, True])
def test_full_busy_guard_does_not_scan_and_ack_allows_allocation(disk):
    index = HostKVTierIndex(2, num_disk_slots=8 if disk else 0)
    slots = [index.stage_attention(owner, 0, owner.encode()) for owner in ("a", "b")]
    assert slots == [0, 1]
    writes = []
    if disk:
        index.confirm_writes(slots)
        writes = index.take_disk_writes(slots)
        assert len(index._host_busy) == index.num_slots
    reclaim = index._reclaim
    index._reclaim = Mock(side_effect=AssertionError("scanned a fully busy arena"))
    assert index.stage_attention("c", 0, b"c") is None
    index._reclaim = reclaim
    # Completing just a's copy makes its one-slot trajectory reclaimable.
    # A memoized miss would fail this check.
    if disk:
        index.confirm_disk_writes(writes[:1])
    else:
        index.confirm_writes(slots[:1])
    assert index.stage_attention("c", 0, b"c") == 0


def test_partial_busy_sets_fall_back_to_existing_reclamation():
    index = HostKVTierIndex(3, num_disk_slots=8)
    slots = [
        index.stage_attention(owner, 0, owner.encode()) for owner in ("a", "b", "c")
    ]
    index.confirm_writes(slots[:2])
    writes = index.take_disk_writes(slots[:1])
    assert len(index._host_busy) == len(index._pending_write) == 1
    # a is disk-busy, c host-write-pending, b is ready and must be reclaimed.
    assert index.stage_attention("d", 0, b"d") == 1
    index.confirm_disk_writes(writes)


def test_free_slot_does_not_require_a_reclamation_scan():
    index = HostKVTierIndex(2)
    index.stage_attention("a", 0, b"a")
    index._reclaim = Mock(side_effect=AssertionError("reclaimed with a free slot"))
    assert index.stage_attention("b", 0, b"b") == 1


def test_zero_capacity_remains_rejected():
    with pytest.raises(AssertionError):
        HostKVTierIndex(0)
