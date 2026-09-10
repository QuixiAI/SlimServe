# SPDX-License-Identifier: Apache-2.0
"""Busy and disk-only trajectories must not materialize slot lists to evict."""

from unittest.mock import Mock

from vllm.v1.core.kv_tier_index import HostKVTierIndex, Trajectory


def test_busy_check_short_circuits_before_later_positions():
    class Unvisited(dict):
        def values(self):
            raise AssertionError("busy check walked past a known busy slot")

    index = HostKVTierIndex(2)
    index._pending_write.add(0)
    trajectory = Trajectory(attn_slots=[{0: 0}, Unvisited({0: 1})])
    assert index._busy(trajectory)


def test_tail_slots_and_late_attention_slots_are_checked():
    index = HostKVTierIndex(3)
    trajectory = Trajectory(attn_slots=[{}, {0: 0}, {1: 1}], tail_state_slots={0: 2})
    assert not index._busy(trajectory)
    index._host_busy[2] = 42
    assert index._busy(trajectory)
    index._host_busy.clear()
    index._pending_write.add(1)
    assert index._busy(trajectory)


def test_reclaim_does_not_materialize_disk_only_or_busy_slots():
    index = HostKVTierIndex(1, num_disk_slots=16)
    disk_only = Trajectory(attn_slots=[{} for _ in range(216)])
    disk_only.host_slots = Mock(side_effect=AssertionError("disk-only materialized"))
    index._trajectories["disk"] = disk_only
    assert index.stage_attention("busy", 0, b"a") == 0
    busy = index._trajectories["busy"]
    busy.host_slots = Mock(side_effect=AssertionError("busy slots materialized"))
    assert not index._reclaim("new")
    assert not index._free


def test_ready_trajectory_still_reclaims_in_lru_order():
    index = HostKVTierIndex(2)
    a = index.stage_attention("a", 0, b"a")
    b = index.stage_attention("b", 0, b"b")
    index.confirm_writes([a, b])
    index.touch("a")
    assert index.stage_attention("c", 0, b"c") == b
    assert list(index._trajectories) == ["a", "c"]


def test_ready_tail_only_trajectory_is_not_skipped():
    index = HostKVTierIndex(1)
    tail = index.stage_tail_states("a", 1, 1, boundary_hash=b"a")
    index.confirm_writes(list(tail.values()))
    assert index.stage_attention("b", 0, b"b") is not None
    assert "a" not in index._trajectories
