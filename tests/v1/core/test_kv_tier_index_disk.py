# SPDX-License-Identifier: Apache-2.0
"""HostKVTierIndex NVMe tier: write-through, demotion, promotion, reclaim."""

from vllm.v1.core.kv_tier_index import HostKVTierIndex


def h(i: int) -> bytes:
    return i.to_bytes(8, "little")


def build(idx, owner, n, base=0, through=True):
    """Stage n attention blocks + a 2-group tail, confirm host writes, and
    (optionally) run the write-through to completion."""
    slots = []
    for i in range(n):
        s = idx.stage_attention(owner, i, h(base + i))
        assert s is not None
        slots.append(s)
    st = idx.stage_tail_states(owner, n, 2, boundary_hash=h(base + n - 1))
    assert st is not None
    host = slots + list(st.values())
    idx.confirm_writes(host)
    writes = idx.take_disk_writes(host)
    assert len(writes) == len(host)
    if through:
        idx.confirm_disk_writes(writes)
    return slots, st, writes


def test_every_host_slot_gets_a_disk_slot_and_writes_through():
    idx = HostKVTierIndex(num_slots=16, num_disk_slots=16)
    slots, st, writes = build(idx, "a", 3, through=False)
    s = idx.stats()
    assert s["disk_used"] == 5 and s["disk_pending"] == 5
    # Busy host slots (disk write reading them) are never reclaimed.
    assert not idx._reclaim(protect="zzz")
    idx.confirm_disk_writes(writes)
    assert idx.stats()["disk_pending"] == 0
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is not None


def test_host_pressure_demotes_a_fully_written_trajectory():
    idx = HostKVTierIndex(num_slots=5, num_disk_slots=32)
    build(idx, "a", 3)  # 5 host slots: full
    assert idx.stats()["used"] == 5
    # A second conversation needs host slots: "a" is demoted, not deleted.
    s = idx.stage_attention("b", 0, h(100))
    assert s is not None
    st = idx.stats()
    assert st["trajectories"] == 2 and st["disk_only"] == 1
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None
    owner, n, attn, tail = hit
    assert owner == "a" and n == 3 and attn == [{}, {}, {}] and tail == {}
    assert idx.needs_promotion(hit)


def test_unwritten_trajectory_is_deleted_not_demoted():
    idx = HostKVTierIndex(num_slots=5, num_disk_slots=32)
    slots, st, writes = build(idx, "a", 3, through=False)
    idx.confirm_disk_writes(writes[:2])  # partial write-through
    # Reclaim must skip "a" while its remaining disk writes hold host slots.
    assert idx.stage_attention("b", 0, h(100)) is None
    idx.confirm_disk_writes(writes[2:])
    # Fully on disk now -> demotion.
    assert idx.stage_attention("b", 0, h(100)) is not None
    assert idx.stats()["disk_only"] == 1


def test_promotion_allocates_host_slots_and_reads_from_disk():
    idx = HostKVTierIndex(num_slots=5, num_disk_slots=32)
    build(idx, "a", 3)  # host full
    sb = idx.stage_attention("b", 0, h(100))  # demotes "a"
    idx.confirm_writes([sb])  # b is reclaimable (not on disk -> deleted)
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    owner, n, _, _ = hit
    assert idx.needs_promotion(hit)
    promoted = idx.promote(owner, n)  # evicts b to make room
    assert promoted is not None
    attn, tail, reads = promoted
    assert len(attn) == 3 and set(tail) == {0, 1} and len(reads) == 5
    assert all(0 in d for d in attn)
    # Until the worker completes the reads the slots are pending and the
    # trajectory is not offered to another request.
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None
    idx.confirm_promotion(owner)
    hit2 = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit2 is not None and not idx.needs_promotion(hit2)
    assert hit2[2] == attn and hit2[3] == tail


def test_promotion_rolls_back_when_host_is_full():
    idx = HostKVTierIndex(num_slots=5, num_disk_slots=32)
    build(idx, "a", 3)
    # Fill the host with an unfinished (busy) trajectory so nothing can be
    # reclaimed for the promotion.
    for i in range(5):
        pass
    idx.stage_attention("b", 0, h(100))  # demotes a, uses 1 slot
    for i in range(1, 5):
        assert idx.stage_attention("b", i, h(100 + i)) is not None
    # "b" is pending (unconfirmed) -> not reclaimable.
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None
    assert idx.promote(hit[0], hit[1]) is None
    assert idx.stats()["pending_writes"] == 5  # only b's
    # Still resumable from disk once room appears.
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is not None


def test_disk_pressure_reclaims_coldest_disk_copies():
    idx = HostKVTierIndex(num_slots=64, num_disk_slots=5)
    build(idx, "a", 3)  # 5 disk slots: full
    slots, st, writes = build(idx, "b", 3)  # reclaims a's disk copies
    s = idx.stats()
    assert s["disk_used"] == 5 and s["trajectories"] == 2
    # "a" is still host-resident and resumable; it just has no disk copy.
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None and hit[0] == "a" and not idx.needs_promotion(hit)


def test_disk_only_trajectory_dies_when_its_disk_is_reclaimed():
    idx = HostKVTierIndex(num_slots=5, num_disk_slots=5)
    build(idx, "a", 3)
    idx.stage_attention("b", 0, h(100))  # demotes a (disk-only now)
    for i in range(1, 5):
        idx.stage_attention("b", i, h(100 + i))  # takes a's disk slots
    assert idx.stats()["trajectories"] == 1
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None


def test_growing_lineage_replaces_disk_tail():
    idx = HostKVTierIndex(num_slots=32, num_disk_slots=32)
    build(idx, "a", 2)
    used = idx.stats()["disk_used"]
    s = idx.stage_attention("a", 2, h(2))
    st = idx.stage_tail_states("a", 3, 2, boundary_hash=h(2))
    host = [s] + list(st.values())
    idx.confirm_writes(host)
    idx.confirm_disk_writes(idx.take_disk_writes(host))
    # One new attention block; the old tail's two disk slots were freed.
    assert idx.stats()["disk_used"] == used + 1
    assert idx.lookup([h(0), h(1), h(2), h(3)])[1] == 3


def test_no_disk_tier_is_the_old_behaviour():
    idx = HostKVTierIndex(num_slots=5)
    build_slots = []
    for i in range(3):
        build_slots.append(idx.stage_attention("a", i, h(i)))
    st = idx.stage_tail_states("a", 3, 2, boundary_hash=h(2))
    idx.confirm_writes(build_slots + list(st.values()))
    assert idx.take_disk_writes(build_slots) == []
    assert idx.stage_attention("b", 0, h(9)) is not None  # deletes a
    assert idx.stats()["trajectories"] == 1


def test_tail_restage_does_not_free_busy_slots():
    """Re-staging a tail while the previous tail's write-through is in
    flight must keep both the busy host slot and its unconfirmed disk slot
    off the free lists until the write confirms (2026-09-04)."""
    from vllm.v1.core.kv_tier_index import HostKVTierIndex

    idx = HostKVTierIndex(8, attn_gids=[0], num_disk_slots=8)
    first = idx.stage_tail_states("o", 2, 1, boundary_hash=b"h1")
    assert first is not None
    h_old = first[0]
    idx.confirm_writes([h_old])
    writes = idx.take_disk_writes([h_old])  # write-through now in flight
    assert writes and writes[0][0] == h_old
    d_old = writes[0][1]
    second = idx.stage_tail_states("o", 4, 1, boundary_hash=b"h2")
    assert second is not None
    h_new = second[0]
    assert h_new != h_old, "busy host slot handed to the new tail"
    d_new = idx._disk_of_host[h_new]
    assert d_new != d_old, "unconfirmed disk slot handed to the new tail"
    assert h_old not in idx._free and d_old not in idx._disk_free
    idx.confirm_disk_writes(writes)
    assert h_old in idx._free and d_old in idx._disk_free
    assert len(set(idx._free)) == len(idx._free)
    assert len(set(idx._disk_free)) == len(idx._disk_free)


# --- main-KV tier slots (host-resident main KV, milestone 4) ---------------


def test_main_slot_reserved_at_alloc_then_confirmed_makes_position_resumable():
    idx = HostKVTierIndex(num_slots=32, num_disk_slots=0, num_main_slots=8)
    assert idx.require_main
    # Reserved before the block fills; the same slot is returned again.
    m0 = idx.reserve_main_slot("a", 0)
    assert m0 is not None and idx.reserve_main_slot("a", 0) == m0
    m1 = idx.reserve_main_slot("a", 1)
    s0 = idx.stage_attention("a", 0, h(0))
    s1 = idx.stage_attention("a", 1, h(1))
    st = idx.stage_tail_states("a", 2, 2, boundary_hash=h(1))
    idx.confirm_writes([s0, s1, *st.values()])
    # Slab rows confirmed but main slots still pending: not resumable.
    assert idx.lookup([h(0), h(1), h(2)]) is None
    idx.confirm_main([m0])
    assert idx.lookup([h(0), h(1), h(2)]) is None  # position 1 still pending
    idx.confirm_main([m1])
    hit = idx.lookup([h(0), h(1), h(2)])
    assert hit is not None and hit[1] == 2
    assert idx.main_slots_for("a", 2) == [m0, m1]
    assert idx.stats()["main_used"] == 2 and idx.stats()["main_pending"] == 0


def test_unfilled_reservations_are_released_at_finish():
    idx = HostKVTierIndex(num_slots=32, num_main_slots=4)
    for i in range(3):
        idx.reserve_main_slot("a", i)
    s0 = idx.stage_attention("a", 0, h(0))
    idx.confirm_writes([s0]); idx.confirm_main([idx.main_slot("a", 0)])
    freed = idx.release_main_reservations("a", filled_upto=1)
    assert len(freed) == 2 and idx.stats()["main_used"] == 1
    assert idx.main_slot("a", 1) is None and idx.main_slot("a", 0) is not None


def test_pinned_main_slots_block_reclaim_until_unpinned():
    idx = HostKVTierIndex(num_slots=64, num_main_slots=2)
    for i in range(2):
        idx.reserve_main_slot("a", i)
        s = idx.stage_attention("a", i, h(i)); idx.confirm_writes([s])
    st = idx.stage_tail_states("a", 2, 2, boundary_hash=h(1)); idx.confirm_writes(list(st.values()))
    idx.confirm_main(idx.main_slots_for("a", 2))
    idx.pin_main("a", 2, "req-x")  # a resumer has the slots rebound
    # Main tier full: a new lineage cannot take a's slots while pinned.
    assert idx.reserve_main_slot("b", 0) is None
    assert idx.lookup([h(0), h(1), h(2)]) is not None
    idx.unpin_main("req-x")
    assert idx.reserve_main_slot("b", 0) is not None  # a reclaimed
    assert idx.stats()["trajectories"] == 1


def test_trajectory_with_main_slots_is_deleted_not_demoted_under_host_pressure():
    idx = HostKVTierIndex(num_slots=5, num_disk_slots=32, num_main_slots=8)
    for i in range(3):
        idx.reserve_main_slot("a", i)
    build(idx, "a", 3)  # fully written through to disk (slab rows)
    idx.confirm_main(idx.main_slots_for("a", 3))
    assert idx.stage_attention("b", 0, h(100)) is not None  # needs a host slot
    st = idx.stats()
    assert st["trajectories"] == 1 and st["disk_only"] == 0 and st["main_used"] == 0
