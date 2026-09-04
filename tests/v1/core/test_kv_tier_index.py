# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HostKVTierIndex: trajectory staging, tail-boundary resume, eviction."""

from vllm.v1.core.kv_tier_index import HostKVTierIndex


def h(i: int) -> bytes:
    return i.to_bytes(8, "little")


def build_traj(idx, owner, n_blocks, tail=True, base=0):
    slots = []
    for i in range(n_blocks):
        s = idx.stage_attention(owner, i, h(base + i))
        assert s is not None
        slots.append(s)
    idx.confirm_writes(slots)
    if tail:
        st = idx.stage_tail(
            owner, n_blocks, [1, 1], boundary_hash=h(base + n_blocks - 1)
        )
        assert st is not None
        idx.confirm_writes([s for g in st for s in g])
    return slots


def test_resumable_only_with_confirmed_tail():
    idx = HostKVTierIndex(num_slots=16)
    slots = build_traj(idx, "a", 3, tail=False)
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None  # no tail state
    st = idx.stage_tail("a", 3, [1, 1], boundary_hash=h(2))
    assert idx.lookup([h(0), h(1), h(2)]) is None  # tail pending
    idx.confirm_writes([s for g in st for s in g])
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None
    owner, n, attn, states = hit
    assert owner == "a" and n == 3 and len(states) == 2
    assert attn == [[sl] for sl in slots]  # one attention group per row


def test_no_hit_on_hash_mismatch_or_short_prompt():
    idx = HostKVTierIndex(num_slots=16)
    build_traj(idx, "a", 3)
    assert idx.lookup([h(0), h(99), h(2)]) is None
    assert idx.lookup([h(0), h(1)]) is None  # prompt shorter than tail


def test_growing_trajectory_replaces_tail():
    idx = HostKVTierIndex(num_slots=32)
    build_traj(idx, "a", 2)
    used_after_first = idx.stats()["used"]
    # The trajectory grows: more attention blocks, new tail.
    for i in range(2, 5):
        s = idx.stage_attention("a", i, h(i))
        idx.confirm_writes([s])
    st = idx.stage_tail("a", 5, [1, 1], boundary_hash=h(4))
    idx.confirm_writes([s for g in st for s in g])
    hit = idx.lookup([h(i) for i in range(6)])
    assert hit is not None and hit[1] == 5
    # Old tail slots were released: net growth is 3 attn + 0 state slots.
    assert idx.stats()["used"] == used_after_first + 3


def test_attention_gap_blocks_resume():
    idx = HostKVTierIndex(num_slots=16)
    s0 = idx.stage_attention("a", 0, h(0))
    s2 = idx.stage_attention("a", 2, h(2))  # gap at 1
    idx.confirm_writes([s0, s2])
    st = idx.stage_tail("a", 3, [1], boundary_hash=h(2))
    idx.confirm_writes([s for g in st for s in g])
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None


def test_eviction_frees_whole_coldest_trajectory():
    idx = HostKVTierIndex(num_slots=8)
    build_traj(idx, "cold", 2)  # 2 attn + 2 state = 4 slots
    build_traj(idx, "hot", 2, base=10)  # 4 more: arena full
    idx.touch("hot")
    s = idx.stage_attention("new", 0, h(99))
    assert s is not None  # reclaimed "cold" wholesale
    idx.confirm_writes([s])
    assert idx.lookup([h(0), h(1), h(2)]) is None
    assert idx.lookup([h(10), h(11), h(12)]) is not None


def test_drop_owner():
    idx = HostKVTierIndex(num_slots=8)
    build_traj(idx, "gone", 2)
    idx.drop_owner("gone")
    assert idx.stats()["used"] == 0


# ---- owner-collision regression (2026-08-29) --------------------------------
# Owners are request-lineage keys, never content-derived. The old scheme
# keyed trajectories on block_hashes[0], collapsing every conversation that
# shared a system prompt into one chimera trajectory that matched nobody.


def test_shared_prefix_conversations_stay_separate():
    """Two conversations share block 0 but diverge after; each must remain
    independently resumable under its own owner."""
    idx = HostKVTierIndex(num_slots=32)
    # conv A: [h0, h1, h2]; conv B: [h0, h101, h102] (shared system block h0)
    for i, bh in enumerate([h(0), h(1), h(2)]):
        idx.confirm_writes([idx.stage_attention("req-A", i, bh)])
    st = idx.stage_tail("req-A", 3, [1], boundary_hash=h(2))
    idx.confirm_writes([s for g in st for s in g])
    for i, bh in enumerate([h(0), h(101), h(102)]):
        idx.confirm_writes([idx.stage_attention("req-B", i, bh)])
    st = idx.stage_tail("req-B", 3, [1], boundary_hash=h(102))
    idx.confirm_writes([s for g in st for s in g])

    hit_a = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit_a is not None and hit_a[0] == "req-A" and hit_a[1] == 3
    hit_b = idx.lookup([h(0), h(101), h(102), h(103)])
    assert hit_b is not None and hit_b[0] == "req-B" and hit_b[1] == 3


def test_tail_hash_mismatch_kills_resume():
    """A tail state paired with another chain's attention blocks must never
    match: resuming with the wrong mamba state corrupts output silently."""
    idx = HostKVTierIndex(num_slots=16)
    slots = [idx.stage_attention("x", i, h(i)) for i in range(3)]
    idx.confirm_writes(slots)
    # Tail recorded by a diverged continuation: its chain hash at the
    # boundary block differs from what is actually staged there.
    st = idx.stage_tail("x", 3, [1], boundary_hash=h(999))
    idx.confirm_writes([s for g in st for s in g])
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None
    # A consistent tail from the true continuation restores resumability.
    st = idx.stage_tail("x", 3, [1], boundary_hash=h(2))
    idx.confirm_writes([s for g in st for s in g])
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is not None


def test_missing_boundary_hash_is_unresumable():
    """Tails saved without a chain hash (or before their attention block is
    staged) never match - resume correctness beats coverage."""
    idx = HostKVTierIndex(num_slots=16)
    slots = [idx.stage_attention("x", i, h(i)) for i in range(3)]
    idx.confirm_writes(slots)
    st = idx.stage_tail("x", 3, [1])  # default empty boundary_hash
    idx.confirm_writes([s for g in st for s in g])
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None


def test_adopted_owner_extends_one_lineage():
    """A resumed conversation staging under the adopted owner grows the
    original trajectory rather than duplicating it."""
    idx = HostKVTierIndex(num_slots=32)
    build_traj(idx, "turn-1", 3)
    hit = idx.lookup([h(i) for i in range(5)])
    assert hit is not None and hit[0] == "turn-1"
    owner = hit[0]  # connector adopts this for the resuming request
    for i in range(3, 5):
        idx.confirm_writes([idx.stage_attention(owner, i, h(i))])
    st = idx.stage_tail(owner, 5, [1, 1], boundary_hash=h(4))
    idx.confirm_writes([s for g in st for s in g])
    assert idx.stats()["trajectories"] == 1
    hit = idx.lookup([h(i) for i in range(6)])
    assert hit is not None and hit[0] == "turn-1" and hit[1] == 5


# --------------------------------------------------------- attention-only
#
# Models with no state groups (MLA/DSA: GLM-5.2, DeepSeek-V4) never stage a
# tail. Gating resumability on one made the tier write-only on exactly the
# models the MI300X profiles serve.


def build_attn_only(idx, owner, n_blocks, base=0, seal=True):
    slots = []
    for i in range(n_blocks):
        s = idx.stage_attention(owner, i, h(base + i))
        assert s is not None
        slots.append(s)
    idx.confirm_writes(slots)
    if seal:
        assert idx.mark_complete(owner)
    return slots


def test_attention_only_resumes_without_a_tail():
    idx = HostKVTierIndex(num_slots=16, requires_tail=False)
    slots = build_attn_only(idx, "a", 3)
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None
    owner, n, attn, states = hit
    assert owner == "a" and n == 3 and states == []
    assert attn == [[sl] for sl in slots]


def test_attention_only_unsealed_trajectory_is_not_matchable():
    """A trajectory its owner is still growing must not be adopted."""
    idx = HostKVTierIndex(num_slots=16, requires_tail=False)
    build_attn_only(idx, "a", 3, seal=False)
    assert idx.lookup([h(0), h(1), h(2)]) is None
    assert idx.mark_complete("a")
    assert idx.lookup([h(0), h(1), h(2)]) is not None


def test_attention_only_growth_unseals_until_finish():
    idx = HostKVTierIndex(num_slots=16, requires_tail=False)
    build_attn_only(idx, "a", 2)
    assert idx.lookup([h(0), h(1)]) is not None
    # The resumed request extends the adopted lineage: no longer matchable
    # until it finishes in turn.
    s = idx.stage_attention("a", 2, h(2))
    idx.confirm_writes([s])
    assert idx.lookup([h(0), h(1), h(2)]) is None
    assert idx.mark_complete("a")
    hit = idx.lookup([h(0), h(1), h(2)])
    assert hit is not None and hit[1] == 3


def test_attention_only_gap_truncates_and_hash_mismatch_misses():
    idx = HostKVTierIndex(num_slots=16, requires_tail=False)
    # Stage 0 and 2, leaving position 1 unstaged: the span ends at the gap.
    for i in (0, 2):
        s = idx.stage_attention("a", i, h(i))
        idx.confirm_writes([s])
    assert idx.mark_complete("a")
    hit = idx.lookup([h(0), h(1), h(2)])
    assert hit is not None and hit[1] == 1
    assert idx.lookup([h(99), h(1), h(2)]) is None


def test_attention_only_pending_write_blocks_resume():
    idx = HostKVTierIndex(num_slots=16, requires_tail=False)
    build_attn_only(idx, "a", 2, seal=False)
    s = idx.stage_attention("a", 2, h(2))  # staged, never confirmed
    assert s is not None
    assert idx.mark_complete("a")
    assert idx.lookup([h(0), h(1), h(2)]) is None


def test_mark_complete_on_unknown_owner_is_false():
    idx = HostKVTierIndex(num_slots=16, requires_tail=False)
    assert idx.mark_complete("nobody") is False


def test_tail_model_still_requires_its_tail():
    """Regression: sealing must not make a state model resumable."""
    idx = HostKVTierIndex(num_slots=16)  # requires_tail defaults True
    build_traj(idx, "a", 3, tail=False)
    idx.mark_complete("a")
    assert idx.lookup([h(0), h(1), h(2)]) is None


# ------------------------------------------------------------- per-group
#
# DeepSeek-V4 forms several attention groups (MLA, SWA, C4/C128
# compressors) whose blocks are distinct slab rows: a logical block needs
# one host slot per group, and a position counts only once every group
# holds it.


def test_multi_group_position_needs_every_group():
    idx = HostKVTierIndex(num_slots=16, requires_tail=False, slots_per_position=2)
    a0 = idx.stage_attention("a", 0, h(0), sub=0)
    a1 = idx.stage_attention("a", 0, h(0), sub=1)
    b0 = idx.stage_attention("a", 1, h(1), sub=0)  # group 1 missing
    idx.confirm_writes([a0, a1, b0])
    assert idx.mark_complete("a")
    hit = idx.lookup([h(0), h(1)])
    assert hit is not None
    owner, n, rows, _ = hit
    assert n == 1 and rows == [[a0, a1]]
    # Staging the missing group completes position 1.
    b1 = idx.stage_attention("a", 1, h(1), sub=1)
    idx.confirm_writes([b1])
    idx.mark_complete("a")
    hit = idx.lookup([h(0), h(1)])
    assert hit is not None and hit[1] == 2 and hit[2] == [[a0, a1], [b0, b1]]


def test_multi_group_restaging_a_held_group_is_refused():
    idx = HostKVTierIndex(num_slots=16, requires_tail=False, slots_per_position=2)
    assert idx.stage_attention("a", 0, h(0), sub=0) is not None
    assert idx.stage_attention("a", 0, h(0), sub=0) is None
    assert idx.stage_attention("a", 0, h(0), sub=1) is not None


def test_multi_group_eviction_frees_every_groups_slots():
    idx = HostKVTierIndex(num_slots=2, requires_tail=False, slots_per_position=2)
    for i in range(2):
        for g in range(2):
            s = idx.stage_attention("a", i, h(i), sub=g)
            assert s is not None
            idx.confirm_writes([s])
    # A position's sub-blocks share one slot row: 2 positions = 2 slots.
    assert idx.stats()["used"] == 2
    idx._free.clear()
    # Arena full: staging for another owner reclaims "a" wholesale.
    assert idx.stage_attention("b", 0, h(50), sub=0) is not None
    assert idx.stats()["trajectories"] == 1 and idx.stats()["used"] == 1


# ----------------------------------------------------------------- pins
#
# lookup() hands out slot ids that the DMA reads a step or more later.
# Without a pin, staging for another request could reclaim the source
# trajectory in between and re-stage those slots ahead of the restore.


def test_pinned_restore_sources_survive_reclaim():
    idx = HostKVTierIndex(num_slots=3, requires_tail=False)
    slots = build_attn_only(idx, "a", 3)
    hit = idx.lookup([h(0), h(1), h(2)])
    assert hit is not None
    pinned = [s for row in hit[2] for s in row]
    idx.pin_slots(pinned)
    # Arena is full and the only trajectory is pinned: staging must fail
    # rather than evict the restore source.
    assert idx.stage_attention("b", 0, h(50)) is None
    assert idx.lookup([h(0), h(1), h(2)]) is not None
    assert idx.stats()["pinned"] == 3
    idx.unpin_slots(pinned)
    assert idx.stats()["pinned"] == 0
    assert idx.stage_attention("b", 0, h(50)) is not None
    assert idx.lookup([h(0), h(1), h(2)]) is None  # reclaimed once unpinned
    assert slots  # (kept for readability)


def test_pins_are_refcounted():
    idx = HostKVTierIndex(num_slots=2, requires_tail=False)
    build_attn_only(idx, "a", 2)
    idx.pin_slots([0, 1])
    idx.pin_slots([0])
    idx.unpin_slots([0, 1])
    assert idx.stats()["pinned"] == 1  # slot 0 still held once
    assert idx.stage_attention("b", 0, h(50)) is None
    idx.unpin_slots([0])
    assert idx.stage_attention("b", 0, h(50)) is not None


def test_drop_owner_respects_pins():
    idx = HostKVTierIndex(num_slots=4, requires_tail=False)
    build_attn_only(idx, "a", 2)
    idx.pin_slots([0])
    idx.drop_owner("a")
    assert idx.stats()["trajectories"] == 1
    idx.unpin_slots([0])
    idx.drop_owner("a")
    assert idx.stats()["trajectories"] == 0


def test_attention_only_lookup_clamps_to_a_shorter_query_prefix():
    """A sealed trajectory longer than the query must still serve the
    query's full prefix (attention-only); with a tail it must not."""
    from vllm.v1.core.kv_tier_index import HostKVTierIndex

    idx = HostKVTierIndex(num_slots=64, requires_tail=False)
    hashes = [f"h{i}".encode() for i in range(6)]
    for i, h in enumerate(hashes):
        idx.stage_attention("own", i, h, sub=0)
    idx.confirm_writes(list(range(6)))
    assert idx.mark_complete("own")
    hit = idx.lookup(hashes[:4])
    assert hit is not None
    owner, n, rows, tails = hit
    assert owner == "own" and n == 4 and len(rows) == 4 and tails == []
    tail_idx = HostKVTierIndex(num_slots=64, requires_tail=True)
    for i, h in enumerate(hashes):
        tail_idx.stage_attention("own", i, h, sub=0)
    tail_idx.confirm_writes(list(range(6)))
    assert tail_idx.lookup(hashes[:4]) is None
