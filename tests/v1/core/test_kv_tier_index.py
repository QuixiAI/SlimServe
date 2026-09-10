# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HostKVTierIndex: trajectory staging, tail-boundary resume, eviction."""

from vllm.v1.core.kv_tier_index import HostKVTierIndex


def h(i: int) -> bytes:
    return i.to_bytes(8, "little")


def build_traj(idx, owner, n_blocks, tail=True, base=0):
    slots = []
    for i in range(n_blocks):
        s = idx.stage_attention(owner, 0, i, h(base + i))
        assert s is not None
        slots.append(s)
    idx.confirm_writes(slots)
    if tail:
        st = idx.stage_tail_states(
            owner, n_blocks, 2, boundary_hash=h(base + n_blocks - 1)
        )
        assert st is not None
        idx.confirm_writes(list(st.values()))
    return slots


def test_resumable_only_with_confirmed_tail():
    idx = HostKVTierIndex(num_slots=16)
    slots = build_traj(idx, "a", 3, tail=False)
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None  # no tail state
    st = idx.stage_tail_states("a", 3, 2, boundary_hash=h(2))
    assert idx.lookup([h(0), h(1), h(2)]) is None  # tail pending
    idx.confirm_writes(list(st.values()))
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None
    owner, n, attn, states = hit
    assert owner == "a" and n == 3 and attn[0][:n] == slots
    assert set(states) == {0, 1}


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
        s = idx.stage_attention("a", 0, i, h(i))
        idx.confirm_writes([s])
    st = idx.stage_tail_states("a", 5, 2, boundary_hash=h(4))
    idx.confirm_writes(list(st.values()))
    hit = idx.lookup([h(i) for i in range(6)])
    assert hit is not None and hit[1] == 5
    # Old tail slots were released: net growth is 3 attn + 0 state slots.
    assert idx.stats()["used"] == used_after_first + 3


def test_attention_gap_blocks_resume():
    idx = HostKVTierIndex(num_slots=16)
    s0 = idx.stage_attention("a", 0, 0, h(0))
    s2 = idx.stage_attention("a", 0, 2, h(2))  # gap at 1
    idx.confirm_writes([s0, s2])
    st = idx.stage_tail_states("a", 3, 1, boundary_hash=h(2))
    idx.confirm_writes(list(st.values()))
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None


def test_eviction_frees_whole_coldest_trajectory():
    idx = HostKVTierIndex(num_slots=8)
    build_traj(idx, "cold", 2)  # 2 attn + 2 state = 4 slots
    build_traj(idx, "hot", 2, base=10)  # 4 more: arena full
    idx.touch("hot")
    s = idx.stage_attention("new", 0, 0, h(99))
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
        idx.confirm_writes([idx.stage_attention("req-A", 0, i, bh)])
    st = idx.stage_tail_states("req-A", 3, 1, boundary_hash=h(2))
    idx.confirm_writes(list(st.values()))
    for i, bh in enumerate([h(0), h(101), h(102)]):
        idx.confirm_writes([idx.stage_attention("req-B", 0, i, bh)])
    st = idx.stage_tail_states("req-B", 3, 1, boundary_hash=h(102))
    idx.confirm_writes(list(st.values()))

    hit_a = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit_a is not None and hit_a[0] == "req-A" and hit_a[1] == 3
    hit_b = idx.lookup([h(0), h(101), h(102), h(103)])
    assert hit_b is not None and hit_b[0] == "req-B" and hit_b[1] == 3


def test_tail_hash_mismatch_kills_resume():
    """A tail state paired with another chain's attention blocks must never
    match: resuming with the wrong mamba state corrupts output silently."""
    idx = HostKVTierIndex(num_slots=16)
    slots = [idx.stage_attention("x", 0, i, h(i)) for i in range(3)]
    idx.confirm_writes(slots)
    # Tail recorded by a diverged continuation: its chain hash at the
    # boundary block differs from what is actually staged there.
    st = idx.stage_tail_states("x", 3, 1, boundary_hash=h(999))
    idx.confirm_writes(list(st.values()))
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None
    # A consistent tail from the true continuation restores resumability.
    st = idx.stage_tail_states("x", 3, 1, boundary_hash=h(2))
    idx.confirm_writes(list(st.values()))
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is not None


def test_missing_boundary_hash_is_unresumable():
    """Tails saved without a chain hash (or before their attention block is
    staged) never match - resume correctness beats coverage."""
    idx = HostKVTierIndex(num_slots=16)
    slots = [idx.stage_attention("x", 0, i, h(i)) for i in range(3)]
    idx.confirm_writes(slots)
    st = idx.stage_tail_states("x", 3, 1)  # default empty boundary_hash
    idx.confirm_writes(list(st.values()))
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
        idx.confirm_writes([idx.stage_attention(owner, 0, i, h(i))])
    st = idx.stage_tail_states(owner, 5, 2, boundary_hash=h(4))
    idx.confirm_writes(list(st.values()))
    assert idx.stats()["trajectories"] == 1
    hit = idx.lookup([h(i) for i in range(6)])
    assert hit is not None and hit[0] == "turn-1" and hit[1] == 5


def test_stateless_tail_makes_attention_only_trajectory_resumable():
    """Attention-only models have no mamba boundary snapshot; without
    stage_attention_tail their trajectories were write-only (staged,
    never matched by lookup)."""
    idx = HostKVTierIndex(num_slots=16)
    slots = build_traj(idx, "a", 3, tail=False)
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None  # the old hole
    assert idx.stage_attention_tail("a") == 3
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None
    owner, n, attn, states = hit
    assert owner == "a" and n == 3 and attn[0][:n] == slots and states == {}


def test_stateless_tail_stops_at_first_gap_and_grows():
    idx = HostKVTierIndex(num_slots=16)
    s0 = idx.stage_attention("a", 0, 0, h(0))
    s2 = idx.stage_attention("a", 0, 2, h(2))  # gap at 1
    idx.confirm_writes([s0, s2])
    assert idx.stage_attention_tail("a") == 1  # span ends at the gap
    s1 = idx.stage_attention("a", 0, 1, h(1))
    idx.confirm_writes([s1])
    assert idx.stage_attention_tail("a") == 3  # regrown span supersedes
    hit = idx.lookup([h(0), h(1), h(2)])
    assert hit is not None and hit[1] == 3


def test_stateless_tail_on_unknown_owner_is_zero():
    idx = HostKVTierIndex(num_slots=16)
    assert idx.stage_attention_tail("ghost") == 0


def test_stateless_tail_supersedes_stale_state_slots():
    """A hybrid trajectory later finished through the stateless path (or a
    reused owner) must free its old tail-state slots."""
    idx = HostKVTierIndex(num_slots=8)
    build_traj(idx, "a", 2, tail=True)  # 2 attn + 2 state slots
    before = len(idx._free)
    assert idx.stage_attention_tail("a") == 2
    assert len(idx._free) == before + 2  # state slots returned
    hit = idx.lookup([h(0), h(1)])
    assert hit is not None and hit[3] == {}


def test_stateless_conversion_preserves_state_slots_during_pending_read():
    idx = HostKVTierIndex(num_slots=6)
    build_traj(idx, "a", 4)
    original = idx.lookup([h(i) for i in range(4)])
    idx.pin_read("a")
    assert idx.stage_attention_tail("a") == 0
    assert idx.lookup([h(i) for i in range(4)]) == original
    assert idx.stage_attention("b", 0, 0, h(100)) is None
    idx.unpin_read("a")
    assert idx.stage_attention_tail("a") == 4
    assert idx.stage_attention("b", 0, 0, h(100)) is not None


def test_stateless_lookup_matches_partial_prefix():
    """A stateless tail routinely crosses into the request's own generated
    thinking tokens, which no follow-up prompt resends. Lookup must match
    the deepest agreeing prefix, not require the exact tail."""
    idx = HostKVTierIndex(num_slots=16)
    slots = build_traj(idx, "a", 4, tail=False)
    assert idx.stage_attention_tail("a") == 4
    # Follow-up agrees for 3 blocks, then carries its own turn.
    hit = idx.lookup([h(0), h(1), h(2), h(999)])
    assert hit is not None
    owner, n, attn, states = hit
    assert owner == "a" and n == 3 and attn[0][:n] == slots[:3]
    # Shorter prompt than the tail also matches what it has.
    hit = idx.lookup([h(0), h(1), h(2)])
    assert hit is not None and hit[1] == 3


def test_stateful_lookup_stays_tail_exact():
    """Mamba state exists only at the boundary: partial prefixes must NOT
    match a stateful trajectory."""
    idx = HostKVTierIndex(num_slots=16)
    build_traj(idx, "a", 3, tail=True)  # 2 state groups at boundary 3
    assert idx.lookup([h(0), h(1), h(999)]) is None
    assert idx.lookup([h(0), h(1)]) is None
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is not None
