# SPDX-License-Identifier: Apache-2.0
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
        st = idx.stage_tail_states(owner, n_blocks, 2)
        assert st is not None
        idx.confirm_writes(list(st.values()))
    return slots


def test_resumable_only_with_confirmed_tail():
    idx = HostKVTierIndex(num_slots=16)
    slots = build_traj(idx, "a", 3, tail=False)
    assert idx.lookup([h(0), h(1), h(2), h(3)]) is None  # no tail state
    st = idx.stage_tail_states("a", 3, 2)
    assert idx.lookup([h(0), h(1), h(2)]) is None  # tail pending
    idx.confirm_writes(list(st.values()))
    hit = idx.lookup([h(0), h(1), h(2), h(3)])
    assert hit is not None
    n, attn, states = hit
    assert n == 3 and attn == slots and set(states) == {0, 1}


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
    st = idx.stage_tail_states("a", 5, 2)
    idx.confirm_writes(list(st.values()))
    hit = idx.lookup([h(i) for i in range(6)])
    assert hit is not None and hit[0] == 5
    # Old tail slots were released: net growth is 3 attn + 0 state slots.
    assert idx.stats()["used"] == used_after_first + 3


def test_attention_gap_blocks_resume():
    idx = HostKVTierIndex(num_slots=16)
    s0 = idx.stage_attention("a", 0, h(0))
    s2 = idx.stage_attention("a", 2, h(2))  # gap at 1
    idx.confirm_writes([s0, s2])
    st = idx.stage_tail_states("a", 3, 1)
    idx.confirm_writes(list(st.values()))
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
