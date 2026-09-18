"""Content-addressed semantic KDA checkpoints in the host tier."""

from vllm.v1.core.kv_tier_index import HostKVTierIndex


def h(value: int) -> bytes:
    return value.to_bytes(8, "little")


def stage_attention(
    index: HostKVTierIndex, owner: str, hashes: list[bytes]
) -> None:
    slots = [
        index.stage_attention(owner, logical, block_hash)
        for logical, block_hash in enumerate(hashes)
    ]
    assert all(slot is not None for slot in slots)
    index.confirm_writes([slot for slot in slots if slot is not None])


def stage_semantic(
    index: HostKVTierIndex, owner: str, boundary: int, hashes: list[bytes]
) -> None:
    slots = index.stage_semantic_states(
        owner,
        boundary,
        num_state_groups=1,
        boundary_hash=hashes[boundary - 1],
    )
    assert slots is not None
    index.confirm_writes(list(slots.values()))


def test_same_tokens_reuse_checkpoint_across_request_ids():
    hashes = [h(i) for i in range(5)]
    index = HostKVTierIndex(16)
    stage_attention(index, "request-a", hashes[:4])
    stage_semantic(index, "request-a", 3, hashes)

    # Lookup carries no request id: the full chained token hashes are the key.
    hit = index.lookup(hashes)
    assert hit is not None
    owner, boundary, attention, states = hit
    assert owner == "request-a"
    assert boundary == 3
    assert len(attention) == 3
    assert set(states) == {0}


def test_shared_first_block_does_not_collide_after_divergence():
    saved = [h(0), h(1), h(2)]
    index = HostKVTierIndex(16)
    stage_attention(index, "saved", saved)
    stage_semantic(index, "saved", 3, saved)

    assert index.lookup([h(0), h(99), h(2), h(3)]) is None


def test_deepest_matching_semantic_checkpoint_wins():
    hashes = [h(i) for i in range(6)]
    index = HostKVTierIndex(24)
    stage_attention(index, "conversation", hashes[:5])
    stage_semantic(index, "conversation", 2, hashes)
    stage_semantic(index, "conversation", 4, hashes)

    hit = index.lookup(hashes)
    assert hit is not None and hit[1] == 4

    # A branch after block 2 cannot pair the block-4 KDA state with another
    # attention history; it safely falls back to the shallower checkpoint.
    branch = [h(0), h(1), h(90), h(91), h(92)]
    hit = index.lookup(branch)
    assert hit is not None and hit[1] == 2


def test_pending_semantic_state_is_not_visible():
    hashes = [h(i) for i in range(3)]
    index = HostKVTierIndex(8)
    stage_attention(index, "pending", hashes)
    slots = index.stage_semantic_states("pending", 2, 1, hashes[1])
    assert slots is not None
    assert index.lookup(hashes) is None
    index.confirm_writes(list(slots.values()))
    assert index.lookup(hashes) is not None


def test_checkpoint_limit_retains_deepest_boundaries():
    hashes = [h(i) for i in range(6)]
    index = HostKVTierIndex(24, semantic_checkpoint_limit=2)
    stage_attention(index, "bounded", hashes[:5])
    stage_semantic(index, "bounded", 1, hashes)
    stage_semantic(index, "bounded", 2, hashes)
    stage_semantic(index, "bounded", 4, hashes)

    trajectory = index._trajectories["bounded"]
    assert set(trajectory.semantic_states) == {2, 4}


def test_semantic_checkpoint_demotes_to_nvme_and_promotes_exact_boundary():
    hashes = [h(i) for i in range(5)]
    index = HostKVTierIndex(5, num_disk_slots=24)
    attention_slots = [
        index.stage_attention("saved", logical, block_hash)
        for logical, block_hash in enumerate(hashes[:3])
    ]
    shallow = index.stage_semantic_states("saved", 2, 1, hashes[1])
    deep = index.stage_semantic_states("saved", 3, 1, hashes[2])
    assert None not in attention_slots and shallow is not None and deep is not None
    host_slots = [
        *attention_slots,
        *shallow.values(),
        *deep.values(),
    ]
    index.confirm_writes(host_slots)
    writes = index.take_disk_writes(host_slots)
    assert len(writes) == len(host_slots)
    index.confirm_disk_writes(writes)

    # Force the saved trajectory out of pinned RAM, then release the pressure
    # trajectory so the exact selected checkpoint can be promoted again.
    busy = index.stage_attention("pressure", 0, h(100))
    assert busy is not None
    index.confirm_writes([busy])
    pressure_writes = index.take_disk_writes([busy])
    index.confirm_disk_writes(pressure_writes)
    saved = index._trajectories["saved"]
    assert not saved.host_slots()
    assert all(cp.disk_state_slots for cp in saved.semantic_states.values())
    index.drop_owner("pressure")

    # Divergence after block 2 selects the shallower snapshot. Promotion must
    # not accidentally hydrate the deeper or terminal recurrent state.
    branch = [hashes[0], hashes[1], h(90), h(91)]
    hit = index.lookup(branch)
    assert hit is not None and hit[1] == 2 and hit[3] == {}
    assert index.needs_promotion(hit)
    promoted = index.promote(hit[0], hit[1])
    assert promoted is not None
    attention, states, disk_reads = promoted
    assert len(attention) == 2
    assert set(states) == {0}
    assert len(disk_reads) == 3  # two attention pages plus boundary-2 KDA
    index.confirm_promotion(hit[0])
    ready = index.lookup(branch)
    assert ready is not None and ready[1] == 2
    assert not index.needs_promotion(ready)
