# SPDX-License-Identifier: Apache-2.0
"""CPU scheduler gates for raw (ratio2) and compact (ratio8) indexer pages.

These prove scheduling/alignment only, not GPU page contents or real DMA.
"""

import pytest

from tests.v1.core.test_host_tier_connector import (
    BLOCK,
    FakeBlock,
    FakeKVCacheBlocks,
    FakeRequest,
    alloc,
    h,
    make_connector,
    sched_output,
)
from vllm.v1.core.kv_tier_index import HostKVTierIndex


def mixed_alloc(n_hash, planned, ratio, base=0):
    standard = alloc(n_hash, planned, base)
    indexer = [FakeBlock(base + 300 + i) for i in range((n_hash + ratio - 1) // ratio)]
    return FakeKVCacheBlocks(blocks=(*standard.blocks, indexer))


@pytest.mark.parametrize("ratio", [2, 8])
def test_partial_indexer_page_is_not_saved_or_restored(ratio):
    conn = make_connector(indexer_ratio=ratio)
    assert conn._attn_ratio == {0: 1, 4: ratio}
    assert conn._resume_align == ratio
    # Two complete wide pages and an incomplete third page.
    n = 2 * ratio + 1
    boundary = 2 * ratio
    req = FakeRequest("source", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, mixed_alloc(n, 0, ratio), 0)
    conn.build_connector_meta(sched_output({req.request_id: n * BLOCK}))
    req.num_computed_tokens = n * BLOCK
    fill = conn.build_connector_meta(sched_output({req.request_id: 1}))
    conn.build_connector_meta(sched_output({}))
    fill_ops = [op for ops in fill.offloads.values() for op in ops]
    assert len(fill_ops) == n + 2
    assert {b for b, _, gid in fill_ops if gid == 4} == {300, 301}
    for g, gid in enumerate((2, 3)):
        conn._block_pool.cached[(bytes(h(boundary - 1)), gid)] = (
            conn._block_pool.blocks[95 + g]
        )
    conn.request_finished_all_groups(req, tuple([] for _ in range(5)))
    conn.build_connector_meta(sched_output({}))
    conn.build_connector_meta(sched_output({}))

    fresh = FakeRequest("restored", [h(i) for i in range(n)], num_tokens=n * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, asynchronous = conn.get_num_new_matched_tokens(fresh, 0)
    assert asynchronous and n_ext == boundary * BLOCK
    conn.update_state_after_alloc(
        fresh, mixed_alloc(n + 1, boundary, ratio, 100), n_ext
    )
    restored = conn.build_connector_meta(sched_output({}))
    ops = restored.restores["restored"]
    assert len(ops) == boundary + 2 + 2  # MLA + indexer + two boundary states
    assert {b for _, b, gid in ops if gid == 4} == {400, 401}
    assert len([1 for _, _, gid in ops if gid == 0]) == boundary
    assert not any(gid == 1 for _, _, gid in ops)  # circular buffer resets instead


@pytest.mark.parametrize("ratio", [2, 8])
def test_disk_promotion_preserves_sparse_group_positions(ratio):
    n = 2 * ratio
    capacity = n + 2 + 2  # narrow pages, wide pages, two boundary states
    index = HostKVTierIndex(
        capacity,
        attn_gids=[0, 4],
        attn_ratio={0: 1, 4: ratio},
        num_disk_slots=4 * capacity,
    )
    host_slots = []
    for pos in range(n):
        for gid in sorted(index.due(pos)):
            slot = index.stage_attention("source", pos, h(pos), gid=gid)
            assert slot is not None
            host_slots.append(slot)
    tails = index.stage_tail_states("source", n, 2, boundary_hash=h(n - 1))
    assert tails is not None
    host_slots.extend(tails.values())
    assert len(host_slots) == capacity
    index.confirm_writes(host_slots)
    writes = index.take_disk_writes(host_slots)
    assert len(writes) == capacity
    index.confirm_disk_writes(writes)
    # Force real index demotion through pressure, then make the competing
    # trajectory reclaimable so all original slots can be promoted again.
    extra = index.stage_attention("pressure", 0, h(1000), gid=0)
    assert extra is not None and index.stats()["disk_only"] == 1
    index.confirm_writes([extra])
    hit = index.lookup([h(i) for i in range(n + 1)])
    assert hit is not None and hit[1] == n and index.needs_promotion(hit)
    promoted = index.promote(hit[0], hit[1])
    assert promoted is not None
    attention, tail, reads = promoted
    assert len(reads) == capacity and set(tail) == {0, 1}
    for pos, mapping in enumerate(attention):
        assert set(mapping) == ({0, 4} if (pos + 1) % ratio == 0 else {0})
    expected_host = {slot for mapping in attention for slot in mapping.values()} | set(
        tail.values()
    )
    assert {slot for _, slot in reads} == expected_host
    assert index.lookup([h(i) for i in range(n + 1)]) is None
    index.confirm_promotion("source")
    ready = index.lookup([h(i) for i in range(n + 1)])
    assert ready is not None and not index.needs_promotion(ready)
