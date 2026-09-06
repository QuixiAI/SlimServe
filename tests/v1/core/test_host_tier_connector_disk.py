# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HostTierConnector NVMe tier (scheduler side): write-through staging,
rank-counted completion, demotion under host pressure, promotion on hit."""

from types import SimpleNamespace

from tests.v1.core.test_host_tier_connector import (
    STRIDE,
    BLOCK,
    FakeRequest,
    alloc,
    h,
    make_connector,
    run_conversation,
    sched_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector import (
    HostTierStats,
)


def make_disk_connector(host_gb=1.0, nvme_gb=4.0, ranks=2):
    from unittest.mock import patch

    import vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector as m

    orig = m.HostTierConnector.__init__

    def init(self, vllm_config, role, kv_cache_config):
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "host_tier_gb_per_rank": host_gb,
            "nvme_tier_gb_per_rank": nvme_gb,
        }
        vllm_config.parallel_config = SimpleNamespace(world_size=ranks)
        orig(self, vllm_config, role, kv_cache_config)

    with patch.object(m.HostTierConnector, "__init__", init):
        conn = make_connector()
    assert conn.num_disk_slots > 0
    return conn


def outputs(finished_recving=None, disk_done=None):
    stats = HostTierStats(data={"disk_done": disk_done}) if disk_done else None
    return SimpleNamespace(
        finished_recving=set(finished_recving or ()), kv_connector_stats=stats
    )


def drain_writes(conn, ranks=2):
    """Emulate the worker: every staged write-through batch completes on
    every rank."""
    seqs = list(conn._disk_write_batches)
    for _ in range(ranks):
        conn.update_connector_output(outputs(disk_done={str(s): 1 for s in seqs}))
    return seqs


def test_confirmed_host_writes_are_written_through_and_rank_counted():
    conn = make_disk_connector(ranks=2)
    req, meta, tail_meta, _ = run_conversation(conn, "r1", 3)
    # The meta AFTER a confirmation carries the write-through of exactly
    # the confirmed host slots.
    fill_slots = sorted(s for ops in meta.offloads.values() for _, s, _ in ops)
    tail_slots = [s for ops in tail_meta.offloads.values() for _, s, _ in ops]
    for _ in range(2):
        conn.build_connector_meta(sched_output({}))
    # run_conversation already built the metas that carried the
    # write-through; the scheduler keeps every staged batch until all ranks
    # report it.
    writes = [op for ops in conn._disk_write_batches.values() for op in ops]
    assert sorted(s for s, _ in writes) == sorted(fill_slots + tail_slots)
    st = conn.index.stats()
    assert st["disk_pending"] == 5
    seqs = list(conn._disk_write_batches)
    # One rank reporting is not enough.
    conn.update_connector_output(outputs(disk_done={str(s): 1 for s in seqs}))
    assert conn.index.stats()["disk_pending"] == 5
    conn.update_connector_output(outputs(disk_done={str(s): 1 for s in seqs}))
    assert conn.index.stats()["disk_pending"] == 0
    assert not conn._disk_write_batches


def test_host_pressure_demotes_then_hit_promotes_from_disk():
    conn = make_disk_connector(ranks=1)
    req, meta, tail_meta, _ = run_conversation(conn, "r1", 3)
    for _ in range(2):
        conn.build_connector_meta(sched_output({}))
    drain_writes(conn, ranks=1)
    # Squeeze the host tier: leave exactly the 5 slots r1 holds.
    idx = conn.index
    used = idx.stats()["used"]
    idx._free = idx._free[:0]
    idx.num_slots = used
    # A new unrelated conversation needs host slots -> r1 is demoted.
    run_conversation(conn, "r2", 1, base=200)
    assert idx.stats()["disk_only"] == 1
    # Same conversation comes back: hit, promotion planned with disk reads.
    req3 = FakeRequest(
        "r3", [h(i) for i in range(3)] + [h(9)], num_tokens=4 * BLOCK + 4
    )
    conn.on_new_request(req3)
    # r2 is still holding slots and unfinished writes; free room for the
    # promotion by finishing its write-through.
    drain_writes(conn, ranks=1)
    n, is_async = conn.get_num_new_matched_tokens(req3, 0)
    assert n == 3 * BLOCK and is_async
    assert "r3" in conn._promoted
    conn.update_state_after_alloc(req3, alloc(4, planned=3, base=300), 3 * BLOCK)
    meta3 = conn.build_connector_meta(sched_output({}))
    assert len(meta3.restores["r3"]) == 5  # 3 attention + 2 tail states
    reads = meta3.disk_reads["r3"]
    assert len(reads) == 5
    assert {hs for _, hs in reads} == {hs for hs, _, _ in meta3.restores["r3"]}
    # Until the worker reports the request received, the promoted slots are
    # pending and the trajectory is not offered again.
    other = idx.lookup([h(0), h(1), h(2), h(9)])
    assert other is None or other[0] != "r1"
    conn.update_connector_output(outputs(finished_recving=["r3"]))
    assert "r3" not in conn._promoted
    hit = idx.lookup([h(0), h(1), h(2), h(9)])
    assert hit is not None and hit[0] == "r1" and not idx.needs_promotion(hit)


def test_disk_tier_off_keeps_metadata_empty():
    conn = make_connector()
    assert conn.num_disk_slots == 0
    req, meta, tail_meta, _ = run_conversation(conn, "r1", 2)
    m = conn.build_connector_meta(sched_output({}))
    assert not meta.disk_writes and not tail_meta.disk_writes and not m.disk_writes
    assert not m.disk_reads


def test_host_resident_main_kv_disables_restores_until_rebind():
    """Without milestone 4 the tier has no main-KV rows to restore: a hit
    must not be reported (the request re-prefills) while offloads continue."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector as m

    orig = m.HostTierConnector.__init__

    def init(self, vllm_config, role, kv_cache_config):
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "host_tier_gb_per_rank": 1.0,
            "main_kv_host_resident": True,
        }
        vllm_config.parallel_config = SimpleNamespace(world_size=1)
        orig(self, vllm_config, role, kv_cache_config)

    with patch.object(m.HostTierConnector, "__init__", init):
        conn = make_connector()
    assert conn._main_kv_host_resident and not conn._main_kv_tiered
    req, meta, tail_meta, _ = run_conversation(conn, "r1", 3)
    assert meta.offloads  # fill offloads still staged
    for _ in range(2):
        conn.build_connector_meta(sched_output({}))
    assert conn.index.lookup(req.block_hashes) is not None  # trajectory exists
    again = FakeRequest("r2", [h(i) for i in range(3)] + [h(9)], num_tokens=4 * BLOCK + 4)
    conn.on_new_request(again)
    assert conn.get_num_new_matched_tokens(again, 0) == (0, False)


# --- main-KV tier slots through the connector (milestone 4) -----------------


def make_main_tier_connector(main_gb=1.0):
    from unittest.mock import patch

    import vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector as m

    orig = m.HostTierConnector.__init__

    def init(self, vllm_config, role, kv_cache_config):
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "host_tier_gb_per_rank": 1.0,
            "main_kv_host_resident": True,
            "main_kv_tier_gb_per_rank": main_gb,
        }
        vllm_config.parallel_config = SimpleNamespace(world_size=1)
        # The attention layer's main KV is host-resident: 13 sub-rows of
        # STRIDE bytes per scheduler block.
        kv_cache_config.kv_cache_tensors = [
            SimpleNamespace(
                host_resident=True, block_stride=13 * STRIDE, shared_by=["attn"], gpu_rows=4, sub_blocks=13
            )
        ]
        orig(self, vllm_config, role, kv_cache_config)

    with patch.object(m.HostTierConnector, "__init__", init):
        conn = make_connector()
    assert conn._main_kv_tiered and conn._main_gid == 0
    return conn


def test_main_slots_are_reserved_at_alloc_flushed_at_fill_and_rebound_on_resume():
    conn = make_main_tier_connector()
    req = FakeRequest("r1", [h(i) for i in range(3)], num_tokens=3 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc(3, planned=0), 0)
    new_req = SimpleNamespace(req_id="r1", block_ids=alloc(3, planned=0).blocks)
    new_req.block_ids = tuple([b.block_id for b in g] for g in new_req.block_ids)
    meta0 = conn.build_connector_meta(sched_output({"r1": 3 * BLOCK}, new_reqs=[new_req]))
    # Homes for the three attention blocks arrived before any fill.
    assert set(meta0.main_homes) == {0, 1, 2}
    slots = dict(meta0.main_homes)
    req.num_computed_tokens = 3 * BLOCK
    meta1 = conn.build_connector_meta(sched_output({"r1": 1}))
    flushed = [op for ops in meta1.main_flush.values() for op in ops]
    assert sorted(flushed) == sorted(slots.items())
    assert conn.index.stats()["main_pending"] == 3
    conn.build_connector_meta(sched_output({}))  # confirms slab + main writes
    assert conn.index.stats()["main_pending"] == 0
    # Finish with the tail snapshot cached, as the engine does.
    pool = conn._block_pool
    for g, gid in enumerate((2, 3)):
        pool.cached[(bytes(h(2)), gid)] = pool.blocks[95 + g]
    conn.request_finished_all_groups(req, tuple([] for _ in range(4)))
    conn.build_connector_meta(sched_output({}))
    conn.build_connector_meta(sched_output({}))
    assert conn.index.stats()["main_used"] == 3  # nothing released: all filled
    # Resume: the hit rebinds the new block ids onto the same slots.
    again = FakeRequest("r2", [h(i) for i in range(3)] + [h(9)], num_tokens=4 * BLOCK + 4)
    conn.on_new_request(again)
    n, is_async = conn.get_num_new_matched_tokens(again, 0)
    assert n == 3 * BLOCK and is_async
    conn.update_state_after_alloc(again, alloc(4, planned=3, base=300), 3 * BLOCK)
    meta2 = conn.build_connector_meta(sched_output({}))
    rebinds = meta2.main_rebinds["r2"]
    assert [slot for _, slot in rebinds] == [slots[0], slots[1], slots[2]]
    assert [blk for blk, _ in rebinds] == [300, 301, 302]
    # Pinned while r2 runs: the trajectory cannot be reclaimed.
    assert conn.index._main_pinned and not conn.index._reclaim(protect="zzz")
    conn.request_finished_all_groups(again, tuple([] for _ in range(4)))
    assert not conn.index._main_pinned


def test_unfilled_reservation_is_released_with_the_block_at_finish():
    conn = make_main_tier_connector()
    req = FakeRequest("r1", [h(i) for i in range(3)], num_tokens=3 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc(3, planned=0), 0)
    new_req = SimpleNamespace(req_id="r1", block_ids=tuple([b.block_id for b in g] for g in alloc(3, planned=0).blocks))
    conn.build_connector_meta(sched_output({"r1": 2 * BLOCK}, new_reqs=[new_req]))
    req.num_computed_tokens = 2 * BLOCK  # third block never fills
    conn.build_connector_meta(sched_output({"r1": 1}))
    conn.build_connector_meta(sched_output({}))
    conn.request_finished_all_groups(req, tuple([] for _ in range(4)))
    meta = conn.build_connector_meta(sched_output({}))
    assert meta.main_release == [2]
    assert conn.index.stats()["main_used"] == 2
