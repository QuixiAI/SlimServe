# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HostTierConnector NVMe tier (scheduler side): write-through staging,
rank-counted completion, demotion under host pressure, promotion on hit."""

from types import SimpleNamespace

from tests.v1.core.test_host_tier_connector import (
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
