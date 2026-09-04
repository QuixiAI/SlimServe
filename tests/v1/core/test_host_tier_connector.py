# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HostTierConnector scheduler-side: fill staging, tail save, resume.

Save contract: the tail state saved for boundary B = k * block_size is the
engine's own frozen align-mode snapshot - the pool-cached mamba block keyed
by block_hashes[k - 1] - never a live block read positionally.

Restore contract: the state lands at position k - 1 of each mamba group
(the worker seeds state_idx = (num_computed - 1) // block_size), and the
ring block is zeroed, not restored (engine-internal hit semantics).
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
)

BLOCK = 16
STRIDE = 4096


def h(i: int) -> bytes:
    return i.to_bytes(8, "little") + b"\x00" * 24


@dataclass
class FakeBlock:
    block_id: int
    is_null: bool = False


@dataclass
class FakeKVCacheBlocks:
    blocks: tuple = ()


@dataclass
class FakeRequest:
    request_id: str
    block_hashes: list = field(default_factory=list)
    num_tokens: int = 0
    num_computed_tokens: int = 0


def make_groups():
    attn = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        ),
        layer_names=["attn"],
    )
    ring = SimpleNamespace(
        kv_cache_spec=CircularBufferSpec(
            block_size=8, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        ),
        layer_names=["ring"],
    )
    mamba = [
        SimpleNamespace(
            kv_cache_spec=MambaSpec(
                shapes=((2, 2),), dtypes=(torch.float32,), block_size=BLOCK
            ),
            layer_names=[f"m{i}"],
        )
        for i in range(2)
    ]
    return [attn, ring, *mamba]


def make_connector():
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8),  # deliberately stale
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"host_tier_gb_per_rank": 1.0}
        ),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=make_groups(), kv_cache_tensors=[])
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector."
        "_get_packed_kv_cache_layout",
        return_value=(STRIDE, {}),
    ), patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.base."
        "KVConnectorBase_V1.__init__",
        lambda self, vc, role, kcc: (
            setattr(self, "_vllm_config", vc),
            setattr(self, "_kv_transfer_config", vc.kv_transfer_config),
            setattr(self, "_kv_cache_config", kcc),
            setattr(self, "_role", role),
            setattr(self, "_connector_metadata", None),
        )[-1],
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector import (
            HostTierConnector,
        )

        conn = HostTierConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    class FakePool:
        def __init__(self):
            self.blocks = {i: SimpleNamespace(block_id=i, ref_cnt=1)
                           for i in range(1000)}
            self.touched, self.freed = [], []
            # (block_hash, group_id) -> block, mirroring the engine's
            # prefix cache of frozen align-mode boundary states.
            self.cached: dict[tuple, SimpleNamespace] = {}

        def get_cached_block(self, block_hash, kv_cache_group_ids):
            out = []
            for gid in kv_cache_group_ids:
                blk = self.cached.get((bytes(block_hash), gid))
                if blk is None:
                    return None
                out.append(blk)
            return out

        def touch(self, blocks):
            self.touched.extend(b.block_id for b in blocks)
            for b in blocks:
                b.ref_cnt += 1

        def free_blocks(self, blocks):
            self.freed.extend(b.block_id for b in blocks)
            for b in blocks:
                b.ref_cnt -= 1

    conn.bind_gpu_block_pool(FakePool())
    assert conn.hash_block_size == BLOCK  # from the attn spec, not config
    assert conn.attn_groups == [0]
    assert conn.state_groups == [2, 3]  # mamba only
    assert conn.ring_groups == [1]
    return conn


def sched_output(step_tokens, new_reqs=(), cached=None):
    if cached is None:
        cached = SimpleNamespace(req_ids=[], new_block_ids=[])
    return SimpleNamespace(
        num_scheduled_tokens=step_tokens,
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached,
    )


def alloc(n_attn, planned, base=0):
    """Allocation shape mirroring the engine at external-load admission:
    attention positional; ring exactly one block; mamba groups shaped
    [null] * (planned - 1) + [real tail] (+ live compute blocks after)."""
    attn = [FakeBlock(base + i) for i in range(n_attn)]
    ring = [FakeBlock(base + 90)]
    mamba = []
    for g in range(2):
        gb = [FakeBlock(0, is_null=True)] * max(0, planned - 1)
        gb.append(FakeBlock(base + 95 + g))
        gb.extend(
            FakeBlock(base + 97 + g + i) for i in range(max(0, n_attn - planned))
        )
        mamba.append(gb)
    return FakeKVCacheBlocks(blocks=(attn, ring, *mamba))


def run_conversation(conn, req_id, n_blocks, base=0):
    """Fill a request, freeze its boundary states in the fake pool's prefix
    cache (as the engine's align mode does), and finish it."""
    req = FakeRequest(
        req_id, [h(i) for i in range(n_blocks)], num_tokens=n_blocks * BLOCK + 4
    )
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc(n_blocks, planned=0, base=base), 0)
    conn.build_connector_meta(sched_output({req_id: n_blocks * BLOCK}))
    req.num_computed_tokens = n_blocks * BLOCK
    meta = conn.build_connector_meta(sched_output({req_id: 1}))
    conn.build_connector_meta(sched_output({}))  # confirm writes
    # The engine cached the frozen boundary snapshot for each mamba group
    # when the final boundary was crossed.
    pool = conn._block_pool
    for g, gid in enumerate((2, 3)):
        pool.cached[(bytes(h(n_blocks - 1)), gid)] = pool.blocks[base + 95 + g]
    ok, _ = conn.request_finished_all_groups(
        req, tuple([] for _ in range(4))
    )
    assert ok is False  # never hold blocks (HMA deferred-free corrupts)
    tail_meta = conn.build_connector_meta(sched_output({}))  # issues save
    conn.build_connector_meta(sched_output({}))  # confirms + releases pins
    return req, meta, tail_meta, ok


def test_fill_stages_attention_and_pinned_tail_at_finish():
    conn = make_connector()
    req, meta, tail_meta, async_save = run_conversation(conn, "r1", 3)
    fill_ops = [op for ops in meta.offloads.values() for op in ops]
    assert len(fill_ops) == 3  # attention only during fill
    tail_ops = [op for ops in tail_meta.offloads.values() for op in ops]
    assert len(tail_ops) == 2  # one frozen snapshot per mamba group, no ring
    assert {b for b, _, _ in tail_ops} == {95, 96}  # the pool-cached snapshots
    assert async_save is False
    pool = conn._block_pool
    assert sorted(pool.touched) == sorted(pool.freed)  # pins released
    assert all(pool.blocks[b].ref_cnt == 1 for b in pool.touched)
    assert conn.index.stats()["pending_writes"] == 0
    assert conn.index.stats()["resumable"] == 1


def test_missing_cached_boundary_skips_save():
    conn = make_connector()
    req = FakeRequest("rx", [h(i) for i in range(3)], num_tokens=3 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc(3, planned=0), 0)
    conn.build_connector_meta(sched_output({"rx": 3 * BLOCK}))
    req.num_computed_tokens = 3 * BLOCK
    conn.build_connector_meta(sched_output({"rx": 1}))
    conn.build_connector_meta(sched_output({}))
    # Boundary snapshot evicted before finish: no tail save, no pins.
    ok, _ = conn.request_finished_all_groups(req, tuple([] for _ in range(4)))
    assert ok is False
    tail_meta = conn.build_connector_meta(sched_output({}))
    assert not tail_meta.offloads
    assert not conn._block_pool.touched
    # No tail state was saved, so the hybrid trajectory is not resumable.
    assert conn.index.lookup(req.block_hashes) is None


def test_save_scans_down_to_deepest_cached_boundary():
    """Align mode freezes one state per chunk-end column, so the final
    400-block boundary is often uncached; the save must scan down to the
    deepest boundary every mamba group has and stage the tail there."""
    conn = make_connector()
    pool = conn._block_pool
    req = FakeRequest("rc", [h(i) for i in range(6)], num_tokens=6 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc(6, planned=0), 0)
    conn.build_connector_meta(sched_output({"rc": 6 * BLOCK}))
    req.num_computed_tokens = 6 * BLOCK
    conn.build_connector_meta(sched_output({"rc": 1}))
    conn.build_connector_meta(sched_output({}))
    # Chunk end fell at block 4: only boundary 4 (hash index 3) is cached.
    for g, gid in enumerate((2, 3)):
        pool.cached[(bytes(h(3)), gid)] = pool.blocks[70 + g]
    conn.request_finished_all_groups(req, tuple([] for _ in range(4)))
    tail_meta = conn.build_connector_meta(sched_output({}))
    assert {b for ops in tail_meta.offloads.values() for b, _, _ in ops} == {70, 71}
    conn.build_connector_meta(sched_output({}))

    fresh = FakeRequest("rd", [h(i) for i in range(6)], num_tokens=6 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    # Resumable at the scanned-down boundary, not the final block count.
    assert is_async and n_ext == 4 * BLOCK


def test_resume_round_trip():
    conn = make_connector()
    run_conversation(conn, "r1", 4)

    fresh = FakeRequest(
        "r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8
    )
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    # Resumable at the finished request's final boundary: all 4 blocks.
    assert is_async and n_ext == 4 * BLOCK

    conn.update_state_after_alloc(fresh, alloc(5, planned=4, base=200), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["r2"]
    # 4 attention restores + 2 mamba tail-state restores; the ring is
    # zeroed, not restored.
    assert len(ops) == 4 + 2
    targets = {b for _, b, _ in ops}
    assert {200, 201, 202, 203} <= targets  # attention span
    # Both mamba states land on the position-(k-1) tail blocks.
    assert {295, 296} <= targets
    assert meta.zeros["r2"] == [290]  # the ring block


def test_progressive_clipped_restore():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, _ = conn.get_num_new_matched_tokens(fresh, 0)
    assert n_ext == 4 * BLOCK
    # Scheduler clips to 2 blocks this step.
    conn.update_state_after_alloc(fresh, alloc(2, planned=4, base=300), 2 * BLOCK)
    meta = conn.build_connector_meta(sched_output({}))
    assert len(meta.restores["r2"]) == 2  # attention only, no tail yet
    assert "r2" not in meta.zeros
    # Load continues; scheduler re-queries with the new computed count.
    fresh.num_computed_tokens = 2 * BLOCK
    n_ext2, is_async2 = conn.get_num_new_matched_tokens(fresh, 2 * BLOCK)
    assert is_async2 and n_ext2 == 2 * BLOCK
    conn.update_state_after_alloc(fresh, alloc(4, planned=4, base=300), n_ext2)
    meta2 = conn.build_connector_meta(sched_output({}))
    assert len(meta2.restores["r2"]) == 2 + 2  # final chunk carries the tail
    assert meta2.zeros["r2"] == [390]


def test_mixed_local_and_tier_resume():
    """GPU cache holds most of the prefix; the tier supplies the tail."""
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r5", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    # Scheduler reports 2 blocks already computed locally.
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 2 * BLOCK)
    assert is_async and n_ext == 2 * BLOCK
    # num_computed_tokens stays 0 while waiting (framework behavior).
    conn.update_state_after_alloc(fresh, alloc(4, planned=4, base=400), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["r5"]
    # 2 attention blocks (positions 2, 3) + 2 mamba tail states.
    assert len(ops) == 2 + 2
    targets = {b for _, b, _ in ops}
    assert {402, 403} <= targets
    assert {495, 496} <= targets
    assert meta.zeros["r5"] == [490]


def test_short_prompt_or_mismatch_misses():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    # Prompt ends exactly at the tail boundary: nothing left to compute.
    exact = FakeRequest("r3", [h(i) for i in range(4)], num_tokens=4 * BLOCK)
    assert conn.get_num_new_matched_tokens(exact, 0) == (0, False)
    other = FakeRequest("r4", [h(50 + i) for i in range(6)], num_tokens=99)
    assert conn.get_num_new_matched_tokens(other, 0) == (0, False)
