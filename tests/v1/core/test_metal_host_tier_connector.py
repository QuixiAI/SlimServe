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


def _fake_base_init(self, vc, role, kcc):
    self._vllm_config = vc
    self._kv_transfer_config = vc.kv_transfer_config
    self._kv_cache_config = kcc
    self._role = role
    self._connector_metadata = None


def make_connector(groups=None, hash_unit=None):
    default_groups = groups is None
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8),  # deliberately stale
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"nvme_tier_gb_per_rank": 1.0}
        ),
    )
    # No packed KVCacheTensor records: the connector falls back to
    # _get_packed_kv_cache_layout (patched below), the pre-e6ef60edc path
    # this mock was built around.
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=make_groups() if default_groups else groups,
        kv_cache_tensors=[],
    )
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.metal_host_tier_connector."
            "_get_packed_kv_cache_layout",
            return_value=(STRIDE, {}),
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.metal_host_tier_connector."
            "resolve_kv_cache_block_sizes",
            return_value=(BLOCK, hash_unit if hash_unit else BLOCK),
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.base."
            "KVConnectorBase_V1.__init__",
            _fake_base_init,
        ),
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            metal_host_tier_connector as metal_connector,
        )

        HostTierConnector = metal_connector.HostTierConnector

        conn = HostTierConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    class FakePool:
        def __init__(self):
            self.blocks = {
                i: SimpleNamespace(block_id=i, ref_cnt=1) for i in range(1000)
            }
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
    if default_groups:
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
        gb.extend(FakeBlock(base + 97 + g + i) for i in range(max(0, n_attn - planned)))
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
    ok, _ = conn.request_finished_all_groups(req, tuple([] for _ in range(4)))
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
    assert {b for b, _ in tail_ops} == {95, 96}  # the pool-cached snapshots
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
    assert conn.index.stats()["resumable"] == 0


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
    assert {b for ops in tail_meta.offloads.values() for b, _ in ops} == {70, 71}
    conn.build_connector_meta(sched_output({}))

    fresh = FakeRequest("rd", [h(i) for i in range(6)], num_tokens=6 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    # Resumable at the scanned-down boundary, not the final block count.
    assert is_async and n_ext == 4 * BLOCK


def test_resume_round_trip():
    conn = make_connector()
    run_conversation(conn, "r1", 4)

    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
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
    targets = {b for _, b in ops}
    assert {200, 201, 202, 203} <= targets  # attention span
    # Both mamba states land on the position-(k-1) tail blocks.
    assert {295, 296} <= targets
    assert meta.zeros["r2"] == [290]  # the ring block


def test_pending_restore_protects_slots_from_capacity_eviction():
    from vllm.v1.core.metal_kv_tier_index import HostKVTierIndex

    conn = make_connector()
    conn.index = HostKVTierIndex(6)  # four attention pages and two tail states
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    assert is_async and n_ext == 4 * BLOCK
    conn.update_state_after_alloc(fresh, alloc(5, planned=4, base=200), n_ext)
    other = FakeRequest("r3", fresh.block_hashes, num_tokens=fresh.num_tokens)
    conn.on_new_request(other)
    n_ext3, _ = conn.get_num_new_matched_tokens(other, 0)
    conn.update_state_after_alloc(other, alloc(5, planned=4, base=300), n_ext3)
    # New offloads are issued before restores in the same worker batch.
    # Recycling a planned source here would overwrite it before its read.
    assert conn.index.stage_attention("contender", 0, 0, h(99)) is None
    meta = conn.build_connector_meta(sched_output({}))
    assert len(meta.restores["r2"]) == 6
    conn.update_connector_output(SimpleNamespace(finished_recving={"r2"}))
    assert conn.index.stage_attention("contender", 0, 0, h(99)) is None
    # Cancellation cannot recycle the second reader's issued source either.
    conn.request_finished_all_groups(other, tuple([] for _ in range(4)))
    assert conn.index.stage_attention("contender", 0, 0, h(99)) is None
    conn.update_connector_output(SimpleNamespace(finished_recving={"r3"}))
    assert conn.index.stage_attention("contender", 0, 0, h(99)) is not None


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


def test_missing_state_target_reports_failed_load_and_drops_trajectory():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    assert is_async and n_ext == 4 * BLOCK
    blocks = alloc(5, planned=4, base=200)
    groups = list(blocks.blocks)
    groups[2] = [FakeBlock(0, is_null=True) for _ in groups[2]]
    conn.update_state_after_alloc(fresh, FakeKVCacheBlocks(tuple(groups)), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    assert "r2" not in meta.restores
    assert {200, 201, 202, 203} <= set(meta.failed["r2"])
    next_req = FakeRequest("r3", fresh.block_hashes, num_tokens=fresh.num_tokens)
    assert conn.get_num_new_matched_tokens(next_req, 0)[0] == 0


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
    targets = {b for _, b in ops}
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


def make_attn_only_connector(hash_unit=None):
    """DSV4/muse shape: attention groups only, no recurrent state."""
    groups = [g for g in make_groups() if not hasattr(g.kv_cache_spec, "shapes")]
    conn = make_connector(groups=groups, hash_unit=hash_unit)
    assert conn.state_groups == []
    return conn


def test_attention_only_finish_records_a_resumable_tail():
    """Attention-only models never reach the mamba tail save; without the
    stateless tail their trajectories were staged but never resumable."""
    conn = make_attn_only_connector()
    n = 3
    req = FakeRequest("s1", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    conn.on_new_request(req)
    attn = [FakeBlock(i) for i in range(n)]
    ring = [FakeBlock(90)]
    conn.update_state_after_alloc(req, FakeKVCacheBlocks(blocks=(attn, ring)), 0)
    conn.build_connector_meta(sched_output({"s1": n * BLOCK}))
    req.num_computed_tokens = n * BLOCK
    meta = conn.build_connector_meta(sched_output({"s1": 1}))
    conn.build_connector_meta(sched_output({}))  # confirm writes
    fill_ops = [op for ops in meta.offloads.values() for op in ops]
    assert len(fill_ops) == n
    ok, _ = conn.request_finished_all_groups(req, ([], []))
    assert ok is False
    assert conn.index.stats()["resumable"] == 1
    # A new request with the same prefix resumes from the tier.
    req2 = FakeRequest("s2", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    got, is_async = conn.get_num_new_matched_tokens(req2, 0)
    assert is_async is True and got == n * BLOCK


def test_attention_only_non_hma_finish_records_tail_too():
    conn = make_attn_only_connector()
    req = FakeRequest("s3", [h(i) for i in range(2)], num_tokens=2 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(
        req,
        FakeKVCacheBlocks(blocks=([FakeBlock(0), FakeBlock(1)], [FakeBlock(90)])),
        0,
    )
    conn.build_connector_meta(sched_output({"s3": 2 * BLOCK}))
    req.num_computed_tokens = 2 * BLOCK
    conn.build_connector_meta(sched_output({"s3": 1}))
    conn.build_connector_meta(sched_output({}))
    conn.request_finished(req, [])
    assert conn.index.stats()["resumable"] == 1


def test_fine_grained_hashes_are_subsampled_at_block_ends():
    """With multiple prefix-cacheable groups the engine hashes at the GCD
    of group block sizes (DSV4/Metal: 4-token units under 256-token
    pages). The tier must store/compare the cumulative hash at each
    attention-block END; treating fine hashes as block hashes made every
    live lookup miss at index 1."""
    stride = 4
    conn = make_attn_only_connector(hash_unit=BLOCK // stride)
    assert conn.hash_stride == stride
    n = 3
    fine = [h(i) for i in range(n * stride)]  # 4 hash units per block
    req = FakeRequest("f1", fine, num_tokens=n * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(
        req,
        FakeKVCacheBlocks(blocks=([FakeBlock(i) for i in range(n)], [FakeBlock(90)])),
        0,
    )
    conn.build_connector_meta(sched_output({"f1": n * BLOCK}))
    req.num_computed_tokens = n * BLOCK
    meta = conn.build_connector_meta(sched_output({"f1": 1}))
    conn.build_connector_meta(sched_output({}))
    fill_ops = [op for ops in meta.offloads.values() for op in ops]
    assert len(fill_ops) == n  # one op per BLOCK, not per hash unit
    conn.request_finished_all_groups(req, ([], []))
    assert conn.index.stats()["resumable"] == 1
    # The stored chain must be the block-END hashes (indices 3, 7, 11).
    req2 = FakeRequest("f2", fine, num_tokens=n * BLOCK + 4)
    got, is_async = conn.get_num_new_matched_tokens(req2, 0)
    assert is_async is True and got == n * BLOCK
    # A request diverging INSIDE the last block (different end hash)
    # matches the agreeing prefix: stateless trajectories resume at the
    # deepest matching block, not tail-exactly (mamba-only constraint).
    diverged = fine[: n * stride - 1] + [h(999)]
    req3 = FakeRequest("f3", diverged, num_tokens=n * BLOCK + 4)
    got3, _ = conn.get_num_new_matched_tokens(req3, 0)
    assert got3 == (n - 1) * BLOCK


def test_state_groups_reject_fine_grained_hashing():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="stride"):
        make_connector(hash_unit=BLOCK // 4)  # default groups include mamba


def make_multi_group_connector():
    """DSV4/Metal shape: primary full-attention pages plus a finer-grained
    sliding-window-style attention group (stride 4), plus the ring."""
    base = make_groups()
    swa = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=BLOCK // 4, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        ),
        layer_names=["swa"],
    )
    groups = [base[0], swa, base[1]]  # attn, fine attn, ring
    conn = make_connector(groups=groups)
    assert conn.state_groups == [] and conn.attn_groups == [0, 1]
    assert conn.attn_pos_stride == [1, 4]
    return conn


def test_full_attention_is_primary_when_window_group_is_listed_first():
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    base = make_groups()
    window = SimpleNamespace(
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK // 4,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            sliding_window=BLOCK,
        ),
        layer_names=["window"],
    )
    conn = make_connector(groups=[window, base[0], base[1]])
    assert conn.attn_groups == [1, 0]
    assert conn.hash_block_size == BLOCK
    assert conn.attn_pos_stride == [1, 4]


def _run_multi_group_fill(conn, req_id, n, skip_prefix_pages):
    """Fill n primary blocks; the fine group has 4n pages with the first
    `skip_prefix_pages` window-skipped (null)."""
    req = FakeRequest(req_id, [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    conn.on_new_request(req)
    attn = [FakeBlock(i) for i in range(n)]
    fine = [
        FakeBlock(0, is_null=True) if p < skip_prefix_pages else FakeBlock(100 + p)
        for p in range(n * 4)
    ]
    ring = [FakeBlock(90)]
    conn.update_state_after_alloc(req, FakeKVCacheBlocks(blocks=(attn, fine, ring)), 0)
    conn.build_connector_meta(sched_output({req_id: n * BLOCK}))
    req.num_computed_tokens = n * BLOCK
    meta = conn.build_connector_meta(sched_output({req_id: 1}))
    conn.build_connector_meta(sched_output({}))
    conn.request_finished_all_groups(req, ([], [], []))
    return req, meta


def test_multi_group_offload_saves_every_groups_pages():
    conn = make_multi_group_connector()
    n, skip = 3, 5
    req, meta = _run_multi_group_fill(conn, "m1", n, skip)
    fill_ops = [op for ops in meta.offloads.values() for op in ops]
    # n primary blocks + (4n - skip) real fine pages.
    assert len(fill_ops) == n + (n * 4 - skip)
    assert conn.index.stats()["resumable"] == 1


def test_multi_group_restore_pairs_per_group_and_skips_nulls():
    conn = make_multi_group_connector()
    n, skip = 3, 5
    _run_multi_group_fill(conn, "m2", n, skip)
    req2 = FakeRequest("m2b", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    got, is_async = conn.get_num_new_matched_tokens(req2, 0)
    assert is_async is True and got == n * BLOCK
    attn = [FakeBlock(50 + i) for i in range(n)]
    fine = [
        FakeBlock(0, is_null=True) if p < skip else FakeBlock(200 + p)
        for p in range(n * 4)
    ]
    ring = [FakeBlock(91)]
    conn.update_state_after_alloc(
        req2, FakeKVCacheBlocks(blocks=(attn, fine, ring)), got
    )
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["m2b"]
    assert not meta.failed
    # Same shape as the save: every real target paired, nulls skipped.
    assert len(ops) == n + (n * 4 - skip)
    targets = sorted(b for _, b in ops)
    assert targets == sorted(
        [50 + i for i in range(n)] + [200 + p for p in range(skip, n * 4)]
    )


def test_multi_group_restore_fails_loudly_on_missing_page():
    """A real target block the tier never saved must fail the load through
    meta.failed (framework invalid-block recovery), never wedge."""
    conn = make_multi_group_connector()
    n, skip = 3, 5
    _run_multi_group_fill(conn, "m3", n, skip)
    req2 = FakeRequest("m3b", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    got, _ = conn.get_num_new_matched_tokens(req2, 0)
    assert got == n * BLOCK
    attn = [FakeBlock(50 + i) for i in range(n)]
    # The resume allocates a REAL page where the save had a window skip.
    fine = [
        FakeBlock(0, is_null=True) if p < skip - 1 else FakeBlock(200 + p)
        for p in range(n * 4)
    ]
    ring = [FakeBlock(91)]
    conn.update_state_after_alloc(
        req2, FakeKVCacheBlocks(blocks=(attn, fine, ring)), got
    )
    meta = conn.build_connector_meta(sched_output({}))
    assert "m3b" not in meta.restores
    assert "m3b" in meta.failed and len(meta.failed["m3b"]) > 0
    # The broken trajectory is dropped: no future request can match it.
    req3 = FakeRequest("m3c", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    got3, _ = conn.get_num_new_matched_tokens(req3, 0)
    assert got3 == 0


def test_partial_prefix_resume_saves_an_independent_branch():
    conn = make_multi_group_connector()
    _run_multi_group_fill(conn, "original", 3, 0)
    hashes = [h(0), h(1), h(99)]
    branch = FakeRequest("branch", hashes, num_tokens=3 * BLOCK + 4)
    conn.on_new_request(branch)
    got, _ = conn.get_num_new_matched_tokens(branch, 0)
    assert got == 2 * BLOCK
    assert conn._tracks["branch"].owner == "branch"
    groups = (
        [FakeBlock(50 + i) for i in range(3)],
        [FakeBlock(200 + i) for i in range(12)],
        [FakeBlock(91)],
    )
    conn.update_state_after_alloc(branch, FakeKVCacheBlocks(groups), got)
    conn.build_connector_meta(sched_output({}))
    conn.update_connector_output(SimpleNamespace(finished_recving={"branch"}))
    branch.num_computed_tokens = 3 * BLOCK
    conn.build_connector_meta(sched_output({"branch": 1}))
    conn.build_connector_meta(sched_output({}))
    conn.request_finished_all_groups(branch, tuple([] for _ in groups))
    hit = conn.index.lookup(hashes)
    assert hit is not None and hit[0] == "branch" and hit[1] == 3
    original = conn.index.lookup([h(0), h(1), h(2)])
    assert original is not None and original[0] == "original" and original[1] == 3


def test_slid_out_window_pages_release_their_slots():
    """As a group's window slides, pages that went null in the allocation
    must release their tier slots - without this a 23k-token request held
    ~11k slots (120x the primary cost) and the LRU reclaimed live
    trajectories."""
    conn = make_multi_group_connector()
    n = 3
    req = FakeRequest("w1", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    conn.on_new_request(req)
    attn = [FakeBlock(i) for i in range(n)]
    fine_all = [FakeBlock(100 + p) for p in range(n * 4)]
    ring = [FakeBlock(90)]
    conn.update_state_after_alloc(
        req, FakeKVCacheBlocks(blocks=(attn, fine_all, ring)), 0
    )
    conn.build_connector_meta(sched_output({"w1": n * BLOCK}))
    req.num_computed_tokens = n * BLOCK
    conn.build_connector_meta(sched_output({"w1": 1}))
    conn.build_connector_meta(sched_output({}))
    used_full = conn.index.stats()["used"]
    assert used_full == n + n * 4  # everything staged while in-window
    # The window slides: the first 7 fine pages go null in the allocation.
    slid = [
        FakeBlock(0, is_null=True) if p < 7 else FakeBlock(100 + p)
        for p in range(n * 4)
    ]
    conn.update_state_after_alloc(req, FakeKVCacheBlocks(blocks=(attn, slid, ring)), 0)
    conn.build_connector_meta(sched_output({"w1": 1}))
    assert conn.index.stats()["used"] == used_full - 7
    # Restore pairing after finish still matches the surviving shape.
    conn.request_finished_all_groups(req, ([], [], []))
    req2 = FakeRequest("w2", [h(i) for i in range(n)], num_tokens=n * BLOCK + 4)
    got, _ = conn.get_num_new_matched_tokens(req2, 0)
    assert got == n * BLOCK
    conn.update_state_after_alloc(
        req2,
        FakeKVCacheBlocks(
            blocks=(
                [FakeBlock(50 + i) for i in range(n)],
                [
                    FakeBlock(0, is_null=True) if p < 7 else FakeBlock(200 + p)
                    for p in range(n * 4)
                ],
                [FakeBlock(91)],
            )
        ),
        got,
    )
    meta = conn.build_connector_meta(sched_output({}))
    assert not meta.failed
    assert len(meta.restores["w2"]) == n + (n * 4 - 7)


def test_connector_shutdown_closes_nvme_backend():
    from unittest.mock import Mock

    conn = make_connector()
    conn._dma = Mock()
    conn.shutdown()
    conn._dma.shutdown.assert_called_once_with()
    conn._dma.flush.assert_not_called()
