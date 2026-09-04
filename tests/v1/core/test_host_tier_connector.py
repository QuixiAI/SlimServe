# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HostTierConnector scheduler-side contract on a fake engine.

Positions are scheduler blocks (LCM of group block sizes); ops are
(slot, col, group, gpu_block). Three model shapes are exercised:
Qwen-style hybrid (attention + ring + mamba tail), attention-only
(GLM-5.2), and DeepSeek-V4-style multi-rate (MLA + a finer-grained indexer
+ a sliding-window compressor whose window is the tail).
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

BLOCK = 16


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


def _attn(block_size, names):
    return SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
        ),
        layer_names=list(names),
    )


def make_groups():
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
    return [_attn(BLOCK, ["attn"]), ring, *mamba]


class FakePool:
    def __init__(self):
        self.blocks = {i: SimpleNamespace(block_id=i, ref_cnt=1) for i in range(2000)}
        self.touched, self.freed = [], []
        # (block_hash, group_id) -> block, mirroring the engine's prefix
        # cache of frozen boundary blocks.
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


def build(groups, role=KVConnectorRole.SCHEDULER, tensors=(), gb=1.0):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8),  # deliberately stale
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"host_tier_gb_per_rank": gb}
        ),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=groups, kv_cache_tensors=list(tensors)
    )

    def _base_init(self, vc, role, kcc):
        self._vllm_config = vc
        self._kv_transfer_config = vc.kv_transfer_config
        self._kv_cache_config = kcc
        self._role = role
        self._connector_metadata = None

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1.__init__",
        _base_init,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector import (  # noqa: E501
            HostTierConnector,
        )

        conn = HostTierConnector(vllm_config, role, kv_cache_config)
    if role == KVConnectorRole.SCHEDULER:
        conn.bind_gpu_block_pool(FakePool())
    return conn


def make_connector():
    conn = build(make_groups())
    assert conn.pos_tokens == BLOCK and conn.hash_block_size == BLOCK
    assert conn.attn_groups == [0]
    assert conn.state_groups == [2, 3]  # mamba only
    assert conn.ring_groups == [1]
    assert conn.tail_groups == [2, 3] and conn.requires_tail
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


def targets(ops):
    return {op[3] for op in ops}


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
    assert all(op[1] == 0 and op[2] == 0 for op in fill_ops)
    tail_ops = [op for ops in tail_meta.offloads.values() for op in ops]
    assert len(tail_ops) == 2  # one frozen snapshot per mamba group, no ring
    assert targets(tail_ops) == {95, 96}  # the pool-cached snapshots
    assert {op[2] for op in tail_ops} == {2, 3}
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
    ok, _ = conn.request_finished_all_groups(req, tuple([] for _ in range(4)))
    assert ok is False
    tail_meta = conn.build_connector_meta(sched_output({}))
    assert not tail_meta.offloads
    assert not conn._block_pool.touched
    assert conn.index.stats()["resumable"] == 0


def test_save_scans_down_to_deepest_cached_boundary():
    conn = make_connector()
    pool = conn._block_pool
    req = FakeRequest("rc", [h(i) for i in range(6)], num_tokens=6 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc(6, planned=0), 0)
    conn.build_connector_meta(sched_output({"rc": 6 * BLOCK}))
    req.num_computed_tokens = 6 * BLOCK
    conn.build_connector_meta(sched_output({"rc": 1}))
    conn.build_connector_meta(sched_output({}))
    for g, gid in enumerate((2, 3)):
        pool.cached[(bytes(h(3)), gid)] = pool.blocks[70 + g]
    conn.request_finished_all_groups(req, tuple([] for _ in range(4)))
    tail_meta = conn.build_connector_meta(sched_output({}))
    assert targets(op for ops in tail_meta.offloads.values() for op in ops) == {70, 71}
    conn.build_connector_meta(sched_output({}))

    fresh = FakeRequest("rd", [h(i) for i in range(6)], num_tokens=6 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    assert is_async and n_ext == 4 * BLOCK


def test_resume_round_trip():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    assert is_async and n_ext == 4 * BLOCK
    conn.update_state_after_alloc(fresh, alloc(5, planned=4, base=200), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["r2"]
    assert len(ops) == 4 + 2  # attention span + two mamba tail states
    assert {200, 201, 202, 203} <= targets(ops)
    assert {295, 296} <= targets(ops)  # position-(k-1) tail blocks
    assert meta.zeros["r2"] == [(1, 290)]  # the ring block


def test_progressive_clipped_restore():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, _ = conn.get_num_new_matched_tokens(fresh, 0)
    assert n_ext == 4 * BLOCK
    conn.update_state_after_alloc(fresh, alloc(2, planned=4, base=300), 2 * BLOCK)
    meta = conn.build_connector_meta(sched_output({}))
    assert len(meta.restores["r2"]) == 2  # attention only, no tail yet
    assert "r2" not in meta.zeros
    fresh.num_computed_tokens = 2 * BLOCK
    n_ext2, is_async2 = conn.get_num_new_matched_tokens(fresh, 2 * BLOCK)
    assert is_async2 and n_ext2 == 2 * BLOCK
    conn.update_state_after_alloc(fresh, alloc(4, planned=4, base=300), 2 * BLOCK)
    meta2 = conn.build_connector_meta(sched_output({}))
    assert len(meta2.restores["r2"]) == 2 + 2  # rest of span + tail
    assert meta2.zeros["r2"] == [(1, 390)]


def test_mixed_local_and_tier_resume():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    # Two blocks already local: the tier reports only the remainder.
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 2 * BLOCK)
    assert is_async and n_ext == 2 * BLOCK
    conn.update_state_after_alloc(fresh, alloc(4, planned=4, base=400), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    assert targets(meta.restores["r2"]) == {402, 403, 495, 496}


def test_short_prompt_or_mismatch_misses():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    short = FakeRequest("s", [h(i) for i in range(4)], num_tokens=4 * BLOCK)
    conn.on_new_request(short)
    assert conn.get_num_new_matched_tokens(short, 0) == (0, False)
    other = FakeRequest("o", [h(0), h(1), h(99), h(3)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(other)
    assert conn.get_num_new_matched_tokens(other, 0) == (0, False)


# ---------------------------------------------------- attention-only models
#
# GLM-5.2 shape: MLA main KV plus its indexer in ONE uniform attention group,
# no mamba and no ring. Gating resumability on a mamba tail made the tier
# write-only on exactly these models (issue #17, MI300X).


def make_attn_groups():
    return [_attn(BLOCK, ["mla", "indexer"])]


def make_attn_connector(groups=None):
    return build(groups if groups is not None else make_attn_groups())


def alloc_attn(n_attn, base=0):
    return FakeKVCacheBlocks(blocks=([FakeBlock(base + i) for i in range(n_attn)],))


def run_attn_conversation(conn, req_id, n_blocks, base=0):
    req = FakeRequest(
        req_id, [h(i) for i in range(n_blocks)], num_tokens=n_blocks * BLOCK + 4
    )
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc_attn(n_blocks, base), 0)
    conn.build_connector_meta(sched_output({req_id: n_blocks * BLOCK}))
    req.num_computed_tokens = n_blocks * BLOCK
    meta = conn.build_connector_meta(sched_output({req_id: 1}))
    conn.build_connector_meta(sched_output({}))  # confirm writes
    ok, _ = conn.request_finished_all_groups(req, ([],))
    assert ok is False
    conn.build_connector_meta(sched_output({}))
    return req, meta


def test_attention_only_connector_has_no_tail():
    conn = make_attn_connector()
    assert conn.attn_groups == [0]
    assert conn.state_groups == [] and conn.ring_groups == []
    assert conn.window_groups == [] and not conn.requires_tail
    assert conn.index.requires_tail is False


def test_attention_only_resumes_after_finish():
    """The issue-#17 regression: without this the tier never restores."""
    conn = make_attn_connector()
    _, meta = run_attn_conversation(conn, "r1", 4)
    assert len([op for ops in meta.offloads.values() for op in ops]) == 4
    assert conn.index.stats()["resumable"] == 1
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    assert is_async and n_ext == 4 * BLOCK


def test_attention_only_restore_has_no_tail_ops_or_zeros():
    conn = make_attn_connector()
    run_attn_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, _ = conn.get_num_new_matched_tokens(fresh, 0)
    conn.update_state_after_alloc(fresh, alloc_attn(5, base=200), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["r2"]
    assert len(ops) == 4
    assert targets(ops) == {200, 201, 202, 203}
    assert "r2" not in meta.zeros


def test_attention_only_unfinished_request_is_not_resumable():
    conn = make_attn_connector()
    req = FakeRequest("r1", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc_attn(4), 0)
    conn.build_connector_meta(sched_output({"r1": 4 * BLOCK}))
    req.num_computed_tokens = 4 * BLOCK
    conn.build_connector_meta(sched_output({"r1": 1}))
    conn.build_connector_meta(sched_output({}))
    assert conn.index.stats()["resumable"] == 0
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    assert conn.get_num_new_matched_tokens(fresh, 0) == (0, False)


def test_attention_only_hash_mismatch_misses():
    conn = make_attn_connector()
    run_attn_conversation(conn, "r1", 4)
    other = FakeRequest("r2", [h(0), h(1), h(99), h(3)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(other)
    assert conn.get_num_new_matched_tokens(other, 0) == (0, False)


# ----------------------------------------------- multi-rate (DeepSeek-V4)
#
# MLA at the scheduler block, an indexer paging 4x finer, and a sliding-
# window compressor (block 2, window 4) whose out-of-window blocks the
# engine nulls: the window is the tail. Hashes are at the GCD (2 tokens),
# so a 16-token position closes 8 hashes.


W_BLOCK, W_WIN = 2, 4
IDX_BLOCK = 4


def make_multirate_groups():
    window = SimpleNamespace(
        kv_cache_spec=SlidingWindowSpec(
            block_size=W_BLOCK,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            sliding_window=W_WIN,
        ),
        layer_names=["c4"],
    )
    return [_attn(BLOCK, ["mla"]), _attn(IDX_BLOCK, ["idx"]), window]


def make_multirate_connector():
    conn = build(make_multirate_groups())
    assert conn.pos_tokens == BLOCK and conn.hash_block_size == W_BLOCK
    assert conn.hashes_per_pos == BLOCK // W_BLOCK
    assert conn.attn_groups == [0, 1] and conn.window_groups == [2]
    assert conn.tail_groups == [2] and conn.requires_tail
    assert [g for g, _, _ in conn.sub_layout] == [0, 1, 1, 1, 1]
    return conn


HPP = BLOCK // W_BLOCK  # hashes per position


def mr_hashes(n_pos):
    return [h(i) for i in range(n_pos * HPP)]


def window_range(tokens):
    skipped = max(0, tokens - W_WIN + 1) // W_BLOCK
    return skipped, tokens // W_BLOCK


def alloc_mr(n_pos, base=0, resume_tokens=None):
    """MLA positional; indexer 4 per position; window: all real while
    filling, or the engine's [null] * skipped + [real] * window shape when
    admitted on an external hit of `resume_tokens`."""
    mla = [FakeBlock(base + i) for i in range(n_pos)]
    idx = [FakeBlock(base + 100 + i) for i in range(n_pos * (BLOCK // IDX_BLOCK))]
    n_w = n_pos * (BLOCK // W_BLOCK)
    if resume_tokens is None:
        win = [FakeBlock(base + 500 + i) for i in range(n_w)]
    else:
        first, last = window_range(resume_tokens)
        win = [FakeBlock(0, is_null=True)] * first + [
            FakeBlock(base + 500 + i) for i in range(first, last)
        ]
    return FakeKVCacheBlocks(blocks=(mla, idx, win))


def run_mr_conversation(conn, req_id, n_pos, base=0):
    req = FakeRequest(req_id, mr_hashes(n_pos), num_tokens=n_pos * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc_mr(n_pos, base), 0)
    conn.build_connector_meta(sched_output({req_id: n_pos * BLOCK}))
    req.num_computed_tokens = n_pos * BLOCK
    meta = conn.build_connector_meta(sched_output({req_id: 1}))
    conn.build_connector_meta(sched_output({}))
    # The engine froze the window blocks at the boundary into its prefix
    # cache (hash of window block m = the request hash closing it).
    pool = conn._block_pool
    first, last = window_range(n_pos * BLOCK)
    for m in range(first, last):
        pool.cached[(bytes(h(m)), 2)] = pool.blocks[base + 500 + m]
    ok, _ = conn.request_finished_all_groups(req, ([], [], []))
    assert ok is False
    tail_meta = conn.build_connector_meta(sched_output({}))
    conn.build_connector_meta(sched_output({}))
    return req, meta, tail_meta


def test_multirate_fill_stages_every_subblock_with_columns():
    conn = make_multirate_connector()
    _, meta, tail_meta = run_mr_conversation(conn, "r1", 3)
    ops = [op for ops in meta.offloads.values() for op in ops]
    assert len(ops) == 3 * 5  # per position: 1 MLA + 4 indexer sub-blocks
    mla_bytes = conn.group_bytes[0]
    idx_bytes = conn.group_bytes[1]
    cols = sorted({op[1] for op in ops})
    assert cols == [0] + [mla_bytes + k * idx_bytes for k in range(4)]
    assert conn.slot_bytes == mla_bytes + 4 * idx_bytes
    # One slot row per position: 5 sub-blocks share a slot.
    per_slot: dict[int, int] = {}
    for slot, _, _, _ in ops:
        per_slot[slot] = per_slot.get(slot, 0) + 1
    assert set(per_slot.values()) == {5}
    # Tail = the two in-window compressor blocks at the boundary (48 tok).
    tail_ops = [op for ops in tail_meta.offloads.values() for op in ops]
    first, last = window_range(3 * BLOCK)
    assert last - first == 2
    assert [op[2] for op in tail_ops] == [2, 2]
    assert targets(tail_ops) == {500 + first, 500 + last - 1}
    assert all(op[1] == 0 for op in tail_ops)
    assert conn.index.stats()["resumable"] == 1


def test_multirate_resume_restores_subblocks_and_window():
    conn = make_multirate_connector()
    run_mr_conversation(conn, "r1", 3)
    fresh = FakeRequest("r2", mr_hashes(3), num_tokens=3 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    assert is_async and n_ext == 3 * BLOCK
    conn.update_state_after_alloc(
        fresh, alloc_mr(3, base=200, resume_tokens=3 * BLOCK), n_ext
    )
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["r2"]
    assert len(ops) == 3 * 5 + 2
    assert {200, 201, 202} <= targets(ops)  # MLA
    assert set(range(300, 312)) <= targets(ops)  # indexer 4 per position
    first, last = window_range(3 * BLOCK)
    assert {700 + first, 700 + last - 1} <= targets(ops)  # window blocks
    assert "r2" not in meta.zeros


def test_multirate_missing_window_block_skips_tail_save():
    conn = make_multirate_connector()
    req = FakeRequest("r1", mr_hashes(3), num_tokens=3 * BLOCK + 4)
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc_mr(3), 0)
    conn.build_connector_meta(sched_output({"r1": 3 * BLOCK}))
    req.num_computed_tokens = 3 * BLOCK
    conn.build_connector_meta(sched_output({"r1": 1}))
    conn.build_connector_meta(sched_output({}))
    # Only one of the two window blocks is still cached: no tail.
    first, _ = window_range(3 * BLOCK)
    conn._block_pool.cached[(bytes(h(first)), 2)] = conn._block_pool.blocks[9]
    conn.request_finished_all_groups(req, ([], [], []))
    assert not conn.build_connector_meta(sched_output({})).offloads
    assert conn.index.stats()["resumable"] == 0


# ---------------------------------------------------------------- pins


def test_restore_sources_are_pinned_until_issued():
    conn = make_attn_connector()
    run_attn_conversation(conn, "r1", 4)
    assert conn.index.stats()["pinned"] == 0
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, _ = conn.get_num_new_matched_tokens(fresh, 0)
    assert conn.index.stats()["pinned"] == 4
    conn.index._free.clear()
    assert conn.index.stage_attention("intruder", 0, h(77)) is None
    assert conn.index.lookup([h(i) for i in range(4)]) is not None
    conn.update_state_after_alloc(fresh, alloc_attn(5, base=200), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    assert len(meta.restores["r2"]) == 4
    assert conn.index.stats()["pinned"] == 4
    conn.build_connector_meta(sched_output({}))
    assert conn.index.stats()["pinned"] == 0


def test_abandoned_plan_releases_pins():
    conn = make_attn_connector()
    run_attn_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, _ = conn.get_num_new_matched_tokens(fresh, 0)
    assert conn.index.stats()["pinned"] == 4
    conn.update_state_after_alloc(fresh, alloc_attn(2, base=200), n_ext)
    assert conn.index.stats()["pinned"] == 0
    assert "r2" not in conn._staged_restores


def test_finish_mid_plan_releases_pins():
    conn = make_attn_connector()
    run_attn_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    conn.get_num_new_matched_tokens(fresh, 0)
    assert conn.index.stats()["pinned"] == 4
    conn.request_finished_all_groups(fresh, ([],))
    assert conn.index.stats()["pinned"] == 0


# ------------------------------------------------- worker registration


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _two_layer_group():
    g = _attn(BLOCK, ["l0", "l1"])
    return [g], g.kv_cache_spec


@cuda
def test_register_packed_slab_takes_one_strided_segment_per_group():
    groups, spec = _two_layer_group()
    page = spec.page_size_bytes
    stride, n_blocks = 2 * page, 4
    slab = torch.zeros(n_blocks * stride, dtype=torch.int8, device="cuda")
    rows = slab.view(n_blocks, stride)
    views = {"l0": rows[:, :page], "l1": rows[:, page:]}
    tensors = [
        SimpleNamespace(
            size=slab.numel(), shared_by=["l0"], offset=0, block_stride=stride
        ),
        SimpleNamespace(
            size=slab.numel(), shared_by=["l1"], offset=page, block_stride=stride
        ),
    ]
    conn = build(groups, KVConnectorRole.WORKER, tensors, gb=0.001)
    conn.register_kv_caches(views)
    assert list(conn._dma.segments) == [0]
    assert len(conn._dma.segments[0]) == 1
    assert conn._dma.group_bytes[0] == stride == conn.slot_bytes
    assert conn._dma.num_blocks == n_blocks


@cuda
def test_register_per_layer_tensors_takes_one_segment_per_layer():
    groups, spec = _two_layer_group()
    page = spec.page_size_bytes
    l0 = torch.zeros(4 * page, dtype=torch.int8, device="cuda")
    l1 = torch.zeros(4 * page, dtype=torch.int8, device="cuda")
    tensors = [
        SimpleNamespace(size=l0.numel(), shared_by=["l0"], offset=0, block_stride=0),
        SimpleNamespace(size=l1.numel(), shared_by=["l1"], offset=0, block_stride=0),
    ]
    conn = build(groups, KVConnectorRole.WORKER, tensors, gb=0.001)
    conn.register_kv_caches({"l0": l0, "l1": l1})
    assert [s.width for s in conn._dma.segments[0]] == [page, page]
    assert conn._dma.num_blocks == 4


@cuda
def test_register_refuses_an_unaccounted_layer():
    groups, spec = _two_layer_group()
    page = spec.page_size_bytes
    l0 = torch.zeros(4 * page, dtype=torch.int8, device="cuda")
    l1 = torch.zeros(4 * page, dtype=torch.int8, device="cuda")
    tensors = [
        SimpleNamespace(size=l0.numel(), shared_by=["l0"], offset=0, block_stride=0),
    ]
    conn = build(groups, KVConnectorRole.WORKER, tensors, gb=0.001)
    with pytest.raises(RuntimeError, match="no KV tensor in the planner layout"):
        conn.register_kv_caches({"l0": l0, "l1": l1})


@cuda
def test_register_refuses_group_pool_tensors():
    groups, spec = _two_layer_group()
    page = spec.page_size_bytes
    pool = torch.zeros(4 * page, dtype=torch.int8, device="cuda")
    tensors = [
        SimpleNamespace(
            size=pool.numel(), shared_by=["l0", "l1"], offset=0, block_stride=0
        ),
    ]
    conn = build(groups, KVConnectorRole.WORKER, tensors, gb=0.001)
    with pytest.raises(RuntimeError, match="not block-addressable"):
        conn.register_kv_caches({"l0": pool, "l1": pool})
