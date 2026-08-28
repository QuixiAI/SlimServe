# SPDX-License-Identifier: Apache-2.0
"""HostTierConnector scheduler-side: fill staging, tail save, resume."""

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
    kv_cache_config = SimpleNamespace(kv_cache_groups=make_groups())
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
    assert conn.attn_groups == [0] and conn.state_groups == [1, 2, 3]
    return conn


def sched_output(step_tokens, new_reqs=(), cached=None):
    if cached is None:
        cached = SimpleNamespace(req_ids=[], new_block_ids=[])
    return SimpleNamespace(
        num_scheduled_tokens=step_tokens,
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached,
    )


def alloc(n_attn, base=0, state_last=0):
    """Allocation shape mirroring the engine: attention positional, ring 1,
    mamba groups position-indexed with nulls before the tail."""
    attn = [FakeBlock(base + i) for i in range(n_attn)]
    ring = [FakeBlock(base + 90)]
    # Align-mode keeps two resident state blocks: the frozen previous
    # boundary and the live newest one; earlier positions are nulls.
    mamba = [
        [FakeBlock(0, is_null=True)] * max(0, n_attn - 2)
        + [FakeBlock(base + 95 + g + state_last)][: min(1, max(0, n_attn - 1))]
        + [FakeBlock(base + 97 + g + state_last)]
        for g in range(2)
    ]
    return FakeKVCacheBlocks(blocks=(attn, ring, *mamba))


def finish_blocks(n_attn, base=0):
    # Mamba groups: {exact boundary snapshot at position n-2, live partial
    # at position n-1}; the connector must save the snapshot.
    return (
        [base + i for i in range(n_attn)],
        [base + 90],
        [-1] * (n_attn - 2) + [base + 95, base + 99],
        [-1] * (n_attn - 2) + [base + 96, base + 98],
    )


def run_conversation(conn, req_id, n_blocks, base=0):
    """Fill a request; tail states save continuously as boundaries advance."""
    req = FakeRequest(
        req_id, [h(i) for i in range(n_blocks)], num_tokens=n_blocks * BLOCK + 4
    )
    conn.on_new_request(req)
    conn.update_state_after_alloc(req, alloc(n_blocks, base=base), 0)
    conn.build_connector_meta(sched_output({req_id: n_blocks * BLOCK}))
    req.num_computed_tokens = n_blocks * BLOCK
    meta = conn.build_connector_meta(sched_output({req_id: 1}))
    conn.build_connector_meta(sched_output({}))  # confirm writes
    ok, _ = conn.request_finished_all_groups(req, finish_blocks(n_blocks, base))
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
    assert len(tail_ops) == 3  # ring + one block per mamba group
    assert async_save is False
    pool = conn._block_pool
    assert sorted(pool.touched) == sorted(pool.freed)  # pins released
    assert all(pool.blocks[b].ref_cnt == 1 for b in pool.touched)
    assert conn.index.stats()["pending_writes"] == 0
    assert conn.index.stats()["resumable"] == 1


def test_resume_round_trip():
    conn = make_connector()
    run_conversation(conn, "r1", 4)

    fresh = FakeRequest(
        "r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8
    )
    conn.on_new_request(fresh)
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 0)
    # Resumable at the exact snapshot boundary: 3 of 4 blocks.
    assert is_async and n_ext == 3 * BLOCK

    conn.update_state_after_alloc(fresh, alloc(5, base=200), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["r2"]
    # 3 attention restores + 3 tail-state restores (ring + mamba).
    assert len(ops) == 3 + 3
    targets = {b for _, b in ops}
    assert {200, 201, 202} <= targets  # attention span
    assert len([b for b in targets if b >= 290]) == 3


def test_progressive_clipped_restore():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r2", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    n_ext, _ = conn.get_num_new_matched_tokens(fresh, 0)
    assert n_ext == 3 * BLOCK
    # Scheduler clips to 2 blocks this step.
    conn.update_state_after_alloc(fresh, alloc(2, base=300), 2 * BLOCK)
    meta = conn.build_connector_meta(sched_output({}))
    assert len(meta.restores["r2"]) == 2  # attention only, no tail yet
    # Load completes; scheduler re-queries with the new computed count.
    fresh.num_computed_tokens = 2 * BLOCK
    n_ext2, is_async2 = conn.get_num_new_matched_tokens(fresh, 2 * BLOCK)
    assert is_async2 and n_ext2 == 1 * BLOCK
    conn.update_state_after_alloc(fresh, alloc(4, base=300), n_ext2)
    meta2 = conn.build_connector_meta(sched_output({}))
    assert len(meta2.restores["r2"]) == 1 + 3  # final chunk carries the tail


def test_mixed_local_and_tier_resume():
    """GPU cache holds most of the prefix; the tier supplies the tail."""
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    fresh = FakeRequest("r5", [h(i) for i in range(4)], num_tokens=4 * BLOCK + 8)
    conn.on_new_request(fresh)
    # Scheduler reports 2 blocks already computed locally.
    n_ext, is_async = conn.get_num_new_matched_tokens(fresh, 2 * BLOCK)
    assert is_async and n_ext == 1 * BLOCK
    # num_computed_tokens stays 0 while waiting (framework behavior).
    conn.update_state_after_alloc(fresh, alloc(4, base=400), n_ext)
    meta = conn.build_connector_meta(sched_output({}))
    ops = meta.restores["r5"]
    # 1 attention block (position 2) + 3 tail states.
    assert len(ops) == 1 + 3
    targets = {b for _, b in ops}
    assert 402 in targets
    assert len([b for b in targets if b >= 490]) == 3


def test_short_prompt_or_mismatch_misses():
    conn = make_connector()
    run_conversation(conn, "r1", 4)
    # Prompt ends exactly at the tail boundary: nothing left to compute.
    exact = FakeRequest("r3", [h(i) for i in range(3)], num_tokens=3 * BLOCK)
    assert conn.get_num_new_matched_tokens(exact, 0) == (0, False)
    other = FakeRequest("r4", [h(50 + i) for i in range(6)], num_tokens=99)
    assert conn.get_num_new_matched_tokens(other, 0) == (0, False)
