# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The invalid-block handler on a hybrid (multi-group) allocator.

vLLM's `_update_requests_with_invalid_blocks` unpacked a single block list
per request; on GLM-5.3-Flash (14 KV groups) the first fail-closed tier
restore raised `ValueError: too many values to unpack` and killed the
engine (2026-09-12). The handler now walks every group at its own block
size and truncates at the earliest token any invalid block covers."""

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def _scheduler(group_block_sizes, block_ids_by_req):
    groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=bs))
        for bs in group_block_sizes
    ]
    manager = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        get_block_ids=lambda req_id: block_ids_by_req[req_id],
    )
    sched = object.__new__(Scheduler)
    sched.kv_cache_manager = manager
    sched.block_size = min(group_block_sizes)
    return sched


def _req(rid, computed):
    return SimpleNamespace(request_id=rid, num_computed_tokens=computed)


def test_hybrid_groups_truncate_at_the_earliest_invalid_token():
    # Group 0: 16-token blocks; group 1: 32-token blocks (2 hash blocks).
    sched = _scheduler(
        [16, 32], {"a": ([10, 11, 12, 13], [20, 21]), "b": ([30, 31], [40])}
    )
    reqs = [_req("a", 64), _req("b", 32)]
    # Block 21 (group 1, second block) covers tokens 32..63 of "a";
    # block 30 (group 0, first block) covers tokens 0..15 of "b".
    affected, tokens, evict = sched._update_requests_with_invalid_blocks(
        reqs, {21, 30}, {}, evict_blocks=True
    )
    assert affected == {"a", "b"}
    assert reqs[0].num_computed_tokens == 32
    assert reqs[1].num_computed_tokens == 0
    assert tokens == 32 + 32
    # Invalid blocks and everything downstream in their own group.
    assert evict == {21, 30, 31}


def test_invalid_block_in_a_narrow_group_truncates_by_that_groups_size():
    sched = _scheduler([16, 32], {"a": ([10, 11, 12, 13], [20, 21])})
    reqs = [_req("a", 64)]
    affected, tokens, evict = sched._update_requests_with_invalid_blocks(
        reqs, {13}, {}, evict_blocks=True
    )
    assert affected == {"a"}
    assert reqs[0].num_computed_tokens == 48 and tokens == 16
    assert evict == {13}


def test_only_externally_computed_blocks_are_considered():
    sched = _scheduler([16, 32], {"a": ([10, 11, 12, 13], [20, 21])})
    reqs = [_req("a", 64)]
    # 40 of the 64 tokens were scheduled this step (not external): only
    # blocks covering the first 24 tokens can be invalidated.
    affected, tokens, _ = sched._update_requests_with_invalid_blocks(
        reqs, {13, 21}, {"a": 40}, evict_blocks=False
    )
    assert affected == set() and tokens == 0
    assert reqs[0].num_computed_tokens == 64


def test_shared_invalid_block_is_recomputed_once():
    sched = _scheduler([16], {"a": ([10, 11],), "b": ([10, 12],)})
    reqs = [_req("a", 32), _req("b", 32)]
    affected, tokens, evict = sched._update_requests_with_invalid_blocks(
        reqs, {10}, {}, evict_blocks=True
    )
    assert affected == {"a", "b"}
    assert reqs[0].num_computed_tokens == 0  # recomputes block 10
    assert reqs[1].num_computed_tokens == 32  # shares it, keeps its count
    assert tokens == 32 and evict == {10, 11}
