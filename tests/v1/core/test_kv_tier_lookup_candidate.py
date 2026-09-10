# SPDX-License-Identifier: Apache-2.0
"""Differential checks against the actual serving lookup implementation."""

import copy
import random
from unittest.mock import Mock

import pytest

from benchmarks.kv_tier_lookup_candidate import LegacyLookupIndex
from vllm.v1.core.kv_tier_index import HostKVTierIndex


@pytest.mark.parametrize("seed", range(20))
def test_random_pending_disk_host_gaps_and_prefixes_match(seed):
    rng = random.Random(seed)
    baseline = LegacyLookupIndex(100, [0, 1], 500, {0: 1, 1: 4})
    queries = []
    for owner in range(12):
        length = rng.choice([4, 8, 12, 16])
        hashes = [
            (0 if i < 2 else owner * 100 + i).to_bytes(16, "little")
            for i in range(length)
        ]
        slots = []
        for logical in range(length):
            for gid in baseline.due(logical):
                if rng.random() < 0.08:
                    continue
                slot = baseline.stage_attention(
                    str(owner), logical, hashes[logical], gid
                )
                if slot is not None:
                    slots.append(slot)
        tail = baseline.stage_tail_states(
            str(owner), length, 2, boundary_hash=hashes[-1]
        )
        if tail:
            slots.extend(tail.values())
        confirmed = [s for s in slots if rng.random() > 0.08]
        baseline.confirm_writes(confirmed)
        writes = baseline.take_disk_writes(confirmed)
        baseline.confirm_disk_writes([op for op in writes if rng.random() > 0.15])
        queries.extend([hashes, hashes[:3], hashes + [b"extension"]])
    queries.extend([[], [b"miss"] * 20])
    candidate = copy.deepcopy(baseline)
    candidate.__class__ = HostKVTierIndex
    rng.shuffle(queries)
    for query in queries:
        for partial in [False, True]:
            assert candidate.lookup(query, partial) == baseline.lookup(query, partial)
            assert list(candidate._trajectories) == list(baseline._trajectories)


def test_unrelated_hashes_do_not_walk_page_readiness():
    index = HostKVTierIndex(4)
    slot = index.stage_attention("old", 0, b"old")
    tail = index.stage_tail_states("old", 1, 1, boundary_hash=b"old")
    index.confirm_writes([slot, *tail.values()])
    index._trajectories["old"].resumable_blocks = Mock(
        side_effect=AssertionError("unrelated trajectory readiness scanned")
    )
    assert index.lookup([b"new"]) is None


def test_matching_hashes_still_check_readiness():
    index = HostKVTierIndex(4)
    slot = index.stage_attention("old", 0, b"old")
    tail = index.stage_tail_states("old", 1, 1, boundary_hash=b"old")
    assert index.lookup([b"old"]) is None
    index.confirm_writes([slot, *tail.values()])
    assert index.lookup([b"old"])[0] == "old"


def test_pending_tail_is_rejected_before_copying_full_hash_chain():
    class NoSlice(list):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError("copied chain despite unavailable tail")
            return super().__getitem__(key)

    index = HostKVTierIndex(4)
    index.stage_attention("old", 0, b"old")
    index.stage_tail_states("old", 1, 1, boundary_hash=b"old")
    index._trajectories["old"].hashes = NoSlice([b"old"])
    assert index.lookup([b"old"]) is None
