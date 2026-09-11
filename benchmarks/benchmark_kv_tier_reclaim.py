# SPDX-License-Identifier: Apache-2.0
"""CPU pressure reproducer for cached-prefix staging, not a serving benchmark."""

import argparse
import copy
import hashlib
import json
import statistics
import time

from vllm.v1.core.kv_tier_index import HostKVTierIndex


def pressure_index(blocks=216, host_slots=11915, disk_slots=42366):
    index = HostKVTierIndex(host_slots, [0, 1], disk_slots, {0: 1, 1: 8})
    # Build real trajectories through the production staging/confirmation API.
    # First let old copies reach disk; then withhold new write acknowledgments
    # until every remaining host slot is protected by pending write-through.
    for owner_num in range(1000):
        owner = f"seed-{owner_num}"
        slots = []
        full = False
        for logical in range(blocks):
            h = (owner_num * blocks + logical + 1).to_bytes(16, "little")
            for gid in sorted(index.due(logical)):
                slot = index.stage_attention(owner, logical, h, gid=gid)
                if slot is None:
                    full = True
                    break
                slots.append(slot)
            if full:
                break
        if not full:
            tails = index.stage_tail_states(owner, blocks, 4, boundary_hash=h)
            if tails is not None:
                slots.extend(tails.values())
        index.confirm_writes(slots)
        writes = index.take_disk_writes(slots)
        if owner_num < 120:
            index.confirm_disk_writes(writes)
        if full:
            assert not index._free
            assert len(index._host_busy) == host_slots
            return index
    raise AssertionError("failed to produce host-tier pressure")


def stage_cached_batch(index, requests, blocks=216):
    allocated = 0
    for request in range(requests):
        for logical in range(blocks):
            h = (1000000 + request * blocks + logical).to_bytes(16, "little")
            for gid in sorted(index.due(logical)):
                slot = index.stage_attention(f"new-{request}", logical, h, gid=gid)
                allocated += slot is not None
    return allocated


def state_fingerprint(index):
    # Exclude wall-clock LRU timestamps, retain order, ownership and slot maps.
    state = {
        "trajectories": [
            (
                owner,
                t.hashes,
                t.attn_slots,
                t.tail_state_slots,
                t.tail_boundary,
                t.tail_hash,
                t.tail_pending,
                t.disk_attn,
                t.disk_tail_slots,
                t.disk_tail_boundary,
                t.main_slots,
            )
            for owner, t in index._trajectories.items()
        ],
        "free": index._free,
        "disk_free": index._disk_free,
        "host_busy": index._host_busy,
        "pending": sorted(index._pending_write),
        "disk_pending": sorted(index._disk_pending),
    }
    encoded = json.dumps(state, default=lambda b: b.hex()).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", nargs="+", type=int, default=[1, 8, 16, 32])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    initial = pressure_index()
    results = []
    for requests in args.requests:
        durations = []
        fingerprints = []
        for _ in range(args.repeats):
            index = copy.deepcopy(initial)
            started = time.perf_counter()
            allocated = stage_cached_batch(index, requests)
            durations.append(time.perf_counter() - started)
            assert allocated == 0  # all host slots remain protected
            fingerprints.append(state_fingerprint(index))
        assert len(set(fingerprints)) == 1
        results.append(
            {
                "requests": requests,
                "seconds": durations,
                "median_seconds": statistics.median(durations),
                "state_sha256": fingerprints[0],
                "allocated": allocated,
            }
        )
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
