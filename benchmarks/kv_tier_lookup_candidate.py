# SPDX-License-Identifier: Apache-2.0
"""Frozen pre-hash-first lookup reference versus the serving implementation.

The historical filename is retained for reproducible benchmark commands.
LegacyLookupIndex intentionally keeps the old scan-first lookup so differential
tests and timings do not silently compare the optimized code with itself.
This reference is not imported by serving.
"""

import copy
import json
import statistics
import time

from benchmarks.benchmark_kv_tier_reclaim import pressure_index
from vllm.v1.core.kv_tier_index import HostKVTierIndex


class LegacyLookupIndex(HostKVTierIndex):
    def lookup(self, hashes, allow_partial=False):
        best = None
        best_owner = None
        for owner, traj in list(self._trajectories.items()):
            if owner in self._promotions:
                continue
            if allow_partial:
                span = min(traj.attn_prefix_len(self.due), len(hashes))
                m = 0
                while (
                    m < span
                    and traj.hashes[m] == hashes[m]
                    and not any(
                        s in self._pending_write for s in traj.attn_slots[m].values()
                    )
                ):
                    m += 1
                if m <= 0:
                    continue
                if best is not None and m <= best[1]:
                    continue
                best = (owner, m, [dict(d) for d in traj.attn_slots[:m]], {})
                best_owner = owner
                continue
            n = traj.resumable_blocks(
                self.due,
                self._disk_pending,
                self._main_pending if self.require_main else None,
            )
            if n <= 0 or n > len(hashes):
                continue
            if best is not None and n <= best[1]:
                continue
            if any(
                s in self._pending_write
                for d in traj.attn_slots[:n]
                for s in d.values()
            ):
                continue
            if traj.hashes[:n] != hashes[:n]:
                continue
            tail = {} if traj.tail_pending else dict(traj.tail_state_slots)
            best = (owner, n, [dict(d) for d in traj.attn_slots[:n]], tail)
            best_owner = owner
        if best_owner is not None:
            self.touch(best_owner)
        return best


def main():
    initial = pressure_index()
    hit = next(
        t.hashes[:]
        for t in initial._trajectories.values()
        if t.resumable_blocks(initial.due, initial._disk_pending) > 0
    )
    miss = [(10000000 + i).to_bytes(16, "little") for i in range(216)]
    # Also keep the beginning equal, as conversations sharing a system prompt
    # do: the optimization must compare the chain, not just its first hash.
    shared_prefix_miss = hit[:8] + miss[8:]
    shared = copy.deepcopy(initial)
    common = [b"common-prefix-" + bytes([i]) for i in range(8)]
    for trajectory in shared._trajectories.values():
        count = min(8, len(trajectory.hashes))
        trajectory.hashes[:count] = common[:count]
    pending = HostKVTierIndex(10000, [0], 10000)
    for owner in range(30):
        pending_hashes = [
            (owner * 300 + i + 1).to_bytes(16, "little") for i in range(216)
        ]
        for i, block_hash in enumerate(pending_hashes):
            pending.stage_attention(str(owner), i, block_hash)
        pending.stage_tail_states(str(owner), 216, 4, boundary_hash=pending_hashes[-1])
    for name, fixture, query in [
        ("hit", initial, hit),
        ("miss", initial, miss),
        ("shared_prefix_miss", initial, shared_prefix_miss),
        ("all_share_prefix_miss", shared, common + miss[8:]),
        ("all_tails_pending", pending, pending_hashes),
    ]:
        times = {"baseline": [], "hash_first": []}
        for repeat in range(5):
            for label in list(times) if repeat % 2 == 0 else list(reversed(times)):
                index = copy.deepcopy(fixture)
                index.__class__ = (
                    HostKVTierIndex if label == "hash_first" else LegacyLookupIndex
                )
                expected = LegacyLookupIndex.lookup(index, query)
                assert index.lookup(query) == expected
                start = time.perf_counter()
                for _ in range(100):
                    result = index.lookup(query)
                times[label].append((time.perf_counter() - start) * 1e6 / 100)
                assert result == expected
        print(
            json.dumps(
                {
                    "query": name,
                    "microseconds": times,
                    "median_us": {k: statistics.median(v) for k, v in times.items()},
                    "scope": "CPU lookup only; no serving speedup claim",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
