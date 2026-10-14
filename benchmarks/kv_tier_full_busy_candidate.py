# SPDX-License-Identifier: Apache-2.0
"""Frozen allocator reference versus the serving full-host-busy guard.

Both live collections contain unique host slot IDs, bounded by num_slots.
If either contains the entire arena, every trajectory with host slots is
busy. Reclamation cannot succeed until a completion changes that collection.
No negative cache, extra counters, or completion invalidation are introduced.
"""

import copy
import json
import statistics
import time

from benchmarks.benchmark_kv_tier_reclaim import (
    pressure_index,
    stage_cached_batch,
    state_fingerprint,
)
from vllm.v1.core.kv_tier_index import HostKVTierIndex


class LegacyAllocIndex(HostKVTierIndex):
    def _alloc_slot(self, protect):
        if not self._free and not self._reclaim(protect):
            return None
        slot = self._free.pop()
        self._pending_write.add(slot)
        return slot


def main():
    initial = pressure_index()
    assert set(initial._host_busy) == set(range(initial.num_slots))
    for requests in (1, 8, 16, 32):
        times = {"lazy_reclaim": [], "full_busy_guard": []}
        fingerprints = set()
        for repeat in range(3):
            for label in list(times) if repeat % 2 == 0 else list(reversed(times)):
                index = copy.deepcopy(initial)
                index.__class__ = (
                    HostKVTierIndex if label == "full_busy_guard" else LegacyAllocIndex
                )
                start = time.perf_counter()
                allocated = stage_cached_batch(index, requests)
                times[label].append(time.perf_counter() - start)
                assert allocated == 0
                fingerprints.add(state_fingerprint(index))
        assert len(fingerprints) == 1
        print(
            json.dumps(
                {
                    "requests": requests,
                    "seconds": times,
                    "median_seconds": {
                        k: statistics.median(v) for k, v in times.items()
                    },
                    "state_sha256": next(iter(fingerprints)),
                    "scope": "CPU allocation only; no serving speedup claim",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
