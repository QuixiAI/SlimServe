# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sync-bracketed step-phase profiler for the Metal serving path.

VLLM_QC_PHASE_PROF=1 wraps the step phases (target forward, sample,
drafter propose, plus the compressor's save-partial/full-compress
sub-phases) with torch.mps.synchronize() timers, so each bucket is
GPU-inclusive wall time. The synchronizes serialize the step pipeline, so
absolute step time under this profiler is slightly inflated; the SPLIT is
the diagnostic. Dump lands in /tmp/phaseprof_<pid>.txt, rewritten every
few dozen phase closes — EngineCore exits via os._exit, which skips
atexit, so the periodic rewrite is the only dump that survives a server
shutdown.
"""

import atexit
import contextlib
import os
import time
from collections import defaultdict

_ENABLED = os.environ.get("VLLM_QC_PHASE_PROF") == "1"
_stats: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
_dump_registered = False
_closes = 0
_DUMP_EVERY = 32


def enabled() -> bool:
    return _ENABLED


@contextlib.contextmanager
def phase(name: str):
    if not _ENABLED:
        yield
        return
    import torch

    global _dump_registered, _closes
    if not _dump_registered:
        _dump_registered = True
        atexit.register(_dump)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.mps.synchronize()
        entry = _stats[name]
        entry[0] += 1
        entry[1] += time.perf_counter() - t0
        _closes += 1
        if _closes % _DUMP_EVERY == 0:
            _dump()


def _dump() -> None:
    path = f"/tmp/phaseprof_{os.getpid()}.txt"
    total = sum(s for _, s in _stats.values())
    with open(path, "w") as f:
        f.write(f"{'seconds':>10} {'count':>8} {'ms/call':>9} {'share':>6}  phase\n")
        for name, (n, s) in sorted(_stats.items(), key=lambda kv: -kv[1][1]):
            f.write(
                f"{s:10.3f} {n:8d} {s / n * 1e3:9.2f} {s / total * 100:5.1f}%  {name}\n"
            )
