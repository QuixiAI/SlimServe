# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sync-bracketed step-phase profiler for the Metal serving path.

VLLM_QC_PHASE_PROF=1 wraps the step phases (target forward, sample,
drafter propose, plus the compressor's save-partial/full-compress
sub-phases) with torch.mps.synchronize() timers, so each bucket is
GPU-inclusive wall time. The synchronizes serialize the step pipeline, so
absolute step time under this profiler is slightly inflated; the SPLIT is
the diagnostic. Dump lands in a private mkdtemp dir (path logged), rewritten every
few dozen phase closes — EngineCore exits via os._exit, which skips
atexit, so the periodic rewrite is the only dump that survives a server
shutdown.
"""

import atexit
import contextlib
import functools
import os
import time
from collections import defaultdict

_ENABLED = os.environ.get("VLLM_QC_PHASE_PROF") == "1"
_CENSUS = os.environ.get("VLLM_QC_OP_CENSUS") == "1"
_PYPROF = os.environ.get("VLLM_QC_PYPROF") == "1"
_pyprof = None
_pyprof_steps = 0
_stats: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
_op_counts: dict[str, int] = defaultdict(int)
_op_steps = 0
_dump_registered = False
_closes = 0
_DUMP_EVERY = 32
_PROF_DIR: str | None = None


def _prof_path(name: str) -> str:
    """A file path inside a private, unpredictable per-process directory.

    /tmp names keyed on the pid are symlink-attackable (open(..., "w")
    follows links, letting a local user redirect the truncate+write);
    mkdtemp gives 0700 and an unguessable suffix.
    """
    global _PROF_DIR
    if _PROF_DIR is None:
        import tempfile

        _PROF_DIR = tempfile.mkdtemp(prefix="qc_phaseprof_")
        print(f"[phaseprof] writing dumps under {_PROF_DIR}")
    return os.path.join(_PROF_DIR, name)


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


def wrap(name: str):
    """Decorator form of phase() for bracketing whole methods.

    With VLLM_QC_OP_CENSUS=1 the bracket also counts every torch op
    dispatched inside it (by qualified name), dumping per-step averages to
    /tmp/opcensus_<pid>.txt — the per-step launch census for the Metal
    dispatch-collapse work. Precedence when several flags are set:
    census > pyprof > plain phase timing (the pyprof arm skips the phase
    bracket so the profile is not dominated by the sync calls).
    """

    def deco(fn):
        if not (_ENABLED or _CENSUS or _PYPROF):
            # All profiling off (the common case): decorate nothing, so
            # cross-platform hot paths pay zero overhead.
            return fn

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if _CENSUS:
                with phase(name), _census_mode():
                    return fn(*args, **kwargs)
            if _PYPROF:
                with _pyprof_mode():
                    return fn(*args, **kwargs)
            with phase(name):
                return fn(*args, **kwargs)

        return inner

    return deco


@contextlib.contextmanager
def _pyprof_mode():
    """cProfile the bracketed method (VLLM_QC_PYPROF=1). C-extension time is
    charged to the Python call site — the per-call-site attribution of the
    step's host budget. Dump: /tmp/pyprof_<pid>.txt (top by cumulative and
    by tottime), rewritten every _DUMP_EVERY bracket closes."""
    import cProfile

    global _pyprof, _pyprof_steps
    if _pyprof is None:
        _pyprof = cProfile.Profile()
    _pyprof.enable()
    try:
        yield
    finally:
        _pyprof.disable()
        _pyprof_steps += 1
        if _pyprof_steps % _DUMP_EVERY == 0:
            _dump_pyprof()


def _dump_pyprof() -> None:
    import io
    import pstats

    path = _prof_path(f"pyprof_{os.getpid()}.txt")
    buf = io.StringIO()
    st = pstats.Stats(_pyprof, stream=buf)
    # _pyprof_steps counts execute_model and sample_tokens brackets
    # separately; a serving step closes both.
    buf.write(f"steps={max(1, _pyprof_steps // 2)}\n")
    st.sort_stats("cumulative").print_stats(45)
    buf.write("\n=== by tottime ===\n")
    st.sort_stats("tottime").print_stats(45)
    with open(path, "w") as f:
        f.write(buf.getvalue())


_INTERESTING = {
    "aten.item.default",
    "aten.linear.default",
    "aten.cat.default",
    "aten.index_put_.default",
    "aten.copy_.default",
    "aten._to_copy.default",
    "aten.to.device",
    "aten.to.dtype",
    "aten.to.dtype_layout",
}
_op_stacks: dict[str, int] = defaultdict(int)
_op_stack_time: dict[str, float] = defaultdict(float)


def _stack_key(name: str) -> str:
    import traceback

    frames = [
        f"{os.path.basename(fr.filename)}:{fr.lineno}({fr.name})"
        for fr in traceback.extract_stack(limit=24)
        if "/vllm/" in fr.filename and "metal_phaseprof" not in fr.filename
    ]
    return f"{name} <- " + " <- ".join(frames[-4:])


@contextlib.contextmanager
def _census_mode():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class _OpCensus(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if name in _INTERESTING:
                dev = "?"
                if args and isinstance(args[0], torch.Tensor):
                    dev = args[0].device.type
                key = f"{name}@{dev}"
                t0 = time.perf_counter()
                out = func(*args, **(kwargs or {}))
                dt = time.perf_counter() - t0
                sk = _stack_key(key)
                if sk in _op_stacks or len(_op_stacks) < 400:
                    _op_stacks[sk] += 1
                    _op_stack_time[sk] += dt
                _op_counts[key] += 1
                return out
            _op_counts[name] += 1
            return func(*args, **(kwargs or {}))

    global _op_steps
    with _OpCensus():
        yield
    _op_steps += 1
    if _op_steps % _DUMP_EVERY == 0:
        _dump_census()


def _dump_census() -> None:
    path = _prof_path(f"opcensus_{os.getpid()}.txt")
    total = sum(_op_counts.values())
    # _op_steps counts execute_model and sample_tokens brackets separately;
    # a serving step closes both, so divide by the bracket-pair count.
    steps = max(1, _op_steps // 2)
    with open(path, "w") as f:
        f.write(
            f"{total} op calls over {steps} steps = {total / steps:.1f} calls/step\n"
        )
        f.write(f"{'calls':>10} {'per-step':>9}  op\n")
        for name, n in sorted(_op_counts.items(), key=lambda kv: -kv[1]):
            f.write(f"{n:10d} {n / steps:9.1f}  {name}\n")
        if _op_stacks:
            f.write("\n--- stacks (interesting ops, sorted by TIME) ---\n")
            f.write(f"{'seconds':>10} {'ms/step':>8} {'calls':>10}  stack\n")
            for key, sec in sorted(_op_stack_time.items(), key=lambda kv: -kv[1]):
                f.write(
                    f"{sec:10.3f} {sec / steps * 1e3:8.2f}"
                    f" {_op_stacks[key]:10d}  {key}\n"
                )


def _dump() -> None:
    path = _prof_path(f"phaseprof_{os.getpid()}.txt")
    total = sum(s for _, s in _stats.values())
    with open(path, "w") as f:
        f.write(f"{'seconds':>10} {'count':>8} {'ms/call':>9} {'share':>6}  phase\n")
        for name, (n, s) in sorted(_stats.items(), key=lambda kv: -kv[1][1]):
            f.write(
                f"{s:10.3f} {n:8d} {s / n * 1e3:9.2f} {s / total * 100:5.1f}%  {name}\n"
            )
