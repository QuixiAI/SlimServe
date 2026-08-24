# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU<->GPU sync-point profiler for the Metal serving path.

Enabled with ``VLLM_SYNCPROF=1`` and installed only inside the worker
process (from ``MetalWorker.init_device``). Wraps the torch entry points
that force a host round-trip on MPS and records per-call-site counts and
wall time. A daemon thread rewrites ``/tmp/syncprof_<pid>.txt`` every
``VLLM_SYNCPROF_INTERVAL`` seconds (default 30); no signal handlers are
involved because an unhandled harvest signal terminates the process
silently, which is indistinguishable from a crash.

``VLLM_SYNCPROF_TARGETS`` takes a comma-separated subset of the target
labels to bisect a misbehaving wrapper (default: all).

torch.Event is an immutable C type in torch 2.13 and cannot be wrapped;
the per-step event wait is captured through ``AsyncOutput.get_output``
instead, which is where the engine blocks on it.
"""

import atexit
import os
import sys
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_stats: dict = {}
_installed = False
_t0 = time.monotonic()
_cb_census_read = None


def _site(depth: int = 4) -> tuple:
    frame = sys._getframe(2)
    parts = []
    while frame is not None and len(parts) < depth:
        code = frame.f_code
        if "metal_syncprof" not in code.co_filename:
            parts.append((code.co_filename, frame.f_lineno, code.co_name))
        frame = frame.f_back
    return tuple(parts)


def _record(label: str, dt: float) -> None:
    key = (label, _site())
    entry = _stats.get(key)
    if entry is None:
        _stats[key] = [1, dt]
    else:
        entry[0] += 1
        entry[1] += dt


def _wrap_method(owner, name: str, label: str) -> bool:
    try:
        orig = getattr(owner, name)
    except AttributeError:
        return False

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            _record(label, time.perf_counter() - t0)

    wrapper.__name__ = getattr(orig, "__name__", name)
    try:
        setattr(owner, name, wrapper)
    except (TypeError, AttributeError) as e:
        logger.warning("syncprof: cannot wrap %s: %s", label, e)
        return False
    return True


def _format_dump() -> str:
    lines = [
        f"# syncprof pid={os.getpid()} uptime={time.monotonic() - _t0:.1f}s",
    ]
    if _cb_census_read is not None:
        try:
            created, completed, busy_s = _cb_census_read()
            lines.append(
                f"# cb_census created={created} completed={completed} "
                f"gpu_busy={busy_s:.3f}s"
            )
        except Exception as e:  # noqa: BLE001 - diagnostics must not raise
            lines.append(f"# cb_census read failed: {e}")
    lines.append(f"{'seconds':>10} {'count':>9}  label / site")
    totals: dict = {}
    for (label, _), (count, seconds) in _stats.items():
        t = totals.setdefault(label, [0, 0.0])
        t[0] += count
        t[1] += seconds
    for label, (count, seconds) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"{seconds:10.3f} {count:9d}  TOTAL {label}")
    lines.append("")
    rows = sorted(_stats.items(), key=lambda kv: -kv[1][1])
    for (label, site), (count, seconds) in rows[:150]:
        where = " <- ".join(
            f"{os.path.basename(fn)}:{lineno}:{func}" for fn, lineno, func in site
        )
        lines.append(f"{seconds:10.3f} {count:9d}  [{label}] {where}")
    return "\n".join(lines) + "\n"


def _dump() -> None:
    path = f"/tmp/syncprof_{os.getpid()}.txt"
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write(_format_dump())
        os.replace(tmp, path)
    except OSError:
        pass


def _compressor_pages() -> int:
    import subprocess

    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if "occupied by compressor" in line:
                return int(line.split()[-1].rstrip("."))
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        pass
    return -1


def _dump_loop(interval: float) -> None:
    csv_path = f"/tmp/syncprof_cb_{os.getpid()}.csv"
    import torch

    while True:
        time.sleep(interval)
        _dump()
        if _cb_census_read is not None:
            try:
                created, completed, busy_s = _cb_census_read()
                driver = torch.mps.driver_allocated_memory()
                current = torch.mps.current_allocated_memory()
                comp_bytes = _compressor_pages() * 16384
                with open(csv_path, "a") as f:
                    f.write(
                        f"{time.monotonic() - _t0:.3f},{created},"
                        f"{completed},{busy_s:.6f},{driver},{current},"
                        f"{comp_bytes}\n"
                    )
            except Exception:  # noqa: BLE001 - diagnostics must not raise
                pass


def _install_cb_census() -> None:
    """Arm the extension's command-buffer census on this thread.

    Must run on the thread that owns the MPS stream (the worker init
    thread); forces MPS stream creation first so the extension can reach
    the live command queue.
    """
    global _cb_census_read
    try:
        import torch

        from vllm import _quixicore_C as qc
    except ImportError as e:
        logger.warning("syncprof: cb census unavailable: %s", e)
        return
    if not hasattr(qc, "cb_census_install"):
        logger.warning("syncprof: extension has no cb_census_install")
        return
    try:
        torch.zeros(1, device="mps")
        if qc.cb_census_install():
            _cb_census_read = qc.cb_census_read
            logger.info("syncprof: command-buffer census armed")
        else:
            logger.warning("syncprof: cb_census_install found no factory")
    except Exception as e:  # noqa: BLE001 - diagnostics must not raise
        logger.warning("syncprof: cb census install failed: %s", e)


def install() -> None:
    global _installed
    if _installed or os.environ.get("VLLM_SYNCPROF", "0") != "1":
        return
    _installed = True

    import torch

    targets = [
        (torch.mps, "synchronize", "mps.synchronize"),
        (torch.Tensor, "item", "Tensor.item"),
        (torch.Tensor, "tolist", "Tensor.tolist"),
        (torch.Tensor, "numpy", "Tensor.numpy"),
        (torch.Tensor, "cpu", "Tensor.cpu"),
        (torch.Tensor, "nonzero", "Tensor.nonzero"),
        (torch.Tensor, "__bool__", "Tensor.bool"),
    ]
    # torch.mps.Event stays unwrapped: the async-output path uses torch.Event
    # (an immutable C type that cannot be wrapped); its per-step wait is
    # captured through AsyncOutput.get_output below.
    try:
        from vllm.v1.worker.gpu import async_utils

        targets.append((async_utils.AsyncOutput, "get_output", "AsyncOutput.get_output"))
    except ImportError:
        pass
    try:
        from vllm.v1.worker.gpu import model_runner

        targets.append(
            (model_runner.GPUModelRunner, "execute_model", "execute_model")
        )
    except ImportError:
        pass

    selected = os.environ.get("VLLM_SYNCPROF_TARGETS")
    if selected:
        wanted = {s.strip() for s in selected.split(",") if s.strip()}
        targets = [t for t in targets if t[2] in wanted]

    installed = [label for owner, name, label in targets
                 if _wrap_method(owner, name, label)]
    logger.info("syncprof armed (%d wrappers): %s", len(installed),
                ", ".join(installed))

    _install_cb_census()

    interval = float(os.environ.get("VLLM_SYNCPROF_INTERVAL", "30"))
    threading.Thread(
        target=_dump_loop, args=(interval,), daemon=True, name="syncprof-dump"
    ).start()
    atexit.register(_dump)
