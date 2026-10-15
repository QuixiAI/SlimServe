#!/usr/bin/env python3
"""Run one heavy GPU job at a time, and only when it fits.

Every model-loading run on this box goes through here:

    perf/ltx25_harness/gpu_run.py --need-gb 45 -- python n1_forward_gate.py ...

It takes an exclusive machine-wide lock (other callers wait), refuses to start
unless `--need-gb` plus an OS reserve is available, and refuses when
iogpu.wired_limit_mb would let Metal wire the OS out of memory.

Why: on 2026-10-02 four model-loading processes ran at once (84.7 + 64.2 +
18.1 GiB resident and one more loading, on 128 GiB, with the wired limit at
122880). GPU memory is wired and cannot be compressed or swapped, the
compressor ran out of space, watchdogd was starved for 90 s and the kernel
panicked. See HANDOFF-LTX25.md, "Ops constraints".
"""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from paths import SCRATCH

LOCK = f"{SCRATCH}/gpu.lock"
OS_RESERVE_GB = 16  # never plan into the last 16 GiB
MAX_WIRED_LIMIT_MB = 110_000  # 128 GiB box: leave the OS at least ~20 GiB unwireable


def sysctl(name: str) -> int:
    return int(subprocess.check_output(["sysctl", "-n", name]).split()[0])


def available_gb() -> float:
    page = sysctl("hw.pagesize")
    out = subprocess.check_output(["vm_stat"], text=True)
    pages = {
        line.split(":")[0]: int(line.split(":")[1].strip().rstrip("."))
        for line in out.splitlines()[1:]
        if ":" in line
    }
    free = (
        pages.get("Pages free", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages speculative", 0)
        + pages.get("Pages purgeable", 0)
    )
    return free * page / 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--need-gb", type=float, required=True, help="expected peak memory of the job"
    )
    ap.add_argument(
        "--wait-s", type=float, default=3600, help="how long to wait for the lock"
    )
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    total = sysctl("hw.memsize") / 2**30
    if a.need_gb + OS_RESERVE_GB > total:
        sys.exit(
            f"[gpu_run] refused: job needs {a.need_gb:.0f} GiB + {OS_RESERVE_GB} GiB "
            f"reserve > {total:.0f} GiB RAM"
        )
    wired = sysctl("iogpu.wired_limit_mb")
    if wired > MAX_WIRED_LIMIT_MB:
        sys.exit(
            f"[gpu_run] refused: iogpu.wired_limit_mb={wired} leaves the OS too "
            "little; "
            f"set it to 0 (default) or <= {MAX_WIRED_LIMIT_MB}: "
            "sudo sysctl iogpu.wired_limit_mb=0"
        )
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    with open(LOCK, "a+") as fh:
        deadline = time.time() + a.wait_s
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    sys.exit("[gpu_run] refused: another GPU job still holds the lock")
                time.sleep(2)
        # A process that just exited releases its wired memory over several
        # seconds; wait a little for that before concluding something holds it.
        deadline_mem = time.time() + 90
        while True:
            avail = available_gb()
            if avail >= a.need_gb + OS_RESERVE_GB or time.time() > deadline_mem:
                break
            time.sleep(3)
        if avail < a.need_gb + OS_RESERVE_GB:
            sys.exit(
                f"[gpu_run] refused: {avail:.0f} GiB available, job needs "
                f"{a.need_gb:.0f} + {OS_RESERVE_GB} reserve; something else is "
                "holding memory (check `ps -axm -o rss,pid,comm | sort -nr | head`)"
            )
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()} need={a.need_gb} {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.call(cmd)  # lock is held until this process exits


if __name__ == "__main__":
    sys.exit(main())
