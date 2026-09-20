#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only local token-progress watchdog; never restarts or resets services."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

METRIC_NAMES = {
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:num_requests_running": "running_requests",
}
METRIC_RE = re.compile(r"^([^\s{]+)(?:\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?$")
JOURNAL_FRAME_RE = re.compile(r'File "([^"\n]+)", line (\d+), in ([\w.<>]+)\s*$')
EXCEPTION_RE = re.compile(
    r"(?:^|\s)([A-Za-z_]\w*(?:Error|Exception|Timeout))(?::|\s*$)"
)
MAX_METRICS_BYTES = 4 * 1024 * 1024
MAX_STACK_TARGETS = 12


@dataclass(frozen=True)
class Metrics:
    generation_tokens: float
    prompt_tokens: float
    running_requests: float
    generation_created_min: float | None = None
    generation_created_max: float | None = None


def parse_metrics(text: str) -> Metrics:
    """Aggregate series without retaining labels, which may contain user data."""
    values: dict[str, float] = {}
    created = []
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match is None:
            continue
        if match[1] == "vllm:generation_tokens_created":
            epoch = float(match[2])
            if not math.isfinite(epoch) or epoch < 0:
                raise ValueError("invalid creation epoch")
            created.append(epoch)
            continue
        if match[1] not in METRIC_NAMES:
            continue
        key = METRIC_NAMES[match[1]]
        value = float(match[2])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid metric value")
        values[key] = values.get(key, 0.0) + value
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("metric aggregate overflow")
    if set(values) != set(METRIC_NAMES.values()):
        raise ValueError("required metrics missing")
    return Metrics(
        **values,
        generation_created_min=min(created) if created else None,
        generation_created_max=max(created) if created else None,
    )


class ProgressDetector:
    """A missing sample is unknown, never evidence of token starvation."""

    def __init__(self, stall_seconds: float = 45.0):
        if not math.isfinite(stall_seconds) or stall_seconds <= 0:
            raise ValueError("stall_seconds must be positive and finite")
        self.stall_seconds = stall_seconds
        self.previous: Metrics | None = None
        self.last_progress: float | None = None
        self.stalled = False

    def update(self, now: float, sample: Metrics | None) -> dict:
        if sample is None:
            self.last_progress = None
            return {"state": "metrics_unavailable", "diagnose": False}
        previous = self.previous
        self.previous = sample
        epoch_changed = previous is not None and (
            previous.generation_created_min is not None
            and sample.generation_created_min is not None
            and (previous.generation_created_min, previous.generation_created_max)
            != (sample.generation_created_min, sample.generation_created_max)
        )
        reset = previous is not None and (
            epoch_changed
            or sample.generation_tokens < previous.generation_tokens
            or sample.prompt_tokens < previous.prompt_tokens
        )
        progressed = previous is not None and (
            sample.generation_tokens > previous.generation_tokens
            or sample.prompt_tokens > previous.prompt_tokens
        )
        if reset:
            self.last_progress = now
            self.stalled = False
            return {
                "state": "counter_reset",
                "diagnose": False,
                "reset_evidence": "creation_epoch_changed"
                if epoch_changed
                else "counter_decreased",
            }
        if sample.running_requests == 0:
            state = "recovered_idle" if self.stalled else "idle"
            self.last_progress = now
            self.stalled = False
            return {"state": state, "diagnose": False}
        if progressed:
            state = "recovered" if self.stalled else "progress"
            self.last_progress = now
            self.stalled = False
            return {"state": state, "diagnose": False}
        if (
            self.last_progress is None
            or previous is None
            or previous.running_requests == 0
        ):
            self.last_progress = now
        idle_seconds = max(0.0, now - self.last_progress)
        diagnose = idle_seconds >= self.stall_seconds and not self.stalled
        if diagnose:
            self.stalled = True
        return {
            "state": "stall"
            if diagnose
            else "stall_ongoing"
            if self.stalled
            else "waiting",
            "no_progress_seconds": round(idle_seconds, 3),
            "diagnose": diagnose,
        }


def local_metrics_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    try:
        local = (
            parsed.hostname == "localhost"
            or ipaddress.ip_address(parsed.hostname).is_loopback
        )
    except ValueError:
        local = False
    if (
        parsed.scheme != "http"
        or not local
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "metrics URL must be plain HTTP on loopback without credentials or query"
        )
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_metrics(url: str, timeout: float = 3) -> Metrics:
    # Ignore proxy environment variables: local metrics must stay local.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(url, timeout=timeout) as response:
        body = response.read(MAX_METRICS_BYTES + 1)
    if len(body) > MAX_METRICS_BYTES:
        raise ValueError("metrics response exceeds limit")
    return parse_metrics(body.decode("utf-8"))


def run_readonly(argv: list[str], timeout: float = 5) -> tuple[str | None, dict]:
    if timeout <= 0:
        return None, {"available": False, "error_type": "duration_exhausted"}
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, {"available": False, "error_type": type(exc).__name__}
    if result.returncode:
        return None, {"available": False, "returncode": result.returncode}
    return result.stdout, {"available": True}


def service_main_pid(unit: str, timeout: float = 3) -> tuple[list[int], dict]:
    output, status = run_readonly(
        ["systemctl", "--user", "show", "--property=MainPID", "--value", "--", unit],
        timeout=timeout,
    )
    if output is None:
        return [], status
    value = output.strip()
    if not value.isdigit():
        return [], {"available": False, "error_type": "invalid_main_pid"}
    pid = int(value)
    if pid == 0:
        return [], {"available": False, "error_type": "service_inactive"}
    return [pid], {"available": True, "main_pid": pid}


def prioritized_stack_targets(rows: list[dict], roots: list[int]) -> list[int]:
    def priority(row):
        name = row["name"].lower()
        if "worker" in name:
            return 0, row["pid"]
        if "engine" in name:
            return 1, row["pid"]
        if row["pid"] in roots:
            return 2, row["pid"]
        return 3, row["pid"]

    return [row["pid"] for row in sorted(rows, key=priority)]


def process_snapshot(pids: list[int], timeout: float = 5) -> dict:
    output, status = run_readonly(
        ["ps", "-eo", "pid=,ppid=,stat=,comm=,wchan:32="], timeout=timeout
    )
    if output is None:
        return status
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0].isdigit() and fields[1].isdigit():
            rows.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "state": fields[2],
                    "name": fields[3],
                    "wchan": fields[4],
                }
            )
    if pids:
        selected = set(pids)
        while True:
            children = {row["pid"] for row in rows if row["ppid"] in selected}
            if children <= selected:
                break
            selected.update(children)
        rows = [row for row in rows if row["pid"] in selected]
    return {**status, "processes": rows[:256], "truncated": len(rows) > 256}


def safe_journal_frames(text: str) -> list[dict]:
    """Never retain journal messages, source-code lines, prompts, or arguments."""
    frames = []
    for line in text.splitlines():
        match = JOURNAL_FRAME_RE.search(line)
        if match:
            frames.append(
                {
                    "file": Path(match[1]).name,
                    "line": int(match[2]),
                    "function": match[3],
                }
            )
            continue
        exception = EXCEPTION_RE.search(line)
        if exception:
            frames.append({"exception_type": exception[1]})
        lower = line.lower()
        if "sample_tokens" in lower and ("timeout" in lower or "timed out" in lower):
            frames.append({"category": "sample_tokens_timeout"})
        elif "engine" in lower and ("dead" in lower or "failed" in lower):
            frames.append({"category": "engine_failure"})
    return frames[-128:]


def journal_snapshot(units: list[str], system: bool, timeout: float = 5) -> dict:
    argv = ["journalctl", "--no-pager", "--output=cat", "--since=-5 min", "--lines=200"]
    if not system:
        argv.append("--user")
    for unit in units:
        argv.extend(["--unit", unit])
    output, status = run_readonly(argv, timeout=timeout)
    return {
        **status,
        "frames": safe_journal_frames(output) if output is not None else [],
    }


def safe_spy_threads(text: str) -> list[dict]:
    """Allowlist JSON fields; discard process command lines and all locals."""
    traces = json.loads(text)
    if not isinstance(traces, list):
        raise ValueError("expected a list of thread traces")

    def is_main(trace):
        return trace.get("thread_name") == "MainThread" or (
            isinstance(trace.get("pid"), int)
            and trace.get("os_thread_id") == trace["pid"]
        )

    # Native dumps can contain hundreds of helper threads. Retain the main
    # Python thread first, then active/GIL-owning threads, before applying caps.
    traces = [trace for trace in traces if isinstance(trace, dict)]
    traces.sort(
        key=lambda trace: (
            not is_main(trace),
            not (trace.get("active") or trace.get("owns_gil")),
        )
    )
    threads = []
    for trace in traces[:128]:
        if not isinstance(trace, dict):
            continue
        thread = {
            key: trace[key]
            for key in ("pid", "thread_id", "os_thread_id", "active", "owns_gil")
            if isinstance(trace.get(key), (int, bool))
        }
        if is_main(trace):
            thread["main_thread"] = True
        frames = []
        for frame in trace.get("frames", [])[:256]:
            if not isinstance(frame, dict):
                continue
            name, filename, line = (
                frame.get("name"),
                frame.get("filename"),
                frame.get("line"),
            )
            if (
                isinstance(name, str)
                and (filename is None or isinstance(filename, str))
                and (line is None or isinstance(line, int))
            ):
                frames.append(
                    {
                        "function": name[:256],
                        "file": Path(filename).name[:256] if filename else None,
                        "line": line,
                    }
                )
        thread["frames"] = frames
        threads.append(thread)
    return threads


def stack_snapshot(
    executable: str | None,
    pids: list[int],
    deadline: float = math.inf,
    sudo: bool = False,
) -> list[dict]:
    if executable is None:
        return [{"available": False, "error_type": "not_requested"}]
    if not pids:
        return [{"available": False, "error_type": "no_target_pid"}]
    results = []
    for pid in pids[:MAX_STACK_TARGETS]:
        argv = [executable, "dump", "--native", "--json", "--pid", str(pid)]
        if sudo:
            argv = ["sudo", "-n", *argv]
        output, status = run_readonly(argv, timeout=min(8, deadline - time.monotonic()))
        threads = []
        if output is not None:
            try:
                threads = safe_spy_threads(output)
            except (ValueError, TypeError):
                status = {"available": False, "error_type": "invalid_stack_json"}
        results.append({"pid": pid, **status, "threads": threads})
    return results


def diagnostic(args, deadline: float) -> dict:
    roots = args.pid
    discovery = None
    if args.service_unit:
        roots, discovery = service_main_pid(
            args.service_unit, min(3, deadline - time.monotonic())
        )
    if args.service_unit and not roots:
        tree = {"available": False, "error_type": "no_service_pid", "processes": []}
    else:
        tree = process_snapshot(roots, min(5, deadline - time.monotonic()))
    # Discover the current service root for each stall, so a restart cannot
    # leave the watchdog attaching to stale PIDs. Workers precede API/helpers.
    targets = (
        prioritized_stack_targets(tree.get("processes", []), roots) if roots else []
    )
    result = {
        "process_tree": tree,
        "journal": journal_snapshot(
            args.journal_unit, args.system_journal, min(5, deadline - time.monotonic())
        ),
        "python_native_stacks": stack_snapshot(
            args.py_spy, targets, deadline, sudo=args.sudo_py_spy
        ),
        "stack_target_pids": targets[:MAX_STACK_TARGETS],
        "stack_targets_truncated": len(targets) > MAX_STACK_TARGETS,
    }
    if discovery is not None:
        result["service_discovery"] = discovery
    return result


def private_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("output must be a regular file")
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)
    except BaseException:
        os.close(descriptor)
        raise


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("seconds must be positive and finite")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-url", type=local_metrics_url, default="http://127.0.0.1:8001/metrics"
    )
    parser.add_argument("--interval", type=positive_seconds, default=5.0)
    parser.add_argument("--stall-seconds", type=positive_seconds, default=45.0)
    parser.add_argument("--duration", type=positive_seconds, default=3600.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".cache/slimserve/progress-watchdog/progress.jsonl",
    )
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument(
        "--service-unit",
        help="user service whose current MainPID roots each stall snapshot",
    )
    roots.add_argument(
        "--pid",
        type=int,
        action="append",
        default=[],
        help="process-tree root and optional py-spy target; repeat up to four",
    )
    parser.add_argument("--journal-unit", action="append", default=[])
    parser.add_argument("--system-journal", action="store_true")
    parser.add_argument(
        "--py-spy",
        help="optional executable path; dumps Python/native frames without locals",
    )
    parser.add_argument(
        "--sudo-py-spy",
        action="store_true",
        help="run only py-spy with sudo -n; requires --py-spy and a process root",
    )
    args = parser.parse_args(argv)
    if args.sudo_py_spy and (not args.py_spy or not (args.pid or args.service_unit)):
        parser.error("--sudo-py-spy requires --py-spy and --pid or --service-unit")
    if args.duration > 3600 or args.interval > 60:
        parser.error("duration must be <=3600 seconds and interval <=60 seconds")
    if len(args.pid) > 4 or any(pid <= 0 for pid in args.pid):
        parser.error("provide at most four positive process IDs")
    detector = ProgressDetector(args.stall_seconds)
    started = time.monotonic()
    deadline = started + args.duration
    with private_output(args.output) as output:
        while time.monotonic() - started < args.duration:
            tick = time.monotonic()
            error = None
            try:
                sample = fetch_metrics(
                    args.metrics_url, timeout=max(0.001, min(3, deadline - tick))
                )
            except (OSError, ValueError, urllib.error.URLError) as exc:
                sample = None
                error = type(exc).__name__
            now = time.monotonic()
            state = detector.update(now, sample)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(now - started, 3),
                "metrics": asdict(sample) if sample is not None else None,
                **state,
            }
            if error:
                record["metrics_error_type"] = error
            if state["diagnose"]:
                record["diagnostic"] = diagnostic(args, deadline)
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            remaining = args.duration - (time.monotonic() - started)
            pause = min(args.interval - (time.monotonic() - tick), remaining)
            if pause > 0:
                time.sleep(pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
