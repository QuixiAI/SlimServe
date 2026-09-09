#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attribute GPU work to model steps using host launch/CUPTI correlation.

A GPU kernel can run after its host step has exited. Match its CUDA runtime
or driver launch inside the execute annotation, not GPU timestamps inside
that CPU interval. Keep every annotated step and report unmatched GPU work.
All times are profiler observations in microseconds, not benchmark rates.
"""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.analyze_cuda_graph_trace import busy_union

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def group_durations(events):
    groups = collections.defaultdict(list)
    for event in events:
        groups[(event["cat"], event["name"])].append(event)
    return [
        {
            "category": category,
            "name": name,
            "count": len(rows),
            "duration_sum_us": sum(row["dur"] for row in rows),
        }
        for (category, name), rows in sorted(
            groups.items(), key=lambda item: -sum(e["dur"] for e in item[1])
        )
    ]


def analyze(events):
    steps = sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and e.get("name", "").startswith("execute_")
            and e.get("ph") == "X"
        ),
        key=lambda e: e["ts"],
    )
    launches = {}
    for event in events:
        correlation = event.get("args", {}).get("correlation")
        if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        if correlation is None:
            continue
        matches = [
            i
            for i, step in enumerate(steps)
            if (event["pid"], event["tid"]) == (step["pid"], step["tid"])
            and step["ts"] <= event["ts"] < step["ts"] + step["dur"]
        ]
        if len(matches) > 1:
            raise ValueError("overlapping execute annotations on the launch thread")
        if matches:
            if correlation in launches and launches[correlation] != matches[0]:
                raise ValueError("ambiguous CUPTI correlation across model steps")
            launches[correlation] = matches[0]
    grouped = collections.defaultdict(list)
    unmatched = []
    for event in events:
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        args = event.get("args", {})
        step = launches.get(args.get("correlation"))
        if step is None:
            unmatched.append(event)
        else:
            grouped[(step, args.get("device"))].append(event)
    result = []
    for i, step in enumerate(steps):
        devices = []
        for (index, device), rows in grouped.items():
            if index != i:
                continue
            start = min(e["ts"] for e in rows)
            span = max(e["ts"] + e["dur"] for e in rows) - start
            kernels = [e for e in rows if e["cat"] == "kernel"]
            devices.append(
                {
                    "device": device,
                    "gpu_start_us": start,
                    "gpu_span_us": span,
                    "gpu_busy_union_us": busy_union(rows),
                    "kernel_busy_union_us": busy_union(kernels),
                    "no_gpu_activity_us": span - busy_union(rows),
                    "kernel_count": len(kernels),
                    "operations": group_durations(rows),
                }
            )
        result.append(
            {
                "annotation": step["name"],
                "host_start_us": step["ts"],
                "host_duration_us": step["dur"],
                "devices": devices,
            }
        )
    return {
        "diagnostic_only": True,
        "method": __doc__,
        "steps": result,
        "unmatched_gpu_count": len(unmatched),
        "unmatched_gpu_operations": group_durations(unmatched),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    results = {}
    for path in args.traces:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            result = analyze(json.load(stream)["traceEvents"])
        with path.open("rb") as stream:
            result["source_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        results[str(path)] = result
        print(
            f"{path}: {len(result['steps'])} steps, "
            f"{result['unmatched_gpu_count']} unmatched GPU operations",
            flush=True,
        )
    with args.output.open("x") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
