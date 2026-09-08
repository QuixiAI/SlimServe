#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attribute GPU graph replays by CUPTI correlation, not CPU launch windows.

The CPU can enqueue several replays ahead of the GPU. A host-timestamp window
therefore cannot reliably identify one GPU step. Every observed replay is kept;
node-count differences are reported, never used to discard a measurement.
Timings are profiler observations in microseconds, not serving TPS.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path


def busy_union(events):
    end = float("-inf")
    total = 0.0
    for event in sorted(events, key=lambda e: e["ts"]):
        stop = event["ts"] + event["dur"]
        total += max(0.0, stop - max(end, event["ts"]))
        end = max(end, stop)
    return total


def distribution(values):
    return (
        {
            "count": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "max": max(values),
        }
        if values
        else None
    )


def analyze(events):
    launches = {
        event["args"]["correlation"]: event
        for event in events
        if event.get("name") == "cudaGraphLaunch"
        and "correlation" in event.get("args", {})
    }
    groups = collections.defaultdict(list)
    for event in events:
        args = event.get("args", {})
        if event.get("cat") == "kernel" and args.get("graph id"):
            groups[(args.get("device"), args["graph id"], args["correlation"])].append(
                event
            )
    replays = []
    for (device, graph, correlation), kernels in sorted(groups.items()):
        kernels.sort(key=lambda e: e["ts"])
        start = kernels[0]["ts"]
        span = max(e["ts"] + e["dur"] for e in kernels) - start
        busy = busy_union(kernels)
        streams = collections.defaultdict(list)
        names = collections.defaultdict(list)
        for event in kernels:
            streams[event["args"]["stream"]].append(event)
            names[event["name"]].append(event["dur"])
        gaps = []
        for stream in streams.values():
            gaps.extend(
                b["ts"] - (a["ts"] + a["dur"]) for a, b in zip(stream, stream[1:])
            )
        launch = launches.get(correlation)
        replays.append(
            {
                "device": device,
                "graph_id": graph,
                "correlation": correlation,
                "kernel_count": len(kernels),
                "span_us": span,
                "kernel_busy_union_us": busy,
                "no_kernel_active_us": span - busy,
                "kernel_duration_sum_us": sum(e["dur"] for e in kernels),
                "same_stream_gap_us": distribution(gaps),
                "gpu_start_after_host_launch_us": start - launch["ts"]
                if launch
                else None,
                "kernels": {
                    name: {"count": len(values), "duration_sum_us": sum(values)}
                    for name, values in sorted(names.items())
                },
            }
        )
    summaries = []
    for device, graph in sorted({(r["device"], r["graph_id"]) for r in replays}):
        rows = [r for r in replays if (r["device"], r["graph_id"]) == (device, graph)]
        summaries.append(
            {
                "device": device,
                "graph_id": graph,
                "observed_kernel_counts": dict(
                    collections.Counter(r["kernel_count"] for r in rows)
                ),
                **{
                    key: distribution([r[key] for r in rows])
                    for key in (
                        "span_us",
                        "kernel_busy_union_us",
                        "no_kernel_active_us",
                        "kernel_duration_sum_us",
                    )
                },
            }
        )
    return {"summaries": summaries, "replays": replays}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {}
    for path in args.traces:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            result = analyze(json.load(stream)["traceEvents"])
        results[str(path)] = result
        print(path)
        print(json.dumps(result["summaries"], indent=2))
    if args.output:
        with args.output.open("x") as stream:
            json.dump(results, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
