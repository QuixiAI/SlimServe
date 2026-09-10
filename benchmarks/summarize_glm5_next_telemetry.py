#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correlate exact-run JSON with passive nvidia-smi samples (not a profiler).

Run windows are approximate: JSON mtime minus client wall duration. Metric
scraping and serialization can shift both boundaries. Idle spans report the
distance between observed zero-utilization samples, not exact idle duration.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def summarize_samples(rows: list[dict], expected_gpus: int = 8) -> dict:
    if not rows:
        return {"samples": 0, "all_gpu_idle_observed_spans": []}
    # Each nvidia-smi polling cycle enumerates devices in index order. Require
    # every device before treating a cycle as evidence of all-GPU idleness.
    cycles: list[list[dict]] = []
    for row in rows:
        if not cycles or row["index"] <= cycles[-1][-1]["index"]:
            cycles.append([])
        cycles[-1].append(row)
    spans: list[dict] = []
    start = end = None

    def flush() -> None:
        if start is not None:
            spans.append(
                {
                    "first_sample_utc": start,
                    "last_sample_utc": end,
                    "observed_span_seconds": end - start,
                }
            )

    for cycle in cycles:
        stamp = cycle[0]["timestamp"]
        idle = {r["index"] for r in cycle} == set(range(expected_gpus)) and all(
            r["utilization"] == 0 for r in cycle
        )
        if not idle or (end is not None and stamp - end > 5):
            flush()
            start = end = None
        if idle:
            if start is None:
                start = stamp
            end = stamp
    flush()
    return {
        "samples": len(rows),
        "gpu_utilization_mean_percent": statistics.mean(r["utilization"] for r in rows),
        "sm_clock_min_mhz": min(r["sm_clock"] for r in rows),
        "sm_clock_max_mhz": max(r["sm_clock"] for r in rows),
        "power_median_watts": statistics.median(r["power"] for r in rows),
        "all_gpu_idle_observed_spans": spans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    samples = []
    with (args.run_root / "gpu-telemetry.csv").open() as source:
        reader = csv.reader(source, skipinitialspace=True)
        next(reader)
        for row in reader:
            if len(row) != 10:
                continue  # A live writer may leave its final line incomplete.
            samples.append(
                {
                    "timestamp": datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f")
                    .replace(tzinfo=timezone.utc)
                    .timestamp(),
                    "index": int(row[1]),
                    "sm_clock": float(row[3].split()[0]),
                    "power": float(row[5].split()[0]),
                    "utilization": float(row[8].split()[0]),
                }
            )
    results = {}
    for path in sorted(args.run_root.glob("*/exact_c*_r*.json")):
        try:
            run = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # Timed run not finished yet.
        end = path.stat().st_mtime
        start = end - run["wall_seconds"]
        rows = [r for r in samples if start <= r["timestamp"] <= end]
        results[str(path.relative_to(args.run_root))] = {
            "aggregate_output_tps": run["aggregate_output_tps"],
            "wall_seconds": run["wall_seconds"],
            "exact": run["exact"],
            "cache_metrics": run.get("cache_metrics"),
            "approximate_start_utc": start,
            "approximate_end_utc": end,
            **summarize_samples(rows, args.gpus),
        }
    print(json.dumps({"window_caveat": __doc__, "runs": results}, indent=2))


if __name__ == "__main__":
    main()
