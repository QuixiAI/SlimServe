#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Join isolated Marlin counter rows to the exact recorded routing cases.

Cold-cache measurements are not end-to-end throughput or a serving ceiling.
Missing metrics, units, launches or incomplete case records are hard errors.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = {
    "gpu__time_duration.sum": "us",
    "dram__bytes_op_read.sum": "byte",
    "dram__bytes_op_write.sum": "byte",
    "lts__t_sectors_op_read.sum": "sector",
    "smsp__inst_executed.sum": "inst",
}
UNIT_SCALES = {
    "us": {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6},
    "byte": {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9},
    "sector": {"sector": 1.0},
    "inst": {"inst": 1.0},
}


def analyze(raw, cases):
    with raw.open() as stream:
        reader = csv.DictReader(stream)
        units = next(reader, None)
        rows = list(reader)
    if units is None or any(
        units.get(name) not in UNIT_SCALES[unit] for name, unit in METRICS.items()
    ):
        raise ValueError(f"missing metrics or unexpected units; require {METRICS}")
    factors = {name: UNIT_SCALES[unit][units[name]] for name, unit in METRICS.items()}
    if cases["status"] != "complete" or len(rows) != 2 * len(cases["cases"]):
        raise ValueError("incomplete cases or unexpected number of Marlin launches")
    measured, groups = [], defaultdict(list)
    for i, row in enumerate(rows):
        if int(row["ID"]) != i or "Marlin<" not in row["Kernel Name"]:
            raise ValueError(
                "counter order/name does not match the bracketed execution"
            )
        values = {
            name: float(row[name].replace(",", "")) * factors[name] for name in METRICS
        }
        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("invalid counter value")
        us = values["gpu__time_duration.sum"]
        if us <= 0:
            raise ValueError("nonpositive kernel duration")
        case = cases["cases"][i // 2]
        gate_up = i % 2 == 0
        n, k = (1024, 4096) if gate_up else (4096, 512)
        unique_bytes = case["unique_experts"] * n * k * (0.5 + 1 / 16)
        item = {
            "case": case["label"],
            "phase": "gate_up" if gate_up else "down",
            "batch": case["batch"],
            "layer": case["layer"],
            "kernel_id": i,
            "unique_experts": case["unique_experts"],
            "kernel_name": row["Kernel Name"],
            "unique_weight_footprint_bytes": unique_bytes,
            "dram_read_bytes": values["dram__bytes_op_read.sum"],
            "dram_write_bytes": values["dram__bytes_op_write.sum"],
            "l2_read_sectors": values["lts__t_sectors_op_read.sum"],
            "warp_instructions": values["smsp__inst_executed.sum"],
            "profiled_us": us,
            "dram_read_to_unique_weight_footprint": values["dram__bytes_op_read.sum"]
            / unique_bytes,
            "l2_read_bytes_to_unique_weight_footprint": values[
                "lts__t_sectors_op_read.sum"
            ]
            * 32
            / unique_bytes,
            "profiled_dram_read_GB_per_s": values["dram__bytes_op_read.sum"]
            / us
            / 1000,
        }
        measured.append(item)
        groups[(case["batch"], item["phase"])].append(item)
    summary = []
    for (batch, phase), items in sorted(groups.items()):
        entry = {"batch": batch, "phase": phase, "count": len(items)}
        for key in (
            "profiled_us",
            "dram_read_to_unique_weight_footprint",
            "l2_read_bytes_to_unique_weight_footprint",
            "profiled_dram_read_GB_per_s",
        ):
            values = [r[key] for r in items]
            entry[key] = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
        summary.append(entry)
    return {"diagnostic_only": True, "rows": measured, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = analyze(args.raw, json.loads(args.cases.read_text()))
    result["inputs"] = {"raw": str(args.raw), "cases": str(args.cases)}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
