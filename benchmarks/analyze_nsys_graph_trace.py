#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize Nsight whole-graph spans without inferring kernel busy time.

Input is an SQLite export of a graph-granularity capture, not a node trace.
Keep every graph record, rank/context/stream identity and invalid boundary row.
This measures observed graph latency, not unprofiled latency or recoverable TPS.
"""

import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
        database.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
        if table not in tables and "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
            raise ValueError(f"no whole-graph activity table in {path}")
        rows = (
            [
                dict(row)
                for row in database.execute(
                    "SELECT start, end, deviceId, globalPid, contextId, streamId, "
                    f"graphId, graphExecId, correlationId FROM {table} "
                    "ORDER BY start, deviceId"
                )
            ]
            if table in tables
            else []
        )
        kernel_count = (
            database.execute(
                "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0]
            if "CUPTI_ACTIVITY_KIND_KERNEL" in tables
            else 0
        )
        graph_launches = (
            [
                dict(row)
                for row in database.execute(
                    "SELECT r.start, r.end, r.globalTid, r.correlationId, "
                    "s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                    "JOIN StringIds s ON r.nameId=s.id "
                    "WHERE s.value LIKE 'cudaGraphLaunch%' ORDER BY r.start"
                )
            ]
            if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables and "StringIds" in tables
            else []
        )
        diagnostics = (
            [dict(row) for row in database.execute("SELECT * FROM DIAGNOSTIC_EVENT")]
            if "DIAGNOSTIC_EVENT" in tables
            else []
        )
    groups = defaultdict(list)
    invalid = []
    identity = (
        "deviceId",
        "globalPid",
        "contextId",
        "streamId",
        "graphId",
        "graphExecId",
    )
    for row in rows:
        if row["end"] < row["start"]:
            invalid.append(row)
            continue
        groups[tuple(row[key] for key in identity)].append(row)
    summaries = []
    for key, samples in groups.items():
        durations = [(row["end"] - row["start"]) / 1000 for row in samples]
        summaries.append(
            {
                **dict(zip(identity, key)),
                "count": len(samples),
                "duration_us": {
                    "min": min(durations),
                    "median": statistics.median(durations),
                    "max": max(durations),
                    "mean": statistics.mean(durations),
                },
                "records": samples,
            }
        )
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "status": "complete" if rows and not invalid else "incomplete",
        "graph_record_count": len(rows),
        "missing_graph_activity_table": table not in tables,
        "runtime_graph_launch_count": len(graph_launches),
        "runtime_graph_launches": graph_launches,
        "ordinary_kernel_record_count": kernel_count,
        "collection_diagnostics": diagnostics,
        "invalid_boundary_records": invalid,
        "groups": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"method": __doc__, "reports": [analyze(path) for path in args.database]}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "reports": len(result["reports"]),
                "graph_records": sum(
                    report["graph_record_count"] for report in result["reports"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
