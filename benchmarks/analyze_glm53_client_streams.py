#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explain exact-token client timing windows without replacing benchmark TPS.

The recorded client_decode_tps spans the earliest first token to the latest
last token: staggered cold prefill can occur inside it. Report every request's
first/last arrival and inter-chunk gaps, plus the intersection in which all
requests have begun and none has finished. Intersection arrival rate is an
observational diagnostic, NOT engine/GPU throughput or a baseline substitute.
No samples, slow gaps, requests or starts are selected away from the source.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(values):
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def analyze(data):
    rows = data["requests"]
    if not rows:
        raise ValueError("no requests")
    arrivals = []
    for row in rows:
        chunks = row["chunks"]
        times = [row["start"] + chunk["seconds"] for chunk in chunks]
        if (
            len(times) < 2
            or any(not math.isfinite(t) for t in times)
            or any(b < a for a, b in zip(times, times[1:]))
            or not math.isclose(times[0], row["first"], rel_tol=0, abs_tol=1e-6)
            or not math.isclose(times[-1], row["last"], rel_tol=0, abs_tol=1e-6)
            or times[0] < row["start"]
        ):
            raise ValueError("invalid or inconsistent chunk timestamps")
        counts = [chunk["tokens"] for chunk in chunks]
        if (
            any(type(n) is not int or n <= 0 for n in counts)
            or sum(counts) != row["usage"]["completion_tokens"]
        ):
            raise ValueError("chunk token counts disagree with usage")
        arrivals.append(times)
    origin = min(row["start"] for row in rows)
    begin = max(times[0] for times in arrivals)
    end = min(times[-1] for times in arrivals)
    overlap_tokens = [
        sum(
            chunk["tokens"]
            for t, chunk in zip(times, row["chunks"])
            if begin < t <= end
        )
        for times, row in zip(arrivals, rows)
    ]
    details = []
    for index, (row, times) in enumerate(zip(rows, arrivals)):
        gaps = [1000 * (b - a) for a, b in zip(times, times[1:])]
        details.append(
            {
                "request": index,
                "seed": row.get("seed"),
                "cached_tokens": row["usage"]
                .get("prompt_tokens_details", {})
                .get("cached_tokens"),
                "first_from_round_start_s": times[0] - origin,
                "last_from_round_start_s": times[-1] - origin,
                "inter_chunk_ms": summarize(gaps),
                "max_gap_after_chunk": gaps.index(max(gaps)),
                "tokens_in_all_active_intersection": overlap_tokens[index],
            }
        )
    return {
        "diagnostic_only": True,
        "cache_policy": data.get("cache_policy"),
        "concurrency": len(rows),
        "recorded_e2e_tps": data["aggregate_output_tps"],
        "recorded_client_decode_tps": data["client_decode_tps"],
        "first_token_spread_s": begin - min(times[0] for times in arrivals),
        "all_active_intersection": {
            "start_from_round_start_s": begin - origin,
            "end_from_round_start_s": end - origin,
            "duration_s": max(0.0, end - begin),
            "observed_tokens": sum(overlap_tokens),
            "client_arrival_tps": sum(overlap_tokens) / (end - begin)
            if end > begin
            else None,
            "boundary_rule": "latest first < event <= earliest last",
        },
        "requests": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in args.receipts:
        raw = path.read_bytes()
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                **analyze(json.loads(raw)),
            }
        )
    with args.output.open("x") as handle:
        json.dump({"method": __doc__, "reports": reports}, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"reports": len(reports), "output": str(args.output)}))


if __name__ == "__main__":
    main()
