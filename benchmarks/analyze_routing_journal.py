#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize observed expert reuse in actual concurrent decode steps.

Optional bytes are unique weight footprint, NOT measured DRAM traffic, effective
bandwidth, or a physical serving ceiling. Use hardware counters for those claims.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def distribution(values):
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def analyze(path, per_expert_bytes=None):
    groups = collections.defaultdict(list)
    kinds = collections.Counter()
    header = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["kind"]
            if kind not in {
                "header", "decode", "invalid", "skip", "limit", "step_limit"
            }:
                raise ValueError(f"unsupported schema-1 routing record kind: {kind!r}")
            kinds[kind] += 1
            if kind == "header":
                if header is not None or row["schema"] != 1:
                    raise ValueError("unsupported or repeated journal header")
                header = row
            elif kind == "invalid":
                raise ValueError(f"invalid routing capture retained in {path}")
            elif kind == "decode":
                if header is None:
                    raise ValueError("decode record before header")
                routes = row["routes"]
                batch = len(row["request_ids"])
                layers = header["num_layers"] - header["first_moe_layer"]
                top_k = header["top_k"]
                if len(routes) != batch or any(len(r) != layers for r in routes):
                    raise ValueError("invalid recorded routing dimensions")
                uniques, max_loads, padded = [], [], []
                for layer in range(layers):
                    selected = [r[layer] for r in routes]
                    if any(
                        len(ids) != top_k
                        or len(set(ids)) != top_k
                        or any(not 0 <= e < header["num_experts"] for e in ids)
                        for ids in selected
                    ):
                        raise ValueError("invalid recorded expert IDs")
                    counts = collections.Counter(e for ids in selected for e in ids)
                    uniques.append(len(counts))
                    max_loads.append(max(counts.values()))
                    # Current BF16-input Marlin small-decode tile is eight rows.
                    padded.append(sum((n + 7) // 8 * 8 for n in counts.values()))
                groups[batch].append(
                    {
                        "step": row["step"],
                        "unique_experts_by_layer": uniques,
                        "max_tokens_per_expert_by_layer": max_loads,
                        "marlin_m8_padded_rows_by_layer": padded,
                    }
                )
    if header is None:
        raise ValueError("empty routing journal")
    summary = {}
    for batch, rows in sorted(groups.items()):
        unique_totals = [sum(r["unique_experts_by_layer"]) for r in rows]
        item = {
            "steps": len(rows),
            "unique_experts_per_layer": distribution(
                [v for r in rows for v in r["unique_experts_by_layer"]]
            ),
            "sum_layer_unique_experts": distribution(unique_totals),
            "max_tokens_per_expert": max(
                v for r in rows for v in r["max_tokens_per_expert_by_layer"]
            ),
            "marlin_m8_route_utilization": distribution(
                [
                    batch
                    * header["top_k"]
                    * len(r["unique_experts_by_layer"])
                    / sum(r["marlin_m8_padded_rows_by_layer"])
                    for r in rows
                ]
            ),
        }
        if per_expert_bytes is not None:
            item["unique_weight_footprint_bytes_not_dram"] = distribution(
                [total * per_expert_bytes for total in unique_totals]
            )
        summary[batch] = item
    return {
        "path": str(path),
        "header": header,
        "record_kinds": dict(kinds),
        "assumed_bytes_per_expert_per_layer_per_rank": per_expert_bytes,
        "summary": summary,
        "steps": dict(groups),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-expert-bytes", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.per_expert_bytes is not None and args.per_expert_bytes < 1:
        parser.error("per-expert bytes must be positive")
    result = analyze(args.journal, args.per_expert_bytes)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))
    if not result["summary"]:
        raise SystemExit("no decode observations captured; summary retained")


if __name__ == "__main__":
    main()
