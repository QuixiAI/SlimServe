#!/usr/bin/env python3
"""Classify warm DSV4 CUDA-graph work for a TP2/TP4 scaling comparison."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def _category(event: dict) -> str:
    name = event["name"]
    grid_x = event.get("args", {}).get("grid", [0])[0]

    if "fused_allreduce_post_pre_urgent" in name:
        return "mhc_urgent_collective_transition"
    if "fused_post_pre_deferred" in name:
        return "mhc_deferred_replicated"
    if "aligned_q8_0_q8_1_gemv_kernel" in name:
        rows_match = re.search(r"gemv_kernel<[^,]+, (\d+),", name)
        # The 1536x4096 fused_wqa_wkv is explicitly disable_tp=True. The
        # remaining aligned-Q8 shapes are TP-sharded attention/shared outputs.
        if rows_match and rows_match.group(1) == "1" and grid_x == 1536:
            return "replicated_fixed_q8_projection"
        return "tp_sharded_quantized_linear_moe"
    if any(
        fragment in name
        for fragment in (
            "iq2_xxs_gate_up",
            "q2_k_down_sum_repacked",
            "grouped_q8_0_q8_1",
            "q8_0_gate_up_swiglu",
        )
    ):
        return "tp_sharded_quantized_linear_moe"
    if any(
        fragment in name
        for fragment in (
            "mla_decode_fp8_v",
            "dsv4_attention_reduce_active_channels",
            "fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel",
        )
    ):
        return "tp_sharded_attention_intended"
    if any(
        fragment in name
        for fragment in (
            "indexer_paged_mqa_logits",
            "persistent_topk_kernel",
            "sparse_topk_tlen",
            "dsv4_projection::bf16_fp32_gemv",
            "dsv4_router::bf16",
            "_dsv4_topk_kernel",
            "_save_partial_states_kernel",
            "_fused_indexer_q_rope_quant_kernel",
            "_fused_kv_compress_norm_rope_insert_indexer_attn",
        )
    ):
        return "replicated_indexer_router_projection"
    return "other_graph_work"


def _load_run(trace_dir: Path, graph_id: int) -> dict:
    rank_files = sorted(trace_dir.glob("*rank*.json"))
    if not rank_files:
        raise ValueError(f"no rank trace JSON files found under {trace_dir}")

    rank_results = []
    replay_counts = []
    node_counts = []
    for path in rank_files:
        with path.open() as trace_file:
            events = [
                event
                for event in json.load(trace_file)["traceEvents"]
                if event.get("cat") == "kernel"
                and event.get("args", {}).get("graph id") == graph_id
            ]
        if not events:
            raise ValueError(f"graph {graph_id} has no kernel events in {path}")

        occurrences = defaultdict(int)
        for event in events:
            occurrences[event["args"]["graph node id"]] += 1
        replay_count_set = set(occurrences.values())
        if len(replay_count_set) != 1:
            raise ValueError(
                f"graph nodes in {path} have inconsistent replay counts: "
                f"{sorted(replay_count_set)}"
            )
        replay_count = replay_count_set.pop()
        replay_counts.append(replay_count)
        node_counts.append(len(occurrences))

        categories = defaultdict(float)
        for event in events:
            categories[_category(event)] += event["dur"] / replay_count
        rank_results.append(categories)

    if len(set(replay_counts)) != 1 or len(set(node_counts)) != 1:
        raise ValueError(
            f"rank traces disagree: replay_counts={replay_counts}, "
            f"node_counts={node_counts}"
        )

    categories = sorted(set().union(*rank_results))
    mean_us = {
        category: statistics.mean(rank[category] for rank in rank_results)
        for category in categories
    }
    return {
        "rank_count": len(rank_files),
        "replay_count": replay_counts[0],
        "nodes_per_replay": node_counts[0],
        "mean_kernel_us_per_replay": mean_us,
        "mean_total_kernel_us_per_replay": sum(mean_us.values()),
    }


def _build_report(args: argparse.Namespace) -> dict:
    tp2 = _load_run(args.tp2_trace_dir, args.graph_id)
    tp4 = _load_run(args.tp4_trace_dir, args.graph_id)
    tp2_ms = 1000.0 / args.tp2_tps
    tp4_ms = 1000.0 / args.tp4_tps
    target_tps = 1.5 * args.tp2_tps
    target_ms = 1000.0 / target_tps

    # Fit T(tp) = replicated + parallel_work / tp to the exact TP2/TP4 runs.
    parallel_work_ms = 4.0 * (tp2_ms - tp4_ms)
    replicated_ms = tp2_ms - parallel_work_ms / 2.0

    comparisons = {}
    for category in sorted(
        set(tp2["mean_kernel_us_per_replay"])
        | set(tp4["mean_kernel_us_per_replay"])
    ):
        tp2_us = tp2["mean_kernel_us_per_replay"].get(category, 0.0)
        tp4_us = tp4["mean_kernel_us_per_replay"].get(category, 0.0)
        comparisons[category] = {
            "tp2_us": tp2_us,
            "tp4_us": tp4_us,
            "delta_us": tp4_us - tp2_us,
            "tp2_over_tp4": tp2_us / tp4_us if tp4_us else None,
        }

    return {
        "graph_id": args.graph_id,
        "tp2_trace_dir": str(args.tp2_trace_dir),
        "tp4_trace_dir": str(args.tp4_trace_dir),
        "tp2": tp2,
        "tp4": tp4,
        "category_comparison": comparisons,
        "exact_throughput": {
            "tp2_tps": args.tp2_tps,
            "tp4_tps": args.tp4_tps,
            "speedup": args.tp4_tps / args.tp2_tps,
            "minimum_tp4_tps": target_tps,
            "tp2_ms_per_token": tp2_ms,
            "tp4_ms_per_token": tp4_ms,
            "minimum_tp4_ms_per_token": target_ms,
            "missing_tp4_ms_per_token": tp4_ms - target_ms,
        },
        "amdahl_fit": {
            "replicated_ms": replicated_ms,
            "parallel_work_ms": parallel_work_ms,
            "replicated_fraction_of_tp2": replicated_ms / tp2_ms,
            "replicated_fraction_of_tp4": replicated_ms / tp4_ms,
            "maximum_tp2_to_infinite_tp_speedup": tp2_ms / replicated_ms,
        },
    }


def _print_report(report: dict) -> None:
    exact = report["exact_throughput"]
    fit = report["amdahl_fit"]
    print(
        f"exact: TP2={exact['tp2_tps']:.3f} tok/s, "
        f"TP4={exact['tp4_tps']:.3f} tok/s, "
        f"speedup={exact['speedup']:.3f}x"
    )
    print(
        f"gate: TP4>={exact['minimum_tp4_tps']:.3f} tok/s "
        f"({exact['minimum_tp4_ms_per_token']:.3f} ms/token); "
        f"missing={exact['missing_tp4_ms_per_token']:.3f} ms/token"
    )
    print(
        f"Amdahl fit: replicated={fit['replicated_ms']:.3f} ms, "
        f"parallel_work={fit['parallel_work_ms']:.3f} ms, "
        f"replicated={100 * fit['replicated_fraction_of_tp4']:.1f}% of TP4"
    )
    print()
    print(
        "category                                      TP2 us     TP4 us"
        "      delta   speedup"
    )
    for category, values in report["category_comparison"].items():
        speedup = values["tp2_over_tp4"]
        print(
            f"{category:45} {values['tp2_us']:9.3f} "
            f"{values['tp4_us']:9.3f} {values['delta_us']:+10.3f} "
            f"{speedup:8.3f}x"
        )
    print()
    for label in ("tp2", "tp4"):
        run = report[label]
        print(
            f"{label.upper()}: ranks={run['rank_count']}, "
            f"replays={run['replay_count']}, "
            f"nodes/replay={run['nodes_per_replay']}, "
            f"mean aggregate kernel={run['mean_total_kernel_us_per_replay']:.3f} us"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp2-trace-dir", type=Path, required=True)
    parser.add_argument("--tp4-trace-dir", type=Path, required=True)
    parser.add_argument("--graph-id", type=int, default=26)
    parser.add_argument("--tp2-tps", type=float, required=True)
    parser.add_argument("--tp4-tps", type=float, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = _build_report(args)
    _print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as output_file:
            json.dump(report, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
