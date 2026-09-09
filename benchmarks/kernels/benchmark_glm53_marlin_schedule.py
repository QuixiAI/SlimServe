#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated A/B/A of existing Marlin launch controls, never a serving benchmark.

Uses actual TP4 quant bytes and predetermined captured routes. Activations are
synthetic; down inputs come from the unchanged serving gate/up and activation.
Rotates distinct layers/routes rather than timing one cache-hot expert matrix.
No quantization, native binary, serving dispatcher or clock changes are made.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import torch

from benchmarks.kernels.profile_glm53_marlin import load_layer, route_cases


def parse_config(value):
    if value == "auto":
        return (-1, -1, -1)
    try:
        k, n, blocks = map(int, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected K,N,blocks or auto") from error
    if (k, n) not in ((64, 128), (128, 64), (128, 128), (64, 256)):
        raise argparse.ArgumentTypeError("tile is not in the generated kernel set")
    if not 1 <= blocks <= 4:
        raise argparse.ArgumentTypeError("blocks must be 1..4")
    return k, n, blocks


def footprint(cases, phase):
    if phase == "moe":
        return footprint(cases, "gate_up") + footprint(cases, "down")
    if phase not in ("gate_up", "down"):
        raise ValueError("unknown phase")
    experts = defaultdict(set)
    for case in cases:
        experts[case["layer"]].update(e for row in case["expert_ids"] for e in row)
    n, k = (1024, 4096) if phase == "gate_up" else (4096, 512)
    return sum(map(len, experts.values())) * n * k * 9 // 16


def prepare_case(weights, batch, record, layer_id, gemm=None):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.scalar_type import scalar_types

    if gemm is None:
        gemm = ops.moe_wna16_marlin_gemm

    ids_cpu = [r[layer_id - 3] for r in record["routes"]]
    if len(ids_cpu) != batch or any(
        len(row) != 8 or len(set(row)) != 8 or min(row) < 0 or max(row) >= 288
        for row in ids_cpu
    ):
        raise ValueError("invalid captured expert IDs")
    ids = torch.tensor(ids_cpu, device="cuda", dtype=torch.int32)
    sorted_ids, expert_ids, padded = moe_align_block_size(ids, 8, 288)
    topk_weights = torch.full((batch, 8), 2.5 / 8, device="cuda")
    x = torch.randn(batch, 4096, device="cuda", dtype=torch.bfloat16)
    gate = torch.empty(batch * 8, 1024, device="cuda", dtype=torch.bfloat16)
    down_input = torch.empty(batch * 8, 512, device="cuda", dtype=torch.bfloat16)
    down = torch.empty(batch * 8, 4096, device="cuda", dtype=torch.bfloat16)
    w13, s13, g13, w2, s2, g2, workspace = weights

    def call(phase, config):
        if phase == "moe":
            call("gate_up", config[0])
            apply_moe_activation(MoEActivation.SILU, down_input, gate, clamp_limit=10.0)
            return call("down", config[1])
        is_gate = phase == "gate_up"
        return gemm(
            x if is_gate else down_input,
            gate if is_gate else down,
            w13 if is_gate else w2,
            None,
            s13 if is_gate else s2,
            None,
            g13 if is_gate else g2,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            padded,
            topk_weights,
            moe_block_size=8,
            top_k=8 if is_gate else 1,
            mul_topk_weights=not is_gate,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=batch if is_gate else batch * 8,
            size_n=1024 if is_gate else 4096,
            size_k=4096 if is_gate else 512,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
            thread_k=config[0],
            thread_n=config[1],
            blocks_per_sm=config[2],
        )

    call("gate_up", (-1, -1, -1))
    apply_moe_activation(MoEActivation.SILU, down_input, gate, clamp_limit=10.0)
    refs = {phase: call(phase, (-1, -1, -1)).clone() for phase in ("gate_up", "down")}
    refs["moe"] = refs["down"]
    return {
        "call": call,
        "refs": refs,
        "input": x,
        "routing_weights": topk_weights,
        "outputs": {
            "gate_up": gate,
            "activation": down_input,
            "down": down,
            "moe": down,
        },
        "identity": {
            "layer": layer_id,
            "batch": batch,
            "step": record["step"],
            "expert_ids": ids_cpu,
            "padded_rows": padded.item(),
        },
    }


def correctness(cases, phase, config):
    rows = []
    for case in cases:
        value = case["call"](phase, config)
        ref = case["refs"][phase]
        error = value.float() - ref.float()
        row = {
            "layer": case["identity"]["layer"],
            "step": case["identity"]["step"],
            "max_abs": error.abs().max().item(),
            "normalized_rms": (
                error.square().mean().sqrt()
                / ref.float().square().mean().sqrt().clamp_min(1e-12)
            ).item(),
            "changed_fraction": (value != ref).float().mean().item(),
            "finite": bool(torch.isfinite(value).all()),
        }
        rows.append(row)
        # This is a local comparison against installed auto scheduling, not
        # an independent quant oracle or end-to-end quality qualification.
        torch.testing.assert_close(value, ref, rtol=0.01, atol=0.01)
    return rows


def capture(cases, phase, config):
    for case in cases:
        case["call"](phase, config)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            case["call"](phase, config)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(graph, count, replays):
    for _ in range(3):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (replays * count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[3, 9, 15, 21, 27, 33, 39, 44]
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--cases-per-batch", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Test the composed gate/up, activation and down path",
    )
    parser.add_argument("--gate-config", type=parse_config, default=(128, 64, 1))
    parser.add_argument("--down-config", type=parse_config, default=(64, 128, 2))
    parser.add_argument(
        "--configs",
        type=parse_config,
        nargs="+",
        default=[
            (k, n, b)
            for k, n in ((64, 128), (128, 64), (128, 128), (64, 256))
            for b in (1, 2, 3, 4)
        ],
    )
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.rank < 4:
        parser.error("output must be new and rank must be 0..3")
    if (
        min(args.cases_per_batch, args.rounds, args.replays) < 1
        or any(not 3 <= layer <= 44 for layer in args.layers)
        or any(not 1 <= b <= 16 for b in args.batch)
    ):
        parser.error("positive counts, target layers 3..44, batches 1..16 only")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs already have compute processes: {active}")
    selected = route_cases(args.journal, args.batch, args.cases_per_batch)
    result = {
        "status": "running",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "journal_sha256": hashlib.sha256(args.journal.read_bytes()).hexdigest(),
        "layers": {},
        "cases": [],
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        torch.manual_seed(103)
        prepared = {}
        for layer in args.layers:
            weights, identity = load_layer(args.model, layer, args.rank)
            result["layers"][layer] = identity
            for batch, record in selected:
                case = prepare_case(weights, batch, record, layer)
                prepared[(batch, record["step"], layer)] = case
                result["cases"].append(case["identity"])
            save()
            print(f"prepared layer {layer}", flush=True)
        for batch in args.batch:
            cases = [
                prepared[(b, record["step"], layer)]
                for b, record in selected
                if b == batch
                for layer in args.layers
            ]
            phase_configs = (
                {"moe": [(args.gate_config, args.down_config)]}
                if args.pipeline
                else {"gate_up": args.configs, "down": args.configs}
            )
            for phase, configs in phase_configs.items():
                auto = ((-1, -1, -1), (-1, -1, -1)) if phase == "moe" else (-1, -1, -1)
                baseline = capture(cases, phase, auto)
                for config in configs:
                    row = {
                        "batch": batch,
                        "phase": phase,
                        "config": config,
                        "unique_weight_footprint_bytes": footprint(
                            [c["identity"] for c in cases], phase
                        ),
                        "status": "running",
                        "samples": [],
                    }
                    result["measurements"].append(row)
                    save()
                    try:
                        row["correctness"] = correctness(cases, phase, config)
                        candidate = capture(cases, phase, config)
                    except RuntimeError as error:
                        # Invalid shared-memory/tile configurations are explicit
                        # rejections, not silently missing sweep points.
                        if "Invalid thread config" not in str(
                            error
                        ) and "Unsupported shapes" not in str(error):
                            raise
                        row.update(status="unsupported", error=str(error))
                        save()
                        continue
                    for _ in range(args.rounds):
                        row["samples"].append(
                            {
                                name: measure(graph, len(cases), args.replays)
                                for name, graph in (
                                    ("a_before_us", baseline),
                                    ("b_us", candidate),
                                    ("a_after_us", baseline),
                                )
                            }
                        )
                        save()
                    row["median_us"] = {
                        key: statistics.median(s[key] for s in row["samples"])
                        for key in ("a_before_us", "b_us", "a_after_us")
                    }
                    row["median_paired_speedup"] = statistics.median(
                        (s["a_before_us"] + s["a_after_us"]) / (2 * s["b_us"])
                        for s in row["samples"]
                    )
                    row["status"] = "complete"
                    save()
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ("correctness", "samples")
                            }
                        ),
                        flush=True,
                    )
                    del candidate
                del baseline
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
