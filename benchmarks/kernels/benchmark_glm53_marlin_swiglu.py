#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4.1: fixed N64 gate/up pairing and fused clamped-SwiGLU epilogue.

Actual TP4 checkpoint bytes and captured routes, synthetic BF16 inputs. Compare
the COMPLETE gate/up, activation, weighted-down path to installed Marlin, not
just a cache-hot activation kernel. One candidate, no launch-parameter sweep.
"""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_marlin_schedule import (
    footprint,
    measure,
    prepare_case,
)
from benchmarks.kernels.profile_glm53_marlin import load_layer, route_cases
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.scalar_type import scalar_types


def build(path):
    from torch.utils.cpp_extension import load

    path.mkdir(parents=True, exist_ok=True)
    return load(
        name="marlin_glm_swiglu",
        sources=[str(Path(__file__).with_name("marlin_glm_swiglu.cu").resolve())],
        extra_include_paths=[str(Path(__file__).resolve().parents[2] / "csrc")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-static-global-template-stub=false",
            "-gencode=arch=compute_120f,code=sm_120f",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        build_directory=str(path),
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--diagnostic-timing",
        action="store_true",
        help="Record all comparison failures and timing; NEVER qualify a candidate",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    extension = build(args.build_dir)
    if args.build_only:
        return
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs busy: {active}")
    torch.manual_seed(1053)
    result = {
        "status": "running",
        "roadmap": "4.1",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "journal_sha256": hashlib.sha256(args.journal.read_bytes()).hexdigest(),
        "weights": {},
        "checks": [],
        "measurements": [],
        "comparison_failures": [],
        "diagnostic_timing": args.diagnostic_timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    auto = (-1, -1, -1)
    cases = {batch: [] for batch in (1, 8, 16)}
    selected = route_cases(args.journal, [1], 15) + route_cases(
        args.journal, [8, 16], 3
    )
    try:
        for layer in (3, 23, 44):
            original, identity = load_layer(args.model, layer, 0)
            paired, paired_identity = load_layer(args.model, layer, 0, gate_up_tile=64)
            assert identity == paired_identity
            result["weights"][layer] = identity
            w13, s13, g13 = paired[:3]
            workspace = paired[-1]
            temp = torch.empty(188 * 4 * 16 * 128, dtype=torch.float32, device="cuda")
            for batch, record in selected:
                case = prepare_case(original, batch, record, layer)

                def candidate(
                    case=case, w13=w13, s13=s13, g13=g13, workspace=workspace, temp=temp
                ):
                    extension.run(
                        case["input"],
                        case["outputs"]["activation"],
                        w13,
                        s13,
                        g13,
                        workspace,
                        temp,
                        *case["alignment"],
                        case["routing_weights"],
                    )
                    return case["call"]("down", auto)

                def control(case=case):
                    return case["call"]("moe", (auto, auto))

                case["candidate"] = candidate
                case["control"] = control
                for changed in (False, True):
                    if changed:
                        # Graph/reused workspace correctness must not depend on
                        # the first input, including the actual clamp branches.
                        case["input"].mul_(32)
                    ref = control().clone()
                    activation_ref = case["outputs"]["activation"].clone()
                    value = candidate()
                    activation = case["outputs"]["activation"]
                    activation_passed = True
                    try:
                        torch.testing.assert_close(
                            activation, activation_ref, rtol=0.01, atol=0.01
                        )
                    except AssertionError as comparison_error:
                        activation_passed = False
                        # Localize a failing element with the SAME paired
                        # weights/schedule but the installed unfused epilogue.
                        # This diagnostic neither relaxes nor replaces the gate.
                        paired_gate = torch.empty_like(case["outputs"]["gate_up"])
                        ops.moe_wna16_marlin_gemm(
                            case["input"],
                            paired_gate,
                            w13,
                            None,
                            s13,
                            None,
                            g13,
                            None,
                            None,
                            None,
                            workspace,
                            *case["alignment"],
                            case["routing_weights"],
                            moe_block_size=8,
                            top_k=8,
                            mul_topk_weights=False,
                            b_q_type=scalar_types.float4_e2m1f,
                            size_m=batch,
                            size_n=1024,
                            size_k=4096,
                            is_k_full=True,
                            use_atomic_add=False,
                            use_fp32_reduce=True,
                            is_zp_float=False,
                            thread_k=128 if batch == 1 else 64,
                            thread_n=64 if batch == 1 else 128,
                            blocks_per_sm=2 if batch == 1 else 3,
                        )
                        restored = paired_gate.view(-1, 16, 2, 32).transpose(1, 2)
                        restored = restored.reshape(-1, 1024).contiguous()
                        paired_activation = torch.empty_like(activation)
                        apply_moe_activation(
                            MoEActivation.SILU,
                            paired_activation,
                            restored,
                            clamp_limit=10.0,
                        )
                        result["failure_fixture"] = {
                            **case["identity"],
                            "changed_input": changed,
                            "fused_vs_paired_activation_exact": torch.equal(
                                activation, paired_activation
                            ),
                            "fused_vs_paired_max_abs": (
                                activation.float() - paired_activation.float()
                            )
                            .abs()
                            .max()
                            .item(),
                        }
                        failure_id = len(result["comparison_failures"])
                        result["comparison_failures"].append(
                            {
                                **result["failure_fixture"],
                                "phase": "activation",
                                "error": str(comparison_error),
                            }
                        )
                        torch.save(
                            {
                                "input": case["input"].cpu(),
                                "original_gate": case["outputs"]["gate_up"].cpu(),
                                "paired_gate": restored.cpu(),
                                "original_activation": activation_ref.cpu(),
                                "paired_activation": paired_activation.cpu(),
                                "fused_activation": activation.cpu(),
                            },
                            args.output.with_suffix(f".failure-{failure_id}.pt"),
                        )
                        if not args.diagnostic_timing:
                            raise
                    output_passed = True
                    try:
                        torch.testing.assert_close(value, ref, rtol=0.01, atol=0.01)
                    except AssertionError as comparison_error:
                        output_passed = False
                        result["comparison_failures"].append(
                            {
                                **case["identity"],
                                "changed_input": changed,
                                "phase": "down",
                                "error": str(comparison_error),
                            }
                        )
                        if not args.diagnostic_timing:
                            raise
                    error = value.float() - ref.float()
                    result["checks"].append(
                        {
                            **case["identity"],
                            "changed_input": changed,
                            "activation_passed": activation_passed,
                            "output_passed": output_passed,
                            "activation_exact": torch.equal(activation, activation_ref),
                            "max_abs": error.abs().max().item(),
                            "nrms": (
                                error.square().mean().sqrt()
                                / ref.float().square().mean().sqrt().clamp_min(1e-12)
                            ).item(),
                        }
                    )
                case["input"].div_(32)
                cases[batch].append(case)
            del original, paired
            gc.collect()
            save()
            print(
                f"layer {layer}: checks recorded; "
                f"{len(result['comparison_failures'])} cumulative comparison failures",
                flush=True,
            )
        for batch, fixtures in cases.items():
            weight_bytes = footprint([c["identity"] for c in fixtures], "moe")
            if weight_bytes <= 3 * 128 * 1024**2:
                raise ValueError(f"insufficient weight footprint: {weight_bytes}")
            graphs = {}
            for arm in ("control", "candidate"):
                for case in fixtures:
                    case[arm]()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for case in fixtures:
                        case[arm]()
                graphs[arm] = graph
            # Replay with changed contents, independently compare the complete
            # captured outputs. Each graph sees the exact same live inputs.
            for case in fixtures:
                case["input"].neg_()
            graphs["control"].replay()
            expected = [c["outputs"]["down"].clone() for c in fixtures]
            graphs["candidate"].replay()
            torch.cuda.synchronize()
            graph_passed = True
            for case, ref in zip(fixtures, expected):
                try:
                    torch.testing.assert_close(
                        case["outputs"]["down"], ref, rtol=0.01, atol=0.01
                    )
                except AssertionError as comparison_error:
                    graph_passed = False
                    result["comparison_failures"].append(
                        {
                            **case["identity"],
                            "phase": "changed_graph",
                            "error": str(comparison_error),
                        }
                    )
                    if not args.diagnostic_timing:
                        raise
            samples = []
            for round_id in range(3):
                row = {"round": round_id}
                for arm, label in (
                    ("control", "before"),
                    ("candidate", "candidate"),
                    ("control", "after"),
                ):
                    row[label] = measure(graphs[arm], len(fixtures), 10)
                samples.append(row)
            medians = {
                key: statistics.median(r[key] for r in samples)
                for key in ("before", "candidate", "after")
            }
            result["measurements"].append(
                {
                    "batch": batch,
                    "fixtures": len(fixtures),
                    "weight_bytes": weight_bytes,
                    "changed_input_graph_parity": graph_passed,
                    "samples_us": samples,
                    "median_us": medians,
                }
            )
            print(f"batch {batch}: {medians}", flush=True)
            save()
        result["status"] = (
            "diagnostic_timing_only" if args.diagnostic_timing else "complete"
        )
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
