#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-working-set FP8 shared-expert launch sweep, not serving performance.

Same installed baseline, actual TP4 checkpoint bytes and BF16 inputs as the
cache control. Both A/A2 and B rotate distinct weight allocations exceeding
three SM120 L2 capacities. Sweep existing kernel NT/warp/stage settings only.
Independent FP64 oracle: NRMS <= .004 and per-row peak-normalized error <= 2^-7;
every graph output must equal its eager variant on initial and changed inputs.
All prescribed configurations and A/B/A samples are recorded. A candidate must
also pass a routed-expert contention test and real-profile validation before
any serving change: small isolated shared-expert wins can be completely hidden.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_fp8_cache import (
    digest_tensor,
    load_weights,
    oracle_weight,
    rotation_count,
)
from benchmarks.kernels.benchmark_mhc_output_parallel import build

CONFIGS = tuple((n, w, s) for n in (8, 16, 32) for w in (4, 8) for s in (4, 8))


def parse_config(text):
    try:
        config = tuple(map(int, text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NT,WARPS,STAGES") from error
    if config not in CONFIGS:
        raise argparse.ArgumentTypeError("configuration is not in the fixed probe set")
    return config


def oracle_errors(value, expected):
    if value.shape != expected.shape:
        raise ValueError("oracle output shapes differ")
    if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
        return {
            "finite": False,
            "normalized_rms": None,
            "row_peak_error": None,
            "passed": False,
        }
    delta = value.double() - expected
    result = {
        "finite": True,
        "normalized_rms": float(
            delta.square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-30)
        ),
        "row_peak_error": float(
            (delta.abs() / expected.abs().amax(1, keepdim=True).clamp_min(1e-30)).max()
        ),
    }
    result["passed"] = (
        result["finite"]
        and result["normalized_rms"] <= 0.004
        and result["row_peak_error"] <= 2**-7
    )
    return result


def capture(call, x, banks, expected):
    for _ in range(3):
        call(x, *banks[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [call(x, q, scale) for q, scale in banks]
    graph.replay()
    if not torch.equal(
        torch.stack(outputs), expected.expand(len(banks), *expected.shape)
    ):
        raise ValueError("initial rotation graph disagrees with its eager variant")
    return graph, outputs


def measure(graph, copies, replays):
    for _ in range(3):
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (copies * replays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 23, 44])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--configs", type=parse_config, nargs="+", default=CONFIGS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=16)
    args = parser.parse_args()
    if (
        not 0 <= args.rank < 4
        or min(args.rounds, args.replays) < 1
        or any(not 3 <= layer <= 44 for layer in args.layers)
        or any(not 1 <= batch <= 16 for batch in args.batches)
        or len(set(args.configs)) != len(args.configs)
    ):
        parser.error("TP4 rank, target layers/batches, positive counts, unique configs")
    if not args.build_only:
        if args.output is None or args.output.exists():
            parser.error("a run requires a new output directory")
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            parser.error(f"GPU compute processes already active: {active}")
    extension = build(args.build_dir, name="fp8_launch_probe")
    if args.build_only:
        return
    args.output.mkdir(parents=True)
    record = args.output / "summary.json"
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        Path(__file__).with_name("fp8_launch_probe.cu"),
        Path(__file__).with_name("benchmark_glm53_fp8_cache.py"),
        Path(__file__).with_name("benchmark_mhc_output_parallel.py"),
        root / "csrc/quixicore/serving/fp8_decode_gemm.cuh",
        root / "csrc/quixicore/serving/bf16_decode_gemm.cuh",
    ]
    result = {
        "status": "running",
        "diagnostic_only": True,
        "method": __doc__,
        "command": sys.argv,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "extension_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "weights": [],
        "baseline_checks": [],
        "cases": [],
    }

    def save():
        record.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        import vllm._quixicore_C as native

        from vllm.quixicore.ops import quixicore_ops as qc

        torch.set_num_threads(4)
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("requires the SM120 target with 128 MiB L2")
        result["gpu"] = torch.cuda.get_device_name()
        result["resources"] = extension.resources()
        if {tuple(r["config"]) for r in result["resources"]} != set(CONFIGS):
            raise ValueError("compiled and requested configuration sets disagree")
        native_path = Path(native.__file__).resolve()
        result["native_path"] = str(native_path)
        with native_path.open("rb") as handle:
            result["native_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        for layer in args.layers:
            weights = load_weights(
                args.model / "fp8-swapset.safetensors", layer, args.rank
            )
            for kind, (cpu_q, cpu_s) in weights.items():
                count = rotation_count(cpu_q.numel() * cpu_q.element_size())
                result["weights"].append(
                    {
                        "layer": layer,
                        "kind": kind,
                        "rank": args.rank,
                        "shape": list(cpu_q.shape),
                        "weight_sha256": digest_tensor(cpu_q),
                        "scale_sha256": digest_tensor(cpu_s),
                        "copies": count,
                        "weight_bytes": count * cpu_q.numel() * cpu_q.element_size(),
                    }
                )
                save()
                banks = [(cpu_q.cuda(), cpu_s.cuda()) for _ in range(count)]
                if len({q.data_ptr() for q, _ in banks}) != count:
                    raise ValueError("weight rotation aliases allocations")
                dequant = oracle_weight(cpu_q, cpu_s)
                for batch in args.batches:
                    generator = torch.Generator().manual_seed(3100 + layer * 31 + batch)
                    cpu_x = torch.randn(
                        batch, cpu_q.shape[1], generator=generator
                    ).bfloat16()
                    x = cpu_x.cuda()
                    expected = cpu_x.double() @ dequant.t()
                    baseline_row = {
                        "layer": layer,
                        "kind": kind,
                        "batch": batch,
                        "input_sha256": digest_tensor(cpu_x),
                        "status": "checking",
                    }
                    result["baseline_checks"].append(baseline_row)
                    save()
                    installed = qc.decode_gemm_fp8(x, *banks[0])
                    baseline_errors = oracle_errors(installed.cpu(), expected)
                    baseline_row["oracle"] = baseline_errors
                    if not baseline_errors["passed"]:
                        baseline_row["status"] = "failed"
                        raise ValueError(
                            f"installed baseline fails oracle: {baseline_errors}"
                        )
                    baseline_row["isolated_auto_exact"] = torch.equal(
                        installed, extension.run(x, *banks[0], 0, 0, 0)
                    )
                    if not baseline_row["isolated_auto_exact"]:
                        baseline_row["status"] = "failed"
                        raise ValueError(
                            "isolated auto config differs from installed serving"
                        )
                    baseline_row["status"] = "complete"
                    save()
                    a_graph, a_outputs = capture(
                        qc.decode_gemm_fp8, x, banks, installed
                    )
                    for config in args.configs:
                        x.copy_(cpu_x)

                        def candidate(activation, q, scales, config=config):
                            return extension.run(activation, q, scales, *config)

                        eager = candidate(x, *banks[0])
                        row = {
                            "layer": layer,
                            "kind": kind,
                            "batch": batch,
                            "config": config,
                            "input_sha256": digest_tensor(cpu_x),
                            "status": "checking",
                            "baseline_oracle": baseline_errors,
                            "candidate_oracle": oracle_errors(eager.cpu(), expected),
                            "rounds": [],
                        }
                        result["cases"].append(row)
                        save()
                        if not row["candidate_oracle"]["passed"]:
                            raise ValueError("candidate fails independent FP64 oracle")
                        b_graph, b_outputs = capture(candidate, x, banks, eager)
                        for repeat in range(args.rounds):
                            x.copy_(cpu_x if repeat % 2 == 0 else -cpu_x)
                            sample = {"repeat": repeat + 1}
                            row["rounds"].append(sample)
                            for phase, graph, outputs, reference in (
                                ("A", a_graph, a_outputs, installed),
                                ("B", b_graph, b_outputs, eager),
                                ("A2", a_graph, a_outputs, installed),
                            ):
                                sample[phase] = measure(graph, count, args.replays)
                                ref = reference if repeat % 2 == 0 else -reference
                                if not torch.equal(
                                    torch.stack(outputs), ref.expand(count, *ref.shape)
                                ):
                                    raise ValueError(
                                        "changed-input graph/eager mismatch"
                                    )
                                save()
                        row["median_us"] = {
                            phase: statistics.median(r[phase] for r in row["rounds"])
                            for phase in ("A", "B", "A2")
                        }
                        row["median_paired_throughput_change_pct"] = statistics.median(
                            (0.5 * (r["A"] + r["A2"]) / r["B"] - 1) * 100
                            for r in row["rounds"]
                        )
                        row["status"] = "complete"
                        save()
                        print(json.dumps(row), flush=True)
                        del b_graph, b_outputs, graph, outputs
                    del a_graph, a_outputs
                del banks, dequant
        result["status"] = "complete"
    except BaseException as error:
        result["status"], result["error"] = "failed", repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
