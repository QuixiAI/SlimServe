#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated SM120 mHC prefill tensor-core probe, not a serving operator.

Same losslessly stored BF16 fn/activations, FP32 dot accumulation and existing
20-iteration finalize/apply kernels. Dot summation order changes. Gate every
actual site against installed BF16 serving and sampled independent FP64
partials; require exact fused residual bits and changed-input graph/eager
parity. Timing rotates six copies of all 90 fn matrices (>3x128MiB L2), with
shared immutable activations per bank to bound memory. No serving gain follows
from these isolated graphs; real-profile and sanitizer gates remain required.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_mhc_storage import (
    exact,
    lossless_bf16,
    selected_check_sites,
    timing_rows,
)
from benchmarks.kernels.benchmark_mhc_output_parallel import (
    build,
    checkpoint_parameters,
    compare,
    inputs,
)


def oracle_rows(batch):
    if batch < 64:
        raise ValueError("prefill probe requires at least 64 rows")
    return sorted({0, 1, 15, 16, 31, 32, 63, min(64, batch - 1), batch - 1})


def partial_errors(residual, fn, partial):
    """CPU FP64 oracle with the candidate's explicit 32 stream-slice splits.

    Inputs are sampled *after* the fused residual has passed its bit gate;
    that operation is independently checked against the installed kernel.
    This oracle does not reuse tensor-core or native partial calculations.
    """
    if (
        residual.ndim != 3
        or tuple(residual.shape[1:]) != (4, 4096)
        or tuple(fn.shape) != (24, 16384)
        or tuple(partial.shape) != (residual.shape[0], 32, 25)
    ):
        raise ValueError("invalid mHC oracle shapes")
    rows = oracle_rows(residual.shape[0])
    r = residual[rows].cpu().double().reshape(len(rows), 4, 32, 128)
    r = r.permute(0, 2, 1, 3).reshape(len(rows), 32, 512)
    w = fn.cpu().double().reshape(24, 4, 32, 128)
    w = w.permute(2, 0, 1, 3).reshape(32, 24, 512)
    expected = torch.einsum("tsk,snk->tsn", r, w)
    sq = r.square().sum(-1)
    observed = partial[rows].cpu().double()
    result = {
        "rows": rows,
        "finite": bool(
            torch.isfinite(observed).all()
            and torch.isfinite(r).all()
            and torch.isfinite(w).all()
        ),
    }
    if not result["finite"]:
        return {**result, "passed": False}
    delta = observed[:, :, :24] - expected
    result["dot_normalized_rms"] = float(
        delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)
    )
    result["dot_row_peak_error"] = float(
        (delta.abs() / expected.abs().amax(-1, keepdim=True).clamp_min(1e-30)).max()
    )
    result["square_relative_error"] = float(
        ((observed[:, :, 24] - sq).abs() / sq.clamp_min(1e-30)).max()
    )
    result["passed"] = (
        result["dot_normalized_rms"] <= 2e-6
        and result["dot_row_peak_error"] <= 2e-5
        and result["square_relative_error"] <= 1e-5
    )
    return result


def installed(data, fused):
    from vllm.quixicore.ops import quixicore_ops as qc

    constants = [1e-5, 1e-6, 1e-6, 2.0, 20, None, 0.0]
    if fused:
        return qc.dsv4_mhc_fused_post_pre(*data, *constants)
    _, residual, _, _, fn, scale, base = data
    return [residual, *qc.dsv4_mhc_pre(residual, fn, scale, base, *constants)]


def check_outputs(reference, candidate):
    if len(reference) != 4 or len(candidate) != 4:
        raise ValueError("unexpected operator output count")
    for ref, got in zip(reference, candidate):
        if ref.shape != got.shape or ref.dtype != got.dtype:
            raise ValueError("operator output shape or dtype changed")
    try:
        exact(reference[:1], candidate[:1])
        return compare(reference, candidate)
    except AssertionError as error:
        # Preserve the original gate and report the failed value, not only an
        # empty AssertionError. Compute this extra diagnostic only on failure.
        details = []
        for index, (ref, got) in enumerate(zip(reference, candidate)):
            finite = bool(torch.isfinite(ref).all() and torch.isfinite(got).all())
            row = {"output": index, "dtype": str(ref.dtype), "finite": finite}
            if finite and ref.numel():
                delta = (got.float() - ref.float()).abs()
                peak = ref.float().abs().amax(-1, keepdim=True).clamp_min(1e-6)
                normalized = delta / peak
                flat = int(normalized.argmax())
                row.update(
                    max_abs=float(delta.max()),
                    row_peak_error=float(normalized.max()),
                    max_flat_index=flat,
                    reference=float(ref.flatten()[flat]),
                    candidate=float(got.flatten()[flat]),
                )
            details.append(row)
        raise AssertionError(
            f"{error}; output diagnostics: {json.dumps(details)}"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-sites", type=int, nargs="+")
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[64, 65, 128, 129, 7616]
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=4)
    args = parser.parse_args()
    try:
        check_sites = selected_check_sites(args.check_sites)
    except ValueError as error:
        parser.error(str(error))
    if args.check_sites is not None and not args.check_only:
        parser.error("bounded site selection is for correctness/sanitizers only")
    if min(args.rounds, args.replays) < 1 or any(
        not 64 <= t <= 7616 for t in args.batch
    ):
        parser.error("positive rounds/replays, prefill batches 64..7616")
    settings = {
        "VLLM_DSV4_MHC_MODE": "0",
        "VLLM_DSV4_MHC_COOP_MAX_T": "8",
        "VLLM_DSV4_MHC_SPLITS": "64",
        "VLLM_DSV4_MHC_PREFILL_MIN_T": "64",
    }
    for key, value in settings.items():
        if key in os.environ and os.environ[key] != value:
            parser.error(f"requires {key}={value}")
        os.environ[key] = value
    if not args.build_only:
        if args.output is None or args.output.exists():
            parser.error("a run requires a new output directory")
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            parser.error(f"GPU compute processes already active: {active}")
    extension = build(args.build_dir, name="mhc_prefill_tc_probe")
    if args.build_only:
        return
    args.output.mkdir(parents=True)
    record = args.output / "summary.json"
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        Path(__file__).with_name("mhc_prefill_tc_probe.cu"),
        Path(__file__).with_name("benchmark_glm53_mhc_storage.py"),
        Path(__file__).with_name("benchmark_mhc_output_parallel.py"),
        root / "csrc/quixicore/serving/mhc_ampere.cuh",
        root / "csrc/quixicore/serving/bf16_decode_gemm.cuh",
        root / "csrc/quixicore/tm_cuda/tm_cuda_serving.cu",
    ]
    result = {
        "status": "running",
        "diagnostic_only": True,
        "method": __doc__,
        "command": sys.argv,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "extension_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "environment": settings,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "checks": [],
        "timings": [],
    }

    def save():
        record.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        import vllm._quixicore_C as native

        torch.set_num_threads(4)
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("requires the SM120 target")
        result["gpu"] = torch.cuda.get_device_name()
        result["resources"] = extension.resources()
        with Path(native.__file__).open("rb") as stream:
            result["native_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        sites, result["parameters"] = checkpoint_parameters(args.model)
        if len(sites) != 90:
            raise ValueError("expected all 90 actual parameter sets")
        narrow = [lossless_bf16(site[0]) for site in sites]
        save()
        for batch in args.batch:
            for site in check_sites:
                for magnitude in (0.01, 1.0, 100.0):
                    data = inputs(batch, 301 + site, magnitude)[:7]
                    data[4:] = [narrow[site], *sites[site][1:]]
                    for fused in (False, True):
                        row = {
                            "batch": batch,
                            "site": site,
                            "magnitude": magnitude,
                            "fused": fused,
                            # The unfused graph checks mutate these same
                            # buffers before the fused eager comparison.
                            "eager_input_seed": (
                                2001 + site * 3 + 2 if fused else 301 + site
                            ),
                            "graph_input_seeds": [
                                2001 + site * 3 + replay for replay in range(3)
                            ],
                            "status": "checking",
                        }
                        result["checks"].append(row)
                        save()
                        ref = installed(data, fused)
                        got = extension.run(*data, fused)
                        row["eager"] = check_outputs(ref, got)
                        ro, partial = extension.run(*data, fused, True)
                        exact(ref[:1], [ro])
                        row["partial_oracle"] = partial_errors(ro, data[4], partial)
                        if not row["partial_oracle"]["passed"]:
                            raise ValueError("independent FP64 partial oracle failed")
                        del ro, partial, ref, got
                        graph = torch.cuda.CUDAGraph()
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            a = installed(data, fused)
                            b = extension.run(*data, fused)
                        row["graphs"] = []
                        for replay in range(3):
                            fresh = inputs(batch, 2001 + site * 3 + replay, magnitude)
                            for target, source in zip(data[:4], fresh[:4]):
                                target.copy_(source)
                            graph.replay()
                            row["graphs"].append(check_outputs(a, b))
                            exact(b, extension.run(*data, fused))
                        row["status"] = "complete"
                        del graph, a, b, fresh
                        save()
                if site % 15 == 0:
                    print(f"batch{batch}: checked site{site + 1}/90", flush=True)
            if args.check_only:
                continue
            # Same six distinct weight banks in both arms. No reference FP32
            # storage conversion is timed; baseline is installed paired BF16.
            rows = timing_rows(
                sites, narrow, batch, shared_activations=True, weight_banks=6
            )
            fn_bytes = sum(row[1][4].numel() * row[1][4].element_size() for row in rows)
            if fn_bytes <= 3 * 128 * 2**20:
                raise ValueError(
                    "timing working set does not exceed three L2 capacities"
                )
            graphs = []
            torch.cuda.reset_peak_memory_stats()
            for call in (installed, lambda data, fused: extension.run(*data, fused)):
                for row in rows:
                    call(row[1], row[2])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for row in rows:
                        call(row[1], row[2])
                graphs.append(graph)
            timing = {
                "batch": batch,
                "sites_per_graph": len(rows),
                "fn_bytes": fn_bytes,
                "fn_l2_ratio": fn_bytes / (128 * 2**20),
                "activation_sets": 6,
                "capture_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "rounds": [],
            }
            result["timings"].append(timing)
            for repeat in range(args.rounds):
                sample = {"repeat": repeat + 1}
                timing["rounds"].append(sample)
                for phase, variant in (("A", 0), ("B", 1), ("A2", 0)):
                    graph = graphs[variant]
                    for _ in range(3):
                        graph.replay()
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(args.replays):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    sample[phase] = (
                        start.elapsed_time(end) * 1000 / (args.replays * len(rows))
                    )
                    save()
            timing["median_us"] = {
                phase: statistics.median(r[phase] for r in timing["rounds"])
                for phase in ("A", "B", "A2")
            }
            timing["median_paired_throughput_change_pct"] = statistics.median(
                (0.5 * (r["A"] + r["A2"]) / r["B"] - 1) * 100 for r in timing["rounds"]
            )
            print(json.dumps(timing), flush=True)
            del graphs, graph, rows, data
            save()
        result["status"] = "complete"
    except BaseException as error:
        result["status"], result["error"] = "failed", repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
