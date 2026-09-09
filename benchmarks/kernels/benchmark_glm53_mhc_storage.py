#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lossless BF16 storage for GLM's upcast mHC fn, with unchanged FP32 math.

Isolated kernels only. Require bit-exact output to installed FP32 serving and
changed-input graph parity over every actual checkpoint site. Fixed A/B/A
timings rotate two independently allocated copies of all 90 parameter sets:
even the narrower fn working set exceeds SM120's 128 MiB L2.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_mhc_output_parallel import (
    build,
    checkpoint_parameters,
    inputs,
)


def lossless_bf16(value):
    narrow = value.bfloat16()
    if not torch.isfinite(value).all() or not torch.equal(value, narrow.float()):
        raise ValueError("fn storage change would alter checkpoint values")
    return narrow


def exact(reference, candidate):
    if len(reference) != len(candidate):
        raise AssertionError("output arity changed")
    for i, (a, b) in enumerate(zip(reference, candidate)):
        if a.dtype != b.dtype or not torch.equal(a, b):
            raise AssertionError(f"output {i} is not bit-exact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.replays, *args.batch) < 1 or max(args.batch) > 7616:
        parser.error("positive rounds/replays/batches, at most 7616 rows")
    if not args.check_only and not args.build_only and max(args.batch) > 128:
        parser.error("large prefill shapes require --check-only to bound GPU memory")
    if not args.build_only:
        if args.output is None or args.output.exists():
            parser.error("a run requires a new output path")
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            parser.error(f"GPUs already have compute processes: {active}")
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
    extension = build(args.build_dir, name="mhc_storage_probe")
    if args.build_only:
        return
    from vllm.quixicore.ops import quixicore_ops as qc

    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        Path(__file__).with_name("mhc_storage_probe.cu"),
        Path(__file__).with_name("benchmark_mhc_output_parallel.py"),
        root / "csrc/quixicore/serving/mhc_ampere.cuh",
    ]
    result = {
        "status": "running",
        "diagnostic_only": True,
        "method": __doc__,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "extension_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "serving_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "vllm").glob("_quixicore_C*.so")
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "environment": settings,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "checks": [],
        "timings": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def installed(data, fused):
        x, residual, post, comb, fn, scale, base = data
        constants = [1e-5, 1e-6, 1e-6, 2.0, 20, None, 0.0]
        if fused:
            return qc.dsv4_mhc_fused_post_pre(*data, *constants)
        return [residual, *qc.dsv4_mhc_pre(residual, fn, scale, base, *constants)]

    try:
        sites, result["parameters"] = checkpoint_parameters(args.model)
        narrow = [lossless_bf16(site[0]) for site in sites]
        result["site_count"] = len(sites)
        if len(sites) != 90:
            raise ValueError("expected all 90 GLM mHC sites")
        save()
        for batch in args.batch:
            for site, weights in enumerate(sites):
                for magnitude in (0.01, 1.0, 100.0):
                    data = inputs(batch, 301 + site, magnitude)[:7]
                    data[4:7] = weights
                    candidate = [*data[:4], narrow[site], *data[5:]]
                    for fused in (False, True):
                        row = {
                            "batch": batch,
                            "site": site,
                            "magnitude": magnitude,
                            "fused": fused,
                            "status": "checking",
                        }
                        result["checks"].append(row)
                        original = extension.run(*data, fused)
                        exact(installed(data, fused), original)
                        exact(original, extension.run(*candidate, fused))
                        # Both storage forms share the same changing activations.
                        graph = torch.cuda.CUDAGraph()
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            a = extension.run(*data, fused)
                            b = extension.run(*candidate, fused)
                        for replay in range(3):
                            fresh = inputs(batch, 2001 + site * 3 + replay, magnitude)
                            for target, source in zip(data[:4], fresh[:4]):
                                target.copy_(source)
                            graph.replay()
                            exact(a, b)
                        row["status"] = "bit_exact"
                    save()
                if site % 15 == 0:
                    print(f"batch {batch}: checked site {site + 1}/90", flush=True)
            if args.check_only:
                continue
            rows = []
            for bank in range(2):
                for site, weights in enumerate(sites):
                    data = inputs(batch, 5001 + site + bank * 90)[:7]
                    data[4:7] = [weights[0].clone(), *weights[1:]]
                    candidate = [*data[:4], narrow[site].clone(), *data[5:]]
                    rows.append((data, candidate, site != 0))
            size = sum(row[1][4].numel() * row[1][4].element_size() for row in rows)
            if size <= 128 * 2**20:
                raise ValueError("narrow weight rotation does not exceed SM120 L2")
            graphs = []
            for variant in (0, 1):
                for row in rows:
                    extension.run(*row[variant], row[2])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for row in rows:
                        extension.run(*row[variant], row[2])
                graphs.append(graph)
            timing = {
                "batch": batch,
                "sites_per_graph": len(rows),
                "narrow_fn_bytes": size,
                "rounds": [],
            }
            result["timings"].append(timing)
            for _ in range(args.rounds):
                sample = {}
                for label, variant in (("A", 0), ("B", 1), ("A2", 0)):
                    graph = graphs[variant]
                    for _ in range(3):
                        graph.replay()
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    for _ in range(args.replays):
                        graph.replay()
                    b.record()
                    b.synchronize()
                    sample[label] = (
                        a.elapsed_time(b) * 1000 / (args.replays * len(rows))
                    )
                timing["rounds"].append(sample)
                save()
            timing["median_paired_throughput_change_pct"] = statistics.median(
                (0.5 * (r["A"] + r["A2"]) / r["B"] - 1) * 100 for r in timing["rounds"]
            )
            print(json.dumps(timing), flush=True)
            del graphs, graph, rows, data, candidate
            save()
        result["status"] = "complete"
    except Exception as exc:
        result["status"], result["error"] = "failed", repr(exc)
        save()
        raise
    save()


if __name__ == "__main__":
    main()
