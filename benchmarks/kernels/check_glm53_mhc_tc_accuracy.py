#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Separate FP64 accuracy census; does not modify or rename strict parity.

See perf/glm53-mhc-tc-accuracy-contract.md. No timings or serving integration.
Every eager row is checked; graph oracle rows are explicitly sampled, while
graph/eager bit checks cover all outputs. The original failed seed is included.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_mhc_prefill_tc import (
    check_outputs,
    installed,
    oracle_rows,
    partial_errors,
)
from benchmarks.kernels.benchmark_glm53_mhc_storage import exact, lossless_bf16
from benchmarks.kernels.benchmark_mhc_output_parallel import (
    checkpoint_parameters,
    inputs,
)
from benchmarks.kernels.mhc_fp64_oracle import accuracy_pair

PROBE_SHA = "45d5e817520a56bbc818e3e237d54b1feea3cb11385540aaabb1fec8fb7980e3"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_plan(batches, sites):
    if (
        not batches
        or not sites
        or len(set(batches)) != len(batches)
        or len(set(sites)) != len(sites)
        or any(t not in (64, 65, 128, 129, 7616) for t in batches)
        or any(not 0 <= site < 90 for site in sites)
    ):
        raise ValueError("unique prescribed batches and sites0..89 required")
    return [
        {
            "batch": t,
            "site": site,
            "magnitude": mag,
            "fused": fused,
            "eager_seed": 2003 + site * 3 if fused else 301 + site,
            "graph_seeds": [12001 + site * 3 + replay for replay in range(3)],
        }
        for t in batches
        for site in sites
        for mag in (0.01, 1.0, 100.0)
        for fused in (False, True)
    ]


def strict_diagnostic(reference, candidate):
    try:
        return {"passed": True, "outputs": check_outputs(reference, candidate)}
    except AssertionError as error:
        return {"passed": False, "error": str(error)}


def check_case(plan, site, extension, row):
    t, fused, magnitude = plan["batch"], plan["fused"], plan["magnitude"]
    data = inputs(t, plan["eager_seed"], magnitude)[:7]
    data[4:] = site
    ref, got = installed(data, fused), extension.run(*data, fused)
    exact(ref[:1], got[:1])
    row["strict_parity_diagnostic"] = strict_diagnostic(ref, got)
    row["eager_fp64_all_rows"] = accuracy_pair(ref[0], *data[4:], ref[1:], got[1:])
    if not row["eager_fp64_all_rows"]["passed"]:
        raise AssertionError("independent all-row eager accuracy failed")
    ro, partial = extension.run(*data, fused, True)
    exact(ref[:1], [ro])
    row["partial_oracle"] = partial_errors(ro, data[4], partial)
    if not row["partial_oracle"]["passed"]:
        raise AssertionError("unchanged partial oracle failed")
    del ref, got, ro, partial
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a, b = installed(data, fused), extension.run(*data, fused)
    selected = oracle_rows(t)
    row["graphs"] = []
    for seed in plan["graph_seeds"]:
        phase = {"seed": seed, "oracle_rows": selected, "status": "checking"}
        row["graphs"].append(phase)
        fresh = inputs(t, seed, magnitude)
        for target, source in zip(data[:4], fresh[:4]):
            target.copy_(source)
        graph.replay()
        exact(a[:1], b[:1])
        exact(a, installed(data, fused))
        exact(b, extension.run(*data, fused))
        # Finiteness of all graph outputs is checked by strict diagnostic;
        # its output mismatch can be recorded without becoming an accuracy pass.
        for tensor in (*a, *b):
            if not torch.isfinite(tensor).all():
                raise AssertionError("nonfinite graph output")
        phase["all_output_bits_match_eager"] = True
        phase["strict_parity_diagnostic"] = strict_diagnostic(a, b)
        phase["fp64_sampled_rows"] = accuracy_pair(
            a[0][selected],
            *data[4:],
            [v[selected] for v in a[1:]],
            [v[selected] for v in b[1:]],
        )
        if not phase["fp64_sampled_rows"]["passed"]:
            raise AssertionError("independent sampled graph accuracy failed")
        ro, partial = extension.run(*data, fused, True)
        exact(a[:1], [ro])
        phase["partial_oracle"] = partial_errors(ro, data[4], partial)
        if not phase["partial_oracle"]["passed"]:
            raise AssertionError("changed-input partial oracle failed")
        phase["status"] = "complete"
        del ro, partial, fresh
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[64, 65, 128, 129, 7616]
    )
    parser.add_argument("--sites", type=int, nargs="+", default=list(range(90)))
    args = parser.parse_args()
    try:
        plan = case_plan(args.batch, args.sites)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("output directory must be new; no overwrite or silent resume")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPU compute processes already active: {active}")
    if sha(args.extension) != PROBE_SHA:
        parser.error("extension differs from the frozen original tensor-core candidate")
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
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__),
        Path(__file__).with_name("mhc_fp64_oracle.py"),
        Path(__file__).with_name("benchmark_glm53_mhc_prefill_tc.py"),
        Path(__file__).with_name("benchmark_glm53_mhc_storage.py"),
        Path(__file__).with_name("benchmark_mhc_output_parallel.py"),
        Path(__file__).with_name("mhc_prefill_tc_probe.cu"),
        root / "perf/glm53-mhc-tc-accuracy-contract.md",
        root / "csrc/quixicore/serving/mhc_ampere.cuh",
        root / "csrc/quixicore/serving/bf16_decode_gemm.cuh",
        root / "csrc/quixicore/tm_cuda/tm_cuda_serving.cu",
    ]
    hashes = {str(p.relative_to(root)): sha(p) for p in paths}
    result = {
        "status": "running",
        "diagnostic_only": True,
        "contract": "glm53-mhc-tc-accuracy-v1",
        "method": __doc__,
        "command": sys.argv,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "source_sha256": hashes,
        "extension_sha256": PROBE_SHA,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "environment": settings,
        "planned_cases": len(plan),
        "plan": plan,
        "completed_cases": 0,
        "completed_graph_phases": 0,
        "strict_parity_failed_comparisons": 0,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    args.output.mkdir(parents=True)
    journal = args.output / "checks.jsonl"
    summary = args.output / "summary.json"

    def save():
        temp = summary.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(summary)

    save()
    started = time.monotonic()
    try:
        import vllm._quixicore_C as native

        torch.set_num_threads(4)
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("SM120 target required")
        spec = importlib.util.spec_from_file_location(
            "mhc_prefill_tc_probe", args.extension
        )
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        result["native_sha256"] = sha(native.__file__)
        result["gpu"] = torch.cuda.get_device_name()
        result["resources"] = extension.resources()
        sites, result["parameters"] = checkpoint_parameters(args.model)
        if len(sites) != 90:
            raise ValueError("all90 parameter sets required")
        sites = [[lossless_bf16(s[0]), *s[1:]] for s in sites]
        with journal.open("x") as stream:
            for item in plan:
                row = {**item, "status": "checking"}
                result["current_case"] = item
                save()
                case_started = time.monotonic()
                try:
                    check_case(item, sites[item["site"]], extension, row)
                    row["status"] = "complete"
                    result["completed_cases"] += 1
                    result["completed_graph_phases"] += len(row["graphs"])
                    result["strict_parity_failed_comparisons"] += sum(
                        not d["passed"]
                        for d in [
                            row["strict_parity_diagnostic"],
                            *(g["strict_parity_diagnostic"] for g in row["graphs"]),
                        ]
                    )
                except BaseException as error:
                    row["status"], row["error"] = "failed", repr(error)
                    raise
                finally:
                    row["elapsed_seconds"] = time.monotonic() - case_started
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    save()
                if result["completed_cases"] % 6 == 0:
                    print(
                        json.dumps(
                            {
                                k: result[k]
                                for k in (
                                    "completed_cases",
                                    "planned_cases",
                                    "strict_parity_failed_comparisons",
                                )
                            }
                        ),
                        flush=True,
                    )
        if hashes != {str(p.relative_to(root)): sha(p) for p in paths}:
            raise ValueError("source changed during census")
        if (
            sha(native.__file__) != result["native_sha256"]
            or sha(args.extension) != PROBE_SHA
        ):
            raise ValueError("native library changed during census")
        result["status"] = "complete"
        result.pop("current_case", None)
    except BaseException as error:
        result["status"], result["error"] = "failed", repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["journal_sha256"] = sha(journal) if journal.exists() else None
        save()


if __name__ == "__main__":
    main()
