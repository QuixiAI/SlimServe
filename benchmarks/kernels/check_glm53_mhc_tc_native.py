#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove native integration is bit-exact to the independently qualified probe.

This is not a new FP64 census or a serving-performance qualification. It uses
the full prescribed 2700-case/8100-phase plan and verifies the original receipt,
probe binary, kernel source, dependencies and actual model parameter identities.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_mhc_prefill_tc import installed
from benchmarks.kernels.benchmark_glm53_mhc_storage import exact, lossless_bf16
from benchmarks.kernels.benchmark_glm53_mhc_tc_qualified import (
    BATCHES,
    qualified_census,
    require,
)
from benchmarks.kernels.benchmark_mhc_output_parallel import (
    checkpoint_parameters,
    inputs,
)
from benchmarks.kernels.check_glm53_mhc_tc_accuracy import PROBE_SHA, case_plan, sha


def qualified_kernel_body(probe, native):
    """Only the containing namespace and host wrapper may differ."""
    original = probe.split("constexpr int H =", 1)[1].split(
        "\ntemplate <bool FUSED>\nstatic std::vector<Tensor> run_typed", 1
    )[0]
    copied = native.split("constexpr int H =", 1)[1].rsplit("}", 1)[0]
    require(original.strip() == copied.strip(), "qualified kernel body changed")


def check_case(item, site, extension, row):
    batch, fused, magnitude = item["batch"], item["fused"], item["magnitude"]
    data = inputs(batch, item["eager_seed"], magnitude)[:7]
    data[4:] = site
    exact(extension.run(*data, fused), installed(data, fused))
    row["eager_all_output_bits_match_probe"] = True
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reference, candidate = extension.run(*data, fused), installed(data, fused)
    row["graphs"] = []
    for seed in item["graph_seeds"]:
        phase = {"seed": seed, "status": "checking"}
        row["graphs"].append(phase)
        fresh = inputs(batch, seed, magnitude)
        for target, value in zip(data[:4], fresh[:4]):
            target.copy_(value)
        graph.replay()
        exact(reference, candidate)
        exact(reference, extension.run(*data, fused))
        exact(candidate, installed(data, fused))
        phase.update(
            all_output_bits_match_probe=True,
            both_graphs_match_eager=True,
            status="complete",
        )
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "new output directory required")
    census, receipt = qualified_census(args.census)
    root = Path(__file__).resolve().parents[2]
    # Host dispatch is the intentional integration change. All originally
    # qualified GPU dependencies, oracle, input and gate sources stay frozen.
    changed_host = "csrc/quixicore/tm_cuda/tm_cuda_serving.cu"
    for path, digest in census["source_sha256"].items():
        if path != changed_host:
            require(sha(root / path) == digest, f"qualified source changed: {path}")
    probe = root / "benchmarks/kernels/mhc_prefill_tc_probe.cu"
    header = root / "csrc/quixicore/serving/glm53_mhc_prefill_tc.cuh"
    qualified_kernel_body(probe.read_text(), header.read_text())
    extension_path = Path(census["settings"]["extension"])
    require(sha(extension_path) == PROBE_SHA, "qualified probe binary changed")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    require(not active, f"GPU compute processes already active: {active}")
    settings = {**census["environment"], "VLLM_GLM5_MHC_PREFILL_TC": "1"}
    for key, value in settings.items():
        require(
            key not in os.environ or os.environ[key] == value, f"requires {key}={value}"
        )
        os.environ[key] = value
    paths = set(census["source_sha256"]) | {
        str(Path(__file__).relative_to(root)),
        str(header.relative_to(root)),
        "benchmarks/kernels/benchmark_glm53_mhc_tc_qualified.py",
        "vllm/quixicore/ops.py",
        "vllm/model_executor/layers/glm5_next_mhc_ops.py",
        "vllm/model_executor/models/glm5_next.py",
    }
    hashes = {p: sha(root / p) for p in sorted(paths)}
    plan = case_plan(BATCHES, list(range(90)))
    result = {
        "status": "running",
        "method": __doc__,
        "command": sys.argv,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "source_sha256": hashes,
        "qualified_accuracy": receipt,
        "extension_sha256": PROBE_SHA,
        "environment": settings,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "plan": plan,
        "planned_cases": len(plan),
        "completed_cases": 0,
        "completed_graph_phases": 0,
    }
    args.output.mkdir(parents=True)
    journal, summary = args.output / "checks.jsonl", args.output / "summary.json"

    def save():
        temp = summary.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(summary)

    save()
    started = time.monotonic()
    try:
        import vllm._quixicore_C as native

        from vllm.quixicore.ops import quixicore_ops as qc

        torch.set_num_threads(4)
        require(torch.cuda.get_device_capability() == (12, 0), "SM120 required")
        require(qc.get_glm53_mhc_prefill_tc() == 1, "native candidate not enabled")
        require(qc.get_dsv4_mhc_prefill_min_t() == 64, "prefill selector changed")
        result["native_sha256"] = sha(native.__file__)
        result["gpu"] = torch.cuda.get_device_name()
        spec = importlib.util.spec_from_file_location(
            "mhc_prefill_tc_probe", extension_path
        )
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        sites, result["parameters"] = checkpoint_parameters(
            Path(census["settings"]["model"])
        )
        require(
            result["parameters"] == census["parameters"], "model parameters changed"
        )
        require(len(sites) == 90, "all 90 parameter sites required")
        sites = [[lossless_bf16(s[0]), *s[1:]] for s in sites]
        with journal.open("x") as stream:
            for item in plan:
                row = {**item, "status": "checking"}
                result["current_case"] = item
                try:
                    check_case(item, sites[item["site"]], extension, row)
                    row["status"] = "complete"
                    result["completed_cases"] += 1
                    result["completed_graph_phases"] += len(row["graphs"])
                except BaseException as error:
                    row.update(status="failed", error=repr(error))
                    raise
                finally:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    save()
                if result["completed_cases"] % 90 == 0:
                    print(
                        json.dumps(
                            {
                                k: result[k]
                                for k in (
                                    "completed_cases",
                                    "planned_cases",
                                    "completed_graph_phases",
                                )
                            }
                        ),
                        flush=True,
                    )
        require(
            hashes == {p: sha(root / p) for p in paths}, "source changed during census"
        )
        require(
            sha(native.__file__) == result["native_sha256"],
            "native changed during census",
        )
        require(sha(extension_path) == PROBE_SHA, "probe changed during census")
        _, final_receipt = qualified_census(args.census)
        require(final_receipt == receipt, "qualification receipt changed during census")
        result["status"] = "complete"
        result.pop("current_case", None)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["journal_sha256"] = sha(journal) if journal.exists() else None
        save()


if __name__ == "__main__":
    main()
