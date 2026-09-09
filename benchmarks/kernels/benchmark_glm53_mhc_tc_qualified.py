#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold isolated A/B/A timing, gated on the complete independent FP64 census.

This does not satisfy sanitizer, real-profile or model-quality promotion gates.
The original strict parity census remains failed. Five rounds/four replays,
three warmups, six banks of all90 actual fn matrices, all five prescribed sizes.
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_mhc_prefill_tc import installed, oracle_rows
from benchmarks.kernels.benchmark_glm53_mhc_storage import lossless_bf16
from benchmarks.kernels.benchmark_mhc_output_parallel import (
    checkpoint_parameters,
    inputs,
)
from benchmarks.kernels.check_glm53_mhc_tc_accuracy import PROBE_SHA, case_plan, sha

BATCHES = [64, 65, 128, 129, 7616]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_accuracy(stats, rows):
    require(stats["passed"] and stats["rows"] == rows, "accuracy rows/status mismatch")
    for arm in ("reference", "candidate"):
        data = stats[arm]
        require(data["passed"], f"{arm} failed accuracy")
        require(data["elements"] == rows * 4096, "hidden-coordinate coverage mismatch")
        require(
            not any(
                data[k] for k in ("violations", "post_violations", "comb_violations")
            ),
            "pointwise accuracy violation",
        )
        for key, value in data.items():
            require(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                f"invalid accuracy metric {arm}/{key}",
            )
        nrms = math.sqrt(data["squared_error"] / max(data["ideal_squared_sum"], 1e-60))
        require(
            math.isclose(nrms, data["normalized_rms"], rel_tol=1e-12, abs_tol=1e-30),
            "inconsistent RMS calculation",
        )
    require(
        stats["candidate_rms_noninferior"]
        and stats["candidate"]["normalized_rms"]
        <= 1.001 * stats["reference"]["normalized_rms"] + 1e-7,
        "candidate RMS regression",
    )


def validate_partial(data, batch):
    require(
        data["passed"] and data["finite"] and data["rows"] == oracle_rows(batch),
        "partial oracle status/rows mismatch",
    )
    for key, limit in (
        ("dot_normalized_rms", 2e-6),
        ("dot_row_peak_error", 2e-5),
        ("square_relative_error", 1e-5),
    ):
        require(
            math.isfinite(data[key]) and 0 <= data[key] <= limit,
            f"partial oracle violation: {key}",
        )


def qualified_census(directory):
    """Fail closed on bounded, interrupted, changed or internally invalid runs."""
    summary = directory / "summary.json"
    data = json.loads(summary.read_bytes())
    plan = case_plan(BATCHES, list(range(90)))
    require(
        data["status"] == "complete" and data["contract"] == "glm53-mhc-tc-accuracy-v1",
        "full independent accuracy census must be complete",
    )
    require(
        data["planned_cases"] == data["completed_cases"] == len(plan)
        and data["completed_graph_phases"] == 8100,
        "full 2700-case/8100-phase census required",
    )
    require(
        data["plan"] == plan and data["extension_sha256"] == PROBE_SHA,
        "census plan or candidate changed",
    )
    journal = directory / "checks.jsonl"
    require(sha(journal) == data["journal_sha256"], "census journal digest mismatch")
    strict_failures, count = 0, 0
    with journal.open() as stream:
        for count, line in enumerate(stream, 1):
            require(count <= len(plan), "extra census row")
            row, item = json.loads(line), plan[count - 1]
            require(
                row["status"] == "complete"
                and all(row[k] == v for k, v in item.items()),
                "census row incomplete, reordered or changed",
            )
            validate_accuracy(row["eager_fp64_all_rows"], item["batch"])
            validate_partial(row["partial_oracle"], item["batch"])
            strict_failures += not row["strict_parity_diagnostic"]["passed"]
            require(len(row["graphs"]) == 3, "missing graph phase")
            for phase, seed in zip(row["graphs"], item["graph_seeds"]):
                require(
                    phase["status"] == "complete"
                    and phase["seed"] == seed
                    and phase["all_output_bits_match_eager"],
                    "graph status/seed/bit gate failed",
                )
                require(
                    phase["oracle_rows"] == oracle_rows(item["batch"]),
                    "graph sampled rows changed",
                )
                validate_accuracy(phase["fp64_sampled_rows"], len(phase["oracle_rows"]))
                validate_partial(phase["partial_oracle"], item["batch"])
                strict_failures += not phase["strict_parity_diagnostic"]["passed"]
    require(count == len(plan), "missing census rows")
    require(
        strict_failures == data["strict_parity_failed_comparisons"],
        "strict parity failure count changed",
    )
    return data, {
        "summary_sha256": sha(summary),
        "journal_sha256": sha(journal),
        "completed_cases": count,
        "graph_phases": 8100,
        "strict_parity_failed_comparisons": strict_failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "new output directory required")
    census, receipt = qualified_census(args.census)
    root = Path(__file__).resolve().parents[2]
    source_hashes = {
        **census["source_sha256"],
        str(Path(__file__).relative_to(root)): sha(__file__),
    }
    for path, digest in source_hashes.items():
        require(
            sha(root / path) == digest, f"source changed since qualification: {path}"
        )
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    require(not active, f"GPU compute processes already active: {active}")
    for key, value in census["environment"].items():
        require(
            key not in os.environ or os.environ[key] == value, f"requires {key}={value}"
        )
        os.environ[key] = value
    extension_path = Path(census["settings"]["extension"])
    require(sha(extension_path) == PROBE_SHA, "qualified candidate binary changed")
    import vllm._quixicore_C as native

    require(
        sha(native.__file__) == census["native_sha256"],
        "qualified baseline binary changed",
    )
    torch.set_num_threads(4)
    require(torch.cuda.get_device_capability() == (12, 0), "SM120 target required")
    spec = importlib.util.spec_from_file_location(
        "mhc_prefill_tc_probe", extension_path
    )
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    args.output.mkdir(parents=True)
    result = {
        "status": "running",
        "diagnostic_only": True,
        "method": __doc__,
        "command": sys.argv,
        "accuracy_receipt": receipt,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "source_sha256": source_hashes,
        "native_sha256": census["native_sha256"],
        "extension_sha256": PROBE_SHA,
        "environment": census["environment"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "resources": extension.resources(),
        "timings": [],
    }

    def save():
        temp = args.output / "summary.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.output / "summary.json")

    save()
    try:
        sites, result["parameters"] = checkpoint_parameters(
            Path(census["settings"]["model"])
        )
        require(
            result["parameters"] == census["parameters"],
            "model parameters changed since census",
        )
        sites = [[lossless_bf16(s[0]), *s[1:]] for s in sites]
        for batch in BATCHES:
            # Same six activation seeds/90-site rotation as the original screen.
            # Only used BF16 weights are allocated, no unused FP32 bank copies.
            rows = []
            for bank in range(6):
                activations = inputs(batch, 5001 + bank * 90)[:4]
                for site, weights in enumerate(sites):
                    rows.append(
                        ([*activations, weights[0].clone(), *weights[1:]], site != 0)
                    )
            fn_bytes = sum(data[4].numel() * data[4].element_size() for data, _ in rows)
            require(
                len(rows) == 540 and fn_bytes > 3 * 128 * 2**20,
                "cold fn working set too small",
            )
            graphs = []
            torch.cuda.reset_peak_memory_stats()
            for call in (installed, lambda data, fused: extension.run(*data, fused)):
                for data, fused in rows:
                    call(data, fused)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for data, fused in rows:
                        call(data, fused)
                graphs.append(graph)
            timing = {
                "batch": batch,
                "sites_per_graph": 540,
                "fn_bytes": fn_bytes,
                "fn_l2_ratio": fn_bytes / (128 * 2**20),
                "activation_sets": 6,
                "capture_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "rounds": [],
            }
            result["timings"].append(timing)
            for repeat in range(5):
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
                    for _ in range(4):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    sample[phase] = start.elapsed_time(end) * 1000 / (4 * 540)
                    save()
            timing["median_us"] = {
                phase: statistics.median(r[phase] for r in timing["rounds"])
                for phase in ("A", "B", "A2")
            }
            timing["median_paired_throughput_change_pct"] = statistics.median(
                (0.5 * (r["A"] + r["A2"]) / r["B"] - 1) * 100 for r in timing["rounds"]
            )
            print(json.dumps(timing), flush=True)
            del graphs, graph, rows, data, activations
            save()
        for path, digest in source_hashes.items():
            require(sha(root / path) == digest, f"source changed during timing: {path}")
        require(
            sha(native.__file__) == census["native_sha256"]
            and sha(extension_path) == PROBE_SHA,
            "binary changed during timing",
        )
        result["status"] = "complete"
    except BaseException as error:
        result["status"], result["error"] = "failed", repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
