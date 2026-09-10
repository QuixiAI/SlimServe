# SPDX-License-Identifier: Apache-2.0
"""CPU inspection of qualified indexer sources and actual AOT root mapping.

This produces a base report, not runnable per-arm manifests or a GPU protocol.
"""

import argparse
import base64
import copy
import importlib
import subprocess
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import load_checked, require, verify
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from benchmarks.kernels.glm53_artifact_roots import discover, read_store
from benchmarks.kernels.glm53_indexer_correction_loader import SCHEMA
from benchmarks.kernels.prepare_glm53_geometry_loader import (
    current_sources,
    graph_inventory,
    loader_source_files,
)
from benchmarks.kernels.prepare_glm53_kv_loader import CONTRACTS_SHA, MAPPING_SHA
from slimserve.rmsnorm_diagnostic import NAMESPACE, sha

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "perf/results/2026-09-10"
QUALIFICATION_SHA = "c6ac3d0399af92be467ef47831f512c4edee60ff08fe0772afb399c59a18cc65"
SELECTION_CAPACITY = 8192  # Explicit diagnostic envelope; serving must check it.


def build_base():
    folder = RESULTS / "indexer-correction-v1"
    analysis_path = folder / "analysis.json"
    analysis = load_checked(analysis_path, QUALIFICATION_SHA)
    require(
        analysis["status"] == "complete"
        and analysis["cases"] == 120
        and not analysis["failures"]
        and analysis["corrected_oracle_max_ulp"] <= 1
        and analysis["production_qualified"] is False,
        "completed isolated correction qualification required",
    )
    manifest_path = RESULTS / "runtime-control/indexer-correction-manifest-v1.json"
    old = load_checked(manifest_path, analysis["manifest_sha256"])
    summary_path = folder / "summary.json"
    summary = load_checked(summary_path, analysis["summary_sha256"])
    require(
        summary["status"] == "complete"
        and len(summary["binaries"]) == 4
        and summary["manifest_sha256"] == sha(manifest_path),
        "qualified summary differs",
    )
    contracts_path = RESULTS / "runtime-control/attention-contracts.json"
    contracts = load_checked(contracts_path, CONTRACTS_SHA)
    mapping_path = RESULTS / "runtime-control/norm-graph-role-pairs.json"
    mapping = load_checked(mapping_path, MAPPING_SHA)
    original = Path(old["original_namespace"])
    require(original.name == NAMESPACE, "wrong original namespace")
    kernel = ROOT / "benchmarks/kernels/glm53_indexer_correction.py"
    require(
        sha(kernel) == old["sources"][str(kernel)],
        "qualified correction kernel changed",
    )
    receipts = {
        str(p): sha(p)
        for p in (
            analysis_path,
            manifest_path,
            summary_path,
            contracts_path,
            mapping_path,
            kernel,
        )
    }
    for path, digest in contracts["receipts"].items():
        require(sha(path) == digest, "attention contract receipt changed")
        receipts[path] = digest
    targets = {}
    for rank in range(4):
        (source,) = [r for r in old["records"] if r["rank"] == rank]
        binary = summary["binaries"][rank]
        require(
            binary["rank"] == rank
            and binary["original_selected"] == source["selected"]
            and binary["original_source_sha256"] == source["source_sha256"]
            and binary["original_cubin_sha256"] == source["cubin_sha256"],
            "original source/binary join differs",
        )
        relative = str(Path(source["source"]).relative_to(original))
        require(
            old["original_files"][relative] == source["source_sha256"],
            "target absent from original inventory",
        )
        for field in ("source", "debug_source", "ptx"):
            require(
                sha(source[field]) == source[field + "_sha256"],
                "original provenance changed",
            )
            receipts[source[field]] = source[field + "_sha256"]
        cubins = []
        for name, digest in binary["correction_artifacts"].items():
            path = folder / name
            require(
                path.resolve() == path
                and path.is_relative_to(folder)
                and sha(path) == digest,
                "qualified correction artifact drift",
            )
            receipts[str(path)] = digest
            if path.suffix == ".cubin":
                cubins.append(digest)
        require(
            len(cubins) == 1 and binary["correction_metadata"] == old["options"],
            "correction image/config join differs",
        )
        uses = []
        for pair in contracts["pairs"]:
            if pair["rank"] != rank:
                continue
            mapped = pair["combo"]
            require(
                mapped["source"] == source["source"]
                and mapped["source_sha256"] == source["source_sha256"]
                and mapped["symbol"] == source["info"]["kernel"]
                and mapped["config"] == source["selected"]["config"],
                "target graph map differs",
            )
            uses.append(dict(graph=pair["old_graph"], symbol=mapped["symbol"]))
        require(
            len(uses) == 2 and len({r["graph"] for r in uses}) == 2,
            "two real target graph uses required",
        )
        target = {
            k: source[k]
            for k in (
                "source",
                "source_sha256",
                "debug_source",
                "debug_source_sha256",
                "selected",
                "cubin_sha256",
            )
        }
        target.update(
            relative=relative,
            kernel=source["info"]["kernel"],
            static_graph_uses=uses,
            correction=dict(
                source=str(kernel),
                source_sha256=sha(kernel),
                relative="correction_sources/glm53_indexer_correction.py",
                debug_source=str(kernel),
                debug_source_sha256=sha(kernel),
                kernel="correct_indexer",
                selected=dict(
                    hash=base64.b32encode(bytes.fromhex(binary["correction_hash"]))
                    .decode()
                    .rstrip("="),
                    config=dict(num_warps=1, num_stages=1),
                ),
                cubin_sha256=cubins[0],
            ),
        )
        targets[str(rank)] = [target]
    modules = (
        "glm53_indexer_correction_loader",
        "audit_glm53_indexer_correction_graphs",
        "glm53_kv_loader",
        "glm53_loader_hooks",
        "glm53_binary_observer",
        "glm53_artifact_roots",
        "audit_glm53_geometry_graphs",
    )
    additions = [
        Path(__file__),
        *loader_source_files(),
        *[
            Path(importlib.import_module("benchmarks.kernels." + n).__file__)
            for n in modules
        ],
        Path(
            importlib.import_module(
                "torch._inductor.runtime.static_triton_launcher"
            ).__file__
        ),
    ]
    previous = copy.deepcopy(old["sources"])
    released = {}
    for name in (
        "perf/glm53-indexer-correction-protocol.md",
        "perf/glm53-attention-isolation.md",
    ):
        path = str(ROOT / name)
        before = previous.pop(path, None)
        if before is not None:
            additions.append(Path(path))
            if sha(path) != before:
                released[path] = dict(previous=before, current=sha(path))
    sources, refreshed = current_sources(previous, additions)
    sources.update(receipts)
    base = dict(
        schema=SCHEMA,
        namespace=NAMESPACE,
        qualification_sha256=QUALIFICATION_SHA,
        contracts_sha256=CONTRACTS_SHA,
        original_namespace=str(original),
        original_files=old["original_files"],
        targets=targets,
        expected_graphs=graph_inventory(mapping, original),
        expected_gpu_config=summary["gpu_config"],
        selection_capacity=SELECTION_CAPACITY,
        sources=sources,
        released_helper_changes={**released, **refreshed},
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    roots = {}
    for rank in range(4):
        store, _ = read_store(original / f"rank_{rank}_0/model")
        require(
            store.num_artifacts() == 7 and store.num_entries() == 46,
            "AOT matrix changed",
        )
        roots[str(rank)] = discover(store, original, base["expected_graphs"][str(rank)])
    base["artifact_roots"] = roots
    verify(base)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve prior report")
    base = build_base()
    write_new(args.output, base)
    print(
        f"Inspected 4 sources, 8 AOT target bindings; {len(base['sources'])} receipts"
    )


if __name__ == "__main__":
    main()
