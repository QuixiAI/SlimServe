# SPDX-License-Identifier: Apache-2.0
"""Prepare KV-only AOT qualification from pinned completed evidence. CPU only."""

import argparse
import copy
import importlib
import json
import shutil
import subprocess
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import load_checked, require, verify
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from benchmarks.kernels.glm53_artifact_roots import discover, read_store
from benchmarks.kernels.glm53_kv_loader import SCHEMA
from benchmarks.kernels.prepare_glm53_geometry_loader import (
    current_sources,
    graph_inventory,
    loader_source_files,
)
from slimserve.rmsnorm_diagnostic import NAMESPACE, sha

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "perf/results/2026-09-10"
QUALIFICATION_SHA = "bfaa7a495e7b69f228661a4402f5a6d5229d53b6950b6e7ee8cfa6275975144d"
CONTRACTS_SHA = "8337ae7e4ec383563ed2df0230f21e9424556feccbec3a93a61deea94caff3c8"
MAPPING_SHA = "320a6f39c4f92cd78459f21e23653e1e900870620da29bed923f5cba5dbe315e"
ORDER = tuple((mode, rank) for mode in ("control", "kv") for rank in range(4))


def qualified_targets(old, summary, contracts, folder):
    require(
        len(old["records"]) == len(summary["binaries"]) == 8,
        "eight qualified images required",
    )
    require(len(contracts["pairs"]) == 8, "eight attention graph pairs required")
    original = Path(old["original_namespace"])
    targets, receipts = {}, {}
    for rank in range(4):
        records = {}
        for arm in ("combo", "split"):
            sources = [
                r for r in old["records"] if (r["rank"], r["arm"]) == (rank, arm)
            ]
            images = [
                r for r in summary["binaries"] if (r["rank"], r["arm"]) == (rank, arm)
            ]
            require(
                len(sources) == len(images) == 1,
                "one qualified source/image per arm/rank",
            )
            source, image = sources[0], images[0]
            require(
                source["info"]["widths"]
                == ([512, 1536, 128] if arm == "combo" else [512])
                and all(
                    image[k] == source[k]
                    for k in ("selected", "source_sha256", "cubin_sha256")
                ),
                "qualified source/image join differs",
            )
            for field in ("source", "debug_source", "ptx"):
                path = Path(source[field])
                require(
                    path.resolve() == path and sha(path) == source[field + "_sha256"],
                    "qualified source drift",
                )
                receipts[str(path)] = sha(path)
            for field in ("source", "cubin"):
                path = folder / image[field]
                require(
                    path.resolve() == path
                    and path.is_relative_to(folder)
                    and sha(path) == image[field + "_sha256"],
                    "completed probe artifact drift",
                )
                receipts[str(path)] = sha(path)
            cubin = Path(source["ptx"]).with_suffix(".cubin")
            require(sha(cubin) == source["cubin_sha256"], "qualified cubin changed")
            receipts[str(cubin)] = sha(cubin)
            relative = (
                str(Path(source["source"]).relative_to(original))
                if arm == "combo"
                else f"kv_sources/rank-{rank}/{Path(source['source']).name}"
            )
            records[arm] = dict(
                source=source["source"],
                source_sha256=source["source_sha256"],
                relative=relative,
                kernel=source["info"]["kernel"],
                debug_source=source["debug_source"],
                debug_source_sha256=source["debug_source_sha256"],
                selected=copy.deepcopy(source["selected"]),
                cubin_sha256=source["cubin_sha256"],
            )
            if arm == "combo":
                require(
                    old["original_files"].get(relative) == source["source_sha256"],
                    "target outside original snapshot",
                )
        uses = []
        for pair in contracts["pairs"]:
            if pair["rank"] != rank:
                continue
            for mapped, record in (
                (pair["combo"], records["combo"]),
                (pair["split"][0], records["split"]),
            ):
                require(
                    all(mapped[k] == record[k] for k in ("source", "source_sha256"))
                    and mapped["symbol"] == record["kernel"]
                    and mapped["config"] == record["selected"]["config"],
                    "attention map/qualification join differs",
                )
            graph = Path(pair["old_graph"])
            require(
                sha(graph) == old["original_files"][str(graph.relative_to(original))],
                "mapped graph drift",
            )
            uses.append(dict(graph=str(graph), symbol=pair["combo"]["symbol"]))
        require(
            len(uses) == len({u["graph"] for u in uses}) == 2,
            "two graph uses per rank required",
        )
        targets[str(rank)] = [
            dict(records["combo"], kv=records["split"], static_graph_uses=uses)
        ]
    return targets, receipts


def evidence():
    folder = RESULTS / "kv-overwrite-v1"
    analysis_path = folder / "analysis.json"
    analysis = load_checked(analysis_path, QUALIFICATION_SHA)
    require(
        analysis["status"] == "complete"
        and analysis["cases"] == 120
        and analysis["binaries_verified"] == 8
        and not analysis["failures"]
        and analysis["historical_indexer_oracle_pass"] is False
        and analysis["production_qualified"] is False,
        "completed adapter qualification required",
    )
    manifest_path = RESULTS / "runtime-control/kv-overwrite-v1-manifest.json"
    old = load_checked(manifest_path, analysis["manifest_sha256"])
    summary_path = folder / "summary.json"
    summary = load_checked(summary_path, analysis["summary_sha256"])
    require(
        summary["status"] == "complete"
        and summary["manifest_sha256"] == sha(manifest_path)
        and len(summary["records"]) == 120,
        "incomplete adapter summary",
    )
    contract_path = RESULTS / "runtime-control/attention-contracts.json"
    contracts = load_checked(contract_path, CONTRACTS_SHA)
    require(contracts["status"] == "complete", "completed attention map required")
    mapping_path = RESULTS / "runtime-control/norm-graph-role-pairs.json"
    mapping = load_checked(mapping_path, MAPPING_SHA)
    targets, receipts = qualified_targets(old, summary, contracts, folder)
    for path, digest in contracts["receipts"].items():
        require(sha(path) == digest, "attention map evidence changed")
        receipts[path] = digest
    for path in (
        analysis_path,
        manifest_path,
        summary_path,
        contract_path,
        mapping_path,
    ):
        receipts[str(path)] = sha(path)
    original = Path(old["original_namespace"])
    require(original.name == NAMESPACE, "wrong original namespace")
    old = dict(old, expected_gpu_config=summary["gpu_config"])
    return old, targets, graph_inventory(mapping, original), receipts


def source_files():
    modules = (
        "glm53_kv_loader",
        "glm53_loader_hooks",
        "glm53_binary_observer",
        "glm53_attention_overwrite",
        "glm53_rmsnorm_geometry",
        "glm53_artifact_roots",
        "audit_glm53_kv_graphs",
        "audit_glm53_geometry_graphs",
        "audit_glm53_geometry_loader",
        "check_glm53_geometry_loader",
        "check_glm53_kv_loader",
        "audit_glm53_kv_loader",
        "prepare_glm53_geometry_loader",
        "check_glm53_attention_norms",
        "check_glm53_rmsnorm_geometry",
    )
    return [
        Path(__file__),
        *loader_source_files(),
        *(
            Path(importlib.import_module("benchmarks.kernels." + name).__file__)
            for name in modules
        ),
        ROOT / "slimserve/rmsnorm_diagnostic.py",
        ROOT / "vllm/compilation/caching.py",
        ROOT / "perf/glm53-attention-isolation.md",
        Path(importlib.import_module("torch._inductor.codecache").__file__),
        Path(
            importlib.import_module(
                "torch._inductor.runtime.static_triton_launcher"
            ).__file__
        ),
    ]


def build_base():
    old, targets, graphs, receipts = evidence()
    previous = dict(old["sources"])
    # This single protocol is deliberately updated between closed experiments.
    # Do not release arbitrary docs, raw evidence, serving, compiler or native files.
    protocol = str(ROOT / "perf/glm53-attention-isolation.md")
    protocol_before = previous.pop(protocol)
    sources, refreshed = current_sources(previous, source_files())
    if sha(protocol) != protocol_before:
        refreshed[protocol] = dict(previous=protocol_before, current=sha(protocol))
    sources.update(receipts)
    base = dict(
        schema=SCHEMA,
        namespace=NAMESPACE,
        qualification_sha256=QUALIFICATION_SHA,
        contracts_sha256=CONTRACTS_SHA,
        original_namespace=old["original_namespace"],
        original_files=old["original_files"],
        expected_gpu_config=old["expected_gpu_config"],
        targets=targets,
        expected_graphs=graphs,
        sources=sources,
        released_helper_changes=refreshed,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    verify(base)
    roots = {}
    for rank in range(4):
        store, _ = read_store(Path(base["original_namespace"]) / f"rank_{rank}_0/model")
        require(
            store.num_artifacts() == 7 and store.num_entries() == 46,
            "unexpected AOT matrix",
        )
        roots[str(rank)] = discover(
            store, Path(base["original_namespace"]), graphs[str(rank)]
        )
    base["artifact_roots"] = roots
    return base


def prepare(output, *, inspect_only=False):
    require(
        not output.exists(), "new series/report required; preserve previous attempts"
    )
    base = build_base()
    if inspect_only:
        write_new(output, base)
        print(
            json.dumps(dict(status="inspected", output=str(output), sha256=sha(output)))
        )
        return
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "commit the complete protocol and implementation before preparation",
    )
    output = output.resolve()
    original = Path(base["original_namespace"])
    require(
        not output.is_relative_to(original) and not original.is_relative_to(output),
        "cache overlap",
    )
    require(
        not any(p.is_symlink() for p in original.rglob("*")),
        "original cache contains aliases",
    )
    output.mkdir(parents=True)
    runs = []
    for mode, rank in ORDER:
        folder = output / f"{mode}-rank{rank}"
        cache = folder / "cache"
        private = cache / "torch_compile_cache/torch_aot_compile" / NAMESPACE
        shutil.copytree(original, private)
        require(
            all(sha(private / p) == h for p, h in base["original_files"].items()),
            "private copy differs",
        )
        private_sources = {}
        for targets in base["targets"].values():
            for target in targets:
                record = target["kv"]
                dest = private / record["relative"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                require(not dest.exists(), "KV source copy collision")
                shutil.copyfile(record["source"], dest)
                require(sha(dest) == record["source_sha256"], "KV source copy differs")
                private_sources[record["relative"]] = record["source_sha256"]
        manifest = dict(
            base,
            rank=rank,
            mode=mode,
            cache_root=str(cache),
            private_namespace=str(private),
            private_sources=private_sources,
            receipts=str(folder / "receipts"),
        )
        path = folder / "manifest.json"
        write_new(path, manifest)
        runs.append(
            dict(rank=rank, mode=mode, manifest=str(path), manifest_sha256=sha(path))
        )
    verify(base)
    write_new(
        output / "preparation.json",
        dict(
            status="prepared",
            runs=runs,
            sources=base["sources"],
            original_files=len(base["original_files"]),
            qualification_sha256=QUALIFICATION_SHA,
        ),
    )
    print(
        json.dumps(dict(status="prepared", output=str(output), runs=len(runs))),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    prepare(args.output, inspect_only=args.inspect_only)


if __name__ == "__main__":
    main()
