# SPDX-License-Identifier: Apache-2.0
"""Prepare new serving caches from completed, pinned all-rank AOT qualification.

CPU only. Never run old HEAD-frozen qualification readers after their freeze ends.
The only permitted changed former sources are the three explicit new-hook
integration sites and the shared offline audit extraction. All qualified live
loader/compiler/kernel/native files stay exact.
"""

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import load_checked, require, verify
from benchmarks.kernels.check_glm53_geometry_loader import ORDER, check_launch
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from slimserve.rmsnorm_diagnostic import sha
from slimserve.rmsnorm_geometry import AOT_CLOSURE_SHA, AOT_PAIR_SHA, SERVING_SCHEMA

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_SITES = {
    ROOT / "slimserve/cli.py",
    ROOT / "vllm/v1/worker/gpu_model_runner.py",
    ROOT / "benchmarks/benchmark_glm53_campaign.py",
    ROOT / "benchmarks/kernels/audit_glm53_geometry_loader.py",
}
CASES = (
    ("control", "control"),
    ("geometry", "geometry"),
    ("return-control", "control"),
)


def completed_evidence(pair_path, closure_path):
    pair = load_checked(pair_path, AOT_PAIR_SHA)
    closure = load_checked(closure_path, AOT_CLOSURE_SHA)
    require(
        pair["status"] == closure["status"] == "complete"
        and pair["non_target_exact"] is True
        and closure["pair_sha256"] == AOT_PAIR_SHA
        and not closure["gpu_processes_after"].strip()
        and [(r["mode"], r["rank"]) for r in pair["runs"]] == list(ORDER),
        "completed eight-case AOT qualification required",
    )
    receipts = {str(p.resolve()): sha(p) for p in (pair_path, closure_path)}
    common, qualified = None, {m: {} for m in ("control", "geometry")}
    reference = None
    for row in pair["runs"]:
        folder = pair_path.parent / f"{row['mode']}-rank{row['rank']}"
        report_path = folder / "run/analysis.json"
        report = load_checked(report_path, row["analysis_sha256"])
        path = folder / "manifest.json"
        manifest = load_checked(path, report["manifest_sha256"])
        check_launch(folder, sha(path))
        require(
            report["status"] == "complete"
            and report["graphs"] == report["artifact_roots"] == 7
            and report["bound_launchers"] == 25
            and (row["mode"] == "control" or report["non_target_exact_vs_control"]),
            "incomplete qualified serving reference",
        )
        for filename, digest in report["receipts"].items():
            require(sha(filename) == digest, "completed AOT receipt changed")
            receipts[filename] = digest
        for p in (
            path,
            report_path,
            folder / "launch.json",
            folder / "load.log",
            folder / "audit.log",
        ):
            receipts[str(p.resolve())] = sha(p)
        fields = {
            k: manifest[k]
            for k in (
                "schema",
                "qualification_sha256",
                "namespace",
                "sources",
                "original_namespace",
                "original_files",
                "targets",
                "expected_graphs",
                "artifact_roots",
            )
        }
        if common is None:
            common = fields
            reference = dict(path=str(path.resolve()), sha256=sha(path))
        require(fields == common, "all-rank qualification manifests disagree")
        graph = folder / "run/graph-bindings.json"
        qualified[row["mode"]][str(row["rank"])] = dict(
            path=str(graph.resolve()), sha256=sha(graph)
        )
    return common, qualified, reference, receipts


def integration_sources(previous, additions):
    sources, changed = {}, {}
    for name, digest in previous.items():
        path = Path(name).resolve()
        current = sha(path)
        if current != digest:
            require(
                path in INTEGRATION_SITES,
                f"qualified non-integration source changed: {path}",
            )
            changed[str(path)] = dict(qualified=digest, integration=current)
        sources[str(path)] = current
    for path in additions:
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
    return sources, changed


def serving_sources(previous):
    from benchmarks.kernels import audit_glm53_geometry_serving, glm53_geometry_serving
    from slimserve import rmsnorm_geometry
    from slimserve.campaign_sources import PATHS

    return integration_sources(
        previous,
        [
            Path(__file__),
            Path(glm53_geometry_serving.__file__),
            Path(rmsnorm_geometry.__file__),
            Path(audit_glm53_geometry_serving.__file__),
            *sorted(INTEGRATION_SITES),
            *(ROOT / p for p in PATHS),
            ROOT / "benchmarks/kernels/glm53_geometry_workload.py",
            ROOT / "benchmarks/kernels/run_glm53_geometry_serving.py",
            ROOT / "benchmarks/analyze_glm53_deterministic_serving.py",
            ROOT / "benchmarks/analyze_glm53_quality_pair.py",
            ROOT / "slimserve/registry.py",
            ROOT / "slimserve/hardware.py",
            ROOT / "perf/glm53-rmsnorm-geometry-protocol.md",
        ],
    )


def prepare(pair_path, closure_path, output):
    require(
        not output.exists(), "new serving series required; preserve earlier attempts"
    )
    old, graphs, reference, receipts = completed_evidence(pair_path, closure_path)
    from benchmarks.kernels.glm53_geometry_workload import reference_evidence

    workload, workload_sources, _, _ = reference_evidence()
    sources, changes = serving_sources(old["sources"])
    sources.update(receipts)
    sources.update(workload_sources)
    original, output = Path(old["original_namespace"]), output.resolve()
    require(
        not output.is_relative_to(original) and not original.is_relative_to(output),
        "serving copies overlap original namespace",
    )
    base = copy.deepcopy(old)
    base.update(
        serving_schema=SERVING_SCHEMA,
        aot_qualification_sha256=AOT_PAIR_SHA,
        aot_closure_sha256=AOT_CLOSURE_SHA,
        aot_pair_path=str(pair_path.resolve()),
        aot_closure_path=str(closure_path.resolve()),
        qualified_manifest=reference,
        qualified_graphs=graphs,
        sources=sources,
        integration_changes=changes,
        workload=workload,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    verify(base)
    require(
        not any(p.is_symlink() for p in original.rglob("*")),
        "original cache contains aliases",
    )
    output.mkdir(parents=True)
    rows = []
    for label, mode in CASES:
        folder = output / label
        cache = folder / "cache"
        private = cache / "torch_compile_cache/torch_aot_compile" / base["namespace"]
        shutil.copytree(original, private)
        require(
            all(sha(private / p) == h for p, h in base["original_files"].items()),
            "new serving copy differs from originals",
        )
        manifest = dict(
            base,
            label=label,
            mode=mode,
            cache_root=str(cache),
            private_namespace=str(private),
            receipts=str(folder / "receipts"),
            worker_receipts=str(folder / "worker-receipts"),
        )
        path = folder / "manifest.json"
        write_new(path, manifest)
        rows.append(
            dict(label=label, mode=mode, manifest=str(path), manifest_sha256=sha(path))
        )
    verify(base)
    write_new(
        output / "preparation.json",
        dict(
            status="prepared",
            runs=rows,
            sources=sources,
            integration_changes=changes,
            git_commit=base["git_commit"],
        ),
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                runs=len(rows),
                preparation_sha256=sha(output / "preparation.json"),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        not subprocess.check_output(["git", "status", "--short"], text=True).strip(),
        "commit integration/protocol before freezing serving sources",
    )
    prepare(args.qualification.resolve(), args.closure.resolve(), args.output)


if __name__ == "__main__":
    main()
