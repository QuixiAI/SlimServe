# SPDX-License-Identifier: Apache-2.0
"""Prepare one control/KV/return serving series from completed AOT receipts.

CPU only. No historical source-frozen reader is rerun. The qualified live KV
loader, compiler, adapter, graph inventory and native code must remain exact.
"""

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import load_checked, require, verify
from benchmarks.kernels.check_glm53_geometry_loader import check_launch
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from benchmarks.kernels.prepare_glm53_geometry_serving import integration_sources
from benchmarks.kernels.prepare_glm53_kv_loader import ORDER
from slimserve.kv_diagnostic import AOT_PAIR_SHA, CASES, SERVING_SCHEMA
from slimserve.rmsnorm_diagnostic import sha

ROOT = Path(__file__).resolve().parents[2]
PAIR = ROOT / "perf/results/2026-09-10/kv-aot-qualification-v2/pair-analysis.json"
# Only the offline event-lifecycle checker and closed-stage notebook change
# among the formerly frozen files. Serving hooks/client files are added below.
INTEGRATION_SITES = {
    ROOT / "benchmarks/kernels/audit_glm53_kv_loader.py",
    ROOT / "perf/glm53-attention-isolation.md",
}
COMMON_FIELDS = (
    "schema",
    "qualification_sha256",
    "contracts_sha256",
    "namespace",
    "sources",
    "original_namespace",
    "original_files",
    "targets",
    "expected_graphs",
    "artifact_roots",
    "private_sources",
    "expected_gpu_config",
)


def completed_evidence(pair_path):
    pair_path = pair_path.resolve()
    pair = load_checked(pair_path, AOT_PAIR_SHA)
    require(
        pair["status"] == "complete"
        and pair["non_target_exact"] is True
        and [(r["mode"], r["rank"]) for r in pair["runs"]] == list(ORDER),
        "completed eight-case KV AOT qualification required",
    )
    receipts = {str(pair_path): AOT_PAIR_SHA}
    common, reference = None, None
    graphs = {m: {} for m in ("control", "kv")}
    for row in pair["runs"]:
        mode, rank = row["mode"], row["rank"]
        folder = pair_path.parent / f"{mode}-rank{rank}"
        report_path, path = folder / "run/analysis.json", folder / "manifest.json"
        report = load_checked(report_path, row["analysis_sha256"])
        manifest = load_checked(path, report["manifest_sha256"])
        check_launch(folder, sha(path))
        launch = json.loads((folder / "launch.json").read_text())
        require(
            report["status"] == "complete"
            and report["graphs"] == report["artifact_roots"] == 7
            and report["bound_launchers"] == 25
            and report["target_bindings"] == 2
            and report["appended_launchers"] == (2 if mode == "kv" else 0)
            and (mode == "control" or report["non_target_exact_vs_control"] is True)
            and report["auditor_sha256"] == pair["auditor_sha256"]
            and (manifest["mode"], manifest["rank"]) == (mode, rank)
            and launch["gpu_config_before"]
            == launch["gpu_config_after"]
            == manifest["expected_gpu_config"],
            "incomplete KV serving reference or hardware identity changed",
        )
        for filename, digest in report["receipts"].items():
            require(sha(filename) == digest, "completed KV AOT receipt changed")
            receipts[str(Path(filename).resolve())] = digest
        for p in (
            path,
            report_path,
            folder / "launch.json",
            folder / "load.log",
            folder / "audit.log",
        ):
            receipts[str(p)] = sha(p)
        fields = {k: manifest[k] for k in COMMON_FIELDS}
        if common is None:
            common = fields
            reference = dict(path=str(path), sha256=sha(path))
        require(fields == common, "all-rank KV qualification manifests disagree")
        graph = folder / "run/graph-bindings.json"
        require(
            report["receipts"].get(str(graph)) == sha(graph),
            "qualified KV graph is not an audited receipt",
        )
        graphs[mode][str(rank)] = dict(path=str(graph), sha256=sha(graph))
    return common, graphs, reference, receipts


def serving_sources(previous):
    from slimserve.campaign_sources import PATHS

    return integration_sources(
        previous,
        [
            Path(__file__),
            *(ROOT / p for p in PATHS),
            *(
                ROOT / f"benchmarks/kernels/{name}.py"
                for name in (
                    "prepare_glm53_geometry_serving",
                    "glm53_geometry_workload",
                    "audit_glm53_geometry_serving",
                    "run_glm53_geometry_serving",
                )
            ),
            ROOT / "benchmarks/analyze_glm53_deterministic_serving.py",
            ROOT / "slimserve/registry.py",
            ROOT / "slimserve/hardware.py",
            ROOT / "slimserve/profiles.json",
            ROOT / "perf/glm53-kv-serving-protocol.md",
            *sorted(INTEGRATION_SITES),
        ],
        allowed_changes=INTEGRATION_SITES,
    )


def prepare(pair_path, output, *, inspect_only=False):
    require(
        not output.exists(),
        "new serving series/report required; preserve previous attempts",
    )
    old, graphs, reference, receipts = completed_evidence(pair_path)
    from benchmarks.kernels.glm53_geometry_workload import reference_evidence

    workload, workload_sources, _, _ = reference_evidence()
    sources, changes = serving_sources(old["sources"])
    sources.update(receipts)
    sources.update(workload_sources)
    base = copy.deepcopy(old)
    base.update(
        serving_schema=SERVING_SCHEMA,
        aot_qualification_sha256=AOT_PAIR_SHA,
        aot_pair_path=str(pair_path.resolve()),
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
    if inspect_only:
        write_new(output, base)
        print(
            json.dumps(dict(status="inspected", path=str(output), sha256=sha(output))),
            flush=True,
        )
        return
    require(
        not subprocess.check_output(["git", "status", "--short"], text=True).strip(),
        "commit integration/protocol before freezing serving sources",
    )
    original, output = Path(old["original_namespace"]), output.resolve()
    require(
        not output.is_relative_to(original) and not original.is_relative_to(output),
        "serving copies overlap original namespace",
    )
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
        copied = {}
        for targets in base["targets"].values():
            for target in targets:
                kv = target["kv"]
                source, dest = Path(kv["source"]), private / kv["relative"]
                require(
                    dest.resolve() == dest
                    and dest.is_relative_to(private)
                    and not dest.exists()
                    and sha(source) == kv["source_sha256"],
                    "private KV copy escaped, collided or changed",
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                copied[kv["relative"]] = sha(dest)
        require(copied == base["private_sources"], "private KV sources differ")
        require(
            {
                str(p.relative_to(private)): sha(p)
                for p in private.rglob("*")
                if p.is_file()
            }
            == {**base["original_files"], **base["private_sources"]},
            "fresh serving copy differs from qualified files",
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
    parser.add_argument("--qualification", type=Path, default=PAIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    prepare(args.qualification.resolve(), args.output, inspect_only=args.inspect_only)


if __name__ == "__main__":
    main()
