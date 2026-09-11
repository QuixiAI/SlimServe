# SPDX-License-Identifier: Apache-2.0
"""Offline serving load/capture provenance, separate from model quality and TPS."""

import argparse
import json
from functools import partial
from pathlib import Path

from benchmarks.kernels.audit_glm53_geometry_loader import (
    check_graph_records,
    check_root_records,
    json_lines,
)
from benchmarks.kernels.check_glm53_attention_norms import require, verify
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from benchmarks.kernels.glm53_geometry_serving import compare_qualified
from slimserve.glm53_serving_diagnostic import policy as serving_policy
from slimserve.rmsnorm_diagnostic import sha
from slimserve.rmsnorm_geometry import AOT_CLOSURE_SHA, SERVING_SCHEMA

PHASES = ("before-forward", "capture-before", "capture-after")


def check_worker(
    manifest,
    manifest_sha,
    summary,
    snapshots,
    binaries,
    roots,
    lifecycle,
    targets,
    qualified,
    *,
    graph_checker=None,
):
    graph_checker = graph_checker or check_graph_records
    rank = manifest["rank"]
    require(
        summary["status"] == "capture-qualified"
        and summary["manifest_sha256"] == manifest_sha
        and summary["rank"] == rank
        and summary["mode"] == manifest["mode"]
        and summary["target_sealed"] is True
        and summary["observer_globally_sealed"] is False
        and summary["artifacts"] == summary["loaded_artifacts"] == 7
        and summary["submodules"] == 46
        and summary["model_sha256"]
        == manifest["original_files"][f"rank_{rank}_0/model"]
        and set(summary["snapshots"]) == set(snapshots) == set(PHASES),
        "incomplete serving AOT/capture qualification",
    )
    require(
        len(summary["static_bundles"]) == 7
        and all(r["expected"] == r["loaded"] > 0 for r in summary["static_bundles"]),
        "serving static bundle fallback or missing coverage",
    )
    events = [r["event"] for r in lifecycle]
    require(
        [e for e in events if e != "aot_store_reuse_verified"]
        == [
            "installed",
            "aot_load_begin",
            "aot_qualified_before_forward",
            "capture_begin",
            "capture_qualified",
        ],
        "serving lifecycle incomplete or out of order",
    )
    require(
        all(
            i > events.index("aot_qualified_before_forward")
            for i, e in enumerate(events)
            if e == "aot_store_reuse_verified"
        ),
        "store reuse before qualification",
    )
    results = {}
    for phase in PHASES:
        graph, modules = snapshots[phase]
        # Extra non-root startup/capture modules are recorded, not misclassified
        # as model roots. They must be exact to their receipts and private paths;
        # overwriting an original source still rejects. All root kernels match
        # the completed qualification, including every non-target selection.
        result = check_root_records(
            manifest, graph, modules, roots, allow_additional_non_roots=True
        )
        result.update(
            graph_checker(
                manifest,
                manifest_sha,
                graph,
                binaries,
                targets,
                require_global_seal=False,
            )
        )
        compare_qualified(graph, qualified)
        results[phase] = result
    return results


def audit_worker(path, manifest, rank):
    folder = Path(manifest["worker_receipts"]) / f"rank-{rank}"
    output = folder / "analysis.json"
    require(not output.exists(), "preserve prior worker audit")
    receipts = {}
    result = dict(
        status="failed",
        scope=__doc__,
        rank=rank,
        mode=manifest["mode"],
        manifest_sha256=sha(path),
        receipts=receipts,
    )

    def read(name, lines=False):
        name = Path(name)
        receipts[str(name)] = sha(name)
        return json_lines(name) if lines else json.loads(name.read_text())

    try:
        summary = read(folder / "summary.json")
        require(
            summary["source_sha256"]
            == sha(Path(__file__).with_name("glm53_geometry_serving.py")),
            "serving hook source changed",
        )
        snapshots = {}
        for phase in PHASES:
            record = summary["snapshots"][phase]
            for kind in ("bindings", "modules"):
                expected = folder / f"{phase}-{kind}.json"
                require(
                    record[kind] == str(expected)
                    and record[kind + "_sha256"] == sha(expected),
                    "serving snapshot path/digest changed",
                )
            snapshots[phase] = (read(record["bindings"]), read(record["modules"]))
        binaries = read(folder / "binary-loads.jsonl", True)
        roots = read(folder / "artifact-roots.jsonl", True)
        lifecycle = read(folder / "lifecycle.jsonl", True)
        graph_checker = None
        if manifest.get("serving_schema") in (
            "glm53-kv-serving-v1",
            "glm53-indexer-correction-serving-v1",
            "glm53-prompt-score-serving-v1",
        ):
            from benchmarks.kernels.audit_glm53_kv_loader import (
                check_graph_records as graph_checker,
            )

            target_events = read(folder / "loader-events.jsonl", True)
            if manifest["serving_schema"] in (
                "glm53-indexer-correction-serving-v1",
                "glm53-prompt-score-serving-v1",
            ):
                from benchmarks.kernels import audit_glm53_indexer_correction_loader
                from benchmarks.kernels.glm53_indexer_correction_serving import (
                    check_runtime_records,
                )

                graph_checker = partial(
                    graph_checker, workflow=audit_glm53_indexer_correction_loader
                )
                result["runtime_envelope"] = check_runtime_records(
                    manifest, summary, read(folder / "runtime-envelope.jsonl", True)
                )
        else:
            target_events = [
                read(Path(manifest["receipts"]) / f"target-{i}/rank-{rank}.jsonl", True)
                for i in range(len(manifest["targets"][str(rank)]))
            ]
        reference = manifest["qualified_graphs"][manifest["mode"]][str(rank)]
        require(
            sha(reference["path"]) == reference["sha256"],
            "qualified graph receipt changed",
        )
        qualified = read(reference["path"])
        result["phases"] = check_worker(
            dict(manifest, rank=rank),
            sha(path),
            summary,
            snapshots,
            binaries,
            roots,
            lifecycle,
            target_events,
            qualified,
            graph_checker=graph_checker,
        )
        result["status"] = "complete"
    except Exception as error:
        result["error"] = repr(error)
    folder.mkdir(parents=True, exist_ok=True)
    result["auditor_sha256"] = sha(__file__)
    write_new(output, result)
    return result


def audit(path):
    path = path.resolve()
    manifest = json.loads(path.read_text())
    policy = serving_policy(manifest)
    require(
        manifest["aot_qualification_sha256"] == policy.AOT_PAIR_SHA
        and (
            policy.SERVING_SCHEMA != SERVING_SCHEMA
            or manifest["aot_closure_sha256"] == AOT_CLOSURE_SHA
        )
        and manifest["worker_receipts"] == str(path.parent / "worker-receipts")
        and manifest["receipts"] == str(path.parent / "receipts"),
        "unexpected serving qualification manifest",
    )
    verify(manifest)
    results = [audit_worker(path, manifest, rank) for rank in range(4)]
    verify(manifest)
    record = dict(
        status="complete"
        if all(r["status"] == "complete" for r in results)
        else "failed",
        scope=__doc__,
        manifest_sha256=sha(path),
        workers=[
            dict(
                rank=r["rank"],
                status=r["status"],
                analysis_sha256=sha(
                    Path(manifest["worker_receipts"])
                    / f"rank-{r['rank']}/analysis.json"
                ),
            )
            for r in results
        ],
        auditor_sha256=sha(__file__),
    )
    write_new(path.parent / "worker-analysis.json", record)
    print(json.dumps(record), flush=True)
    return record["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.manifest) else 1)


if __name__ == "__main__":
    main()
