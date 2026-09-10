# SPDX-License-Identifier: Apache-2.0
"""Offline KV AOT graph/driver/controller receipt joins. No GPU use."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.kernels.audit_glm53_geometry_graphs import compare_unrelated
from benchmarks.kernels.audit_glm53_geometry_loader import (
    binary_records,
    check_aot_summary,
    check_root_records,
    graph_records,
    json_lines,
)
from benchmarks.kernels.check_glm53_attention_norms import require, verify
from benchmarks.kernels.check_glm53_geometry_loader import check_launch, read_json
from benchmarks.kernels.check_glm53_kv_loader import (
    ORDER as ORDER,
)
from benchmarks.kernels.check_glm53_kv_loader import (
    prior_receipts as prior_receipts,
)
from benchmarks.kernels.check_glm53_kv_loader import (
    read_manifest as read_manifest,
)
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from slimserve.rmsnorm_diagnostic import sha

CANDIDATE_MODE = "kv"
EXTRA_KEY = "kv"
EVENT_PREFIX = "kv"
DISPATCH = "combo_then_kv"
LOADER_SOURCE = Path(__file__).with_name("glm53_kv_loader.py")
RUNNER_SOURCE = Path(__file__).with_name("check_glm53_kv_loader.py")


def check_extra(extra, manifest):
    pass


def check_graph_records(
    manifest,
    manifest_sha,
    graphs,
    binary_events,
    events,
    *,
    require_global_seal=True,
    workflow=None,
):
    api = sys.modules[__name__] if workflow is None else workflow
    images = binary_records(
        manifest, binary_events, require_global_seal=require_global_seal
    )
    rank, mode = manifest["rank"], manifest["mode"]
    private = Path(manifest["private_namespace"])

    def check_target(row, target, image):
        require(
            row["selected"] == [target["selected"]]
            and row["cubin_sha256"] == target["cubin_sha256"]
            and image["metadata"]["name"] == target["kernel"],
            "original combo selection changed",
        )
        if mode == "control":
            require(
                row["dispatch"] == "direct_combo" and row["appended"] is None,
                "control launch path differs",
            )
            return
        require(
            mode == api.CANDIDATE_MODE and row["dispatch"] == api.DISPATCH,
            "extra dispatch missing",
        )
        extra, saved = row["appended"], target[api.EXTRA_KEY]
        observed = images[extra["observed_binary_index"]]
        require(
            extra["source"] == saved["relative"]
            and sha(private / extra["source"])
            == extra["source_sha256"]
            == saved["source_sha256"]
            and extra["selected"] == saved["selected"]
            and extra["cubin_sha256"]
            == observed["cubin_sha256"]
            == saved["cubin_sha256"]
            and Path(observed["path"]).parent.name == saved["selected"]["hash"]
            and observed["metadata"]["name"] == saved["kernel"]
            and observed["metadata"]["num_warps"]
            == saved["selected"]["config"]["num_warps"],
            "appended KV source/config/binary join differs",
        )
        api.check_extra(extra, manifest)

    result = graph_records(manifest, graphs, images, check_target)
    require(
        all(
            "appended" not in r and "dispatch" not in r
            for r in graphs["bindings"]
            if not r["target"]
        ),
        "non-target dispatch intervention",
    )
    prefix = api.EVENT_PREFIX
    seals = [i for i, e in enumerate(events) if e["event"] == f"{prefix}_sealed"]
    require(len(seals) == 1, "KV controller requires exactly one seal")
    seal_index = seals[0]
    require(
        events
        and events[0]["event"] == f"{prefix}_begin"
        and (not require_global_seal or seal_index == len(events) - 1)
        and all(e["rank"] == rank for e in events),
        "KV controller lifecycle incomplete",
    )
    begin, seal = events[0], events[seal_index]
    require(
        begin["mode"] == mode
        and begin["manifest_sha256"] == manifest_sha
        and begin["source_sha256"] == sha(api.LOADER_SOURCE),
        "KV controller identity changed",
    )
    owners, bindings = {}, set()
    for event in events[1:seal_index]:
        if event["event"] == f"{prefix}_binding":
            index = event["binding_index"]
            require(
                index == len(owners) + 1
                and event["mode"] == mode
                and event["resolved_by"] in ("direct", "graph", "upstream"),
                "KV target resolution sequence changed",
            )
            owners[index] = event["source"]
        else:
            require(
                event["event"] == f"{prefix}_graph_binding",
                "unexpected extra-launch controller event",
            )
            require(
                owners.get(event["binding_index"]) == event["source"],
                "KV graph bound before target",
            )
            graph = Path(event["graph"])
            require(graph.is_relative_to(private), "KV graph outside private cache")
            bindings.add(
                (str(graph.relative_to(private)), event["symbol"], event["source"])
            )
    expected = {
        (r["graph"], r["symbol"], r["source"])
        for r in graphs["bindings"]
        if r["target"]
    }
    require(
        bindings == expected
        and seal["targets"] == len(owners) > 0
        and seal["graph_bindings"] == sum(r["target"] for r in graphs["bindings"]),
        "KV controller and independent graph coverage differ",
    )
    # Serving leaves observation/hooks open for later non-target compilation.
    # Only exact callbacks for already-bound globals may follow target sealing.
    prior_graph_events = [
        e for e in events[1:seal_index] if e["event"] == f"{prefix}_graph_binding"
    ]
    require(
        all(e in prior_graph_events for e in events[seal_index + 1 :]),
        "new or changed KV controller event after seal",
    )
    result["appended_launchers"] = (
        sum(r["target"] for r in graphs["bindings"])
        if mode == api.CANDIDATE_MODE
        else 0
    )
    return result


def audit(path, *, workflow=None):
    api = sys.modules[__name__] if workflow is None else workflow
    path = path.resolve()
    output = path.parent / "run"
    require(not (output / "analysis.json").exists(), "preserve prior audit")
    receipts = {}

    def read(name, lines=False):
        receipts[str(name)] = sha(name)
        return json_lines(name) if lines else read_json(name)

    report = dict(status="failed", manifest_sha256=sha(path), receipts=receipts)
    try:
        manifest, preparation = api.read_manifest(path)
        summary = read(output / "summary.json")
        report["summary_sha256"] = sha(output / "summary.json")
        check_aot_summary(
            manifest,
            sha(path),
            summary,
            expected_weights=getattr(api, "EXPECTED_WEIGHTS", 0),
        )
        require(
            summary["source_sha256"] == sha(api.RUNNER_SOURCE)
            and summary["lifecycle_sha256"]
            == sha(Path(__file__).with_name("check_glm53_geometry_loader.py"))
            and summary["model_sha256"]
            == manifest["original_files"][f"rank_{manifest['rank']}_0/model"]
            and summary["prior_receipts"]
            == api.prior_receipts(path, manifest, preparation),
            "runner/model/predecessor identity changed",
        )
        graphs = read(output / "graph-bindings.json")
        report.update(
            check_root_records(
                manifest,
                graphs,
                read(output / "module-inventory.json"),
                read(output / "artifact-roots.jsonl", True),
            )
        )
        report.update(
            check_graph_records(
                manifest,
                sha(path),
                graphs,
                read(output / "binary-loads.jsonl", True),
                read(output / "loader-events.jsonl", True),
                workflow=api,
            )
        )
        require(
            report["observed_binary_objects"] == summary["observed_binary_objects"],
            "binary object count changed",
        )
        require(
            report["graphs"] == 7
            and report["target_bindings"] == 2
            and report["bound_launchers"] == 25,
            "unexpected real AOT binding matrix",
        )
        if manifest["mode"] == api.CANDIDATE_MODE:
            control = (
                path.parent.parent
                / f"control-rank{manifest['rank']}/run/graph-bindings.json"
            )
            compare_unrelated(read(control), graphs)
            report["non_target_exact_vs_control"] = True
        check_leaf = getattr(api, "check_leaf", None)
        if check_leaf is not None:
            report.update(check_leaf(manifest, graphs, summary, read))
        verify(manifest)
        report["status"] = "complete"
    except Exception as error:
        report["error"] = repr(error)
    report["auditor_sha256"] = sha(api.__file__)
    write_new(output / "analysis.json", report)
    print(json.dumps(report), flush=True)
    return report["status"] == "complete"


def compare(series, *, workflow=None):
    api = sys.modules[__name__] if workflow is None else workflow
    require(not (series / "pair-analysis.json").exists(), "preserve final audit")
    reports = []
    for mode, rank in api.ORDER:
        path = series / f"{mode}-rank{rank}/manifest.json"
        manifest, preparation = api.read_manifest(path)
        check_launch(path.parent, sha(path))
        report = read_json(path.parent / "run/analysis.json")
        require(
            report["status"] == "complete"
            and report["manifest_sha256"] == sha(path)
            and report["summary_sha256"] == sha(path.parent / "run/summary.json")
            and (mode == "control" or report["non_target_exact_vs_control"] is True),
            "all eight qualified loads required",
        )
        for filename, digest in report["receipts"].items():
            require(sha(filename) == digest, "audited receipt changed")
        api.prior_receipts(path, manifest, preparation)
        reports.append(
            dict(
                mode=mode,
                rank=rank,
                analysis_sha256=sha(path.parent / "run/analysis.json"),
                **{k: report[k] for k in getattr(api, "PAIR_FIELDS", ())},
                **{
                    k: report[k]
                    for k in (
                        "graphs",
                        "target_bindings",
                        "bound_launchers",
                        "appended_launchers",
                    )
                },
            )
        )
    result = dict(
        status="complete",
        scope=getattr(
            api,
            "SCOPE",
            "AOT loader/binary/graph coverage; not numerical/model/TPS qualification",
        ),
        runs=reports,
        non_target_exact=True,
        auditor_sha256=sha(api.__file__),
    )
    write_new(series / "pair-analysis.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "compare"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        raise SystemExit(0 if audit(args.path) else 1)
    compare(args.path.resolve())


if __name__ == "__main__":
    main()
