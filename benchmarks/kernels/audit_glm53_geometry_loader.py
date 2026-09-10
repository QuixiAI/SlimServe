# SPDX-License-Identifier: Apache-2.0
"""Offline join of actual graph, binary-load and controller receipts. No GPU use."""

import argparse
import base64
import json
from pathlib import Path

from benchmarks.kernels.audit_glm53_geometry_graphs import (
    compare_unrelated,
    exported_call,
    run_symbols,
)
from benchmarks.kernels.check_glm53_attention_norms import require, verify
from benchmarks.kernels.check_glm53_geometry_loader import (
    ORDER,
    check_launch,
    prior_receipts,
    read_json,
    read_manifest,
)
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from slimserve.rmsnorm_diagnostic import expected_receipt, sha


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_root_records(manifest, graphs, modules, events):
    """Join the complete module catalog to observed serialized artifact roots.

    Extra call exports are imported helpers only when they are original-exact
    sources AND absent from the complete actual artifact-root set. They cannot
    substitute for a missing root. Root launcher coverage is still audited below.
    """
    private = Path(manifest["private_namespace"])
    expected = manifest["artifact_roots"][str(manifest["rank"])]
    by_artifact = {row["artifact"]: row for row in expected}
    require(
        len(by_artifact) == len(expected)
        and {row["graph"]: row["graph_sha256"] for row in expected}
        == manifest["expected_graphs"][str(manifest["rank"])],
        "artifact provenance and expected root sources differ",
    )
    by_module = {}
    for index, row in enumerate(modules, 1):
        path = Path(row["path"])
        require(
            row["module_index"] == index
            and path.resolve() == path
            and path.is_relative_to(private),
            f"module inventory has invalid path/index: {path}",
        )
        relative = str(path.relative_to(private))
        require(
            row.get("source_sha256")
            == sha(path)
            == manifest["original_files"].get(relative),
            f"imported module differs from original source: {path}",
        )
        if row["callable"]:
            definition, cls, _ = exported_call(path.read_text())
            require(
                row["call_filename"] == str(path)
                and row["call_line"] == definition.lineno
                and row["bound"] is (cls is not None),
                f"imported call export differs from source: {path}",
            )
        by_module[index] = row
    phases = {key: [] for key in by_artifact}
    bound = {}
    for event in events:
        key = event["artifact"]
        require(
            key in by_artifact
            and {k: event[k] for k in by_artifact[key]} == by_artifact[key],
            "observed serialized artifact provenance changed",
        )
        phases[key].append(event["event"])
        if event["event"] == "artifact_root_bound":
            row = by_module[event["module_index"]]
            require(
                row["path"] == str(private / event["graph"])
                and row["source_sha256"] == event["graph_sha256"]
                and row["callable"],
                "artifact root does not join actual loaded module",
            )
            bound[key] = event["module_index"]
    require(
        all(
            p
            == [
                "artifact_deserialize_begin",
                "artifact_root_bound",
                "artifact_deserialize_complete",
            ]
            for p in phases.values()
        )
        and len(set(bound.values())) == len(expected),
        "incomplete or repeated artifact-root load sequence",
    )
    indices = {
        row["graph"]: i + 1
        for i, row in enumerate(sorted(expected, key=lambda r: r["artifact"]))
    }
    require(
        all(row["module_index"] == indices[row["graph"]] for row in graphs["bindings"]),
        "root launcher inventory differs from observed artifact modules",
    )
    return dict(
        artifact_roots=len(expected),
        imported_modules=len(modules),
        non_root_imported_modules=len(modules) - len(expected),
    )


def check_records(
    manifest, manifest_sha, summary, graphs, binary_events, target_events
):
    rank, mode = manifest["rank"], manifest["mode"]
    require(
        summary["status"] == "complete"
        and summary["rank"] == rank
        and summary["mode"] == mode
        and summary["manifest_sha256"] == manifest_sha
        and summary["model_forward_calls"] == summary["weight_tensors_loaded"] == 0
        and summary["artifacts"] == summary["loaded_artifacts"] == 7
        and summary["submodules"] == 46
        and summary["original_cache_unchanged"] is True,
        "incomplete or mismatched AOT qualification",
    )
    require(
        len(summary["static_bundles"]) == 7
        and all(b["expected"] == b["loaded"] > 0 for b in summary["static_bundles"]),
        "incomplete static bundle coverage",
    )
    loaded = [r for r in binary_events if r["event"] == "binary_loaded"]
    sealed = [r for r in binary_events if r["event"] == "binary_observer_sealed"]
    require(
        len(sealed) == 1
        and sealed[0]["rank"] == rank
        and sealed[0]["objects"] == len(loaded) == summary["observed_binary_objects"],
        "binary observer not completely sealed",
    )
    require(
        [r["index"] for r in loaded] == list(range(1, len(loaded) + 1)),
        "binary object indices changed",
    )
    images = {r["index"]: r for r in loaded}
    private = Path(manifest["private_namespace"])
    original = Path(manifest["original_namespace"])
    seen_images, is_sealed = set(), False
    for event in binary_events:
        require(event["rank"] == rank, "binary event rank changed")
        require(
            event["event"]
            in ("binary_loaded", "binary_load_reuse", "binary_observer_sealed"),
            "unexpected binary event",
        )
        if event["event"] == "binary_load_reuse":
            require(event["index"] in seen_images, "unobserved reused binary")
        elif event["event"] == "binary_loaded":
            require(not is_sealed, "binary load after global seal")
            seen_images.add(event["index"])
        else:
            is_sealed = True
    for event in loaded:
        path = Path(event["path"])
        metadata = event["metadata"]
        key = base64.b32encode(bytes.fromhex(metadata["hash"])).decode().rstrip("=")
        require(
            path.resolve() == path
            and path
            == private / f"inductor_cache/triton/{rank}/{key}/{metadata['name']}.cubin",
            "binary load outside exact rank-local cache",
        )
        require(
            sha(path) == event["cubin_sha256"]
            and len(event["handles"]) == 2
            and all(type(h) is int and h != 0 for h in event["handles"]),
            "binary load receipt changed",
        )
    targets = {t["relative"]: t for t in manifest["targets"][str(rank)]}
    expected = {
        (str(Path(u["graph"]).relative_to(original)), u["symbol"], t["relative"])
        for t in targets.values()
        for u in t["static_graph_uses"]
    }
    expected_graphs = manifest["expected_graphs"][str(rank)]
    actual, seen_graphs, seen_rows = set(), set(), set()
    for row in graphs["bindings"]:
        graph, source = private / row["graph"], private / row["source"]
        require(
            row["graph"] in expected_graphs
            and sha(graph) == row["graph_sha256"] == expected_graphs[row["graph"]],
            "graph report source changed",
        )
        require(
            row["source"] in manifest["original_files"]
            and sha(source)
            == row["source_sha256"]
            == manifest["original_files"][row["source"]],
            "kernel report source changed",
        )
        key = (row["module_index"], row["graph"], row["symbol"])
        require(key not in seen_rows, "duplicate actual module/global binding")
        seen_rows.add(key)
        seen_graphs.add(row["graph"])
        referenced = row["symbol"] in run_symbols(graph.read_text())
        require(
            row["referenced"] == referenced, "reported executable graph use changed"
        )
        image = images[row["observed_binary_index"]]
        require(len(row["selected"]) == 1, "unresolved graph launcher")
        selected = row["selected"][0]
        require(
            selected["hash"] == Path(image["path"]).parent.name
            and selected["config"]["num_warps"] == image["metadata"]["num_warps"]
            and row["cubin_sha256"] == image["cubin_sha256"],
            "graph/binary observation join changed",
        )
        target = targets.get(row["source"])
        require(
            row["target"] is (target is not None),
            "reported target classification changed",
        )
        if target is not None:
            key = (row["graph"], row["symbol"], row["source"])
            require(
                key in expected
                and referenced
                and row["selected"] == expected_receipt(target["configs"][mode])
                and row["cubin_sha256"] == target["configs"][mode]["cubin_sha256"],
                "wrong target selection or coverage",
            )
            actual.add(key)
    require(
        seen_graphs == set(expected_graphs)
        and graphs["graphs"] == len(seen_graphs)
        and actual == expected
        and graphs["target_bindings"] == len(expected),
        "incomplete actual graph coverage",
    )
    require(len(target_events) == len(targets), "missing target receipt streams")
    for target, events in zip(targets.values(), target_events):
        begin = [r for r in events if r["event"] == "begin"]
        seal = [r for r in events if r["event"] == "sealed"]
        require(
            len(begin) == len(seal) == 1
            and begin[0]["manifest_sha256"] == manifest_sha
            and begin[0]["rank"] == rank
            and begin[0]["mode"] == mode
            and seal[0]["sources"] == 1
            and seal[0]["targets"] > 0,
            "controller stream not qualified and sealed",
        )
        selections = [r for r in events if r["event"] == "launcher" and r["target"]]
        expected_selected = expected_receipt(target["configs"][mode])
        require(
            bool(selections)
            and all(
                r["rank"] == rank
                and r["filename"] == str(private / target["relative"])
                and r["after"] == expected_selected
                for r in selections
            ),
            "controller target selection changed",
        )
        require(
            [r["resolution_index"] for r in selections]
            == list(range(1, len(selections) + 1))
            and seal[0]["resolutions"] == len(selections),
            "controller resolution sequence changed",
        )
        binding_ids = set()
        for row in selections:
            first = row["binding_index"] not in binding_ids
            require(
                row["repeated"] is not first
                and row["before"]
                == (
                    expected_receipt(target["configs"]["control"])
                    if first
                    else expected_selected
                ),
                "controller first/reused selection changed",
            )
            binding_ids.add(row["binding_index"])
        require(
            binding_ids == set(range(1, seal[0]["targets"] + 1)),
            "controller binding indices changed",
        )
        bindings = [r for r in events if r["event"] == "graph_binding"]
        actual_bindings = {
            (r["graph"], r["symbol"])
            for r in graphs["bindings"]
            if r["source"] == target["relative"]
        }
        recorded = set()
        for row in bindings:
            graph = Path(row["module"])
            require(
                graph.is_relative_to(private)
                and sha(graph) == row["module_sha256"]
                and row["selected"] == expected_selected
                and row["filename"] == str(private / target["relative"]),
                "controller graph receipt changed",
            )
            recorded.add((str(graph.relative_to(private)), row["symbol"]))
        require(
            recorded == actual_bindings,
            "controller and independent graph inventories differ",
        )
        actual_count = sum(
            row["source"] == target["relative"] for row in graphs["bindings"]
        )
        coverage = [row for row in events if row["event"] == "graph_coverage"]
        require(
            bool(coverage)
            and all(
                row["rank"] == rank and row["bindings"] == actual_count
                for row in coverage
            ),
            "controller coverage count differs from actual globals",
        )
    return dict(
        graphs=len(seen_graphs),
        target_bindings=len(actual),
        bound_launchers=len(graphs["bindings"]),
        observed_binary_objects=len(images),
    )


def audit(path):
    path = path.resolve()
    output = path.parent / "run"
    require(not (output / "analysis.json").exists(), "preserve prior audit")
    receipts = {}

    def read(name, lines=False):
        receipts[str(name)] = sha(name)
        return json_lines(name) if lines else read_json(name)

    report = dict(
        status="failed", scope=__doc__, manifest_sha256=sha(path), receipts=receipts
    )
    try:
        manifest, preparation = read_manifest(path)
        summary = read(output / "summary.json")
        report["summary_sha256"] = sha(output / "summary.json")
        require(
            summary["source_sha256"]
            == sha(Path(__file__).with_name("check_glm53_geometry_loader.py")),
            "runner source changed",
        )
        require(
            summary.get("model_sha256")
            == manifest["original_files"][f"rank_{manifest['rank']}_0/model"],
            "cached model receipt changed",
        )
        require(
            summary["prior_receipts"] == prior_receipts(path, manifest, preparation),
            "predecessor gate changed",
        )
        graphs = read(output / "graph-bindings.json")
        modules = read(output / "module-inventory.json")
        roots = read(output / "artifact-roots.jsonl", True)
        report.update(check_root_records(manifest, graphs, modules, roots))
        binaries = read(output / "binary-loads.jsonl", True)
        targets = [
            read(
                Path(manifest["receipts"])
                / f"target-{i}/rank-{manifest['rank']}.jsonl",
                True,
            )
            for i in range(len(manifest["targets"][str(manifest["rank"])]))
        ]
        report.update(
            check_records(manifest, sha(path), summary, graphs, binaries, targets)
        )
        if manifest["mode"] == "geometry":
            control = (
                path.parent.parent
                / f"control-rank{manifest['rank']}/run/graph-bindings.json"
            )
            compare_unrelated(read(control), graphs)
            report["non_target_exact_vs_control"] = True
        verify(manifest)
        report["status"] = "complete"
    except Exception as error:
        report["error"] = repr(error)
    report["auditor_sha256"] = sha(__file__)
    write_new(output / "analysis.json", report)
    print(json.dumps(report), flush=True)
    return report["status"] == "complete"


def compare(series):
    require(not (series / "pair-analysis.json").exists(), "preserve final audit")
    reports = []
    for mode, rank in ORDER:
        path = series / f"{mode}-rank{rank}/manifest.json"
        manifest, preparation = read_manifest(path)
        check_launch(path.parent, sha(path))
        report = read_json(path.parent / "run/analysis.json")
        require(
            report["status"] == "complete"
            and report["manifest_sha256"] == sha(path)
            and report["summary_sha256"] == sha(path.parent / "run/summary.json"),
            "all eight loads and audits must pass",
        )
        require(
            mode == "control" or report["non_target_exact_vs_control"] is True,
            "unrelated selections not compared",
        )
        for filename, digest in report["receipts"].items():
            require(sha(filename) == digest, "audited receipt changed")
        prior_receipts(path, manifest, preparation)
        reports.append(
            dict(
                mode=mode,
                rank=rank,
                analysis_sha256=sha(path.parent / "run/analysis.json"),
                **{
                    k: report[k]
                    for k in ("graphs", "target_bindings", "bound_launchers")
                },
            )
        )
    result = dict(
        status="complete",
        scope=(
            "Actual AOT-loader/binary/graph coverage only; "
            "not model correctness, numerical execution or TPS"
        ),
        runs=reports,
        non_target_exact=True,
        auditor_sha256=sha(__file__),
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
