#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prescribed fresh-A/fresh-B/cached-A GLM53 qualification, never a best-start run.

Preparation creates only fresh private caches and a frozen manifest. Audits are
offline; no generated code is executed. Every failed gate remains in its output.
"""

import argparse
import copy
import itertools
import json
import math
import statistics
import subprocess
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import (
    compare,
    delta_summary,
    input_identity,
    require,
    verified_quality,
)
from benchmarks.analyze_glm53_reduction_receipts import audit_receipt, sha
from benchmarks.benchmark_glm53_prefill import summarize_response

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "perf/results/2026-09-09"
SERIES = ROOT / "perf/results/2026-09-10/deterministic-no-combo-serving"
ARMS = ("fresh-a", "fresh-b", "cached-a")
PROMPT = Path("/home/tiny/.local/scratch/slimserve-glm53/prompt-source.txt")


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def snapshot(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}


def save(path, result):
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def prepare():
    require(not SERIES.exists(), "new series directory required")
    require(not git("status", "--short"), "clean source required")
    original = read(BASE / "native-order-quality/summary.json")
    frontend = SERIES.parent / "runtime-control/no-combo-frontend-analysis.json"
    require(
        sha(frontend)
        == "6facb3efedd788e3dc527fd3687d09ddefd61096a528d8cc876c8b55e313a6f9",
        "frontend qualification changed",
    )
    sources = set(original["benchmark_implementation_sha256"]) | {
        "slimserve/profiles.json",
        "slimserve/deterministic_reductions.py",
        "slimserve/reduction_receipts.py",
        "slimserve/rmsnorm_diagnostic.py",
        "benchmarks/kernels/check_glm53_cached_rmsnorm.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/config/compilation.py",
        "vllm/compilation/compiler_interface.py",
        "benchmarks/analyze_glm53_deterministic_serving.py",
        "benchmarks/analyze_glm53_reduction_receipts.py",
        "benchmarks/analyze_glm53_quality_pair.py",
    }
    cache_manifest = (
        BASE / "rmsnorm-complete-graph-serving-caches/control/manifest.json"
    )
    manifest = read(cache_manifest)
    require(
        snapshot(Path(manifest["original_namespace"])) == manifest["original_files"],
        "original cache changed",
    )
    require(sha(PROMPT) == original["source_sha256"], "prompt changed")
    for name, digest in original["runtime"]["native_sha256"].items():
        require(sha(ROOT / name) == digest, "native binary changed")
    SERIES.mkdir()
    for name in ("cache-a", "cache-b"):
        (SERIES / name).mkdir()
    result = dict(
        status="prepared",
        git_commit=git("rev-parse", "HEAD"),
        sources={name: sha(ROOT / name) for name in sorted(sources)},
        native_sha256=original["runtime"]["native_sha256"],
        source_sha256=original["source_sha256"],
        original_manifest=str(cache_manifest),
        original_manifest_sha256=sha(cache_manifest),
        heuristic_sha256=read(
            read(
                SERIES.parent / "deterministic-reduction-no-combo-frontend/summary.json"
            )["receipt"]
        )["heuristic_source_sha256"],
        compiler_options={
            "deterministic": True,
            "combo_kernels": False,
            "benchmark_combo_kernel": False,
        },
        frontend_audit_sha256=sha(frontend),
        caches={name: snapshot(SERIES / name) for name in ("cache-a", "cache-b")},
        arms=list(ARMS),
    )
    save(SERIES / "manifest.json", result)
    print(json.dumps(result, indent=2))


def preflight(arm):
    manifest = read(SERIES / "manifest.json")
    require(not git("status", "--short"), "clean source required")
    require(git("rev-parse", "HEAD") == manifest["git_commit"], "source changed")
    require(not (SERIES / arm).exists(), "preserve prior start")
    if arm != "fresh-a":
        previous = "fresh-a" if arm == "fresh-b" else "fresh-b"
        require(
            read(SERIES / f"{previous}-analysis.json")["status"] == "complete",
            "preceding arm must pass",
        )
    cache = SERIES / ("cache-b" if arm == "fresh-b" else "cache-a")
    files = snapshot(cache)
    expected = (
        read(SERIES / "fresh-a-analysis.json")["cache_files"]
        if arm == "cached-a"
        else {}
    )
    require(files == expected, "fresh/return cache precondition failed")
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "GPU workload active",
    )
    result = dict(
        status="ready",
        arm=arm,
        cache_root=str(cache),
        cache_files=files,
        git_commit=manifest["git_commit"],
        manifest_sha256=sha(SERIES / "manifest.json"),
    )
    save(SERIES / f"{arm}-preflight.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "cache_files"}))


def timing(run, summary):
    require(
        len(run["warmups"]) == 3 and len(run["measurements"]) == 9,
        "wrong timing matrix",
    )
    require(
        [(m["repeat"], m["concurrency"]) for m in run["measurements"]]
        == list(itertools.product((1, 2, 3), (1, 8, 16))),
        "timing order changed",
    )
    receipts, salts, tps = [], set(), {c: [] for c in (1, 8, 16)}
    for warmup, names in (
        (True, run["warmups"]),
        (False, [m["path"] for m in run["measurements"]]),
    ):
        for name in names:
            path = ROOT / name
            data = read(path)
            c = len(data["requests"])
            require(
                c in tps and data["cache_policy"] == "isolated-cold",
                "wrong workload/cache policy",
            )
            for request in data["requests"]:
                usage = request["usage"]
                require(
                    tuple(
                        usage[k]
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    )
                    == (1000, 300, 1300),
                    "wrong exact token counts",
                )
                require(
                    usage["prompt_tokens_details"]["cached_tokens"] == 0, "cache hit"
                )
                require(
                    len(request["token_ids"]) == 300
                    and not request["replacement_characters"],
                    "invalid output",
                )
                salt = request["cache_salt"]
                require(bool(salt) and salt not in salts, "reused cache salt")
                salts.add(salt)
            require(
                math.isclose(
                    300 * c / data["wall_seconds"],
                    data["aggregate_output_tps"],
                    rel_tol=0,
                    abs_tol=1e-12,
                ),
                "TPS mismatch",
            )
            if not warmup:
                tps[c].append(data["aggregate_output_tps"])
            receipts.append(
                dict(path=str(path), sha256=sha(path), warmup=warmup, requests=c)
            )
    require(
        sum(r["requests"] for r in receipts if r["warmup"]) == 25,
        "wrong warmup request count",
    )
    require(
        sum(r["requests"] for r in receipts if not r["warmup"]) == 75,
        "wrong timed request count",
    )
    for c, values in tps.items():
        require(
            summary["aggregates"][str(c)]
            == dict(
                count=3,
                median=statistics.median(values),
                min=min(values),
                max=max(values),
            ),
            "TPS aggregate mismatch",
        )
    return receipts


def prefill(run, source_sha256):
    path = ROOT / run["prefill_path"] / "summary.json"
    document = read(path)
    require(
        document["status"] == "complete" and document["source_sha256"] == source_sha256,
        "invalid prefill summary",
    )
    expected = [
        (c, warm, rep)
        for c in (32768, 131072)
        for warm, reps in ((True, (1,)), (False, (1, 2, 3)))
        for rep in reps
    ]
    require(
        [(r["context"], r["warmup"], r["repeat"]) for r in document["requests"]]
        == expected,
        "wrong prefill matrix",
    )
    receipts, prompts, salts = [], {}, set()
    for row in document["requests"]:
        request_path = ROOT / row["path"]
        request = read(request_path)
        require(
            row["status"] == request["status"] == "complete" and request["done"],
            "incomplete prefill",
        )
        require(request["request"]["max_tokens"] == 8, "prefill output count changed")
        actual = summarize_response(request)
        require(
            actual == request["summary"] == row["summary"], "prefill metrics mismatch"
        )
        require(actual["prompt_tokens"] == row["context"], "wrong prefill context")
        salt = request["request"]["cache_salt"]
        require(salt and salt not in salts, "reused prefill salt")
        salts.add(salt)
        ids = request["request"]["prompt"]
        require(
            ids == prompts.setdefault(row["context"], ids), "prefill prompt changed"
        )
        receipts.append(dict(path=str(request_path), sha256=sha(request_path)))
    for context in (32768, 131072):
        rows = [
            r["summary"]
            for r in document["requests"]
            if r["context"] == context and not r["warmup"]
        ]
        aggregate = {
            key: dict(
                median=statistics.median(r[key] for r in rows),
                min=min(r[key] for r in rows),
                max=max(r[key] for r in rows),
            )
            for key in rows[0]
        }
        require(
            aggregate
            == document["aggregates"][str(context)]
            == run["prefill"][str(context)],
            "prefill aggregate mismatch",
        )
    return dict(
        path=str(path),
        sha256=sha(path),
        requests=receipts,
        aggregates=document["aggregates"],
    )


def quality(summary):
    rows = summary["runs"][0]["quality_passes"]
    require([r["repeat"] for r in rows] == [1, 2, 3], "three quality passes required")
    documents, values = [], []
    for row in rows:
        document = read(ROOT / row["path"])
        text = verified_quality(document)
        require(
            row["status"] == "complete" and row["summary"] == document["summary"],
            "quality summary mismatch",
        )
        require(document["summary"]["all_needles_rank_first"], "needle ranking failed")
        responses = [r["response"] for r in document["text"]] + [
            c["response"] for r in document["needles"] for c in r["candidates"]
        ]
        require(
            len(responses) == 56
            and all(
                r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                for r in responses
            ),
            "quality cache proof failed",
        )
        needles = [
            s
            for r in document["needles"]
            for c in r["candidates"]
            for s in c["per_token"]
        ]
        require(len(needles) == 168, "wrong needle score count")
        documents.append(document)
        values.append(
            dict(
                text=text,
                needles=needles,
                path=row["path"],
                sha256=sha(ROOT / row["path"]),
            )
        )
    identity = input_identity(documents[0])
    require(
        all(input_identity(d) == identity for d in documents), "quality inputs changed"
    )
    return documents, values


def audit(arm, result):
    manifest = read(SERIES / "manifest.json")
    preflight_path = SERIES / f"{arm}-preflight.json"
    ready = read(preflight_path)
    require(
        ready["status"] == "ready"
        and ready["arm"] == arm
        and ready["git_commit"] == manifest["git_commit"]
        and ready["manifest_sha256"] == sha(SERIES / "manifest.json"),
        "missing frozen cache preflight",
    )
    result["preflight_sha256"] = sha(preflight_path)
    require(
        not git("status", "--short")
        and git("rev-parse", "HEAD") == manifest["git_commit"],
        "source freeze violated",
    )
    for name, digest in {**manifest["sources"], **manifest["native_sha256"]}.items():
        require(sha(ROOT / name) == digest, f"source/native changed: {name}")
    require(
        sha(Path(manifest["original_manifest"]))
        == manifest["original_manifest_sha256"],
        "cache manifest changed",
    )
    original_cache = read(manifest["original_manifest"])
    require(
        snapshot(Path(original_cache["original_namespace"]))
        == original_cache["original_files"],
        "original cache changed",
    )
    require(
        sha(
            ROOT
            / ".venv/lib/python3.12/site-packages/torch/_inductor/runtime"
            / "triton_heuristics.py"
        )
        == manifest["heuristic_sha256"],
        "compiler runtime changed",
    )
    original = read(BASE / "native-order-quality/summary.json")
    folder = SERIES / arm
    summary = read(folder / "summary.json")
    require(
        summary["status"] == "complete"
        and not summary["git_status"]
        and summary["git_commit"] == manifest["git_commit"],
        "incomplete or unfrozen run",
    )
    require(
        summary["diagnostic_only"] and not summary["throughput_is_baseline_eligible"],
        "incorrect baseline eligibility",
    )
    require(
        summary["compatible_profiles"] == ["glm53-nvfp4-4"]
        and len(summary["runs"]) == 1,
        "wrong profile/start count",
    )
    require(
        summary["source_sha256"] == manifest["source_sha256"], "wrong prompt source"
    )
    plan = copy.deepcopy(original["plan"])
    plan["engine"]["compilation_config"]["inductor_compile_config"] = manifest[
        "compiler_options"
    ]
    require(summary["plan"] == plan, "recipe/plan changed")
    cache = SERIES / ("cache-b" if arm == "fresh-b" else "cache-a")
    env = {
        **original["environment"],
        "VLLM_CACHE_ROOT": str(cache),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
    }
    require(summary["environment"] == env, "unexpected environment")
    require(
        summary["runtime"]
        == {
            **original["runtime"],
            "gpu_before_start": summary["runtime"]["gpu_before_start"],
        },
        "runtime identity changed",
    )
    require(
        len(summary["benchmark_implementation_sha256"]) == 24,
        "source receipt coverage changed",
    )
    require(
        all(
            manifest["sources"][name] == digest
            for name, digest in summary["benchmark_implementation_sha256"].items()
        ),
        "benchmark sources changed",
    )
    run = summary["runs"][0]
    require(
        run["status"] == run["teardown"]["status"] == "complete"
        and run["teardown"]["returncode"] == 0,
        "incomplete teardown",
    )
    require(
        run["teardown"]["gpu_release"]["status"] == "complete"
        and not run["teardown"]["gpu_release"]["samples"][-1]["owned_active_pids"],
        "GPU not released",
    )
    require("--deterministic-reductions" in run["argv"], "candidate flag absent")
    require(
        run["canaries"]["text"]["answer"] == "4"
        and run["canaries"]["image"]["answer"].lower() == "red",
        "canary failure",
    )
    result.update(
        summary_sha256=sha(folder / "summary.json"),
        startup_seconds=run["startup_seconds"],
        aggregates=summary["aggregates"],
        timing=timing(run, summary),
        prefill=prefill(run, manifest["source_sha256"]),
        teardown=run["teardown"],
    )
    prior = read(SERIES / "fresh-a-analysis.json") if arm != "fresh-a" else None
    if prior:
        require(prior["status"] == "complete", "first arm must pass before continuing")
    previous_paths = (
        {r["path"] for r in prior["graph_receipts"]} if arm == "cached-a" else set()
    )
    paths = [
        p
        for p in (cache / "glm53-reduction-receipts").glob("*.json")
        if str(p) not in previous_paths
    ]
    require(len(paths) == 4, "four current TP rank receipts required")
    graphs, receipts, ranks = [], [], set()
    for path in sorted(paths):
        document = read(path)
        require(
            document["compiler_options"]
            == {
                **manifest["compiler_options"],
                "enable_auto_functionalized_v2": False,
            },
            "live compiler options changed",
        )
        rank = document["rank"]
        require(rank in range(4) and rank not in ranks, "duplicate/wrong TP rank")
        ranks.add(rank)
        graphs.extend(
            audit_receipt(
                document,
                cache_root=cache,
                rank=rank,
                source_sha256=manifest["sources"]["slimserve/reduction_receipts.py"],
                heuristic_sha256=manifest["heuristic_sha256"],
            )
        )
        receipts.append(dict(path=str(path), sha256=sha(path), rank=rank))
    result.update(graphs=graphs, graph_receipts=receipts)
    if prior:
        # Module/object alias multiplicity is process-local, not a policy choice.
        canonical = lambda rows: {json.dumps(r, sort_keys=True) for r in rows}
        result["graph_choices_match_fresh_a"] = canonical(graphs) == canonical(
            prior["graphs"]
        )
    documents, scores = quality(summary)
    result["quality_receipts"] = [
        {k: v for k, v in s.items() if k not in ("text", "needles")} for s in scores
    ]
    result["within_start_exact"] = all(
        (s["text"], s["needles"]) == (scores[0]["text"], scores[0]["needles"])
        for s in scores
    )
    comparisons = []
    for reference in (
        "native-order-quality",
        "native-order-serialized",
        "native-order-async-return",
        "stable-align-quality-diagnostic",
    ):
        reference_summary = read(BASE / reference / "summary.json")
        ref_docs, ref_scores = quality(reference_summary)
        require(
            all(input_identity(d) == input_identity(documents[0]) for d in ref_docs),
            "reference inputs differ",
        )
        comparisons.extend(
            dict(
                reference=reference,
                a=a["path"],
                b=b["path"],
                text=delta_summary(a["text"], b["text"]),
                needles=delta_summary(a["needles"], b["needles"]),
                exact=a["text"] == b["text"] and a["needles"] == b["needles"],
            )
            for a, b in itertools.product(ref_scores, scores)
        )
    result["historical_comparisons"] = comparisons
    controls = [
        read(
            ROOT
            / read(BASE / name / "summary.json")["runs"][0]["quality_passes"][0]["path"]
        )
        for name in ("native-order-quality", "native-order-async-return")
    ]
    result["unchanged_window_quality_gate"] = compare(controls, documents)
    if prior:
        _, a_scores = quality(read(SERIES / "fresh-a/summary.json"))
        result["cross_start_exact"] = all(
            a["text"] == b["text"] and a["needles"] == b["needles"]
            for a, b in itertools.product(a_scores, scores)
        )
    log_path = folder / "boot-1/server.log"
    log = log_path.read_text()
    result.update(
        server_log_sha256=sha(log_path),
        warnings=[
            line
            for line in log.splitlines()
            if any(
                s in line.lower() for s in ("warning", "allocation failed", "leaked")
            )
        ],
        cache_files=snapshot(cache),
        manifest_sha256=sha(SERIES / "manifest.json"),
    )
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "GPU still occupied",
    )
    require(result["within_start_exact"], "within-start score equality failed")
    require(
        result["unchanged_window_quality_gate"]["passed"],
        "unchanged window quality gate failed",
    )
    if prior:
        require(
            result["cross_start_exact"],
            "independent/cached start score equality failed",
        )
        require(
            result["graph_choices_match_fresh_a"],
            "graph reduction choices changed across starts",
        )
    result["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", *ARMS))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    if args.preflight:
        preflight(args.stage)
        return
    path = SERIES / f"{args.stage}-analysis.json"
    require(not path.exists(), "preserve prior audit")
    result = dict(
        status="running",
        arm=args.stage,
        diagnostic_only=True,
        auditor_sha256=sha(__file__),
    )
    try:
        audit(args.stage, result)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save(path, result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "arm",
                        "error",
                        "startup_seconds",
                        "within_start_exact",
                        "cross_start_exact",
                        "graph_choices_match_fresh_a",
                        "aggregates",
                    )
                    if k in result
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
