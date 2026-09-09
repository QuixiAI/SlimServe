#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit the graph-complete normalization-only pair; preserve failed score gates."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from pathlib import Path

from benchmarks.analyze_glm53_client_streams import analyze
from benchmarks.analyze_glm53_quality_pair import (
    delta_summary,
    input_identity,
    verified_quality,
)
from benchmarks.analyze_glm53_rmsnorm_graphs import audit_rank

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def git(*argv):
    return subprocess.check_output(["git", *argv], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["control", "legacy"])
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    BASE = ROOT / "perf/results/2026-09-09"
    NAMES = dict(
        control="rmsnorm-noop-complete-graph-control",
        legacy="rmsnorm-legacy-complete-graph-only",
    )
    RUN = BASE / NAMES[args.arm]
    OUT = BASE / f"runtime-control/{NAMES[args.arm]}-analysis.json"
    COMMIT = args.commit

    assert not OUT.exists()
    assert git("rev-parse", "HEAD").decode().strip() == COMMIT
    assert not git("status", "--short").strip()
    original = read(BASE / "native-order-quality/summary.json")
    candidate = read(RUN / "summary.json")
    assert candidate["status"] == "complete" and candidate["git_commit"] == COMMIT
    assert not candidate["git_status"]
    assert (
        candidate["diagnostic_only"]
        and not candidate["throughput_is_baseline_eligible"]
    )
    assert len(candidate["runs"]) == 1 and candidate["compatible_profiles"] == [
        "glm53-nvfp4-4"
    ]
    assert candidate["plan"] == original["plan"]
    manifest_path = (
        BASE / f"rmsnorm-complete-graph-serving-caches/{args.arm}/manifest.json"
    )
    manifest = read(manifest_path)
    expected_env = dict(original["environment"])
    expected_env.update(
        VLLM_FORCE_AOT_LOAD="1",
        VLLM_CACHE_ROOT=manifest["cache_root"],
        SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC=args.arm,
        SLIMSERVE_GLM53_RMSNORM_MANIFEST=str(manifest_path),
    )
    assert candidate["environment"] == expected_env
    assert candidate["source_sha256"] == original["source_sha256"]
    sources = candidate["benchmark_implementation_sha256"]
    assert len(sources) == 22
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest
        assert hashlib.sha256(git("show", f"{COMMIT}:{name}")).hexdigest() == digest
        if name in original["benchmark_implementation_sha256"] and name not in (
            "slimserve/cli.py",
            "benchmarks/benchmark_glm53_campaign.py",
        ):
            assert digest == original["benchmark_implementation_sha256"][name]
    assert candidate["runtime"] == {
        **original["runtime"],
        "gpu_before_start": candidate["runtime"]["gpu_before_start"],
    }
    for name, digest in candidate["runtime"]["native_sha256"].items():
        assert sha(ROOT / name) == digest

    old = Path(manifest["original_namespace"])
    snapshot = manifest["original_files"]
    assert {str(p.relative_to(old)) for p in old.rglob("*") if p.is_file()} == set(
        snapshot
    )
    assert all(sha(old / name) == digest for name, digest in snapshot.items())
    private = Path(manifest["private_namespace"])
    private_seed_changes = [
        name for name, digest in snapshot.items() if sha(private / name) != digest
    ]
    assert not [name for name in private_seed_changes if name.endswith(".best_config")]
    qualification_path = (
        BASE / "runtime-control/rmsnorm-complete-graph-qualification-analysis.json"
    )
    assert (
        sha(qualification_path)
        == "f480e98300607517aad2b1b59a9cda71146dc5e85793b1d913c843ea140e0297"
    )
    qualification = read(qualification_path)
    assert qualification["status"] == "complete"
    auditor_sources = {
        name: sha(ROOT / name)
        for name in (
            "benchmarks/analyze_glm53_rmsnorm_intervention.py",
            "benchmarks/analyze_glm53_rmsnorm_graphs.py",
        )
    }
    for name, digest in auditor_sources.items():
        assert hashlib.sha256(git("show", f"{COMMIT}:{name}")).hexdigest() == digest
    launchers, binding_receipts = [], []
    for rank in range(4):
        qualified_folder = BASE / f"rmsnorm-complete-graph-{args.arm}-rank{rank}"
        qualified_manifest = read(qualified_folder / "manifest.json")
        qualified_private = Path(qualified_manifest["private_namespace"])
        graph_path = qualified_folder / "loaded-graph-bindings.json"
        (loader,) = [
            r
            for r in qualification["loaders"]
            if r["rank"] == rank and r["mode"] == args.arm
        ]
        assert sha(graph_path) == loader["graph_sha256"]
        graph = read(graph_path)
        assert graph["unintercepted"] == graph["wrong_selected"] == 0
        expected_graphs = {}
        for row in graph["graph_bindings"]:
            name = str(Path(row["module"]).relative_to(qualified_private))
            assert row["intercepted"]
            assert sha(Path(row["module"])) == snapshot[name]
            expected_graphs[name, row["name"]] = snapshot[name]
        assert len(expected_graphs) == graph["graph_binding_count"]
        path = Path(manifest["receipts"]) / f"rank-{rank}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        checked = audit_rank(
            records,
            rank=rank,
            arm=args.arm,
            manifest=manifest,
            manifest_sha256=sha(manifest_path),
            source_sha256=sources["slimserve/rmsnorm_diagnostic.py"],
            expected_graphs=expected_graphs,
            hash_file=sha,
        )
        launchers.extend(checked.pop("launchers"))
        binding_receipts.append(
            dict(
                **checked,
                path=str(path),
                sha256=sha(path),
                qualified_graph_sha256=sha(graph_path),
            )
        )

    if args.arm == "legacy":
        control = read(BASE / f"runtime-control/{NAMES['control']}-analysis.json")
        assert control["control_exact_to_all_native_passes"]
        assert control["auditor_sources"] == auditor_sources
        assert control["complete_graph_intervention"]
        # Compare unrelated source/config choices, never alias-object multiplicities.
        canonical = lambda rows: {
            (r["rank"], r["filename"], json.dumps(r["before"], sort_keys=True))
            for r in rows
            if not r["target"]
        }
        assert canonical(control["launchers"]) == canonical(launchers)
        normalize_graphs = lambda audit: {
            (r["rank"], Path(g["module"]).name, g["symbol"], g["module_sha256"])
            for r in audit
            for g in r["graph_bindings"]
        }
        assert normalize_graphs(control["binding_receipts"]) == normalize_graphs(
            binding_receipts
        )

    run = candidate["runs"][0]
    assert run["status"] == run["teardown"]["status"] == "complete"
    assert run["teardown"]["returncode"] == 0
    assert run["teardown"]["gpu_release"]["status"] == "complete"
    assert not run["teardown"]["gpu_release"]["samples"][-1]["owned_active_pids"]
    assert (
        run["canaries"]["text"]["answer"] == "4"
        and run["canaries"]["image"]["answer"].lower() == "red"
    )
    assert len(run["warmups"]) == 3 and len(run["measurements"]) == 9
    assert [(m["repeat"], m["concurrency"]) for m in run["measurements"]] == list(
        itertools.product((1, 2, 3), (1, 8, 16))
    )
    receipts, windows = [], []
    for group, paths in [
        ("warmup", run["warmups"]),
        ("timed", [m["path"] for m in run["measurements"]]),
    ]:
        for name in paths:
            path = ROOT / name
            data = read(path)
            assert data["cache_policy"] == "isolated-cold"
            for request in data["requests"]:
                usage = request["usage"]
                assert (
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    usage["total_tokens"],
                ) == (1000, 300, 1300)
                assert usage["prompt_tokens_details"]["cached_tokens"] == 0
                assert (
                    len(request["token_ids"]) == 300
                    and not request["replacement_characters"]
                )
            assert math.isclose(
                300 * len(data["requests"]) / data["wall_seconds"],
                data["aggregate_output_tps"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            receipts.append(
                dict(
                    path=str(path),
                    sha256=sha(path),
                    group=group,
                    requests=len(data["requests"]),
                )
            )
            if group == "timed":
                windows.append(dict(path=str(path), **analyze(data)))
    assert {
        g: sum(r["requests"] for r in receipts if r["group"] == g)
        for g in ("warmup", "timed")
    } == {"warmup": 25, "timed": 75}
    for c in (1, 8, 16):
        values = [w["recorded_e2e_tps"] for w in windows if w["concurrency"] == c]
        assert candidate["aggregates"][str(c)] == dict(
            count=3, median=statistics.median(values), min=min(values), max=max(values)
        )

    references = [
        (
            "old-legacy",
            "stable-align-quality-diagnostic",
            "7a262eee2748daec06d90f55238a362bd0bf4af3ebe4731f5273d98ca5915222",
        ),
        (
            "native",
            "native-order-quality",
            "05fe6f0c9357062b8f9b322e36d0c89558b891d84f325b877789b01d9e6ca338",
        ),
        (
            "native-serialized",
            "native-order-serialized",
            "777665432df2ba8dc5f4ebbda5e00c93028d9070d66dc7eb52a889f33c5cfaab",
        ),
        (
            "native-return",
            "native-order-async-return",
            "7c02b7b269c3e3f86dfa12c8fab03e787a6d607158ae2afeb404789fcf32de41",
        ),
    ]
    if args.arm == "legacy":
        references.append(
            ("control", NAMES["control"], sha(BASE / NAMES["control"] / "summary.json"))
        )
    references.append((args.arm, NAMES[args.arm], sha(RUN / "summary.json")))
    quality, identity = [], None
    for arm, folder, digest in references:
        path = BASE / folder / "summary.json"
        assert sha(path) == digest
        summary = read(path)
        assert (
            summary["status"] == "complete"
            and len(summary["runs"][0]["quality_passes"]) == 3
        )
        for repeat, row in enumerate(summary["runs"][0]["quality_passes"], 1):
            assert row["status"] == "complete" and row["repeat"] == repeat
            path = ROOT / row["path"]
            data = read(path)
            scores = verified_quality(data)
            assert (
                row["summary"] == data["summary"]
                and data["summary"]["all_needles_rank_first"]
            )
            if identity is None:
                identity = input_identity(data)
            assert input_identity(data) == identity
            responses = [r["response"] for r in data["text"]] + [
                c["response"] for r in data["needles"] for c in r["candidates"]
            ]
            assert len(responses) == 56
            assert all(
                r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                for r in responses
            )
            needles = [
                x
                for r in data["needles"]
                for c in r["candidates"]
                for x in c["per_token"]
            ]
            assert len(needles) == 168
            quality.append(
                dict(
                    arm=arm,
                    repeat=repeat,
                    path=str(path),
                    sha256=sha(path),
                    text=scores,
                    needles=needles,
                    summary=data["summary"],
                )
            )
    pairs = [
        dict(
            a=[a["arm"], a["repeat"]],
            b=[b["arm"], b["repeat"]],
            text=delta_summary(a["text"], b["text"]),
            needle_tokens=delta_summary(a["needles"], b["needles"]),
            exact=a["text"] == b["text"] and a["needles"] == b["needles"],
        )
        for a, b in itertools.combinations(quality, 2)
    ]
    within = [p for p in pairs if p["a"][0] == p["b"][0] == args.arm]
    native_pairs = [
        p for p in pairs if p["a"][0].startswith("native") and p["b"][0] == args.arm
    ]
    assert len(within) == 3 and len(native_pairs) == 27
    log = (RUN / "boot-1/server.log").read_text()
    assert log.count("Directly load AOT compilation from path " + str(private)) == 4
    assert "Compiling model again due to a load failure" not in log
    result = dict(
        status="complete",
        arm=args.arm,
        diagnostic_only=True,
        no_exclusions=True,
        complete_graph_intervention=True,
        auditor_sources=auditor_sources,
        qualification_sha256=sha(qualification_path),
        all_candidate_repeats_exact=all(p["exact"] for p in within),
        control_exact_to_all_native_passes=all(p["exact"] for p in native_pairs)
        if args.arm == "control"
        else None,
        candidate_mean_logprobs=[
            r["summary"]["mean_text_logprob"] for r in quality if r["arm"] == args.arm
        ],
        quality_requests=168,
        scored_text_tokens=12288,
        scored_needle_tokens=504,
        warmup_requests=25,
        timed_requests=75,
        request_receipts=receipts,
        quality_receipts=[
            {k: v for k, v in r.items() if k not in ("text", "needles")}
            for r in quality
        ],
        pairs=pairs,
        windows=windows,
        aggregates=candidate["aggregates"],
        startup_seconds=run["startup_seconds"],
        teardown=run["teardown"],
        benchmark_sources_verified=len(sources),
        original_cache_files_verified=len(snapshot),
        private_seed_changes=private_seed_changes,
        binding_receipts=binding_receipts,
        launchers=launchers,
        native_sha256=candidate["runtime"]["native_sha256"],
        analyzer_sha256=sha(Path(__file__)),
        server_log_sha256=sha(RUN / "boot-1/server.log"),
        allocator_oom_warnings=[
            line
            for line in log.splitlines()
            if "memory allocation failed with OOM" in line
        ],
        limitation=(
            "Graph inventory is independently qualified and live object coverage "
            "is checked before capture. Exact no-op equality required. "
            "No production promotion or fresh-compilation claim."
        ),
    )
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "pairs",
                    "quality_receipts",
                    "request_receipts",
                    "windows",
                    "teardown",
                    "native_sha256",
                    "launchers",
                    "binding_receipts",
                )
            },
            indent=2,
        )
    )
    if not result["all_candidate_repeats_exact"] or (
        args.arm == "control" and not result["control_exact_to_all_native_passes"]
    ):
        raise SystemExit(
            "Exact score gate failed; comparisons preserved. "
            "Do not proceed to legacy if control failed."
        )


if __name__ == "__main__":
    main()
