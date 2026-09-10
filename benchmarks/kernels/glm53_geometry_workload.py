# SPDX-License-Identifier: Apache-2.0
"""Fixed full-model causal workload; quality failure is never a performance win.

Controls must reproduce the original scores exactly. Geometry scores are an
observation (including failed unchanged quality floors), provided the workload,
needles and within-start repetition qualify. This allows the prescribed return
control to test reversibility without promoting a failed-quality intervention.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from benchmarks.analyze_glm53_deterministic_serving import prefill, quality, timing
from benchmarks.analyze_glm53_quality_pair import (
    compare,
    delta_summary,
    input_identity,
    require,
)
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from slimserve.rmsnorm_diagnostic import sha
from slimserve.rmsnorm_geometry import FLAG, MANIFEST

ROOT = Path(__file__).resolve().parents[2]
PROMPT = Path("/home/tiny/.local/scratch/slimserve-glm53/prompt-source.txt")
MODEL = Path("/raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4")
REFERENCE_SUMMARIES = {
    "control": (
        "perf/results/2026-09-09/native-order-quality/summary.json",
        "05fe6f0c9357062b8f9b322e36d0c89558b891d84f325b877789b01d9e6ca338",
    ),
    "return-control": (
        "perf/results/2026-09-09/native-order-async-return/summary.json",
        "7c02b7b269c3e3f86dfa12c8fab03e787a6d607158ae2afeb404789fcf32de41",
    ),
    "failed-no-combo": (
        "perf/results/2026-09-10/deterministic-no-combo-serving/fresh-a/summary.json",
        "09b656f73988ef577e39ec69def7bc99518e5f7fcec5e2bbafa8e7c6cca49e10",
    ),
}
# Whole response documents, checked against their completed historical audits.
# Summary means alone are not sufficient to pin a per-token reference vector.
REFERENCE_QUALITY_SHAS = {
    "control": [
        "a7979e4db521b72e101e2c4f928b7f02d4597dbf62c553bf51bb1c58a68cedcd",
        "52cfddec232f36a4c822f19b36db04a6826bf8b3de241ee41141453e123d9e5a",
        "05fe6c9b696b6beaee1e0bd0927bfa4374627e2a6d6c947cd1743cdb3859dd12",
    ],
    "return-control": [
        "3df9a84bb8b9924fa39bad7234ec568d13b5a657b92e38a0a3402e454d696792",
        "376aadae24ea54036f8629965b31a51b8a6437aad005a4c6c3c5bef101cff458",
        "2ce8bc221c98beb842f052f1194b4ab60888fadba7648ba2080eee17aabc6a84",
    ],
    "failed-no-combo": [
        "d6823a808cd1017efc50e9d6b6c0e44c960552a4dcadd2869dc19e4f592f9c4a",
        "db98c71b2757fb1c9ab9d781c0fb507eb5994a26a8b29ca03cc61d839f5e6333",
        "a8a40524af0e577c1f42bed20b816fa3193b1027c8c196db42f5cdce20929018",
    ],
}


def read(path):
    return json.loads(Path(path).read_text())


def vectors_exact(left, right):
    return all(left[k] == right[k] for k in ("text", "needles"))


def hardware_identity(snapshot):
    rows = list(csv.reader(io.StringIO(snapshot), skipinitialspace=True))
    require(
        len(rows) == 5 and all(len(r) == 15 for r in rows),
        "four-GPU runtime inventory required",
    )
    # Dynamic clocks, used memory, temperature and power draw are observations.
    # GPU IDs/topology/model/driver/capacity and configured limits stay fixed.
    return [[r[i].strip() for i in (0, 1, 2, 3, 4, 5, 10, 12)] for r in rows[1:]]


def reference_evidence():
    """Consume pinned completed data; never rerun the stopped compiler series."""
    sources, summaries, docs, scores = {}, {}, {}, {}
    for label, (relative, digest) in REFERENCE_SUMMARIES.items():
        path = ROOT / relative
        require(sha(path) == digest, "historical workload summary changed")
        sources[str(path)] = digest
        summaries[label] = read(path)
        docs[label], scores[label] = quality(summaries[label])
        require(
            [r["sha256"] for r in scores[label]] == REFERENCE_QUALITY_SHAS[label],
            "historical whole quality response changed",
        )
        for row in scores[label]:
            sources[str((ROOT / row["path"]).resolve())] = row["sha256"]
        require(
            all(vectors_exact(scores[label][0], s) for s in scores[label]),
            "historical within-start equality changed",
        )
    identity = input_identity(docs["control"][0])
    require(
        all(input_identity(d) == identity for ds in docs.values() for d in ds),
        "historical quality inputs differ",
    )
    require(
        vectors_exact(scores["control"][0], scores["return-control"][0]),
        "historical original control scores differ",
    )
    failed_gate = compare(
        [docs["control"][0], docs["return-control"][0]], docs["failed-no-combo"]
    )
    require(
        not failed_gate["passed"]
        and all(
            sum(not w["passed"] for w in r["windows"]) == 12
            for r in failed_gate["candidates"]
        ),
        "historical failed 12-window result changed",
    )
    original = summaries["control"]
    require(sha(PROMPT) == original["source_sha256"], "prompt source changed")
    sources[str(PROMPT)] = sha(PROMPT)
    for name, digest in original["runtime"]["native_sha256"].items():
        require(sha(ROOT / name) == digest, "native binary changed")
        sources[str(ROOT / name)] = digest
    contract = dict(
        schema="glm53-geometry-workload-v1",
        prompt=str(PROMPT),
        prompt_sha256=sha(PROMPT),
        model=str(MODEL),
        reference_summaries={
            label: dict(path=str(ROOT / p), sha256=h)
            for label, (p, h) in REFERENCE_SUMMARIES.items()
        },
        plan=original["plan"],
        environment=original["environment"],
        runtime=original["runtime"],
    )
    return contract, sources, docs, scores


def command(path, manifest):
    return [
        str(ROOT / "benchmarks/benchmark_glm53_campaign.py"),
        "--profile",
        "glm53-nvfp4-4",
        "--source",
        manifest["workload"]["prompt"],
        "--output",
        str(path.parent / "campaign"),
        "--boots",
        "1",
        "--repeats",
        "3",
        "--concurrency",
        "1",
        "8",
        "16",
        "--input-tokens",
        "1000",
        "--output-tokens",
        "300",
        "--cold-prefix",
        "--quality",
        "--quality-repeats",
        "3",
        "--prefill",
    ]


def expected_environment(path, manifest):
    return {
        **manifest["workload"]["environment"],
        "VLLM_CACHE_ROOT": manifest["cache_root"],
        "TORCHINDUCTOR_CACHE_DIR": str(
            Path(manifest["private_namespace"]) / "inductor_cache"
        ),
        "VLLM_FORCE_AOT_LOAD": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_CACHE_AUTOTUNING": "1",
        FLAG: manifest["mode"],
        MANIFEST: str(path),
    }


def check_summary(path, manifest, summary):
    spec = manifest["workload"]
    require(
        summary["status"] == "complete"
        and summary["git_commit"] == manifest["git_commit"]
        and not summary["git_status"]
        and summary["diagnostic_only"] is True
        and summary["throughput_is_baseline_eligible"] is False
        and summary["compatible_profiles"] == ["glm53-nvfp4-4"]
        and len(summary["runs"]) == 1,
        "incomplete/unfrozen diagnostic or wrong profile/start count",
    )
    require(
        summary["command"] == command(path, manifest)
        and summary["source_sha256"] == spec["prompt_sha256"]
        and summary["plan"] == spec["plan"]
        and summary["environment"] == expected_environment(path, manifest),
        "prescribed workload, recipe or environment changed",
    )
    require(
        summary["runtime"]
        == {
            **spec["runtime"],
            "gpu_before_start": summary["runtime"]["gpu_before_start"],
        },
        "runtime/package/native/affinity identity changed",
    )
    require(
        hardware_identity(summary["runtime"]["gpu_before_start"])
        == hardware_identity(spec["runtime"]["gpu_before_start"]),
        "GPU topology/driver/capacity/power configuration changed",
    )
    from slimserve.campaign_sources import PATHS

    require(
        summary["benchmark_implementation_sha256"]
        == {name: manifest["sources"][str(ROOT / name)] for name in PATHS},
        "benchmark source freeze differs from loaded client",
    )
    run = summary["runs"][0]
    teardown = run["teardown"]
    require(
        run["boot"] == 1
        and run["status"] == teardown["status"] == "complete"
        and teardown["returncode"] == 0
        and teardown["gpu_release"]["status"] == "complete"
        and not teardown["gpu_release"]["samples"][-1]["owned_active_pids"],
        "incomplete workload or owned-worker teardown",
    )
    require(
        run["canaries"]["text"]["answer"] == "4"
        and run["canaries"]["image"]["answer"].lower() == "red",
        "text/image canary failed",
    )
    require(
        run["argv"]
        == [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "slimserve.cli",
            "glm53-nvfp4-4",
            "--serve",
            "--host",
            "127.0.0.1",
            "--port",
            run["argv"][8],
            "-y",
            "--request-metrics",
        ]
        and 0 < int(run["argv"][8]) < 65536,
        "unexpected serving command",
    )
    folder = path.parent / "campaign/boot-1"
    require(
        run["warmups"] == [str(folder / f"warmup-c{c}.json") for c in (1, 8, 16)]
        and [m["path"] for m in run["measurements"]]
        == [
            str(folder / f"repeat-{r}-c{c}.json") for r in (1, 2, 3) for c in (1, 8, 16)
        ]
        and [r["path"] for r in run["quality_passes"]]
        == [
            str(folder / name)
            for name in (
                "quality.json",
                "quality-repeat-2.json",
                "quality-repeat-3.json",
            )
        ]
        and run["prefill_path"] == str(folder / "prefill"),
        "workload paths or matrix changed",
    )
    return run


def check_scores(label, documents, scores, reference_docs, reference_scores):
    identity = input_identity(reference_docs["control"][0])
    require(
        all(input_identity(d) == identity for d in documents),
        "candidate/reference prompt IDs differ",
    )
    repeated = all(vectors_exact(scores[0], s) for s in scores)
    gate = compare(
        [reference_docs["control"][0], reference_docs["return-control"][0]],
        documents,
    )
    comparisons = {}
    for name, rows in reference_scores.items():
        comparisons[name] = dict(
            exact=vectors_exact(rows[0], scores[0]),
            text=delta_summary(rows[0]["text"], scores[0]["text"]),
            needles=delta_summary(rows[0]["needles"], scores[0]["needles"]),
        )
    control = label != "geometry"
    # A geometry quality regression is retained as a diagnostic observation. The
    # existing floors are evaluated unchanged and never labelled a quality pass.
    passed = repeated and (
        not control or (gate["passed"] and comparisons["control"]["exact"])
    )
    return dict(
        within_start_exact=repeated,
        unchanged_window_quality_gate=gate,
        historical_comparisons=comparisons,
        diagnostic_gate_passed=passed,
        quality_passed=gate["passed"],
        production_qualified=False,
    )


def audit_workload(path, manifest, result):
    receipts = result["receipts"]

    def keep(name):
        name = Path(name).resolve()
        receipts[str(name)] = sha(name)
        return read(name)

    spec, sources, ref_docs, ref_scores = reference_evidence()
    require(spec == manifest["workload"], "prepared workload contract changed")
    require(
        all(manifest["sources"].get(p) == h for p, h in sources.items()),
        "reference workload receipts missing from source freeze",
    )
    summary = keep(path.parent / "campaign/summary.json")
    run = check_summary(path, manifest, summary)
    # Retain every generated workload file, including partial failures, separately
    # from these success gates. The launcher also hashes the full case inventory.
    result.update(
        summary_sha256=sha(path.parent / "campaign/summary.json"),
        startup_seconds=run["startup_seconds"],
        diagnostic_tps=summary["aggregates"],
        teardown=run["teardown"],
    )
    result["timing"] = timing(run, summary)
    for row in result["timing"]:
        data = keep(row["path"])
        require(
            [r["seed"] for r in data["requests"]]
            == list(range(42, 42 + len(data["requests"]))),
            "timing seed sequence changed",
        )
    result["prefill"] = prefill(run, spec["prompt_sha256"])
    result["prefill_prompt_sha256"] = {}
    keep(result["prefill"]["path"])
    for row in result["prefill"]["requests"]:
        request = keep(row["path"])
        ids = request["request"]["prompt"]
        key = str(len(ids))
        digest = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        require(
            result["prefill_prompt_sha256"].setdefault(key, digest) == digest,
            "cold prefill prompt changed within start",
        )
        require(
            {
                k: v
                for k, v in request["request"].items()
                if k not in ("prompt", "cache_salt")
            }
            == dict(
                model=spec["plan"]["engine"]["served_model_name"],
                max_tokens=8,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                seed=42,
                ignore_eos=True,
                stream=True,
                return_token_ids=True,
                stream_options={"include_usage": True},
            ),
            "cold prefill sampling/workload changed",
        )
    documents, scores = quality(summary)
    for row in scores:
        keep(ROOT / row["path"])
    result["quality_receipts"] = [
        dict(path=r["path"], sha256=r["sha256"]) for r in scores
    ]
    result.update(
        check_scores(manifest["label"], documents, scores, ref_docs, ref_scores)
    )
    require(
        result["diagnostic_gate_passed"], "control/repetition diagnostic gate failed"
    )


def audit(path):
    from benchmarks.kernels.audit_glm53_geometry_serving import audit as audit_workers
    from benchmarks.kernels.check_glm53_attention_norms import verify

    path = path.resolve()
    manifest = read(path)
    output = path.parent / "workload-analysis.json"
    require(not output.exists(), "preserve prior workload audit")
    result = dict(
        status="failed",
        scope=__doc__,
        label=manifest["label"],
        manifest_sha256=sha(path),
        auditor_sha256=sha(__file__),
        receipts={},
        production_qualified=False,
    )
    try:
        verify(manifest)
        require(audit_workers(path), "all-rank serving loader audit failed")
        audit_workload(path, manifest, result)
        verify(manifest)
        result["status"] = "complete"
    except Exception as error:
        result["error"] = repr(error)
    write_new(output, result)
    print(json.dumps({k: result[k] for k in ("status", "label")}), flush=True)
    return result["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.manifest) else 1)


if __name__ == "__main__":
    main()
