# SPDX-License-Identifier: Apache-2.0
import copy
import json
import shutil
from pathlib import Path

import pytest

from benchmarks.analyze_glm53_quality_pair import verified_quality
from benchmarks.kernels import glm53_geometry_workload as workload
from slimserve.campaign_sources import PATHS
from tests.slimserve.test_quality_pair_analysis import docs as quality_docs
from tests.slimserve.test_quality_pair_analysis import update_window


@pytest.fixture
def docs():
    return quality_docs.__wrapped__()


def scores(documents):
    return [
        dict(
            text=verified_quality(d),
            needles=[
                v for n in d["needles"] for c in n["candidates"] for v in c["per_token"]
            ],
        )
        for d in documents
    ]


@pytest.mark.parametrize("label", ["control", "geometry", "kv", "return-control"])
def test_original_scores_pass_diagnostic_without_production_promotion(docs, label):
    refs = {
        "control": docs[:1],
        "return-control": docs[1:2],
        "failed-no-combo": docs[:1],
    }
    result = workload.check_scores(
        label, docs[2:], scores(docs[2:]), refs, {k: scores(v) for k, v in refs.items()}
    )
    assert result["quality_passed"] and result["diagnostic_gate_passed"]
    assert not result["production_qualified"]


@pytest.mark.parametrize(
    "label,allowed",
    [("control", False), ("geometry", True), ("kv", True), ("return-control", False)],
)
def test_geometry_regression_is_observation_not_relaxed_quality_gate(
    docs, label, allowed
):
    refs = {"control": docs[:1], "return-control": docs[1:2]}
    for d in docs[2:]:
        for i in range(12):
            update_window(d, i, -2.1)
    refs["failed-no-combo"] = docs[2:3]
    result = workload.check_scores(
        label, docs[2:], scores(docs[2:]), refs, {k: scores(v) for k, v in refs.items()}
    )
    assert result["diagnostic_gate_passed"] is allowed
    assert not result["quality_passed"] and not result["production_qualified"]
    assert result["historical_comparisons"]["failed-no-combo"]["exact"]
    assert all(
        sum(not w["passed"] for w in r["windows"]) == 12
        for r in result["unchanged_window_quality_gate"]["candidates"]
    )


@pytest.mark.parametrize("label", ["control", "geometry", "kv", "return-control"])
def test_nonrepeatable_scores_stop_every_arm(docs, label):
    refs = {"control": docs[:1], "return-control": docs[1:2]}
    update_window(docs[3], 0, -2.001)
    result = workload.check_scores(
        label, docs[2:], scores(docs[2:]), refs, {k: scores(v) for k, v in refs.items()}
    )
    assert not result["diagnostic_gate_passed"]
    assert result["quality_passed"]


def test_control_must_be_exact_even_inside_quality_band(docs):
    refs = {"control": docs[:1], "return-control": docs[1:2]}
    for d in docs[2:]:
        update_window(d, 0, -2.001)
    result = workload.check_scores(
        "return-control",
        docs[2:],
        scores(docs[2:]),
        refs,
        {k: scores(v) for k, v in refs.items()},
    )
    assert result["quality_passed"] and not result["diagnostic_gate_passed"]


def summary_fixture(tmp_path):
    path = tmp_path / "control/manifest.json"
    folder = path.parent / "campaign/boot-1"
    manifest = dict(
        serving_schema="glm53-rmsnorm-geometry-serving-v1",
        label="control",
        mode="control",
        git_commit="frozen",
        cache_root=str(path.parent / "cache"),
        private_namespace=str(path.parent / "cache/private"),
        sources={str(workload.ROOT / p): "digest" for p in PATHS},
        workload=dict(
            prompt="prompt",
            prompt_sha256="prompt-digest",
            plan={"recipe": "fixed"},
            environment={"SLIMSERVE_CACHE": "/raid/weights"},
            runtime={"packages": {}},
        ),
    )
    gpu = "\n".join(
        [
            ",".join(["header"] * 15),
            *[
                ",".join(
                    [
                        str(r),
                        f"gpu-{r}",
                        "bus",
                        "name",
                        "driver",
                        "mem",
                        "used",
                        "util",
                        "clock",
                        "clock",
                        "max",
                        "state",
                        "limit",
                        "draw",
                        "temp",
                    ]
                )
                for r in range(4)
            ],
        ]
    )
    manifest["workload"]["runtime"]["gpu_before_start"] = gpu
    run = dict(
        boot=1,
        status="complete",
        argv=[
            str(workload.ROOT / ".venv/bin/python"),
            "-m",
            "slimserve.cli",
            "glm53-nvfp4-4",
            "--serve",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "-y",
            "--request-metrics",
        ],
        teardown=dict(
            status="complete",
            returncode=0,
            gpu_release=dict(status="complete", samples=[{"owned_active_pids": []}]),
        ),
        canaries={"text": {"answer": "4"}, "image": {"answer": "red"}},
        warmups=[str(folder / f"warmup-c{c}.json") for c in (1, 8, 16)],
        measurements=[
            dict(path=str(folder / f"repeat-{r}-c{c}.json"))
            for r in (1, 2, 3)
            for c in (1, 8, 16)
        ],
        quality_passes=[
            dict(path=str(folder / p))
            for p in ("quality.json", "quality-repeat-2.json", "quality-repeat-3.json")
        ],
        prefill_path=str(folder / "prefill"),
    )
    summary = dict(
        status="complete",
        git_commit="frozen",
        git_status="",
        diagnostic_only=True,
        throughput_is_baseline_eligible=False,
        compatible_profiles=["glm53-nvfp4-4"],
        command=workload.command(path, manifest),
        source_sha256="prompt-digest",
        plan=copy.deepcopy(manifest["workload"]["plan"]),
        environment=workload.expected_environment(path, manifest),
        runtime={"packages": {}, "gpu_before_start": gpu},
        benchmark_implementation_sha256={p: "digest" for p in PATHS},
        runs=[run],
    )
    return path, manifest, summary


def test_exact_summary_contract(tmp_path):
    path, manifest, summary = summary_fixture(tmp_path)
    assert workload.check_summary(path, manifest, summary) is summary["runs"][0]
    assert "--prefill" in summary["command"]


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "extra-start",
        "dirty",
        "commit",
        "eligible",
        "profile",
        "command",
        "plan",
        "environment",
        "source",
        "native",
        "missing-source",
        "teardown",
        "release",
        "canary",
        "server-command",
        "warmup-order",
        "quality-path",
    ],
)
def test_summary_rejects_changed_or_incomplete_evidence(tmp_path, mutation):
    path, manifest, s = summary_fixture(tmp_path)
    r = s["runs"][0]
    if mutation == "incomplete":
        s["status"] = "failed"
    elif mutation == "extra-start":
        s["runs"].append(copy.deepcopy(r))
    elif mutation == "dirty":
        s["git_status"] = "M source.py"
    elif mutation == "commit":
        s["git_commit"] = "other"
    elif mutation == "eligible":
        s["throughput_is_baseline_eligible"] = True
    elif mutation == "profile":
        s["compatible_profiles"].append("glm53-nvfp4-8")
    elif mutation == "command":
        s["command"].append("--deterministic-reductions")
    elif mutation == "plan":
        s["plan"]["recipe"] = "wrong"
    elif mutation == "environment":
        s["environment"]["NCCL_P2P_DISABLE"] = "1"
    elif mutation == "source":
        s["source_sha256"] = "different"
    elif mutation == "native":
        s["runtime"]["native_sha256"] = {"other": "binary"}
    elif mutation == "missing-source":
        s["benchmark_implementation_sha256"].pop(PATHS[0])
    elif mutation == "teardown":
        r["teardown"]["returncode"] = -9
    elif mutation == "release":
        r["teardown"]["gpu_release"]["samples"][0]["owned_active_pids"] = [12]
    elif mutation == "canary":
        r["canaries"]["image"]["answer"] = "blue"
    elif mutation == "server-command":
        r["argv"].append("--deterministic-reductions")
    elif mutation == "warmup-order":
        r["warmups"].reverse()
    elif mutation == "quality-path":
        r["quality_passes"][0]["path"] = str(Path("/tmp/other.json"))
    with pytest.raises(ValueError):
        workload.check_summary(path, manifest, s)


@pytest.mark.parametrize(
    "column,allowed",
    [
        (1, False),
        (4, False),
        (5, False),
        (6, True),
        (8, True),
        (10, False),
        (12, False),
        (13, True),
    ],
)
def test_runtime_distinguishes_dynamic_telemetry_from_hardware_configuration(
    tmp_path, column, allowed
):
    path, manifest, summary = summary_fixture(tmp_path)
    rows = summary["runtime"]["gpu_before_start"].splitlines()
    row = rows[1].split(",")
    row[column] = "changed"
    rows[1] = ",".join(row)
    summary["runtime"]["gpu_before_start"] = "\n".join(rows)
    if allowed:
        workload.check_summary(path, manifest, summary)
    else:
        with pytest.raises(ValueError, match="GPU topology"):
            workload.check_summary(path, manifest, summary)


def test_retained_real_workload_replay_is_a_failed_quality_observation(tmp_path):
    """CPU rehearsal using real responses, not a new model or serving claim.

    Only wrapper metadata/paths are adapted to the new protocol. Every token, score,
    timing, sampling body and usage field is copied unchanged from the terminal run.
    The loader audit is deliberately separate; no CUDA or graph code executes here.
    """
    previous = workload.ROOT / workload.REFERENCE_SUMMARIES["failed-no-combo"][0]
    if not previous.exists():
        pytest.skip("local campaign receipts are not distributed with the repository")
    spec, sources, _, _ = workload.reference_evidence()
    path = tmp_path / "geometry/manifest.json"
    folder = path.parent / "campaign"
    shutil.copytree(previous.parent, folder)

    def relocate(value):
        if isinstance(value, str) and value.startswith(str(previous.parent)):
            return str(folder) + value[len(str(previous.parent)) :]
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        return value

    summary = relocate(workload.read(folder / "summary.json"))
    from slimserve.campaign_sources import snapshot

    manifest = dict(
        serving_schema="glm53-rmsnorm-geometry-serving-v1",
        workload=spec,
        label="geometry",
        mode="geometry",
        git_commit="cpu-replay",
        cache_root=str(path.parent / "cache"),
        private_namespace=str(path.parent / "cache/private"),
        sources={
            **sources,
            **{str(workload.ROOT / p): h for p, h in snapshot().items()},
        },
    )
    summary.update(
        git_commit="cpu-replay",
        command=workload.command(path, manifest),
        benchmark_implementation_sha256=snapshot(),
        plan=spec["plan"],
        environment=workload.expected_environment(path, manifest),
    )
    summary["runs"][0]["argv"].remove("--deterministic-reductions")
    (folder / "summary.json").write_text(json.dumps(summary))
    prefill_path = folder / "boot-1/prefill/summary.json"
    prefill_path.write_text(json.dumps(relocate(workload.read(prefill_path))))
    result = {"receipts": {}}
    workload.audit_workload(path, manifest, result)
    assert result["diagnostic_gate_passed"] and not result["quality_passed"]
    assert result["historical_comparisons"]["failed-no-combo"]["exact"]
    assert len(result["timing"]) == 12 and len(result["prefill"]["requests"]) == 8
    assert len(result["quality_receipts"]) == 3
    assert set(result["prefill_prompt_sha256"]) == {"32768", "131072"}
    assert not result["production_qualified"]
