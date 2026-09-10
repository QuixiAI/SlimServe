# SPDX-License-Identifier: Apache-2.0
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_glm53_prompt_score_rollout as rollout
from tests.slimserve.test_quality_pair_analysis import docs as docs  # noqa: F401


def observation(warnings=0):
    return dict(
        quality={"passed": True},
        aggregates={c: {"median": v} for c, v in (("1", 157), ("8", 580), ("16", 780))},
        prefill={
            "aggregates": {
                c: {"engine_scheduled_to_first_token_ms": {"median": v}}
                for c, v in (("32768", 2600), ("131072", 10900))
            }
        },
        allocation_warnings=["OOM"] * warnings,
    )


@pytest.mark.parametrize("change", [None, "decode", "prefill", "spread", "all_slow"])
def test_fixed_performance_gate_never_hides_a_bad_start(change):
    audits = {label: observation() for label, _ in rollout.ORDER}
    if change == "decode":
        audits["candidate-2"]["aggregates"]["8"]["median"] *= 0.97
    elif change == "prefill":
        audits["candidate-3"]["prefill"]["aggregates"]["131072"][
            "engine_scheduled_to_first_token_ms"
        ]["median"] *= 1.03
    elif change == "spread":
        audits["control"]["aggregates"]["1"]["median"] *= 1.03
    elif change == "all_slow":
        for row in audits.values():
            row["aggregates"]["1"]["median"] = 100
    result = rollout.performance_gate(audits)
    assert result["passed"] is (change is None)


def test_only_memory_flag_changes_and_no_diagnostic_cache_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setenv("SLIMSERVE_GLM53_INDEXER_CORRECTION", "correction")
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    before, after = [rollout.environment(tmp_path, flag) for flag in ("0", "1")]
    assert {k for k in before | after if before.get(k) != after.get(k)} == {
        rollout.FLAG
    }
    assert before["SLIMSERVE_GLM53_NATIVE_ORDER"] == "0"
    assert not any(
        k in before
        for k in (
            "VLLM_FORCE_AOT_LOAD",
            "SLIMSERVE_GLM53_INDEXER_CORRECTION",
            "NCCL_P2P_DISABLE",
        )
    )
    argv = rollout.command(tmp_path)
    assert "--quality-repeats" not in argv and "--deterministic-reductions" not in argv
    assert argv[argv.index("--boots") + 1] == "1"


def test_real_plan_resolves_paths_inside_prescribed_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path / "unrelated"))
    plan = rollout.expected_plan(tmp_path / "control", "0")
    assert plan["profile_id"] == "glm53-nvfp4-4"
    assert plan["weight_recipe"]["id"] == "glm53-redhatai-nvfp4-fp8-kda-tp4-v1"
    assert plan["engine"]["tensor_parallel_size"] == 4 and plan["speculative"] is False


@pytest.mark.parametrize("failure", [None, "serve", "quality", "interrupt"])
def test_prescribed_sequence_stops_and_preserves_failures(
    tmp_path, monkeypatch, docs, failure
):
    monkeypatch.setattr(rollout, "ROOT", tmp_path)
    output = tmp_path / "perf/results/rollout"
    monkeypatch.setattr(sys, "argv", ["rollout", "--output", str(output)])
    monkeypatch.setattr(rollout, "historical_evidence", lambda: docs[:2])
    monkeypatch.setattr(
        rollout, "freeze", lambda: {"fixed": True, "gpu_config": "hardware"}
    )
    monkeypatch.setattr(rollout, "gpu_config", lambda: "hardware")
    monkeypatch.setattr(rollout, "check_freeze", lambda _: None)
    monkeypatch.setattr(rollout, "gpu_processes", lambda: "")
    calls, stopped = [], []

    def execute(argv, **kwargs):
        if argv[0] == "systemctl":
            stopped.append(argv)
            return SimpleNamespace(returncode=0)
        calls.append(argv)
        assert "MemoryMax=150G" in argv and "MemorySwapMax=0" in argv
        assert kwargs["env"][rollout.FLAG] == rollout.ORDER[len(calls) - 1][1]
        assert kwargs["env"]["SLIMSERVE_GLM53_NATIVE_ORDER"] == "0"
        assert not list(Path(kwargs["env"]["VLLM_CACHE_ROOT"]).iterdir())
        kwargs["stdout"].write("preserved launch\n")
        if failure == "interrupt":
            raise RuntimeError("interrupted")
        return SimpleNamespace(returncode=1 if failure == "serve" else 0)

    def audit(folder, flag, frozen, references):
        report = observation(warnings=8 if flag == "0" else 0)
        report["quality"]["passed"] = failure != "quality"
        return report, docs[2]

    monkeypatch.setattr(rollout.subprocess, "run", execute)
    monkeypatch.setattr(rollout, "audit", audit)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            rollout.main()
    else:
        rollout.main()
    record = json.loads((output / "rollout.json").read_text())
    assert record["status"] == ("failed" if failure else "passed")
    assert len(calls) == (1 if failure else 5)
    assert list(record["runs"]) == [label for label, _ in rollout.ORDER][: len(calls)]
    for row in record["runs"].values():
        assert row["files"] == rollout.inventory(Path(row["folder"]))
    if failure:
        assert record["runs"]["control"]["status"] == "failed"
        assert record["failure_source_freeze_verified"] is True
    assert len(stopped) == int(failure == "interrupt")
    with pytest.raises(ValueError, match="new in-repo"):
        rollout.main()


def audit_fixture(tmp_path, monkeypatch, docs):
    folder = tmp_path / "candidate-1"
    (folder / "campaign/boot-1").mkdir(parents=True)
    quality = folder / "campaign/boot-1/quality.json"
    quality.write_text(json.dumps(docs[2]))
    (folder / "campaign/boot-1/server.log").write_text("ordinary log\n")
    frozen = dict(
        git_commit="commit", prompt_sha256="prompt", runtime={"gpu_before_start": "gpu"}
    )
    summary = dict(
        status="complete",
        git_commit="commit",
        git_status="",
        compatible_profiles=["glm53-nvfp4-4"],
        command=rollout.command(folder),
        environment=rollout.policy(folder, "1"),
        plan={"fixed": True},
        benchmark_implementation_sha256={"source": "hash"},
        source_sha256="prompt",
        diagnostic_only=False,
        throughput_is_baseline_eligible=True,
        runtime=copy.deepcopy(frozen["runtime"]),
        aggregates={},
    )
    run = dict(
        status="complete",
        teardown={
            "status": "complete",
            "returncode": 0,
            "gpu_release": {"status": "complete"},
        },
        canaries={"text": {"answer": "4"}, "image": {"answer": "red"}},
        quality_passes=[{"status": "complete", "elapsed_seconds": 90}],
        quality_path=str(quality),
        startup_seconds=160,
    )
    summary["runs"] = [run]
    monkeypatch.setattr(rollout, "expected_plan", lambda *a: {"fixed": True})
    monkeypatch.setattr(rollout, "snapshot", lambda: {"source": "hash"})
    monkeypatch.setattr(rollout, "hardware_identity", lambda s: s)
    monkeypatch.setattr(rollout, "timing", lambda *a: [])
    monkeypatch.setattr(rollout, "prefill", lambda *a: {})
    return folder, frozen, summary


@pytest.mark.parametrize(
    "change",
    [
        None,
        "plan",
        "diagnostic",
        "environment",
        "native",
        "source",
        "teardown",
        "canary",
        "quality_count",
    ],
)
def test_real_audit_admission_and_warning_census(tmp_path, monkeypatch, docs, change):
    folder, frozen, summary = audit_fixture(tmp_path, monkeypatch, docs)
    if change == "plan":
        summary["plan"]["fixed"] = False
    elif change == "diagnostic":
        summary["diagnostic_only"] = True
    elif change == "environment":
        summary["environment"]["VLLM_FORCE_AOT_LOAD"] = "1"
    elif change == "native":
        summary["runtime"]["native"] = "changed"
    elif change == "source":
        summary["benchmark_implementation_sha256"]["source"] = "changed"
    elif change == "teardown":
        summary["runs"][0]["teardown"]["returncode"] = 1
    elif change == "canary":
        summary["runs"][0]["canaries"]["text"]["answer"] = "wrong"
    elif change == "quality_count":
        summary["runs"][0]["quality_passes"] *= 2
    (folder / "campaign/summary.json").write_text(json.dumps(summary))
    if change:
        with pytest.raises(ValueError):
            rollout.audit(folder, "1", frozen, docs[:2])
    else:
        (folder / "campaign/boot-1/server.log").write_text(
            "[rank0] memory allocation failed with OOM: 4718592000\n"
        )
        report, document = rollout.audit(folder, "1", frozen, docs[:2])
        assert len(report["allocation_warnings"]) == 1 and report["quality"]["passed"]
        assert document == docs[2]
