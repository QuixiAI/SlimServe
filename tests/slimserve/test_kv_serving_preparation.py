# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kernels import glm53_geometry_workload as workload
from benchmarks.kernels import prepare_glm53_kv_serving as preparation
from benchmarks.kernels import run_glm53_geometry_serving as runner
from slimserve import glm53_ordering
from slimserve import kv_diagnostic as policy
from slimserve.glm53_serving_diagnostic import cases
from slimserve.glm53_serving_diagnostic import policy as serving_policy
from slimserve.rmsnorm_diagnostic import sha
from tests.slimserve.test_geometry_serving_runner import fixture as runner_fixture
from tests.slimserve.test_kv_loader_audit import prepared_fixture


def completed_fixture(tmp_path, monkeypatch):
    base = prepared_fixture(tmp_path, monkeypatch)
    base["expected_gpu_config"] = "qualified-hardware"
    base["private_sources"] = {
        t["kv"]["relative"]: t["kv"]["source_sha256"]
        for targets in base["targets"].values()
        for t in targets
    }
    reference_path = tmp_path / "qualified.json"
    reference_path.write_text(json.dumps(base))
    ref = dict(path=str(reference_path), sha256=sha(reference_path))
    graphs = {mode: {str(r): ref for r in range(4)} for mode in ("control", "kv")}
    monkeypatch.setattr(
        preparation,
        "completed_evidence",
        lambda *_: (base, graphs, ref, {str(reference_path): sha(reference_path)}),
    )
    monkeypatch.setattr(workload, "reference_evidence", lambda: ({}, {}, {}, {}))
    original_check = subprocess.check_output
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kw: ""
        if cmd == ["git", "status", "--short"]
        else original_check(cmd, **kw),
    )
    return base


def test_prepare_independent_kv_copies_and_roundtrip_policy(tmp_path, monkeypatch):
    base = completed_fixture(tmp_path, monkeypatch)
    output = tmp_path / "serving"
    preparation.prepare(tmp_path / "pair.json", output)
    prep = json.loads((output / "preparation.json").read_text())
    assert [(r["label"], r["mode"]) for r in prep["runs"]] == list(policy.CASES)
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    for label, mode in policy.CASES:
        path = output / label / "manifest.json"
        manifest = json.loads(path.read_text())
        monkeypatch.setenv(policy.FLAG, mode)
        monkeypatch.setenv(policy.MANIFEST, str(path))
        monkeypatch.setenv("VLLM_CACHE_ROOT", manifest["cache_root"])
        monkeypatch.setenv(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path(manifest["private_namespace"]) / "inductor_cache"),
        )
        assert policy.read_manifest() == (manifest, path)
        private = Path(manifest["private_namespace"])
        assert runner.inventory(private) == {
            **base["original_files"],
            **base["private_sources"],
        }
        for targets in base["targets"].values():
            for target in targets:
                assert not (private / target["kv"]["relative"]).samefile(
                    target["kv"]["source"]
                )
        assert serving_policy(manifest) is policy and cases(manifest) == policy.CASES
    # Mutating one candidate's source cannot silently mutate another's copy.
    relative = next(iter(base["private_sources"]))
    kv_manifest = json.loads((output / "kv/manifest.json").read_text())
    (Path(kv_manifest["private_namespace"]) / relative).write_text("changed-copy")
    assert sha(private / relative) == base["private_sources"][relative]
    with pytest.raises(ValueError, match="preserve previous"):
        preparation.prepare(tmp_path / "pair.json", output)
    manifest["targets"]["0"][0]["kv"]["cubin_sha256"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        policy.read_manifest()


def test_inspection_never_prepares_caches(tmp_path, monkeypatch):
    completed_fixture(tmp_path, monkeypatch)
    report = tmp_path / "inspection.json"
    monkeypatch.setattr(
        preparation.shutil,
        "copytree",
        lambda *_: pytest.fail("inspection copied caches"),
    )
    preparation.prepare(tmp_path / "pair.json", report, inspect_only=True)
    assert json.loads(report.read_text())["serving_schema"] == policy.SERVING_SCHEMA
    assert not list(tmp_path.rglob("launch.json"))


def test_serving_sources_cannot_release_loader_or_conflicting_aliases(tmp_path):
    source = tmp_path / "qualified-loader.py"
    source.write_text("qualified")
    digest = sha(source)
    alias = tmp_path / "alias.py"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="conflicting qualified source aliases"):
        preparation.serving_sources({str(source): digest, str(alias): "conflict"})
    source.write_text("changed")
    with pytest.raises(ValueError, match="non-integration source changed"):
        preparation.serving_sources({str(source): digest})


@pytest.mark.parametrize("kind,memory", [("serve", 150), ("audit", 8)])
def test_kv_runner_uses_exact_workload_resources_and_separate_flag(
    tmp_path, monkeypatch, kind, memory
):
    path, manifest = runner_fixture(tmp_path, "kv")
    manifest.update(serving_schema=policy.SERVING_SCHEMA, mode="kv")
    env = runner.environment(path, manifest)
    assert env[policy.FLAG] == "kv" and env[policy.MANIFEST] == str(path)
    assert "SLIMSERVE_GLM53_RMSNORM_GEOMETRY" not in env

    def execute(argv, **kwargs):
        assert f"MemoryMax={memory}G" in argv and "MemorySwapMax=0" in argv
        assert argv[3].startswith("--unit=glm53-kv-")
        assert kwargs["env"][policy.FLAG] == "kv"
        kwargs["stdout"].write("fixed KV attempt\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    record = {}
    assert runner.execute(kind, path, manifest, record) == 0
    assert record[kind]["log_sha256"] == sha(path.parent / f"{kind}.log")


def test_kv_return_checks_control_then_kv_not_geometry(tmp_path, monkeypatch):
    path, manifest = runner_fixture(tmp_path, "return-control")
    manifest["serving_schema"] = policy.SERVING_SCHEMA
    seen = []
    monkeypatch.setattr(runner, "check_completed", lambda p: seen.append(p.parent.name))
    monkeypatch.setattr(runner, "inventory", lambda _: {"wrong": "cache"})
    with pytest.raises(ValueError, match="fresh private cache"):
        runner.preflight(path, manifest)
    assert seen == ["control", "kv"]


@pytest.mark.parametrize(
    "mutation", [None, "non-target", "return-target", "prefill", "schema"]
)
def test_kv_complete_closure_keeps_quality_failure_and_kernel_guards(
    tmp_path, monkeypatch, mutation
):
    manifests = {}
    for label, mode in policy.CASES:
        path, manifest = runner_fixture(tmp_path, label)
        manifest.update(serving_schema=policy.SERVING_SCHEMA, mode=mode)
        if mutation == "schema" and label == "kv":
            manifest["serving_schema"] = "glm53-rmsnorm-geometry-serving-v1"
        path.write_text(json.dumps(manifest))
        manifests[path] = manifest
        audit = dict(
            diagnostic_tps={},
            prefill_prompt_sha256={"32768": "prompt", "131072": "long"},
            quality_passed=label != "kv",
            historical_comparisons={
                "control": {"exact": label != "kv"},
                "failed-no-combo": {"exact": label == "kv"},
            },
        )
        if label == "kv" and mutation == "prefill":
            audit["prefill_prompt_sha256"]["131072"] = "changed"
        (path.parent / "workload-analysis.json").write_text(json.dumps(audit))
        for rank in range(4):
            folder = path.parent / f"worker-receipts/rank-{rank}"
            folder.mkdir(parents=True)
            target = dict(
                graph="graph",
                symbol="target",
                source="target",
                target=True,
                appended={"cubin_sha256": "kv", "observed_binary_index": 10}
                if label == "kv"
                else None,
            )
            other = dict(
                graph="graph",
                symbol="side",
                source="side",
                target=False,
                cubin_sha256="side",
            )
            if label == "kv" and mutation == "non-target":
                other["cubin_sha256"] = "changed"
            if label == "return-control" and mutation == "return-target":
                target["appended"] = {"cubin_sha256": "unexpected"}
            (folder / "capture-after-bindings.json").write_text(
                json.dumps(dict(bindings=[target, other]))
            )
    monkeypatch.setattr(runner, "frozen_manifest", lambda p: manifests[p])
    monkeypatch.setattr(runner, "gpu_processes", lambda: "")
    monkeypatch.setattr(runner, "check_completed", lambda p: {"label": p.parent.name})
    assert runner.close_series(tmp_path) is (mutation is None)
    result = runner.read(tmp_path / "closure.json")
    assert not result["production_qualified"]
    if mutation is None:
        assert result["kv_reproduces_failed_no_combo"]
        assert result["controls_exact_to_original"] and not result["kv_quality_passed"]
        assert not any(k.startswith("geometry_") for k in result)


def test_gpu_configuration_must_match_completed_qualification(tmp_path, monkeypatch):
    from benchmarks.kernels import check_glm53_kda_gate

    manifest, record = {"expected_gpu_config": "qualified"}, {}
    monkeypatch.setattr(check_glm53_kda_gate, "gpu_config", lambda: "changed")
    with pytest.raises(ValueError, match="identity/driver/power"):
        runner.record_gpu_config(manifest, record, "gpu_config_after")
    assert record["gpu_config_after"] == "changed"


def test_unknown_schema_never_defaults_to_control():
    with pytest.raises(ValueError, match="unknown diagnostic"):
        serving_policy({"serving_schema": "unknown"})


def test_interrupted_serving_is_audited_once_then_terminal(tmp_path, monkeypatch):
    path, manifest = runner_fixture(tmp_path)
    manifest["serving_schema"] = policy.SERVING_SCHEMA
    monkeypatch.setattr(runner, "frozen_manifest", lambda _: manifest)
    monkeypatch.setattr(runner, "preflight", lambda *_: {})
    monkeypatch.setattr(runner, "gpu_processes", lambda: "")
    attempts = []

    def execute(kind, path, manifest, record):
        attempts.append(kind)
        record[kind] = dict(status="failed" if kind == "serve" else "exited")
        if kind == "serve":
            raise KeyboardInterrupt()
        return 1

    monkeypatch.setattr(runner, "execute", execute)
    with pytest.raises(KeyboardInterrupt):
        runner.launch(path)
    assert attempts == ["serve", "audit"]
    assert runner.read(path.parent / "launch.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        runner.launch(path)
