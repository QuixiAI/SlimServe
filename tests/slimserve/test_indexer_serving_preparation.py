# SPDX-License-Identifier: Apache-2.0
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kernels import glm53_geometry_workload as workload
from benchmarks.kernels import prepare_glm53_indexer_correction_serving as preparation
from benchmarks.kernels import run_glm53_geometry_serving as runner
from slimserve import glm53_ordering
from slimserve import indexer_correction_diagnostic as policy
from slimserve.rmsnorm_diagnostic import sha
from tests.slimserve.test_geometry_serving_runner import fixture as runner_fixture
from tests.slimserve.test_kv_loader_audit import prepared_fixture


def completed_fixture(tmp_path, monkeypatch):
    base = prepared_fixture(tmp_path, monkeypatch)
    base.update(
        schema=policy.SCHEMA,
        selection_capacity=8192,
        weight_sha256={},
        qualified_leaf_cases=[],
        expected_gpu_config="hardware",
    )
    # All four ranks share one qualified JIT file; copy exactly once per cache.
    correction = base["targets"]["0"][0]["kv"]
    for targets in base["targets"].values():
        for target in targets:
            target.pop("kv")
            target["correction"] = copy.deepcopy(correction)
    base["private_sources"] = {correction["relative"]: correction["source_sha256"]}
    path = tmp_path / "qualified.json"
    path.write_text(json.dumps(base))
    reference = dict(path=str(path), sha256=sha(path))
    graphs = {
        m: {str(r): reference for r in range(4)} for m in ("control", "correction")
    }
    monkeypatch.setattr(
        preparation,
        "completed_evidence",
        lambda *_: (base, graphs, reference, {str(path): sha(path)}),
    )
    monkeypatch.setattr(workload, "reference_evidence", lambda: ({}, {}, {}, {}))
    original_check = subprocess.check_output
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kw: (
            "" if cmd == ["git", "status", "--short"] else original_check(cmd, **kw)
        ),
    )
    return base


def test_private_shared_source_copies_and_manifest_roundtrip(tmp_path, monkeypatch):
    base = completed_fixture(tmp_path, monkeypatch)
    output = tmp_path / "serving"
    preparation.prepare(tmp_path / "pair.json", output)
    prepared = json.loads((output / "preparation.json").read_text())
    assert [(r["label"], r["mode"]) for r in prepared["runs"]] == list(policy.CASES)
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    private_sources = []
    relative = next(iter(base["private_sources"]))
    for label, mode in policy.CASES:
        path = output / label / "manifest.json"
        manifest = json.loads(path.read_text())
        private = Path(manifest["private_namespace"])
        monkeypatch.setenv(policy.FLAG, mode)
        monkeypatch.setenv(policy.MANIFEST, str(path))
        monkeypatch.setenv("VLLM_CACHE_ROOT", manifest["cache_root"])
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(private / "inductor_cache"))
        assert policy.read_manifest() == (manifest, path)
        assert runner.inventory(private) == {
            **base["original_files"],
            **base["private_sources"],
        }
        private_sources.append(private / relative)
    assert len({p.stat().st_ino for p in private_sources}) == 3
    private_sources[1].write_text("mutated candidate copy")
    assert (
        sha(private_sources[0])
        == sha(private_sources[2])
        == base["private_sources"][relative]
    )
    with pytest.raises(ValueError, match="preserve previous"):
        preparation.prepare(tmp_path / "pair.json", output)
    manifest["selection_capacity"] = 16384
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        policy.read_manifest()


def test_qualified_kernel_and_loader_never_released(tmp_path):
    path = tmp_path / "qualified.py"
    path.write_text("original")
    digest = sha(path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="non-integration source changed"):
        preparation.serving_sources({str(path): digest})


@pytest.mark.parametrize("kind,memory", [("serve", 150), ("audit", 8)])
def test_correction_runner_separate_flag_resources(tmp_path, monkeypatch, kind, memory):
    path, manifest = runner_fixture(tmp_path, "correction")
    manifest.update(serving_schema=policy.SERVING_SCHEMA, mode="correction")
    env = runner.environment(path, manifest)
    assert env[policy.FLAG] == "correction"
    assert env[policy.MANIFEST] == str(path)
    assert "SLIMSERVE_GLM53_KV_DIAGNOSTIC" not in env

    def execute(argv, **kwargs):
        assert f"MemoryMax={memory}G" in argv and "MemorySwapMax=0" in argv
        assert argv[3].startswith("--unit=glm53-correction-")
        assert kwargs["env"][policy.FLAG] == "correction"
        kwargs["stdout"].write("prescribed attempt\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    record = {}
    assert runner.execute(kind, path, manifest, record) == 0


def test_return_checks_control_and_correction_predecessors(tmp_path, monkeypatch):
    path, manifest = runner_fixture(tmp_path, "return-control")
    manifest["serving_schema"] = policy.SERVING_SCHEMA
    seen = []
    monkeypatch.setattr(runner, "check_completed", lambda p: seen.append(p.parent.name))
    monkeypatch.setattr(runner, "inventory", lambda _: {"wrong": "cache"})
    with pytest.raises(ValueError, match="fresh private cache"):
        runner.preflight(path, manifest)
    assert seen == ["control", "correction"]


@pytest.mark.parametrize("change", [None, "count", "phases", "capacity", "leaves"])
def test_bound_leaf_qualification_is_required(change):
    report, row = dict(leaf_cases=60, leaf_phase_observations=300), dict(leaf_cases=60)
    manifest = dict(selection_capacity=8192, qualified_leaf_cases=[None] * 120)
    if change == "count":
        report["leaf_cases"] = 59
    elif change == "phases":
        report["leaf_phase_observations"] = 299
    elif change == "capacity":
        manifest["selection_capacity"] = 16384
    elif change == "leaves":
        manifest["qualified_leaf_cases"].pop()
    if change:
        with pytest.raises(ValueError, match="bound-leaf"):
            preparation.check_completed_report(report, row, manifest)
    else:
        preparation.check_completed_report(report, row, manifest)
