# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kernels import glm53_geometry_workload as workload
from benchmarks.kernels import prepare_glm53_indexer_correction_serving as original
from benchmarks.kernels import prepare_glm53_prompt_score_serving as preparation
from benchmarks.kernels import run_glm53_geometry_serving as runner
from slimserve import glm53_ordering
from slimserve import prompt_score_diagnostic as policy
from slimserve.glm53_serving_diagnostic import cases
from slimserve.glm53_serving_diagnostic import policy as serving_policy
from tests.slimserve.test_geometry_serving_runner import fixture as runner_fixture
from tests.slimserve.test_indexer_serving_preparation import completed_fixture


def test_prepare_control_only_loaders_with_separate_prompt_arms(tmp_path, monkeypatch):
    base = completed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(preparation, "completed_evidence", original.completed_evidence)
    output = tmp_path / "series"
    preparation.prepare(tmp_path / "pair.json", output)
    prepared = json.loads((output / "preparation.json").read_text())
    assert [(r["label"], r["mode"]) for r in prepared["runs"]] == list(policy.CASES)
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    for label, mode in policy.CASES:
        path = output / label / "manifest.json"
        manifest = json.loads(path.read_text())
        monkeypatch.setenv(policy.FLAG, mode)
        monkeypatch.setenv(policy.MANIFEST, str(path))
        monkeypatch.setenv(policy.CHUNKS, "1" if label == "chunked" else "0")
        monkeypatch.setenv("VLLM_CACHE_ROOT", manifest["cache_root"])
        monkeypatch.setenv(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path(manifest["private_namespace"]) / "inductor_cache"),
        )
        assert policy.read_manifest() == (manifest, path)
        assert serving_policy(manifest) is policy and cases(manifest) == policy.CASES
        assert manifest["mode"] == "control"
        assert runner.inventory(Path(manifest["private_namespace"])) == {
            **base["original_files"],
            **base["private_sources"],
        }
        monkeypatch.setenv(policy.CHUNKS, "0" if label == "chunked" else "1")
        with pytest.raises(ValueError, match="prescribed arm"):
            policy.read_manifest()


@pytest.mark.parametrize(
    "label,chunk_flag", [("control", "0"), ("chunked", "1"), ("return-control", "0")]
)
@pytest.mark.parametrize("kind,memory", [("serve", 150), ("audit", 8)])
def test_prescribed_flag_and_scopes(
    tmp_path, monkeypatch, label, chunk_flag, kind, memory
):
    path, manifest = runner_fixture(tmp_path, label)
    manifest.update(serving_schema=policy.SERVING_SCHEMA, mode="control")
    env = runner.environment(path, manifest)
    assert env[policy.FLAG] == "control" and env[policy.CHUNKS] == chunk_flag
    assert not any(
        env.get(k)
        for k in ("SLIMSERVE_GLM53_INDEXER_CORRECTION", "SLIMSERVE_GLM53_KV_DIAGNOSTIC")
    )

    def execute(argv, **kwargs):
        assert f"MemoryMax={memory}G" in argv and "MemorySwapMax=0" in argv
        assert kwargs["env"][policy.CHUNKS] == chunk_flag
        kwargs["stdout"].write("prescribed\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.execute(kind, path, manifest, {}) == 0


def test_noop_by_default_and_no_arithmetic_intervention(monkeypatch):
    monkeypatch.delenv(policy.FLAG, raising=False)
    policy.install(object())
    policy.validate_plan(object())
    monkeypatch.setenv(policy.FLAG, "correction")
    with pytest.raises(ValueError, match="unchanged control"):
        policy.mode()
    monkeypatch.setenv(policy.FLAG, "control")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setenv("SLIMSERVE_GLM53_INDEXER_CORRECTION", "control")
    with pytest.raises(ValueError, match="cannot combine"):
        policy.validate_environment()


def test_model_score_candidate_requires_exact_equality(monkeypatch):
    docs, scores = [{}], [dict(text=[1.0], needles=[2.0])]
    references = {"control": [dict(text=[1.0], needles=[2.0])]}
    monkeypatch.setattr(workload, "input_identity", lambda _: {})
    monkeypatch.setattr(workload, "compare", lambda *a: {"passed": True})
    reference_docs = {"control": [{}], "return-control": [{}]}
    assert workload.check_scores("chunked", docs, scores, reference_docs, references)[
        "diagnostic_gate_passed"
    ]
    scores[0]["text"][0] += 0.001
    assert not workload.check_scores(
        "chunked", docs, scores, reference_docs, references
    )["diagnostic_gate_passed"]


def test_return_checks_both_predecessors(tmp_path, monkeypatch):
    path, manifest = runner_fixture(tmp_path, "return-control")
    manifest.update(serving_schema=policy.SERVING_SCHEMA, mode="control")
    seen = []
    monkeypatch.setattr(runner, "check_completed", lambda p: seen.append(p.parent.name))
    monkeypatch.setattr(runner, "inventory", lambda _: {"wrong": "cache"})
    with pytest.raises(ValueError, match="fresh private cache"):
        runner.preflight(path, manifest)
    assert seen == ["control", "chunked"]


def test_qualified_code_cannot_be_released(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text("changed")
    with pytest.raises(ValueError, match="non-integration source changed"):
        preparation.serving_sources({str(path): "previous"})


def test_install_uses_qualified_control_lifecycle_once(tmp_path, monkeypatch):
    import torch

    from benchmarks.kernels import glm53_indexer_correction_serving as lifecycle
    from tests.slimserve.test_indexer_correction_serving import runner_fixture

    value = runner_fixture()
    manifest = {"mode": "control"}
    path = tmp_path / "manifest.json"
    installed = []

    class Control:
        def __init__(self, rank, data, filename):
            assert rank == 0 and data is manifest and filename == path

        def install(self, runner):
            installed.append(runner)

    monkeypatch.setenv(policy.FLAG, "control")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setattr(policy, "read_manifest", lambda: (manifest, path))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(lifecycle, "ServingIndexerCorrection", Control)
    policy.install(value)
    assert installed == [value]
    with pytest.raises(ValueError, match="once"):
        policy.install(value)
