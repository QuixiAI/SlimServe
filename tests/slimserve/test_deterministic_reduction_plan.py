# SPDX-License-Identifier: Apache-2.0
import copy
from dataclasses import replace

import pytest

from slimserve import cli
from slimserve.deterministic_reductions import diagnostic_plan
from slimserve.hardware import Machine
from slimserve.registry import resolve


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    from slimserve import glm53_ordering

    for flag in (
        *glm53_ordering.LEGACY_FLAGS,
        glm53_ordering.FLAG,
        "SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC",
        "TORCHINDUCTOR_DETERMINISTIC",
        "TORCHINDUCTOR_BATCH_INVARIANT",
        "TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS",
        "VLLM_BATCH_INVARIANT",
    ):
        monkeypatch.delenv(flag, raising=False)


def test_candidate_changes_only_recorded_option(monkeypatch):
    before = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    snapshot = copy.deepcopy(before)
    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    candidate = diagnostic_plan(before)
    assert before == snapshot
    assert resolve("glm53-nvfp4-4", "rtx6000", 4, None) == before
    expected = copy.deepcopy(before.engine)
    expected["compilation_config"]["inductor_compile_config"] = {"deterministic": True}
    assert candidate == replace(before, engine=expected)
    assert diagnostic_plan(candidate) == candidate


def test_real_compilation_hash_changes_and_frontend_receives_policy(monkeypatch):
    import torch._inductor.config as inductor
    import torch.utils._triton as triton_utils
    from torch._inductor.codegen.triton import TritonKernel

    from vllm.config import CompilationConfig

    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    baseline = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    candidate = diagnostic_plan(baseline)
    old = CompilationConfig(**baseline.engine["compilation_config"])
    new = CompilationConfig(**candidate.engine["compilation_config"])
    assert old.compute_hash() != new.compute_hash()
    assert new.inductor_compile_config["deterministic"] is True
    # Avoid GPU probing for the backend identifier; execute the real metadata
    # generator with the real compiler options. GPU codegen still needs testing.
    monkeypatch.setattr(
        triton_utils, "triton_hash_with_backend", lambda: "test-backend"
    )
    with inductor.patch(new.inductor_compile_config):
        metadata = TritonKernel.inductor_meta_common()
    assert metadata["deterministic"] is True
    assert metadata["batch_invariant"] is False


def test_requires_native_order():
    with pytest.raises(ValueError, match="require native"):
        diagnostic_plan(resolve("glm53-nvfp4-4", "rtx6000", 4, None))


@pytest.mark.parametrize(
    "flag",
    [
        "SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC",
        "TORCHINDUCTOR_DETERMINISTIC",
        "TORCHINDUCTOR_BATCH_INVARIANT",
        "TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS",
        "VLLM_BATCH_INVARIANT",
    ],
)
def test_mixed_modes_rejected(monkeypatch, flag):
    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError):
        diagnostic_plan(resolve("glm53-nvfp4-4", "rtx6000", 4, None))


def test_other_platform_rejected(monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    with pytest.raises(ValueError, match="requires glm53"):
        diagnostic_plan(resolve("glm53-nvfp4-4", "a100", 4, None))


def test_conflicting_engine_option_rejected(monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    baseline = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    engine = copy.deepcopy(baseline.engine)
    engine["compilation_config"]["inductor_compile_config"] = {"deterministic": False}
    with pytest.raises(ValueError, match="conflicting"):
        diagnostic_plan(replace(baseline, engine=engine))


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_dry_run_uses_candidate_only_when_requested(monkeypatch, enabled):
    monkeypatch.setattr(
        cli.hardware, "detect", lambda: Machine("rtx6000", "RTX6000", 4)
    )
    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    captured = []
    monkeypatch.setattr(cli, "_show", captured.append)
    monkeypatch.setattr(cli.fetch, "ensure", lambda *a, **k: pytest.fail("download"))
    args = ["glm53-nvfp4-4", "--dry-run"]
    if enabled:
        args.append("--deterministic-reductions")
    assert cli.main(args) == 0
    plan = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    assert captured == [diagnostic_plan(plan) if enabled else plan]
