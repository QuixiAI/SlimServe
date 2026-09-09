# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from slimserve import glm53_ordering
from slimserve import rmsnorm_diagnostic as diagnostic
from slimserve.registry import resolve


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    keys = (
        diagnostic.FLAG,
        diagnostic.MANIFEST,
        glm53_ordering.FLAG,
        "CUDA_LAUNCH_BLOCKING",
        "VLLM_FORCE_AOT_LOAD",
        "VLLM_GLM5_MHC_PREFILL_TC",
        "VLLM_GLM5_MHC_BF16_FN",
        *(
            "SLIMSERVE_GLM53_" + name
            for name in (
                "MODEL_JOURNAL",
                "MOE_JOURNAL",
                "INDEX_JOURNAL",
                "SCORE_JOURNAL",
            )
        ),
        *glm53_ordering.LEGACY_FLAGS,
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def fake_launcher(identity):
    return SimpleNamespace(
        cache_hash=identity,
        config=SimpleNamespace(kwargs={"XBLOCK": 1}, num_warps=8, num_stages=1),
    )


def setup_intervention(tmp_path, arm="control"):
    source = tmp_path / "norm.py"
    source.write_text("# source-bound fixture\n")
    target = dict(
        filename=source.name,
        source_sha256=diagnostic.sha(source),
        configs=[dict(triton_cache_hash="old"), dict(triton_cache_hash="new")],
    )
    manifest = dict(receipts=str(tmp_path / "receipts"), targets={"0": target})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    instance = diagnostic.Intervention(0, manifest, path, arm)
    tuner = SimpleNamespace(
        filename=str(source),
        launchers=[fake_launcher("new")],
        compile_results=["original"],
        configs=["original"],
        _cached_launcher="stale",
        save_cache_hook="old cache",
    )
    return instance, tuner


def test_default_off_does_not_require_torch_or_config():
    assert diagnostic.mode() == ""
    diagnostic.install(object())
    diagnostic.validate_plan(object())


@pytest.mark.parametrize("value", ["1", "0", "native", "random"])
def test_invalid_mode_fails_closed(monkeypatch, value):
    monkeypatch.setenv(diagnostic.FLAG, value)
    with pytest.raises(ValueError, match="absent, control or legacy"):
        diagnostic.mode()


@pytest.mark.parametrize("arm,identity", [("control", "new"), ("legacy", "old")])
def test_exact_replacement_and_seal(tmp_path, arm, identity):
    instance, tuner = setup_intervention(tmp_path, arm)
    calls = []

    def compile_config(saved):
        calls.append(saved)
        return SimpleNamespace(
            make_launcher=lambda: fake_launcher(saved["triton_cache_hash"])
        )

    assert instance.replace(tuner, compile_config) is tuner
    assert tuner.launchers[0].cache_hash == identity
    assert tuner._cached_launcher is None
    assert tuner.save_cache_hook is None
    assert tuner.configs is None
    assert calls == [dict(triton_cache_hash=identity)]
    instance.seal()
    with pytest.raises(ValueError, match="rebound"):
        instance.replace(tuner, compile_config)
    records = [json.loads(line) for line in instance.path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["begin", "launcher", "sealed"]
    assert records[1]["before"][0]["hash"] == "new"
    assert records[1]["after"][0]["hash"] == identity


def test_untargeted_launcher_is_not_compiled_or_modified(tmp_path):
    instance, tuner = setup_intervention(tmp_path)
    tuner.filename = str(tmp_path / "other.py")
    before = vars(tuner).copy()
    instance.replace(tuner, lambda *_: pytest.fail("unrelated compilation"))
    assert vars(tuner) == before
    with pytest.raises(ValueError, match="exactly one"):
        instance.seal()


@pytest.mark.parametrize("bad", ["source", "old_hash", "multiple", "replacement_hash"])
def test_replacement_gates_fail_before_changing_launcher(tmp_path, bad):
    instance, tuner = setup_intervention(tmp_path)
    if bad == "source":
        instance.target["source_sha256"] = "wrong"
    elif bad == "old_hash":
        tuner.launchers[0].cache_hash = "wrong"
    elif bad == "multiple":
        tuner.launchers *= 2
    before = vars(tuner).copy()
    with pytest.raises(ValueError):
        instance.replace(
            tuner,
            lambda *_: SimpleNamespace(make_launcher=lambda: fake_launcher("wrong")),
        )
    assert vars(tuner) == before


def test_relocation_checks_bytes_and_detaches_original_cache_save_hook(tmp_path):
    instance, tuner = setup_intervention(tmp_path)
    old, new = tmp_path / "original", tmp_path / "private"
    old.mkdir()
    new.mkdir()
    for folder in (old, new):
        (folder / "norm.py").write_text("identical\n")
    instance.manifest.update(original_namespace=str(old), private_namespace=str(new))
    tuner.filename = str(old / "norm.py")
    instance.relocate(tuner)
    assert tuner.filename == str(new / "norm.py")
    assert tuner.save_cache_hook is None
    tuner.filename = str(old / "norm.py")
    (new / "norm.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source differs"):
        instance.relocate(tuner)


@pytest.mark.parametrize(
    "key,value",
    [
        (glm53_ordering.FLAG, "0"),
        ("VLLM_FORCE_AOT_LOAD", "0"),
        ("CUDA_LAUNCH_BLOCKING", "1"),
        ("VLLM_GLM5_MHC_PREFILL_TC", "1"),
        ("VLLM_GLM5_MHC_BF16_FN", "0"),
        ("SLIMSERVE_GLM53_MODEL_JOURNAL", "1"),
    ],
)
def test_plan_rejects_other_modes(monkeypatch, key, value):
    monkeypatch.setenv(diagnostic.FLAG, "control")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(diagnostic, "read_manifest", lambda: None)
    with pytest.raises(ValueError):
        diagnostic.validate_plan(resolve("glm53-nvfp4-4", "rtx6000", 4, None))


def test_plan_requires_fixed_recipe_and_profile(monkeypatch):
    monkeypatch.setenv(diagnostic.FLAG, "control")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setattr(diagnostic, "read_manifest", lambda: None)
    plan = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    diagnostic.validate_plan(plan)
    for changed in (replace(plan, weight_recipe=None), replace(plan, platform="a100")):
        with pytest.raises(ValueError):
            diagnostic.validate_plan(changed)
