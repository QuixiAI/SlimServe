# SPDX-License-Identifier: Apache-2.0
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
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
        config=SimpleNamespace(
            kwargs={"XBLOCK": 1, "R0_BLOCK": 1024}, num_warps=8, num_stages=1
        ),
    )


def setup_intervention(tmp_path, arm="control"):
    source = tmp_path / "norm.py"
    source.write_text("# source-bound fixture\n")
    target = dict(
        filename=source.name,
        source_sha256=diagnostic.sha(source),
        configs=[
            dict(
                triton_cache_hash=key,
                XBLOCK=1,
                R0_BLOCK=1024,
                num_warps=8,
                num_stages=1,
            )
            for key in ("old", "new")
        ],
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
    assert calls == [instance.target["configs"][int(arm == "control")]]
    instance.seal()
    before = vars(tuner).copy()
    assert instance.replace(tuner, compile_config) is tuner
    assert vars(tuner) == before
    assert len(calls) == 1
    records = [json.loads(line) for line in instance.path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "begin",
        "launcher",
        "sealed",
        "launcher",
    ]
    assert records[-1]["repeated"] and records[-1]["sealed"]
    assert records[1]["before"][0]["hash"] == "new"
    assert records[1]["after"][0]["hash"] == identity


def test_untargeted_launcher_is_not_compiled_or_modified(tmp_path):
    instance, tuner = setup_intervention(tmp_path)
    tuner.filename = str(tmp_path / "other.py")
    before = vars(tuner).copy()
    instance.replace(tuner, lambda *_: pytest.fail("unrelated compilation"))
    assert vars(tuner) == before
    with pytest.raises(ValueError, match="missing source-bound"):
        instance.seal()


@pytest.mark.parametrize("arm", ["control", "legacy"])
def test_multiple_aot_objects_for_one_source_are_all_replaced(tmp_path, arm):
    instance, first = setup_intervention(tmp_path, arm)
    second = SimpleNamespace(**vars(first))
    compile_config = lambda saved: SimpleNamespace(
        make_launcher=lambda: fake_launcher(saved["triton_cache_hash"])
    )
    instance.replace(first, compile_config)
    instance.replace(second, compile_config)
    assert first.launchers[0].cache_hash == second.launchers[0].cache_hash
    assert first.launchers[0].cache_hash == ("new" if arm == "control" else "old")
    instance.seal()
    records = [json.loads(line) for line in instance.path.read_text().splitlines()]
    assert [r["binding_index"] for r in records if r.get("target")] == [1, 2]
    assert records[-1]["targets"] == 2


def compile_fake(saved):
    return SimpleNamespace(
        make_launcher=lambda: fake_launcher(saved["triton_cache_hash"])
    )


@pytest.mark.parametrize("arm", ["control", "legacy"])
def test_aliases_and_concurrent_resolution_do_not_repeat_upstream(tmp_path, arm):
    instance, tuner = setup_intervention(tmp_path, arm)
    instance.relocate = lambda obj: None
    future = SimpleNamespace(static_autotuner=tuner)
    alias = SimpleNamespace(static_autotuner=tuner)
    barrier = threading.Barrier(8)
    calls = []

    def upstream(future, timeout=None):
        calls.append(timeout)
        return future.static_autotuner

    def resolve(i):
        barrier.wait(timeout=5)
        return instance.resolve(future if i % 2 else alias, upstream, compile_fake, 7)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(result is tuner for result in pool.map(resolve, range(8)))
    assert calls == [7]
    assert len(instance.replaced) == 1 and instance.resolutions == 8
    assert tuner.launchers[0].cache_hash == ("new" if arm == "control" else "old")
    instance.seal()
    instance.resolve(alias, upstream, compile_fake)
    assert calls == [7] and instance.resolutions == 9


def test_strong_reference_prevents_id_reuse(tmp_path):
    instance, fixture = setup_intervention(tmp_path)

    class Tuner:
        pass

    tuner = Tuner()
    vars(tuner).update(vars(fixture))
    reference = weakref.ref(tuner)
    instance.replace(tuner, compile_fake)
    identity = id(tuner)
    del tuner
    assert reference() is not None
    assert instance.replaced[identity] is reference()


@pytest.mark.parametrize("change", ["source", "hash", "config", "filename"])
def test_repeat_cannot_bypass_source_or_launcher_checks(tmp_path, change):
    instance, tuner = setup_intervention(tmp_path)
    instance.replace(tuner, compile_fake)
    if change == "source":
        instance.target["source_sha256"] = "wrong"
    elif change == "filename":
        tuner.filename += ".different"
    elif change == "hash":
        tuner.launchers[0].cache_hash = "wrong"
    else:
        tuner.launchers[0].config.num_warps = 16
    with pytest.raises(ValueError):
        instance.replace(tuner, compile_fake)


def test_native_reset_can_be_reapplied_only_before_seal(tmp_path):
    instance, tuner = setup_intervention(tmp_path, "legacy")
    instance.replace(tuner, compile_fake)
    tuner.launchers = [fake_launcher("new")]
    tuner._cached_launcher = "stale native callable"
    instance.replace(tuner, compile_fake)
    assert tuner.launchers[0].cache_hash == "old" and tuner._cached_launcher is None
    instance.seal()
    tuner.launchers = [fake_launcher("new")]
    with pytest.raises(ValueError, match="cached launcher"):
        instance.replace(tuner, compile_fake)


def test_new_target_after_seal_fails(tmp_path):
    instance, tuner = setup_intervention(tmp_path)
    second = SimpleNamespace(**vars(tuner))
    instance.replace(tuner, compile_fake)
    instance.seal()
    with pytest.raises(ValueError, match="new RMSNorm target"):
        instance.replace(second, compile_fake)


def test_unseen_legacy_launcher_is_not_accepted_as_native(tmp_path):
    instance, tuner = setup_intervention(tmp_path, "legacy")
    tuner.launchers = [fake_launcher("old")]
    with pytest.raises(ValueError, match="cached launcher"):
        instance.replace(tuner, compile_fake)


def graph_module(tmp_path, **bindings):
    source = tmp_path / "graph.py"
    source.write_text("# graph fixture\n")
    return SimpleNamespace(__file__=str(source), call=lambda: None, **bindings)


def test_graph_hook_covers_non_static_objects_and_aliases(tmp_path):
    instance, tuner = setup_intervention(tmp_path, "legacy")
    instance.relocate = lambda obj: None
    other = SimpleNamespace(**vars(tuner))
    module = graph_module(tmp_path, norm=tuner, alias=tuner, other=other)
    instance.bind_graph(module, compile_fake)
    assert len(instance.replaced) == 2
    assert len(instance.graph_bindings) == 3
    assert tuner.launchers[0].cache_hash == other.launchers[0].cache_hash == "old"
    instance.verify_graphs([module])
    instance.seal()
    instance.bind_graph(module, compile_fake)
    instance.verify_graphs([module])


def test_static_source_coverage_does_not_imply_graph_coverage(tmp_path):
    instance, tuner = setup_intervention(tmp_path, "legacy")
    instance.replace(tuner, compile_fake)
    missing = SimpleNamespace(**vars(tuner))
    missing.launchers = [fake_launcher("new")]
    module = graph_module(tmp_path, norm=missing)
    with pytest.raises(ValueError, match="uncovered"):
        instance.verify_graphs([module])


@pytest.mark.parametrize("change", ["object", "hash", "source", "filename", "removed"])
def test_graph_coverage_detects_post_load_changes(tmp_path, change):
    instance, tuner = setup_intervention(tmp_path)
    instance.relocate = lambda obj: None
    module = graph_module(tmp_path, norm=tuner)
    instance.bind_graph(module, compile_fake)
    if change == "object":
        module.norm = SimpleNamespace(**vars(tuner))
    elif change == "filename":
        tuner.filename += ".renamed"
    elif change == "removed":
        del module.norm
    elif change == "hash":
        tuner.launchers[0].cache_hash = "wrong"
    else:
        instance.target["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="uncovered"):
        instance.verify_graphs([module])


def test_graph_verification_requires_a_real_binding(tmp_path):
    instance, tuner = setup_intervention(tmp_path)
    instance.replace(tuner, compile_fake)
    with pytest.raises(ValueError, match="missing RMSNorm graph coverage"):
        instance.verify_graphs([])


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
