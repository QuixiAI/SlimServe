# SPDX-License-Identifier: Apache-2.0
"""Actual Torch load/capture plumbing, CPU fake driver; not GPU qualification."""

import copy
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch._functorch._aot_autograd.aot_autograd_result import (
    BundledAOTAutogradResult,
    BundledCompiledForward,
)
from torch._inductor.codecache import PyCodeCache
from torch._inductor.output_code import CompiledFxGraph, maybe_realign_inputs
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch._inductor.triton_bundler import TritonBundler
from torch._inductor.utils import BoxedBool
from torch.utils._ordered_set import OrderedSet

from benchmarks.kernels.audit_glm53_geometry_loader import json_lines
from benchmarks.kernels.audit_glm53_geometry_serving import audit_worker, check_worker
from benchmarks.kernels.audit_glm53_kv_loader import check_graph_records
from benchmarks.kernels.glm53_artifact_roots import discover
from benchmarks.kernels.glm53_geometry_serving import compare_qualified, stable_bindings
from benchmarks.kernels.glm53_kv_serving import ServingKV
from slimserve import glm53_ordering
from slimserve import kv_diagnostic as policy
from slimserve.registry import resolve
from slimserve.rmsnorm_diagnostic import sha
from tests.slimserve.test_binary_observer import make_compiled
from tests.slimserve.test_kv_loader import fixture
from vllm.compilation.caching import StandaloneCompiledArtifacts


def serving_fixture(
    tmp_path,
    monkeypatch,
    mode="kv",
    *,
    kernel_fixture=None,
    serving_type=ServingKV,
    serving_policy=policy,
    separate_owners=False,
):
    f = (kernel_fixture or fixture)(tmp_path, monkeypatch, mode)
    manifest = f.manifest
    original, private = Path(manifest["original_namespace"]), f.loader.private
    target = manifest["targets"]["0"][0]
    extra_key = f.loader.controller.extra_key
    if separate_owners:
        from torch._inductor.codecache import StaticAutotunerFuture

        second = type(f.tuner).__new__(type(f.tuner))
        vars(second).update(vars(f.tuner))
        second.launchers = []

        def precompile(**kwargs):
            second.launchers = [second.compile_results[0].make_launcher()]

        second.precompile = precompile
        second_future = StaticAutotunerFuture(second)
        second_future.reload_kernel_from_src = lambda: pytest.fail("source reload")
        sys.modules["glm53_kv_fixture"].future2 = second_future
    side, side_path, _ = make_compiled(f.loader.cache / "triton/0", "side")
    relative = "inductor_cache/ss/side.py"
    for root in (original, private):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unrelated static source\n")
    manifest["original_files"][relative] = sha(path)
    tuner = SimpleNamespace(filename=str(path), compile_results=[side], launchers=[])

    def side_result():
        if not tuner.launchers:
            tuner.run = side.make_launcher()
            tuner.launchers = [tuner.run]
        return tuner

    sys.modules["glm53_kv_fixture"].side_result = side_result
    store, graphs, rows = StandaloneCompiledArtifacts(), {}, []
    target["static_graph_uses"] = []
    for index in range(7):
        relative_graph = f"inductor_cache/ra/graph{index}.py"
        text = "from glm53_kv_fixture import side_result\nside = side_result()\n"
        if index < 2:
            future_name = "future2" if separate_owners and index == 1 else "future"
            text += (
                f"from glm53_kv_fixture import {future_name} as future\n"
                "combo = future.result()\n"
            )
        text += "def call(args, stream):\n"
        if index < 2:
            text += "    combo.run(*args, stream=stream)\n"
        text += "    side.run(*args, stream=stream)\n"
        for root in (original, private):
            path = root / relative_graph
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        graphs[relative_graph] = manifest["original_files"][relative_graph] = sha(path)
        if index < 2:
            target["static_graph_uses"].append(
                dict(graph=str(original / relative_graph), symbol="combo")
            )
        graph = CompiledFxGraph.__new__(CompiledFxGraph)
        graph.cache_key, graph.source_code, graph.cache_linemap = path.stem, text, []
        graph.current_callable = graph.compiled_fn_runner = None
        graph.inputs_to_check, graph.mutated_input_idxs = [0, 4], OrderedSet([4])
        graph._defers_input_alignment, graph._wrap_compiled_regions = True, False
        entry = BundledAOTAutogradResult.__new__(BundledAOTAutogradResult)
        entry.compiled_fw = BundledCompiledForward.__new__(BundledCompiledForward)
        entry.compiled_fw.result, entry.compiled_bw = graph, None
        blob = pickle.dumps(pickle.dumps((entry, {}, {})))
        for n in range(46 // 7 + (index < 46 % 7)):
            store.insert(f"graph{index}layer{n}", "shape", blob)
        for symbol in ["combo", "side"] if index < 2 else ["side"]:
            is_target = symbol == "combo"
            source = target["relative"] if is_target else relative
            row = dict(
                graph=relative_graph,
                graph_sha256=sha(path),
                symbol=symbol,
                source=source,
                source_sha256=sha(private / source),
                target=is_target,
                referenced=True,
                selected=[target["selected"]]
                if is_target
                else [
                    dict(
                        hash=side_path.parent.name,
                        config=dict(
                            side.config.kwargs,
                            num_warps=side.config.num_warps,
                            num_stages=side.config.num_stages,
                        ),
                    )
                ],
                cubin_sha256=target["cubin_sha256"] if is_target else sha(side_path),
                module_index=index + 100,
                observed_binary_index=100,
            )
            if is_target:
                row.update(
                    dispatch="direct_combo"
                    if mode == "control"
                    else (
                        "combo_then_indexer_correction"
                        if extra_key == "correction"
                        else "combo_then_kv"
                    ),
                    appended=None,
                )
                if mode != "control":
                    kv = target[extra_key]
                    row["appended"] = dict(
                        source=kv["relative"],
                        source_sha256=kv["source_sha256"],
                        selected=kv["selected"],
                        cubin_sha256=kv["cubin_sha256"],
                        observed_binary_index=200,
                    )
                    if extra_key == "correction":
                        row["appended"].update(
                            selection_capacity=8192,
                            selection_bytes=8194 * 128,
                            selection_dtype="uint8",
                            selection_device="cuda:0",
                        )
            rows.append(row)
    manifest["expected_graphs"] = {"0": graphs}
    manifest["artifact_roots"] = {"0": discover(store, private, graphs)}
    model = private / "rank_0_0/model"
    model.parent.mkdir()
    model.write_bytes(b"model-fixture")
    manifest["original_files"]["rank_0_0/model"] = sha(model)
    qualified = tmp_path / "qualified.json"
    qualified.write_text(json.dumps(dict(graphs=7, target_bindings=2, bindings=rows)))
    manifest.update(
        serving_schema=serving_policy.SERVING_SCHEMA,
        mode=mode,
        receipts=str(tmp_path / "targets"),
        worker_receipts=str(tmp_path / "workers"),
        qualified_graphs={
            mode: {"0": dict(path=str(qualified), sha256=sha(qualified))}
        },
    )
    path = tmp_path / "serving-manifest.json"
    path.write_text(json.dumps(manifest))

    def deserialize(data):
        TritonBundler.load_autotuners(["static-fixture"])
        graph = pickle.loads(data)[0].compiled_fw.result
        graph.after_deserialization(SimpleNamespace(unwrap=lambda g: {}))
        maybe_realign_inputs(
            BoxedBool(False), graph, graph.inputs_to_check, graph.mutated_input_idxs
        )
        return SimpleNamespace(graph=graph)

    monkeypatch.setattr(AOTCompiledArtifact, "deserialize", staticmethod(deserialize))
    monkeypatch.setattr(
        TritonBundler, "load_autotuners", classmethod(lambda cls, ts: ts)
    )
    diagnostic = serving_type(0, manifest, path)
    diagnostic.loader.compile_replacement = f.loader.compile_replacement
    return diagnostic, store, f


def audited_fixture(tmp_path, monkeypatch, mode="kv"):
    diagnostic, store, f = serving_fixture(tmp_path, monkeypatch, mode)

    def capture():
        assert diagnostic.loader.controller.sealed
        assert not diagnostic.loader.observer.sealed
        later, _, _ = make_compiled(diagnostic.loader.cache / "triton/0", "late")
        later.make_launcher()
        path = diagnostic.private / "inductor_cache/gg/nonroot.py"
        path.write_text("def call(): pass\n")
        PyCodeCache.load_by_key_path("nonroot", str(path))
        # Cached callbacks may recur, but cannot add a binding after sealing.
        root = next(
            m
            for m in diagnostic.roots.roots(store, PyCodeCache.modules)
            if hasattr(m, "combo")
        )
        diagnostic.loader.controller.bind_graph(
            root, diagnostic.loader.compile_replacement
        )
        return 42

    runner = SimpleNamespace(capture_model=capture)
    original = StandaloneCompiledArtifacts.load_all
    diagnostic.install(runner)
    try:
        store.load_all()
        store.load_all()
        assert runner.capture_model() == 42
        assert diagnostic.summary["status"] == "capture-qualified"
    finally:
        diagnostic.close()
    assert StandaloneCompiledArtifacts.load_all is original
    assert runner.capture_model is capture and not diagnostic.loader.installed
    assert all(s.closed for s in diagnostic.streams.values())
    snapshots = {
        phase: (
            json.loads(Path(row["bindings"]).read_text()),
            json.loads(Path(row["modules"]).read_text()),
        )
        for phase, row in diagnostic.summary["snapshots"].items()
    }
    records = (
        dict(diagnostic.manifest, rank=0),
        sha(diagnostic.path),
        diagnostic.summary,
        snapshots,
        json_lines(diagnostic.folder / "binary-loads.jsonl"),
        json_lines(diagnostic.folder / "artifact-roots.jsonl"),
        json_lines(diagnostic.folder / "lifecycle.jsonl"),
        json_lines(diagnostic.folder / "loader-events.jsonl"),
        diagnostic.qualified,
    )
    return records, diagnostic, f


@pytest.mark.parametrize("mode", ["control", "kv"])
def test_actual_serving_lifecycle_and_offline_audit(tmp_path, monkeypatch, mode):
    records, diagnostic, _ = audited_fixture(tmp_path, monkeypatch, mode)
    assert all(e["event"].startswith("binary_") for e in records[4])
    assert all(e["event"].startswith("kv_") for e in records[7])
    result = check_worker(*records, graph_checker=check_graph_records)
    assert result["before-forward"]["graphs"] == 7
    assert result["capture-after"]["target_bindings"] == 2
    assert result["capture-after"]["appended_launchers"] == (2 if mode == "kv" else 0)
    assert result["capture-after"]["imported_modules"] == 8
    report = audit_worker(diagnostic.path, diagnostic.manifest, 0)
    assert report["status"] == "complete", report


@pytest.mark.parametrize(
    "change",
    [
        "extra_index",
        "extra_binary",
        "extra_config",
        "extra_source",
        "late_owner",
        "late_graph",
        "extra_seal",
        "missing_seal",
        "non_target",
    ],
)
def test_serving_audit_rejects_changed_appended_or_unrelated_evidence(
    tmp_path, monkeypatch, change
):
    records, _, _ = audited_fixture(tmp_path, monkeypatch)
    records = copy.deepcopy(records)
    graph = records[3]["capture-after"][0]
    target = next(r for r in graph["bindings"] if r["target"])
    events = records[7]
    if change.startswith("extra_") and change != "extra_seal":
        key, value = {
            "extra_index": ("observed_binary_index", 10000),
            "extra_binary": ("cubin_sha256", "wrong"),
            "extra_config": ("selected", {}),
            "extra_source": ("source_sha256", "wrong"),
        }[change]
        target["appended"][key] = value
    elif change == "late_owner":
        events.append(
            copy.deepcopy(next(e for e in events if e["event"] == "kv_binding"))
        )
    elif change == "late_graph":
        event = copy.deepcopy(
            next(e for e in events if e["event"] == "kv_graph_binding")
        )
        event["symbol"] = "foreign"
        events.append(event)
    elif change == "extra_seal":
        events.append(
            copy.deepcopy(next(e for e in events if e["event"] == "kv_sealed"))
        )
    elif change == "missing_seal":
        events[:] = [e for e in events if e["event"] != "kv_sealed"]
    else:
        next(r for r in graph["bindings"] if not r["target"])["cubin_sha256"] = "wrong"
    with pytest.raises((ValueError, KeyError)):
        check_worker(*records, graph_checker=check_graph_records)


def test_stable_comparison_ignores_only_local_indices():
    row = dict(
        graph="root",
        source="source",
        symbol="symbol",
        module_index=1,
        observed_binary_index=2,
        appended=dict(
            observed_binary_index=3,
            selected={"hash": "hash"},
            cubin_sha256="image",
            source="kv",
        ),
    )
    original = dict(graphs=1, target_bindings=1, bindings=[row])
    other = copy.deepcopy(original)
    other["bindings"][0]["appended"]["observed_binary_index"] = 42
    compare_qualified(original, other)
    assert row["appended"]["observed_binary_index"] == 3
    other["bindings"][0]["appended"]["cubin_sha256"] = "changed"
    with pytest.raises(ValueError, match="source/config/binary"):
        compare_qualified(original, other)
    assert "selected" in stable_bindings(original)[0]["appended"]


def test_default_kv_policy_is_inert(monkeypatch):
    monkeypatch.delenv(policy.FLAG, raising=False)
    policy.install(object())
    policy.validate_plan(object())


@pytest.mark.parametrize(
    "change",
    [
        None,
        "dtype",
        "kv_dtype",
        "tp",
        "ep",
        "pp",
        "model",
        "speculation",
        "compiler",
        "gpu",
    ],
)
def test_worker_scope_is_checked_before_loader_install(tmp_path, monkeypatch, change):
    import torch

    from benchmarks.kernels import glm53_kv_serving

    monkeypatch.setenv(policy.FLAG, "kv")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda: (12, 0) if change != "gpu" else (8, 0),
    )
    monkeypatch.setattr(
        policy, "read_manifest", lambda: ({}, tmp_path / "manifest.json")
    )
    installed = []
    monkeypatch.setattr(
        glm53_kv_serving,
        "ServingKV",
        lambda *a: SimpleNamespace(install=lambda r: installed.append(r)),
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=45)
        ),
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
        ),
        compilation_config=SimpleNamespace(inductor_compile_config={}),
        speculative_config=None,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
    )
    if change == "dtype":
        runner.dtype = torch.float16
    elif change == "kv_dtype":
        runner.kv_cache_dtype = torch.float8_e4m3fn
    elif change == "tp":
        runner.parallel_config.tensor_parallel_size = 2
    elif change == "ep":
        runner.parallel_config.enable_expert_parallel = True
    elif change == "pp":
        runner.parallel_config.pipeline_parallel_size = 2
    elif change == "model":
        runner.model_config.hf_text_config.hidden_size = 2048
    elif change == "speculation":
        runner.speculative_config = object()
    elif change == "compiler":
        runner.compilation_config.inductor_compile_config["combo_kernels"] = False
    if change:
        with pytest.raises(ValueError):
            policy.install(runner)
        assert not installed
    else:
        policy.install(runner)
        assert installed == [runner]
        with pytest.raises(ValueError, match="installed once"):
            policy.install(runner)


@pytest.mark.parametrize(
    "failure",
    ["early_capture", "fallback", "capture_mutation", "extra_store", "foreign_hook"],
)
def test_kv_failure_closes_streams_and_restores_owned_hooks(
    tmp_path, monkeypatch, failure
):
    diagnostic, store, f = serving_fixture(tmp_path, monkeypatch)

    def capture():
        if failure == "capture_mutation":
            f.results["kv"].config.num_stages = 2

    runner = SimpleNamespace(capture_model=capture)
    original = StandaloneCompiledArtifacts.load_all
    if failure == "fallback":
        monkeypatch.setattr(
            TritonBundler, "load_autotuners", classmethod(lambda cls, ts: [])
        )
    diagnostic.install(runner)
    with pytest.raises(ValueError):
        if failure == "early_capture":
            runner.capture_model()
        elif failure == "foreign_hook":
            runner.capture_model = lambda: None
            diagnostic.close()
        else:
            store.load_all()
            if failure == "extra_store":
                StandaloneCompiledArtifacts().load_all()
            runner.capture_model()
    assert diagnostic.closed and not diagnostic.loader.installed
    assert StandaloneCompiledArtifacts.load_all is original
    assert all(s.closed for s in diagnostic.streams.values())
    if failure != "foreign_hook":
        assert runner.capture_model is capture
        assert diagnostic.summary["status"] == "failed"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("SLIMSERVE_GLM53_RMSNORM_GEOMETRY", "geometry"),
        ("SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC", "control"),
        ("VLLM_FORCE_AOT_LOAD", "0"),
        (glm53_ordering.FLAG, "0"),
        ("TORCHINDUCTOR_DETERMINISTIC", "1"),
        ("VLLM_GLM5_MHC_BF16_FN", "0"),
    ],
)
def test_policy_rejects_conflicting_modes(monkeypatch, flag, value):
    monkeypatch.setenv(policy.FLAG, "kv")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setenv(flag, value)
    monkeypatch.setattr(
        policy,
        "read_manifest",
        lambda: pytest.fail("conflicting mode reached manifest"),
    )
    with pytest.raises(ValueError):
        policy.validate_plan(resolve("glm53-nvfp4-4", "rtx6000", 4, None))
