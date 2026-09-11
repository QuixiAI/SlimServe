# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.kernels import prepare_glm53_kv_loader as preparation
from benchmarks.kernels.audit_glm53_kv_loader import check_graph_records
from benchmarks.kernels.check_glm53_kv_loader import read_manifest
from benchmarks.kernels.glm53_artifact_roots import discover
from benchmarks.kernels.glm53_kv_loader import SCHEMA, KVLoader
from slimserve.rmsnorm_diagnostic import NAMESPACE, sha
from tests.slimserve.test_artifact_roots import serialized_fixture
from tests.slimserve.test_kv_loader import fixture, load_graphs


def actual_receipts(tmp_path, monkeypatch, mode="kv"):
    f = fixture(tmp_path, monkeypatch, mode)
    f.manifest.update(rank=0, mode=mode)
    path = f.loader.manifest_path
    path.write_text(json.dumps(f.manifest))
    events = []
    replacement = f.loader.compile_replacement
    f.loader = KVLoader(0, f.manifest, path, mode, emit=events.append)
    f.loader.compile_replacement = replacement
    with f.loader.intercept():
        modules = load_graphs(f)
        graphs = f.loader.controller.seal(modules)
        f.loader.observer.seal()
        f.loader.controller.verify_graphs(modules)
    return (
        f.manifest,
        sha(path),
        graphs,
        [e for e in events if e["event"].startswith("binary_")],
        [e for e in events if e["event"].startswith("kv_")],
    )


@pytest.mark.parametrize("mode", ["control", "kv"])
def test_offline_audit_joins_real_torch_loader_receipts(tmp_path, monkeypatch, mode):
    records = actual_receipts(tmp_path, monkeypatch, mode)
    result = check_graph_records(*records)
    assert result["graphs"] == result["target_bindings"] == 2
    assert result["appended_launchers"] == (2 if mode == "kv" else 0)


@pytest.mark.parametrize(
    "change",
    [
        "missing_graph",
        "dispatch",
        "original",
        "extra_index",
        "extra_config",
        "extra_source",
        "extra_image",
        "unsealed_binary",
        "unsealed_controller",
        "begin",
        "graph_receipt",
        "binding_index",
        "late_begin",
        "coverage",
    ],
)
def test_offline_kv_audit_rejects_changed_receipts(tmp_path, monkeypatch, change):
    manifest, digest, graphs, binaries, events = actual_receipts(tmp_path, monkeypatch)
    if change == "missing_graph":
        graphs["bindings"].pop()
    elif change == "dispatch":
        graphs["bindings"][0]["dispatch"] = "direct_combo"
    elif change == "original":
        graphs["bindings"][0]["selected"][0]["hash"] = "wrong"
    elif change == "extra_index":
        graphs["bindings"][0]["appended"]["observed_binary_index"] = 1
    elif change == "extra_config":
        graphs["bindings"][0]["appended"]["selected"]["config"]["num_stages"] = 2
    elif change == "extra_source":
        graphs["bindings"][0]["appended"]["source"] = "missing.py"
    elif change == "extra_image":
        graphs["bindings"][0]["appended"]["cubin_sha256"] = "0" * 64
    elif change == "unsealed_binary":
        binaries.pop()
    elif change == "unsealed_controller":
        events.pop()
    elif change == "begin":
        events[0]["manifest_sha256"] = "wrong"
    elif change == "graph_receipt":
        events[:] = [e for e in events if e["event"] != "kv_graph_binding"]
    elif change == "binding_index":
        next(e for e in events if e["event"] == "kv_binding")["binding_index"] = 8
    elif change == "late_begin":
        events.insert(-1, copy.deepcopy(events[0]))
    else:
        events[-1]["graph_bindings"] = 20
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        check_graph_records(manifest, digest, graphs, binaries, events)


def prepared_fixture(tmp_path, monkeypatch):
    original = tmp_path / NAMESPACE
    graphs, roots, targets = {}, {}, {}
    for rank in range(4):
        store, graph_map, _ = serialized_fixture(original, count=7, rank=rank)
        graphs[str(rank)] = graph_map
        roots[str(rank)] = discover(store, original, graph_map)
        source = original / f"inductor_cache/co/combo{rank}.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# combo rank {rank}\n")
        kv = tmp_path / f"kv{rank}.py"
        kv.write_text(f"# kv rank {rank}\n")
        record = dict(
            source=str(source),
            source_sha256=sha(source),
            relative=str(source.relative_to(original)),
            kernel="combo",
            debug_source=str(source),
            debug_source_sha256=sha(source),
            selected={},
            cubin_sha256="0" * 64,
        )
        targets[str(rank)] = [
            dict(
                record,
                kv=dict(
                    record,
                    source=str(kv),
                    relative=f"kv_sources/rank-{rank}/kv.py",
                    source_sha256=sha(kv),
                    debug_source=str(kv),
                    debug_source_sha256=sha(kv),
                ),
                static_graph_uses=[
                    dict(graph=str(original / p), symbol="combo")
                    for p in list(graph_map)[:2]
                ],
            )
        ]
    base = dict(
        schema=SCHEMA,
        namespace=NAMESPACE,
        qualification_sha256=preparation.QUALIFICATION_SHA,
        contracts_sha256=preparation.CONTRACTS_SHA,
        original_namespace=str(original),
        original_files={
            str(p.relative_to(original)): sha(p)
            for p in original.rglob("*")
            if p.is_file()
        },
        targets=targets,
        expected_graphs=graphs,
        artifact_roots=roots,
        sources={str(p.resolve()): sha(p) for p in preparation.source_files()},
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    monkeypatch.setattr(preparation, "build_base", lambda: copy.deepcopy(base))
    original_check = subprocess.check_output
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kw: (
            "" if cmd == ["git", "status", "--porcelain"] else original_check(cmd, **kw)
        ),
    )
    return base


def test_full_private_preparation_and_frozen_manifest_roundtrip(tmp_path, monkeypatch):
    base = prepared_fixture(tmp_path, monkeypatch)
    output = tmp_path / "series"
    preparation.prepare(output)
    for mode, rank in preparation.ORDER:
        path = output / f"{mode}-rank{rank}/manifest.json"
        manifest, prepared = read_manifest(path)
        assert len(prepared["runs"]) == 8
        assert (manifest["mode"], manifest["rank"]) == (mode, rank)
        private = Path(manifest["private_namespace"])
        for relative, digest in base["original_files"].items():
            copied = private / relative
            assert sha(copied) == digest
            assert not copied.samefile(Path(base["original_namespace"]) / relative)
        assert len(manifest["private_sources"]) == 4
        for relative, digest in manifest["private_sources"].items():
            assert sha(private / relative) == digest
        names = {Path(p).name for p in manifest["sources"]}
        assert {
            "codecache.py",
            "static_triton_launcher.py",
            "glm53_loader_hooks.py",
        } <= names
    with pytest.raises(ValueError, match="new series"):
        preparation.prepare(output)


def test_inspection_does_not_copy_caches_or_create_launches(tmp_path, monkeypatch):
    prepared_fixture(tmp_path, monkeypatch)
    report = tmp_path / "inspection.json"
    preparation.prepare(report, inspect_only=True)
    data = json.loads(report.read_text())
    assert "private_namespace" not in data and "mode" not in data
    assert len(data["artifact_roots"]) == 4
    assert not list(tmp_path.glob("**/launch.json"))


def test_preparation_rejects_uncommitted_implementation(tmp_path, monkeypatch):
    prepared_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: " M kernel.py\n")
    output = tmp_path / "series"
    with pytest.raises(ValueError, match="commit the complete protocol"):
        preparation.prepare(output)
    assert not output.exists()


def test_shared_no_weights_runner_loads_real_store_and_never_forwards(
    tmp_path, monkeypatch
):
    import torch
    from torch._functorch._aot_autograd.aot_autograd_result import (
        BundledAOTAutogradResult,
        BundledCompiledForward,
    )
    from torch._inductor.output_code import maybe_realign_inputs
    from torch._inductor.standalone_compile import AOTCompiledArtifact
    from torch._inductor.utils import BoxedBool

    from benchmarks.kernels import check_glm53_geometry_loader as common
    from benchmarks.kernels import check_glm53_kv_loader as runner
    from benchmarks.kernels.audit_glm53_geometry_loader import check_aot_summary
    from vllm.compilation.caching import StandaloneCompiledArtifacts

    f = fixture(tmp_path, monkeypatch)
    original, private = Path(f.manifest["original_namespace"]), f.loader.private
    _, graphs, compiled_graphs = serialized_fixture(original, count=7)
    combo_source = (original / "inductor_cache/gg/graph0.py").read_text()
    store = StandaloneCompiledArtifacts()
    for index, (relative, graph) in enumerate(zip(graphs, compiled_graphs)):
        if index < 2:
            # Ensure even the target graph body fails immediately if invoked.
            graph.source_code = combo_source.replace(
                "    return combo.run",
                "    raise AssertionError('no forward')\n    return combo.run",
            )
        (original / relative).write_text(graph.source_code)
        dest = private / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, dest)
        graphs[relative] = sha(dest)
        forward = BundledCompiledForward.__new__(BundledCompiledForward)
        forward.result = graph
        entry = BundledAOTAutogradResult.__new__(BundledAOTAutogradResult)
        entry.compiled_fw, entry.compiled_bw = forward, None
        blob = pickle.dumps(pickle.dumps((entry, {}, {})))
        for n in range(46 // 7 + (index < 46 % 7)):
            store.insert(f"graph{index}layer{n}", "shape", blob)
    for folder in (original, private):
        model = folder / "rank_0_0/model"
        model.parent.mkdir()
        model.write_bytes(b"owned CPU fixture; read_store supplies its store")
    f.manifest.update(
        rank=0,
        mode="kv",
        sources={},
        cache_root=str(tmp_path / "cache"),
        original_files={
            str(p.relative_to(original)): sha(p)
            for p in original.rglob("*")
            if p.is_file()
        },
        expected_graphs={"0": graphs},
        artifact_roots={"0": discover(store, private, graphs)},
    )
    f.manifest["targets"]["0"][0]["static_graph_uses"] = [
        dict(graph=str(original / p), symbol="combo") for p in list(graphs)[:2]
    ]
    path = f.loader.manifest_path
    path.write_text(json.dumps(f.manifest))

    def make_loader(rank, manifest, manifest_path, mode, **kwargs):
        loader = KVLoader(rank, manifest, manifest_path, mode, **kwargs)
        loader.compile_replacement = f.loader.compile_replacement
        return loader

    def deserialize(data):
        graph = pickle.loads(data)[0].compiled_fw.result
        graph.after_deserialization(SimpleNamespace(unwrap=lambda g: {}))
        maybe_realign_inputs(
            BoxedBool(False), graph, graph.inputs_to_check, graph.mutated_input_idxs
        )
        return SimpleNamespace(graph=graph)

    monkeypatch.setattr(runner, "read_manifest", lambda _: (f.manifest, {}))
    monkeypatch.setattr(runner, "prior_receipts", lambda *a: [])
    monkeypatch.setattr(runner, "LOADER", make_loader)
    monkeypatch.setattr(common, "read_store", lambda _: (store, {}))
    monkeypatch.setattr(AOTCompiledArtifact, "deserialize", staticmethod(deserialize))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    original_check = subprocess.check_output
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kw: "" if cmd[0] == "nvidia-smi" else original_check(cmd, **kw),
    )
    with patch.dict(os.environ, dict(os.environ)):
        runner.run(path)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == "complete" and summary["loaded_artifacts"] == 7
    assert summary["submodules"] == 46
    assert summary["weight_tensors_loaded"] == summary["model_forward_calls"] == 0
    assert all(c["event"] == "driver_load" for calls in f.calls.values() for c in calls)
    graphs = json.loads((tmp_path / "run/graph-bindings.json").read_text())
    assert graphs["graphs"] == 7 and graphs["target_bindings"] == 2
    # This fixture supplies the target future directly, not real Triton bundles.
    # A complete runner result alone MUST NOT qualify that missing coverage.
    with pytest.raises(ValueError, match="static bundle coverage"):
        check_aot_summary(f.manifest, sha(path), summary)


@pytest.mark.parametrize("load_code", [0, 7])
def test_launcher_prescribes_resources_audits_failure_and_never_retries(
    tmp_path, monkeypatch, load_code
):
    from benchmarks.kernels import check_glm53_kv_loader as runner

    folder = tmp_path / "control-rank0"
    folder.mkdir()
    path = folder / "manifest.json"
    manifest = dict(rank=0, mode="control", expected_gpu_config="four fixed GPUs")
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "read_manifest", lambda _: (manifest, {}))
    monkeypatch.setattr(runner, "prior_receipts", lambda *args: [])
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda cmd, **kw: "four fixed GPUs" if "--query-gpu=" in cmd[1] else "",
    )
    commands = []

    def execute(command, *, stdout, stderr):
        commands.append(command)
        stdout.write("retained child log\n")
        return SimpleNamespace(returncode=load_code if len(commands) == 1 else 0)

    monkeypatch.setattr(subprocess, "run", execute)
    if load_code:
        with pytest.raises(ValueError, match="load/audit/release"):
            runner.launch(path)
    else:
        runner.launch(path)
    record = json.loads((folder / "launch.json").read_text())
    assert record["status"] == ("failed" if load_code else "complete")
    assert (
        record["gpu_config_before"] == record["gpu_config_after"] == "four fixed GPUs"
    )
    assert len(commands) == 2
    assert "MemoryMax=16G" in commands[0] and "MemoryMax=8G" in commands[1]
    assert all("MemorySwapMax=0" in c for c in commands)
    assert runner.MODULE in commands[0] and runner.AUDITOR_MODULE in commands[1]
    assert (
        record["load"]["exit_code"] == load_code and record["audit"]["exit_code"] == 0
    )
    with pytest.raises(ValueError, match="already attempted"):
        runner.launch(path)
    assert len(commands) == 2
