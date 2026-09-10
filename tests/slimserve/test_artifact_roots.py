# SPDX-License-Identifier: Apache-2.0
import copy
import pickle
from types import FunctionType, SimpleNamespace

import pytest
from torch._functorch._aot_autograd.aot_autograd_result import (
    BundledAOTAutogradResult,
    BundledCompiledForward,
)
from torch._inductor.codecache import PyCodeCache
from torch._inductor.output_code import CompiledFxGraph, maybe_realign_inputs
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch._inductor.utils import BoxedBool
from torch.utils._ordered_set import OrderedSet

from benchmarks.kernels.audit_glm53_geometry_loader import check_root_records
from benchmarks.kernels.glm53_artifact_roots import (
    ArtifactRootObserver,
    digest_bytes,
    discover,
    module_inventory,
)
from slimserve.rmsnorm_diagnostic import sha
from vllm.compilation.caching import StandaloneCompiledArtifacts


def serialized_fixture(namespace, count=2, rank=0):
    store = StandaloneCompiledArtifacts()
    graphs, results = {}, []
    for index in range(count):
        key = f"croot{rank}n{index}"
        relative = f"inductor_cache/{key[1:3]}/{key}.py"
        source = (
            "class Runner:\n"
            "    def __init__(self): self.partitions = []\n"
            "    def call(self): raise AssertionError('must never execute')\n"
            "runner = Runner()\ncall = runner.call\n"
            if index % 2
            else "def call(): raise AssertionError('must never execute')\n"
        )
        path = namespace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        graph = CompiledFxGraph.__new__(CompiledFxGraph)
        graph.cache_key = key
        graph.source_code = source
        graph.cache_linemap = []
        graph.current_callable = None
        graph.compiled_fn_runner = None
        graph.inputs_to_check = [0, 4]
        graph.mutated_input_idxs = OrderedSet([4] if index % 2 else [])
        graph._defers_input_alignment = True
        graph._wrap_compiled_regions = False
        forward = BundledCompiledForward.__new__(BundledCompiledForward)
        forward.result = graph
        entry = BundledAOTAutogradResult.__new__(BundledAOTAutogradResult)
        entry.compiled_fw = forward
        entry.compiled_bw = None
        payload = pickle.dumps((entry, {}, {}))
        # Exercise the real store's content-addressed insertion and deduplication.
        for n in range(46 // count + (index < 46 % count)):
            store.insert(f"sub{index}n{n}", "shape", pickle.dumps(payload))
        graphs[relative] = sha(path)
        results.append(graph)
    return store, graphs, results


def test_serialized_sources_match_real_store_and_no_postcompile(tmp_path, monkeypatch):
    store, graphs, _ = serialized_fixture(tmp_path)
    monkeypatch.setattr(
        CompiledFxGraph, "after_deserialization", lambda *a: pytest.fail("postcompile")
    )
    rows = discover(store, tmp_path, graphs)
    assert len(rows) == 2 and sum(len(r["submodules"]) for r in rows) == 46
    assert {r["graph"]: r["graph_sha256"] for r in rows} == graphs


@pytest.mark.parametrize(
    "change",
    [
        "digest",
        "graph_source",
        "missing",
        "extra",
        "loaded",
        "backward",
        "serialized_source",
        "entry_map",
    ],
)
def test_serialized_roots_reject_drift(tmp_path, change):
    store, graphs, _ = serialized_fixture(tmp_path)
    if change == "digest":
        first = next(iter(store.submodule_bytes_store))
        store.submodule_bytes_store[first] += b"changed"
    elif change == "graph_source":
        (tmp_path / next(iter(graphs))).write_text("changed")
    elif change == "missing":
        graphs.pop(next(iter(graphs)))
    elif change == "extra":
        graphs["inductor_cache/xx/extra.py"] = "0" * 64
    elif change == "loaded":
        store.loaded_submodule_store["anything"] = object()
    elif change == "entry_map":
        store.submodule_bytes["unknown"] = "unbound"
    else:
        old = next(iter(store.submodule_bytes_store))
        bundle = pickle.loads(pickle.loads(store.submodule_bytes_store.pop(old)))
        if change == "backward":
            bundle[0].compiled_bw = True
        else:
            bundle[0].compiled_fw.result.source_code += "# drift\n"
        blob = pickle.dumps(pickle.dumps(bundle))
        new = digest_bytes(blob)
        store.submodule_bytes_store[new] = blob
        store.submodule_bytes = {
            k: new if v == old else v for k, v in store.submodule_bytes.items()
        }
    with pytest.raises(ValueError):
        discover(store, tmp_path, graphs)


def live_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor_cache"))
    monkeypatch.setattr(PyCodeCache, "modules_no_attr", {})
    monkeypatch.setattr(PyCodeCache, "linemaps", {})
    monkeypatch.setattr(PyCodeCache, "modules", [])
    store, graphs, _ = serialized_fixture(tmp_path, count=4)
    expected = discover(store, tmp_path, graphs)
    events = []
    observer = ArtifactRootObserver(expected, tmp_path, emit=events.append)

    def deserialize(data):
        graph = pickle.loads(data)[0].compiled_fw.result
        graph.after_deserialization(SimpleNamespace(unwrap=lambda g: {}))
        maybe_realign_inputs(
            BoxedBool(False), graph, graph.inputs_to_check, graph.mutated_input_idxs
        )
        return SimpleNamespace(graph=graph)

    # Real vLLM concurrent load_all, CompiledFxGraph.after_deserialization,
    # write_to_disk and PyCodeCache imports. Only the outer AOT wrapper is
    # reduced; nothing invokes call and no GPU APIs are mocked or used.
    monkeypatch.setattr(AOTCompiledArtifact, "deserialize", staticmethod(deserialize))
    return (
        store,
        graphs,
        expected,
        events,
        observer,
        AOTCompiledArtifact,
        store.load_all,
    )


def live_receipts(tmp_path, monkeypatch):
    store, graphs, expected, events, observer, artifact, load_all = live_fixture(
        tmp_path, monkeypatch
    )
    helper = tmp_path / "inductor_cache/hh/helper.py"
    helper.parent.mkdir()
    helper.write_text(
        "def call(): raise AssertionError('benchmark helper must not execute')\n"
    )
    PyCodeCache.load_by_key_path("helper", str(helper))
    before = (artifact.__dict__["deserialize"], CompiledFxGraph.after_deserialization)
    with observer.intercept(artifact_class=artifact):
        load_all()
        roots = observer.roots(store, PyCodeCache.modules)
        assert len(roots) == 4 and len(PyCodeCache.modules) == 5
        assert all(m.__file__ != str(helper) for m in roots)
    assert (
        artifact.__dict__["deserialize"],
        CompiledFxGraph.after_deserialization,
    ) == before
    manifest = dict(
        rank=0,
        private_namespace=str(tmp_path),
        expected_graphs={"0": graphs},
        artifact_roots={"0": expected},
        original_files={**graphs, str(helper.relative_to(tmp_path)): sha(helper)},
    )
    bindings = dict(
        bindings=[
            dict(graph=row["graph"], module_index=i + 1)
            for i, row in enumerate(expected)
        ]
    )
    return manifest, bindings, module_inventory(PyCodeCache.modules), events


def test_real_after_deserialize_exports_and_helper_separation(tmp_path, monkeypatch):
    result = check_root_records(*live_receipts(tmp_path, monkeypatch))
    assert result == dict(
        artifact_roots=4, imported_modules=5, non_root_imported_modules=1
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_root",
        "wrong_payload",
        "wrong_module",
        "wrong_source",
        "late_root",
        "wrong_order",
        "duplicate",
        "helper_drift",
        "binding_index",
        "call_line",
    ],
)
def test_offline_artifact_root_join_rejects_drift(tmp_path, monkeypatch, change):
    manifest, bindings, modules, events = copy.deepcopy(
        live_receipts(tmp_path, monkeypatch)
    )
    if change == "missing_root":
        events.pop()
    elif change == "wrong_payload":
        events[0]["payload_sha256"] = "wrong"
    elif change == "wrong_module":
        next(r for r in events if r["event"] == "artifact_root_bound")[
            "module_index"
        ] = 1
    elif change == "wrong_source":
        events[0]["graph_sha256"] = "wrong"
    elif change == "late_root":
        events.append(copy.deepcopy(events[0]))
    elif change == "wrong_order":
        events.reverse()
    elif change == "duplicate":
        events += events
    elif change == "helper_drift":
        modules[0]["source_sha256"] = "wrong"
    elif change == "binding_index":
        bindings["bindings"][0]["module_index"] = 99
    else:
        modules[0]["call_line"] += 1
    with pytest.raises((ValueError, KeyError)):
        check_root_records(manifest, bindings, modules, events)


@pytest.mark.parametrize(
    "change",
    [
        "returned_artifact",
        "call",
        "runner",
        "module_missing",
        "graph_source",
        "unobserved",
        "repeated",
        "foreign_hook",
    ],
)
def test_live_artifact_observer_rejects_drift(tmp_path, monkeypatch, change):
    store, _, _, _, observer, artifact, load_all = live_fixture(tmp_path, monkeypatch)
    with observer.intercept(artifact_class=artifact):
        if change == "graph_source":
            original = CompiledFxGraph.after_deserialization

            # Change the live result before the real observer sees it.
            def changed(graph, *args):
                graph.source_code += "# altered\n"
                return original(graph, *args)

            CompiledFxGraph.after_deserialization = changed
            try:
                with pytest.raises(ValueError, match="source/key/path"):
                    load_all()
            finally:
                CompiledFxGraph.after_deserialization = original
            return
        load_all()
        root = next(iter(store.loaded_submodule_store))
        graph, module, _ = observer.bindings[root]
        if change == "returned_artifact":
            store.loaded_submodule_store[root] = object()
        elif change == "call":
            graph.current_callable = lambda: None
        elif change == "runner":
            graph.compiled_fn_runner = SimpleNamespace(partitions=[])
        elif change == "module_missing":
            PyCodeCache.modules.remove(module)
        elif change == "unobserved":
            with pytest.raises(ValueError, match="outside observed"):
                graph.after_deserialization(SimpleNamespace(unwrap=lambda g: {}))
            return
        elif change == "repeated":
            with pytest.raises(ValueError, match="repeated"):
                artifact.deserialize(
                    pickle.loads(next(iter(store.submodule_bytes_store.values())))
                )
            return
        elif change == "foreign_hook":
            with (
                pytest.raises(ValueError, match="conflicting"),
                observer.intercept(artifact_class=artifact),
            ):
                pass
            return
        with pytest.raises(ValueError):
            observer.roots(store, PyCodeCache.modules)


@pytest.mark.parametrize(
    "change",
    ["model", "indices", "mutations", "globals", "code", "direct", "nested", "plan"],
)
def test_only_exact_writeback_closure_is_accepted(tmp_path, monkeypatch, change):
    store, _, _, _, observer, artifact, load_all = live_fixture(tmp_path, monkeypatch)
    with observer.intercept(artifact_class=artifact):
        load_all()
        graph, module, original = next(
            value
            for value in observer.bindings.values()
            if value[0].current_callable is not value[2]
        )
        wrapper = graph.current_callable
        cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))
        if change == "model":
            cells["model"].cell_contents = lambda: None
        elif change == "indices":
            cells["inputs_to_check"].cell_contents = [0]
        elif change == "mutations":
            cells["mutated_input_idxs"].cell_contents = OrderedSet([4])
        elif change == "globals":
            graph.current_callable = FunctionType(
                wrapper.__code__, {}, closure=wrapper.__closure__
            )
        elif change == "code":
            wrapper.__code__ = wrapper.__code__.replace(co_name="changed")
        elif change == "direct":
            graph.current_callable = original
        elif change == "nested":
            maybe_realign_inputs(
                BoxedBool(False), graph, graph.inputs_to_check, graph.mutated_input_idxs
            )
        else:
            graph.inputs_to_check.append(99)
        with pytest.raises(ValueError):
            observer.roots(store, PyCodeCache.modules)


@pytest.mark.parametrize(
    "field",
    [
        "wrapper_source_sha256",
        "callable_line",
        "closure_model_is_original",
        "exact_writeback_globals",
        "closure_inputs_to_check",
        "runner_is_original",
    ],
)
def test_offline_writeback_receipts_reject_changed_provenance(
    tmp_path, monkeypatch, field
):
    manifest, bindings, modules, events = live_receipts(tmp_path, monkeypatch)
    state = next(
        e["state"]
        for e in events
        if e["event"] == "artifact_post_compile_state" and not e["state"]["direct"]
    )
    state[field] = None
    with pytest.raises(ValueError):
        check_root_records(manifest, bindings, modules, events)
