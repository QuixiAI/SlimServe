# SPDX-License-Identifier: Apache-2.0
"""Real Torch future/code-cache/static-launcher APIs; only the CUDA driver is fake."""

import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch._inductor.codecache import PyCodeCache, StaticAutotunerFuture
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from benchmarks.kernels import glm53_kv_loader as loader_module
from benchmarks.kernels.audit_glm53_geometry_graphs import compare_unrelated
from benchmarks.kernels.audit_glm53_kv_graphs import inventory
from benchmarks.kernels.glm53_kv_loader import SCHEMA, KVLoader
from slimserve.rmsnorm_diagnostic import sha
from tests.slimserve.test_binary_observer import make_compiled


def attention_compiled(root, name):
    config = dict(XBLOCK=2, R0_BLOCK=1024, num_warps=8, num_stages=1)
    if name == "kv":
        config = dict(XBLOCK=2, num_warps=1, num_stages=1)
    compiled, path, calls = make_compiled(root, name, config)
    if name == "combo":
        args = [f"in_ptr{i}" for i in range(5)] + ["out_ptr1", "out_ptr3", "out_ptr6"]
        args += [f"xnumel_{i}" for i in range(3)]
        compiled.inductor_meta = dict(
            grid_type="SequentialComboKernelGrid",
            combo_grid_meta=dict(
                num_kernels=3,
                min_blocks=None,
                autotune_grouping=True,
                default_config=None,
                no_x_dim_0=None,
                xnumel_0=None,
                no_x_dim_1=False,
                xnumel_1=None,
                no_x_dim_2=None,
                xnumel_2=None,
            ),
        )
    else:
        args = ["in_ptr0", "in_ptr1", "out_ptr1", "xnumel", "r0_numel"]
    compiled.kernel.arg_names = args
    compiled.kernel.arg_tys = "O" * len(args)
    compiled.compile_meta["signature"] = {
        arg: "i32" if "numel" in arg else "*bf16" for arg in args
    }
    return compiled, path, calls, config


def fixture(tmp_path, monkeypatch, mode="kv"):
    original, private = tmp_path / "original", tmp_path / "private"
    cache = private / "inductor_cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache / "triton/0"))
    monkeypatch.setattr(PyCodeCache, "modules_no_attr", {})
    monkeypatch.setattr(PyCodeCache, "linemaps", {})
    monkeypatch.setattr(PyCodeCache, "modules", [])
    records, results, driver_calls = {}, {}, {}
    for name in ("combo", "kv"):
        relative = f"inductor_cache/aa/{name}.py"
        for folder in (original, private):
            source = folder / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# {name}\n")
        result, path, calls, config = attention_compiled(cache / "triton/0", name)
        results[name], driver_calls[name] = result, calls
        records[name] = dict(
            relative=relative,
            source=str(original / relative),
            kernel=name,
            source_sha256=sha(source),
            debug_source=str(original / relative),
            debug_source_sha256=sha(source),
            cubin_sha256=sha(path),
            selected=dict(hash=path.parent.name, config=config),
        )
    target = dict(records["combo"], kv=records["kv"], static_graph_uses=[])
    tuner = CachingAutotuner.__new__(CachingAutotuner)
    tuner.filename = target["source"]
    tuner.compile_results, tuner.launchers = [results["combo"]], []
    tuner.configs, tuner.save_cache_hook = (
        None,
        lambda *_: pytest.fail("old cache write"),
    )
    # Deliberately NOT the same callable as launchers[0]. Must be cleared and bypassed.
    tuner._cached_launcher = lambda *_: pytest.fail("cached fast path bypass")
    rechecks = []

    def precompile(**kwargs):
        tuner.launchers = [tuner.compile_results[0].make_launcher()]

    tuner.precompile = precompile
    tuner.recheck_autotune_cache = lambda **kw: rechecks.append(kw)
    future = StaticAutotunerFuture(tuner)
    future.reload_kernel_from_src = lambda: pytest.fail("source reload")
    registry = ModuleType("glm53_kv_fixture")
    registry.future = future
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    source_text = (
        "from glm53_kv_fixture import future\n"
        "combo = future.result()\n"
        "def call(args, stream):\n"
        "    return combo.run(*args, stream=stream)\n"
    )
    graphs, files = {}, {r["relative"]: r["source_sha256"] for r in records.values()}
    for index in range(2):
        relative = f"inductor_cache/gg/graph{index}.py"
        for folder in (original, private):
            graph = folder / relative
            graph.parent.mkdir(parents=True, exist_ok=True)
            graph.write_text(source_text)
        graphs[relative] = files[relative] = sha(graph)
        target["static_graph_uses"].append(
            dict(graph=str(original / relative), symbol="combo")
        )
    manifest = dict(
        schema=SCHEMA,
        original_namespace=str(original),
        private_namespace=str(private),
        targets={"0": [target]},
        expected_graphs={"0": graphs},
        original_files=files,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    loader = KVLoader(0, manifest, manifest_path, mode)
    compiles = []

    def replacement(record):
        compiles.append(record)
        return results["kv"]

    loader.compile_replacement = replacement
    return SimpleNamespace(
        loader=loader,
        manifest=manifest,
        tuner=tuner,
        results=results,
        calls=driver_calls,
        future=future,
        rechecks=rechecks,
        compiles=compiles,
    )


def load_graphs(f):
    return [
        PyCodeCache.load_by_key_path(relative, str(f.loader.private / relative))
        for relative in f.manifest["expected_graphs"]["0"]
    ]


@pytest.mark.parametrize("mode", ["control", "kv"])
def test_real_loader_dispatch_and_static_driver_lifecycle(tmp_path, monkeypatch, mode):
    f = fixture(tmp_path, monkeypatch, mode)
    before = (
        StaticAutotunerFuture.result,
        PyCodeCache.__dict__["load_by_key_path"],
        StaticallyLaunchedCudaKernel.load_kernel,
    )
    # A real CachingAutotuner instance inherits the dispatcher until binding.
    assert f.tuner.run.__func__ is CachingAutotuner.run
    with f.loader.intercept():
        modules = load_graphs(f)
        assert len(f.rechecks) == 1
        assert len(f.compiles) == int(mode == "kv")
        original_lists = f.tuner.compile_results, f.tuner.launchers
        report = f.loader.controller.seal(modules)
        assert report["graphs"] == report["target_bindings"] == 2
        assert not f.loader.observer.sealed
        for module in modules:
            args = (*[object() for _ in range(8)], 65, 65, 65)
            module.call(args, 123)
            call = f.calls["combo"][-1]
            assert call["event"] == "driver_launch"
            assert call["args"][-2] == args
            assert call["args"][-1] == 123
            if mode == "kv":
                extra = f.calls["kv"][-1]
                assert extra["event"] == "driver_launch"
                assert extra["args"][-2] == (args[0], args[1], args[5], 65, 512)
                assert extra["args"][-1] == 123
        assert f.tuner.compile_results is original_lists[0]
        assert f.tuner.launchers is original_lists[1]
        assert f.future.result() is f.tuner and len(f.rechecks) == 1
        assert StaticAutotunerFuture(f.tuner).result() is f.tuner
        assert len(f.rechecks) == 1
        assert load_graphs(f) == modules
        assert f.loader.controller.verify_graphs(modules) == report
        unrelated, _, _ = make_compiled(f.loader.cache / "triton/0", "later")
        unrelated.make_launcher()  # Target seal does not seal unrelated compilation.
    assert before == (
        StaticAutotunerFuture.result,
        PyCodeCache.__dict__["load_by_key_path"],
        StaticallyLaunchedCudaKernel.load_kernel,
    )
    if mode == "control":
        assert f.calls["kv"] == []


def test_concurrent_graph_imports_resolve_shared_target_once(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    with f.loader.intercept(), ThreadPoolExecutor(max_workers=4) as pool:
        modules = sum(list(pool.map(lambda _: load_graphs(f), range(4))), [])
        assert len(f.rechecks) == len(f.compiles) == 1
        assert all(m.combo is f.tuner for m in modules)
        f.loader.controller.seal(modules)


@pytest.mark.parametrize(
    "change",
    [
        "run",
        "cached",
        "combo",
        "kv",
        "handle",
        "config",
        "source",
        "foreign_source",
        "graph",
        "missing",
        "call",
        "original_result",
        "missing_launcher",
        "save_hook",
    ],
)
def test_independent_inventory_rejects_drift(tmp_path, monkeypatch, change):
    f = fixture(tmp_path, monkeypatch)
    with f.loader.intercept():
        modules = load_graphs(f)
        f.loader.controller.seal(modules)
        tuner = f.tuner
        if change == "run":
            tuner.run = tuner.launchers[0]
        elif change == "cached":
            tuner._cached_launcher = tuner.launchers[0]
        elif change == "combo":
            tuner.run.__self__.combo = lambda *_: None
        elif change == "kv":
            tuner.run.__self__.split_kv = tuner.launchers[0]
        elif change == "handle":
            f.results["kv"].kernel.function = -1
        elif change == "config":
            f.results["kv"].config.num_warps = 4
        elif change == "source":
            Path(tuner.filename).write_text("# drift\n")
        elif change == "foreign_source":
            tuner.filename = str(tmp_path / "combo.py")
        elif change == "graph":
            Path(modules[0].__file__).write_text("# drift\n")
        elif change == "missing":
            modules.pop()
        elif change == "call":
            modules[0].call = lambda *_: None
        elif change == "original_result":
            tuner.compile_results = [f.results["kv"]]
        elif change == "missing_launcher":
            tuner.launchers = []
        else:
            tuner.save_cache_hook = lambda *_: None
        with pytest.raises(ValueError):
            inventory(modules, f.manifest, 0, "kv", f.loader.observer)
        with pytest.raises(ValueError):
            f.loader.controller.verify_graphs(modules)


def test_inventory_does_not_trust_controller_coverage(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    with f.loader.intercept():
        modules = load_graphs(f)
        f.loader.controller.graph_bindings.clear()
        assert (
            inventory(modules, f.manifest, 0, "kv", f.loader.observer)[
                "target_bindings"
            ]
            == 2
        )
        with pytest.raises(ValueError, match="coverage"):
            f.loader.controller.seal(modules)


def test_finished_module_binding_without_static_future(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    with f.loader.intercept():
        f.tuner.precompile()
        module = ModuleType("finished")
        module.call = lambda: None
        module.combo = f.tuner
        f.loader.controller.bind_graph(module, f.loader.compile_replacement)
        extra = f.tuner.run.__self__.split_kv
        assert extra.__globals__["runner"].__self__ is f.results["kv"].kernel
        f.loader.observer.verify_launcher(f.results["kv"], extra)
        assert not f.rechecks and len(f.compiles) == 1


@pytest.mark.parametrize("which", ["future", "graph"])
def test_new_target_or_graph_after_seal_fails(tmp_path, monkeypatch, which):
    f = fixture(tmp_path, monkeypatch)
    with f.loader.intercept():
        f.loader.controller.seal(load_graphs(f))
        if which == "future":
            tuner = CachingAutotuner.__new__(CachingAutotuner)
            vars(tuner).update(vars(f.tuner))
            future = StaticAutotunerFuture(tuner)
            with pytest.raises(ValueError, match="after seal"):
                future.result()
        else:
            module = ModuleType("late")
            module.call = lambda: None
            module.combo = f.tuner
            with pytest.raises(ValueError, match="after seal"):
                f.loader.controller.bind_graph(module, f.loader.compile_replacement)
        assert len(f.rechecks) == 1


@pytest.mark.parametrize("which", ["future", "cache"])
def test_foreign_hook_restoration(tmp_path, monkeypatch, which):
    f = fixture(tmp_path, monkeypatch)
    future_hook = StaticAutotunerFuture.result
    cache_hook = PyCodeCache.__dict__["load_by_key_path"]
    kernel_hook = StaticallyLaunchedCudaKernel.load_kernel
    foreign = lambda *_: None
    try:
        with (
            pytest.raises(ValueError, match="foreign loader hook"),
            f.loader.intercept(),
        ):
            if which == "future":
                StaticAutotunerFuture.result = foreign
            else:
                PyCodeCache.load_by_key_path = classmethod(foreign)
        assert StaticallyLaunchedCudaKernel.load_kernel is kernel_hook
        if which == "future":
            assert StaticAutotunerFuture.result is foreign
            assert PyCodeCache.__dict__["load_by_key_path"] is cache_hook
        else:
            assert StaticAutotunerFuture.result is future_hook
            assert PyCodeCache.__dict__["load_by_key_path"].__func__ is foreign
    finally:
        StaticAutotunerFuture.result = future_hook
        PyCodeCache.load_by_key_path = cache_hook


def test_compiler_uses_qualified_source_provenance_and_config(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    calls, configs = [], []

    def load(*args, **kw):
        calls.append((args, kw))
        return SimpleNamespace(_precompile_config=lambda config: configs.append(config))

    monkeypatch.setattr(loader_module, "load_preserving_provenance", load)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    record = f.manifest["targets"]["0"][0]["kv"]
    for _ in range(2):
        KVLoader.compile_replacement(f.loader, record)
    assert len(calls) == 1 and len(configs) == 2
    args, kw = calls[0]
    assert args[:3] == (
        f.loader.private / record["relative"],
        Path(record["source"]),
        "kv",
    )
    assert kw == dict(debug_source=Path(record["debug_source"]))
    assert configs[0].kwargs == dict(XBLOCK=2)
    assert configs[0].num_warps == 1 and configs[0].num_stages == 1


def test_non_target_comparison_retains_full_source_config_binary_gate():
    row = dict(
        target=False,
        source="side.py",
        graph="graph.py",
        symbol="side",
        cubin_sha256="a",
    )
    original = dict(graphs=2, target_bindings=2, bindings=[row])
    candidate = copy.deepcopy(original)
    compare_unrelated(original, candidate)
    candidate["bindings"][0]["cubin_sha256"] = "b"
    with pytest.raises(ValueError, match="non-target"):
        compare_unrelated(original, candidate)


@pytest.mark.parametrize("which", ["private_source", "extra_image", "foreign_run"])
def test_invalid_initial_binding_stops_before_appended_driver_load(
    tmp_path, monkeypatch, which
):
    f = fixture(tmp_path, monkeypatch)
    if which == "private_source":
        (f.loader.private / "inductor_cache/aa/kv.py").write_text("# changed\n")
    elif which == "extra_image":
        f.results["kv"].kernel.cubin_raw = b"changed"
    else:
        f.tuner.run = lambda *_: None
    with f.loader.intercept(), pytest.raises(ValueError):
        load_graphs(f)
    assert not f.calls["kv"]


def test_actual_non_target_inventory_is_unchanged_and_detects_drift(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch)
    with f.loader.intercept():
        modules = load_graphs(f)
        result, _, _ = make_compiled(f.loader.cache / "triton/0", "side")
        relative = "inductor_cache/ss/side.py"
        source = f.loader.private / relative
        source.parent.mkdir()
        source.write_text("# side\n")
        f.manifest["original_files"][relative] = sha(source)
        tuner = SimpleNamespace(
            filename=str(source),
            compile_results=[result],
            launchers=[result.make_launcher()],
        )
        modules[0].side = tuner
        original = inventory(modules, f.manifest, 0, "kv", f.loader.observer)
        (row,) = [r for r in original["bindings"] if not r["target"]]
        assert row["source"] == relative and not row["referenced"]
        assert row["cubin_sha256"] == f.loader.observer.digest(result)
        assert "appended" not in row
        result.config.num_stages = 2
        changed = inventory(modules, f.manifest, 0, "kv", f.loader.observer)
        with pytest.raises(ValueError, match="non-target"):
            compare_unrelated(original, changed)


@pytest.mark.parametrize(
    "marker", ["_glm53_geometry_loader", "_glm53_rmsnorm_diagnostic"]
)
def test_conflicting_diagnostic_hook_is_not_replaced(tmp_path, monkeypatch, marker):
    f = fixture(tmp_path, monkeypatch)
    foreign = lambda *_: None
    setattr(foreign, marker, True)
    monkeypatch.setattr(StaticAutotunerFuture, "result", foreign)
    with pytest.raises(ValueError, match="conflicting"), f.loader.intercept():
        pass
    assert StaticAutotunerFuture.result is foreign


def test_later_non_target_future_is_not_transformed(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    with f.loader.intercept():
        f.loader.controller.seal(load_graphs(f))
        result, _, _ = make_compiled(f.loader.cache / "triton/0", "side")
        source = f.loader.private / "inductor_cache/aa/side.py"
        source.write_text("# side\n")
        tuner = SimpleNamespace(
            filename=str(source),
            compile_results=[result],
            launchers=[],
            save_cache_hook=None,
        )
        rechecks = []
        tuner.recheck_autotune_cache = lambda **kw: rechecks.append(kw)
        tuner.precompile = lambda **kw: setattr(
            tuner, "launchers", [result.make_launcher()]
        )
        future = StaticAutotunerFuture(tuner)
        future.reload_kernel_from_src = lambda: pytest.fail("source reload")
        assert future.result() is tuner and len(rechecks) == 1
        assert "run" not in vars(tuner) and tuner.compile_results == [result]
        assert len(f.loader.controller.owners) == 1
