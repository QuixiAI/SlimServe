# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import triton
from torch._inductor.codecache import PyCodeCache, StaticAutotunerFuture
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

from benchmarks.kernels import glm53_geometry_loader as loader_module
from benchmarks.kernels.audit_glm53_geometry_graphs import (
    compare_unrelated,
    inventory,
    run_symbols,
)
from benchmarks.kernels.glm53_geometry_loader import GeometryLoader
from benchmarks.kernels.glm53_rmsnorm_geometry import CONTROL, GEOMETRY, SCHEMA
from benchmarks.kernels.prepare_glm53_geometry_loader import (
    current_sources,
    qualified_targets,
)
from slimserve.rmsnorm_diagnostic import sha
from tests.slimserve.test_binary_observer import make_compiled


def fixture(tmp_path, monkeypatch, mode="geometry"):
    original, private = tmp_path / "original", tmp_path / "private"
    cache = private / "inductor_cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cache))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setattr(PyCodeCache, "modules_no_attr", {})
    monkeypatch.setattr(PyCodeCache, "linemaps", {})
    monkeypatch.setattr(PyCodeCache, "modules", [])
    targets, tuners, pairs, rechecks, files = [], [], {}, [], {}
    graph_relative = "inductor_cache/gg/graph.py"
    for index in range(4):
        name = f"norm{index}"
        relative = f"inductor_cache/aa/{name}.py"
        for folder in (original, private):
            source = folder / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# {name}\n")
        files[relative] = sha(source)
        configs, pair = {}, {}
        for arm, config in (("control", CONTROL), ("geometry", GEOMETRY)):
            result, path, _ = make_compiled(cache / "triton/0", name, config)
            configs[arm] = dict(
                config, triton_cache_hash=path.parent.name, cubin_sha256=sha(path)
            )
            pair[arm] = result
        pairs[relative] = pair
        if index < 3:
            targets.append(
                dict(
                    relative=relative,
                    kernel=name,
                    source_sha256=sha(source),
                    configs=configs,
                    debug_source=str(original / relative),
                    debug_source_sha256=sha(source),
                    static_graph_uses=[
                        dict(graph=str(original / graph_relative), symbol=name)
                    ],
                )
            )
        tuner = SimpleNamespace(
            filename=str(original / relative),
            compile_results=[pair["control"]],
            launchers=[],
            configs=None,
            save_cache_hook=None,
            _cached_launcher=None,
        )

        def precompile(*, tuner=tuner, **kwargs):
            tuner.launchers = [tuner.compile_results[0].make_launcher()]

        tuner.precompile = precompile
        tuner.recheck_autotune_cache = lambda *, index=index, **kwargs: rechecks.append(
            index
        )
        future = StaticAutotunerFuture(tuner)
        future.reload_kernel_from_src = lambda: pytest.fail(
            "source reload not expected"
        )
        tuners.append(future)
    registry = ModuleType("glm53_geometry_loader_test_registry")
    registry.futures = tuners
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    text = "from glm53_geometry_loader_test_registry import futures\n"
    text += "\n".join(f"norm{i} = futures[{i}].result()" for i in range(4))
    text += "\n# same future is resolved again, but must not redo upstream selection\n"
    text += "futures[0].result()\n"
    text += "def call():\n" + "\n".join(f"    norm{i}.run()" for i in range(4)) + "\n"
    for folder in (original, private):
        graph = folder / graph_relative
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_text(text)
    files[graph_relative] = sha(graph)
    manifest = dict(
        schema=SCHEMA,
        original_namespace=str(original),
        private_namespace=str(private),
        targets={"0": targets},
        original_files=files,
        expected_graphs={"0": {graph_relative: sha(graph)}},
        receipts=str(tmp_path / "receipts"),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loader = GeometryLoader(0, manifest, path, mode)
    loader.compile_replacement = lambda target, saved: pairs[target["relative"]][mode]
    return loader, manifest, graph, rechecks, pairs


@pytest.mark.parametrize("mode", ["control", "geometry"])
def test_installed_codecache_future_and_static_driver_lifecycle(
    tmp_path, monkeypatch, mode
):
    loader, manifest, graph, rechecks, pairs = fixture(tmp_path, monkeypatch, mode)
    before = (
        StaticAutotunerFuture.result,
        PyCodeCache.__dict__["load_by_key_path"],
        StaticallyLaunchedCudaKernel.load_kernel,
    )
    with loader.intercept():
        module = PyCodeCache.load_by_key_path("geometry-fixture", str(graph))
        assert rechecks == [0, 1, 2, 3]
        report = inventory(PyCodeCache.modules, manifest, 0, mode, loader.observer)
        assert report["graphs"] == 1 and report["target_bindings"] == 3
        loader.controller.seal(PyCodeCache.modules)
        assert not loader.observer.sealed
        loader.controller.verify_graphs(PyCodeCache.modules)
        assert PyCodeCache.load_by_key_path("geometry-fixture", str(graph)) is module
        assert rechecks == [0, 1, 2, 3]
        # Target seal must not forbid a legitimate late, non-target binary load.
        unrelated, _, _ = make_compiled(loader.cache / "triton/0", "later")
        unrelated.make_launcher()
        loader.observer.seal()
    assert (
        StaticAutotunerFuture.result,
        PyCodeCache.__dict__["load_by_key_path"],
        StaticallyLaunchedCudaKernel.load_kernel,
    ) == before
    loader.close()


@pytest.mark.parametrize(
    "change",
    [
        "missing_graph",
        "missing_target",
        "wrong_binary",
        "wrong_config",
        "wrong_call_globals",
        "changed_graph",
        "changed_source",
        "foreign_source",
        "missing_launcher",
    ],
)
def test_independent_inventory_rejects_changed_live_bindings(
    tmp_path, monkeypatch, change
):
    loader, manifest, graph, _, pairs = fixture(tmp_path, monkeypatch)
    with loader.intercept():
        module = PyCodeCache.load_by_key_path("geometry-fixture", str(graph))
        modules = PyCodeCache.modules
        if change == "missing_graph":
            modules = []
        elif change == "missing_target":
            del module.norm0
        elif change == "wrong_binary":
            module.norm0.compile_results[0].kernel.function = -1
        elif change == "wrong_config":
            module.norm0.launchers[0].config.num_stages = 2
        elif change == "wrong_call_globals":
            module.call = lambda: None
        elif change == "changed_graph":
            graph.write_text(graph.read_text() + "# changed\n")
        elif change == "changed_source":
            Path(module.norm0.filename).write_text("# changed source\n")
        elif change == "foreign_source":
            module.norm0.filename = str(tmp_path / "norm0.py")
        else:
            module.norm0.launchers = []
        with pytest.raises(ValueError):
            inventory(modules, manifest, 0, "geometry", loader.observer)
    loader.close()


def test_auditor_does_not_depend_on_controller_coverage_receipts(tmp_path, monkeypatch):
    loader, manifest, graph, _, _ = fixture(tmp_path, monkeypatch)
    with loader.intercept():
        PyCodeCache.load_by_key_path("geometry-fixture", str(graph))
        for child in loader.controller.controllers.values():
            child.graph_bindings.clear()
        report = inventory(
            PyCodeCache.modules, manifest, 0, "geometry", loader.observer
        )
        assert report["target_bindings"] == 3
        with pytest.raises(ValueError):
            loader.controller.seal(PyCodeCache.modules)
    loader.close()


@pytest.mark.parametrize("which", ["future", "cache"])
def test_foreign_hook_is_preserved_and_other_hooks_restore(
    tmp_path, monkeypatch, which
):
    loader, _, _, _, _ = fixture(tmp_path, monkeypatch)
    original_result = StaticAutotunerFuture.result
    original_load = PyCodeCache.__dict__["load_by_key_path"]
    original_kernel = StaticallyLaunchedCudaKernel.load_kernel
    foreign = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match="foreign loader hook"), loader.intercept():
            if which == "future":
                StaticAutotunerFuture.result = foreign
            else:
                PyCodeCache.load_by_key_path = classmethod(foreign)
        assert StaticallyLaunchedCudaKernel.load_kernel is original_kernel
        if which == "future":
            assert StaticAutotunerFuture.result is foreign
            assert PyCodeCache.__dict__["load_by_key_path"] is original_load
        else:
            assert PyCodeCache.__dict__["load_by_key_path"].__func__ is foreign
            assert StaticAutotunerFuture.result is original_result
    finally:
        StaticAutotunerFuture.result = original_result
        PyCodeCache.load_by_key_path = original_load
        loader.close()


def test_nested_hook_fails_without_changing_outer_hook(tmp_path, monkeypatch):
    loader, _, _, _, _ = fixture(tmp_path, monkeypatch)
    with loader.intercept():
        hook = StaticAutotunerFuture.result
        with pytest.raises(ValueError, match="conflicting"), loader.intercept():
            pass
        assert StaticAutotunerFuture.result is hook
    loader.close()


def test_parallel_real_codecache_imports_keep_atomic_target_selection(
    tmp_path, monkeypatch
):
    loader, manifest, graph, rechecks, _ = fixture(tmp_path, monkeypatch)
    with loader.intercept(), ThreadPoolExecutor(max_workers=4) as pool:
        # Different attrs force distinct actual Python modules for the same source.
        modules = list(
            pool.map(
                lambda index: PyCodeCache.load_by_key_path(
                    f"geometry-{index}", str(graph), attrs={"instance": index}
                ),
                range(4),
            )
        )
        assert len({id(module) for module in modules}) == 4
        assert all(rechecks.count(index) == 1 for index in range(3))
        report = inventory(
            PyCodeCache.modules, manifest, 0, "geometry", loader.observer
        )
        assert report["target_bindings"] == 3 and len(report["bindings"]) == 16
        loader.controller.seal(PyCodeCache.modules)
    loader.close()


def test_replacement_compiler_retains_provenance_and_rank_cache(tmp_path, monkeypatch):
    loader, manifest, _, _, _ = fixture(tmp_path, monkeypatch)
    target = manifest["targets"]["0"][0]
    calls, configs = [], []

    def load(copied, original, kernel, name, *, debug_source):
        calls.append((copied, original, kernel, name, debug_source))
        return SimpleNamespace(_precompile_config=lambda config: configs.append(config))

    monkeypatch.setattr(loader_module, "load_preserving_provenance", load)
    monkeypatch.setattr(torch.cuda, "device", lambda rank: nullcontext())
    for mode in ("control", "geometry"):
        GeometryLoader.compile_replacement(loader, target, target["configs"][mode])
    assert len(calls) == 1 and len(configs) == 2
    copied, original, kernel, _, debug = calls[0]
    assert str(original) == target["debug_source"] == str(debug)
    assert copied == loader.private / target["relative"] and kernel == target["kernel"]
    assert [c.kwargs["R0_BLOCK"] for c in configs] == [4096, 1024]
    assert [c.num_warps for c in configs] == [16, 8]
    loader.close()


def test_real_caching_autotuner_materializes_exact_rank_cache(tmp_path, monkeypatch):
    loader, _, _, _, _ = fixture(tmp_path, monkeypatch)
    loader.check_cache()
    assert "TRITON_CACHE_DIR" not in os.environ
    tuner = CachingAutotuner(
        SimpleNamespace(__name__="fixture_norm", src="def fixture_norm(): pass"),
        dict(
            device=SimpleNamespace(
                index=0, type="cuda", warp_size=32, max_threads_per_block=1024
            )
        ),
        [triton.Config({"XBLOCK": 1}, num_warps=8)],
        None,
        [],
        False,
        None,
    )
    assert os.environ["TRITON_CACHE_DIR"] == str(loader.cache / "triton/0")
    loader.check_cache()
    assert not tuner.compile_results  # Constructor only: no GPU compilation/load.
    loader.close()


@pytest.mark.parametrize("value", ["", "shared", "wrong_rank", "alias"])
def test_cache_materialization_rejects_other_paths(tmp_path, monkeypatch, value):
    loader, _, _, _, _ = fixture(tmp_path, monkeypatch)
    if value == "wrong_rank":
        value = str(loader.cache / "triton/1")
    elif value == "alias":
        path = tmp_path / "alias"
        path.symlink_to(loader.cache / "triton/0", target_is_directory=True)
        value = str(path)
    monkeypatch.setenv("TRITON_CACHE_DIR", value)
    with pytest.raises(ValueError, match="cache binding"):
        loader.check_cache()
    loader.close()


@pytest.mark.parametrize(
    "change", ["shared_cache", "wrong_inductor", "source", "debug"]
)
def test_compiler_rejects_provenance_or_cache_drift(tmp_path, monkeypatch, change):
    loader, manifest, _, _, _ = fixture(tmp_path, monkeypatch)
    target = manifest["targets"]["0"][0]
    if change == "shared_cache":
        monkeypatch.setenv("TRITON_CACHE_DIR", "")
    elif change == "wrong_inductor":
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    elif change == "source":
        (loader.private / target["relative"]).write_text("# drift\n")
    else:
        target["debug_source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        GeometryLoader.compile_replacement(
            loader, target, target["configs"]["geometry"]
        )
    loader.close()


def test_run_symbols_ignores_embedded_compile_time_docstring():
    assert run_symbols(
        '"""def call(): unused.run()"""\ndef call():\n    actual.run()\n'
    ) == {"actual"}


def test_non_target_comparison_ignores_handles_but_not_selection():
    a = dict(
        graphs=7,
        target_bindings=9,
        bindings=[
            dict(
                graph="g",
                symbol="k",
                source="s",
                target=False,
                selected=["control"],
                observed_binary_index=1,
                module_index=2,
            )
        ],
    )
    b = copy.deepcopy(a)
    b["bindings"][0].update(observed_binary_index=12, module_index=3)
    compare_unrelated(a, b)
    b["bindings"][0]["selected"] = ["different"]
    with pytest.raises(ValueError, match="non-target"):
        compare_unrelated(a, b)


def qualified_fixture(tmp_path):
    original = tmp_path / "original"
    manifest = dict(original_namespace=str(original), original_files={}, targets=[])
    binaries = []
    for rank, count in enumerate((3, 3, 3, 4)):
        for n in range(count):
            index = len(manifest["targets"])
            name = f"norm{index}"
            relative = f"inductor_cache/aa/{name}.py"
            source = original / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# {index}\n")
            target = dict(
                rank=rank,
                relative=relative,
                source=str(source),
                source_sha256=sha(source),
                debug_source=str(source),
                debug_source_sha256=sha(source),
                kernel=name,
            )
            for arm, config in (("control", CONTROL), ("geometry", GEOMETRY)):
                key = f"KEY{index}{arm}"
                path = original / f"inductor_cache/triton/{rank}/{key}/{name}.cubin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(key.encode())
                manifest["original_files"][str(path.relative_to(original))] = sha(path)
                binaries.append(
                    dict(
                        source_index=index,
                        rank=rank,
                        arm=arm,
                        source_sha256=sha(source),
                        selected=dict(hash=key, config=config),
                        cubin_sha256=sha(path),
                    )
                )
                if arm == "control":
                    target["control"] = dict(
                        config, triton_cache_hash=key, cubin_sha256=sha(path)
                    )
            manifest["targets"].append(target)
    return manifest, binaries


def test_source_bound_qualification_join(tmp_path):
    manifest, binaries = qualified_fixture(tmp_path)
    targets = qualified_targets(manifest, binaries)
    assert [len(t) for t in targets.values()] == [3, 3, 3, 4]
    assert targets["0"][0]["configs"]["geometry"]["triton_cache_hash"] == "KEY0geometry"


@pytest.mark.parametrize(
    "change",
    ["order", "rank", "config", "binary", "debug", "source", "control", "count"],
)
def test_source_bound_join_rejects_drift(tmp_path, change):
    manifest, binaries = qualified_fixture(tmp_path)
    if change == "order":
        binaries[0], binaries[1] = binaries[1], binaries[0]
    elif change == "rank":
        binaries[0]["rank"] = 1
    elif change == "config":
        binaries[1]["selected"]["config"] = CONTROL
    elif change == "binary":
        binaries[1]["cubin_sha256"] = "0" * 64
    elif change == "debug":
        manifest["targets"][0]["debug_source_sha256"] = "0" * 64
    elif change == "source":
        Path(manifest["targets"][0]["source"]).write_text("# drift\n")
    elif change == "control":
        manifest["targets"][0]["control"]["triton_cache_hash"] = "wrong"
    else:
        binaries.pop()
    with pytest.raises(ValueError):
        qualified_targets(manifest, binaries)


def test_preparation_cannot_refresh_changed_non_helper_artifacts(tmp_path):
    native = tmp_path / "native.so"
    native.write_bytes(b"old")
    before = {str(native): sha(native)}
    assert current_sources(before, [Path(__file__)])[1] == {}
    native.write_bytes(b"changed")
    with pytest.raises(ValueError, match="non-helper"):
        current_sources(before, [])
