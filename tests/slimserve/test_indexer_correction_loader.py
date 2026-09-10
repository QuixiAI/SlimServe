# SPDX-License-Identifier: Apache-2.0
"""CPU launcher mechanics; native driver and device allocation are simulated."""

import base64
import hashlib
from types import SimpleNamespace

import pytest
import torch
from triton.compiler import ASTSource

from benchmarks.kernels import glm53_indexer_correction_loader as module
from benchmarks.kernels import prepare_glm53_indexer_correction_loader as prepare
from benchmarks.kernels.audit_glm53_indexer_correction_graphs import inventory
from benchmarks.kernels.glm53_binary_observer import StaticCudaBinaryObserver
from benchmarks.kernels.glm53_indexer_correction import correct_indexer
from tests.slimserve.test_indexer_correction import tensors
from tests.slimserve.test_kv_loader import fixture as kv_fixture
from tests.slimserve.test_kv_loader import load_graphs


def triton_image(tmp_path):
    identity = hashlib.sha256(b"fixture key").hexdigest()
    key = base64.b32encode(bytes.fromhex(identity)).decode().rstrip("=")
    path = tmp_path / key / "correct_indexer.cubin"
    path.parent.mkdir(parents=True)
    image = b"fixture cubin"
    path.write_bytes(image)
    compiled = SimpleNamespace(
        hash=identity,
        src=ASTSource(correct_indexer, signature=module.SIGNATURE),
        asm={"cubin": image},
        metadata=SimpleNamespace(**module.OPTIONS, shared=0, num_ctas=1),
    )
    record = dict(
        kernel="correct_indexer",
        selected=dict(hash=key, config=dict(num_warps=1, num_stages=1)),
        cubin_sha256=module.sha(path),
    )
    return compiled, record, path


def test_actual_static_constructor_observer_and_n_grid_preserve_six_argument_abi(
    tmp_path,
):
    compiled, record, path = triton_image(tmp_path)
    result = module.to_static(compiled, record, 0, tmp_path)
    assert result.kernel.cubin_raw == b"fixture cubin"
    assert result.kernel.function is None
    calls = []
    result.kernel.C_impl = SimpleNamespace(
        _load_kernel=lambda *a: (10, 20, 32, 0),
        _launch_kernel=lambda *a: calls.append(a),
        _unload_kernel=lambda *_: None,
    )
    observer = StaticCudaBinaryObserver(
        0,
        [tmp_path],
        expected_images={
            (path.parent.name, record["kernel"]): {record["cubin_sha256"]}
        },
    )
    with observer.intercept():
        launcher = result.make_launcher()
        observer.verify_launcher(result, launcher)
        args = (*[object() for _ in range(5)], 7616)
        launcher(*args, stream=123)
        assert calls == [(20, 7616, 1, 1, 1, 0, "OOOOOi", args, 123)]
        assert observer.digest(result) == record["cubin_sha256"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "change",
    [
        "key",
        "bytes",
        "disk",
        "signature",
        "warps",
        "fusion",
        "name",
        "config",
        "arg_order",
    ],
)
def test_static_bridge_rejects_unqualified_kernel_before_driver_load(tmp_path, change):
    compiled, record, path = triton_image(tmp_path)
    if change == "key":
        record["selected"]["hash"] = "wrong"
    elif change == "bytes":
        compiled.asm["cubin"] = b"other image"
    elif change == "disk":
        path.write_bytes(b"changed")
    elif change == "signature":
        compiled.src.signature = {**module.SIGNATURE, "N": "i64"}
    elif change == "warps":
        compiled.metadata.num_warps = 4
    elif change == "fusion":
        compiled.metadata.enable_fp_fusion = True
    elif change == "config":
        record["selected"]["config"]["num_stages"] = 2
    elif change == "arg_order":
        compiled.src.fn = SimpleNamespace(
            __name__="correct_indexer", arg_names=list(reversed(module.SIGNATURE))
        )
    else:
        record["kernel"] = "wrong"
    with pytest.raises(ValueError):
        module.to_static(compiled, record, 0, tmp_path)
    assert not torch.cuda.is_initialized()


def test_bound_adapter_uses_fixed_guarded_arena_views_without_allocating(monkeypatch):
    args, _ = tensors(3)
    storage = torch.full((10, 128), 171, dtype=torch.uint8)
    calls = []

    def combo(*a, **kw):
        calls.append(("combo", a, kw))
        return "original result"

    def extra(*a, **kw):
        calls.append(("extra", a, kw))

    adapter = module.BoundIndexerCorrection(combo, extra, storage, 8)

    def forbidden(*a, **kw):
        raise AssertionError("allocation forbidden in dispatch")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "full", forbidden)
    assert adapter.run(*args, stream=123) == "original result"
    assert [c[0] for c in calls] == ["combo", "extra"]
    launched = calls[1][1]
    assert (
        launched[0] is args[0]
        and launched[1] is args[3]
        and launched[2] is args[4]
        and launched[3] is args[7]
    )
    assert launched[4].data_ptr() == storage.data_ptr() + 128
    assert launched[4].shape == (3, 128) and launched[5] == 3
    assert calls[1][2] == dict(stream=123)


@pytest.mark.parametrize(
    "change", ["overflow", "rebound", "dtype", "shape", "run", "combo", "correction"]
)
def test_arena_and_dispatch_drift_are_rejected(change):
    args, _ = tensors(3)

    def original(*a, **kw):
        raise AssertionError("must not launch")

    def extra(*a, **kw):
        raise AssertionError("must not launch")

    storage = torch.full((10, 128), 171, dtype=torch.uint8)
    adapter = module.BoundIndexerCorrection(original, extra, storage, 8)
    if change == "overflow":
        with pytest.raises(ValueError, match="exceed"):
            adapter.run(*args[:-3], 9, 9, 9, stream=0)
        return
    if change == "rebound":
        adapter.storage = storage.clone()
    elif change == "dtype":
        adapter.storage = storage.float()
    elif change == "shape":
        adapter.capacity = 7
    elif change == "run":
        adapter.run = original
    elif change == "combo":
        adapter.combo = extra
    else:
        adapter.correction = original
    with pytest.raises(ValueError):
        module.check_adapter(adapter, original, extra, 8)


def correction_fixture(tmp_path, monkeypatch, mode):
    f = kv_fixture(tmp_path, monkeypatch, mode="control")
    f.manifest["schema"] = module.SCHEMA
    f.manifest["selection_capacity"] = 8192
    target = f.manifest["targets"]["0"][0]
    target["correction"] = target.pop("kv")
    f.loader = module.IndexerCorrectionLoader(
        0, f.manifest, tmp_path / "manifest.json", mode
    )
    f.loader.compile_replacement = lambda _: f.results["kv"]

    def make_adapter(controller, launcher, pair):
        storage = torch.full((8194, 128), 171, dtype=torch.uint8)
        adapter = module.BoundIndexerCorrection(launcher, pair[1], storage, 8192)
        return adapter, adapter.run

    monkeypatch.setattr(
        module.IndexerCorrectionIntervention, "make_adapter", make_adapter
    )
    return f


def test_control_actual_future_and_graph_inventory_preserve_original_dispatch(
    tmp_path, monkeypatch
):
    f = correction_fixture(tmp_path, monkeypatch, "control")
    with f.loader.intercept():
        modules = load_graphs(f)
        report = inventory(modules, f.manifest, 0, "control", f.loader.observer)
        assert report["target_bindings"] == 2
        assert all(r["dispatch"] == "direct_combo" for r in report["bindings"])
        f.loader.controller.seal(modules)


def test_candidate_controller_binds_extra_static_image_and_rejects_arena_drift(
    tmp_path, monkeypatch
):
    f = correction_fixture(tmp_path, monkeypatch, "correction")
    with f.loader.intercept():
        modules = load_graphs(f)
        binding = next(iter(f.loader.controller.owners.values()))
        f.loader.controller.verify(binding)
        assert (
            f.tuner.run.__self__.correction.__globals__["runner"].__self__
            is f.results["kv"].kernel
        )
        # The CPU arena is deliberately not claimed to be a rank0 GPU tensor.
        with pytest.raises(ValueError, match="wrong rank"):
            inventory(modules, f.manifest, 0, "correction", f.loader.observer)
        binding["adapter"].storage = binding["adapter"].storage.clone()
        with pytest.raises(ValueError, match="arena"):
            f.loader.controller.verify(binding)


def test_candidate_binding_rejects_changed_extra_handle(tmp_path, monkeypatch):
    f = correction_fixture(tmp_path, monkeypatch, "correction")
    with f.loader.intercept():
        load_graphs(f)
        binding = next(iter(f.loader.controller.owners.values()))
        f.results["kv"].kernel.function = -1
        with pytest.raises(ValueError, match="handles"):
            f.loader.controller.verify(binding)


def test_cpu_base_joins_completed_correction_and_actual_aot_roots():
    if not (prepare.RESULTS / "indexer-correction-v1/analysis.json").exists():
        pytest.skip("campaign evidence unavailable")
    base = prepare.build_base()
    assert base["schema"] == module.SCHEMA
    assert base["selection_capacity"] == 8192
    assert all(len(base["artifact_roots"][str(r)]) == 7 for r in range(4))
    assert all(
        len(base["targets"][str(r)][0]["static_graph_uses"]) == 2 for r in range(4)
    )
    cubins = {
        base["targets"][str(r)][0]["correction"]["cubin_sha256"] for r in range(4)
    }
    assert cubins == {
        "bd00effc35c7c0619ce74d3819aa3322cc110e423ce157ce855994f30188453e"
    }
