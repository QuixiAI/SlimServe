# SPDX-License-Identifier: Apache-2.0
import base64
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import triton
from torch._inductor.runtime.static_triton_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

from benchmarks.kernels.glm53_binary_observer import StaticCudaBinaryObserver
from benchmarks.kernels.glm53_rmsnorm_geometry import (
    CONTROL,
    GEOMETRY,
    SCHEMA,
    MultiIntervention,
)
from slimserve.rmsnorm_diagnostic import sha


def make_compiled(root, name="norm", config=None, rank=0):
    config = config or GEOMETRY
    identity = hashlib.sha256(
        (name + json.dumps(config, sort_keys=True)).encode()
    ).hexdigest()
    key = base64.b32encode(bytes.fromhex(identity)).decode().rstrip("=")
    path = root / key / f"{name}.cubin"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (name + identity).encode()
    path.write_bytes(raw)
    kernel = StaticallyLaunchedCudaKernel.__new__(StaticallyLaunchedCudaKernel)
    kernel.name, kernel.hash = name, identity
    kernel.num_warps, kernel.shared = config["num_warps"], 0
    kernel.cubin_path, kernel.cubin_raw = str(path), raw
    kernel.module = kernel.function = None
    kernel.arg_names = ["x", "xnumel"]
    kernel.declared_constexprs = kernel.full_constexprs = []
    kernel.has_global_scratch = kernel.has_profile_scratch = False
    kernel.arg_tys = "OO"
    calls = []

    def load(filename, actual_name, shared, device):
        assert (filename, actual_name, shared, device) == (str(path), name, 0, rank)
        assert path.read_bytes() == raw
        calls.append(dict(event="driver_load", path=filename))
        return id(kernel), id(kernel) + 1, 32, 0

    kernel.C_impl = SimpleNamespace(
        _load_kernel=load,
        _unload_kernel=lambda handle: None,
        _launch_kernel=lambda *args: calls.append(
            dict(event="driver_launch", args=args)
        ),
    )
    compiled = StaticTritonCompileResult(
        kernel,
        triton.Config(
            {k: v for k, v in config.items() if k not in ("num_warps", "num_stages")},
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        ),
        dict(
            device=rank,
            device_type="cuda",
            constants={},
            signature=dict(x="*bf16", xnumel="i32"),
        ),
        dict(grid_type="Grid1D"),
    )
    return compiled, path, calls


def observer_fixture(tmp_path, monkeypatch):
    root = tmp_path / "triton"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(root))
    compiled, path, calls = make_compiled(root)
    expected = {(path.parent.name, compiled.kernel.name): {sha(path)}}
    events = []
    observer = StaticCudaBinaryObserver(
        0, [root], expected_images=expected, emit=events.append
    )
    return observer, compiled, path, calls, events


def test_actual_static_make_load_and_reuse_have_binary_proof(tmp_path, monkeypatch):
    observer, compiled, path, calls, events = observer_fixture(tmp_path, monkeypatch)
    original = StaticallyLaunchedCudaKernel.load_kernel
    local = StaticallyLaunchedCudaKernel.__dict__.get("load_kernel")
    with observer.intercept():
        assert observer.digest(compiled) == sha(path)
        launcher = compiled.make_launcher()
        assert compiled.kernel.cubin_raw is compiled.kernel.cubin_path is None
        observer.verify_launcher(compiled, launcher)
        assert observer.digest(compiled) == sha(path)
        launcher(None, 3, stream=0)
        assert calls[-1]["event"] == "driver_launch"
        assert calls[-1]["args"][0] == compiled.kernel.function
        observer.seal()
        again = compiled.make_launcher()
        observer.verify_launcher(compiled, again)
        assert sum(c["event"] == "driver_load" for c in calls) == 1
        assert len(observer.records) == 1
    assert StaticallyLaunchedCudaKernel.load_kernel is original
    assert StaticallyLaunchedCudaKernel.__dict__.get("load_kernel") is local
    assert [r["event"] for r in events] == [
        "binary_loaded",
        "binary_observer_sealed",
        "binary_load_reuse",
    ]


def test_serialized_object_without_raw_bytes_uses_actual_preload_driver_path(
    tmp_path, monkeypatch
):
    observer, compiled, path, calls, _ = observer_fixture(tmp_path, monkeypatch)
    compiled.kernel.cubin_raw = None
    with pytest.raises(ValueError, match="unavailable"):
        observer.digest(compiled)
    with observer.intercept():
        launcher = compiled.make_launcher()
        observer.verify_launcher(compiled, launcher)
        assert observer.digest(compiled) == sha(path)
    assert len(calls) == 1


def test_unobserved_loaded_object_cannot_borrow_a_disk_receipt(tmp_path, monkeypatch):
    observer, compiled, _, _, _ = observer_fixture(tmp_path, monkeypatch)
    compiled.make_launcher()
    with pytest.raises(ValueError, match="lacks pre-load"):
        observer.digest(compiled)
    with observer.intercept(), pytest.raises(ValueError, match="no pre-load proof"):
        compiled.make_launcher()
    assert not observer.records


@pytest.mark.parametrize(
    "change", ["root", "alias", "key", "name", "raw", "expected", "rank"]
)
def test_invalid_preload_stops_before_driver_call(tmp_path, monkeypatch, change):
    observer, compiled, path, calls, _ = observer_fixture(tmp_path, monkeypatch)
    if change == "root":
        observer.roots = {tmp_path / "other"}
    elif change == "alias":
        alias = path.with_name("alias.cubin")
        alias.symlink_to(path)
        compiled.kernel.cubin_path = str(alias)
    elif change == "key":
        compiled.kernel.hash = "00" * 32
    elif change == "name":
        compiled.kernel.name = "wrong"
    elif change == "raw":
        compiled.kernel.cubin_raw = b"different debug bytes"
    elif change == "expected":
        observer.expected[(path.parent.name, compiled.kernel.name)] = {"wrong"}
    else:
        observer.rank = 1
    with observer.intercept(), pytest.raises(ValueError):
        compiled.make_launcher()
    assert calls == [] and not observer.records


@pytest.mark.parametrize(
    "change", ["closed", "function", "module", "hash", "warps", "name", "raw"]
)
def test_loaded_proof_is_invalidated_by_changed_object_state(
    tmp_path, monkeypatch, change
):
    observer, compiled, _, _, _ = observer_fixture(tmp_path, monkeypatch)
    with observer.intercept():
        launcher = compiled.make_launcher()
    kernel = compiled.kernel
    if change == "closed":
        kernel.close()
    elif change in ("function", "module"):
        setattr(kernel, change, -1)
    elif change == "hash":
        kernel.hash = "00" * 32
    elif change == "warps":
        kernel.num_warps = 16
    elif change == "name":
        kernel.name = "different"
    else:
        kernel.cubin_raw = b"unobserved"
    with pytest.raises(ValueError, match="changed"):
        observer.digest(compiled)
    with pytest.raises(ValueError, match="changed"):
        observer.verify_launcher(compiled, launcher)


def test_changed_disk_image_cannot_sneak_through_make_launcher_reuse(
    tmp_path, monkeypatch
):
    observer, compiled, path, calls, _ = observer_fixture(tmp_path, monkeypatch)
    with observer.intercept():
        compiled.make_launcher()
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="unqualified"):
            compiled.make_launcher()
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["runner", "static_flag", "hash", "shared", "warps"])
def test_claimed_cache_key_is_not_proof_of_launcher_binding(
    tmp_path, monkeypatch, change
):
    observer, compiled, _, _, _ = observer_fixture(tmp_path, monkeypatch)
    with observer.intercept():
        launcher = compiled.make_launcher()
    if change == "runner":
        launcher.__globals__["runner"] = lambda *_: None
    elif change == "static_flag":
        launcher._is_static = False
    elif change == "hash":
        launcher.cache_hash = "different"
    elif change == "shared":
        launcher.shared = 4
    else:
        launcher.config.num_warps = 16
    with pytest.raises(ValueError, match="not bound"):
        observer.verify_launcher(compiled, launcher)


def test_concurrent_make_launcher_aliases_load_driver_once(tmp_path, monkeypatch):
    observer, compiled, _, calls, _ = observer_fixture(tmp_path, monkeypatch)
    barrier = threading.Barrier(8)

    def make(_):
        barrier.wait(timeout=5)
        return compiled.make_launcher()

    with observer.intercept(), ThreadPoolExecutor(max_workers=8) as pool:
        launchers = list(pool.map(make, range(8)))
    assert len(observer.records) == len(calls) == 1
    for launcher in launchers:
        observer.verify_launcher(compiled, launcher)


def test_seal_and_context_cleanup_fail_closed(tmp_path, monkeypatch):
    observer, compiled, path, _, _ = observer_fixture(tmp_path, monkeypatch)
    original = StaticallyLaunchedCudaKernel.load_kernel
    with pytest.raises(ValueError, match="empty"):
        observer.seal()
    with observer.intercept():
        with pytest.raises(ValueError, match="twice"), observer.intercept():
            pass
        other = StaticCudaBinaryObserver(0, [path.parent.parent])
        with pytest.raises(ValueError, match="twice"), other.intercept():
            pass
        compiled.make_launcher()
        observer.seal()
        late, _, _ = make_compiled(path.parent.parent, "late")
        with pytest.raises(ValueError, match="after seal"):
            late.make_launcher()
    assert StaticallyLaunchedCudaKernel.load_kernel is original


def test_failed_driver_load_does_not_publish_binary_proof(tmp_path, monkeypatch):
    observer, compiled, _, _, _ = observer_fixture(tmp_path, monkeypatch)
    original = StaticallyLaunchedCudaKernel.load_kernel

    def failed(*_):
        raise RuntimeError("driver failure")

    compiled.kernel.C_impl._load_kernel = failed
    with pytest.raises(RuntimeError, match="driver failure"), observer.intercept():
        compiled.make_launcher()
    assert not observer.records
    assert StaticallyLaunchedCudaKernel.load_kernel is original


def test_foreign_hook_change_is_not_overwritten_on_exit(tmp_path, monkeypatch):
    observer, _, _, _, _ = observer_fixture(tmp_path, monkeypatch)
    local = StaticallyLaunchedCudaKernel.__dict__.get("load_kernel")

    def foreign(kernel, device):
        pass

    try:
        with pytest.raises(ValueError, match="hook changed"), observer.intercept():
            StaticallyLaunchedCudaKernel.load_kernel = foreign
        assert StaticallyLaunchedCudaKernel.load_kernel is foreign
    finally:
        if local is None:
            delattr(StaticallyLaunchedCudaKernel, "load_kernel")
        else:
            StaticallyLaunchedCudaKernel.load_kernel = local


def test_multi_source_controller_handles_consumed_static_images(tmp_path, monkeypatch):
    root = tmp_path / "triton"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(root))
    original, private = tmp_path / "original", tmp_path / "private"
    targets, tuners, compiled_pairs, expected = [], [], [], {}
    for index in range(3):
        name = f"norm{index}"
        relative = f"inductor_cache/aa/{name}.py"
        for folder in (original, private):
            source = folder / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# {name}\n")
        configs, pair = {}, {}
        for arm, config in (("control", CONTROL), ("geometry", GEOMETRY)):
            result, path, _ = make_compiled(root, name, config)
            configs[arm] = dict(
                config, triton_cache_hash=path.parent.name, cubin_sha256=sha(path)
            )
            expected[(path.parent.name, name)] = {sha(path)}
            pair[arm] = result
        targets.append(
            dict(relative=relative, source_sha256=sha(source), configs=configs)
        )
        compiled_pairs.append(pair)
    manifest = dict(
        schema=SCHEMA,
        original_namespace=str(original),
        private_namespace=str(private),
        targets={"0": targets},
        receipts=str(tmp_path / "receipts"),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    observer = StaticCudaBinaryObserver(0, [root], expected_images=expected)
    controller = MultiIntervention(
        0, manifest, manifest_path, "geometry", binary_observer=observer
    )

    def replacement(target, saved):
        index = int(target["filename"].removeprefix("norm").removesuffix(".py"))
        return compiled_pairs[index]["geometry"]

    with observer.intercept():
        for target, pair in zip(targets, compiled_pairs):
            native = pair["control"]
            tuner = SimpleNamespace(
                filename=str(original / target["relative"]),
                launchers=[native.make_launcher()],
                compile_results=[native],
                configs=None,
                _cached_launcher=None,
                save_cache_hook=None,
            )
            assert native.kernel.cubin_raw is None
            tuners.append(tuner)
        graph_path = tmp_path / "graph.py"
        graph_path.write_text("# graph\n")
        module = SimpleNamespace(
            __file__=str(graph_path),
            call=lambda: None,
            **{f"norm{i}": tuner for i, tuner in enumerate(tuners)},
        )
        controller.bind_graph(module, replacement)
        controller.seal([module])
        controller.bind_graph(module, replacement)
        controller.verify_graphs([module])
    assert len(observer.records) == 6
    assert all(t.compile_results[0].kernel.cubin_raw is None for t in tuners)
    assert all(t.launchers[0].config.num_warps == 8 for t in tuners)
