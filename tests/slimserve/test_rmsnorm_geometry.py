# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from benchmarks.kernels.glm53_rmsnorm_geometry import (
    CONTROL,
    GEOMETRY,
    SCHEMA,
    MultiIntervention,
)
from benchmarks.kernels.prepare_glm53_rmsnorm_geometry import select_changes
from slimserve.rmsnorm_diagnostic import sha


def compiled(saved):
    cubin = saved["triton_cache_hash"].encode()
    return SimpleNamespace(
        kernel=SimpleNamespace(asm={"cubin": cubin}),
        make_launcher=lambda: SimpleNamespace(
            cache_hash=saved["triton_cache_hash"],
            config=SimpleNamespace(
                kwargs={k: saved[k] for k in ("XBLOCK", "R0_BLOCK")},
                num_warps=saved["num_warps"],
                num_stages=saved["num_stages"],
            ),
        ),
    )


def compile_target(target, saved):
    return compiled(saved)


def setup(tmp_path, mode="geometry", count=3):
    original, private = tmp_path / "original", tmp_path / "private"
    targets, tuners = [], []
    for index in range(count):
        relative = f"inductor_cache/aa/norm{index}.py"
        for folder in (original, private):
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# source {index}\n")
        configs = {}
        for arm, config in (("control", CONTROL), ("geometry", GEOMETRY)):
            identity = f"{index}-{arm}"
            configs[arm] = dict(
                config,
                triton_cache_hash=identity,
                cubin_sha256=hashlib.sha256(identity.encode()).hexdigest(),
            )
        targets.append(
            dict(relative=relative, source_sha256=sha(path), configs=configs)
        )
        native = compiled(configs["control"])
        tuners.append(
            SimpleNamespace(
                filename=str(original / relative),
                launchers=[native.make_launcher()],
                compile_results=[native],
                configs=["old"],
                save_cache_hook="old",
                _cached_launcher="old",
            )
        )
    manifest = dict(
        schema=SCHEMA,
        original_namespace=str(original),
        private_namespace=str(private),
        targets={"0": targets},
        receipts=str(tmp_path / "receipts"),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    controller = MultiIntervention(0, manifest, manifest_path, mode)
    return controller, tuners


def graph(tmp_path, tuners):
    source = tmp_path / "graph.py"
    source.write_text("# graph fixture\n")
    return SimpleNamespace(
        __file__=str(source),
        call=lambda: None,
        **{f"norm{i}": t for i, t in enumerate(tuners)},
    )


@pytest.mark.parametrize("mode", ["control", "geometry"])
def test_multi_source_aliases_resolve_upstream_once_per_object(tmp_path, mode):
    controller, tuners = setup(tmp_path, mode)
    # Every source has two distinct futures referring to the same object.
    futures = [SimpleNamespace(static_autotuner=t) for t in tuners for _ in range(2)]
    calls, compiles = [], []
    barrier = threading.Barrier(len(futures))

    def upstream(future, timeout=None):
        calls.append((id(future.static_autotuner), timeout))
        # Simulate the upstream recheck that would undo an earlier substitution.
        tuner = future.static_autotuner
        native = compiled(controller.owner(tuner).target["configs"][1])
        tuner.compile_results = [native]
        tuner.launchers = [native.make_launcher()]
        return tuner

    def compile_once(target, saved):
        compiles.append(target["filename"])
        return compiled(saved)

    def resolve(future):
        barrier.wait(timeout=5)
        return controller.resolve(future, upstream, compile_once, timeout=7)

    with ThreadPoolExecutor(max_workers=len(futures)) as pool:
        results = list(pool.map(resolve, futures))
    assert all(r is f.static_autotuner for r, f in zip(results, futures))
    assert sorted(calls) == sorted((id(t), 7) for t in tuners)
    assert len(compiles) == len(tuners)
    for index, tuner in enumerate(tuners):
        assert tuner.launchers[0].cache_hash == f"{index}-{mode}"
        assert tuner.save_cache_hook is None and tuner._cached_launcher is None
        assert tuner.filename.startswith(str(controller.private))
    module = graph(tmp_path, tuners)
    controller.bind_graph(module, compile_once)
    controller.seal([module])
    for future in futures:
        controller.resolve(future, upstream, compile_once)
    assert len(calls) == len(tuners) and len(compiles) == len(tuners)
    for child in controller.controllers.values():
        records = [json.loads(line) for line in child.path.read_text().splitlines()]
        assert records[0]["mode"] == mode
        assert records[1]["schema"] == SCHEMA


def test_graph_loader_covers_ordinary_objects_aliases_and_distinct_targets(tmp_path):
    controller, tuners = setup(tmp_path, count=4)
    other = copy.copy(tuners[0])
    module = graph(tmp_path, [*tuners, tuners[0], other])
    controller.bind_graph(module, compile_target)
    assert len(controller.owners) == 5
    assert sum(len(c.graph_bindings) for c in controller.controllers.values()) == 6
    controller.seal([module])
    controller.bind_graph(module, compile_target)
    controller.verify_graphs([module])


def test_missing_source_prevents_all_seals(tmp_path):
    controller, tuners = setup(tmp_path)
    module = graph(tmp_path, tuners[:-1])
    controller.bind_graph(module, compile_target)
    with pytest.raises(ValueError, match="missing RMSNorm graph coverage"):
        controller.seal([module])
    assert not controller.sealed
    assert all(not c.sealed for c in controller.controllers.values())


def test_static_replacement_does_not_count_as_graph_coverage(tmp_path):
    controller, tuners = setup(tmp_path)
    for tuner in tuners:
        controller.replace(tuner, compile_target)
    with pytest.raises(ValueError, match="missing RMSNorm graph coverage"):
        controller.seal([])


@pytest.mark.parametrize(
    "change",
    [
        "filename",
        "unknown_path",
        "original_path",
        "removed",
        "new_object",
        "cubin",
        "source",
    ],
)
def test_changed_bindings_are_rejected_after_seal(tmp_path, change):
    controller, tuners = setup(tmp_path)
    module = graph(tmp_path, tuners)
    controller.bind_graph(module, compile_target)
    controller.seal([module])
    tuner = tuners[0]
    if change == "filename":
        tuner.filename = tuners[1].filename
    elif change == "unknown_path":
        tuner.filename = str(tmp_path / "norm0.py")
    elif change == "original_path":
        tuner.filename = str(controller.original / "inductor_cache/aa/norm0.py")
    elif change == "removed":
        del module.norm0
    elif change == "new_object":
        module.norm0 = copy.copy(tuner)
    elif change == "cubin":
        tuner.compile_results[0].kernel.asm["cubin"] = b"different debug image"
    else:
        from pathlib import Path

        Path(tuner.filename).write_text("# changed\n")
    with pytest.raises(ValueError):
        controller.verify_graphs([module])


def test_missing_graph_and_late_alias_are_rejected(tmp_path):
    controller, tuners = setup(tmp_path)
    module = graph(tmp_path, tuners)
    controller.bind_graph(module, compile_target)
    controller.seal([module])
    with pytest.raises(ValueError, match="loader inventory"):
        controller.verify_graphs([])
    module.late_alias = tuners[0]
    with pytest.raises(ValueError, match="new geometry graph binding"):
        controller.bind_graph(module, compile_target)


def test_late_static_target_is_rejected_before_upstream_resolution(tmp_path):
    controller, tuners = setup(tmp_path)
    module = graph(tmp_path, tuners)
    controller.bind_graph(module, compile_target)
    controller.seal([module])
    with pytest.raises(ValueError, match="new geometry target binding"):
        controller.resolve(
            SimpleNamespace(static_autotuner=copy.copy(tuners[0])),
            lambda *_: pytest.fail("upstream resolution after seal"),
            compile_target,
        )


def test_unrelated_static_source_is_relocated_without_numerical_change(tmp_path):
    controller, tuners = setup(tmp_path)
    for root in (controller.original, controller.private):
        (root / "other.py").write_text("# unrelated\n")
    tuner = copy.copy(tuners[0])
    tuner.filename = str(controller.original / "other.py")
    before = tuner.launchers
    calls = []

    def upstream(future, timeout=None):
        calls.append(timeout)
        return future.static_autotuner

    controller.resolve(
        SimpleNamespace(static_autotuner=tuner),
        upstream,
        lambda *_: pytest.fail("unrelated compilation"),
        timeout=9,
    )
    assert calls == [9]
    assert tuner.launchers is before
    assert tuner.filename == str(controller.private / "other.py")
    assert tuner.save_cache_hook is None


@pytest.mark.parametrize("stage", ["native", "replacement", "cached"])
def test_binary_bytes_are_checked_before_replacement(tmp_path, stage):
    controller, tuners = setup(tmp_path)
    tuner = tuners[0]
    if stage == "cached":
        controller.replace(tuner, compile_target)
        cached = tuner.compile_results[0]
        native = compiled(controller.owner(tuner).target["configs"][1])
        tuner.compile_results = [native]
        tuner.launchers = [native.make_launcher()]
        cached.kernel.asm["cubin"] = b"changed cached binary"
    elif stage == "native":
        tuner.compile_results[0].kernel.asm["cubin"] = b"changed original binary"
    before = tuner.launchers

    def bad_compile(target, saved):
        result = compiled(saved)
        result.kernel.asm["cubin"] = b"changed replacement binary"
        return result

    with pytest.raises(ValueError, match="cubin"):
        controller.replace(
            tuner, bad_compile if stage == "replacement" else compile_target
        )
    assert tuner.launchers is before


@pytest.mark.parametrize(
    "change", ["schema", "mode", "duplicate", "overlap", "path", "config", "digest"]
)
def test_manifest_rejects_wrong_scope_or_geometry(tmp_path, change):
    controller, _ = setup(tmp_path)
    manifest = copy.deepcopy(controller.manifest)
    mode = "geometry"
    if change == "schema":
        manifest["schema"] = 1
    elif change == "mode":
        mode = "legacy"
    elif change == "duplicate":
        manifest["targets"]["0"].append(manifest["targets"]["0"][0])
    elif change == "overlap":
        manifest["private_namespace"] = manifest["original_namespace"]
    elif change == "path":
        manifest["targets"]["0"][0]["relative"] = "../norm.py"
    elif change == "config":
        manifest["targets"]["0"][0]["configs"]["geometry"]["R0_BLOCK"] = 2048
    else:
        manifest["targets"]["0"][0]["configs"]["control"]["cubin_sha256"] = (
            "key-not-bytes"
        )
    with pytest.raises(ValueError):
        MultiIntervention(0, manifest, tmp_path / "manifest.json", mode)


def discovery_fixture():
    changes, inventory = [], []
    for index, rank in enumerate([0] * 3 + [1] * 3 + [2] * 3 + [3] * 4):
        source = f"/original/source{index}.py"
        changes.append(
            dict(
                rank=rank,
                old_source=source,
                old_config=CONTROL,
                new_config=GEOMETRY,
                body="body",
                roles=["input_layernorm" if index < 6 else "post_attention_layernorm"],
            )
        )
        inventory.append(
            dict(
                rank=rank,
                source=source,
                arm="combo",
                selected=dict(config=CONTROL),
                info=dict(body_sha256="body", widths=[4096]),
            )
        )
    return dict(correspondence=dict(unique_changed_sources=changes)), inventory


def test_discovery_requires_exact_thirteen_source_correspondence():
    mapping, inventory = discovery_fixture()
    chosen = select_changes(mapping, inventory)
    assert len(chosen) == 13
    mapping["correspondence"]["unique_changed_sources"].reverse()
    inventory.reverse()
    assert select_changes(mapping, inventory) == chosen


@pytest.mark.parametrize(
    "change", ["missing_rank", "duplicate", "role", "source", "body", "config", "width"]
)
def test_discovery_rejects_incomplete_or_drifted_targets(change):
    mapping, inventory = discovery_fixture()
    changes = mapping["correspondence"]["unique_changed_sources"]
    if change == "missing_rank":
        changes.pop()
    elif change == "duplicate":
        changes[1] = changes[0]
    elif change == "role":
        changes[0]["roles"] = ["final_mean_and_norm"]
    elif change == "source":
        inventory[0]["arm"] = "split"
    elif change == "body":
        inventory[0]["info"]["body_sha256"] = "other"
    elif change == "config":
        changes[0]["new_config"] = CONTROL
    else:
        inventory[0]["info"]["widths"] = [1536]
    with pytest.raises(ValueError):
        select_changes(mapping, inventory)
