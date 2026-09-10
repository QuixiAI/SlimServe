# SPDX-License-Identifier: Apache-2.0
import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
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

from benchmarks.kernels import glm53_geometry_serving as serving
from benchmarks.kernels.glm53_artifact_roots import discover
from slimserve import glm53_ordering
from slimserve import rmsnorm_geometry as policy
from slimserve.registry import resolve
from slimserve.rmsnorm_diagnostic import expected_receipt, sha
from tests.slimserve.test_binary_observer import make_compiled
from tests.slimserve.test_geometry_loader import fixture
from vllm.compilation.caching import StandaloneCompiledArtifacts


def serving_fixture(tmp_path, monkeypatch, mode="geometry", count=1):
    unused, manifest, old_graph, _, pairs = fixture(tmp_path, monkeypatch, mode)
    unused.close()  # Keep its unused CPU-fixture receipts; use a new serving folder.
    private, original = (
        Path(manifest["private_namespace"]),
        Path(manifest["original_namespace"]),
    )
    graph_files = [f"inductor_cache/ra/graph{i}.py" for i in range(count)]
    for relative in graph_files:
        for root in (private, original):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(old_graph.read_text())
        manifest["original_files"][relative] = sha(private / relative)
    graph_path = private / graph_files[0]
    manifest["expected_graphs"] = {"0": {p: sha(private / p) for p in graph_files}}
    for target in manifest["targets"]["0"]:
        (use,) = target["static_graph_uses"]
        target["static_graph_uses"] = [
            dict(use, graph=str(original / p)) for p in graph_files
        ]
    model = private / "rank_0_0/model"
    model.parent.mkdir()
    model.write_bytes(b"owned-model-fixture")
    manifest["original_files"]["rank_0_0/model"] = sha(model)

    store = StandaloneCompiledArtifacts()
    for graph_index, relative in enumerate(graph_files):
        graph = CompiledFxGraph.__new__(CompiledFxGraph)
        graph.cache_key, graph.source_code, graph.cache_linemap = (
            Path(relative).stem,
            (private / relative).read_text(),
            [],
        )
        graph.current_callable = graph.compiled_fn_runner = None
        graph.inputs_to_check, graph.mutated_input_idxs = [0, 4], OrderedSet([4])
        graph._defers_input_alignment, graph._wrap_compiled_regions = True, False
        entry = BundledAOTAutogradResult.__new__(BundledAOTAutogradResult)
        entry.compiled_fw = BundledCompiledForward.__new__(BundledCompiledForward)
        entry.compiled_fw.result, entry.compiled_bw = graph, None
        blob = pickle.dumps(pickle.dumps((entry, {}, {})))
        for index in range(46 // count + (graph_index < 46 % count)):
            store.insert(f"graph{graph_index}layer{index}", "shape", blob)
    manifest["artifact_roots"] = {
        "0": discover(store, private, manifest["expected_graphs"]["0"])
    }

    # Build the expected source/config/binary receipt without loading a GPU object.
    rows = []
    for index, (source, pair) in enumerate(pairs.items()):
        target = index < 3
        arm = mode if target else "control"
        compiled = pair[arm]
        image = Path(compiled.kernel.cubin_path)
        selected = dict(
            compiled.config.kwargs,
            num_warps=compiled.config.num_warps,
            num_stages=compiled.config.num_stages,
            triton_cache_hash=image.parent.name,
        )
        for graph_index, root in enumerate(manifest["artifact_roots"]["0"]):
            rows.append(
                dict(
                    graph=root["graph"],
                    graph_sha256=root["graph_sha256"],
                    symbol=f"norm{index}",
                    source=source,
                    source_sha256=sha(private / source),
                    target=target,
                    referenced=True,
                    selected=expected_receipt(selected),
                    cubin_sha256=sha(image),
                    module_index=graph_index + 1,
                    observed_binary_index=index + 1,
                )
            )
    qualified = tmp_path / "qualified-graphs.json"
    qualified.write_text(
        json.dumps(dict(graphs=count, target_bindings=3 * count, bindings=rows))
    )
    manifest.update(
        mode=mode,
        receipts=str(tmp_path / "serving-targets"),
        worker_receipts=str(tmp_path / "workers"),
        qualified_graphs={
            mode: {"0": dict(path=str(qualified), sha256=sha(qualified))}
        },
    )
    path = tmp_path / "serving-manifest.json"
    path.write_text(json.dumps(manifest))

    def deserialize(data):
        # Real store/graph/cache/controller APIs; only the outer AOT wrapper and
        # static bundle transport are reduced. Driver calls use the existing mock.
        TritonBundler.load_autotuners(["static-fixture"])
        result = pickle.loads(data)[0].compiled_fw.result
        result.after_deserialization(SimpleNamespace(unwrap=lambda g: {}))
        maybe_realign_inputs(
            BoxedBool(False), result, result.inputs_to_check, result.mutated_input_idxs
        )
        return SimpleNamespace(graph=result)

    monkeypatch.setattr(AOTCompiledArtifact, "deserialize", staticmethod(deserialize))
    monkeypatch.setattr(
        TritonBundler, "load_autotuners", classmethod(lambda cls, tuners: tuners)
    )
    diagnostic = serving.ServingGeometry(0, manifest, path)
    diagnostic.loader.compile_replacement = lambda target, saved: pairs[
        target["relative"]
    ][mode]
    return diagnostic, store, graph_path, pairs


@pytest.mark.parametrize("mode", ["control", "geometry"])
def test_real_serving_lifecycle_seals_targets_but_allows_later_non_target_loads(
    tmp_path, monkeypatch, mode
):
    diagnostic, store, _, _ = serving_fixture(tmp_path, monkeypatch, mode)
    captures = []

    def capture():
        assert (
            diagnostic.loader.controller.sealed
            and not diagnostic.loader.observer.sealed
        )
        later, _, _ = make_compiled(
            diagnostic.loader.cache / "triton/0", "capture_later"
        )
        later.make_launcher()
        captures.append(True)
        return 42

    runner = SimpleNamespace(capture_model=capture)
    original_load = StandaloneCompiledArtifacts.load_all
    diagnostic.install(runner)
    try:
        store.load_all()
        assert diagnostic.loaded and diagnostic.summary["status"] == "aot-qualified"
        assert (
            diagnostic.loader.controller.sealed
            and not diagnostic.loader.observer.sealed
        )
        store.load_all()  # The same complete store is rechecked, never reloaded.
        later, _, _ = make_compiled(
            diagnostic.loader.cache / "triton/0", "warmup_later"
        )
        later.make_launcher()
        assert runner.capture_model() == 42
        assert (
            captures == [True] and diagnostic.summary["status"] == "capture-qualified"
        )
        assert set(diagnostic.summary["snapshots"]) == {
            "before-forward",
            "capture-before",
            "capture-after",
        }
        assert diagnostic.loader.installed and not diagnostic.loader.observer.sealed
    finally:
        diagnostic.close()
    assert StandaloneCompiledArtifacts.load_all is original_load
    assert runner.capture_model is capture
    assert all(s.closed for s in diagnostic.streams.values())


@pytest.mark.parametrize(
    "failure",
    [
        "early_capture",
        "fallback",
        "extra_store",
        "source",
        "config",
        "capture_mutation",
        "capture_exception",
    ],
)
def test_serving_failures_preserve_raw_state_and_restore_hooks(
    tmp_path, monkeypatch, failure
):
    diagnostic, store, graph_path, _ = serving_fixture(tmp_path, monkeypatch)

    def capture():
        if failure == "capture_exception":
            raise RuntimeError("capture fixture failed")
        if failure == "capture_mutation":
            graph_path.write_text(graph_path.read_text() + "# changed during capture\n")

    runner = SimpleNamespace(capture_model=capture)
    if failure == "fallback":
        monkeypatch.setattr(
            TritonBundler, "load_autotuners", classmethod(lambda cls, tuners: [])
        )
    original = StandaloneCompiledArtifacts.load_all
    diagnostic.install(runner)
    with pytest.raises((ValueError, RuntimeError)):
        if failure == "early_capture":
            runner.capture_model()
        else:
            store.load_all()
            if failure == "extra_store":
                StandaloneCompiledArtifacts().load_all()
            elif failure == "source":
                graph_path.write_text(graph_path.read_text() + "# changed\n")
            elif failure == "config":
                module = diagnostic.roots.roots(store, PyCodeCache.modules)[0]
                module.norm0.launchers[0].config.num_warps = 1
            runner.capture_model()
    assert diagnostic.closed and diagnostic.summary["status"] == "failed"
    assert list(diagnostic.folder.glob("failed-*-modules.json"))
    assert (
        StandaloneCompiledArtifacts.load_all is original
        and runner.capture_model is capture
    )


def test_close_preserves_a_foreign_capture_hook(tmp_path, monkeypatch):
    diagnostic, _, _, _ = serving_fixture(tmp_path, monkeypatch)
    runner = SimpleNamespace(capture_model=lambda: None)
    original = StandaloneCompiledArtifacts.load_all
    diagnostic.install(runner)
    foreign = lambda: None
    runner.capture_model = foreign
    with pytest.raises(ValueError, match="foreign serving"):
        diagnostic.close()
    assert (
        runner.capture_model is foreign
        and StandaloneCompiledArtifacts.load_all is original
    )
    assert not diagnostic.loader.installed and diagnostic.closed


def test_default_geometry_path_never_inspects_a_runner_or_plan(monkeypatch):
    monkeypatch.delenv(policy.FLAG, raising=False)
    policy.install(object())
    policy.validate_plan(object())


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(policy.FLAG, "control")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setattr(policy, "read_manifest", lambda: None)


@pytest.mark.parametrize(
    "flag,value",
    [
        (glm53_ordering.FLAG, "0"),
        ("VLLM_FORCE_AOT_LOAD", "0"),
        ("SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC", "control"),
        ("TORCHINDUCTOR_DETERMINISTIC", "1"),
        ("VLLM_GLM5_MHC_PREFILL_TC", "1"),
        ("VLLM_GLM5_MHC_BF16_FN", "0"),
        ("SLIMSERVE_GLM53_MODEL_JOURNAL", "1"),
    ],
)
def test_geometry_policy_rejects_other_modes(enabled, monkeypatch, flag, value):
    monkeypatch.setenv(flag, value)
    with pytest.raises(ValueError):
        policy.validate_plan(resolve("glm53-nvfp4-4", "rtx6000", 4, None))


def test_geometry_policy_requires_fixed_recipe_and_original_compiler(enabled):
    plan = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    policy.validate_plan(plan)
    changed_engine = copy.deepcopy(plan.engine)
    changed_engine["compilation_config"]["inductor_compile_config"] = {
        "combo_kernels": False
    }
    for changed in (
        replace(plan, weight_recipe=None),
        replace(plan, platform="a100"),
        replace(plan, engine=changed_engine),
        replace(plan, engine={**plan.engine, "kv_cache_dtype": "fp8"}),
        replace(plan, engine={**plan.engine, "dtype": "float16"}),
    ):
        with pytest.raises(ValueError):
            policy.validate_plan(changed)


@pytest.mark.parametrize(
    "change", [None, "dtype", "kv_dtype", "tp", "speculation", "compiler"]
)
def test_worker_installation_scope_and_precision(
    enabled, tmp_path, monkeypatch, change
):
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
    installs = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(
        policy, "read_manifest", lambda: ({}, tmp_path / "manifest.json")
    )
    monkeypatch.setattr(
        serving,
        "ServingGeometry",
        lambda *a: SimpleNamespace(install=lambda r: installs.append(r)),
    )
    if change == "dtype":
        runner.dtype = torch.float16
    elif change == "kv_dtype":
        runner.kv_cache_dtype = torch.float8_e4m3fn
    elif change == "tp":
        runner.parallel_config.tensor_parallel_size = 2
    elif change == "speculation":
        runner.speculative_config = object()
    elif change == "compiler":
        runner.compilation_config.inductor_compile_config["deterministic"] = True
    if change:
        with pytest.raises(ValueError):
            policy.install(runner)
        assert not installs
    else:
        policy.install(runner)
        assert installs == [runner]
        with pytest.raises(ValueError, match="installed once"):
            policy.install(runner)


def test_failed_install_cache_preflight_closes_receipts(tmp_path, monkeypatch):
    diagnostic, _, _, _ = serving_fixture(tmp_path, monkeypatch)
    runner = SimpleNamespace(capture_model=lambda: None)
    original = StandaloneCompiledArtifacts.load_all
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "shared-wrong-cache"))
    with pytest.raises(ValueError, match="cache binding"):
        diagnostic.install(runner)
    assert diagnostic.closed and diagnostic.summary["phase"] == "install"
    assert StandaloneCompiledArtifacts.load_all is original
    assert all(stream.closed for stream in diagnostic.streams.values())


def audited_fixture(tmp_path, monkeypatch, mode="geometry"):
    from benchmarks.kernels.audit_glm53_geometry_loader import json_lines

    diagnostic, store, _, _ = serving_fixture(tmp_path, monkeypatch, mode, count=7)

    def capture():
        path = diagnostic.private / "inductor_cache/hh/additional.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def call(): pass\n")
        PyCodeCache.load_by_key_path("additional", str(path))
        extra, _, _ = make_compiled(
            diagnostic.loader.cache / "triton/0", "new_non_target"
        )
        extra.make_launcher()

    runner = SimpleNamespace(capture_model=capture)
    diagnostic.install(runner)
    try:
        store.load_all()
        runner.capture_model()
    finally:
        diagnostic.close()
    snapshots = {
        phase: (
            json.loads(Path(row["bindings"]).read_text()),
            json.loads(Path(row["modules"]).read_text()),
        )
        for phase, row in diagnostic.summary["snapshots"].items()
    }
    return (
        dict(diagnostic.manifest, rank=0),
        sha(diagnostic.path),
        diagnostic.summary,
        snapshots,
        json_lines(diagnostic.folder / "binary-loads.jsonl"),
        json_lines(diagnostic.folder / "artifact-roots.jsonl"),
        json_lines(diagnostic.folder / "lifecycle.jsonl"),
        [json_lines(c.path) for c in diagnostic.loader.controller.controllers.values()],
        diagnostic.qualified,
    ), diagnostic


@pytest.mark.parametrize("mode", ["control", "geometry"])
def test_offline_audit_accepts_real_cpu_serving_lifecycle(tmp_path, monkeypatch, mode):
    from benchmarks.kernels.audit_glm53_geometry_serving import (
        audit_worker,
        check_worker,
    )

    records, diagnostic = audited_fixture(tmp_path, monkeypatch, mode)
    result = check_worker(*records)
    assert result["before-forward"]["graphs"] == 7
    assert result["capture-after"]["imported_modules"] == 8
    assert result["capture-after"]["target_bindings"] == 21
    report = audit_worker(diagnostic.path, diagnostic.manifest, 0)
    assert report["status"] == "complete", report


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "missing_phase",
        "count",
        "model",
        "fallback",
        "early_capture",
        "sealed_observer",
        "target",
        "non_target",
        "root_event",
        "new_source",
        "overwritten_source",
        "controller",
        "binary",
    ],
)
def test_offline_serving_audit_rejects_missing_or_changed_evidence(
    tmp_path, monkeypatch, change
):
    from benchmarks.kernels.audit_glm53_geometry_serving import check_worker

    records, _ = audited_fixture(tmp_path, monkeypatch)
    (
        manifest,
        digest,
        summary,
        snapshots,
        binaries,
        roots,
        lifecycle,
        targets,
        qualified,
    ) = copy.deepcopy(records)
    if change == "status":
        summary["status"] = "aot-qualified"
    elif change == "missing_phase":
        snapshots.pop("capture-after")
    elif change == "count":
        summary["loaded_artifacts"] = 6
    elif change == "model":
        summary["model_sha256"] = "wrong"
    elif change == "fallback":
        summary["static_bundles"][0]["loaded"] = 0
    elif change == "early_capture":
        lifecycle[1], lifecycle[3] = lifecycle[3], lifecycle[1]
    elif change == "sealed_observer":
        binaries.append(dict(event="binary_observer_sealed", rank=0, objects=8))
    elif change in ("target", "non_target"):
        row = next(
            r
            for r in snapshots["capture-after"][0]["bindings"]
            if r["target"] is (change == "target")
        )
        row["selected"][0]["config"]["num_warps"] = 1
    elif change == "root_event":
        roots.pop()
    elif change == "new_source":
        snapshots["capture-after"][1][-1]["source_sha256"] = "wrong"
    elif change == "overwritten_source":
        row = snapshots["capture-after"][1][-1]
        relative = str(Path(row["path"]).relative_to(manifest["private_namespace"]))
        manifest["original_files"][relative] = "different-original"
    elif change == "controller":
        targets.pop()
    else:
        binaries[0]["cubin_sha256"] = "wrong"
    with pytest.raises((ValueError, KeyError)):
        check_worker(
            manifest,
            digest,
            summary,
            snapshots,
            binaries,
            roots,
            lifecycle,
            targets,
            qualified,
        )


def test_integration_source_refresh_is_explicit_and_narrow(tmp_path, monkeypatch):
    from benchmarks.kernels import prepare_glm53_geometry_serving as preparation

    site, loader = tmp_path / "site.py", tmp_path / "qualified-loader.py"
    site.write_text("before")
    loader.write_text("qualified")
    before = {str(site): sha(site), str(loader): sha(loader)}
    monkeypatch.setattr(preparation, "INTEGRATION_SITES", {site})
    site.write_text("new call site")
    sources, changes = preparation.integration_sources(before, [])
    assert set(changes) == {str(site)} and sources[str(loader)] == before[str(loader)]
    loader.write_text("unexpected change")
    with pytest.raises(ValueError, match="non-integration"):
        preparation.integration_sources(before, [])


def test_prepare_three_independent_caches_and_manifest_validation(
    tmp_path, monkeypatch
):
    from benchmarks.kernels import prepare_glm53_geometry_serving as preparation
    from benchmarks.kernels.glm53_rmsnorm_geometry import SCHEMA
    from benchmarks.kernels.prepare_glm53_geometry_loader import qualified_targets
    from slimserve.rmsnorm_diagnostic import NAMESPACE
    from tests.slimserve.test_artifact_roots import serialized_fixture
    from tests.slimserve.test_geometry_loader import qualified_fixture

    old, binaries = qualified_fixture(tmp_path)
    original = Path(old["original_namespace"])
    old["targets"] = qualified_targets(old, binaries)
    old.update(
        schema=SCHEMA,
        namespace=NAMESPACE,
        sources={},
        artifact_roots={},
        expected_graphs={},
    )
    dependency = tmp_path / "runtime.py"
    dependency.write_text("# CPU-only dependency fixture\n")
    alias = tmp_path / "venv-runtime.py"
    alias.symlink_to(dependency)
    old["sources"] = {str(p): sha(p) for p in (dependency, alias)}
    for rank in range(4):
        store, graphs, _ = serialized_fixture(original, count=7, rank=rank)
        old["expected_graphs"][str(rank)] = graphs
        old["artifact_roots"][str(rank)] = discover(store, original, graphs)
    old["original_files"] = {
        str(p.relative_to(original)): sha(p) for p in original.rglob("*") if p.is_file()
    }
    ref = tmp_path / "qualified-manifest.json"
    ref.write_text(json.dumps(old))
    reference = dict(path=str(ref), sha256=sha(ref))
    graphs = {m: {str(r): reference for r in range(4)} for m in ("control", "geometry")}
    monkeypatch.setattr(
        preparation,
        "completed_evidence",
        lambda *a: (old, graphs, reference, {str(ref): sha(ref)}),
    )
    output = tmp_path / "serving"
    from benchmarks.kernels import glm53_geometry_workload as workload

    monkeypatch.setattr(workload, "reference_evidence", lambda: ({}, {}, {}, {}))
    preparation.prepare(tmp_path / "pair.json", tmp_path / "closure.json", output)
    from slimserve.campaign_sources import snapshot

    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    for label, mode in preparation.CASES:
        path = output / label / "manifest.json"
        data = json.loads(path.read_text())
        assert all(
            data["sources"][str(preparation.ROOT / p)] == digest
            for p, digest in snapshot().items()
        )
        monkeypatch.setenv(policy.FLAG, mode)
        monkeypatch.setenv(policy.MANIFEST, str(path))
        monkeypatch.setenv("VLLM_CACHE_ROOT", data["cache_root"])
        monkeypatch.setenv(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path(data["private_namespace"]) / "inductor_cache"),
        )
        actual, _ = policy.read_manifest()
        assert actual == data
        for relative, digest in old["original_files"].items():
            copy_path = Path(data["private_namespace"]) / relative
            assert sha(copy_path) == digest and not copy_path.samefile(
                original / relative
            )
    data["targets"]["0"][0]["configs"]["geometry"]["R0_BLOCK"] = 4096
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest changed"):
        policy.read_manifest()
    with pytest.raises(ValueError, match="new serving series"):
        preparation.prepare(tmp_path / "pair.json", tmp_path / "closure.json", output)


def test_source_aliases_keep_identical_receipts_but_reject_conflicts(tmp_path):
    dependency = tmp_path / "runtime.py"
    dependency.write_text("# fixture\n")
    alias = tmp_path / "alias.py"
    alias.symlink_to(dependency)
    receipts = {str(p): sha(p) for p in (dependency, alias)}
    assert policy.canonical_sources(receipts) == {str(dependency): sha(dependency)}
    receipts[str(alias)] = "different"
    with pytest.raises(ValueError, match="conflicting qualified source aliases"):
        policy.canonical_sources(receipts)


def test_real_prepared_manifest_roundtrip_with_current_source_freeze(
    tmp_path, monkeypatch
):
    """Actual v1 metadata/qualified receipts; no cache copies, GPU or model load.

    A fresh CPU fixture refreshes only the current implementation hashes and local
    case paths. It must not rewrite or reuse the terminal v1 manifest/attempt.
    """
    import subprocess

    from benchmarks.kernels.prepare_glm53_geometry_serving import CASES, ROOT

    original = (
        ROOT
        / "perf/results/2026-09-10/rmsnorm-geometry-serving-v1/control/manifest.json"
    )
    if not original.exists():
        pytest.skip("local completed campaign receipts are not distributed")
    base = json.loads(original.read_text())
    base["sources"] = {name: sha(name) for name in base["sources"]}
    base["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    runs = []
    for label, mode in CASES:
        folder = tmp_path / label
        folder.mkdir()
        cache = folder / "cache"
        data = dict(
            base,
            label=label,
            mode=mode,
            cache_root=str(cache),
            private_namespace=str(
                cache / "torch_compile_cache/torch_aot_compile" / base["namespace"]
            ),
            receipts=str(folder / "receipts"),
            worker_receipts=str(folder / "worker-receipts"),
        )
        path = folder / "manifest.json"
        path.write_text(json.dumps(data))
        runs.append(
            dict(label=label, mode=mode, manifest=str(path), manifest_sha256=sha(path))
        )
    (tmp_path / "preparation.json").write_text(
        json.dumps(
            dict(
                status="prepared",
                sources=base["sources"],
                git_commit=base["git_commit"],
                runs=runs,
            )
        )
    )
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    for row in runs:
        path = Path(row["manifest"])
        data = json.loads(path.read_text())
        monkeypatch.setenv(policy.FLAG, row["mode"])
        monkeypatch.setenv(policy.MANIFEST, str(path))
        monkeypatch.setenv("VLLM_CACHE_ROOT", data["cache_root"])
        monkeypatch.setenv(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path(data["private_namespace"]) / "inductor_cache"),
        )
        actual, checked = policy.read_manifest()
        assert actual == data and checked == path
