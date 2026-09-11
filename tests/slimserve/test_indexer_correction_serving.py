# SPDX-License-Identifier: Apache-2.0
"""CPU integration tests; CUDA allocation/driver/stream queries are simulated."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch

from benchmarks.kernels import glm53_indexer_correction_serving as serving
from benchmarks.kernels import prepare_glm53_indexer_correction_serving as preparation
from benchmarks.kernels.audit_glm53_geometry_loader import json_lines
from benchmarks.kernels.audit_glm53_geometry_serving import audit_worker
from benchmarks.kernels.glm53_indexer_correction_loader import (
    BoundIndexerCorrection,
    IndexerCorrectionIntervention,
)
from slimserve import glm53_ordering
from slimserve import indexer_correction_diagnostic as policy
from slimserve.glm53_serving_diagnostic import cases
from slimserve.glm53_serving_diagnostic import policy as serving_policy
from slimserve.registry import resolve
from tests.slimserve.test_indexer_correction_loader import correction_fixture
from tests.slimserve.test_kv_serving import serving_fixture


def runner_fixture():
    return NS(
        max_num_tokens=8192,
        scheduler_config=NS(max_num_batched_tokens=8192),
        cudagraph_batch_sizes=[1, 2, 4, 8, 16, 32, 64],
        parallel_config=NS(
            data_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            rank=0,
        ),
        compilation_config=NS(
            pass_config=NS(enable_sp=False), inductor_compile_config={}
        ),
        model_config=NS(hf_text_config=NS(hidden_size=4096, num_hidden_layers=45)),
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        speculative_config=None,
    )


def runtime_fixture(monkeypatch):
    diagnostic = serving.ServingIndexerCorrection.__new__(
        serving.ServingIndexerCorrection
    )
    diagnostic.summary = dict(
        status="installed",
        runtime_envelope=serving.runtime_envelope(runner_fixture(), 8192),
    )
    diagnostic.owner_thread = diagnostic.last_stream = diagnostic.live_stream = None
    diagnostic.observed = set()
    events, barriers = [], []
    diagnostic.emit = lambda kind, row: events.append(row)
    stream = NS(cuda_stream=123)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: barriers.append(stream.cuda_stream)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return diagnostic, events, barriers, stream


def result(padded, ubatch=False, dp=None):
    return None, NS(num_tokens=padded), ubatch, dp, None


@pytest.mark.parametrize(
    "change",
    ["capacity", "maximum", "scheduler", "capture", "dp", "dcp", "ubatch", "sp"],
)
def test_actual_runtime_configuration_rejected(change):
    runner, capacity = runner_fixture(), 8192
    if change == "capacity":
        capacity = 16384
    elif change == "maximum":
        runner.max_num_tokens = 16384
    elif change == "scheduler":
        runner.scheduler_config.max_num_batched_tokens = 4096
    elif change == "capture":
        runner.cudagraph_batch_sizes.append(8193)
    elif change == "dp":
        runner.parallel_config.data_parallel_size = 2
    elif change == "dcp":
        runner.parallel_config.decode_context_parallel_size = 2
    elif change == "ubatch":
        runner.parallel_config.use_ubatching = True
    else:
        runner.compilation_config.pass_config.enable_sp = True
    with pytest.raises(ValueError, match="bounded serialized"):
        serving.runtime_envelope(runner, capacity)


@pytest.mark.parametrize(
    "tokens,padded,ubatch,dp",
    [
        (8192, 8193, False, None),
        (0, 1, False, None),
        (2, 1, False, None),
        (True, 1, False, None),
        (1, 1, True, None),
        (1, 1, False, []),
    ],
)
def test_real_padding_output_is_checked_before_forward(
    monkeypatch, tokens, padded, ubatch, dp
):
    diagnostic, events, _, _ = runtime_fixture(monkeypatch)
    with pytest.raises(ValueError, match="actual padded"):
        diagnostic.observe_padding(tokens, result(padded, ubatch, dp))
    assert not events


def test_stream_transitions_serialize_and_live_stream_is_fixed(monkeypatch):
    diagnostic, events, barriers, stream = runtime_fixture(monkeypatch)
    diagnostic.observe_padding(3, result(4))
    diagnostic.observe_padding(3, result(4))
    assert len(events) == 1 and not barriers
    stream.cuda_stream = 456
    diagnostic.observe_padding(3, result(4))
    assert barriers == [456] and events[-1]["transition_barrier"]
    diagnostic.summary["status"] = "capture-qualified"
    stream.cuda_stream = 123
    diagnostic.observe_padding(1, result(1))
    assert barriers == [456, 123]
    stream.cuda_stream = 789
    with pytest.raises(ValueError, match="live forward stream"):
        diagnostic.observe_padding(1, result(1))
    monkeypatch.setattr(serving.threading, "get_ident", lambda: -1)
    with pytest.raises(ValueError, match="concurrent worker thread"):
        diagnostic.observe_padding(1, result(1))


def test_no_transition_barrier_can_be_inserted_inside_capture(monkeypatch):
    diagnostic, _, barriers, stream = runtime_fixture(monkeypatch)
    diagnostic.observe_padding(1, result(1))
    stream.cuda_stream = 456
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="transition inside capture"):
        diagnostic.observe_padding(1, result(1))
    assert not barriers


class SimulatedCudaTensor(torch.Tensor):
    """CPU storage reporting the rank device only for plumbing tests."""

    @property
    def device(self):
        return torch.device("cuda:0")


def native_fixture(tmp_path, monkeypatch, mode):
    fixture = correction_fixture(tmp_path, monkeypatch, mode)

    def adapter(controller, launcher, pair):
        storage = torch.full((8194, 128), 171, dtype=torch.uint8).as_subclass(
            SimulatedCudaTensor
        )
        instance = BoundIndexerCorrection(launcher, pair[1], storage, 8192)
        return instance, instance.run

    monkeypatch.setattr(IndexerCorrectionIntervention, "make_adapter", adapter)
    return fixture


@pytest.mark.parametrize("mode", ["control", "correction"])
def test_real_torch_aot_capture_and_independent_worker_audit(
    tmp_path, monkeypatch, mode
):
    runtime_fixture(monkeypatch)
    diagnostic, store, _ = serving_fixture(
        tmp_path,
        monkeypatch,
        mode,
        kernel_fixture=native_fixture,
        serving_type=serving.ServingIndexerCorrection,
        serving_policy=policy,
        separate_owners=True,
    )
    runner = runner_fixture()
    runner._determine_batch_execution_and_padding = lambda num_tokens: result(
        num_tokens
    )

    def capture():
        assert diagnostic.loader.controller.sealed
        runner._determine_batch_execution_and_padding(num_tokens=64)
        return 42

    runner.capture_model = capture
    diagnostic.install(runner)
    try:
        store.load_all()
        runner._determine_batch_execution_and_padding(8192)
        assert runner.capture_model() == 42
        runner._determine_batch_execution_and_padding(1)
    finally:
        diagnostic.close()
    assert all(s.closed for s in diagnostic.streams.values())
    assert runner.capture_model is capture
    assert all(
        r["event"].startswith("binary_")
        for r in json_lines(diagnostic.folder / "binary-loads.jsonl")
    )
    assert all(
        r["event"].startswith("indexer_correction_")
        for r in json_lines(diagnostic.folder / "loader-events.jsonl")
    )
    report = audit_worker(diagnostic.path, diagnostic.manifest, 0)
    assert report["status"] == "complete", report
    assert report["runtime_envelope"]["max_padded"] == 8192
    assert report["runtime_envelope"]["arena_bindings"] == (
        2 if mode == "correction" else 0
    )
    assert report["phases"]["capture-after"]["appended_launchers"] == (
        2 if mode == "correction" else 0
    )


@pytest.mark.parametrize(
    "mutation",
    [None, "barrier", "padding", "thread", "stream", "arena", "coverage", "phase"],
)
def test_runtime_audit_rejects_mutated_receipts(monkeypatch, mutation):
    diagnostic, events, _, stream = runtime_fixture(monkeypatch)
    diagnostic.observe_padding(8192, result(8192))
    stream.cuda_stream = 456
    diagnostic.summary["status"] = "capture-qualified"
    diagnostic.observe_padding(1, result(1))
    diagnostic.observe_padding(3, result(4))
    summary = dict(
        diagnostic.summary,
        arena_snapshots={
            p: [
                dict(address=1234, bytes=8194 * 128),
                dict(address=5678, bytes=8194 * 128),
            ]
            for p in ("before-forward", "capture-before", "capture-after")
        },
    )
    if mutation == "barrier":
        events[1]["transition_barrier"] = False
    elif mutation == "padding":
        events[0]["padded"] = 8193
    elif mutation == "thread":
        events[-1]["thread"] = -1
    elif mutation == "stream":
        events[-1]["stream"] = 789
    elif mutation == "arena":
        summary["arena_snapshots"]["capture-after"][0]["address"] += 1
    elif mutation == "coverage":
        events[:] = events[:1]
    elif mutation == "phase":
        events[-1]["phase"] = "startup"
    if mutation is None:
        assert serving.check_runtime_records(
            dict(mode="correction", selection_capacity=8192), summary, events
        )["single_live_stream"]
    else:
        with pytest.raises(ValueError):
            serving.check_runtime_records(
                dict(mode="correction", selection_capacity=8192), summary, events
            )


def test_disabled_policy_is_inert(monkeypatch):
    monkeypatch.delenv(policy.FLAG, raising=False)
    policy.install(object())
    policy.validate_plan(object())


@pytest.mark.parametrize(
    "conflict",
    [
        "SLIMSERVE_GLM53_KV_DIAGNOSTIC",
        "SLIMSERVE_GLM53_RMSNORM_GEOMETRY",
        "SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC",
        "TORCHINDUCTOR_DETERMINISTIC",
    ],
)
def test_conflicting_diagnostics_rejected(monkeypatch, conflict):
    monkeypatch.setenv(policy.FLAG, "correction")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setenv(conflict, "1")
    with pytest.raises(ValueError):
        policy.validate_environment()


@pytest.mark.parametrize(
    "change",
    [
        None,
        "gpu",
        "dtype",
        "kv",
        "model",
        "tp",
        "pp",
        "ep",
        "speculation",
        "compiler",
        "rows",
    ],
)
def test_worker_admission_and_single_install(monkeypatch, tmp_path, change):
    monkeypatch.setenv(policy.FLAG, "correction")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda: (8, 0) if change == "gpu" else (12, 0),
    )
    monkeypatch.setattr(
        policy,
        "read_manifest",
        lambda: ({"selection_capacity": 8192}, tmp_path / "manifest.json"),
    )
    installed = []
    monkeypatch.setattr(
        serving,
        "ServingIndexerCorrection",
        lambda *a: NS(install=lambda runner: installed.append(runner)),
    )
    runner = runner_fixture()
    if change == "dtype":
        runner.dtype = torch.float16
    elif change == "kv":
        runner.kv_cache_dtype = torch.float16
    elif change == "model":
        runner.model_config.hf_text_config.hidden_size = 2048
    elif change == "tp":
        runner.parallel_config.tensor_parallel_size = 2
    elif change == "pp":
        runner.parallel_config.pipeline_parallel_size = 2
    elif change == "ep":
        runner.parallel_config.enable_expert_parallel = True
    elif change == "speculation":
        runner.speculative_config = object()
    elif change == "compiler":
        runner.compilation_config.inductor_compile_config["combo_kernels"] = False
    elif change == "rows":
        runner.max_num_tokens = 16384
    if change:
        with pytest.raises(ValueError):
            policy.install(runner)
        assert not installed
    else:
        policy.install(runner)
        assert installed == [runner]
        with pytest.raises(ValueError, match="once"):
            policy.install(runner)


@pytest.mark.parametrize("change", [None, "dtype", "kv", "combo", "deterministic"])
def test_plan_preserves_recipe_and_compiler(monkeypatch, change):
    monkeypatch.setenv(policy.FLAG, "correction")
    monkeypatch.setenv(glm53_ordering.FLAG, "1")
    monkeypatch.setenv("VLLM_FORCE_AOT_LOAD", "1")
    plan = copy.deepcopy(resolve("glm53-nvfp4-4", "rtx6000", 4, None))
    seen = []
    monkeypatch.setattr(policy, "read_manifest", lambda: seen.append(True))
    if change == "dtype":
        plan.engine["dtype"] = "float16"
    elif change == "kv":
        plan.engine["kv_cache_dtype"] = "fp8"
    elif change in ("combo", "deterministic"):
        plan.engine["compilation_config"]["inductor_compile_config"] = (
            {"combo_kernels": False} if change == "combo" else {"deterministic": True}
        )
    if change:
        with pytest.raises(ValueError):
            policy.validate_plan(plan)
        assert not seen
    else:
        policy.validate_plan(plan)
        assert seen == [True]


def test_completed_evidence_joins_every_bound_leaf_without_expired_reader():
    if not preparation.PAIR.exists():
        pytest.skip("campaign receipts unavailable")
    base, graphs, reference, receipts = preparation.completed_evidence(preparation.PAIR)
    assert base["selection_capacity"] == 8192
    assert len(base["qualified_leaf_cases"]) == 120
    assert set(graphs) == {"control", "correction"}
    assert all(len(ranks) == 4 for ranks in graphs.values())
    assert reference["path"] in receipts
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mode", ["control", "correction"])
def test_schema_and_case_dispatch(mode):
    manifest = dict(serving_schema=policy.SERVING_SCHEMA, mode=mode)
    assert serving_policy(manifest) is policy
    assert cases(manifest) == policy.CASES
