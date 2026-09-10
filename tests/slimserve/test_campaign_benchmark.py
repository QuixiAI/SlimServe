# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "glm53_campaign", directory / "benchmark_glm53_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_script_can_import_owned_kernel_helpers_without_pythonpath(tmp_path):
    root = Path(__file__).resolve().parents[2]
    helper = root / "benchmarks/kernels/check_glm53_attention_norms.py"
    code = f"""
import importlib.util, runpy, sys
assert importlib.util.find_spec('benchmarks') is None
sys.path.insert(0, {str(root / "benchmarks")!r})
runpy.run_path({str(root / "benchmarks/benchmark_glm53_campaign.py")!r})
from benchmarks.kernels import check_glm53_attention_norms
assert check_glm53_attention_norms.__file__ == {str(helper)!r}
print('direct-script-helper-import-passed')
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "direct-script-helper-import-passed" in result.stdout


@pytest.mark.parametrize(
    "active",
    [
        None,
        "SLIMSERVE_GLM53_SCORE_JOURNAL",
        "SLIMSERVE_GLM53_PROMPT_SCORE_DIAGNOSTIC",
        "SLIMSERVE_GLM53_INDEX_JOURNAL",
        "SLIMSERVE_GLM53_MODEL_JOURNAL",
        "SLIMSERVE_GLM53_MOE_JOURNAL",
        "SLIMSERVE_GLM53_CANONICAL_MOE",
        "SLIMSERVE_GLM53_STABLE_ROUTE",
        "SLIMSERVE_GLM53_STABLE_ALIGN",
        "SLIMSERVE_GLM53_NATIVE_ORDER",
        "SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER",
        "SLIMSERVE_GLM53_CANONICAL_INDEX_TIES",
        "SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED",
    ],
)
def test_observers_and_ordering_intervention_cannot_be_baselines(monkeypatch, active):
    bench = _load(monkeypatch)
    keys = (
        "SLIMSERVE_GLM53_SCORE_JOURNAL",
        "SLIMSERVE_GLM53_PROMPT_SCORE_DIAGNOSTIC",
        "SLIMSERVE_GLM53_INDEX_JOURNAL",
        "SLIMSERVE_GLM53_MODEL_JOURNAL",
        "SLIMSERVE_GLM53_MOE_JOURNAL",
        "SLIMSERVE_GLM53_CANONICAL_MOE",
        "SLIMSERVE_GLM53_STABLE_ROUTE",
        "SLIMSERVE_GLM53_STABLE_ALIGN",
        "SLIMSERVE_GLM53_NATIVE_ORDER",
        "SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER",
        "SLIMSERVE_GLM53_CANONICAL_INDEX_TIES",
        "SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    args = SimpleNamespace(routing=False, cuda_traces=False, quality_repeats=1)
    if active:
        monkeypatch.setenv(active, "1")
    assert bench.diagnostic_only(args) is bool(active)
    for key, value in (
        ("routing", True),
        ("cuda_traces", True),
        ("quality_repeats", 3),
        ("jit_monitor_verbose", True),
        ("deterministic_reductions", True),
    ):
        assert bench.diagnostic_only(SimpleNamespace(**(vars(args) | {key: value})))


@pytest.mark.parametrize("mode", ["control", "legacy", "invalid"])
def test_rmsnorm_intervention_is_never_a_baseline(monkeypatch, mode):
    bench = _load(monkeypatch)
    monkeypatch.setenv("SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC", mode)
    args = SimpleNamespace(routing=False, cuda_traces=False, quality_repeats=1)
    assert bench.diagnostic_only(args)
    sources = bench.benchmark_sources()
    assert "slimserve/rmsnorm_diagnostic.py" in sources
    assert "vllm/v1/worker/gpu_model_runner.py" in sources
    assert "benchmarks/kernels/check_glm53_cached_rmsnorm.py" in sources


@pytest.mark.parametrize("mode", ["control", "geometry", "invalid"])
def test_geometry_intervention_is_never_a_baseline(monkeypatch, mode):
    bench = _load(monkeypatch)
    monkeypatch.setenv("SLIMSERVE_GLM53_RMSNORM_GEOMETRY", mode)
    args = SimpleNamespace(routing=False, cuda_traces=False, quality_repeats=1)
    assert bench.diagnostic_only(args)
    sources = bench.benchmark_sources()
    assert "slimserve/rmsnorm_geometry.py" in sources
    assert "benchmarks/kernels/glm53_geometry_serving.py" in sources
    assert "benchmarks/kernels/glm53_artifact_roots.py" in sources


@pytest.mark.parametrize("mode", ["control", "kv", "invalid"])
def test_kv_intervention_is_never_a_baseline(monkeypatch, mode):
    bench = _load(monkeypatch)
    monkeypatch.setenv("SLIMSERVE_GLM53_KV_DIAGNOSTIC", mode)
    args = SimpleNamespace(routing=False, cuda_traces=False, quality_repeats=1)
    assert bench.diagnostic_only(args)
    sources = bench.benchmark_sources()
    assert "slimserve/kv_diagnostic.py" in sources
    assert "benchmarks/kernels/glm53_kv_serving.py" in sources
    assert "benchmarks/kernels/glm53_kv_loader.py" in sources


@pytest.mark.parametrize("mode", ["control", "correction", "invalid"])
def test_indexer_correction_is_never_a_baseline(monkeypatch, mode):
    bench = _load(monkeypatch)
    monkeypatch.setenv("SLIMSERVE_GLM53_INDEXER_CORRECTION", mode)
    args = SimpleNamespace(routing=False, cuda_traces=False, quality_repeats=1)
    assert bench.diagnostic_only(args)
    sources = bench.benchmark_sources()
    assert "slimserve/indexer_correction_diagnostic.py" in sources
    assert "benchmarks/kernels/glm53_indexer_correction_serving.py" in sources
    assert "benchmarks/kernels/glm53_indexer_correction_loader.py" in sources


@pytest.mark.parametrize("enabled", [False, True])
def test_deterministic_plan_and_server_command_agree(monkeypatch, tmp_path, enabled):
    from slimserve.hardware import Machine

    bench = _load(monkeypatch)
    monkeypatch.setenv("SLIMSERVE_GLM53_NATIVE_ORDER", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setattr(bench.hardware, "detect", lambda: Machine("rtx6000", "RTX", 4))
    monkeypatch.setattr(bench, "compatible_profile_ids", lambda _: ["glm53-nvfp4-4"])
    monkeypatch.setattr(bench, "get_tokenizer", lambda *a, **k: "tokenizer")
    monkeypatch.setattr(bench, "exact_prompts", lambda *a, **k: ["prompt"])
    monkeypatch.setattr(bench, "runtime_identity", lambda: {})
    monkeypatch.setattr(bench, "require_gpus_free", lambda *a: None)
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: "test")
    source = tmp_path / "source.txt"
    source.write_text("source")
    output = tmp_path / "run"
    args = ["campaign", "--source", str(source), "--output", str(output)]
    if enabled:
        args.append("--deterministic-reductions")
    monkeypatch.setattr(bench.sys, "argv", args)
    launched = []

    def capture(argv, **kwargs):
        launched.append(argv)
        raise RuntimeError("test stops before launching a process")

    monkeypatch.setattr(bench.subprocess, "Popen", capture)
    with pytest.raises(RuntimeError, match="test stops before launching"):
        bench.main()
    assert len(launched) == 1
    assert ("--deterministic-reductions" in launched[0]) is enabled
    receipt = json.loads((output / "summary.json").read_text())
    assert receipt["environment"]["TRITON_CACHE_DIR"] == str(tmp_path / "triton")
    assert receipt["environment"]["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "inductor"
    )
    options = receipt["plan"]["engine"]["compilation_config"].get(
        "inductor_compile_config", {}
    )
    assert (options.get("deterministic") is True) is enabled
    assert (
        "slimserve/deterministic_reductions.py"
        in receipt["benchmark_implementation_sha256"]
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--cuda-trace-concurrency", "8"],
        ["--cuda-traces", "--cuda-trace-concurrency", "4"],
        ["--cuda-traces", "--concurrency", "8"],
        ["--quality-repeats", "0"],
        ["--quality-repeats", "-1", "--quality"],
        ["--quality-repeats", "2"],
    ],
)
def test_invalid_campaign_configuration_fails_before_hardware_probe(
    monkeypatch, options
):
    bench = _load(monkeypatch)
    monkeypatch.setattr(bench.hardware, "detect", lambda: pytest.fail("hardware probe"))
    monkeypatch.setattr(
        bench.sys,
        "argv",
        ["campaign", "--source", "unused", "--output", "unused", *options],
    )
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


@pytest.mark.parametrize("repeats", [1, 3])
@pytest.mark.parametrize("failure", [None, "score", "needle"])
def test_quality_passes_preserve_every_pass_without_replacement(
    monkeypatch, tmp_path, repeats, failure
):
    bench = _load(monkeypatch)
    calls, snapshots = [], []
    run = {}
    expected_calls = min(2, repeats) if failure else repeats

    def scorer(args, tokenizer):
        calls.append(args)
        assert (
            args.pairs == 32 and args.prefix_tokens == 512 and args.score_tokens == 128
        )
        assert args.needle_contexts == [1024, 8192, 32768]
        assert args.needle_positions == [0.25, 0.75]
        assert tokenizer == "tokenizer"
        if failure == "score" and len(calls) == expected_calls:
            raise RuntimeError("scoring failed")
        return {
            "summary": {
                "all_needles_rank_first": not (
                    failure == "needle" and len(calls) == expected_calls
                ),
                "mean_text_logprob": -float(len(calls)),
            }
        }

    monkeypatch.setitem(
        bench.sys.modules, "benchmark_glm53_quality", SimpleNamespace(run=scorer)
    )
    args = SimpleNamespace(quality_repeats=repeats, source=tmp_path / "source")

    def check():
        bench.quality_passes(
            args,
            "url",
            "model",
            "model-dir",
            "tokenizer",
            tmp_path,
            run,
            lambda: snapshots.append(json.loads(json.dumps(run))),
        )

    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            check()
    else:
        check()
    assert len(calls) == expected_calls
    assert [a.output.name for a in calls] == ["quality.json"] + [
        f"quality-repeat-{i}.json" for i in range(2, expected_calls + 1)
    ]
    assert run["quality_path"] == str(tmp_path / "quality.json")
    assert len(snapshots) == expected_calls * 2
    assert snapshots[-1] == run
    assert run["quality_passes"][-1]["status"] == ("failed" if failure else "complete")
    if repeats > 1 or failure != "score":
        assert run["quality"]["mean_text_logprob"] == -1.0
    assert all(r["status"] == "complete" for r in run["quality_passes"][:-1])


@pytest.mark.parametrize("fails", [False, True])
def test_teardown_always_persists_completed_work_and_failure(
    monkeypatch, tmp_path, fails
):
    bench = _load(monkeypatch)
    run = {"status": "complete", "quality": {"passed": True}, "prefill": [123]}
    receipt = {"runs": [run]}
    record = tmp_path / "summary.json"

    def stop(process):
        if fails:
            raise RuntimeError("owned group still exists")
        return []

    monkeypatch.setattr(bench, "stop_owned", stop)
    monkeypatch.setattr(bench, "process_group_snapshot", lambda _: [])
    monkeypatch.setattr(bench, "gpu_compute_pids", lambda: [])
    if fails:
        with pytest.raises(RuntimeError, match="owned group"):
            bench.finish_owned_run(
                SimpleNamespace(pid=11, returncode=0), run, receipt, record
            )
    else:
        bench.finish_owned_run(
            SimpleNamespace(pid=11, returncode=0), run, receipt, record
        )
    saved = json.loads(record.read_text())["runs"][0]
    assert saved["status"] == ("failed" if fails else "complete")
    assert saved["teardown"]["status"] == ("failed" if fails else "complete")
    assert saved["quality"] == {"passed": True}
    assert saved["prefill"] == [123]


def test_gpu_release_waits_for_owned_driver_entries_not_foreign_work(monkeypatch):
    bench = _load(monkeypatch)
    reports = iter([[12, 99], [12, 99], [99]])
    monkeypatch.setattr(bench, "gpu_compute_pids", lambda: next(reports))
    monkeypatch.setattr(bench.time, "sleep", lambda _: None)
    evidence = {}
    bench.wait_owned_gpu_release({11, 12}, evidence)
    assert evidence["status"] == "complete"
    assert [r["owned_active_pids"] for r in evidence["samples"]] == [[12], [12], []]
    assert all(r["other_active_pids"] == [99] for r in evidence["samples"])


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(2)])
def test_startup_interrupt_marks_receipt_failed_before_teardown(
    monkeypatch, tmp_path, error
):
    bench = _load(monkeypatch)
    run = {"status": "starting", "measurements": []}
    receipt = {"status": "running", "runs": [run]}
    record = tmp_path / "summary.json"
    monkeypatch.setattr(bench, "process_group_snapshot", lambda _: [])
    monkeypatch.setattr(bench, "stop_owned", lambda _: [])
    monkeypatch.setattr(bench, "gpu_compute_pids", lambda: [])
    with pytest.raises(type(error)):
        try:
            bench.record_run_failure(run, receipt, error)
        finally:
            bench.finish_owned_run(
                SimpleNamespace(pid=11, returncode=1), run, receipt, record
            )
    saved = json.loads(record.read_text())
    assert saved["status"] == saved["runs"][0]["status"] == "failed"
    assert saved["runs"][0]["error"] == repr(error)
    assert saved["runs"][0]["teardown"]["status"] == "complete"
    assert saved["runs"][0]["measurements"] == []


def test_gpu_release_timeout_keeps_measurements_and_teardown_receipt(
    monkeypatch, tmp_path
):
    bench = _load(monkeypatch)
    zombie = {"pid": 12, "ppid": 1, "state": "Z"}
    monkeypatch.setattr(bench, "process_group_snapshot", lambda _: [zombie])
    monkeypatch.setattr(bench, "stop_owned", lambda _: [zombie])
    monkeypatch.setattr(bench, "gpu_compute_pids", lambda: [12])
    ticks = iter([0, 31])
    monkeypatch.setattr(bench.time, "monotonic", lambda: next(ticks))
    run = {"status": "complete", "measurements": [{"exact": True}]}
    receipt = {"status": "running", "runs": [run]}
    record = tmp_path / "summary.json"
    with pytest.raises(RuntimeError, match="GPU processes did not release"):
        bench.finish_owned_run(
            SimpleNamespace(pid=11, returncode=0), run, receipt, record
        )
    saved = json.loads(record.read_text())
    assert saved["status"] == "failed"
    teardown = saved["runs"][0]["teardown"]
    assert teardown["status"] == "failed"
    assert teardown["remaining_zombies"] == [zombie]
    assert teardown["gpu_release"]["samples"][0]["owned_active_pids"] == [12]
    assert saved["runs"][0]["measurements"] == [{"exact": True}]


def test_next_boot_refuses_foreign_gpu_work_and_persists_failure(monkeypatch, tmp_path):
    bench = _load(monkeypatch)
    monkeypatch.setattr(bench, "gpu_compute_pids", lambda: [99])
    receipt = {"status": "running", "runs": [{"status": "complete"}]}
    record = tmp_path / "summary.json"
    with pytest.raises(RuntimeError, match="already have compute"):
        bench.require_gpus_free(receipt, record, 2)
    saved = json.loads(record.read_text())
    assert saved["status"] == "failed"
    assert saved["blocked_before_boot"] == 2
    assert saved["runs"] == [{"status": "complete"}]


def test_gpu_pid_query_deduplicates_and_has_a_timeout(monkeypatch):
    bench = _load(monkeypatch)

    def query(argv, **kwargs):
        assert "--query-compute-apps=pid" in argv
        assert kwargs == {"text": True, "timeout": 10}
        return "12\n11\n12\n"

    monkeypatch.setattr(bench.subprocess, "check_output", query)
    assert bench.gpu_compute_pids() == [11, 12]


def test_failed_gpu_query_is_never_interpreted_as_free_hardware(monkeypatch, tmp_path):
    bench = _load(monkeypatch)

    def query():
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(bench, "gpu_compute_pids", query)
    evidence = {}
    with pytest.raises(RuntimeError, match="driver query failed"):
        bench.wait_owned_gpu_release({11}, evidence)
    assert evidence["status"] == "failed"
    receipt = {"runs": []}
    record = tmp_path / "summary.json"
    with pytest.raises(RuntimeError, match="driver query failed"):
        bench.require_gpus_free(receipt, record, 1)
    assert json.loads(record.read_text())["status"] == "failed"


@pytest.mark.parametrize("live", [False, True])
def test_stop_owned_records_zombies_but_never_accepts_live_workers(monkeypatch, live):
    bench = _load(monkeypatch)
    members = [{"pid": 12, "ppid": 99, "state": "S" if live else "Z"}]
    monkeypatch.setattr(bench, "process_group_snapshot", lambda pgid: members)
    signals = []
    monkeypatch.setattr(
        bench.os, "killpg", lambda pgid, sig: signals.append((pgid, sig))
    )
    clock = iter(range(0, 1000, 100))
    monkeypatch.setattr(bench.time, "monotonic", lambda: next(clock))
    process = SimpleNamespace(pid=11, wait=lambda timeout: 0)
    if live:
        with pytest.raises(RuntimeError, match="remaining processes"):
            bench.stop_owned(process)
        assert signals == [
            (11, signal)
            for signal in (
                bench.signal.SIGINT,
                bench.signal.SIGTERM,
                bench.signal.SIGKILL,
            )
        ]
    else:
        assert bench.stop_owned(process) == members
        assert signals == [(11, bench.signal.SIGINT)]


def test_process_group_snapshot_keeps_zombies_and_filters_ownership(
    monkeypatch, tmp_path
):
    bench = _load(monkeypatch)
    for pid, comm, state, ppid, pgid in [
        (11, "name with ) parens", "S", 10, 11),
        (12, "worker", "Z", 99, 11),
        (13, "unrelated", "R", 10, 13),
    ]:
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {pgid} 0 0")
    (tmp_path / "self").mkdir()
    (tmp_path / "14").mkdir()  # Exited between listing and reading stat.
    monkeypatch.setattr(bench, "Path", lambda path: tmp_path)
    assert bench.process_group_snapshot(11) == [
        {"pid": 11, "ppid": 10, "state": "S"},
        {"pid": 12, "ppid": 99, "state": "Z"},
    ]


@pytest.mark.parametrize("first_tokens", [1, 2])
@pytest.mark.parametrize("cache_salt", [None, "isolated-request"])
def test_stream_counts_token_ids_not_text_events(monkeypatch, first_tokens, cache_salt):
    bench = _load(monkeypatch)
    chunks = [
        {"choices": [{"text": "", "token_ids": []}]},
        {"choices": [{"text": "hello", "token_ids": list(range(first_tokens))}]},
        {"choices": [{"text": " world", "token_ids": list(range(first_tokens, 3))}]},
        {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 3}},
    ]
    content = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    content += b"data: [DONE]\n\n"
    bodies = []

    def urlopen(req, **kwargs):
        bodies.append(json.loads(req.data))
        return io.BytesIO(content)

    monkeypatch.setattr(bench.urllib.request, "urlopen", urlopen)
    times = iter([10.0, 11.0, 13.0, 14.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))
    result = bench.request(
        "http://localhost", "test", "prompt", 3, 42, cache_salt=cache_salt
    )
    assert result["ttft_seconds"] == 1.0
    assert result["decode_seconds"] == 2.0
    assert result["tokens_after_first_chunk"] == 3 - first_tokens
    assert result["end"] - result["start"] == 4.0
    assert result["text"] == "hello world"
    assert bodies[0].get("cache_salt") == cache_salt
    assert ("cache_salt" in bodies[0]) == (cache_salt is not None)
    assert result["cache_salt"] == cache_salt


def test_truncated_stream_is_a_failure_not_a_fast_result(monkeypatch):
    bench = _load(monkeypatch)
    monkeypatch.setattr(
        bench.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"data: [DONE]\n\n")
    )
    with pytest.raises(ValueError, match="incomplete stream"):
        bench.request("http://localhost", "test", "prompt", 3, 42)


def test_native_receipt_includes_moe_and_allocator(monkeypatch, tmp_path):
    bench = _load(monkeypatch)
    directory = tmp_path / "vllm"
    directory.mkdir()
    names = [
        "_C_stable_libtorch.abi3.so",
        "_moe_C_stable_libtorch.abi3.so",
        "cumem_allocator.abi3.so",
    ]
    for name in names:
        (directory / name).write_bytes(name.encode())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bench, "gpu_snapshot", lambda: "test GPU")
    receipt = bench.runtime_identity()
    assert receipt["native_sha256"] == {
        f"vllm/{name}": hashlib.sha256(name.encode()).hexdigest() for name in names
    }


@pytest.mark.parametrize("bad_counts", [False, True])
@pytest.mark.parametrize("cold", [False, True])
def test_observer_return_repeats_same_matrix_and_retains_failed_raw(
    monkeypatch, tmp_path, bad_counts, cold
):
    bench = _load(monkeypatch)
    calls = []

    def round_requests(base, model, prompts, tokens, seed, cold_prefix=False):
        assert cold_prefix == cold
        calls.append((prompts, tokens, seed))
        return {
            "aggregate_output_tps": 123.0,
            "requests": [
                {
                    "usage": {
                        "prompt_tokens": 999 if bad_counts else 1000,
                        "completion_tokens": tokens,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                    "replacement_characters": 0,
                }
            ],
        }

    monkeypatch.setattr(bench, "round_requests", round_requests)
    monkeypatch.setattr(bench, "gpu_snapshot", lambda: "gpu")
    args = argparse.Namespace(
        repeats=3,
        concurrency=[1, 8],
        input_tokens=1000,
        output_tokens=300,
        cold_prefix=cold,
    )
    prompts = {1: ["a"], 8: ["b"] * 8}
    if bad_counts:
        with pytest.raises(ValueError, match="observer return token-count"):
            bench.observer_return_rounds(args, "url", "model", prompts, tmp_path)
        raw = json.loads((tmp_path / "observer-return-1-c1.json").read_text())
        assert raw["exact"] is False
        assert len(calls) == 1
    else:
        rows = bench.observer_return_rounds(args, "url", "model", prompts, tmp_path)
        assert len(rows) == 6
        assert calls == [(prompts[c], 300, 42) for _ in range(3) for c in (1, 8)]
        assert len(list(tmp_path.glob("observer-return-*.json"))) == 6


@pytest.mark.parametrize(
    "other", [["--traces"], ["--routing"], ["--prefill", "--prefill-traces"]]
)
def test_cuda_trace_rejects_mixed_observers_before_hardware_probe(monkeypatch, other):
    bench = _load(monkeypatch)
    monkeypatch.setattr(
        bench.hardware, "detect", lambda: pytest.fail("unexpected hardware probe")
    )
    monkeypatch.setattr(
        bench.sys,
        "argv",
        [
            "campaign",
            "--source",
            "unused",
            "--output",
            "unused",
            "--cuda-traces",
            *other,
        ],
    )
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_source_receipt_is_frozen_and_rejects_edits(monkeypatch):
    bench = _load(monkeypatch)
    original = bench.require_benchmark_sources()
    assert "slimserve/stream.py" in original
    original["slimserve/stream.py"] = "edited"
    assert bench.require_benchmark_sources()["slimserve/stream.py"] != "edited"
    monkeypatch.setattr(bench, "benchmark_sources", lambda: original)
    with pytest.raises(RuntimeError, match="source changed after import"):
        bench.require_benchmark_sources()


@pytest.mark.parametrize("cached", [0, 1, None, False, "0"])
def test_cold_prefix_requires_explicit_integer_zero(monkeypatch, cached):
    bench = _load(monkeypatch)
    result = {
        "requests": [{"usage": {"prompt_tokens_details": {"cached_tokens": cached}}}]
    }
    assert bench.cold_prefix_verified(result) == (type(cached) is int and cached == 0)
    assert not bench.cold_prefix_verified({"requests": []})


def test_cold_rounds_isolate_each_request_without_changing_prompts(monkeypatch):
    bench = _load(monkeypatch)
    salts = iter(["first-round", "second-round"])
    monkeypatch.setattr(bench.secrets, "token_hex", lambda _: next(salts))
    calls = []

    def request(base, model, prompt, tokens, seed, event, cache_salt):
        calls.append((prompt, tokens, seed, cache_salt))
        return {
            "first": 1,
            "last": 2,
            "tokens_after_first_chunk": 299,
            "ttft_seconds": 0.1,
            "usage": {"completion_tokens": 300},
        }

    monkeypatch.setattr(bench, "request", request)
    for _ in range(2):
        row = bench.round_requests(
            "url", "model", ["a", "b"], 300, 42, cold_prefix=True
        )
        assert row["cache_policy"] == "isolated-cold"
    assert sorted(calls) == sorted(
        [
            (prompt, 300, 42 + i, f"{salt}:{i}")
            for salt in ("first-round", "second-round")
            for i, prompt in enumerate(("a", "b"))
        ]
    )
