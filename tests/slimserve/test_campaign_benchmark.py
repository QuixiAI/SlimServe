# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import importlib.util
import io
import json
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


@pytest.mark.parametrize(
    "options",
    [
        ["--cuda-trace-concurrency", "8"],
        ["--cuda-traces", "--cuda-trace-concurrency", "4"],
        ["--cuda-traces", "--concurrency", "8"],
    ],
)
def test_cuda_single_range_configuration_fails_before_hardware_probe(
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
    if fails:
        with pytest.raises(RuntimeError, match="owned group"):
            bench.finish_owned_run(SimpleNamespace(returncode=0), run, receipt, record)
    else:
        bench.finish_owned_run(SimpleNamespace(returncode=0), run, receipt, record)
    saved = json.loads(record.read_text())["runs"][0]
    assert saved["status"] == ("failed" if fails else "complete")
    assert saved["teardown"]["status"] == ("failed" if fails else "complete")
    assert saved["quality"] == {"passed": True}
    assert saved["prefill"] == [123]


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
