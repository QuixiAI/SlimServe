# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

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


@pytest.mark.parametrize("first_tokens", [1, 2])
def test_stream_counts_token_ids_not_text_events(monkeypatch, first_tokens):
    bench = _load(monkeypatch)
    chunks = [
        {"choices": [{"text": "", "token_ids": []}]},
        {"choices": [{"text": "hello", "token_ids": list(range(first_tokens))}]},
        {"choices": [{"text": " world", "token_ids": list(range(first_tokens, 3))}]},
        {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 3}},
    ]
    content = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    content += b"data: [DONE]\n\n"
    monkeypatch.setattr(
        bench.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content)
    )
    times = iter([10.0, 11.0, 13.0, 14.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))
    result = bench.request("http://localhost", "test", "prompt", 3, 42)
    assert result["ttft_seconds"] == 1.0
    assert result["decode_seconds"] == 2.0
    assert result["tokens_after_first_chunk"] == 3 - first_tokens
    assert result["end"] - result["start"] == 4.0
    assert result["text"] == "hello world"


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
def test_observer_return_repeats_same_matrix_and_retains_failed_raw(
    monkeypatch, tmp_path, bad_counts
):
    bench = _load(monkeypatch)
    calls = []

    def round_requests(base, model, prompts, tokens, seed):
        calls.append((prompts, tokens, seed))
        return {
            "aggregate_output_tps": 123.0,
            "requests": [
                {
                    "usage": {
                        "prompt_tokens": 999 if bad_counts else 1000,
                        "completion_tokens": tokens,
                    },
                    "replacement_characters": 0,
                }
            ],
        }

    monkeypatch.setattr(bench, "round_requests", round_requests)
    monkeypatch.setattr(bench, "gpu_snapshot", lambda: "gpu")
    args = argparse.Namespace(
        repeats=3, concurrency=[1, 8], input_tokens=1000, output_tokens=300
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
