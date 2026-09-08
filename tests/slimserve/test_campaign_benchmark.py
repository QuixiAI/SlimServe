# SPDX-License-Identifier: Apache-2.0
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
