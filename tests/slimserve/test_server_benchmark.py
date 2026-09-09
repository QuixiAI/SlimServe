# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def bench(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "glm53_server", directory / "benchmark_glm53_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenizer_dir(path, **changes):
    path.mkdir()
    data = {"model": {"vocab": {"hello": 0}}, "normalizer": None, "truncation": None}
    data.update(changes)
    (path / "tokenizer.json").write_text(json.dumps(data))
    return path


class Tokenizer:
    def __init__(self, token=1, count=131072):
        self.token, self.count = token, count

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [self.token] * self.count


def test_serialized_defaults_require_actual_source_equality(bench, tmp_path):
    a = tokenizer_dir(tmp_path / "a", truncation={"max_length": 2048})
    b = tokenizer_dir(tmp_path / "b")
    _, result = bench.tokenizer_receipt(a, b, "source", load=lambda _: Tokenizer())
    assert result["source_tokens"] == 131072
    assert result["reference"]["serialized_defaults"]["truncation"] == {
        "max_length": 2048
    }
    assert result["semantic_components_equal"]
    with pytest.raises(ValueError, match="different source IDs"):
        bench.tokenizer_receipt(
            a, b, "source", load=lambda p: Tokenizer(token=1 if p == str(a) else 2)
        )
    with pytest.raises(ValueError, match="untruncated"):
        bench.tokenizer_receipt(a, b, "source", load=lambda _: Tokenizer(count=2048))


def test_matching_vocab_does_not_hide_normalizer_difference(bench, tmp_path):
    a = tokenizer_dir(tmp_path / "a")
    b = tokenizer_dir(tmp_path / "b", normalizer={"type": "NFC"})
    with pytest.raises(ValueError, match="semantic components"):
        bench.tokenizer_receipt(
            a, b, "source", load=lambda _: pytest.fail("must reject before loading")
        )


def result_for(c):
    return {
        "aggregate_output_tps": 100.0,
        "client_decode_tps": 110.0,
        "requests": [
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 300,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "token_ids": list(range(300)),
                "replacement_characters": 0,
            }
            for _ in range(c)
        ],
    }


@pytest.mark.parametrize(
    "defect", ["prompt", "output", "stream", "replacement", "count"]
)
def test_incomplete_round_never_becomes_fast_result(bench, defect):
    result = result_for(1)
    row = result["requests"][0]
    if defect == "prompt":
        row["usage"]["prompt_tokens"] = 999
    elif defect == "output":
        row["usage"]["completion_tokens"] = 299
    elif defect == "stream":
        row["token_ids"].pop()
    elif defect == "replacement":
        row["replacement_characters"] = 1
    else:
        result["requests"] = []
    with pytest.raises(ValueError):
        bench.check_round(result, 1)


@pytest.mark.parametrize("failure", [None, "counts", "image", "cache", "canary-only"])
def test_fixed_workload_uses_shared_function_and_retains_failure(
    bench, monkeypatch, tmp_path, failure
):
    source = tmp_path / "source.txt"
    source.write_text("source")
    args = argparse.Namespace(
        url="http://127.0.0.1:8123",
        model="reference",
        source=source,
        reference_tokenizer=tmp_path,
        tokenizer=tmp_path,
        output=tmp_path / "run",
        repeats=3,
        quality=False,
        prefill=False,
        cold_prefix=failure == "cache",
        canary_only=failure == "canary-only",
    )
    monkeypatch.setattr(bench, "tokenizer_receipt", lambda *a: (Tokenizer(), {}))
    monkeypatch.setattr(
        bench, "exact_prompts", lambda tok, src, c, n, off, repeat: ["prompt"] * c
    )

    def chat(url, model, messages, **kwargs):
        image = isinstance(messages[0]["content"], list)
        answer = ("RedRed" if failure == "image" else "red") if image else "4"
        kwargs["on_event"]({"choices": [{"delta": {"content": answer}}]})
        return [answer]

    monkeypatch.setattr(bench, "chat_completion", chat)
    monkeypatch.setattr(bench, "gpu_snapshot", lambda: "no GPU used")
    calls = []

    def fake_round(url, model, prompts, output, seed, cold_prefix=False):
        assert cold_prefix == (failure == "cache")
        calls.append((len(prompts), output, seed))
        result = result_for(len(prompts))
        if failure == "counts" and len(calls) == 4:
            result["requests"][0]["usage"]["completion_tokens"] = 299
        if failure == "cache" and len(calls) == 4:
            result["requests"][0]["usage"]["prompt_tokens_details"]["cached_tokens"] = (
                1000
            )
        return result

    monkeypatch.setattr(bench, "round_requests", fake_round)
    if failure == "image":
        with pytest.raises(RuntimeError, match="image check failed"):
            bench.run(args)
        receipt = json.loads((args.output / "summary.json").read_text())
        assert receipt["status"] == "failed"
        assert (
            receipt["canaries"]["image"]["response_events"][0]["choices"][0]["delta"][
                "content"
            ]
            == "RedRed"
        )
        assert not calls
    elif failure in ("counts", "cache"):
        with pytest.raises(
            ValueError, match="cold-prefix" if failure == "cache" else "exact-token"
        ):
            bench.run(args)
        receipt = json.loads((args.output / "summary.json").read_text())
        assert receipt["status"] == "failed"
        assert len(receipt["measurements"]) == 1
        assert (args.output / "repeat-1-c1.json").is_file()
        assert len(calls) == 4  # No retry or selection of a later successful round.
    else:
        receipt = bench.run(args)
        assert receipt["status"] == "complete"
        if failure == "canary-only":
            assert not calls and "aggregates" not in receipt
            assert receipt["diagnostic_only"]
        else:
            assert calls == [(c, 300, 42) for _ in range(4) for c in (1, 8, 16)]
            assert all(row["count"] == 3 for row in receipt["aggregates"].values())
