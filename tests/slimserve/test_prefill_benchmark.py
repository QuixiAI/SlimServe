# SPDX-License-Identifier: Apache-2.0
import argparse
import io
import itertools
import json

import pytest

from benchmarks import benchmark_glm53_prefill as bench
from slimserve import cli, registry


def events(cached=0, done=True):
    values = [
        {"choices": [{"text": "", "token_ids": []}]},
        {"choices": [{"text": "a", "token_ids": [1]}]},
        {"choices": [{"text": "b", "token_ids": [2]}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
            "metrics": {"time_to_first_token_ms": 100, "queue_time_ms": 2},
        },
    ]
    raw = b"".join(b"data: " + json.dumps(v).encode() + b"\n\n" for v in values)
    return raw + (b"data: [DONE]\n\n" if done else b"")


def mock_http(monkeypatch, raw):
    monkeypatch.setattr(
        bench.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(raw)
    )
    times = itertools.count(10, 0.1)
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))


def test_prefill_has_separate_client_engine_and_cache_evidence(monkeypatch, tmp_path):
    mock_http(monkeypatch, events())
    output = tmp_path / "request.json"
    summary = bench.request(
        "http://localhost", "glm", [4, 5, 6, 7], 2, output, "unique-salt"
    )
    assert summary["cached_tokens"] == 0
    assert summary["client_ttft_ms"] == pytest.approx(200)
    assert summary["engine_scheduled_to_first_token_ms"] == 100
    assert summary["engine_queue_ms"] == 2
    assert summary["effective_input_tokens_per_engine_ttft_second"] == 40
    row = json.loads(output.read_text())
    assert row["request"]["cache_salt"] == "unique-salt"
    assert row["request"]["prompt"] == [4, 5, 6, 7]
    assert row["status"] == "complete" and len(row["events"]) == 4


@pytest.mark.parametrize("cached", [None, 1, False, "0"])
def test_missing_or_nonzero_cache_evidence_is_retained_failure(
    monkeypatch, tmp_path, cached
):
    mock_http(monkeypatch, events(cached))
    output = tmp_path / "request.json"
    with pytest.raises(ValueError, match="cold-prefix proof"):
        bench.request("http://localhost", "glm", [4, 5, 6, 7], 2, output, "salt")
    row = json.loads(output.read_text())
    assert row["status"] == "failed"
    assert row["usage"]["prompt_tokens_details"]["cached_tokens"] == cached
    assert len(row["events"]) == 4


def test_truncated_stream_is_not_a_prefill_result(monkeypatch, tmp_path):
    mock_http(monkeypatch, events(done=False))
    output = tmp_path / "request.json"
    with pytest.raises(ValueError, match="end-of-stream"):
        bench.request("http://localhost", "glm", [4, 5, 6, 7], 2, output, "salt")
    assert json.loads(output.read_text())["status"] == "failed"


def test_warmups_and_repetitions_keep_same_ids_with_distinct_salts(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.txt"
    source.write_text("source")
    calls = []

    def request(url, model, prompt, tokens, path, salt):
        calls.append((prompt, salt, path.name))
        return {"prompt_tokens": len(prompt), "client_ttft_ms": len(calls)}

    monkeypatch.setattr(bench, "request", request)
    args = argparse.Namespace(
        source=source,
        output=tmp_path / "results",
        contexts=[4],
        warmups=1,
        repeats=2,
        output_tokens=2,
        url="url",
        model="glm",
    )
    tokenizer = argparse.Namespace(encode=lambda *a, **k: list(range(10)))
    result = bench.run(args, tokenizer)
    assert [c[0] for c in calls] == [list(range(4))] * 3
    assert len({c[1] for c in calls}) == 3
    assert result["aggregates"]["4"]["client_ttft_ms"]["median"] == 2.5
    assert [r["warmup"] for r in result["requests"]] == [True, False, False]


def test_metrics_cli_defaults_off_and_serializes_existing_engine_options(
    monkeypatch, capsys
):
    assert not cli._parser().parse_args([]).request_metrics
    # A dry-run proves both flags reach the actual resolved profile; no launch.
    monkeypatch.setattr(
        cli.hardware,
        "detect",
        lambda: argparse.Namespace(
            platform="rtx6000",
            count=4,
            memory_bytes=None,
            host_ram_bytes=188 * 1024**3,
            known=True,
            device_name="RTX PRO 6000",
        ),
    )
    captured = []
    monkeypatch.setattr(cli, "_show", captured.append)
    assert cli.main(["glm53-nvfp4-4", "--request-metrics", "--dry-run"]) == 0
    plan = captured[0]
    assert plan.engine["enable_prompt_tokens_details"] is True
    assert plan.engine["enable_per_request_metrics"] is True
    baseline = registry.resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    assert {
        k: v
        for k, v in plan.engine.items()
        if k not in ("enable_prompt_tokens_details", "enable_per_request_metrics")
    } == baseline.engine
