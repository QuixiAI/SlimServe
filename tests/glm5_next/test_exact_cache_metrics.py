# SPDX-License-Identifier: Apache-2.0
"""Cache evidence uses the exact client's existing out-of-timer snapshots."""

import io
import json
import sys

import pytest

from benchmarks import benchmark_dsv4_exact as bench


def test_snapshot_reads_once_for_both_metric_families(monkeypatch):
    calls = []
    body = b"""
vllm:spec_decode_num_drafts_total{model_name="target",engine="0"} 3
vllm:spec_decode_num_drafts_total{model_name="other",engine="0"} 99
vllm:prompt_tokens_total{model_name="target",engine="0"} 20
vllm:prompt_tokens_total{model_name="target",engine="1"} 30
"""

    def urlopen(url, **kwargs):
        calls.append(url)
        return io.BytesIO(body)

    monkeypatch.setattr(bench.urllib.request, "urlopen", urlopen)
    spec, cache = bench.metric_snapshot("http://unused/metrics", "target")
    assert calls == ["http://unused/metrics"]
    assert spec["spec_decode_drafts"] == 3
    assert cache["prompt_tokens"] == 50
    assert cache["preemptions"] is None


def test_metrics_disabled_keeps_cache_unknown(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled metrics must not perform HTTP")

    monkeypatch.setattr(bench.urllib.request, "urlopen", unexpected)
    spec, cache = bench.metric_snapshot("none", "target")
    assert all(value == 0 for value in spec.values())
    assert all(value is None for value in cache.values())
    assert bench.metric_counters("none", "target") == spec


@pytest.mark.parametrize("dump_responses", [False, True])
def test_main_snapshots_exclude_warmup_and_timed_http(
    monkeypatch, tmp_path, capsys, dump_responses
):
    source = tmp_path / "source.txt"
    source.write_text("abcdefghijklmnop")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exact",
            "--model",
            "/unused/model",
            "--served-model-name",
            "target",
            "--source",
            str(source),
            "--concurrency",
            "2",
            "--input-tokens",
            "8",
            "--output-tokens",
            "10",
            "--allow-no-spec",
        ],
    )

    response_dir = tmp_path / "responses"
    if dump_responses:
        sys.argv.extend(["--dump-responses", str(response_dir)])

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(map(ord, text))

        def decode(self, ids):
            return "".join(map(chr, ids))

    monkeypatch.setattr(bench, "get_tokenizer", lambda _: Tokenizer())
    events = []
    snapshots = iter([(16, 14, 2), (32, 28, 4)])

    def urlopen(url, **kwargs):
        events.append("metrics")
        prompts, cached, requests = next(snapshots)
        return io.BytesIO(
            (
                f'vllm:prompt_tokens_total{{model_name="target"}} {prompts}\n'
                f'vllm:prompt_tokens_cached_total{{model_name="target"}} {cached}\n'
                'vllm:num_preemptions_total{model_name="target"} 0\n'
                "vllm:request_prefill_kv_computed_tokens_count"
                f'{{model_name="target"}} {requests}\n'
                "vllm:request_prefill_kv_computed_tokens_sum"
                f'{{model_name="target"}} {prompts - cached}\n'
            ).encode()
        )

    def completion(url, model, prompt, output_tokens, *args):
        events.append("warmup" if output_tokens == 8 else "request")
        return {
            "seconds": 1.0,
            "response": {
                "usage": {"prompt_tokens": 8, "completion_tokens": output_tokens},
                "choices": [{
                    "text": "healthy output text",
                    "routed_experts": "opaque routing payload preserved verbatim",
                }],
            },
        }

    ticks = iter([10.0, 12.0])

    def timer():
        events.append("timer")
        return next(ticks)

    monkeypatch.setattr(bench.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(bench, "request_completion", completion)
    monkeypatch.setattr(bench.time, "perf_counter", timer)
    original_write = bench.Path.write_text

    def write(path, text, *args, **kwargs):
        if path.name.startswith("response_"):
            events.append("dump")
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(bench.Path, "write_text", write)
    bench.main()
    result = json.loads(capsys.readouterr().out)
    assert events == [
        "warmup",
        "warmup",
        "metrics",
        "timer",
        "request",
        "request",
        "timer",
        "metrics",
    ] + (["dump", "dump"] if dump_responses else [])
    if dump_responses:
        for index in range(2):
            response = json.loads((response_dir / f"response_{index}.json").read_text())
            assert response["choices"][0]["routed_experts"] == (
                "opaque routing payload preserved verbatim"
            )
            assert response["usage"]["completion_tokens"] == 10
    else:
        assert not response_dir.exists()
    assert result["exact"]
    assert result["aggregate_output_tps"] == 10
    assert result["cache_metrics"]["prompt_tokens"] == 16
    assert result["cache_metrics"]["cached_prompt_tokens"] == 14
    assert result["cache_metrics"]["prefill_computed_tokens"] == 2
    assert result["cache_metrics"]["prefill_requests"] == 2
    assert result["cache_metrics"]["preemptions"] == 0
    assert result["cache_metrics"]["external_prefix_hit_tokens"] is None
