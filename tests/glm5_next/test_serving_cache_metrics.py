# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.serving_cache_metrics import cache_metric_delta, parse_cache_metrics


def test_cache_metrics_aggregate_only_selected_model():
    body = """
vllm:prompt_tokens_total{engine="0",model_name="target"} 100
vllm:prompt_tokens_total{engine="1",model_name="target"} 200
vllm:prompt_tokens_total{engine="0",model_name="other"} 10000
vllm:prompt_tokens_cached_total{engine="0",model_name="target"} 280
vllm:num_preemptions_total{engine="0",model_name="target"} 0
vllm:request_prefill_kv_computed_tokens_count{model_name="target"} 2
vllm:request_prefill_kv_computed_tokens_sum{model_name="target"} 20
vllm:request_prefill_kv_computed_tokens_bucket{model_name="target",le="+Inf"} 2
"""
    result = parse_cache_metrics(body, "target")
    assert result["prompt_tokens"] == 300
    assert result["cached_prompt_tokens"] == 280
    assert result["preemptions"] == 0
    assert result["prefill_requests"] == 2
    assert result["prefill_computed_tokens"] == 20
    assert result["external_prefix_hit_tokens"] is None


def test_missing_metrics_are_unknown_not_zero():
    empty = parse_cache_metrics("", "target")
    assert all(value is None for value in empty.values())
    assert all(value is None for value in cache_metric_delta(empty, {}).values())


def test_delta_preserves_unknown_and_zero():
    before = {"prompt_tokens": 100, "cached_prompt_tokens": 80, "preemptions": 3}
    after = {"prompt_tokens": 300, "cached_prompt_tokens": 270, "preemptions": 3}
    delta = cache_metric_delta(before, after)
    assert delta["prompt_tokens"] == 200
    assert delta["cached_prompt_tokens"] == 190
    assert delta["preemptions"] == 0
    assert delta["prefill_requests"] is None


def test_counter_reset_is_not_a_negative_workload():
    with pytest.raises(ValueError, match="counter reset"):
        cache_metric_delta({"prompt_tokens": 10}, {"prompt_tokens": 1})


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1"])
def test_invalid_counters_fail_closed(value):
    body = f'vllm:prompt_tokens_total{{model_name="target"}} {value}\n'
    with pytest.raises(ValueError, match="invalid cache metric counter"):
        parse_cache_metrics(body, "target")
