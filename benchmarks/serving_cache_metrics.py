# SPDX-License-Identifier: Apache-2.0
"""Optional cache/recompute evidence for exact-token serving measurements.

Missing counters are unknown, not zero. Sum across the selected model's
engine labels; these deltas require an otherwise idle serving endpoint.
"""

import math

from prometheus_client.parser import text_string_to_metric_families

CACHE_METRICS = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "cached_prompt_tokens": "vllm:prompt_tokens_cached_total",
    "prefix_query_tokens": "vllm:prefix_cache_queries_total",
    "prefix_hit_tokens": "vllm:prefix_cache_hits_total",
    "external_prefix_hit_tokens": "vllm:external_prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
    "prefill_computed_tokens": "vllm:request_prefill_kv_computed_tokens_sum",
    "prefill_requests": "vllm:request_prefill_kv_computed_tokens_count",
}


def parse_cache_metrics(body: str, model: str) -> dict[str, float | None]:
    result = dict.fromkeys(CACHE_METRICS)
    reverse = {name: key for key, name in CACHE_METRICS.items()}
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            key = reverse.get(sample.name)
            if key is not None and sample.labels.get("model_name") == model:
                if not math.isfinite(sample.value) or sample.value < 0:
                    raise ValueError(f"invalid cache metric counter: {key}")
                result[key] = (result[key] or 0.0) + sample.value
    return result


def cache_metric_delta(before, after) -> dict[str, float | None]:
    result = {}
    for key in CACHE_METRICS:
        first, last = before.get(key), after.get(key)
        result[key] = None if first is None or last is None else last - first
        if result[key] is not None and result[key] < 0:
            raise ValueError(f"cache metric counter reset during measurement: {key}")
    return result
