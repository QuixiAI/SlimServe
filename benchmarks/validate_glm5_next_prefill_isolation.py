# SPDX-License-Identifier: Apache-2.0
"""Distinct-record cached-prefix recall through the real registered profile.

This is an end-to-end correctness check, not a throughput benchmark. Aggregate
prefix-hit counters establish cache reuse, not exact scheduler chunk membership;
test_prefill_query_row_map.py separately forces that metadata/selection case.
"""

import argparse
import concurrent.futures
import json
import threading
import urllib.request
from pathlib import Path

from benchmarks.serving_cache_metrics import cache_metric_delta, parse_cache_metrics
from vllm.tokenizers.registry import get_tokenizer


def check_reply(response, marker, other_markers):
    content = response["choices"][0]["message"].get("content") or ""
    assert marker in content, (marker, content)
    assert not any(other in content for other in other_markers), content


def aligned_prime_ids(tokenizer, question):
    # This installed tokenizer returns BatchEncoding for tokenize=True, not
    # a flat token list. Render text explicitly and encode without adding a
    # second set of special tokens around the rendered chat template.
    rendered = tokenizer.apply_chat_template(
        [dict(role="user", content=question)], tokenize=False,
        add_generation_prompt=True, thinking=True, enable_thinking=True,
    )
    assert isinstance(rendered, str)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    assert len(ids) > 9217
    return ids[:9217]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8400")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 16, 32])
    args = parser.parse_args()
    assert min(args.concurrency) > 0
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = get_tokenizer(args.tokenizer)
    source = (Path(__file__).parent / "sonnet.txt").read_text()
    source_ids = tokenizer.encode(source)
    assert source_ids
    filler = tokenizer.decode((source_ids * ((11999 // len(source_ids)) + 1))[:12000])
    assert len(tokenizer.encode(filler)) >= 11500

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
            return parse_cache_metrics(response.read().decode(), args.model)

    def batch(prompts, limit, *, raw_prefix=False):
        barrier = threading.Barrier(len(prompts))

        def submit(prompt):
            body = dict(model=args.model, max_tokens=limit, temperature=0, seed=42)
            if raw_prefix:
                body["prompt"] = prompt
                endpoint = "/v1/completions"
            else:
                body["messages"] = [dict(role="user", content=prompt)]
                endpoint = "/v1/chat/completions"
            request = urllib.request.Request(
                args.base_url + endpoint, json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            barrier.wait(timeout=30)
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.load(response)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            return list(pool.map(submit, prompts))

    for concurrency in args.concurrency:
        markers = [f"cedar-{concurrency:02d}-{index:03d}-violet"
                   for index in range(concurrency)]
        docs = [f"Record {index}. The unique verification marker is {marker}.\n"
                "Remember that marker for a later question about this record.\n"
                + filler + "\nEND OF RECORD.\n"
                for index, marker in enumerate(markers)]
        questions = [
            doc + "What is the unique verification marker assigned at the beginning "
            "of this record? Reply with only that exact marker, with no explanation."
            for doc in docs
        ]
        # Hybrid KDA boundary-state caching needs an aligned checkpoint. Prime
        # exactly two 4608-token blocks plus one token, as in the exact harness.
        # Chat tokenization must agree with the server; counters below fail
        # closed if it does not. The later query extends this prefix by ~2800
        # tokens, ensuring a real prefill rather than a single decode token.
        prefixes = [aligned_prime_ids(tokenizer, question) for question in questions]
        assert all(len(prefix) == 9217 for prefix in prefixes)
        prime = batch(prefixes, 1, raw_prefix=True)
        (args.out / f"c{concurrency}-prime.json").write_text(
            json.dumps(prime, indent=2))
        before = metrics()
        responses = batch(questions, 400)
        delta = cache_metric_delta(before, metrics())
        artifact = dict(concurrency=concurrency, markers=markers,
                        responses=responses, cache_metrics=delta,
                        scheduler_chunk_membership_proven=False)
        # Always preserve the response before a quality/cache assertion.
        (args.out / f"c{concurrency}-recall.json").write_text(
            json.dumps(artifact, indent=2))
        assert delta["prefix_hit_tokens"] is not None
        assert delta["prefix_hit_tokens"] >= concurrency * 9216, delta
        assert delta["prefill_computed_tokens"] is not None
        assert delta["prefill_computed_tokens"] > concurrency, delta
        for index, response in enumerate(responses):
            check_reply(response, markers[index], markers[:index] + markers[index + 1:])
        print(json.dumps(dict(concurrency=concurrency, recall_passed=concurrency,
                              cache_metrics=delta)), flush=True)


if __name__ == "__main__":
    main()
