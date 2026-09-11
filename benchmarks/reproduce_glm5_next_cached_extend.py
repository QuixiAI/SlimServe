# SPDX-License-Identifier: Apache-2.0
"""Isolate cold vs prefix-reused generation without modifying the live engine."""

import argparse
import json
import urllib.request
from pathlib import Path

from benchmarks.serving_cache_metrics import cache_metric_delta, parse_cache_metrics
from benchmarks.validate_glm5_next_prefill_isolation import aligned_prime_ids
from vllm.tokenizers.registry import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8400")
    parser.add_argument("--tokenizer",
                        default="/home/ubuntu/models/GLM-5.3-Flash-NVFP4")
    parser.add_argument("--salt", required=True, help="Use a fresh salt for every run")
    args = parser.parse_args()
    model = "GLM-5.3-Flash"
    tokenizer = get_tokenizer(args.tokenizer)
    source_ids = tokenizer.encode((Path(__file__).parent / "sonnet.txt").read_text())
    filler = tokenizer.decode((source_ids * ((11999 // len(source_ids)) + 1))[:12000])
    marker = "cedar-01-000-violet"
    question = (f"Record 0. The unique verification marker is {marker}.\n"
                "Remember that marker for a later question about this record.\n"
                + filler + "\nEND OF RECORD.\n"
                "What is the unique verification marker assigned at the beginning "
                "of this record? Reply with only that exact marker, "
                "with no explanation.")

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
            return parse_cache_metrics(response.read().decode(), model)

    def request(endpoint, **kwargs):
        body = dict(model=model, temperature=0, seed=42, **kwargs)
        before = metrics()
        req = urllib.request.Request(args.base_url + endpoint,
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as response:
            result = json.load(response)
        return dict(response=result,
                    cache_metrics=cache_metric_delta(before, metrics()))

    for arm in ("cold", "cached"):
        salt = args.salt + "-" + arm
        if arm == "cached":
            prime = request("/v1/completions",
                            prompt=aligned_prime_ids(tokenizer, question),
                            max_tokens=1, cache_salt=salt)
            print(json.dumps(dict(arm="prime", **prime)), flush=True)
        result = request("/v1/chat/completions",
                         messages=[dict(role="user", content=question)],
                         max_tokens=400, cache_salt=salt)
        print(json.dumps(dict(arm=arm, marker=marker, **result)), flush=True)


if __name__ == "__main__":
    main()
