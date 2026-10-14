# SPDX-License-Identifier: Apache-2.0
"""Save repeated greedy completions and draft counters from a running profile.

This is a repeatability diagnostic, not an exact-token throughput baseline.
Run separately on non-speculative and speculative boots for comparison.
"""

import argparse
import json
from pathlib import Path

from benchmark_dsv4_exact import (
    exact_prompts,
    get_tokenizer,
    metric_counters,
    request_completion,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8400")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--logprobs", type=int, default=None)
    args = parser.parse_args()
    tokenizer = get_tokenizer(args.tokenizer)
    source = Path(__file__).with_name("sonnet.txt").read_text()
    prompt = exact_prompts(tokenizer, source, 1, 1000, 0, False)[0]
    model = "GLM-5.3-Flash"
    responses = []
    for repeat in range(args.repeats):
        before = metric_counters(args.base_url + "/metrics", model)
        result = request_completion(
            args.base_url + "/v1/completions",
            model,
            prompt,
            300,
            600,
            0.0,
            1.0,
            -1,
            42,
            logprobs=args.logprobs,
        )
        after = metric_counters(args.base_url + "/metrics", model)
        result["draft_counters"] = {k: after[k] - before[k] for k in before}
        result["repeat"] = repeat
        responses.append(result)
    texts = [r["response"]["choices"][0]["text"] for r in responses]
    print(json.dumps({"identical": len(set(texts)) == 1, "runs": responses}, indent=2))


if __name__ == "__main__":
    main()
