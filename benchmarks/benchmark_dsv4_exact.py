#!/usr/bin/env python3
"""Exact-shape DSV4 throughput benchmark for a local OpenAI completion API."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

from vllm.tokenizers.registry import get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local GGUF path")
    parser.add_argument("--source", required=True, help="Natural-text prompt source")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--concurrency", type=int, choices=(1, 8), required=True)
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=2000)
    parser.add_argument(
        "--prompt-overhead",
        type=int,
        default=0,
        help="Server-added wrapper tokens included in API prompt usage",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def exact_prompts(
    tokenizer, source: str, count: int, token_count: int
) -> list[str]:
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    if len(source_ids) < token_count:
        raise ValueError(
            f"source has {len(source_ids)} tokens, need at least {token_count}"
        )

    max_start = len(source_ids) - token_count
    stride = max(1, max_start // max(1, count - 1))
    prompts: list[str] = []
    candidate = 0
    while len(prompts) < count and candidate <= max_start:
        text = tokenizer.decode(source_ids[candidate : candidate + token_count])
        if len(tokenizer.encode(text, add_special_tokens=False)) == token_count:
            prompts.append(text)
        candidate += stride if prompts else 1
    if len(prompts) != count:
        raise ValueError(f"could only construct {len(prompts)} exact prompts")
    return prompts


def request_completion(
    url: str, prompt: str, output_tokens: int, timeout: float
) -> dict[str, object]:
    body = json.dumps(
        {
            "model": "deepseek-v4-flash",
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": False,
            # vLLM honors this extension. DS4 may ignore it until its server
            # adapter exposes the same benchmark control.
            "ignore_eos": True,
        }
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {"seconds": time.perf_counter() - started, "response": payload}


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer(args.model)
    prompts = exact_prompts(
        tokenizer,
        Path(args.source).read_text(),
        args.concurrency,
        args.input_tokens - args.prompt_overhead,
    )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                request_completion,
                args.url,
                prompt,
                args.output_tokens,
                args.timeout,
            )
            for prompt in prompts
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started

    prompt_counts: list[int] = []
    completion_counts: list[int] = []
    latencies: list[float] = []
    response_sha256: list[str] = []
    for result in results:
        response = result["response"]
        assert isinstance(response, dict)
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError(f"response has no usage object: {response}")
        prompt_counts.append(int(usage["prompt_tokens"]))
        completion_counts.append(int(usage["completion_tokens"]))
        latencies.append(float(result["seconds"]))
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"response has no choices: {response}")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
            raise RuntimeError(f"completion has no text: {response}")
        response_sha256.append(
            hashlib.sha256(choice["text"].encode("utf-8")).hexdigest()
        )

    summary = {
        "concurrency": args.concurrency,
        "requested_input_tokens": args.input_tokens,
        "requested_output_tokens": args.output_tokens,
        "prompt_tokens": prompt_counts,
        "completion_tokens": completion_counts,
        "wall_seconds": wall_seconds,
        "aggregate_output_tps": sum(completion_counts) / wall_seconds,
        "request_latency_mean_seconds": statistics.mean(latencies),
        "request_latency_median_seconds": statistics.median(latencies),
        "response_sha256": response_sha256,
        "exact": prompt_counts == [args.input_tokens] * args.concurrency
        and completion_counts == [args.output_tokens] * args.concurrency,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["exact"]:
        raise SystemExit("server did not honor the exact benchmark token counts")


if __name__ == "__main__":
    main()
