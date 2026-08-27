#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact-shape DSV4 throughput benchmark for a local OpenAI completion API."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Reserve stdout for the benchmark's JSON before vLLM configures its log
# handlers at import time.
os.environ["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"

from vllm.tokenizers.registry import get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local GGUF path")
    parser.add_argument(
        "--served-model-name",
        default="DeepSeek-V4-Flash",
        help="Model name to send to the OpenAI-compatible endpoint",
    )
    parser.add_argument("--source", required=True, help="Natural-text prompt source")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument(
        "--metrics-url",
        help="Prometheus endpoint; defaults to /metrics on the completion host. "
        "Pass 'none' for servers without metrics; spec-decode counters are "
        "then reported as zero and the draft-token gate is skipped",
    )

    def _concurrency(value: str) -> int:
        parsed = int(value)
        if not 1 <= parsed <= 128:
            raise argparse.ArgumentTypeError("concurrency must be 1..128")
        return parsed

    parser.add_argument("--concurrency", type=_concurrency, required=True)
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=2000)
    # Sampling: greedy/temp-0 is banned stack-wide. When these are left
    # unset the request omits them, so the server applies the model's
    # SHIPPED generation_config defaults; the default seed keeps runs
    # deterministic and sha-pinnable under real sampling. Every value is
    # still overridable for diagnostics, but never silently greedy.
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-no-spec",
        action="store_true",
        help="permit a plain-decoding server with no speculative draft metrics",
    )
    parser.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="Starting source-token offset; vary it between prefix-cached runs",
    )
    parser.add_argument(
        "--repeat-source",
        action="store_true",
        help="Repeat the tokenized source when it is shorter than the prompt",
    )
    parser.add_argument(
        "--prompt-overhead",
        type=int,
        default=0,
        help="Server-added wrapper tokens included in API prompt usage",
    )
    parser.add_argument(
        "--warmup-output-tokens",
        type=int,
        default=8,
        help="Prime each prompt before timing; set to 0 to measure cold serving",
    )
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument(
        "--dump-completions",
        default=None,
        help="Directory to write each completion's raw text; used to compare "
        "outputs token-for-token across deliberate numeric changes where the "
        "sha alone cannot say how close a mismatch is.",
    )
    return parser.parse_args()


SPEC_METRICS = {
    "spec_decode_drafts": "vllm:spec_decode_num_drafts_total",
    "spec_decode_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "spec_decode_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def metric_counters(url: str, served_model_name: str) -> dict[str, float]:
    if url == "none":
        return dict.fromkeys(SPEC_METRICS, 0.0)
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read().decode("utf-8")
    counters = dict.fromkeys(SPEC_METRICS, 0.0)
    model_label = f'model_name="{served_model_name}"'
    for line in body.splitlines():
        if not line or line.startswith("#") or model_label not in line:
            continue
        for key, metric_name in SPEC_METRICS.items():
            if line.startswith(f"{metric_name}{{"):
                counters[key] += float(line.rsplit(maxsplit=1)[1])
    return counters


def exact_prompts(
    tokenizer,
    source: str,
    count: int,
    token_count: int,
    prompt_offset: int,
    repeat_source: bool,
) -> list[str]:
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    if len(source_ids) < token_count:
        if not repeat_source:
            raise ValueError(
                f"source has {len(source_ids)} tokens, need at least {token_count}"
            )
        repeats = (token_count + len(source_ids) - 1) // len(source_ids)
        source_ids = (source_ids * repeats)[:token_count]

    max_start = len(source_ids) - token_count
    if prompt_offset < 0 or prompt_offset > max_start:
        raise ValueError(f"prompt offset must be between 0 and {max_start}")
    stride = max(1, (max_start - prompt_offset) // max(1, count - 1))
    prompts: list[str] = []
    candidate = prompt_offset
    while len(prompts) < count and candidate <= max_start:
        text = tokenizer.decode(source_ids[candidate : candidate + token_count])
        if len(tokenizer.encode(text, add_special_tokens=False)) == token_count:
            prompts.append(text)
        candidate += stride if prompts else 1
    if len(prompts) != count:
        raise ValueError(f"could only construct {len(prompts)} exact prompts")
    return prompts


def request_completion(
    url: str,
    served_model_name: str,
    prompt: str,
    output_tokens: int,
    timeout: float,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": served_model_name,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "seed": seed,
        "stream": False,
        # vLLM honors this extension. DS4 may ignore it until its server
        # adapter exposes the same benchmark control.
        "ignore_eos": True,
    }
    # Omitted knobs fall through to the model's shipped generation_config.
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {"seconds": time.perf_counter() - started, "response": payload}


def main() -> None:
    args = parse_args()
    if args.metrics_url is None:
        parsed_url = urllib.parse.urlsplit(args.url)
        args.metrics_url = urllib.parse.urlunsplit(
            (parsed_url.scheme, parsed_url.netloc, "/metrics", "", "")
        )
    tokenizer = get_tokenizer(args.model)
    prompts = exact_prompts(
        tokenizer,
        Path(args.source).read_text(),
        args.concurrency,
        args.input_tokens - args.prompt_overhead,
        args.prompt_offset,
        args.repeat_source,
    )

    if args.warmup_output_tokens:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            warmups = [
                executor.submit(
                    request_completion,
                    args.url,
                    args.served_model_name,
                    prompt,
                    args.warmup_output_tokens,
                    args.timeout,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.seed,
                )
                for prompt in prompts
            ]
            for warmup in warmups:
                warmup.result()

    # Sampled after the warmup so warmup drafts stay out of the delta.
    metrics_before = metric_counters(args.metrics_url, args.served_model_name)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                request_completion,
                args.url,
                args.served_model_name,
                prompt,
                args.output_tokens,
                args.timeout,
                args.temperature,
                args.top_p,
                args.top_k,
                args.seed,
            )
            for prompt in prompts
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started
    metrics_after = metric_counters(args.metrics_url, args.served_model_name)
    spec_metrics = {
        key: metrics_after[key] - metrics_before[key] for key in SPEC_METRICS
    }

    prompt_counts: list[int] = []
    completion_counts: list[int] = []
    latencies: list[float] = []
    response_sha256: list[str] = []
    dump_dir = Path(args.dump_completions) if args.dump_completions else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    chars_per_token: list[float] = []
    for index, result in enumerate(results):
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
        completion_tokens = int(usage["completion_tokens"])
        chars_per_token.append(
            len(choice["text"]) / completion_tokens if completion_tokens else 0.0
        )
        if dump_dir is not None:
            (dump_dir / f"completion_{index}.txt").write_text(choice["text"])

    summary = {
        "model": str(Path(args.model).resolve()),
        "served_model_name": args.served_model_name,
        "concurrency": args.concurrency,
        "requested_input_tokens": args.input_tokens,
        "requested_output_tokens": args.output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "prompt_offset": args.prompt_offset,
        "warmup_output_tokens": args.warmup_output_tokens,
        "prompt_tokens": prompt_counts,
        "completion_tokens": completion_counts,
        "wall_seconds": wall_seconds,
        "aggregate_output_tps": sum(completion_counts) / wall_seconds,
        "request_latency_mean_seconds": statistics.mean(latencies),
        "request_latency_median_seconds": statistics.median(latencies),
        "response_sha256": response_sha256,
        "chars_per_token": chars_per_token,
        **spec_metrics,
        "exact": prompt_counts == [args.input_tokens] * args.concurrency
        and completion_counts == [args.output_tokens] * args.concurrency,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["exact"]:
        raise SystemExit("server did not honor the exact benchmark token counts")
    if (
        args.metrics_url != "none"
        and not args.allow_no_spec
        and summary["spec_decode_draft_tokens"] <= 0
    ):
        raise SystemExit("server produced no speculative draft tokens")
    # A degenerate server (NaN logits -> special-token loop) still returns the
    # exact token counts, but the tokens decode to almost no visible text.
    # Healthy prose/code sits well above 1 char/token; the BOS loop is ~0.
    degenerate = [ratio for ratio in chars_per_token if ratio < 0.5]
    if degenerate:
        raise SystemExit(
            f"{len(degenerate)}/{len(chars_per_token)} responses look degenerate "
            f"(chars/token {min(degenerate):.3f}); throughput is not valid"
        )


if __name__ == "__main__":
    main()
