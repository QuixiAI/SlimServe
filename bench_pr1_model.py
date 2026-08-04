#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-level coherence + throughput benchmark for PR1 Marlin HIP."""

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--gpu-util", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--bench-runs", type=int, default=3)
    parser.add_argument("--coherence-max-tokens", type=int, default=120)
    parser.add_argument("--warmup-max-tokens", type=int, default=50)
    parser.add_argument("--bench-max-tokens", type=int, default=200)
    parser.add_argument("--bench-min-tokens", type=int, default=200)
    parser.add_argument("--use-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = args.label or args.model

    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60, flush=True)
    print(f"Benchmarking: {label}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"gpu_memory_utilization={args.gpu_util}", flush=True)
    print(f"max_num_batched_tokens={args.max_num_batched_tokens}", flush=True)
    print("=" * 60, flush=True)

    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        disable_log_stats=True,
    )
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    llm = LLM(**llm_kwargs)

    # Coherence check output.
    coherence_prompt = (
        "Explain why eclipses happen and why they do not occur every month."
    )
    coherence_params = SamplingParams(
        temperature=0.0, max_tokens=args.coherence_max_tokens
    )
    t0 = time.perf_counter()
    coherence_out = llm.generate(
        [coherence_prompt], coherence_params, use_tqdm=args.use_tqdm
    )[0]
    t1 = time.perf_counter()
    text = coherence_out.outputs[0].text.strip()
    coherence_tokens = len(coherence_out.outputs[0].token_ids)
    coherence_tps = coherence_tokens / (t1 - t0)
    print(f"COHERENCE_TOKENS={coherence_tokens}", flush=True)
    print(f"COHERENCE_SECONDS={t1 - t0:.3f}", flush=True)
    print(f"COHERENCE_TPS={coherence_tps:.3f}", flush=True)
    print("COHERENCE_OUTPUT_START", flush=True)
    print(text, flush=True)
    print("COHERENCE_OUTPUT_END", flush=True)

    # Warmup.
    if args.warmup_max_tokens > 0:
        warmup_params = SamplingParams(
            temperature=0.0, max_tokens=args.warmup_max_tokens
        )
        llm.generate(["Hello"], warmup_params, use_tqdm=args.use_tqdm)

    # Throughput benchmark.
    bench_prompt = (
        "Explain the theory of general relativity in detail, covering "
        "spacetime curvature, the equivalence principle, and gravitational "
        "waves."
    )
    bench_params = SamplingParams(
        temperature=0.0,
        min_tokens=args.bench_min_tokens,
        max_tokens=args.bench_max_tokens,
    )

    tps = []
    for run_idx in range(max(args.bench_runs, 0)):
        t0 = time.perf_counter()
        outputs = llm.generate([bench_prompt], bench_params, use_tqdm=args.use_tqdm)
        t1 = time.perf_counter()
        token_count = len(outputs[0].outputs[0].token_ids)
        rate = token_count / (t1 - t0)
        tps.append(rate)
        print(
            f"RUN{run_idx + 1}_TOKENS={token_count} RUN{run_idx + 1}_TPS={rate:.3f}",
            flush=True,
        )

    if tps:
        avg_tps = sum(tps) / len(tps)
        print(f"AVG_TPS={avg_tps:.3f}", flush=True)


if __name__ == "__main__":
    main()
