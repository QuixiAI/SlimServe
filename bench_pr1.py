#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark tokens/sec for PR1 dense models."""

import os
import sys
import time


def main():
    model_name = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else model_name

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from vllm import LLM, SamplingParams

    print(f"\n{'=' * 60}", flush=True)
    print(f"Benchmarking: {label}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    llm = LLM(
        model=model_name,
        max_model_len=512,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=200,
        min_tokens=200,
    )

    # Warmup
    print("Warming up...", flush=True)
    llm.generate(["Hello"], sampling_params)

    # Benchmark: single request, measure decode throughput
    prompt = (
        "Explain the theory of general relativity in detail, covering "
        "spacetime curvature, the equivalence principle, and gravitational waves."
    )

    print("Benchmarking (3 runs)...", flush=True)
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()
        n_tokens = len(outputs[0].outputs[0].token_ids)
        elapsed = t1 - t0
        tps = n_tokens / elapsed
        times.append(tps)
        print(
            f"  Run {i + 1}: {n_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s",
            flush=True,
        )

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg:.1f} tok/s", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
