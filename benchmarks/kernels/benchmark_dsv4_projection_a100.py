#!/usr/bin/env python3
import argparse
import json

import torch

from vllm.quixicore.ops import quixicore_ops


def time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    results = []
    for tokens in (1, 2, 4, 8):
        x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
        for rows in (64, 256, 1024):
            weight = torch.randn(
                rows, 4096, device="cuda", dtype=torch.bfloat16
            ) / 64
            reference = torch.mm(x, weight.T, out_dtype=torch.float32)
            actual = quixicore_ops.dsv4_projection_gemv(x, weight)
            actual_bf16 = quixicore_ops.dsv4_projection_gemv(x, weight, True)
            results.append(
                {
                    "tokens": tokens,
                    "rows": rows,
                    "native_us": time_us(
                        lambda: quixicore_ops.dsv4_projection_gemv(x, weight),
                        args.warmup,
                        args.iterations,
                    ),
                    "cublas_us": time_us(
                        lambda: torch.mm(x, weight.T, out_dtype=torch.float32),
                        args.warmup,
                        args.iterations,
                    ),
                    "max_abs_error": float((actual - reference).abs().max()),
                    "mean_abs_error": float((actual - reference).abs().mean()),
                    "bf16_exact": bool(torch.equal(actual_bf16, reference.bfloat16())),
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
