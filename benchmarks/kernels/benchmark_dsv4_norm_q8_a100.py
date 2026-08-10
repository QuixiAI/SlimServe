# SPDX-License-Identifier: Apache-2.0
"""Benchmark and bitwise-check fused DSV4 RMSNorm plus Q8_1 packing."""

import argparse
import json
import statistics

import torch

from vllm import _custom_ops as custom_ops
from vllm.model_executor.layers.quantization.gguf import ops


def median_us(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(7)
    x = torch.randn(
        args.tokens, 4096, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    reference_output = torch.empty_like(x)

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        custom_ops.rms_norm(reference_output, x, weight, 1e-6)
        return reference_output, ops.ggml_quantize_q8_1(reference_output)

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        return ops.ggml_dsv4_rms_norm_q8_1(x, weight, 1e-6)

    ref_output, ref_quant = reference()
    fused_output, fused_quant = fused()
    torch.cuda.synchronize()
    if not torch.equal(ref_output, fused_output):
        raise AssertionError("fused RMSNorm output is not bitwise exact")
    if not torch.equal(ref_quant, fused_quant):
        raise AssertionError("fused Q8_1 output is not bitwise exact")

    reference_us = median_us(reference, args.warmup, args.iterations)
    fused_us = median_us(fused, args.warmup, args.iterations)
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "reference_us": round(reference_us, 3),
                "fused_us": round(fused_us, 3),
                "speedup": round(reference_us / fused_us, 4),
                "saved_us": round(reference_us - fused_us, 3),
                "bf16_bitwise_exact": True,
                "q8_1_bitwise_exact": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
