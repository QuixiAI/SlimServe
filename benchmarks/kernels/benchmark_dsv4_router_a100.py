# SPDX-License-Identifier: Apache-2.0
"""Benchmark the native DeepSeek-V4 A100 router paths."""

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from vllm.quixicore.ops import quixicore_ops


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
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    weight = (torch.randn(256, 4096, device="cuda") * 0.02).bfloat16()
    tid2eid = torch.randint(0, 256, (1024, 6), device="cuda", dtype=torch.int32)
    routed_scaling_factor = 2.5
    results = []
    for tokens in range(1, 9):
        x = (torch.randn(tokens, 4096, device="cuda") * 0.05).bfloat16()
        input_ids = torch.arange(tokens, device="cuda", dtype=torch.int64)
        native = quixicore_ops.dsv4_router_gemm(x, weight)
        fp32_reference = torch.mm(x, weight.T, out_dtype=torch.float32)
        fallback = F.linear(x, weight).float()
        selected = tid2eid[input_ids]
        reference_scores = torch.sqrt(F.softplus(native)).gather(1, selected.long())
        reference_weights = (
            reference_scores
            * routed_scaling_factor
            / reference_scores.sum(dim=1, keepdim=True)
        )
        hash_weights, hash_ids = quixicore_ops.dsv4_hash_router(
            x, weight, input_ids, tid2eid, routed_scaling_factor
        )
        results.append(
            {
                "tokens": tokens,
                "native_us": median_us(
                    lambda: quixicore_ops.dsv4_router_gemm(x, weight),
                    args.warmup,
                    args.iterations,
                ),
                "fallback_bf16_cast_us": median_us(
                    lambda: F.linear(x, weight).float(),
                    args.warmup,
                    args.iterations,
                ),
                "cublas_fp32_us": median_us(
                    lambda: torch.mm(x, weight.T, out_dtype=torch.float32),
                    args.warmup,
                    args.iterations,
                ),
                "hash_fused_us": median_us(
                    lambda: quixicore_ops.dsv4_hash_router(
                        x, weight, input_ids, tid2eid, routed_scaling_factor
                    ),
                    args.warmup,
                    args.iterations,
                ),
                "hash_unfused_us": median_us(
                    lambda: torch.sqrt(
                        F.softplus(quixicore_ops.dsv4_router_gemm(x, weight))
                    ).gather(1, selected.long()),
                    args.warmup,
                    args.iterations,
                ),
                "hash_ids_match": bool(torch.equal(hash_ids, selected)),
                "hash_weight_max_abs_error": float(
                    (hash_weights - reference_weights).abs().max().item()
                ),
                "max_abs_error_vs_fp32": float(
                    (native - fp32_reference).abs().max().item()
                ),
                "top6_matches_fp32": bool(
                    torch.equal(native.topk(6).indices, fp32_reference.topk(6).indices)
                ),
                "top6_matches_fallback": bool(
                    torch.equal(native.topk(6).indices, fallback.topk(6).indices)
                ),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
