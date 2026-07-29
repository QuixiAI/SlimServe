# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the in-tree GGUF Q2_K MoE kernels at serving shapes."""

import argparse
import statistics

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.gguf import ops

Q2_K_TYPE = 10
Q2_K_BLOCK_BYTES = 84
Q2_K_BLOCK_SIZE = 256


def make_q2_weights(
    experts: int, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    blocks_per_row = cols // Q2_K_BLOCK_SIZE
    weights = torch.empty(
        (experts, rows, blocks_per_row * Q2_K_BLOCK_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    block = torch.empty(Q2_K_BLOCK_BYTES, dtype=torch.uint8, device=device)
    block[:16] = 0x21
    block[16:80] = 0xE4
    block[80:84] = torch.tensor(
        [0.125, 0.0625], dtype=torch.float16, device=device
    ).view(torch.uint8)
    weights.view(experts, rows, blocks_per_row, Q2_K_BLOCK_BYTES).copy_(block)
    return weights


def time_us(function, warmup: int, iterations: int) -> float:
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
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("w13", "w2"), default="w13")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compare-vector", action="store_true")
    args = parser.parse_args()

    if args.shape == "w13":
        rows, cols, top_k = 2048, 6144, 8
        tokens = args.tokens or 64
    else:
        rows, cols, top_k = 6144, 1024, 1
        tokens = args.tokens or 512

    device = torch.device("cuda")
    torch.manual_seed(0)
    inputs = torch.randn((tokens, cols), dtype=torch.bfloat16, device=device)
    route = torch.arange(tokens * top_k, dtype=torch.int32, device=device)
    topk_ids = (route % args.experts).view(tokens, top_k)
    weights = make_q2_weights(args.experts, rows, cols, device)
    sorted_ids, expert_ids, padded_tokens = moe_align_block_size(
        topk_ids, ops.ggml_moe_get_block_size(Q2_K_TYPE), args.experts
    )

    def matrix() -> torch.Tensor:
        return ops.ggml_moe_a8(
            inputs,
            weights,
            sorted_ids,
            expert_ids,
            padded_tokens,
            Q2_K_TYPE,
            rows,
            top_k,
            tokens,
        )

    matrix_output = matrix()
    matrix_us = time_us(matrix, args.warmup, args.iterations)
    result = {
        "shape": args.shape,
        "tokens": tokens,
        "routed_rows": tokens * top_k,
        "matrix_us": round(matrix_us, 3),
        "padded_rows": int(padded_tokens.item()),
    }

    if args.compare_vector:

        def vector() -> torch.Tensor:
            return ops.ggml_moe_a8_vec(
                inputs, weights, topk_ids, top_k, Q2_K_TYPE, rows, tokens
            )

        vector_output = vector()
        torch.testing.assert_close(matrix_output, vector_output, rtol=0.02, atol=0.5)
        result["vector_us"] = round(time_us(vector, args.warmup, args.iterations), 3)
        result["max_abs_diff"] = round(
            float((matrix_output - vector_output).abs().max()), 6
        )

    print(result)


if __name__ == "__main__":
    main()
