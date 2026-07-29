# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark in-tree GGUF linear kernels at GLM serving shapes."""

import argparse
import statistics

import torch

from vllm.model_executor.layers.quantization.gguf import ops

Q8_0_TYPE = 8
Q8_0_BLOCK_BYTES = 34
Q8_0_BLOCK_SIZE = 32

SHAPES = {
    "o_proj": (6144, 8192),
    "dense_gate_up": (12288, 6144),
    "q_b_proj": (6144, 2048),
}


def make_q8_weights(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    blocks_per_row = cols // Q8_0_BLOCK_SIZE
    weights = torch.empty(
        (rows, blocks_per_row * Q8_0_BLOCK_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    block = torch.empty(Q8_0_BLOCK_BYTES, dtype=torch.uint8, device=device)
    block[:2] = torch.tensor([0.125], dtype=torch.float16, device=device).view(
        torch.uint8
    )
    block[2:] = torch.arange(32, dtype=torch.uint8, device=device)
    weights.view(rows, blocks_per_row, Q8_0_BLOCK_BYTES).copy_(block)
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
    parser.add_argument("--shape", choices=SHAPES, default="o_proj")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compare-vector", action="store_true")
    args = parser.parse_args()

    rows, cols = SHAPES[args.shape]
    device = torch.device("cuda")
    torch.manual_seed(0)
    inputs = torch.randn((args.tokens, cols), dtype=torch.bfloat16, device=device)
    weights = make_q8_weights(rows, cols, device)

    def matrix() -> torch.Tensor:
        return ops.ggml_mul_mat_a8(weights, inputs, Q8_0_TYPE, rows)

    matrix_output = matrix()
    result = {
        "shape": args.shape,
        "tokens": args.tokens,
        "matrix_us": round(time_us(matrix, args.warmup, args.iterations), 3),
    }

    if args.compare_vector:

        def vector() -> torch.Tensor:
            return ops.ggml_mul_mat_vec_a8(weights, inputs, Q8_0_TYPE, rows)

        vector_output = vector()
        torch.testing.assert_close(matrix_output, vector_output, rtol=0.02, atol=0.5)
        result["vector_us"] = round(time_us(vector, args.warmup, args.iterations), 3)
        result["max_abs_diff"] = round(
            float((matrix_output - vector_output).abs().max()), 6
        )

    print(result)


if __name__ == "__main__":
    main()
