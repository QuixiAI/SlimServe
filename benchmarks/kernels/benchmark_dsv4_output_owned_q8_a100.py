#!/usr/bin/env python3
"""Validate and time the reusable DSV4 output-owned Q8 TP projection."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.quantization.gguf import ops


def make_q8_weight(
    rows: int, cols: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    blocks = cols // 32
    weight = torch.empty((rows, blocks, 34), dtype=torch.uint8, device=device)
    scales = torch.rand(
        (rows, blocks), generator=generator, dtype=torch.float32, device=device
    ).mul_(0.001).add_(0.0001).to(torch.float16)
    codes = torch.randint(
        -127,
        128,
        (rows, blocks, 32),
        generator=generator,
        dtype=torch.int8,
        device=device,
    )
    weight[..., :2].copy_(scales.view(torch.uint8).reshape(rows, blocks, 2))
    weight[..., 2:].copy_(codes.view(torch.uint8))
    return weight.view(rows, blocks * 34)


def graph_time_us(
    graph: torch.cuda.CUDAGraph, warmup: int, iterations: int
) -> float:
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def run_shape(
    comm: CustomAllreduce,
    rank: int,
    world_size: int,
    full_cols: int,
    tokens: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    device = torch.device("cuda", rank)
    local_cols = full_cols // world_size
    local_rows = 4096 // world_size
    generator = torch.Generator(device=device).manual_seed(
        2026080900 + full_cols + tokens
    )
    full_input = torch.randn(
        (tokens, full_cols), generator=generator, device=device
    ).to(torch.bfloat16)
    local_input = full_input[:, rank * local_cols : (rank + 1) * local_cols]
    local_quant = ops.ggml_quantize_q8_1(local_input.contiguous())
    full_quant = ops.ggml_quantize_q8_1(full_input)
    packed_weight = make_q8_weight(
        local_rows, full_cols, device, 2026081000 + full_cols + rank
    )
    aligned_weight = ops.ggml_dsv4_repack_q8_0_aligned(packed_weight)
    expected = ops.ggml_dsv4_mul_mat_vec_aligned_q8_0(
        aligned_weight, full_input, full_quant, local_rows, 2
    )

    timings: dict[str, float] = {}
    errors: dict[str, float] = {}
    for rows_per_cta in (1, 2, 4):
        actual = comm.dsv4_output_owned_q8(
            local_quant, aligned_weight, rows_per_cta
        )
        torch.cuda.synchronize()
        error = (actual.float() - expected.float()).abs().max().item()
        if error != 0.0:
            raise AssertionError(
                f"output-owned Q8 mismatch for rows={rows_per_cta}: {error}"
            )
        graph = torch.cuda.CUDAGraph()
        with comm.capture(), torch.cuda.graph(graph):
            captured = comm.dsv4_output_owned_q8(
                local_quant, aligned_weight, rows_per_cta
            )
        timing = graph_time_us(graph, warmup, iterations)
        graph_error = (captured.float() - expected.float()).abs().max().item()
        if graph_error != 0.0:
            raise AssertionError(
                "captured output-owned Q8 mismatch for "
                f"rows={rows_per_cta}: {graph_error}"
            )
        timings[f"rows{rows_per_cta}_us"] = timing
        errors[f"rows{rows_per_cta}_max_abs_error"] = graph_error

    return {
        "rank": rank,
        "world_size": world_size,
        "tokens": tokens,
        "full_cols": full_cols,
        "local_cols": local_cols,
        "local_rows": local_rows,
        **timings,
        **errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-cols", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise ValueError("output-owned DSV4 Q8 requires TP2, TP4, or TP8")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    local_results = [
        run_shape(
            comm,
            rank,
            world_size,
            full_cols,
            tokens,
            args.warmup,
            args.iterations,
        )
        for full_cols in args.full_cols
        for tokens in args.tokens
    ]
    gathered: list[list[dict[str, object]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
