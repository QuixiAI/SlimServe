#!/usr/bin/env python3
"""Validate and time input-owned DSV4 attention projection bundles."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.quantization.gguf import ops as gguf_ops


def graph_us(comm: CustomAllreduce, fn, warmup: int, iterations: int):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        outputs = fn()
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize()
    return outputs, (time.perf_counter() - start) * 1e6 / iterations


def make_q8_weight(
    rows: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    blocks = 4096 // 32
    raw = torch.empty((rows, blocks, 34), dtype=torch.uint8, device=device)
    scales = (
        torch.rand(
            (rows, blocks), device=device, dtype=torch.float16, generator=generator
        )
        * 0.02
        + 0.001
    )
    codes = torch.randint(
        -127,
        128,
        (rows, blocks, 32),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    raw[:, :, :2].copy_(scales.view(torch.uint8).reshape(rows, blocks, 2))
    raw[:, :, 2:].copy_(codes.view(torch.uint8))
    return raw.reshape(rows, blocks * 34)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q8-rows", type=int, default=1536)
    parser.add_argument("--bf16-rows", type=int, nargs=3, default=(576, 64, 576))
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise ValueError("input-owned attention requires TP2, TP4, or TP8")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=group, device=device, max_size=1 << 20)

    generator = torch.Generator(device=device).manual_seed(20260809)
    full_input = (
        torch.randn((1, 4096), generator=generator, device=device) * 0.1
    ).to(torch.bfloat16)
    local_hidden = 4096 // world_size
    local_input = full_input[
        :, rank * local_hidden : (rank + 1) * local_hidden
    ].contiguous()
    full_quant = gguf_ops.ggml_quantize_q8_1(full_input)
    local_quant = gguf_ops.ggml_quantize_q8_1(local_input)
    raw_q8_weight = make_q8_weight(args.q8_rows, device, generator)
    aligned_q8_weight = gguf_ops.ggml_dsv4_repack_q8_0_aligned(raw_q8_weight)
    bf16_weights = [
        (
            torch.randn((rows, 4096), generator=generator, device=device) * 0.01
        ).to(torch.bfloat16)
        for rows in args.bf16_rows
    ]
    router_weight = (
        torch.randn((256, 4096), generator=generator, device=device) * 0.01
    ).to(torch.bfloat16)
    full_residual = (
        torch.randn((1, 4, 4096), generator=generator, device=device) * 0.1
    ).to(torch.bfloat16)
    local_residual = full_residual[
        :, :, rank * local_hidden : (rank + 1) * local_hidden
    ].contiguous()

    def run_owned():
        return comm.dsv4_owned_attention_projections(
            local_input,
            local_quant,
            aligned_q8_weight,
            bf16_weights[0],
            bf16_weights[1],
            bf16_weights[2],
        )

    expected_q8 = gguf_ops.ggml_dsv4_mul_mat_vec_aligned_q8_0(
        aligned_q8_weight, full_input, full_quant, args.q8_rows, 1
    )
    expected_bf16 = [
        full_input.float() @ weight.float().t() for weight in bf16_weights
    ]
    outputs, latency_us = graph_us(comm, run_owned, args.warmup, args.iterations)
    gathered_quant, gather_us = graph_us(
        comm,
        lambda: comm.dsv4_gather_owned_q8(local_quant),
        args.warmup,
        args.iterations,
    )
    router_output, router_us = graph_us(
        comm,
        lambda: comm.dsv4_owned_router(local_input, router_weight),
        args.warmup,
        args.iterations,
    )
    gathered_bf16, gather_bf16_us = graph_us(
        comm,
        lambda: comm.dsv4_gather_owned_bf16(local_residual),
        args.warmup,
        args.iterations,
    )
    errors = []
    for actual, expected in zip(outputs, (expected_q8, *expected_bf16)):
        difference = (actual.float() - expected.float()).abs()
        errors.append(
            {
                "max_abs": float(difference.max()) if difference.numel() else 0.0,
                "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
            }
        )
    result = {
        "rank": rank,
        "world_size": world_size,
        "rows": [args.q8_rows, *args.bf16_rows],
        "errors": errors,
        "graph_us": latency_us,
        "gather_q8_max_abs": float(
            (gathered_quant.view(torch.int32) - full_quant.view(torch.int32))
            .abs()
            .max()
        ),
        "gather_q8_us": gather_us,
        "router_max_abs": float(
            (
                router_output
                - full_input.float() @ router_weight.float().t()
            )
            .abs()
            .max()
        ),
        "router_us": router_us,
        "gather_bf16_max_abs": float(
            (gathered_bf16.float() - full_residual.float()).abs().max()
        ),
        "gather_bf16_us": gather_bf16_us,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
