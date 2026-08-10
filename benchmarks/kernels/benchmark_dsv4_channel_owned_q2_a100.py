#!/usr/bin/env python3
"""Validate the output-stationary DSV4 Q2_K down kernel across TP ranks."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.gguf import ops


def make_q2_weights(
    experts: int, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    blocks_per_row = cols // 256
    weights = torch.empty(
        (experts, rows, blocks_per_row, 84), dtype=torch.uint8, device=device
    )
    weights[..., :16] = torch.randint(
        0, 256, weights[..., :16].shape, dtype=torch.uint8, device=device
    )
    weights[..., 16:80] = torch.randint(
        0, 256, weights[..., 16:80].shape, dtype=torch.uint8, device=device
    )
    dm = torch.tensor([0.002, 0.001], dtype=torch.float16, device=device).view(
        torch.uint8
    )
    weights[..., 80:84] = dm
    return weights.view(experts, rows, blocks_per_row * 84)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4):
        raise ValueError("output-stationary Q2_K requires TP2 or TP4")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    full_k = 2048
    local_k = full_k // world_size
    local_rows = 4096 // world_size
    generator = torch.Generator(device=device).manual_seed(20260809)
    topk_ids = ((torch.arange(args.top_k, device=device) * 7) % args.experts).to(
        torch.int32
    )
    route_weights = torch.softmax(
        torch.randn(args.top_k, generator=generator, device=device), dim=0
    )
    mid = (
        torch.randn(
            (args.top_k, full_k), generator=generator, device=device
        ).mul_(0.1)
        * route_weights[:, None]
    ).to(torch.bfloat16)
    local_mid = mid[:, rank * local_k : (rank + 1) * local_k].contiguous()
    quant_mid = ops.ggml_quantize_q8_1(local_mid)
    raw_weights = make_q2_weights(args.experts, local_rows, full_k, device)
    weights = ops.ggml_dsv4_repack_q2_k(raw_weights, full_k)

    # Generic GGUF W2 is the independent native-format reference. It produces
    # one route row at a time; route weights are already folded into `mid`.
    sorted_ids, expert_ids, padded = moe_align_block_size(
        topk_ids.view(1, -1), 4, args.experts
    )
    reference_routes = ops.ggml_moe_a8(
        mid,
        raw_weights,
        sorted_ids,
        expert_ids,
        padded,
        10,
        local_rows,
        1,
        args.top_k,
    )
    reference = reference_routes.float().sum(dim=0, keepdim=True).to(torch.bfloat16)
    actual = comm.dsv4_channel_owned_q2_down(quant_mid, weights, topk_ids)
    torch.cuda.synchronize()
    error = (actual.float() - reference.float()).abs()

    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        captured = comm.dsv4_channel_owned_q2_down(quant_mid, weights, topk_ids)
    dist.barrier()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.iterations):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / args.iterations
    if not torch.isfinite(captured).all():
        raise AssertionError("output-stationary Q2_K produced non-finite values")

    result = {
        "rank": rank,
        "world_size": world_size,
        "experts": args.experts,
        "top_k": args.top_k,
        "local_k": local_k,
        "local_rows": local_rows,
        "graph_us": elapsed_us,
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
