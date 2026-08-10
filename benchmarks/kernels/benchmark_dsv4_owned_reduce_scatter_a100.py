#!/usr/bin/env python3
"""Validate and time the DSV4 channel-ownership reduce-scatter."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def make_partial(
    rank: int, tokens: int, device: torch.device, seed_offset: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(
        2026080900 + seed_offset + rank
    )
    return torch.randn(
        (tokens, 4096), generator=generator, device=device
    ).mul_(0.01).to(torch.bfloat16)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise ValueError("DSV4 owned reduce-scatter requires TP2, TP4, or TP8")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    local_hidden = 4096 // world_size
    row_start = rank * local_hidden
    peer_inputs = [
        make_partial(peer, args.tokens, device, 0) for peer in range(world_size)
    ]
    peer_addends = [
        make_partial(peer, args.tokens, device, 100) for peer in range(world_size)
    ]
    input = peer_inputs[rank]
    addend = peer_addends[rank]

    expected = torch.zeros(
        (args.tokens, local_hidden), dtype=torch.float32, device=device
    )
    expected_add = torch.zeros_like(expected)
    for peer in range(world_size):
        expected.add_(
            peer_inputs[peer][:, row_start : row_start + local_hidden].float()
        )
        local_combined = (
            peer_inputs[peer].float() + peer_addends[peer].float()
        ).to(torch.bfloat16)
        expected_add.add_(
            local_combined[:, row_start : row_start + local_hidden].float()
        )
    expected = expected.to(torch.bfloat16)
    expected_add = expected_add.to(torch.bfloat16)

    actual = comm.dsv4_owned_reduce_scatter(input)
    actual_add = comm.dsv4_owned_reduce_scatter(input, addend)
    torch.cuda.synchronize()
    eager_error = (actual.float() - expected.float()).abs().max().item()
    eager_add_error = (
        actual_add.float() - expected_add.float()
    ).abs().max().item()
    if eager_error != 0.0 or eager_add_error != 0.0:
        raise AssertionError(
            f"owned reduce-scatter mismatch: {eager_error}, {eager_add_error}"
        )

    graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(graph):
        captured = comm.dsv4_owned_reduce_scatter(input)
    add_graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(add_graph):
        captured_add = comm.dsv4_owned_reduce_scatter(input, addend)
    legacy_graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(legacy_graph):
        legacy_full = comm.custom_all_reduce(input)
        assert legacy_full is not None

    owned_us = graph_time_us(graph, args.warmup, args.iterations)
    owned_add_us = graph_time_us(add_graph, args.warmup, args.iterations)
    legacy_us = graph_time_us(legacy_graph, args.warmup, args.iterations)
    graph_error = (captured.float() - expected.float()).abs().max().item()
    graph_add_error = (
        captured_add.float() - expected_add.float()
    ).abs().max().item()
    legacy_expected = torch.zeros_like(input, dtype=torch.float32)
    for peer_input in peer_inputs:
        legacy_expected.add_(peer_input.float())
    legacy_error = (
        legacy_full.float() - legacy_expected.to(torch.bfloat16).float()
    ).abs().max().item()

    result = {
        "rank": rank,
        "world_size": world_size,
        "tokens": args.tokens,
        "local_hidden": local_hidden,
        "owned_us": owned_us,
        "owned_add_us": owned_add_us,
        "legacy_full_allreduce_us": legacy_us,
        "owned_vs_legacy": legacy_us / owned_us,
        "eager_max_abs_error": eager_error,
        "eager_add_max_abs_error": eager_add_error,
        "graph_max_abs_error": graph_error,
        "graph_add_max_abs_error": graph_add_error,
        "legacy_max_abs_error": legacy_error,
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
