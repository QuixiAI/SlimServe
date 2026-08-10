#!/usr/bin/env python3
"""Validate and time DSV4 head-partial peer top-k on A100."""

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def topk_set(indices: torch.Tensor) -> torch.Tensor:
    return torch.sort(indices.to(torch.int64), dim=-1).values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=16_384)
    parser.add_argument("--topk", type=int, choices=(512, 1024, 2048), default=512)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=8 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    partials = []
    for peer_rank in range(world_size):
        generator = torch.Generator(device=device).manual_seed(20260808 + peer_rank)
        partials.append(
            torch.randn(
                (args.rows, args.context),
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
        )
    logits = partials[rank].contiguous()
    lengths = torch.full(
        (args.rows,), args.context, dtype=torch.int32, device=device
    )
    output = torch.empty((args.rows, args.topk), dtype=torch.int32, device=device)
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)

    expected_logits = torch.zeros_like(logits)
    for partial in partials:
        expected_logits.add_(partial)
    expected = torch.topk(expected_logits, args.topk, dim=-1).indices

    comm.dsv4_indexer_topk(
        logits, lengths, output, workspace, args.topk, args.context
    )
    torch.cuda.synchronize()
    eager_exact = torch.equal(topk_set(output), topk_set(expected))

    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        comm.dsv4_indexer_topk(
            logits, lengths, output, workspace, args.topk, args.context
        )
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = torch.equal(topk_set(output), topk_set(expected))

    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.iterations):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / args.iterations

    result = {
        "rank": rank,
        "world_size": world_size,
        "rows": args.rows,
        "context": args.context,
        "topk": args.topk,
        "eager_exact": eager_exact,
        "graph_exact": graph_exact,
        "graph_us": elapsed_us,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    if not eager_exact or not graph_exact:
        raise RuntimeError("DSV4 peer top-k did not match summed-logit reference")
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
