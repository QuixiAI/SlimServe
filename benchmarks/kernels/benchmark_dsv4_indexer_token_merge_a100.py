#!/usr/bin/env python3
"""Validate and time native DSV4 token-shard indexer candidate merge."""

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def sorted_set(indices: torch.Tensor) -> torch.Tensor:
    return torch.sort(indices.to(torch.int64), dim=-1).values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=16_387)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    topk = 512
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4):
        raise RuntimeError("token merge benchmark requires TP2 or TP4")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=8 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    generator = torch.Generator(device=device).manual_seed(20260809)
    global_logits = torch.randn(
        (args.rows, args.context),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    logits = global_logits[:, rank::world_size].contiguous()
    local_length = logits.shape[1]
    lengths = torch.full(
        (args.rows,), local_length, dtype=torch.int32, device=device
    )
    output = torch.empty((args.rows, topk), dtype=torch.int32, device=device)
    local_indices = torch.empty_like(output)
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    expected = torch.topk(global_logits, min(topk, args.context), dim=-1).indices

    torch.ops._C.persistent_topk(
        logits, lengths, local_indices, workspace, topk, logits.shape[1]
    )
    comm.dsv4_indexer_token_merge(
        logits, lengths, local_indices, output, topk
    )
    torch.cuda.synchronize()
    eager_exact = torch.equal(
        sorted_set(output[:, : expected.shape[1]]), sorted_set(expected)
    )

    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        torch.ops._C.persistent_topk(
            logits, lengths, local_indices, workspace, topk, logits.shape[1]
        )
        comm.dsv4_indexer_token_merge(
            logits, lengths, local_indices, output, topk
        )
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = torch.equal(
        sorted_set(output[:, : expected.shape[1]]), sorted_set(expected)
    )

    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.iterations):
        graph.replay()
    torch.cuda.synchronize()
    graph_us = (time.perf_counter() - start) * 1e6 / args.iterations

    result = {
        "rank": rank,
        "world_size": world_size,
        "rows": args.rows,
        "context": args.context,
        "local_context": local_length,
        "topk": topk,
        "eager_exact": eager_exact,
        "graph_exact": graph_exact,
        "local_topk_plus_merge_graph_us": graph_us,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    if not eager_exact or not graph_exact:
        raise RuntimeError("token-shard merge did not match global top-k")
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
