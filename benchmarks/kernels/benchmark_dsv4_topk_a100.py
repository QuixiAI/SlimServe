#!/usr/bin/env python3
"""Validate and time the DSV4 persistent top-k decode path on A100."""

import argparse
import hashlib
import json
import os

import torch

import vllm._C_stable_libtorch  # noqa: F401


def sorted_indices(indices: torch.Tensor) -> torch.Tensor:
    return torch.sort(indices.to(torch.int64), dim=-1).values


def digest(indices: torch.Tensor) -> str:
    values = sorted_indices(indices).cpu().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=750)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not 512 <= args.context <= 1024:
        raise ValueError("this benchmark targets top-512 contexts up to 1024")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260809)
    logits = torch.randn(
        (args.rows, args.context),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    lengths = torch.full(
        (args.rows,), args.context, dtype=torch.int32, device=device
    )
    output = torch.empty((args.rows, 512), dtype=torch.int32, device=device)
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    expected = torch.topk(logits, 512, dim=-1).indices

    def run() -> None:
        torch.ops._C.persistent_topk(
            logits, lengths, output, workspace, 512, args.context
        )

    run()
    torch.cuda.synchronize()
    eager_exact = torch.equal(sorted_indices(output), sorted_indices(expected))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = torch.equal(sorted_indices(output), sorted_indices(expected))

    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()

    timings_us = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings_us.append(start.elapsed_time(end) * 1_000 / args.iterations)

    result = {
        "device": torch.cuda.get_device_name(),
        "context": args.context,
        "rows": args.rows,
        "topk": 512,
        "cub1024_enabled": os.environ.get("VLLM_DSV4_TOPK_CUB1024") == "1",
        "eager_exact": eager_exact,
        "graph_exact": graph_exact,
        "output_sha256": digest(output),
        "graph_us": timings_us,
        "graph_us_median": sorted(timings_us)[len(timings_us) // 2],
    }
    print(json.dumps(result, indent=2))
    if not eager_exact or not graph_exact:
        raise RuntimeError("persistent top-k did not match torch.topk")


if __name__ == "__main__":
    main()
