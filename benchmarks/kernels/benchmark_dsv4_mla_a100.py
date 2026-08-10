# SPDX-License-Identifier: Apache-2.0
"""Sweep split-K widths for the native DSV4 packed-cache MLA decode op."""

from __future__ import annotations

import argparse

import torch

from vllm.quixicore.ops import quixicore_ops


def decode_e4m3(codes: torch.Tensor) -> torch.Tensor:
    values = codes.to(torch.int32)
    magnitude = values & 0x7F
    exponent = (magnitude >> 3) & 0xF
    mantissa = magnitude & 0x7
    normal = (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0)
    subnormal = mantissa.float() * (2.0**-9)
    decoded = torch.where(exponent == 0, subnormal, normal)
    return torch.where((values & 0x80) != 0, -decoded, decoded)


def packed_cache(blocks: int, block_size: int, device: torch.device) -> torch.Tensor:
    page_bytes = block_size * (576 + 8)
    cache = torch.zeros((blocks, page_bytes), dtype=torch.uint8, device=device)
    data = cache[:, : block_size * 576].view(blocks, block_size, 576)
    scales = cache[:, block_size * 576 :].view(blocks, block_size, 8)
    magnitudes = torch.randint(
        0, 64, data[:, :, :448].shape, dtype=torch.uint8, device=device
    )
    signs = torch.randint(
        0, 2, data[:, :, :448].shape, dtype=torch.uint8, device=device
    )
    data[:, :, :448].copy_(magnitudes | (signs << 7))
    rope = torch.randn(
        (blocks, block_size, 64), dtype=torch.bfloat16, device=device
    )
    data[:, :, 448:].copy_(rope.view(torch.uint8).reshape_as(data[:, :, 448:]))
    scales.fill_(127)
    return cache


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    block_size = 256
    max_tokens = max(args.swa_tokens, args.sparse_tokens)
    blocks = (max_tokens + block_size - 1) // block_size
    cache = packed_cache(blocks, block_size, device)
    q = torch.randn(
        (args.batch, args.heads, 512), dtype=torch.bfloat16, device=device
    )
    empty_bt = torch.empty((args.batch, 0), dtype=torch.int32, device=device)

    def indices(length: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        if length > width:
            raise ValueError(f"length {length} exceeds graph width {width}")
        row = torch.full((width,), -1, dtype=torch.int32, device=device)
        row[:length] = torch.arange(length, dtype=torch.int32, device=device)
        idx = row.repeat(args.batch, 1).contiguous()
        lens = torch.full((args.batch,), length, dtype=torch.int32, device=device)
        return idx, lens

    swa_idx, swa_lens = indices(args.swa_tokens, args.swa_width)
    sparse_idx, sparse_lens = indices(args.sparse_tokens, args.sparse_width)
    sink = torch.zeros((args.heads,), dtype=torch.float32, device=device)

    def invoke(partition_size: int) -> torch.Tensor:
        return quixicore_ops.mla_decode_fp8_sparse_dsv4(
            q,
            cache,
            empty_bt,
            swa_idx,
            swa_lens,
            True,
            cache,
            empty_bt,
            sparse_idx,
            sparse_lens,
            True,
            sink,
            block_size,
            block_size,
            args.scale,
            partition_size,
        )

    reference = invoke(0)
    cache_data = cache[:, : block_size * 576].view(blocks, block_size, 576)
    cache_scales = cache[:, block_size * 576 :].view(blocks, block_size, 8)
    flat_data = cache_data.reshape(-1, 576)
    flat_scales = cache_scales.reshape(-1, 8)

    def reconstruct(length: int) -> torch.Tensor:
        rows = flat_data[:length]
        fp8 = decode_e4m3(rows[:, :448])
        scale = torch.exp2(flat_scales[:length, :7].float() - 127.0)
        fp8 *= scale.repeat_interleave(64, dim=1)
        rope = rows[:, 448:].contiguous().view(torch.bfloat16).reshape(length, 64)
        return torch.cat((fp8, rope.float()), dim=1)

    values = torch.cat(
        (reconstruct(args.swa_tokens), reconstruct(args.sparse_tokens)), dim=0
    )
    scores = torch.einsum("bhd,kd->bhk", q.float(), values) * args.scale
    scores = torch.cat(
        (scores, torch.zeros((*scores.shape[:2], 1), device=device)), dim=2
    )
    probs = torch.softmax(scores, dim=2)[..., :-1]
    torch_reference = torch.einsum("bhk,kd->bhd", probs, values).to(torch.bfloat16)
    reference_error = (reference.float() - torch_reference.float()).abs().max().item()
    torch.cuda.synchronize(device)
    print(
        f"A100 DSV4 MLA: batch={args.batch} heads={args.heads} "
        f"swa={args.swa_tokens} sparse={args.sparse_tokens} "
        f"widths={args.swa_width}+{args.sparse_width} "
        f"torch_max_error={reference_error:.6f}"
    )
    if reference_error > 0.02:
        raise SystemExit("packed MLA reference mismatch")
    for partition_size in args.partition_sizes:
        for _ in range(args.warmup):
            output = invoke(partition_size)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repetitions):
            output = invoke(partition_size)
        end.record()
        end.synchronize()
        us = start.elapsed_time(end) * 1000.0 / args.repetitions
        error = (output.float() - reference.float()).abs()
        print(
            f"partition={partition_size:>3}  eager={us:9.3f} us  "
            f"max_error={error.max().item():.6f}"
        )
        if args.cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = invoke(partition_size)
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize(device)
            start.record()
            for _ in range(args.repetitions):
                graph.replay()
            end.record()
            end.synchronize()
            graph_us = start.elapsed_time(end) * 1000.0 / args.repetitions
            graph_error = (graph_output.float() - reference.float()).abs().max().item()
            print(
                f"partition={partition_size:>3}  graph={graph_us:9.3f} us  "
                f"max_error={graph_error:.6f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--swa-tokens", type=int, default=512)
    parser.add_argument("--sparse-tokens", type=int, default=250)
    parser.add_argument("--swa-width", type=int, default=512)
    parser.add_argument("--sparse-width", type=int, default=512)
    parser.add_argument("--scale", type=float, default=0.04419417382415922)
    parser.add_argument(
        "--partition-sizes", type=int, nargs="+", default=[0, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--cuda-graph", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
