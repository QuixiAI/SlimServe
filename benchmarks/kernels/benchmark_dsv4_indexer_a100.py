# SPDX-License-Identifier: Apache-2.0
"""Validate and time the native DSV4 Ampere paged-indexer logits op."""

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


def random_e4m3(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    magnitude = torch.randint(0, 0x48, shape, dtype=torch.uint8, device=device)
    sign = torch.randint(0, 2, shape, dtype=torch.uint8, device=device) << 7
    return magnitude | sign


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    batch, heads, dim, block_size = 1, args.heads, 128, 256
    blocks = (args.context + block_size - 1) // block_size

    q = random_e4m3((batch, 1, heads, dim), device).view(torch.float8_e4m3fn)
    raw = torch.empty(
        (blocks, block_size * (dim + 4)), dtype=torch.uint8, device=device
    )
    key_bytes = block_size * dim
    keys = raw[:, :key_bytes].view(blocks, block_size, dim)
    keys.copy_(random_e4m3(tuple(keys.shape), device))
    scales = torch.rand((blocks, block_size), dtype=torch.float32, device=device)
    raw[:, key_bytes:].copy_(scales.view(torch.uint8).reshape(blocks, -1))
    cache = raw.view(blocks, block_size, dim + 4)
    weights = torch.rand((batch, heads), dtype=torch.float32, device=device)
    context_lens = torch.tensor([[args.context]], dtype=torch.int32, device=device)
    block_tables = torch.arange(blocks, dtype=torch.int32, device=device)[None, :]

    def invoke(
        rank: int = 0, world_size: int = 1, trivial_topk: int = 0
    ) -> torch.Tensor:
        return quixicore_ops.fp8_paged_mqa_logits(
            q,
            cache,
            weights,
            context_lens,
            block_tables,
            args.max_model_len,
            rank,
            world_size,
            trivial_topk,
        )

    output = invoke()
    q_ref = decode_e4m3(q[0, 0].view(torch.uint8)).cpu()
    k_ref = decode_e4m3(keys.reshape(-1, dim)[: args.context]).cpu()
    score_ref = torch.relu(k_ref @ q_ref.T)
    score_ref = (score_ref * weights[0].cpu()).sum(dim=1)
    score_ref *= scales.reshape(-1)[: args.context].cpu()
    error = output[0, : args.context].cpu() - score_ref
    candidate_exact = True
    if args.tp > 1:
        reconstructed = torch.empty(args.context, dtype=torch.float32, device=device)
        packed_shards = []
        for rank in range(args.tp):
            shard_logits = invoke(rank, args.tp)
            local_len = max(0, (args.context - rank + args.tp - 1) // args.tp)
            global_ids = rank + torch.arange(local_len, device=device) * args.tp
            reconstructed[global_ids] = shard_logits[0, :local_len]

            local_indices = torch.full(
                (1, args.topk), -1, dtype=torch.int32, device=device
            )
            local_k = min(args.topk, local_len)
            if local_k:
                local_indices[0, :local_k] = torch.topk(
                    shard_logits[0, :local_len], local_k
                ).indices.to(torch.int32)
            packed_shards.append(
                quixicore_ops.indexer_pack_tp_candidates(
                    shard_logits, local_indices, rank, args.tp
                )
            )

        shard_error = reconstructed.cpu() - score_ref
        if shard_error.abs().max().item() > 0.02:
            raise SystemExit("token-sharded indexer logits correctness failed")
        gathered = torch.cat(packed_shards, dim=1).contiguous()
        candidate_scores, _ = quixicore_ops.indexer_unpack_tp_scores(gathered)
        candidate_positions = torch.topk(candidate_scores, args.topk, dim=1).indices.to(
            torch.int32
        )
        merged_ids = torch.empty_like(candidate_positions)
        quixicore_ops.indexer_resolve_tp_candidates(
            gathered, candidate_positions, merged_ids
        )
        expected_k = min(args.topk, args.context)
        expected_ids = torch.topk(output[0, : args.context], expected_k).indices
        candidate_exact = torch.equal(
            torch.sort(merged_ids[0, :expected_k].to(torch.int64)).values,
            torch.sort(expected_ids).values,
        )
        if not candidate_exact:
            raise SystemExit("token-sharded indexer top-k merge correctness failed")

    def measure(trivial_topk: int) -> float:
        for _ in range(args.warmup):
            invoke(0, args.tp, trivial_topk)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repetitions):
            invoke(0, args.tp, trivial_topk)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / args.repetitions

    us = measure(0)
    trivial_us = (
        measure(args.topk) if args.compare_trivial and args.context <= args.topk else None
    )
    print(
        f"context={args.context} heads={heads} max_model_len={args.max_model_len} "
        f"tp={args.tp} time_us={us:.3f} "
        f"max_error={error.abs().max().item():.6f} "
        f"mean_error={error.abs().mean().item():.6f}"
        f" candidate_exact={candidate_exact}"
        + (
            f" trivial_time_us={trivial_us:.3f} speedup={us / trivial_us:.3f}x"
            if trivial_us is not None
            else ""
        )
    )
    if error.abs().max().item() > 0.02:
        raise SystemExit("indexer correctness failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--context", type=int, default=1000)
    parser.add_argument("--heads", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--max-model-len", type=int, default=1_048_576)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--compare-trivial", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
