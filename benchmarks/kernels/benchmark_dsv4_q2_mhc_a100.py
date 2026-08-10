#!/usr/bin/env python3
"""Validate the DSV4 pending Q2_K producer-progress mHC boundary."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.quantization.gguf import ops as gguf_ops


def make_iq2_weights(
    experts: int, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    weights = torch.empty(
        (experts, rows, cols // 256, 66), dtype=torch.uint8, device=device
    )
    weights[..., 2:] = torch.randint(
        0, 128, weights[..., 2:].shape, dtype=torch.uint8, device=device
    )
    scale = torch.tensor([0.002], dtype=torch.float16, device=device).view(
        torch.uint8
    )
    weights[..., 0] = scale[0]
    weights[..., 1] = scale[1]
    return weights.flatten(2)


def make_q2_weights(
    experts: int, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    weights = torch.empty(
        (experts, rows, cols // 256, 84), dtype=torch.uint8, device=device
    )
    weights[..., :16] = 0x21
    weights[..., 16:80] = 0xE4
    scales = torch.tensor(
        [0.002, 0.001], dtype=torch.float16, device=device
    ).view(torch.uint8)
    weights[..., 80:84] = scales
    return weights.flatten(2)


def graph_us(
    comm: CustomAllreduce, fn, warmup: int, iterations: int
) -> tuple[float, tuple[torch.Tensor, ...]]:
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        outputs = fn()
        comm.wait_dsv4_mhc(outputs[0])
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4):
        raise RuntimeError("benchmark requires TP2 or TP4")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    hidden = 4096
    intermediate = 2048 // world_size
    top_k = 6
    generator = torch.Generator(device=device).manual_seed(20260808)
    x = (torch.randn((1, hidden), generator=generator, device=device) * 0.05).to(
        torch.bfloat16
    )
    quant_x = gguf_ops.ggml_quantize_q8_1(x)
    topk_ids = (
        (torch.arange(top_k, dtype=torch.int32, device=device) * 3 + rank)
        % args.experts
    ).view(1, top_k)
    topk_weights = torch.softmax(
        torch.randn((1, top_k), generator=generator, device=device), dim=-1
    )
    w1 = gguf_ops.ggml_dsv4_repack_iq2_xxs(
        make_iq2_weights(args.experts, 2 * intermediate, hidden, device), hidden
    )
    w2 = gguf_ops.ggml_dsv4_repack_q2_k(
        make_q2_weights(args.experts, hidden, intermediate, device), intermediate
    )
    dummy = topk_ids.contiguous()

    shared = (
        torch.randn((1, hidden), generator=generator, device=device) * 0.03
    ).to(torch.bfloat16)
    residual = (
        torch.randn((1, 4, hidden), generator=generator, device=device) * 0.05
    ).to(torch.bfloat16)
    post = torch.sigmoid(
        torch.randn((1, 4, 1), generator=generator, device=device)
    ).float()
    comb = torch.softmax(
        torch.randn((1, 4, 4), generator=generator, device=device), dim=1
    ).float()
    fn = (
        torch.randn((24, 4 * hidden), generator=generator, device=device) * 0.01
    ).to(torch.float16)
    scale = torch.tensor([0.3, 0.4, 0.5], device=device, dtype=torch.float32)
    base = torch.zeros(24, device=device, dtype=torch.float32)
    norm_weight = torch.randn(
        (hidden,), generator=generator, device=device, dtype=torch.bfloat16
    )
    mhc_args = (1e-6, 1e-6, 1e-6, 2.0, 20, norm_weight, 1e-6)

    def moe(defer_down: bool) -> torch.Tensor:
        return gguf_ops.ggml_dsv4_moe_a8(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            dummy,
            dummy,
            dummy,
            dummy,
            intermediate,
            hidden,
            top_k,
            1,
            10.0,
            True,
            True,
            quant_x,
            defer_down,
        )

    def reference() -> tuple[torch.Tensor, ...]:
        routed = moe(False)
        return comm.all_reduce_dsv4_mhc(
            routed,
            shared,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            *mhc_args,
        )

    def candidate() -> tuple[torch.Tensor, ...]:
        pending = moe(True)
        return comm.all_reduce_dsv4_q2_mhc(
            pending,
            shared,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            *mhc_args,
        )

    expected = reference()
    actual = candidate()
    torch.cuda.synchronize()
    exact = [torch.equal(a, b) for a, b in zip(actual, expected)]
    max_error = [
        float((a.float() - b.float()).abs().max())
        for a, b in zip(actual, expected)
    ]
    reference_us, _ = graph_us(comm, reference, args.warmup, args.iterations)
    candidate_us, _ = graph_us(comm, candidate, args.warmup, args.iterations)

    routed_static = moe(False)
    torch.cuda.synchronize()

    def reference_transition() -> tuple[torch.Tensor, ...]:
        return comm.all_reduce_dsv4_mhc(
            routed_static,
            shared,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            *mhc_args,
        )

    materialized_moe_us, _ = graph_us(
        comm, lambda: (moe(False),), args.warmup, args.iterations
    )
    pending_moe_us, _ = graph_us(
        comm, lambda: (moe(True),), args.warmup, args.iterations
    )
    reference_transition_us, _ = graph_us(
        comm, reference_transition, args.warmup, args.iterations
    )

    result = {
        "rank": rank,
        "world_size": world_size,
        "intermediate": intermediate,
        "exact": exact,
        "max_error": max_error,
        "reference_graph_us": reference_us,
        "candidate_graph_us": candidate_us,
        "speedup": reference_us / candidate_us,
        "materialized_moe_us": materialized_moe_us,
        "pending_moe_us": pending_moe_us,
        "standalone_q2_us": materialized_moe_us - pending_moe_us,
        "reference_transition_us": reference_transition_us,
        "candidate_boundary_us": candidate_us - pending_moe_us,
        "layer_mismatches": int((actual[3] != expected[3]).sum()),
        "layer_expected_head": expected[3].flatten()[:8].float().tolist(),
        "layer_actual_head": actual[3].flatten()[:8].float().tolist(),
        "layer_expected_rms": float(expected[3].float().square().mean().sqrt()),
        "layer_actual_rms": float(actual[3].float().square().mean().sqrt()),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    if not all(exact):
        raise SystemExit("pending Q2_K mHC output mismatch")
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
