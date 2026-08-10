#!/usr/bin/env python3
"""Validate channel-owned DSV4 mHC and compare its graph latency."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch


def graph_us(comm: CustomAllreduce, fn, warmup: int, iterations: int) -> float:
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
    return (time.perf_counter() - start) * 1e6 / iterations


def main() -> None:
    os.environ.setdefault("VLLM_DSV4_MHC_SCHEDULE", "async")
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--correctness-seeds", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise ValueError("channel-owned mHC requires TP2, TP4, or TP8")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    local_hidden = 4096 // world_size
    channel_slice = slice(rank * local_hidden, (rank + 1) * local_hidden)
    generator = torch.Generator(device=device).manual_seed(20260809)
    xs_full = [
        (torch.randn((1, 4096), generator=generator, device=device) * 0.1).to(
            torch.bfloat16
        )
        for _ in range(args.transitions)
    ]
    addends_full = [
        (torch.randn((1, 4096), generator=generator, device=device) * 0.1).to(
            torch.bfloat16
        )
        for _ in range(args.transitions)
    ]
    reduced_full = [torch.empty_like(x) for x in xs_full]
    xs_local = [x[:, channel_slice].contiguous() for x in xs_full]
    residual_full = (
        torch.randn((1, 4, 4096), generator=generator, device=device) * 0.1
    ).to(torch.bfloat16)
    residual_local = residual_full[:, :, channel_slice].contiguous()
    initial_post = torch.sigmoid(
        torch.randn((1, 4, 1), generator=generator, device=device)
    ).float()
    initial_comb = torch.softmax(
        torch.randn((1, 4, 4), generator=generator, device=device), dim=1
    ).float()
    fns = [
        (
            torch.randn((24, 4 * 4096), generator=generator, device=device)
            * 0.01
        ).to(torch.float16)
        for _ in range(args.transitions)
    ]
    scales = [
        torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32, device=device)
        for _ in range(args.transitions)
    ]
    bases = [
        torch.randn((24,), generator=generator, device=device).mul_(0.01)
        for _ in range(args.transitions)
    ]
    norm_weights = [
        torch.randn((4096,), generator=generator, device=device).to(torch.bfloat16)
        for _ in range(args.transitions)
    ]
    eps = 1e-6
    sinkhorn_repeat = 20

    def run_owned():
        residual = residual_local
        post = initial_post
        comb = initial_comb
        outputs = []
        for index in range(args.transitions):
            result = comm.dsv4_channel_owned_mhc(
                xs_local[index],
                residual,
                post,
                comb,
                fns[index],
                scales[index],
                bases[index],
                norm_weights[index],
                eps,
                eps,
                eps,
                2.0,
                sinkhorn_repeat,
                eps,
            )
            outputs.append(result)
            residual, post, comb = result[:3]
        comm.wait_dsv4_mhc(outputs[-1][0])
        return outputs

    def run_reduced_owned():
        residual = residual_local
        post = initial_post
        comb = initial_comb
        outputs = []
        for index in range(args.transitions):
            result = comm.dsv4_channel_owned_mhc(
                xs_full[index],
                residual,
                post,
                comb,
                fns[index],
                scales[index],
                bases[index],
                norm_weights[index],
                eps,
                eps,
                eps,
                2.0,
                sinkhorn_repeat,
                eps,
                addends_full[index],
            )
            outputs.append(result)
            residual, post, comb = result[:3]
        comm.wait_dsv4_mhc(outputs[-1][0])
        return outputs

    def run_separate_owned():
        residual = residual_local
        post = initial_post
        comb = initial_comb
        outputs = []
        for index in range(args.transitions):
            reduced = comm.dsv4_owned_reduce_scatter(
                xs_full[index], addends_full[index]
            )
            result = comm.dsv4_channel_owned_mhc(
                reduced,
                residual,
                post,
                comb,
                fns[index],
                scales[index],
                bases[index],
                norm_weights[index],
                eps,
                eps,
                eps,
                2.0,
                sinkhorn_repeat,
                eps,
            )
            outputs.append(result)
            residual, post, comb = result[:3]
        comm.wait_dsv4_mhc(outputs[-1][0])
        return outputs

    def run_reference():
        residual = residual_full
        post = initial_post
        comb = initial_comb
        outputs = []
        for index in range(args.transitions):
            residual = mhc_post_torch(xs_full[index], residual, post, comb)
            residual_float = residual.float().view(1, -1)
            projection = residual_float @ fns[index].float().t()
            residual_inverse_rms = torch.rsqrt(
                residual_float.square().mean(dim=-1) + eps
            )
            pre = (
                torch.sigmoid(
                    projection[:, :4]
                    * residual_inverse_rms[:, None]
                    * scales[index][0]
                    + bases[index][:4]
                )
                + eps
            )
            post, comb, layer_input = mhc_pre_torch(
                residual,
                fns[index].float(),
                scales[index],
                bases[index],
                eps,
                eps,
                eps,
                2.0,
                sinkhorn_repeat,
            )
            normalized = (
                layer_input.float()
                * torch.rsqrt(
                    layer_input.float().square().mean(dim=-1, keepdim=True) + eps
                )
                * norm_weights[index].float()
            ).to(torch.bfloat16)
            outputs.append(
                (
                    residual,
                    post,
                    comb,
                    normalized,
                    pre,
                    layer_input,
                    residual_inverse_rms,
                    projection,
                )
            )
        return outputs

    def run_legacy():
        residual = residual_full
        post = initial_post
        comb = initial_comb
        outputs = []
        for index in range(args.transitions):
            local_partial = (
                xs_full[index]
                if rank == 0
                else torch.zeros_like(xs_full[index])
            )
            result = comm.all_reduce_dsv4_mhc(
                local_partial,
                None,
                residual,
                post,
                comb,
                fns[index],
                scales[index],
                bases[index],
                eps,
                eps,
                eps,
                2.0,
                sinkhorn_repeat,
                norm_weights[index],
                eps,
            )
            outputs.append(result)
            residual, post, comb = result[:3]
        comm.wait_dsv4_mhc(outputs[-1][0])
        return outputs

    reduced_graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(reduced_graph):
        reduced_actual = run_reduced_owned()
    dist.barrier()

    correctness = []
    for seed in range(args.correctness_seeds):
        generator.manual_seed(20260809 + seed)
        for full, local in zip(xs_full, xs_local):
            full.copy_(
                (
                    torch.randn(full.shape, generator=generator, device=device)
                    * 0.2
                    + rank * 0.015625
                ).to(torch.bfloat16)
            )
            local.copy_(full[:, channel_slice])
        for addend in addends_full:
            addend.copy_(
                (
                    torch.randn(
                        addend.shape, generator=generator, device=device
                    )
                    * 0.2
                    - rank * 0.0078125
                ).to(torch.bfloat16)
            )
        for index, reduced in enumerate(reduced_full):
            combined = (xs_full[index] + addends_full[index]).to(torch.bfloat16)
            combined_cpu = combined.cpu()
            dist.all_reduce(combined_cpu)
            reduced.copy_(combined_cpu.to(device))
        residual_full.copy_(
            (
                torch.randn(
                    residual_full.shape, generator=generator, device=device
                )
                * 0.2
            ).to(torch.bfloat16)
        )
        residual_local.copy_(residual_full[:, :, channel_slice])
        expected = run_reference()
        original_x = [x.clone() for x in xs_full]
        for full, reduced in zip(xs_full, reduced_full):
            full.copy_(reduced)
        expected_reduced = run_reference()
        for full, original in zip(xs_full, original_x):
            full.copy_(original)
        dist.barrier()
        actual = run_owned()
        reduced_graph.replay()
        torch.cuda.synchronize()
        seed_results = []
        for candidate, reduced_candidate, reference, reduced_reference in zip(
            actual, reduced_actual, expected, expected_reduced
        ):
            pairs = (
                (candidate[0], reference[0][:, :, channel_slice]),
                (candidate[1], reference[1]),
                (candidate[2], reference[2]),
                (candidate[3], reference[3][:, channel_slice]),
            )
            reduced_pairs = (
                (reduced_candidate[0], reduced_reference[0][:, :, channel_slice]),
                (reduced_candidate[1], reduced_reference[1]),
                (reduced_candidate[2], reduced_reference[2]),
                (reduced_candidate[3], reduced_reference[3][:, channel_slice]),
            )
            seed_results.append(
                {
                    "outputs": [
                        {
                            "max_abs": float((a.float() - b.float()).abs().max()),
                            "mean_abs": float((a.float() - b.float()).abs().mean()),
                        }
                        for a, b in pairs
                    ],
                    "reduced_outputs": [
                        {
                            "max_abs": float((a.float() - b.float()).abs().max()),
                            "mean_abs": float((a.float() - b.float()).abs().mean()),
                        }
                        for a, b in reduced_pairs
                    ],
                    "control": {
                        "pre_max_abs": float(
                            (candidate[5][:4] - reference[4].flatten()).abs().max()
                        ),
                        "residual_inverse_rms_abs": float(
                            (
                                candidate[5][4]
                                - reference[6].flatten()[0]
                            ).abs()
                        ),
                        "input_inverse_rms_abs": float(
                            (
                                candidate[5][5]
                                - torch.rsqrt(
                                    reference[5].float().square().mean() + eps
                                )
                            ).abs()
                        ),
                        "local_input_square_sum_abs": float(
                            (
                                candidate[5][6]
                                - reference[5][:, channel_slice]
                                .float()
                                .square()
                                .sum()
                            ).abs()
                        ),
                        "urgent_projection_max_abs": float(
                            (
                                candidate[5][8:12]
                                - reference[7].flatten()[:4]
                            ).abs().max()
                        ),
                        "residual_square_sum_abs": float(
                            (
                                candidate[5][12]
                                - reference[0].float().square().sum()
                            ).abs()
                        ),
                        "prepared_deferred_max_abs": float(
                            (
                                candidate[5][13:33]
                                - torch.cat(
                                    (reference[1].flatten(), reference[2].flatten())
                                )
                            ).abs().max()
                        ),
                    },
                }
            )
        correctness.append(seed_results)

    owned_us = graph_us(comm, run_owned, args.warmup, args.iterations)
    reduced_owned_us = graph_us(
        comm, run_reduced_owned, args.warmup, args.iterations
    )
    separate_owned_us = graph_us(
        comm, run_separate_owned, args.warmup, args.iterations
    )
    legacy_us = graph_us(comm, run_legacy, args.warmup, args.iterations)
    max_abs = max(
        item["max_abs"]
        for seed in correctness
        for transition in seed
        for item in transition["outputs"]
    )
    reduced_max_abs = max(
        item["max_abs"]
        for seed in correctness
        for transition in seed
        for item in transition["reduced_outputs"]
    )
    result = {
        "rank": rank,
        "world_size": world_size,
        "transitions": args.transitions,
        "max_abs_error": max_abs,
        "reduced_max_abs_error": reduced_max_abs,
        "correctness": correctness,
        "owned_graph_us": owned_us,
        "reduced_owned_graph_us": reduced_owned_us,
        "separate_owned_graph_us": separate_owned_us,
        "legacy_graph_us": legacy_us,
        "owned_us_per_transition": owned_us / args.transitions,
        "reduced_owned_us_per_transition": reduced_owned_us / args.transitions,
        "separate_owned_us_per_transition": separate_owned_us / args.transitions,
        "legacy_us_per_transition": legacy_us / args.transitions,
        "speedup": legacy_us / owned_us,
        "reduce_fusion_speedup": separate_owned_us / reduced_owned_us,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
