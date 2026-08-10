#!/usr/bin/env python3
"""Validate replicated-state, rank-owned-input DSV4 mHC on A100."""

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def graph_elapsed_us(
    comm: CustomAllreduce, fn, warmup: int, iterations: int
) -> float:
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        outputs = fn()
        comm.wait_dsv4_mhc(outputs[-1][0])
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--correctness-seeds", type=int, default=8)
    args = parser.parse_args()
    if args.transitions < 2:
        raise ValueError("ownership validation requires at least two transitions")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    generator = torch.Generator(device=device).manual_seed(20260809)
    xs = [
        (
            torch.randn((1, 4096), generator=generator, device=device) * 0.05
            + rank * 0.01
        ).to(torch.bfloat16)
        for _ in range(args.transitions)
    ]
    initial_residual = (
        torch.randn((1, 4, 4096), generator=generator, device=device) * 0.05
    ).to(torch.bfloat16)
    initial_post = torch.sigmoid(
        torch.randn((1, 4, 1), generator=generator, device=device)
    ).float()
    initial_comb = torch.softmax(
        torch.randn((1, 4, 4), generator=generator, device=device), dim=1
    ).float()
    fns = [
        (
            torch.randn(
                (24, 4 * 4096), generator=generator, device=device
            )
            * 0.01
        ).to(torch.float16)
        for _ in range(args.transitions)
    ]
    scales = [
        torch.tensor([0.3, 0.4, 0.5], device=device, dtype=torch.float32)
        for _ in range(args.transitions)
    ]
    bases = [
        torch.randn((24,), generator=generator, device=device).mul_(0.01)
        for _ in range(args.transitions)
    ]
    norm_weights = [
        torch.randn(
            (4096,), generator=generator, device=device, dtype=torch.bfloat16
        )
        for _ in range(args.transitions)
    ]
    mhc_args = (1e-6, 1e-6, 1e-6, 2.0, 20)

    def run_chain(mode: str) -> list[tuple[torch.Tensor, ...]]:
        own_urgent = mode != "legacy"
        own_deferred = mode == "full"
        local_input_owned = mode == "local"
        residual = initial_residual
        post = initial_post
        comb = initial_comb
        outputs = []
        for index in range(args.transitions):
            publish = own_deferred and index + 1 < args.transitions
            result = comm.all_reduce_dsv4_mhc(
                xs[index],
                None,
                residual,
                post,
                comb,
                fns[index],
                scales[index],
                bases[index],
                *mhc_args,
                norm_weights[index],
                1e-6,
                input_prepared=own_deferred and index > 0,
                own_projections=own_urgent,
                publish_prepared=publish,
                local_input_owned=local_input_owned,
            )
            outputs.append(result)
            residual, post, comb = result[:3]
        return outputs

    seed_exact: list[list[list[bool]]] = []
    max_abs_error: list[list[float]] = []
    first_transition_slice_errors: list[float] = []
    first_transition_layer_sample: list[float] = []
    first_transition_nonfinite: list[int] = []
    first_transition_reference_sample: list[float] = []
    first_transition_reference_nonfinite: list[int] = []
    for seed in range(args.correctness_seeds):
        generator.manual_seed(20260809 + seed)
        for index in range(args.transitions):
            xs[index].copy_(
                (
                    torch.randn((1, 4096), generator=generator, device=device)
                    * 0.2
                    + rank * 0.03
                ).to(torch.bfloat16)
            )
        initial_residual.copy_(
            (
                torch.randn(
                    (1, 4, 4096), generator=generator, device=device
                )
                * 0.2
            ).to(torch.bfloat16)
        )
        expected = run_chain("legacy")
        comm.wait_dsv4_mhc(expected[-1][0])
        actual = run_chain("local")
        comm.wait_dsv4_mhc(actual[-1][0])
        torch.cuda.synchronize()
        transition_exact = []
        transition_errors = []
        for index, (candidate, reference) in enumerate(zip(actual, expected)):
            local_hidden = 4096 // world_size
            hidden_start = rank * local_hidden
            quant_start = hidden_start // 32 * 9
            quant_values = local_hidden // 32 * 9
            reference_owned = (
                *reference[:3],
                reference[3][..., hidden_start : hidden_start + local_hidden],
                reference[4][..., quant_start : quant_start + quant_values],
            )
            if seed == 0 and index == 0:
                first_transition_layer_sample = candidate[3][0, :16].float().tolist()
                first_transition_nonfinite = (
                    (~torch.isfinite(candidate[3][0].float()))
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                reference_slice = reference[3][
                    0, hidden_start : hidden_start + local_hidden
                ].float()
                first_transition_reference_sample = reference_slice[:16].tolist()
                first_transition_reference_nonfinite = (
                    (~torch.isfinite(reference_slice))
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                first_transition_slice_errors = [
                    float(
                        (
                            candidate[3].float()
                            - reference[3][
                                ...,
                                peer * local_hidden : (peer + 1) * local_hidden,
                            ].float()
                        )
                        .abs()
                        .max()
                    )
                    for peer in range(world_size)
                ]
            observable = range(len(reference_owned))
            transition_exact.append(
                [
                    torch.equal(candidate[item], reference_owned[item])
                    for item in observable
                ]
            )
            transition_errors.append(
                max(
                    float(
                        (candidate[item].float() - reference_owned[item].float())
                        .abs()
                        .max()
                    )
                    for item in observable
                )
            )
        seed_exact.append(transition_exact)
        max_abs_error.append(transition_errors)

    legacy_us = graph_elapsed_us(
        comm, lambda: run_chain("legacy"), args.warmup, args.iterations
    )
    local_owned_us = graph_elapsed_us(
        comm, lambda: run_chain("local"), args.warmup, args.iterations
    )
    result = {
        "world_size": world_size,
        "transitions": args.transitions,
        "all_exact": all(
            exact
            for seed in seed_exact
            for transition in seed
            for exact in transition
        ),
        "seed_transition_exact": seed_exact,
        "seed_transition_max_abs_error": max_abs_error,
        "first_transition_layer_errors_by_reference_slice": (
            first_transition_slice_errors
        ),
        "first_transition_layer_sample": first_transition_layer_sample,
        "first_transition_nonfinite": first_transition_nonfinite[:32],
        "first_transition_reference_sample": first_transition_reference_sample,
        "first_transition_reference_nonfinite": (
            first_transition_reference_nonfinite[:32]
        ),
        "legacy_graph_us": legacy_us,
        "local_owned_graph_us": local_owned_us,
        "speedup": legacy_us / local_owned_us,
        "legacy_us_per_transition": legacy_us / args.transitions,
        "local_owned_us_per_transition": local_owned_us / args.transitions,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
