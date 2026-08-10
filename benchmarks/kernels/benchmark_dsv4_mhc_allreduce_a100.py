#!/usr/bin/env python3
"""Validate and time the TP-local DSV4 all-reduce + mHC transition."""

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm import _custom_ops as custom_ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.quantization.gguf import ops as gguf_ops
from vllm.quixicore import quixicore_ops


def elapsed_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def graph_elapsed_us(comm, fn, warmup: int, iterations: int) -> float:
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        outputs = fn()
        anchor = outputs[0] if isinstance(outputs, tuple) else outputs
        comm.wait_dsv4_mhc(anchor)
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--correctness-seeds", type=int, default=32)
    parser.add_argument(
        "--fn-dtype", choices=("float16", "float32"), default="float32"
    )
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    generator = torch.Generator(device=device).manual_seed(20260807)
    x = (
        torch.randn((1, 4096), generator=generator, device=device) * 0.05
        + rank * 0.01
    ).to(torch.bfloat16)
    addend = (
        torch.randn((1, 4096), generator=generator, device=device) * 0.03
        - rank * 0.005
    ).to(torch.bfloat16)
    residual = (
        torch.randn((1, 4, 4096), generator=generator, device=device) * 0.05
    ).to(torch.bfloat16)
    post = torch.sigmoid(
        torch.randn((1, 4, 1), generator=generator, device=device)
    ).float()
    comb = torch.softmax(
        torch.randn((1, 4, 4), generator=generator, device=device), dim=1
    ).float()
    fn_native = (
        torch.randn((24, 4 * 4096), generator=generator, device=device) * 0.01
    ).to(torch.float16)
    fn = fn_native if args.fn_dtype == "float16" else fn_native.float()
    scale = torch.tensor([0.3, 0.4, 0.5], device=device, dtype=torch.float32)
    base = torch.zeros(24, device=device, dtype=torch.float32)
    mhc_args = (1e-6, 1e-6, 1e-6, 2.0, 20)

    def reference():
        reduced = comm.all_reduce(x)
        return quixicore_ops.dsv4_mhc_fused_post_pre(
            reduced, residual, post, comb, fn, scale, base, *mhc_args
        )

    def fused():
        result = comm.all_reduce_dsv4_mhc(
            x, None, residual, post, comb, fn, scale, base, *mhc_args
        )
        assert result is not None
        return result

    reference_output = reference()
    reduced_for_expanded = comm.all_reduce(x)
    expanded_reference_output = quixicore_ops.dsv4_mhc_fused_post_pre(
        reduced_for_expanded,
        residual,
        post,
        comb,
        fn.float(),
        scale,
        base,
        *mhc_args,
    )
    fused_output = fused()
    torch.cuda.synchronize()
    errors = [
        float((actual.float().reshape(-1) - expected.float().reshape(-1)).abs().max())
        for actual, expected in zip(fused_output, reference_output)
    ]
    native_expanded_exact = [
        torch.equal(actual, expected)
        for actual, expected in zip(reference_output, expanded_reference_output)
    ]

    def add_reference():
        local = x + addend
        result = comm.all_reduce_dsv4_mhc(
            local, None, residual, post, comb, fn, scale, base, *mhc_args
        )
        assert result is not None
        return result

    def add_fused():
        result = comm.all_reduce_dsv4_mhc(
            x, addend, residual, post, comb, fn, scale, base, *mhc_args
        )
        assert result is not None
        return result

    add_reference_output = add_reference()
    add_fused_output = add_fused()
    torch.cuda.synchronize()
    add_errors = [
        float((actual.float().reshape(-1) - expected.float().reshape(-1)).abs().max())
        for actual, expected in zip(add_fused_output, add_reference_output)
    ]

    norm_weight = torch.randn(
        (4096,), generator=generator, device=device, dtype=torch.bfloat16
    )

    def norm_reference():
        result = fused()
        norm_output = torch.empty_like(result[3])
        custom_ops.rms_norm(norm_output, result[3], norm_weight, 1e-6)
        quant_output = gguf_ops.ggml_quantize_q8_1(norm_output)
        return (*result[:3], norm_output, quant_output)

    def norm_fused():
        return comm.all_reduce_dsv4_mhc(
            x,
            None,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            *mhc_args,
            norm_weight,
            1e-6,
        )

    def norm_separate_fused():
        result = fused()
        norm_output, quant_output = gguf_ops.ggml_dsv4_rms_norm_q8_1(
            result[3], norm_weight, 1e-6
        )
        return (*result[:3], norm_output, quant_output)

    def add_norm_reference():
        result = add_fused()
        norm_output = torch.empty_like(result[3])
        custom_ops.rms_norm(norm_output, result[3], norm_weight, 1e-6)
        quant_output = gguf_ops.ggml_quantize_q8_1(norm_output)
        return (*result[:3], norm_output, quant_output)

    def add_norm_fused():
        return comm.all_reduce_dsv4_mhc(
            x,
            addend,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            *mhc_args,
            norm_weight,
            1e-6,
        )

    norm_reference_output = norm_reference()
    norm_fused_output = norm_fused()
    norm_separate_fused_output = norm_separate_fused()
    add_norm_reference_output = add_norm_reference()
    add_norm_fused_output = add_norm_fused()
    torch.cuda.synchronize()
    norm_exact = [
        torch.equal(actual, expected)
        for actual, expected in zip(norm_fused_output, norm_reference_output)
    ]
    norm_separate_fused_exact = [
        torch.equal(actual, expected)
        for actual, expected in zip(
            norm_separate_fused_output, norm_reference_output
        )
    ]
    add_norm_exact = [
        torch.equal(actual, expected)
        for actual, expected in zip(
            add_norm_fused_output, add_norm_reference_output
        )
    ]
    norm_seed_exact: list[list[bool]] = []
    for seed in range(args.correctness_seeds):
        generator.manual_seed(20260807 + seed)
        x.copy_(
            (
                torch.randn((1, 4096), generator=generator, device=device) * 0.2
                + rank * 0.03
            ).to(torch.bfloat16)
        )
        residual.copy_(
            (
                torch.randn(
                    (1, 4, 4096), generator=generator, device=device
                )
                * 0.2
            ).to(torch.bfloat16)
        )
        post.copy_(
            torch.sigmoid(
                torch.randn((1, 4, 1), generator=generator, device=device)
            )
        )
        comb.copy_(
            torch.softmax(
                torch.randn((1, 4, 4), generator=generator, device=device),
                dim=1,
            )
        )
        fn.copy_(
            torch.randn(
                (24, 4 * 4096), generator=generator, device=device
            )
            .mul_(0.03)
            .to(torch.float16)
        )
        norm_weight.copy_(
            torch.randn(
                (4096,),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
        )
        expected = norm_reference()
        actual = norm_fused()
        torch.cuda.synchronize()
        norm_seed_exact.append(
            [torch.equal(a, e) for a, e in zip(actual, expected)]
        )

    graph_expected = norm_reference()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        graph_actual = norm_fused()
        comm.wait_dsv4_mhc(graph_actual[0])
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    graph_norm_exact = [
        torch.equal(actual, expected)
        for actual, expected in zip(graph_actual, graph_expected)
    ]

    result = {
        "world_size": world_size,
        "fn_dtype": args.fn_dtype,
        "native_expanded_exact": native_expanded_exact,
        "reference_us": elapsed_us(reference, args.warmup, args.iterations),
        "fused_us": elapsed_us(fused, args.warmup, args.iterations),
        "reference_graph_us": graph_elapsed_us(
            comm, reference, args.warmup, args.iterations
        ),
        "fused_graph_us": graph_elapsed_us(
            comm, fused, args.warmup, args.iterations
        ),
        "local_add_reference_us": elapsed_us(
            add_reference, args.warmup, args.iterations
        ),
        "local_add_fused_us": elapsed_us(add_fused, args.warmup, args.iterations),
        "local_add_reference_graph_us": graph_elapsed_us(
            comm, add_reference, args.warmup, args.iterations
        ),
        "local_add_fused_graph_us": graph_elapsed_us(
            comm, add_fused, args.warmup, args.iterations
        ),
        "norm_reference_graph_us": graph_elapsed_us(
            comm, norm_reference, args.warmup, args.iterations
        ),
        "norm_fused_graph_us": graph_elapsed_us(
            comm, norm_fused, args.warmup, args.iterations
        ),
        "norm_separate_fused_graph_us": graph_elapsed_us(
            comm, norm_separate_fused, args.warmup, args.iterations
        ),
        "local_add_norm_reference_graph_us": graph_elapsed_us(
            comm, add_norm_reference, args.warmup, args.iterations
        ),
        "local_add_norm_fused_graph_us": graph_elapsed_us(
            comm, add_norm_fused, args.warmup, args.iterations
        ),
        "norm_exact": norm_exact,
        "norm_separate_fused_exact": norm_separate_fused_exact,
        "local_add_norm_exact": add_norm_exact,
        "graph_norm_exact": graph_norm_exact,
        "norm_seed_exact": norm_seed_exact,
        "max_abs_error": errors,
        "local_add_max_abs_error": add_errors,
        "reference_post": reference_output[1].flatten().tolist(),
        "fused_post": fused_output[1].flatten().tolist(),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
