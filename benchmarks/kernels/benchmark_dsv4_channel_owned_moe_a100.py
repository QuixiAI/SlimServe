#!/usr/bin/env python3
"""Validate the native DSV4 W1-to-output-stationary-W2 TP boundary."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.quantization.gguf import ops


def make_q2_weights(
    experts: int, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    blocks_per_row = cols // 256
    weights = torch.empty(
        (experts, rows, blocks_per_row, 84), dtype=torch.uint8, device=device
    )
    weights[..., :80] = torch.randint(
        0, 256, weights[..., :80].shape, dtype=torch.uint8, device=device
    )
    dm = torch.tensor([0.00002, 0.00001], dtype=torch.float16, device=device).view(
        torch.uint8
    )
    weights[..., 80:84] = dm
    return weights.view(experts, rows, blocks_per_row * 84)


def make_iq2_weights(
    experts: int, rows: int, hidden: int, device: torch.device, seed: int
) -> torch.Tensor:
    block_bytes = 66
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0,
        256,
        (experts, rows, hidden // 256 * block_bytes),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )


def make_q8_weights(
    rows: int, cols: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    blocks = cols // 32
    weights = torch.empty((rows, blocks, 34), dtype=torch.uint8, device=device)
    scales = torch.full(
        (rows, blocks), 0.0002, dtype=torch.float16, device=device
    ).view(torch.uint8)
    weights[..., :2] = scales.view(rows, blocks, 2)
    weights[..., 2:] = torch.randint(
        0,
        256,
        (rows, blocks, 32),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    return weights.view(rows, blocks * 34)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4):
        raise ValueError("channel-owned DSV4 MoE requires TP2 or TP4")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllreduce(group=cpu_group, device=device, max_size=1 << 20)
    if comm.disabled:
        raise RuntimeError("vLLM custom all-reduce is unavailable")

    hidden = 4096
    full_intermediate = 2048
    local_intermediate = full_intermediate // world_size
    local_rows = hidden // world_size
    top_k = 6
    generator = torch.Generator(device=device).manual_seed(2026080900)
    x = torch.randn((1, hidden), generator=generator, device=device).to(
        torch.bfloat16
    )
    quant_x = ops.ggml_quantize_q8_1(x)
    topk_ids = ((torch.arange(top_k, device=device) * 7) % args.experts).to(
        torch.int32
    )
    topk_weights = torch.softmax(
        torch.randn((1, top_k), generator=generator, device=device), dim=-1
    )
    raw_w1 = make_iq2_weights(
        args.experts,
        2 * local_intermediate,
        hidden,
        device,
        2026080910 + rank,
    )
    w1 = ops.ggml_dsv4_repack_iq2_xxs(raw_w1, hidden)
    raw_w2 = make_q2_weights(args.experts, local_rows, full_intermediate, device)
    w2 = ops.ggml_dsv4_repack_q2_k(raw_w2, full_intermediate)

    shared_values = []
    shared_quants = []
    for peer in range(world_size):
        peer_generator = torch.Generator(device=device).manual_seed(
            2026080930 + peer
        )
        peer_values = torch.randn(
            (1, local_intermediate), generator=peer_generator, device=device
        ).to(torch.bfloat16)
        shared_values.append(peer_values)
        shared_quants.append(ops.ggml_quantize_q8_1(peer_values))
    shared_quant = shared_quants[rank]
    full_shared = torch.cat(shared_values, dim=1)
    full_shared_quant = torch.cat(shared_quants, dim=1)
    raw_shared_w2 = make_q8_weights(
        local_rows, full_intermediate, device, 2026080940 + rank
    )
    shared_w2 = ops.ggml_dsv4_repack_q8_0_aligned(raw_shared_w2)

    addends = []
    for peer in range(world_size):
        peer_generator = torch.Generator(device=device).manual_seed(2026080920 + peer)
        addends.append(
            torch.randn(
                (1, hidden), generator=peer_generator, device=device
            ).mul_(0.01).to(torch.bfloat16)
        )
    addend = addends[rank]
    row_start = rank * local_rows

    ignored = topk_ids.view(1, -1).contiguous()
    padded = torch.tensor([top_k], dtype=torch.int32, device=device)

    def make_pending() -> torch.Tensor:
        return ops.ggml_dsv4_moe_a8(
            x,
            w1,
            w2,
            topk_weights,
            ignored,
            ignored,
            ignored,
            ignored,
            padded,
            local_intermediate,
            local_rows,
            top_k,
            1,
            7.0,
            True,
            True,
            quant_x,
            True,
        )

    pending = make_pending()
    payload_ints = top_k * (local_intermediate // 32 * 9)
    quant_mid = pending.view(torch.int32).flatten()[:payload_ints].view(
        top_k, local_intermediate // 32 * 9
    )
    owned_w1_quant = ops.ggml_dsv4_moe_w1_a8(
        x,
        w1,
        topk_weights,
        ignored,
        ignored,
        ignored,
        padded,
        local_intermediate,
        top_k,
        1,
        7.0,
        True,
        quant_x,
    )
    if not torch.equal(owned_w1_quant, quant_mid):
        raise AssertionError("output-owned W1 changed the packed Q8_1 handoff")
    peer_quant_cpu: list[torch.Tensor | None] = [None] * world_size
    dist.all_gather_object(peer_quant_cpu, owned_w1_quant.cpu())
    full_quant_mid = torch.cat(
        [peer.to(device) for peer in peer_quant_cpu if peer is not None], dim=-1
    )
    routed_reference = comm.dsv4_channel_owned_q2_down(quant_mid, w2, topk_ids)
    owned_down = ops.ggml_dsv4_moe_down_output_owned(
        w2, full_quant_mid, ignored, 1, top_k
    )
    routed_actual = comm.dsv4_channel_owned_q2_down_pending(
        pending, torch.zeros_like(addend)
    )
    routed_error = (routed_actual.float() - routed_reference.float()).abs()
    owned_down_error = (owned_down.float() - routed_reference.float()).abs()
    expected_float = routed_reference.float()
    for peer_addend in addends:
        expected_float.add_(
            peer_addend[:, row_start : row_start + local_rows].float()
        )
    expected = expected_float.to(torch.bfloat16)
    actual = comm.dsv4_channel_owned_q2_down_pending(pending, addend)
    producer_owned = comm.dsv4_channel_owned_moe(
        quant_x, w1, w2, topk_weights, topk_ids, addend
    )
    shared_reference = ops.ggml_dsv4_mul_mat_vec_aligned_q8_0(
        shared_w2,
        full_shared,
        full_shared_quant,
        local_rows,
        1,
    )
    output_owned_expected = routed_reference + shared_reference
    output_owned = comm.dsv4_output_owned_moe(
        quant_x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        shared_quant,
        shared_w2,
    )
    torch.cuda.synchronize()
    error = (actual.float() - expected.float()).abs()
    producer_owned_error = (producer_owned.float() - expected.float()).abs()
    output_owned_error = (
        output_owned.float() - output_owned_expected.float()
    ).abs()
    if float(routed_error.max()) != 0.0:
        raise AssertionError(
            f"pending handoff changed routed output: {float(routed_error.max())}"
        )
    if float(owned_down_error.max()) > 0.015625:
        raise AssertionError(
            "output-owned full-K down exceeded BF16 tolerance: "
            f"{float(owned_down_error.max())}"
        )
    if float(error.max()) != 0.0:
        raise AssertionError(
            f"shared folding changed rank-ordered output: {float(error.max())}"
        )
    if float(producer_owned_error.max()) != 0.0:
        raise AssertionError(
            "producer-owned fused path changed output: "
            f"{float(producer_owned_error.max())}"
        )
    if float(output_owned_error.max()) > 0.03125:
        raise AssertionError(
            "output-owned shared Q8 fold exceeded BF16 tolerance: "
            f"{float(output_owned_error.max())}"
        )
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with comm.capture(), torch.cuda.graph(graph):
        captured_pending = make_pending()
        captured = comm.dsv4_channel_owned_q2_down_pending(
            captured_pending, addend
        )
    dist.barrier()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.iterations):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / args.iterations
    if not torch.isfinite(captured).all():
        raise AssertionError("channel-owned DSV4 MoE produced non-finite values")

    w1_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(w1_graph):
        w1_graph_pending = make_pending()
    q2_graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(q2_graph):
        q2_graph_output = comm.dsv4_channel_owned_q2_down_pending(
            pending, addend
        )
    producer_owned_graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(producer_owned_graph):
        producer_owned_graph_output = comm.dsv4_channel_owned_moe(
            quant_x, w1, w2, topk_weights, topk_ids, addend
        )
    output_owned_graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(output_owned_graph):
        output_owned_graph_output = comm.dsv4_output_owned_moe(
            quant_x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            shared_quant,
            shared_w2,
        )
    def graph_time_us(candidate: torch.cuda.CUDAGraph) -> float:
        dist.barrier()
        for _ in range(args.warmup):
            candidate.replay()
        torch.cuda.synchronize()
        dist.barrier()
        graph_start = time.perf_counter()
        for _ in range(args.iterations):
            candidate.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - graph_start) * 1e6 / args.iterations

    w1_us = graph_time_us(w1_graph)
    q2_us = graph_time_us(q2_graph)
    producer_owned_us = graph_time_us(producer_owned_graph)
    output_owned_us = graph_time_us(output_owned_graph)
    # w1_graph_pending is opaque Q8/header storage viewed as BF16 and may
    # legally contain NaN bit patterns. Only the consumed output is numeric.
    if not torch.isfinite(q2_graph_output).all():
        raise AssertionError("component graph produced non-finite values")
    if not torch.isfinite(producer_owned_graph_output).all():
        raise AssertionError("producer-owned graph produced non-finite values")
    if not torch.isfinite(output_owned_graph_output).all():
        raise AssertionError("output-owned graph produced non-finite values")
    producer_owned_graph_error = (
        producer_owned_graph_output.float() - expected.float()
    ).abs()
    if float(producer_owned_graph_error.max()) != 0.0:
        raise AssertionError(
            "captured producer-owned path changed output: "
            f"{float(producer_owned_graph_error.max())}"
        )
    output_owned_graph_error = (
        output_owned_graph_output.float() - output_owned_expected.float()
    ).abs()
    if float(output_owned_graph_error.max()) > 0.03125:
        raise AssertionError(
            "captured output-owned path exceeded BF16 tolerance: "
            f"{float(output_owned_graph_error.max())}"
        )
    result = {
        "rank": rank,
        "world_size": world_size,
        "experts": args.experts,
        "local_intermediate": local_intermediate,
        "local_rows": local_rows,
        "graph_w1_q2_shared_us": elapsed_us,
        "graph_w1_us": w1_us,
        "graph_q2_shared_us": q2_us,
        "graph_producer_owned_moe_us": producer_owned_us,
        "graph_output_owned_moe_us": output_owned_us,
        "routed_max_abs_error": float(routed_error.max()),
        "output_owned_down_max_abs_error": float(owned_down_error.max()),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "producer_owned_max_abs_error": float(producer_owned_error.max()),
        "producer_owned_graph_max_abs_error": float(
            producer_owned_graph_error.max()
        ),
        "output_owned_max_abs_error": float(output_owned_error.max()),
        "output_owned_mean_abs_error": float(output_owned_error.mean()),
        "output_owned_graph_max_abs_error": float(
            output_owned_graph_error.max()
        ),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(json.dumps(gathered, indent=2))
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
