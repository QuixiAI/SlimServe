# SPDX-License-Identifier: Apache-2.0
"""Benchmark the native DSV4 IQ2_XXS/SwiGLU/Q2_K A100 MoE path."""

import argparse
import hashlib
import json
import statistics

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.gguf import ops

IQ2_XXS = 16
Q2_K = 10


def make_iq2_weights(
    experts: int, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    blocks_per_row = cols // 256
    weights = torch.empty(
        (experts, rows, blocks_per_row, 66), dtype=torch.uint8, device=device
    )
    weights[..., 2:] = torch.randint(
        0, 128, weights[..., 2:].shape, dtype=torch.uint8, device=device
    )
    scale = torch.tensor([0.002], dtype=torch.float16, device=device).view(
        torch.uint8
    )
    weights[..., 0] = scale[0]
    weights[..., 1] = scale[1]
    return weights.view(experts, rows, blocks_per_row * 66)


def make_q2_weights(
    experts: int, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    blocks_per_row = cols // 256
    weights = torch.empty(
        (experts, rows, blocks_per_row, 84), dtype=torch.uint8, device=device
    )
    weights[..., :16] = 0x21
    weights[..., 16:80] = 0xE4
    scales = torch.tensor(
        [0.002, 0.001], dtype=torch.float16, device=device
    ).view(torch.uint8)
    weights[..., 80:84] = scales
    return weights.view(experts, rows, blocks_per_row * 84)


def median_us(
    function, warmup: int, iterations: int, inner_repeats: int
) -> float:
    for _ in range(warmup):
        for _ in range(inner_repeats):
            function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_repeats):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / inner_repeats)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--inner-repeats", type=int, default=1)
    args = parser.parse_args()

    if args.hidden % 256 or args.intermediate % 256:
        raise ValueError("hidden and intermediate must be multiples of 256")

    device = torch.device("cuda")
    torch.manual_seed(7)
    x = (torch.randn(args.tokens, args.hidden, device=device) * 0.05).to(
        torch.bfloat16
    )
    route = torch.arange(
        args.tokens * args.top_k, dtype=torch.int32, device=device
    )
    topk_ids = ((route * 7) % args.experts).view(args.tokens, args.top_k)
    topk_weights = torch.softmax(
        torch.randn(args.tokens, args.top_k, device=device), dim=-1
    )
    w1 = make_iq2_weights(
        args.experts, 2 * args.intermediate, args.hidden, device
    )
    w2 = make_q2_weights(
        args.experts, args.hidden, args.intermediate, device
    )
    w1_repacked = ops.ggml_dsv4_repack_iq2_xxs(w1, args.hidden)
    w2_repacked = ops.ggml_dsv4_repack_q2_k(w2, args.intermediate)
    quant_x = ops.ggml_quantize_q8_1(x)

    routed_rows = args.tokens * args.top_k
    w1_width = 8 if routed_rows >= 256 else 4
    sorted_ids, w1_expert_ids, padded = moe_align_block_size(
        topk_ids, w1_width, args.experts
    )
    w2_expert_ids = (
        w1_expert_ids
        if w1_width == 4
        else w1_expert_ids.repeat_interleave(2)
    )

    def fused_grouped() -> torch.Tensor:
        return ops.ggml_dsv4_moe_a8(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            sorted_ids,
            w1_expert_ids,
            w2_expert_ids,
            padded,
            args.intermediate,
            args.hidden,
            args.top_k,
            args.tokens,
            args.swiglu_limit,
            False,
            False,
        )

    def fused_repacked() -> torch.Tensor:
        return ops.ggml_dsv4_moe_a8(
            x,
            w1,
            w2_repacked,
            topk_weights,
            topk_ids,
            sorted_ids,
            w1_expert_ids,
            w2_expert_ids,
            padded,
            args.intermediate,
            args.hidden,
            args.top_k,
            args.tokens,
            args.swiglu_limit,
            False,
            True,
        )

    def fused_repacked_prequant() -> torch.Tensor:
        return ops.ggml_dsv4_moe_a8(
            x,
            w1,
            w2_repacked,
            topk_weights,
            topk_ids,
            sorted_ids,
            w1_expert_ids,
            w2_expert_ids,
            padded,
            args.intermediate,
            args.hidden,
            args.top_k,
            args.tokens,
            args.swiglu_limit,
            False,
            True,
            quant_x,
        )

    def fused_aligned_prequant() -> torch.Tensor:
        return ops.ggml_dsv4_moe_a8(
            x,
            w1_repacked,
            w2_repacked,
            topk_weights,
            topk_ids,
            sorted_ids,
            w1_expert_ids,
            w2_expert_ids,
            padded,
            args.intermediate,
            args.hidden,
            args.top_k,
            args.tokens,
            args.swiglu_limit,
            True,
            True,
            quant_x,
        )

    # The generic IQ2 kernel is four routes wide. Expand an eight-wide
    # schedule's expert map so this reference sees the same sorted rows.
    generic_w1_expert_ids = (
        w1_expert_ids
        if w1_width == 4
        else w1_expert_ids.repeat_interleave(2)
    )

    def generic() -> torch.Tensor:
        gate_up = ops.ggml_moe_a8(
            x,
            w1,
            sorted_ids,
            generic_w1_expert_ids,
            padded,
            IQ2_XXS,
            2 * args.intermediate,
            args.top_k,
            args.tokens,
        )
        gate = gate_up[:, : args.intermediate].float().clamp(
            max=args.swiglu_limit
        )
        up = gate_up[:, args.intermediate :].float().clamp(
            min=-args.swiglu_limit, max=args.swiglu_limit
        )
        mid = F.silu(gate) * up
        down = ops.ggml_moe_a8(
            mid,
            w2,
            sorted_ids,
            w2_expert_ids,
            padded,
            Q2_K,
            args.hidden,
            1,
            routed_rows,
        )
        return (
            down.float().view(args.tokens, args.top_k, args.hidden)
            * topk_weights[:, :, None]
        ).sum(dim=1)

    baseline_actual = fused_repacked()
    prequant_actual = fused_repacked_prequant()
    if not torch.equal(baseline_actual, prequant_actual):
        raise AssertionError("prequantized native DSV4 MoE output differs")
    aligned_actual = fused_aligned_prequant()
    if not torch.equal(prequant_actual, aligned_actual):
        mismatch = int((prequant_actual != aligned_actual).sum())
        raise AssertionError(
            f"aligned IQ2_XXS native output differs at {mismatch} elements"
        )
    actual = aligned_actual.float()
    output_sha256 = hashlib.sha256(
        aligned_actual.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()
    reference = generic()
    torch.cuda.synchronize()
    error = (actual - reference).abs()
    if not torch.isfinite(actual).all():
        raise AssertionError("native DSV4 output contains non-finite values")

    raw_w1_us = median_us(
        fused_repacked_prequant,
        args.warmup,
        args.iterations,
        args.inner_repeats,
    )
    fused_us = median_us(
        fused_aligned_prequant,
        args.warmup,
        args.iterations,
        args.inner_repeats,
    )
    grouped_us = median_us(
        fused_grouped, args.warmup, args.iterations, args.inner_repeats
    )
    generic_us = median_us(
        generic, args.warmup, args.iterations, args.inner_repeats
    )
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "routed_rows": routed_rows,
                "experts": args.experts,
                "hidden": args.hidden,
                "local_intermediate": args.intermediate,
                "inner_repeats": args.inner_repeats,
                "w1_tile_width": w1_width,
                "swiglu_limit": args.swiglu_limit,
                "padded_rows": int(padded.item()),
                "fused_us": round(fused_us, 3),
                "raw_w1_us": round(raw_w1_us, 3),
                "aligned_w1_speedup": round(raw_w1_us / fused_us, 4),
                "grouped_down_us": round(grouped_us, 3),
                "generic_us": round(generic_us, 3),
                "speedup_vs_grouped": round(grouped_us / fused_us, 4),
                "speedup": round(generic_us / fused_us, 4),
                "mean_abs_error": float(error.mean()),
                "max_abs_error": float(error.max()),
                "output_sha256": output_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
