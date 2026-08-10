# SPDX-License-Identifier: Apache-2.0
"""Correctness and latency checks for the native Ampere DSV4 mHC kernels."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from vllm import _custom_ops as custom_ops
from vllm.model_executor.kernels.mhc.torch import (
    hc_head_fused_torch,
    mhc_post_torch,
    mhc_pre_torch,
)
from vllm.quixicore.ops import quixicore_ops


def elapsed_us(call: Callable[[], object], warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repetitions


def error(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    delta = (actual.float() - expected.float()).abs()
    print(
        f"{label:<16} max={delta.max().item():.7f} "
        f"mean={delta.mean().item():.7f}"
    )


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    tokens, hc, hidden = args.tokens, 4, 4096
    residual = torch.randn((tokens, hc, hidden), dtype=torch.bfloat16, device=device)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
    fn = torch.randn((24, hc * hidden), dtype=torch.float32, device=device) * 0.01
    scale = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32, device=device)
    base = torch.randn((24,), dtype=torch.float32, device=device) * 0.01
    rms_eps = 1e-6
    mix_eps = 1e-6
    sinkhorn_repeat = 4

    ref_post, ref_comb, ref_layer = mhc_pre_torch(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        mix_eps,
        mix_eps,
        2.0,
        sinkhorn_repeat,
    )
    native_post, native_comb, native_layer = quixicore_ops.dsv4_mhc_pre(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        mix_eps,
        mix_eps,
        2.0,
        sinkhorn_repeat,
    )
    error("pre.post", native_post, ref_post.squeeze(-1))
    error("pre.comb", native_comb, ref_comb)
    error("pre.layer", native_layer, ref_layer)

    ref_residual = mhc_post_torch(x, residual, ref_post, ref_comb)
    ref_next_post, ref_next_comb, ref_next_layer = mhc_pre_torch(
        ref_residual,
        fn,
        scale,
        base,
        rms_eps,
        mix_eps,
        mix_eps,
        2.0,
        sinkhorn_repeat,
    )
    native_residual, native_next_post, native_next_comb, native_next_layer = (
        quixicore_ops.dsv4_mhc_fused_post_pre(
            x,
            residual,
            ref_post.squeeze(-1).contiguous(),
            ref_comb.contiguous(),
            fn,
            scale,
            base,
            rms_eps,
            mix_eps,
            mix_eps,
            2.0,
            sinkhorn_repeat,
        )
    )
    error("fused.residual", native_residual, ref_residual)
    error("fused.post", native_next_post, ref_next_post.squeeze(-1))
    error("fused.comb", native_next_comb, ref_next_comb)
    error("fused.layer", native_next_layer, ref_next_layer)

    norm_weight = torch.randn((hidden,), dtype=torch.bfloat16, device=device)
    ref_normalized = (
        ref_next_layer.float()
        * torch.rsqrt(ref_next_layer.float().square().mean(dim=-1, keepdim=True) + rms_eps)
        * norm_weight.float()
    ).to(torch.bfloat16)
    native_normalized = quixicore_ops.dsv4_mhc_fused_post_pre(
        x,
        residual,
        ref_post.squeeze(-1).contiguous(),
        ref_comb.contiguous(),
        fn,
        scale,
        base,
        rms_eps,
        mix_eps,
        mix_eps,
        2.0,
        sinkhorn_repeat,
        norm_weight,
        rms_eps,
    )[3]
    error("fused.norm", native_normalized, ref_normalized)

    native_post_only = quixicore_ops.dsv4_mhc_post(
        x, residual, ref_post.squeeze(-1).contiguous(), ref_comb.contiguous()
    )
    error("post", native_post_only, ref_residual)

    head_fn = fn[:hc].contiguous()
    head_scale = scale[:1].contiguous()
    head_base = base[:hc].contiguous()
    ref_head = hc_head_fused_torch(
        residual, head_fn, head_scale, head_base, rms_eps, mix_eps
    )
    native_head = quixicore_ops.dsv4_hc_head(
        residual, head_fn, head_scale, head_base, rms_eps, mix_eps
    )
    error("head", native_head, ref_head)

    native_us = elapsed_us(
        lambda: quixicore_ops.dsv4_mhc_fused_post_pre(
            x,
            residual,
            ref_post.squeeze(-1).contiguous(),
            ref_comb.contiguous(),
            fn,
            scale,
            base,
            rms_eps,
            mix_eps,
            mix_eps,
            2.0,
            sinkhorn_repeat,
        ),
        args.warmup,
        args.repetitions,
    )
    native_norm_us = elapsed_us(
        lambda: quixicore_ops.dsv4_mhc_fused_post_pre(
            x,
            residual,
            ref_post.squeeze(-1).contiguous(),
            ref_comb.contiguous(),
            fn,
            scale,
            base,
            rms_eps,
            mix_eps,
            mix_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight,
            rms_eps,
        ),
        args.warmup,
        args.repetitions,
    )
    norm_output = torch.empty_like(native_next_layer)

    def native_then_norm() -> None:
        layer = quixicore_ops.dsv4_mhc_fused_post_pre(
            x,
            residual,
            ref_post.squeeze(-1).contiguous(),
            ref_comb.contiguous(),
            fn,
            scale,
            base,
            rms_eps,
            mix_eps,
            mix_eps,
            2.0,
            sinkhorn_repeat,
        )[3]
        custom_ops.rms_norm(norm_output, layer, norm_weight, rms_eps)

    native_then_norm_us = elapsed_us(
        native_then_norm, args.warmup, args.repetitions
    )
    torch_us = elapsed_us(
        lambda: mhc_pre_torch(
            mhc_post_torch(x, residual, ref_post, ref_comb),
            fn,
            scale,
            base,
            rms_eps,
            mix_eps,
            mix_eps,
            2.0,
            sinkhorn_repeat,
        ),
        args.warmup,
        args.repetitions,
    )
    print(
        f"tokens={tokens} native={native_us:.3f} us "
        f"native_norm={native_norm_us:.3f} us "
        f"native_then_norm={native_then_norm_us:.3f} us "
        f"torch={torch_us:.3f} us "
        f"speedup={torch_us / native_us:.2f}x"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    started = time.perf_counter()
    run(parse_args())
    print(f"wall={time.perf_counter() - started:.2f}s")
