# SPDX-License-Identifier: Apache-2.0
"""Validate and time native packed-Q8 DSV4 attention output projection."""

from __future__ import annotations

import argparse

import torch

from vllm.model_executor.layers.quantization.gguf import ops
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _fused_inverse_rope_gptj,
)


def time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def make_q8_weight(
    rows: int, cols: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = cols // 32
    packed = torch.empty((rows, blocks, 34), dtype=torch.uint8, device=device)
    scales = (
        torch.rand((rows, blocks), dtype=torch.float32, device=device)
        .mul_(0.01)
        .add_(0.001)
        .to(torch.float16)
    )
    codes = torch.randint(
        -127, 128, (rows, blocks, 32), dtype=torch.int8, device=device
    )
    packed[:, :, :2].copy_(scales.view(torch.uint8).reshape(rows, blocks, 2))
    packed[:, :, 2:].copy_(codes.view(torch.uint8))
    dequant = (codes.float() * scales.float().unsqueeze(-1)).to(torch.bfloat16)
    return packed.reshape(rows, -1), dequant.reshape(rows, cols)


def run(
    groups: int,
    tokens: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> None:
    heads_per_group = 8
    head_dim = 512
    rope_dim = 64
    rows_per_group = 1024
    local_heads = groups * heads_per_group
    group_dim = heads_per_group * head_dim
    weight, dequant = make_q8_weight(
        groups * rows_per_group, group_dim, device
    )
    o = torch.randn(
        (tokens, local_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    positions = torch.arange(tokens, dtype=torch.int64, device=device) + 17
    theta = torch.randn((128, rope_dim // 2), device=device)
    cos_sin = torch.cat((theta.cos(), theta.sin()), dim=-1).contiguous()

    def quant_reference() -> tuple[torch.Tensor, torch.Tensor]:
        inverse = _fused_inverse_rope_gptj(o, positions, cos_sin, rope_dim)
        grouped = inverse.view(tokens, groups, group_dim)
        token_outputs = []
        for token in range(tokens):
            parts = [
                ops.ggml_mul_mat_vec_a8(
                    weight[
                        group * rows_per_group : (group + 1) * rows_per_group
                    ],
                    grouped[token : token + 1, group],
                    8,
                    rows_per_group,
                )
                for group in range(groups)
            ]
            token_outputs.append(
                torch.stack(parts, dim=1).reshape(1, groups * rows_per_group)
            )
        z = torch.cat(token_outputs)
        return z, ops.ggml_quantize_q8_1(z)

    def native() -> tuple[torch.Tensor, torch.Tensor]:
        return ops.ggml_dsv4_o_proj_q8_0(
            weight, o, positions, cos_sin, groups, rope_dim
        )

    def old_bf16() -> torch.Tensor:
        inverse = _fused_inverse_rope_gptj(o, positions, cos_sin, rope_dim)
        grouped = inverse.view(tokens, groups, group_dim)
        z = torch.einsum(
            "tgd,grd->tgr",
            grouped,
            dequant.view(groups, rows_per_group, group_dim),
        ).reshape(tokens, groups * rows_per_group)
        return ops.ggml_quantize_q8_1(z)

    expected_z, expected_quant = quant_reference()
    actual_z, actual_quant = native()
    torch.cuda.synchronize()
    z_exact = torch.equal(actual_z, expected_z)
    quant_exact = torch.equal(actual_quant, expected_quant)
    max_error = (actual_z.float() - expected_z.float()).abs().max().item()
    native_us = time_us(native, warmup, iterations)
    quant_reference_us = time_us(quant_reference, warmup, iterations)
    old_bf16_us = time_us(old_bf16, warmup, iterations)
    print(
        f"groups={groups} tokens={tokens} native_us={native_us:.3f} "
        f"quant_reference_us={quant_reference_us:.3f} "
        f"old_bf16_us={old_bf16_us:.3f} "
        f"vs_old={old_bf16_us / native_us:.4f} "
        f"z_exact={z_exact} quant_exact={quant_exact} "
        f"max_error={max_error:.8f}"
    )
    if not z_exact or not quant_exact:
        raise SystemExit("native DSV4 Q8 o_proj is not bitwise exact")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    for groups in args.groups:
        run(groups, args.tokens, args.warmup, args.iterations, device)


if __name__ == "__main__":
    main()
