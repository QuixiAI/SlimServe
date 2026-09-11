# SPDX-License-Identifier: Apache-2.0
"""Quarantined full-K=256 NVFP4 down projection on existing Marlin layouts.

One CTA owns an expert-aligned eight-row/output-column tile. No stream-K
locks, global partial sums, weight repack or persistent decoded weight cache.
The physical BF16 weight scale and down epilogue match the owned Marlin
expressions; MMA accumulation order and native GPU parity are separate gates.
This module is not imported by serving.
"""

import torch

from vllm.triton_utils import tl, triton


def weight_bits_cpu(codes, scale_bytes):
    """Power-of-two rebias avoids arithmetic on FP32 subnormal FP4 values."""
    codes, scale_bytes = codes.to(torch.int32), scale_bytes.to(torch.int32)
    magnitude = codes & 7
    fp4 = torch.where(magnitude == 1, 0x3F00, (magnitude << 6) + 0x3F00)
    fp4 = torch.where(magnitude == 0, 0, fp4) | ((codes & 8) << 12)
    scale = ((scale_bytes & 128) << 7) | ((scale_bytes & 127) << 4)
    scale = torch.where(scale_bytes == 0, 0, scale - 7 * 128)
    product = (
        fp4.to(torch.int16).view(torch.bfloat16).float()
        * scale.to(torch.int16).view(torch.bfloat16).float()
    ).bfloat16()
    bits = product.view(torch.int16).to(torch.int32) & 0xFFFF
    absolute = bits & 0x7FFF
    assert ((absolute == 0) | (absolute >= 120 * 128)).all()
    return (torch.where(absolute == 0, 0, absolute - 119 * 128) | (bits & 0x8000)).to(
        torch.int16
    )


def down_epilogue_cpu(tiny_accumulator, router_weight, stored_global_scale):
    """Preserve BOTH BF16 conversions before the final BF16 multiply."""
    factor = (router_weight.float() * stored_global_scale.float()).bfloat16()
    return (tiny_accumulator.bfloat16().float() * factor.float()).bfloat16()


@triton.jit
def _direct_down(
    X,
    W,
    S,
    G,
    SORTED,
    EXPERT,
    PADDED,
    ROUTE,
    OUT,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block, column = tl.program_id(0), tl.program_id(1)
    if block * 8 >= tl.load(PADDED):
        return
    expert = tl.load(EXPERT + block)
    if expert < 0 or expert >= 288:
        return
    m, k = tl.arange(0, 16), tl.arange(0, 256)
    n = column * BLOCK_N + tl.arange(0, BLOCK_N)
    route = tl.load(SORTED + block * 8 + m, m < 8, ROWS)
    valid = (m < 8) & (route >= 0) & (route < ROWS)
    a = tl.load(X + route[:, None] * 256 + k[None, :], valid[:, None], 0)
    # Inverse of owned gptq_marlin_repack's non-act-order BF16 layout.
    kk, nn = k[:, None], n[None, :]
    word = ((kk // 16) * 64 + nn // 64) * 128
    word += (nn % 8) * 16 + ((kk % 8) // 2) * 4 + (nn % 64) // 16
    shift = 4 * (4 * (kk % 2) + 2 * ((nn % 16) // 8) + (kk % 16) // 8)
    code = (tl.load(W + expert * (256 * 4096 // 8) + word) >> shift) & 15
    # Inverse of marlin_permute_scales + S0E5M3 four-byte permutation.
    flat = (kk // 16) * 4096 + nn
    permuted = (flat // 64) * 64 + (flat % 8) * 8 + (flat % 64) // 8
    address = (permuted // 4) * 4 + (permuted % 2) * 2 + (permuted % 4) // 2
    scale_byte = tl.load(S + expert * (16 * 4096) + address).to(tl.int32)
    magnitude = code & 7
    fp4_bits = tl.where(magnitude == 1, 0x3F00, (magnitude << 6) + 0x3F00)
    fp4_bits = tl.where(magnitude == 0, 0, fp4_bits) | ((code & 8) << 12)
    scale_bits = ((scale_byte & 128) << 7) | ((scale_byte & 127) << 4)
    scale_bits = tl.where(scale_byte == 0, 0, scale_bits - 7 * 128)
    fp4 = fp4_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    scale = scale_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    product = (fp4.to(tl.float32) * scale.to(tl.float32)).to(tl.bfloat16)
    bits = product.to(tl.uint16, bitcast=True).to(tl.int32)
    absolute = bits & 0x7FFF
    tiny_bits = tl.where(absolute == 0, 0, absolute - 119 * 128) | (bits & 0x8000)
    b = tiny_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    accumulator = tl.dot(a, b)
    factor = (tl.load(ROUTE + route, valid, 0) * tl.load(G + expert)).to(tl.bfloat16)
    # Marlin down rounds its tiny accumulator BEFORE the huge combined scale.
    result = accumulator.to(tl.bfloat16).to(tl.float32) * factor[:, None].to(tl.float32)
    tl.store(OUT + route[:, None] * 4096 + n[None, :], result, valid[:, None])


def direct_down(
    x,
    packed,
    scales,
    global_scale,
    sorted_ids,
    expert_ids,
    padded_count,
    route_weights,
    *,
    out=None,
    tile=64,
):
    assert tile in (64, 128)
    rows = x.shape[0]
    assert 1 <= rows <= 256 and x.shape == (rows, 256)
    assert x.dtype == torch.bfloat16
    assert packed.shape == (288, 16, 8192) and packed.dtype == torch.int32
    assert scales.shape == (288, 16, 4096) and scales.dtype == torch.float8_e4m3fn
    assert global_scale.shape == (288,) and global_scale.dtype == torch.float32
    assert sorted_ids.ndim == expert_ids.ndim == 1
    assert sorted_ids.numel() % 8 == 0 and expert_ids.numel() >= sorted_ids.numel() // 8
    assert padded_count.numel() == 1 and route_weights.numel() == rows
    assert route_weights.dtype == torch.float32
    assert all(t.dtype == torch.int32 for t in (sorted_ids, expert_ids, padded_count))
    tensors = (
        x,
        packed,
        scales,
        global_scale,
        sorted_ids,
        expert_ids,
        padded_count,
        route_weights,
    )
    assert x.is_cuda and all(
        t.device == x.device and t.is_contiguous() for t in tensors
    )
    if out is None:
        out = torch.empty(rows, 4096, device=x.device, dtype=x.dtype)
    assert out.shape == (rows, 4096) and out.dtype == x.dtype
    assert out.device == x.device and out.is_contiguous()
    _direct_down[(sorted_ids.numel() // 8, 4096 // tile)](
        x,
        packed,
        scales.view(torch.uint8),
        global_scale,
        sorted_ids,
        expert_ids,
        padded_count,
        route_weights,
        out,
        rows,
        tile,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


def compile_only():
    import json

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = dict(
        X="*bf16",
        W="*i32",
        S="*u8",
        G="*fp32",
        SORTED="*i32",
        EXPERT="*i32",
        PADDED="*i32",
        ROUTE="*fp32",
        OUT="*bf16",
    )
    for tile in (64, 128):
        kernel = triton.compile(
            ASTSource(_direct_down, signature, constexprs=dict(ROWS=256, BLOCK_N=tile)),
            target=GPUTarget("cuda", 80, 32),
            options=dict(num_warps=4, num_stages=1, enable_fp_fusion=False),
        )
        print(
            json.dumps(
                dict(
                    tile=tile,
                    shared_bytes=kernel.metadata.shared,
                    scope="Offline compilation only; not GPU correctness/timing",
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    compile_only()
