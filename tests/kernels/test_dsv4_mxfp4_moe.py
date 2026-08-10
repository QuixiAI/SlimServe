# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the fused A100 MXFP4 MoE decode path.

Compares ggml_dsv4_moe_a8_mxfp4 (fused gate/up + SwiGLU + Q8_1 emission +
weighted MXFP4 down) against an fp32 dequantized reference over random MXFP4
expert weights. Tolerances account for the Q8_1 activation and intermediate
quantization the fused path shares with the IQ2_XXS/Q2_K pipeline.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf import ops
from vllm.platforms import current_platform

MXFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _random_mxfp4(experts: int, rows: int, cols: int, device, seed: int):
    """Random raw MXFP4 tensor [experts, rows, cols/32*17] plus its fp32
    dequantization [experts, rows, cols]."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    blocks = cols // 32
    # Scales near 1.0 (e8m0 exponents around 127) keep values in a sane range.
    e = torch.randint(121, 131, (experts, rows, blocks), generator=gen).to(torch.uint8)
    qs = torch.randint(0, 256, (experts, rows, blocks, 16), generator=gen).to(
        torch.uint8
    )
    raw = torch.cat([e.unsqueeze(-1), qs], dim=-1).reshape(experts, rows, blocks * 17)

    scales = torch.pow(2.0, e.to(torch.float32) - 127.0)
    lo = MXFP4_VALUES[(qs & 0xF).long()]
    hi = MXFP4_VALUES[(qs >> 4).long()]
    # GGUF MXFP4 block order matches the MMVQ kernel: 4-byte groups, low
    # nibbles fill values [g*4, g*4+4) and high nibbles [16+g*4, 16+g*4+4).
    lo = lo.reshape(experts, rows, blocks, 4, 4)
    hi = hi.reshape(experts, rows, blocks, 4, 4)
    values = torch.cat([lo.flatten(3), hi.flatten(3)], dim=-1)
    dequant = (values * scales.unsqueeze(-1)).reshape(experts, rows, cols)
    return raw.to(device), dequant.to(device)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 4, 8, 48])
@pytest.mark.parametrize("intermediate", [64, 512])
def test_fused_mxfp4_moe_matches_dequant_reference(tokens, intermediate):
    torch.manual_seed(0)
    device = "cuda"
    experts, top_k, hidden = 16, 6, 256

    w1_raw, w1_ref = _random_mxfp4(experts, 2 * intermediate, hidden, device, 1)
    w2_raw, w2_ref = _random_mxfp4(experts, hidden, intermediate, device, 2)
    x = (torch.randn(tokens, hidden, device=device) * 0.3).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, device=device) + 0.25

    out = ops.ggml_dsv4_moe_a8_mxfp4(
        x,
        w1_raw,
        w2_raw,
        topk_weights.float().contiguous(),
        topk_ids.contiguous(),
        intermediate,
        hidden,
        top_k,
        tokens,
        0.0,
    )

    xf = x.to(torch.float32)
    ref = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for t in range(tokens):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            gates_ups = w1_ref[e].to(torch.float32) @ xf[t]
            gate, up = gates_ups[:intermediate], gates_ups[intermediate:]
            act = (gate / (1.0 + torch.exp(-gate))) * up
            ref[t] += float(topk_weights[t, k]) * (
                w2_ref[e].to(torch.float32) @ act
            )

    out_f = out.to(torch.float32)
    scale = ref.abs().max().clamp(min=1.0)
    rel = (out_f - ref).abs().max() / scale
    assert torch.isfinite(out_f).all()
    assert rel < 0.03, f"fused MXFP4 deviates: rel={float(rel):.4f}"


def _moe_a8_vs_reference(tokens, top_k, hidden, rows, experts, seed):
    """Run ggml_moe_a8 (type 39) through real moe_align metadata and compare
    against the fp32 dequantized reference. Covers the tensor-core grouped
    tile (K % 256 == 0), the 64-wide dp4a fallback (other K), padding
    columns, and expert-row tails."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        moe_align_block_size,
    )

    device = "cuda"
    torch.manual_seed(seed)
    w_raw, w_ref = _random_mxfp4(experts, rows, hidden, device, seed)
    x = (torch.randn(tokens, hidden, device=device) * 0.3).to(torch.bfloat16)
    topk_ids = torch.stack(
        [
            torch.randperm(experts, device=device)[:top_k]
            for _ in range(tokens)
        ]
    ).to(torch.int32)

    align = ops.ggml_moe_get_block_size(39)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, align, experts
    )
    out = ops.ggml_moe_a8(
        x,
        w_raw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        39,
        rows,
        top_k,
        tokens,
    )

    xf = x.to(torch.float32)
    ref = torch.empty(tokens * top_k, rows, device=device, dtype=torch.float32)
    for t in range(tokens):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            ref[t * top_k + k] = w_ref[e].to(torch.float32) @ xf[t]

    out_f = out.to(torch.float32)
    scale = ref.abs().max().clamp(min=1.0)
    rel = (out_f - ref).abs().max() / scale
    assert torch.isfinite(out_f).all()
    assert rel < 0.02, f"moe_a8 MXFP4 deviates: rel={float(rel):.4f}"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [8, 65, 128, 300])
def test_moe_a8_mxfp4_w1_shape(tokens):
    # W1-like: K = hidden, gate|up rows, top_k routes per token.
    _moe_a8_vs_reference(tokens, 6, 256, 128, 16, 3)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [96, 390])
def test_moe_a8_mxfp4_w2_shape(tokens):
    # W2-like: per-route activations (top_k=1), K = intermediate shard,
    # wide output rows (multiple 128-row tiles).
    _moe_a8_vs_reference(tokens, 1, 512, 256, 16, 4)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_moe_a8_mxfp4_row_tail(tokens=70):
    # Output rows not a multiple of the 128-row tile: row-clamp path.
    _moe_a8_vs_reference(tokens, 6, 256, 96, 16, 5)


def _seg_vs_reference(tokens, drop_routes=0, seed=7, repacked=False):
    """Full segmented MoE (perm build + fused-SwiGLU W1 + W2 + reduce) vs the
    fp32 dequant reference. tokens*top_k < 1536 exercises the J=16 tile,
    larger the J=64 tile."""
    device = "cuda"
    torch.manual_seed(seed)
    experts, top_k, hidden, intermediate = 16, 6, 256, 256

    w1_raw, w1_ref = _random_mxfp4(experts, 2 * intermediate, hidden, device, seed)
    w2_raw, w2_ref = _random_mxfp4(experts, hidden, intermediate, device, seed + 1)
    x = (torch.randn(tokens, hidden, device=device) * 0.3).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    if drop_routes:
        flat = topk_ids.view(-1)
        drop = torch.randperm(flat.numel(), device=device)[:drop_routes]
        flat[drop] = -1
    topk_weights = torch.rand(tokens, top_k, device=device) + 0.25

    if repacked:
        w1_in = ops.ggml_dsv4_repack_mxfp4(w1_raw, hidden)
        w2_in = ops.ggml_dsv4_repack_mxfp4(w2_raw, intermediate)
    else:
        w1_in, w2_in = w1_raw, w2_raw
    out = ops.ggml_dsv4_moe_a8_mxfp4_seg(
        x,
        w1_in,
        w2_in,
        topk_weights.float().contiguous(),
        topk_ids.contiguous(),
        intermediate,
        hidden,
        top_k,
        tokens,
        0.0,
        repacked,
        repacked,
    )

    xf = x.to(torch.float32)
    ref = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for t in range(tokens):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            if e < 0:
                continue
            gates_ups = w1_ref[e].to(torch.float32) @ xf[t]
            gate, up = gates_ups[:intermediate], gates_ups[intermediate:]
            act = (gate / (1.0 + torch.exp(-gate))) * up
            ref[t] += float(topk_weights[t, k]) * (
                w2_ref[e].to(torch.float32) @ act
            )

    out_f = out.to(torch.float32)
    scale = ref.abs().max().clamp(min=1.0)
    rel = (out_f - ref).abs().max() / scale
    assert torch.isfinite(out_f).all()
    assert rel < 0.03, f"segmented MXFP4 deviates: rel={float(rel):.4f}"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [65, 128])
def test_seg_mxfp4_moe_j16(tokens):
    _seg_vs_reference(tokens)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [65, 300])
def test_seg_mxfp4_moe_repacked(tokens):
    _seg_vs_reference(tokens, repacked=True)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_seg_mxfp4_moe_j64(tokens=300):
    _seg_vs_reference(tokens)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_seg_mxfp4_moe_invalid_routes(tokens=80):
    # Dropped routes (expert-map -1) must contribute exactly nothing.
    _seg_vs_reference(tokens, drop_routes=60)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_moe_a8_mxfp4_k_fallback(tokens=90):
    # K % 256 != 0 cannot take the tensor-core tile; exercises the 64-wide
    # dp4a fallback against the same 64-wide alignment metadata.
    _moe_a8_vs_reference(tokens, 6, 288, 128, 16, 6)


def _random_gguf_bytes(experts, rows, cols, block_bytes, values_per_block,
                       device, seed, scale_byte_offsets=()):
    """Random raw GGUF blocks; fp16 d fields are patched to ~1.0 so the
    dequant reference stays in a sane range."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    blocks = cols // values_per_block
    raw = torch.randint(
        0, 256, (experts, rows, blocks, block_bytes), generator=gen
    ).to(torch.uint8)
    for off in scale_byte_offsets:
        # fp16 ~= 0.00x-0.03 range: exponent bits low, keeps sums finite.
        raw[..., off] = torch.randint(0, 256, raw.shape[:-1], generator=gen)
        raw[..., off + 1] = 0x2C  # fp16 exponent for ~0.0x magnitudes
    return raw.reshape(experts, rows, blocks * block_bytes).to(device)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [40, 300])
def test_iq2_seg_moe_matches_dequant_reference(tokens):
    """Full hybrid segmented pipeline (perm + IQ2 W1 fused SwiGLU + Q2_K W2 +
    reduce) on repacked weights vs the fp32 ggml_dequantize reference."""
    device = "cuda"
    torch.manual_seed(11)
    experts, top_k, hidden, intermediate = 16, 6, 256, 256

    # IQ2_XXS: 66-byte superblocks (fp16 d at offset 0), 256 values.
    w1_raw = _random_gguf_bytes(
        experts, 2 * intermediate, hidden, 66, 256, device, 21,
        scale_byte_offsets=(0,),
    )
    # Q2_K: 84-byte superblocks (scales[16] | qs[64] | d | dmin).
    w2_raw = _random_gguf_bytes(
        experts, hidden, intermediate, 84, 256, device, 22,
        scale_byte_offsets=(80, 82),
    )

    w1_ref = torch.stack(
        [
            ops.ggml_dequantize(w1_raw[e], 16, 2 * intermediate, hidden,
                                torch.float32)
            for e in range(experts)
        ]
    )
    w2_ref = torch.stack(
        [
            ops.ggml_dequantize(w2_raw[e], 10, hidden, intermediate,
                                torch.float32)
            for e in range(experts)
        ]
    )

    w1_rep = ops.ggml_dsv4_repack_iq2_xxs(w1_raw, hidden)
    w2_rep = ops.ggml_dsv4_repack_q2_k(w2_raw, intermediate)

    x = (torch.randn(tokens, hidden, device=device) * 0.3).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(tokens, top_k, device=device) + 0.25

    out = ops.ggml_dsv4_moe_a8_iq2_seg(
        x,
        w1_rep,
        w2_rep,
        topk_weights.float().contiguous(),
        topk_ids.contiguous(),
        intermediate,
        hidden,
        top_k,
        tokens,
        0.0,
    )

    xf = x.to(torch.float32)
    ref = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for t in range(tokens):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            gates_ups = w1_ref[e] @ xf[t]
            gate, up = gates_ups[:intermediate], gates_ups[intermediate:]
            act = (gate / (1.0 + torch.exp(-gate))) * up
            ref[t] += float(topk_weights[t, k]) * (w2_ref[e] @ act)

    out_f = out.to(torch.float32)
    scale = ref.abs().max().clamp(min=1.0)
    rel = (out_f - ref).abs().max() / scale
    assert torch.isfinite(out_f).all()
    assert rel < 0.03, f"iq2 segmented MoE deviates: rel={float(rel):.4f}"
