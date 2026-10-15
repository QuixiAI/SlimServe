# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor descriptors must not let unwritten page-tail NaNs enter attention.

Run on XPU; deterministic random data models Qwen TP4's non-power-of-two
six-query-head group, interleaved KV pages, and mixed ragged lengths. Every
KV quant mode the TD path dequantizes inside the kernel is covered: none
(bf16 cache), FP8 per-tensor, and INT8 / FP8 per-token-head (whose scales
come from separate per-slot caches).
"""

import pytest
import torch

from vllm.v1.kv_cache_interface import KVQuantMode

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU tensor-descriptor regression"
)

_CACHE_DTYPE = {
    KVQuantMode.NONE: torch.bfloat16,
    KVQuantMode.FP8_PER_TENSOR: torch.float8_e4m3fn,
    KVQuantMode.INT8_PER_TOKEN_HEAD: torch.int8,
    KVQuantMode.FP8_PER_TOKEN_HEAD: torch.float8_e4m3fn,
}


def _random_cache(shape, mode, generator):
    dtype = _CACHE_DTYPE[mode]
    if dtype == torch.int8:
        return torch.randint(-127, 128, shape, generator=generator, dtype=torch.int8)
    return torch.randn(shape, generator=generator).to(dtype)


def _poison(kv_slots, mode):
    """Fill unwritten slots with the cache dtype's NaN (or extreme) pattern."""
    dtype = _CACHE_DTYPE[mode]
    if dtype == torch.bfloat16:
        kv_slots.view(torch.int16).fill_(0x7FC0)  # bf16 quiet NaN
    elif dtype == torch.float8_e4m3fn:
        kv_slots.view(torch.uint8).fill_(0x7F)  # E4M3FN NaN
    else:
        # int8 has no NaN; garbage must still have zero weight.
        kv_slots.fill_(127)


@pytest.mark.parametrize(
    "kv_quant_mode",
    [
        KVQuantMode.NONE,
        KVQuantMode.FP8_PER_TENSOR,
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    ],
    ids=lambda mode: mode.name,
)
@pytest.mark.parametrize("poison_padding", [False, True])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("use_td", [False, True])
def test_unwritten_page_tail_is_masked(
    query_len, use_td, poison_padding, kv_quant_mode
):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    generator = torch.Generator().manual_seed(20260919)
    lengths = [10177, 10191, 10239, 10240, 10178, 10192, 10238, 10237]
    batch, heads, dim, page, segments = len(lengths), 6, 256, 64, 16
    blocks_per_seq = (max(lengths) + page - 1) // page
    slots = blocks_per_seq * page
    per_token_head = kv_quant_mode.is_per_token_head

    q_cpu = torch.randn(batch * query_len, heads, dim, generator=generator).bfloat16()
    kv_cpu = _random_cache((batch, slots, 1, 2 * dim), kv_quant_mode, generator)
    if per_token_head:
        # [seq, slot, kv_head]; kept away from zero so dequantized values
        # stay O(1) for both int8 codes and fp8 values.
        int8 = kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD
        base = 1.0 / 127 if int8 else 1.0
        k_scale_cpu = base * (0.5 + torch.rand(batch, slots, 1, generator=generator))
        v_scale_cpu = base * (0.5 + torch.rand(batch, slots, 1, generator=generator))
    else:
        k_scale_cpu = torch.ones(batch, slots, 1)
        v_scale_cpu = torch.ones(batch, slots, 1)
    # These positions are allocated but lie outside each sequence's logical
    # length and must have no numerical effect, including their scales.
    if poison_padding:
        for seq, length in enumerate(lengths):
            _poison(kv_cpu[seq, length:], kv_quant_mode)
            if per_token_head:
                k_scale_cpu[seq, length:] = float("nan")
                v_scale_cpu[seq, length:] = float("nan")

    reference = []
    for seq, length in enumerate(lengths):
        k_cpu = kv_cpu[seq, :length, 0, :dim].float() * k_scale_cpu[seq, :length]
        v_cpu = kv_cpu[seq, :length, 0, dim:].float() * v_scale_cpu[seq, :length]
        q_seq = q_cpu[seq * query_len : (seq + 1) * query_len].float()
        scores = (q_seq.transpose(0, 1) @ k_cpu.T) * dim**-0.5
        causal = (
            torch.arange(length)[None, :]
            <= torch.arange(length - query_len, length)[:, None]
        )
        scores.masked_fill_(~causal[None, :, :], float("-inf"))
        reference.append((scores.softmax(-1) @ v_cpu).transpose(0, 1))
    reference = torch.cat(reference)

    device = "xpu:0"
    kv = kv_cpu.reshape(-1, page, 1, 2 * dim).to(device)
    key, value = kv.split(dim, dim=-1)
    query = q_cpu.to(device)
    output = torch.empty_like(query)
    scale = torch.ones(1, device=device)
    scale_caches = {}
    if per_token_head:
        scale_caches = {
            "k_scale_cache": k_scale_cpu.reshape(-1, page, 1).to(device),
            "v_scale_cache": v_scale_cpu.reshape(-1, page, 1).to(device),
        }
    scratch = torch.empty(
        batch, heads, segments, dim, dtype=torch.float32, device=device
    )
    maxima = torch.empty(batch, heads, segments, dtype=torch.float32, device=device)
    sums = torch.empty_like(maxima)
    unified_attention(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=(torch.arange(batch + 1, dtype=torch.int32) * query_len).to(
            device
        ),
        max_seqlen_q=query_len,
        seqused_k=torch.tensor(lengths, dtype=torch.int32, device=device),
        max_seqlen_k=max(lengths),
        softmax_scale=dim**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=torch.arange(batch * blocks_per_seq, dtype=torch.int32)
        .reshape(batch, blocks_per_seq)
        .to(device),
        softcap=0.0,
        q_descale=scale,
        k_descale=scale,
        v_descale=scale,
        seq_threshold_3D=128,
        num_par_softmax_segments=segments,
        softmax_segm_output=scratch,
        softmax_segm_max=maxima,
        softmax_segm_expsum=sums,
        kv_quant_mode=kv_quant_mode,
        use_td=use_td,
        **scale_caches,
    )
    actual = output.cpu().float()
    assert torch.isfinite(actual).all(), "Unwritten KV tail contaminated attention"
    # Per-token-head modes round P * v_scale to the cache's compute dtype.
    atol = 0.01 if per_token_head else 0.003
    torch.testing.assert_close(actual, reference, atol=atol, rtol=0.02)
