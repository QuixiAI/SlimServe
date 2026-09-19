"""Tensor descriptors must not let unwritten page-tail NaNs enter attention.

Run on XPU; deterministic random data models Qwen TP4's non-power-of-two
six-query-head group, interleaved FP8 KV pages, and mixed ragged lengths.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU tensor-descriptor regression"
)


@pytest.mark.parametrize("poison_padding", [False, True])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("use_td", [False, True])
def test_unwritten_fp8_page_tail_is_masked(query_len, use_td, poison_padding):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    generator = torch.Generator().manual_seed(20260919)
    lengths = [10177, 10191, 10239, 10240, 10178, 10192, 10238, 10237]
    batch, heads, dim, page, segments = len(lengths), 6, 256, 64, 16
    blocks_per_seq = (max(lengths) + page - 1) // page
    q_cpu = torch.randn(batch * query_len, heads, dim, generator=generator).bfloat16()
    kv_cpu = torch.randn(
        batch, blocks_per_seq * page, 1, 2 * dim, generator=generator
    ).to(torch.float8_e4m3fn)
    # E4M3FN has NaN bit pattern 0x7f. These positions are allocated but lie
    # outside each sequence's logical length and must have no numerical effect.
    if poison_padding:
        for seq, length in enumerate(lengths):
            kv_cpu.view(torch.uint8)[seq, length:] = 0x7F

    reference = []
    for seq, length in enumerate(lengths):
        k_cpu = kv_cpu[seq, :length, 0, :dim].float()
        v_cpu = kv_cpu[seq, :length, 0, dim:].float()
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
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
        use_td=use_td,
    )
    actual = output.cpu().float()
    assert torch.isfinite(actual).all(), "Unwritten KV tail contaminated attention"
    torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.02)
