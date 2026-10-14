# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.quixicore import quixicore_ops
from vllm.v1.attention.backends.metal_attn import MetalAttentionImpl

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal required"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padding", [0, 512])
def test_fp8_cache_roundtrip_with_physical_stride_and_padding(dtype, padding):
    blocks, block_size, heads, dim = 4, 16, 2, 64
    page = block_size * heads * dim
    raw = torch.full(
        (blocks * (2 * page + padding),), 211, dtype=torch.uint8, device="mps"
    )
    cache = raw.as_strided(
        (2, blocks, block_size, heads, dim),
        (page, 2 * page + padding, heads * dim, dim, 1),
    )
    impl = MetalAttentionImpl(4, dim, dim**-0.5, heads, kv_cache_dtype="fp8")
    scales = torch.tensor([0.25, 2.0], device="mps")
    layer = SimpleNamespace(_k_scale=scales, _v_scale=scales * 2)
    # Exactly representable FP8 inputs independently generated on CPU.
    torch.manual_seed(42)
    k = torch.randn(17, heads, dim).to(torch.float8_e4m3fn).float()
    v = torch.randn_like(k).to(torch.float8_e4m3fn).float()
    slots = torch.tensor([32 + i for i in range(16)] + [-1], device="mps")
    key = (k * scales.cpu()[None, :, None]).to(dtype).to("mps")
    value = (v * (scales.cpu() * 2)[None, :, None]).to(dtype).to("mps")
    impl.do_kv_cache_update(layer, key, value, cache, slots)
    table = torch.tensor([2, -1], dtype=torch.int32, device="mps")
    got_k, got_v = quixicore_ops.kv_cache_gather_range_fp8(
        cache[0],
        cache[1],
        table,
        3,
        16,
        scales,
        scales * 2,
        key,
    )
    torch.testing.assert_close(got_k[:13].cpu(), key[3:16].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(got_v[:13].cpu(), value[3:16].cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(got_k[13:]).item() == 0
    assert torch.count_nonzero(got_v[13:]).item() == 0
    physical = raw.cpu().view(blocks, 2 * page + padding)
    assert torch.all(physical[[0, 1, 3]] == 211)
    assert torch.all(physical[:, 2 * page :] == 211)


@pytest.mark.parametrize("causal", [False, True])
def test_fp8_draft_attention_preserves_window_and_sink(causal):
    from vllm.v1.attention.backends.metal_attn import MetalAttentionMetadata

    torch.manual_seed(13)
    heads, dim, tokens = 4, 512, 20
    cache = torch.zeros((2, 2, 16, 1, dim), dtype=torch.uint8, device="mps")
    sink = torch.tensor([-1.0, 0.0, 1.0, 3.0], device="mps")
    impl = MetalAttentionImpl(
        heads, dim, dim**-0.5, 1, sliding_window=8, kv_cache_dtype="fp8", sinks=sink
    )
    layer = SimpleNamespace(
        _k_scale=torch.ones(1, device="mps"), _v_scale=torch.ones(1, device="mps")
    )
    keys = torch.randn(tokens, 1, dim).to(torch.float8_e4m3fn).float()
    values = torch.randn_like(keys).to(torch.float8_e4m3fn).float()
    impl.do_kv_cache_update(
        layer,
        keys.to("mps"),
        values.to("mps"),
        cache,
        torch.arange(tokens, device="mps"),
    )
    q = torch.randn(3, heads, dim)
    metadata = MetalAttentionMetadata(
        num_actual_tokens=3,
        num_reqs=1,
        max_query_len=3,
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="mps"),
        seq_lens_gpu=torch.tensor([tokens], dtype=torch.int32, device="mps"),
        slot_mapping=torch.tensor([17, 18, 19], device="mps"),
        seq_lens_cpu_max=tokens,
        causal=causal,
        _seq_lens_cpu=torch.tensor([tokens]),
    )
    out = torch.empty_like(q, device="mps")
    impl.forward(
        layer,
        q.to("mps"),
        keys[17:].to("mps"),
        values[17:].to("mps"),
        cache,
        metadata,
        out,
    )
    logits = torch.einsum("thd,kd->htk", q, keys[:, 0]) * dim**-0.5
    positions = torch.arange(tokens)
    if causal:
        query_pos = torch.arange(17, 20)
        visible = (positions[None] <= query_pos[:, None]) & (
            positions[None] > query_pos[:, None] - 8
        )
    else:
        visible = positions[None] >= tokens - 3 - 8 + 1
    logits.masked_fill_(~visible, float("-inf"))
    sink_logits = sink.cpu()[:, None, None].expand(heads, 3, 1)
    probabilities = torch.cat((logits, sink_logits), -1).softmax(-1)[..., :-1]
    expected = torch.einsum("htk,kd->thd", probabilities, values[:, 0])
    torch.testing.assert_close(out.cpu(), expected, atol=3e-5, rtol=3e-5)
