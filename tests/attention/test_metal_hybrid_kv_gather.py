# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for Metal's strided hybrid KV-cache gather."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.quixicore.ops import quixicore_ops

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal"
)


def _interleaved_cache(
    num_blocks: int, block_size: int, heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    page_elems = block_size * heads * head_size
    raw = torch.empty(2 * num_blocks * page_elems, dtype=torch.bfloat16, device="mps")
    cache = raw.view(2, num_blocks, block_size, heads, head_size)
    cache.as_strided_(
        size=cache.shape,
        stride=(page_elems, 2 * page_elems, heads * head_size, head_size, 1),
    )
    return cache[0], cache[1]


@pytest.mark.parametrize("packed", [False, True])
def test_kv_scatter_honors_physical_stride_and_preserves_neighbor_layers(packed):
    blocks, block_size, heads, dim = 5, 16, 2, 8
    page = block_size * heads * dim
    stride = 2 * page + (192 if packed else 0)
    raw = torch.full((blocks * stride,), -17, dtype=torch.bfloat16, device="mps")
    key_cache = raw.as_strided(
        (blocks, block_size, heads, dim), (stride, heads * dim, dim, 1)
    )
    value_cache = raw.as_strided(key_cache.shape, key_cache.stride(), page)
    slots = torch.tensor([0, 31, -1, 64, 17], dtype=torch.long, device="mps")
    key = torch.arange(5 * heads * dim, dtype=torch.bfloat16, device="mps").view(
        5, heads, dim
    )
    value = key + 101
    expected = raw.cpu().clone()
    expected_key = expected.as_strided(key_cache.shape, key_cache.stride())
    expected_value = expected.as_strided(value_cache.shape, value_cache.stride(), page)
    for i, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            expected_key[slot // block_size, slot % block_size] = key[i].cpu()
            expected_value[slot // block_size, slot % block_size] = value[i].cpu()
    quixicore_ops.qc_kv_cache_scatter(
        key, value, slots, key_cache, value_cache, heads, dim, block_size, 1
    )
    assert torch.equal(raw.cpu(), expected)


def test_hybrid_kv_gather_honors_block_stride_and_token_range() -> None:
    num_blocks, block_size, heads, head_size = 6, 16, 2, 8
    key_cache, value_cache = _interleaved_cache(
        num_blocks, block_size, heads, head_size
    )

    for block in range(num_blocks):
        rows = torch.arange(block_size, dtype=torch.bfloat16, device="mps").view(
            -1, 1, 1
        )
        key_cache[block] = rows + 100 * block
        value_cache[block] = rows + 100 * block + 1

    block_table = torch.tensor([4, 1, 5], dtype=torch.int32, device="mps")
    token_start, num_tokens = 13, 24
    keys, values = quixicore_ops.kv_cache_gather_range(
        key_cache, value_cache, block_table, token_start, num_tokens
    )

    dense_keys = torch.cat([key_cache[int(block)] for block in block_table.cpu()])
    dense_values = torch.cat([value_cache[int(block)] for block in block_table.cpu()])
    assert torch.equal(keys, dense_keys[token_start : token_start + num_tokens])
    assert torch.equal(values, dense_values[token_start : token_start + num_tokens])


@pytest.mark.parametrize("window", [None, 12])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bound_draft_attention_uses_native_gather_and_exact_gpu_mask(
    monkeypatch, window, dtype
):
    from vllm.v1.attention.backends.metal_attn import MetalAttentionImpl

    blocks, block_size, heads, dim = 10, 8, 2, 8
    page = block_size * heads * dim
    stride = 6 * page
    torch.manual_seed(12)
    raw_cpu = torch.randn(blocks * stride).to(dtype)
    shape = (2, blocks, block_size, heads, dim)
    strides = (page, stride, heads * dim, dim, 1)
    cache_cpu = raw_cpu.as_strided(shape, strides)
    cache = raw_cpu.to("mps").as_strided(shape, strides)
    key_cache, value_cache = cache[0], cache[1]
    table = torch.tensor([[7, 3, 1, 6, 8, 2], [2, 8, 6, 1, 3, 7]])
    actual, bounds = [29, 41], [33, 45]
    query = torch.randn(6, 4, dim).to(dtype)
    output = torch.empty_like(query, device="mps")
    impl = SimpleNamespace(
        sliding_window=window,
        num_kv_heads=heads,
        head_size=dim,
        num_queries_per_kv=2,
        use_native_range_gather=True,
        scale=dim**-0.5,
    )
    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 3, 6]),
        seq_lens_cpu_bound=torch.tensor(bounds),
        seq_lens_gpu=torch.tensor(actual, device="mps", dtype=torch.int32),
        block_table=table.to(device="mps", dtype=torch.int32),
        causal=False,
        num_reqs=2,
    )
    index_select = torch.Tensor.index_select

    def reject_strided_cache_gather(tensor, *args, **kwargs):
        if tensor is key_cache or tensor is value_cache:
            raise AssertionError("strided cache index_select is unsafe on MPS")
        return index_select(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_select", reject_strided_cache_gather)
    MetalAttentionImpl._sdpa_forward(
        impl,
        query.to("mps"),
        output,
        metadata,
        key_cache,
        value_cache,
        blocks,
        block_size,
    )
    for req, length in enumerate(actual):
        begin = max(0, length - 3 - window + 1) if window else 0
        k = cache_cpu[0, table[req]].reshape(-1, heads, dim)[begin:length]
        v = cache_cpu[1, table[req]].reshape(-1, heads, dim)[begin:length]
        expected = torch.nn.functional.scaled_dot_product_attention(
            query[req * 3 : req * 3 + 3].float().transpose(0, 1),
            k.float().repeat_interleave(2, dim=1).transpose(0, 1),
            v.float().repeat_interleave(2, dim=1).transpose(0, 1),
            scale=impl.scale,
        ).transpose(0, 1)
        torch.testing.assert_close(
            output[req * 3 : req * 3 + 3].float().cpu(), expected, atol=0.008, rtol=0.02
        )


@pytest.mark.parametrize("window", [None, 12])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bound_draft_attention_ignores_nonfinite_masked_cache_rows(window, dtype):
    from vllm.v1.attention.backends.metal_attn import MetalAttentionImpl

    blocks, block_size, heads, dim = 6, 8, 2, 8
    torch.manual_seed(31)
    keys = torch.randn(blocks, block_size, heads, dim).to(dtype)
    values = torch.randn_like(keys)
    actual, bound = 29, 33
    begin = max(0, actual - 3 - window + 1) if window else 0
    query = torch.randn(3, heads, dim).to(dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.float().transpose(0, 1),
        keys.view(-1, heads, dim)[begin:actual].float().transpose(0, 1),
        values.view(-1, heads, dim)[begin:actual].float().transpose(0, 1),
        scale=dim**-0.5,
    ).transpose(0, 1)
    # Recycled sliding-window pages need not be zeroed. Masked positions
    # can contain old state bytes, including NaNs and infinities.
    for tensor, poison in ((keys, float("nan")), (values, float("inf"))):
        rows = tensor.view(-1, heads, dim)
        rows[:begin] = poison
        rows[actual:] = poison
    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 3]),
        seq_lens_cpu_bound=torch.tensor([bound]),
        seq_lens_gpu=torch.tensor([actual], device="mps", dtype=torch.int32),
        block_table=torch.arange(blocks, device="mps", dtype=torch.int32)[None],
        causal=False,
        num_reqs=1,
    )
    impl = SimpleNamespace(
        sliding_window=window,
        num_kv_heads=heads,
        head_size=dim,
        num_queries_per_kv=1,
        use_native_range_gather=True,
        scale=dim**-0.5,
    )
    output = torch.empty_like(query, device="mps")
    MetalAttentionImpl._sdpa_forward(
        impl,
        query.to("mps"),
        output,
        metadata,
        keys.to("mps"),
        values.to("mps"),
        blocks,
        block_size,
    )
    torch.testing.assert_close(output.float().cpu(), expected, atol=0.008, rtol=0.02)


@pytest.mark.skipif(
    os.environ.get("VLLM_RUN_LARGE_MPS_TESTS") != "1",
    reason="allocates a 5 GiB cache to cross the signed-32-bit element boundary",
)
def test_hybrid_kv_gather_above_signed_32bit_element_offset() -> None:
    # Qwen3.8's aligned attention page is 832 * 4 * 256 bf16 elements.
    # In the hybrid K/V-interleaved layout, block 1271 starts beyond 2^31
    # elements. torch.index_select silently wraps there on MPS.
    num_blocks, block_size, heads, head_size = 1590, 832, 4, 256
    key_cache, value_cache = _interleaved_cache(
        num_blocks, block_size, heads, head_size
    )
    block, num_tokens = 1271, 29
    block_ids = torch.full((num_tokens,), block, dtype=torch.long, device="mps")
    offsets = torch.arange(num_tokens, dtype=torch.long, device="mps")
    expected_key = (
        (
            torch.arange(num_tokens, dtype=torch.bfloat16, device="mps").view(-1, 1, 1)
            + 71
        )
        .expand(-1, heads, head_size)
        .contiguous()
    )
    expected_value = expected_key + 1
    key_cache[block_ids, offsets] = expected_key
    value_cache[block_ids, offsets] = expected_value

    block_table = torch.tensor([block], dtype=torch.int32, device="mps")
    keys, values = quixicore_ops.kv_cache_gather_range(
        key_cache, value_cache, block_table, 0, num_tokens
    )
    assert torch.equal(keys, expected_key)
    assert torch.equal(values, expected_value)

    # The write path must use the same 64-bit physical addressing as gather.
    slots = block * block_size + offsets
    updated_key = (expected_key + 11).contiguous()
    updated_value = (expected_value + 13).contiguous()
    quixicore_ops.qc_kv_cache_scatter(
        updated_key,
        updated_value,
        slots,
        key_cache,
        value_cache,
        heads,
        head_size,
        block_size,
        1,
    )
    keys, values = quixicore_ops.kv_cache_gather_range(
        key_cache, value_cache, block_table, 0, num_tokens
    )
    assert torch.equal(keys, updated_key)
    assert torch.equal(values, updated_value)

    # Bound-mode draft attention must also stay on the 64-bit gather path.
    from vllm.v1.attention.backends.metal_attn import MetalAttentionImpl

    query = torch.zeros((1, heads, head_size), dtype=torch.bfloat16, device="mps")
    output = torch.empty_like(query)
    impl = SimpleNamespace(
        sliding_window=None,
        num_kv_heads=heads,
        head_size=head_size,
        num_queries_per_kv=1,
        use_native_range_gather=True,
        scale=head_size**-0.5,
    )
    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1]),
        seq_lens_cpu_bound=torch.tensor([num_tokens]),
        seq_lens_gpu=torch.tensor([num_tokens], device="mps", dtype=torch.int32),
        block_table=block_table.view(1, 1),
        causal=False,
        num_reqs=1,
    )
    MetalAttentionImpl._sdpa_forward(
        impl, query, output, metadata, key_cache, value_cache, num_blocks, block_size
    )
    torch.testing.assert_close(
        output.float().cpu(),
        updated_value.float().cpu().mean(dim=0, keepdim=True),
        atol=0.5,
        rtol=0,
    )
