# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for Metal's strided hybrid KV-cache gather."""

import os

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
