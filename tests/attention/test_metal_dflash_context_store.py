# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash context writes must preserve packed neighbors and padding slots."""

import pytest
import torch

from vllm.model_executor.models.muse_glimmer_dflash import _store_kv_at_slots

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padded", [False, True])
def test_context_store_matches_cpu_and_preserves_packed_neighbors(dtype, padded):
    blocks, block_size, heads, dim = 5, 16, 2, 8
    page = block_size * heads * dim
    stride, offset = 4 * page, 37
    raw = torch.full((blocks * stride,), -17, dtype=dtype, device="mps")
    cache = raw.as_strided(
        (2, blocks, block_size, heads, dim),
        (page, stride, heads * dim, dim, 1),
        offset,
    )
    slots = torch.tensor([3, 65, -1 if padded else 21], device="mps", dtype=torch.int32)
    # Precompute supplies sliced K/V projections, not necessarily dense inputs.
    projections = (
        torch.arange(3 * heads * dim * 2, dtype=torch.float32)
        .view(3, heads, 2 * dim)
        .to("mps")
    )
    key, value = projections[..., :dim], projections[..., dim:]
    expected = raw.cpu().clone()
    expected_cache = expected.as_strided(cache.shape, cache.stride(), offset)
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            expected_cache[0, slot // block_size, slot % block_size] = key[row].cpu()
            expected_cache[1, slot // block_size, slot % block_size] = value[row].cpu()
    _store_kv_at_slots(cache, None, slots, key, value)
    assert torch.equal(raw.cpu(), expected)


def test_empty_context_store_is_a_noop():
    cache = torch.full((2, 2, 16, 2, 8), -17, device="mps")
    _store_kv_at_slots(
        cache,
        None,
        torch.empty(0, device="mps", dtype=torch.long),
        torch.empty((0, 2, 8), device="mps"),
        torch.empty((0, 2, 8), device="mps"),
    )
    assert (cache == -17).all()
