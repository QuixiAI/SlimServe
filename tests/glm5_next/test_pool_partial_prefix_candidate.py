# SPDX-License-Identifier: Apache-2.0
"""Kernel gate for a possible finer-grained partial-page prefix cache.

This does NOT implement scheduler lookup, copy-on-write ownership or tier
metadata. It tests one prerequisite: at a four-token-aligned prefix, cloning
a page containing later tokens is safe if subsequent writes target the clone
and scoring obeys the shorter visible length. Old raw-ring contents need
not represent that prefix because the next pool starts entirely after it.
"""

import pytest
import torch

from tests.glm5_next.test_pool_cache import (
    allocate,
    compare_logits,
    slot_ids,
    update_both,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bs", [64, 4608])
@pytest.mark.parametrize("eighths", [1, 3])
@torch.no_grad()
def test_clone_partial_prefix_ignores_future_keys_and_stale_ring(bs, eighths):
    torch.manual_seed(9300 + bs + eighths)
    prefix = bs * eighths // 8
    assert prefix % 4 == 0
    source_table, source_raw, source_cache = allocate(bs, [2 * bs], padding=2)
    _, target_raw, target_cache = allocate(bs, [2 * bs], padding=2)
    target_table = source_table.flip(-1).contiguous()
    ape = torch.randn(4, 128, device="cuda")
    # The raw ring now describes the far-future suffix, not the prefix.
    old_rows = torch.randn(bs - 1, 256, device="cuda", dtype=torch.bfloat16)
    update_both(
        old_rows,
        slot_ids(source_table, 0, 0, bs - 1, bs),
        ape,
        source_raw,
        source_cache,
    )
    snapshot = source_cache._base.clone()
    source_page, target_page = source_table[0, 0], target_table[0, 0]
    target_cache[target_page].copy_(source_cache[source_page])
    target_raw[target_page].copy_(source_raw[source_page])
    compare_logits(target_table, target_raw, target_cache, ape, [0], [prefix])
    cursor = prefix
    for chunk in (1, 2, 5, 17):
        end = cursor + chunk
        replacement = torch.randn(chunk, 256, device="cuda", dtype=torch.bfloat16)
        update_both(
            replacement,
            slot_ids(target_table, 0, cursor, end, bs),
            ape,
            target_raw,
            target_cache,
        )
        compare_logits(
            target_table,
            target_raw,
            target_cache,
            ape,
            [0] * chunk,
            list(range(cursor + 1, end + 1)),
        )
        cursor = end
    assert torch.equal(source_cache._base.view(torch.int16), snapshot.view(torch.int16))
    assert torch.isnan(target_cache._base[:, 1:]).all()
    # Previously completed prefix pools remain byte-identical after append.
    assert torch.equal(
        target_cache[target_page].reshape(-1)[: prefix // 4 * 128],
        source_cache[source_page].reshape(-1)[: prefix // 4 * 128],
    )
