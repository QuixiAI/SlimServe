# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A negative allocation must fail loudly: popleft_n(-k) used to add k to
the free-list count without linking anything."""

import pytest

from vllm.v1.core.block_pool import BlockPool


def test_negative_allocation_is_rejected():
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    free_before = pool.get_num_free_blocks()
    with pytest.raises(ValueError, match="negative"):
        pool.get_new_blocks(-3)
    assert pool.get_num_free_blocks() == free_before
    with pytest.raises(ValueError, match="negative"):
        pool.free_block_queue.popleft_n(-1)
    assert pool.get_new_blocks(0) == []
    got = pool.get_new_blocks(2)
    assert len(got) == 2 and pool.get_num_free_blocks() == free_before - 2
