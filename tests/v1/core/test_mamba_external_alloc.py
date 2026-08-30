# SPDX-License-Identifier: Apache-2.0
"""MambaManager align-mode external (KV-connector) allocation shape.

A tier-resumed request reads exactly one state block - position
``num_computed // block_size - 1`` (the worker seeds state_idx from it) -
so external allocation must mirror the internal-hit shape
([null] * (k - 1) + [real]) instead of densely allocating the whole span.
"""

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK = 16


def make_manager(mode="align", num_blocks=64):
    spec = MambaSpec(
        shapes=((2, 2),),
        dtypes=(torch.float32,),
        block_size=BLOCK,
        mamba_cache_mode=mode,
    )
    pool = BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BLOCK
    )
    mgr = MambaManager(
        spec,
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK,
    )
    return mgr, pool


def test_align_external_alloc_is_nulls_plus_tail():
    mgr, pool = make_manager("align")
    free_before = pool.get_num_free_blocks()
    mgr.allocate_external_computed_blocks("r1", 0, 4 * BLOCK)
    blocks = mgr.req_to_blocks["r1"]
    assert len(blocks) == 4
    assert [b.is_null for b in blocks] == [True, True, True, False]
    # Exactly one real block drawn from the pool.
    assert pool.get_num_free_blocks() == free_before - 1


def test_align_external_alloc_after_local_hit():
    mgr, pool = make_manager("align")
    # Local hit already placed [null, cached] (2 blocks).
    real = pool.get_new_blocks(1)[0]
    mgr.req_to_blocks["r1"].extend([pool.null_block, real])
    mgr.allocate_external_computed_blocks("r1", 2 * BLOCK, 3 * BLOCK)
    blocks = mgr.req_to_blocks["r1"]
    assert len(blocks) == 5
    assert [b.is_null for b in blocks[2:]] == [True, True, False]


def test_align_external_alloc_noop_when_covered():
    mgr, pool = make_manager("align")
    mgr.allocate_external_computed_blocks("r1", 0, 4 * BLOCK)
    free_after = pool.get_num_free_blocks()
    # Re-query with no additional external span: no growth.
    mgr.allocate_external_computed_blocks("r1", 4 * BLOCK, 0)
    assert len(mgr.req_to_blocks["r1"]) == 4
    assert pool.get_num_free_blocks() == free_after
