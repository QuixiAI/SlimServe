# SPDX-License-Identifier: Apache-2.0
"""KVTierDMA: offload/restore roundtrip through the pinned arena."""

import pytest
import torch

from vllm.v1.worker.gpu.kv_tier_dma import KVTierDMA, TierOpBatch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

STRIDE = 4096
BLOCKS = 8
SLOTS = 16


def _dma():
    device = torch.device("cuda")
    backing = torch.zeros(BLOCKS * STRIDE, dtype=torch.int8, device=device)
    return backing, KVTierDMA(backing, STRIDE, SLOTS, device)


def test_offload_then_restore_roundtrip():
    backing, dma = _dma()
    blocks = backing.view(-1, STRIDE)
    src = torch.randint(-128, 127, (STRIDE,), dtype=torch.int8, device="cuda")
    blocks[3].copy_(src)

    dma.issue(TierOpBatch(seq=1, offload=[(3, 7)], restore=[]))
    assert dma.flush() == [1]
    # Clobber the GPU copy, then restore from the host slot into a new block.
    blocks[3].zero_()
    dma.issue(TierOpBatch(seq=2, offload=[], restore=[(7, 5)]))
    dma.fence_restores()
    assert dma.flush() == [2]
    assert torch.equal(blocks[5], src)


def test_batches_complete_in_order():
    backing, dma = _dma()
    blocks = backing.view(-1, STRIDE)
    for b in range(4):
        blocks[b].fill_(b + 1)
    dma.issue(TierOpBatch(seq=10, offload=[(0, 0), (1, 1)], restore=[]))
    dma.issue(TierOpBatch(seq=11, offload=[(2, 2), (3, 3)], restore=[]))
    done = dma.flush()
    assert done == [10, 11]
    for b in range(4):
        assert int(dma.arena[b][0]) == b + 1


def test_empty_batch_is_a_noop():
    _, dma = _dma()
    dma.issue(TierOpBatch(seq=1, offload=[], restore=[]))
    assert dma.flush() == []
