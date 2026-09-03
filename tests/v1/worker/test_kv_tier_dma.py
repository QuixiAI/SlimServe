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

    dma.issue(TierOpBatch(seq=1, offload=[(3, 7, 0)], restore=[]))
    assert dma.flush() == [1]
    # Clobber the GPU copy, then restore from the host slot into a new block.
    blocks[3].zero_()
    dma.issue(TierOpBatch(seq=2, offload=[], restore=[(7, 5, 0)]))
    dma.fence_restores()
    assert dma.flush() == [2]
    assert torch.equal(blocks[5], src)


def test_batches_complete_in_order():
    backing, dma = _dma()
    blocks = backing.view(-1, STRIDE)
    for b in range(4):
        blocks[b].fill_(b + 1)
    dma.issue(TierOpBatch(seq=10, offload=[(0, 0, 0), (1, 1, 0)], restore=[]))
    dma.issue(TierOpBatch(seq=11, offload=[(2, 2, 0), (3, 3, 0)], restore=[]))
    done = dma.flush()
    assert done == [10, 11]
    for b in range(4):
        assert int(dma.arena[b][0]) == b + 1


def test_empty_batch_is_a_noop():
    _, dma = _dma()
    dma.issue(TierOpBatch(seq=1, offload=[], restore=[]))
    assert dma.flush() == []
def _mem_available_mib() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("no MemAvailable")


def test_arena_costs_exactly_its_size_not_the_next_power_of_two():
    """A 1.5 GiB arena must consume ~1.5 GiB of RAM, not 2 GiB.

    torch's caching pinned allocator rounds pin_memory=True allocations up
    to the next power of two; an 88 GiB/rank arena became 128 GiB/rank and
    OOM-killed the 8-rank boot on 2026-09-02. The arena is cudaHostRegister'd
    instead, which pins exactly the pages it owns.
    """
    device = torch.device("cuda")
    stride = 1 << 20
    slots = 1536  # 1.5 GiB: not a power of two
    backing = torch.zeros(2 * stride, dtype=torch.int8, device=device)
    torch.cuda.synchronize()
    before = _mem_available_mib()
    dma = KVTierDMA(backing, stride, slots, device)
    used = before - _mem_available_mib()
    assert dma.arena.is_pinned()
    assert dma.arena.numel() == slots * stride
    # Power-of-two rounding would show ~2048 MiB here.
    assert 1536 * 0.9 <= used <= 1536 * 1.15, used
    dma.release()
    assert not dma.arena.is_pinned()
