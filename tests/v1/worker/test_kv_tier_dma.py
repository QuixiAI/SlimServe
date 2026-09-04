# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVTierDMA: offload/restore roundtrip through the pinned arena."""

import pytest
import torch

from vllm.v1.worker.gpu.kv_tier_dma import KVTierDMA, TierOpBatch, TierSegment

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

STRIDE = 4096
BLOCKS = 8
SLOTS = 16


def _flat(width, device="cuda"):
    return torch.zeros(BLOCKS * width, dtype=torch.int8, device=device)


def _dma():
    device = torch.device("cuda")
    backing = _flat(STRIDE)
    dma = KVTierDMA(
        {0: [TierSegment.from_flat(backing, STRIDE)]}, STRIDE, SLOTS, device
    )
    return backing.view(-1, STRIDE), dma


def test_offload_then_restore_roundtrip():
    blocks, dma = _dma()
    src = torch.randint(-128, 127, (STRIDE,), dtype=torch.int8, device="cuda")
    blocks[3].copy_(src)
    dma.issue(TierOpBatch(seq=1, offload=[(7, 0, 0, 3)], restore=[]))
    assert dma.flush() == [1]
    blocks[3].zero_()
    dma.issue(TierOpBatch(seq=2, offload=[], restore=[(7, 0, 0, 5)]))
    dma.fence_restores()
    assert dma.flush() == [2]
    assert torch.equal(blocks[5], src)


def test_batches_complete_in_order():
    blocks, dma = _dma()
    for b in range(4):
        blocks[b].fill_(b + 1)
    dma.issue(TierOpBatch(seq=10, offload=[(0, 0, 0, 0), (1, 0, 0, 1)], restore=[]))
    dma.issue(TierOpBatch(seq=11, offload=[(2, 0, 0, 2), (3, 0, 0, 3)], restore=[]))
    assert dma.flush() == [10, 11]
    for b in range(4):
        assert int(dma.row(b)[0]) == b + 1


def test_empty_batch_is_a_noop():
    _, dma = _dma()
    dma.issue(TierOpBatch(seq=1, offload=[], restore=[]))
    assert dma.flush() == []


def test_column_addressed_subblocks_share_a_slot_row():
    """A position's sub-blocks from several groups land in one row."""
    device = torch.device("cuda")
    mla, idx = _flat(1024), _flat(256)
    segs = {0: [TierSegment.from_flat(mla, 1024)], 1: [TierSegment.from_flat(idx, 256)]}
    dma = KVTierDMA(segs, 1024 + 4 * 256, SLOTS, device)
    a = torch.randint(-128, 127, (1024,), dtype=torch.int8, device=device)
    bs = [
        torch.randint(-128, 127, (256,), dtype=torch.int8, device=device)
        for _ in range(4)
    ]
    segs[0][0].blocks[2].copy_(a)
    for k in range(4):
        segs[1][0].blocks[10 + k if False else 4 + k].copy_(bs[k])
    ops = [(3, 0, 0, 2)] + [(3, 1024 + k * 256, 1, 4 + k) for k in range(4)]
    dma.issue(TierOpBatch(seq=1, offload=ops, restore=[]))
    assert dma.flush() == [1]
    assert torch.equal(dma.row(3).cuda(), torch.cat([a, *bs]))
    # Restore onto different blocks.
    rops = [(3, 0, 0, 6)] + [(3, 1024 + k * 256, 1, k) for k in range(4)]
    dma.issue(TierOpBatch(seq=2, offload=[], restore=rops, zero=[(0, 2)]))
    dma.fence_restores()
    assert dma.flush() == [2]
    assert torch.equal(segs[0][0].blocks[6], a)
    for k in range(4):
        assert torch.equal(segs[1][0].blocks[k], bs[k])
    assert int(segs[0][0].blocks[2].abs().sum()) == 0


def test_per_layer_segments_restore_every_layer():
    device = torch.device("cuda")
    widths = [1024, 512, 2560]
    layers = [_flat(w) for w in widths]
    segs = {0: [TierSegment.from_flat(t, w) for t, w in zip(layers, widths)]}
    dma = KVTierDMA(segs, STRIDE, SLOTS, device)
    srcs = [
        torch.randint(-128, 127, (w,), dtype=torch.int8, device=device) for w in widths
    ]
    for seg, src in zip(segs[0], srcs):
        seg.blocks[3].copy_(src)
    dma.issue(TierOpBatch(seq=1, offload=[(7, 0, 0, 3)], restore=[]))
    assert dma.flush() == [1]
    assert torch.equal(dma.row(7).cuda(), torch.cat(srcs))
    for seg in segs[0]:
        seg.blocks[3].zero_()
    dma.issue(TierOpBatch(seq=2, offload=[], restore=[(7, 0, 0, 5)], zero=[(0, 3)]))
    dma.fence_restores()
    assert dma.flush() == [2]
    for seg, src in zip(segs[0], srcs):
        assert torch.equal(seg.blocks[5], src)
        assert int(seg.blocks[3].abs().sum()) == 0


def test_strided_slab_view_copies_only_its_own_bytes():
    """A packed-slab group view is block-strided: rows are contiguous but
    separated by other groups' bytes, which must be left untouched."""
    device = torch.device("cuda")
    rows = _flat(STRIDE).view(BLOCKS, STRIDE)
    width = 1024
    seg = TierSegment(rows[:, :width])
    assert not seg.blocks.is_contiguous()
    dma = KVTierDMA({0: [seg]}, width, SLOTS, device)
    rows.fill_(7)
    src = torch.randint(-128, 127, (width,), dtype=torch.int8, device=device)
    seg.blocks[2].copy_(src)
    dma.issue(TierOpBatch(seq=1, offload=[(0, 0, 0, 2)], restore=[]))
    dma.issue(TierOpBatch(seq=2, offload=[], restore=[(0, 0, 0, 6)]))
    dma.fence_restores()
    assert dma.flush() == [1, 2]
    assert torch.equal(seg.blocks[6], src)
    assert bool((rows[6, width:] == 7).all())


def test_group_block_must_fit_the_slot():
    device = torch.device("cuda")
    with pytest.raises(AssertionError, match="exceeds"):
        KVTierDMA({0: [TierSegment.from_flat(_flat(2048), 2048)]}, 1024, SLOTS, device)


def test_segments_must_agree_on_block_count():
    device = torch.device("cuda")
    b = torch.zeros((BLOCKS + 1) * 512, dtype=torch.int8, device=device)
    with pytest.raises(AssertionError, match="block count"):
        KVTierDMA(
            {
                0: [TierSegment.from_flat(_flat(1024), 1024)],
                1: [TierSegment.from_flat(b, 512)],
            },
            STRIDE,
            SLOTS,
            device,
        )


def test_verify_digests_each_op(monkeypatch):
    import vllm.v1.worker.gpu.kv_tier_dma as mod

    monkeypatch.setattr(mod, "_VERIFY", True)
    device = torch.device("cuda")
    segs = {
        0: [TierSegment.from_flat(_flat(2048), 2048)],
        1: [TierSegment.from_flat(_flat(2048), 2048)],
    }
    dma = KVTierDMA(segs, STRIDE, SLOTS, device)
    for g in (0, 1):
        segs[g][0].blocks[1].random_(-128, 127)
    dma.issue(TierOpBatch(seq=1, offload=[(4, 0, 0, 1), (4, 2048, 1, 1)], restore=[]))
    dma.flush()
    dma.issue(TierOpBatch(seq=2, offload=[], restore=[(4, 0, 0, 2), (4, 2048, 1, 2)]))
    dma.fence_restores()
    dma.flush()
    assert dma._gpu_digest(0, 2) == dma._digests[(4, 0)]
    assert dma._gpu_digest(1, 2) == dma._digests[(4, 2048)]
