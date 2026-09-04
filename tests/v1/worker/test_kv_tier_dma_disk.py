# SPDX-License-Identifier: Apache-2.0
"""KVTierDMA + NVMe tier: write-through after the host copy, promotion
reads that gate a restore, and load-error reporting."""

import time

import pytest
import torch

from vllm.v1.worker.gpu.kv_tier_dma import KVTierDMA, TierOpBatch, padded_stride
from vllm.v1.worker.gpu.kv_tier_nvme import NvmeTierFile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

STRIDE = 5000  # deliberately not a page multiple
BLOCKS = 8
SLOTS = 16
DISK = 32


def _dma(tmp_path):
    device = torch.device("cuda")
    backing = torch.zeros(BLOCKS * STRIDE, dtype=torch.int8, device=device)
    disk = NvmeTierFile(str(tmp_path), DISK, padded_stride(STRIDE), threads=2)
    return backing, KVTierDMA(backing, STRIDE, SLOTS, device, disk=disk)


def _spin(dma, cond, timeout=30.0):
    t0 = time.time()
    while not cond():
        dma.pump()
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.002)


def test_write_through_then_promote_restores_bytes(tmp_path):
    backing, dma = _dma(tmp_path)
    try:
        blocks = backing.view(-1, STRIDE)
        src = torch.randint(-128, 127, (STRIDE,), dtype=torch.int8, device="cuda")
        blocks[2].copy_(src)
        # Fill: GPU block 2 -> host slot 7 (batch 1); write-through host 7
        # -> disk 11 (batch 2, "one confirmation later").
        dma.issue(TierOpBatch(seq=1, offload=[(2, 7, 0)], restore=[]))
        dma.issue(TierOpBatch(seq=2, offload=[], restore=[], disk_writes=[(7, 11)]))
        done = []
        _spin(dma, lambda: (done.extend(dma.take_disk_done()) or 2 in done))
        # Demote: clobber the host slot; promote into a fresh slot 9 from
        # disk, then restore slot 9 -> GPU block 5.
        dma.flush()
        dma.arena[7].zero_()
        blocks[5].zero_()
        dma.issue(
            TierOpBatch(
                seq=3, offload=[], restore=[(9, 5, 0)], disk_reads=[(11, 9)], req_id="r"
            )
        )
        # Not issued to the copy stream until the read lands.
        assert dma.poll_done() == [] and dma._deferred
        _spin(dma, lambda: not dma._deferred)
        dma.fence_restores()
        torch.cuda.synchronize()
        assert dma.flush() == [3]
        assert torch.equal(blocks[5], src)
        assert dma.take_invalid_blocks() == set()
    finally:
        dma.release()


def test_later_restore_of_same_request_waits_for_reads(tmp_path):
    backing, dma = _dma(tmp_path)
    try:
        blocks = backing.view(-1, STRIDE)
        a = torch.randint(-128, 127, (STRIDE,), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 127, (STRIDE,), dtype=torch.int8, device="cuda")
        blocks[0].copy_(a)
        blocks[1].copy_(b)
        dma.issue(TierOpBatch(seq=1, offload=[(0, 1, 0), (1, 2, 0)], restore=[]))
        dma.issue(TierOpBatch(seq=2, offload=[], restore=[], disk_writes=[(1, 3), (2, 4)]))
        done = []
        _spin(dma, lambda: (done.extend(dma.take_disk_done()) or 2 in done))
        dma.flush()
        dma.arena[1].zero_()
        dma.arena[2].zero_()
        blocks[6].zero_()
        blocks[7].zero_()
        # Chunk 1 carries the reads for both slots; chunk 2 (no reads) must
        # still wait for them.
        dma.issue(
            TierOpBatch(seq=3, offload=[], restore=[(1, 6, 0)], disk_reads=[(3, 1), (4, 2)], req_id="q")
        )
        dma.issue(TierOpBatch(seq=4, offload=[], restore=[(2, 7, 0)], req_id="q"))
        assert len(dma._deferred) == 2
        _spin(dma, lambda: not dma._deferred)
        dma.fence_restores()
        torch.cuda.synchronize()
        assert sorted(dma.flush()) == [3, 4]
        assert torch.equal(blocks[6], a) and torch.equal(blocks[7], b)
    finally:
        dma.release()


def test_failed_read_reports_invalid_blocks(tmp_path):
    backing, dma = _dma(tmp_path)
    try:
        blocks = backing.view(-1, STRIDE)
        blocks[3].fill_(5)
        # Reading a slot beyond the file's end fails the IO; the restore's
        # target block is reported invalid and never copied.
        dma.disk.num_slots = DISK + 1  # let submit past the bounds check
        dma.issue(
            TierOpBatch(seq=1, offload=[], restore=[(0, 3, 0)], disk_reads=[(DISK, 0)], req_id="x")
        )
        _spin(dma, lambda: not dma._deferred)
        dma.flush()
        assert dma.take_invalid_blocks() == {3}
        assert int(blocks[3][0].item()) == 5  # untouched
    finally:
        dma.release()


def test_verify_digest_survives_promotion(tmp_path, monkeypatch):
    """VLLM_KV_TIER_VERIFY: a promoted row (fresh host slot, bytes read
    back from disk) must be checked against the ORIGINAL offload digest -
    carried by the write-through's IO-time hash - so a faithful disk tier
    reports 0 mismatches (58/58 false mismatches on 2026-09-04)."""
    import vllm.v1.worker.gpu.kv_tier_dma as dma_mod
    import vllm.v1.worker.gpu.kv_tier_nvme as nvme_mod

    monkeypatch.setattr(dma_mod, "_VERIFY", True)
    monkeypatch.setattr(nvme_mod, "_VERIFY_ROWS", True)
    backing, dma = _dma(tmp_path)
    try:
        blocks = backing.view(-1, STRIDE)
        live = 3000  # the group's live bytes (a shorter-than-row page)
        dma._group_nbytes = {0: live}
        src = torch.randint(-128, 127, (STRIDE,), dtype=torch.int8, device="cuda")
        blocks[2].copy_(src)
        dma.issue(TierOpBatch(seq=1, offload=[(2, 7, 0)], restore=[]))
        # Write-through handed BEFORE the worker polled batch 1 (as the
        # scheduler does: it confirms one step later, not on completion).
        dma.issue(TierOpBatch(seq=2, offload=[], restore=[], disk_writes=[(7, 11, 0)]))
        done = []
        _spin(dma, lambda: (done.extend(dma.take_disk_done()) or 2 in done))
        dma.flush()
        original = dma._slot_digests[7]
        dma.arena[7].zero_()
        blocks[5].zero_()
        dma.issue(
            TierOpBatch(
                seq=3, offload=[], restore=[(9, 5, 0)], disk_reads=[(11, 9)], req_id="r"
            )
        )
        done = []
        _spin(dma, lambda: (done.extend(dma.poll_done()) or 3 in done))
        assert dma._slot_digests[9] == original, "promoted row lost its digest"
        assert torch.equal(blocks[5][:live].cpu(), src[:live].cpu())
    finally:
        dma.release()
