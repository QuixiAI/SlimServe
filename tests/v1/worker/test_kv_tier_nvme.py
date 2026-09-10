# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVTierNVMe: offload/restore roundtrip through the unlinked tier file.

Runs on whatever device is present (MPS on Apple Silicon, CPU elsewhere);
the backend's device surface is copy_/zero_ plus optional completion events,
all of which exist everywhere.
"""

import pytest
import torch

import vllm.v1.worker.gpu.kv_tier_nvme as nvme_mod
from vllm.v1.worker.gpu.kv_tier_dma import TierOpBatch
from vllm.v1.worker.gpu.kv_tier_nvme import KVTierNVMe

STRIDE = 4096
BLOCKS = 8
SLOTS = 16


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def tier(tmp_path, monkeypatch):
    # The free-space gate wants tier size + 32 GiB slack; a test tmpdir may
    # be on a small volume, so waive the slack, not the check's existence.
    monkeypatch.setattr(
        KVTierNVMe, "_check_free_space", staticmethod(lambda d, s: None)
    )
    device = _device()
    backing = torch.zeros(BLOCKS * STRIDE, dtype=torch.int8, device=device)
    t = KVTierNVMe(backing, STRIDE, SLOTS, device, str(tmp_path))
    yield backing, t
    if t._io_error is not None:
        with pytest.raises(RuntimeError, match="IO thread failed"):
            t.shutdown()
    else:
        t.shutdown()


def _rand_row(device):
    return torch.randint(-128, 127, (STRIDE,), dtype=torch.int8, device=device)


def test_offload_then_restore_roundtrip(tier):
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    src = _rand_row(backing.device)
    blocks[3].copy_(src)

    t.issue(TierOpBatch(seq=1, offload=[(3, 7)], restore=[]))
    assert t.flush() == [1]
    blocks[3].zero_()
    t.issue(TierOpBatch(seq=2, offload=[], restore=[(7, 5)]))
    t.fence_restores()
    assert t.flush() == [2]
    assert torch.equal(blocks[5].cpu(), src.cpu())


def test_batches_complete_in_order(tier):
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    for b in range(4):
        blocks[b].fill_(b + 1)
    t.issue(TierOpBatch(seq=10, offload=[(0, 0), (1, 1)], restore=[]))
    t.issue(TierOpBatch(seq=11, offload=[(2, 2), (3, 3)], restore=[]))
    done = t.flush()
    assert done == [10, 11]


def test_empty_batch_is_a_noop(tier):
    _, t = tier
    t.issue(TierOpBatch(seq=1, offload=[], restore=[]))
    assert t.flush() == []


def test_restore_reads_after_pending_write_of_same_slot(tier):
    """The confirm-one-step-later protocol: a restore issued after an
    offload of the same slot must observe the offloaded bytes, however
    slow the write is. FIFO in the IO queue is the guarantee."""
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    src = _rand_row(backing.device)
    blocks[2].copy_(src)
    # Same call sequence a scheduler step produces: write slot 9, then a
    # later batch reads slot 9 into another block, no flush in between.
    t.issue(TierOpBatch(seq=1, offload=[(2, 9)], restore=[]))
    t.issue(TierOpBatch(seq=2, offload=[], restore=[(9, 6)]))
    assert t.flush() == [1, 2]
    assert torch.equal(blocks[6].cpu(), src.cpu())


def test_slot_reuse_write_after_write(tier):
    """LRU reclaim reuses a slot for a new trajectory: the second write
    must win for any restore issued after it."""
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    first = _rand_row(backing.device)
    second = _rand_row(backing.device)
    blocks[0].copy_(first)
    blocks[1].copy_(second)
    t.issue(TierOpBatch(seq=1, offload=[(0, 4)], restore=[]))
    t.issue(TierOpBatch(seq=2, offload=[(1, 4)], restore=[]))
    t.issue(TierOpBatch(seq=3, offload=[], restore=[(4, 7)]))
    assert t.flush() == [1, 2, 3]
    assert torch.equal(blocks[7].cpu(), second.cpu())


def test_zero_only_batch_zeroes_and_completes(tier):
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    blocks[5].fill_(11)
    t.issue(TierOpBatch(seq=4, offload=[], restore=[], zero=[5]))
    assert t.flush() == [4]
    assert int(blocks[5].abs().max().item()) == 0


def test_restore_batch_applies_zero_with_it(tier):
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    src = _rand_row(backing.device)
    blocks[0].copy_(src)
    blocks[6].fill_(3)  # stale ring block
    t.issue(TierOpBatch(seq=1, offload=[(0, 2)], restore=[]))
    t.issue(TierOpBatch(seq=2, offload=[], restore=[(2, 1)], zero=[6]))
    assert t.flush() == [1, 2]
    assert torch.equal(blocks[1].cpu(), src.cpu())
    assert int(blocks[6].abs().max().item()) == 0


def test_staging_budget_backpressure(tier, monkeypatch):
    """issue() must not drop offloads when the staging budget is exceeded;
    it waits for the IO thread to drain and every byte still lands."""
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    # Budget of ~2 rows forces the third issue to wait for a drain.
    monkeypatch.setattr(nvme_mod, "_STAGING_BUDGET_BYTES", 2 * STRIDE)
    rows = []
    for b in range(6):
        row = _rand_row(backing.device)
        blocks[b % BLOCKS].copy_(row)
        rows.append(row)
        t.issue(TierOpBatch(seq=100 + b, offload=[(b % BLOCKS, b)], restore=[]))
    assert t.flush() == [100 + b for b in range(6)]
    # Read each slot back and compare.
    for b in range(6):
        t.issue(TierOpBatch(seq=200 + b, offload=[], restore=[(b, 7)]))
        t.flush()
        assert torch.equal(blocks[7].cpu(), rows[b].cpu())


def test_file_is_unlinked_and_sized(tier, tmp_path):
    _, t = tier
    import os

    assert os.fstat(t.fd).st_size == SLOTS * STRIDE
    assert list(tmp_path.iterdir()) == []  # unlinked while open


def test_io_error_surfaces_on_main_thread(tier):
    backing, t = tier
    import os

    blocks = backing.view(-1, STRIDE)
    blocks[0].fill_(1)
    os.close(t.fd)  # simulate the volume dying under the tier
    t.issue(TierOpBatch(seq=1, offload=[(0, 0)], restore=[]))
    with pytest.raises(RuntimeError, match="IO thread failed"):
        t.flush()
    t.fd = os.open(os.devnull, os.O_RDWR)  # let fixture shutdown close something


def test_oversized_mixed_batch_is_bounded_and_completes_once(tier, monkeypatch):
    backing, t = tier
    blocks = backing.view(-1, STRIDE)
    expected = torch.randint(-128, 127, (4, STRIDE), dtype=torch.int8)
    blocks[:4].copy_(expected)
    monkeypatch.setattr(nvme_mod, "_STAGING_BUDGET_BYTES", 2 * STRIDE)
    peak = 0
    reserve = t._reserve_staging

    def checked_reserve(nbytes):
        nonlocal peak
        reserve(nbytes)
        peak = max(peak, t._staged_bytes)
        assert peak <= 2 * STRIDE

    monkeypatch.setattr(t, "_reserve_staging", checked_reserve)
    t.issue(
        TierOpBatch(
            seq=8,
            offload=[(i, i) for i in range(4)],
            restore=[(i, i + 4) for i in range(4)],
            zero=[4],
        )
    )
    assert t.flush() == [8]
    assert t._staged_bytes == 0
    assert torch.equal(blocks[4:].cpu(), expected)


def test_shutdown_drains_closes_and_is_idempotent(tier):
    import os

    backing, t = tier
    backing.fill_(7)
    t.issue(TierOpBatch(seq=1, offload=[(0, 0)], restore=[(0, 1)]))
    t.shutdown()
    assert not t._io_thread.is_alive()
    assert not t._jobs
    with pytest.raises(OSError):
        os.fstat(t.fd)
    t.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        t.issue(TierOpBatch(seq=2, offload=[(0, 0)], restore=[]))


@pytest.mark.parametrize("operation", ["read", "write"])
def test_zero_progress_io_fails_and_shutdown_reclaims(tier, monkeypatch, operation):
    import os

    _, t = tier
    if operation == "read":
        monkeypatch.setattr(os, "preadv", lambda *a: 0)
        batch = TierOpBatch(seq=1, offload=[], restore=[(0, 0)])
    else:
        monkeypatch.setattr(os, "pwrite", lambda *a: 0)
        batch = TierOpBatch(seq=1, offload=[(0, 0)], restore=[])
    t.issue(batch)
    with pytest.raises(RuntimeError, match="IO thread failed"):
        t.flush()


def test_partial_io_roundtrip(tier, monkeypatch):
    import os

    backing, t = tier
    expected = _rand_row(backing.device)
    backing.view(-1, STRIDE)[0].copy_(expected)
    write, read = os.pwrite, os.preadv
    monkeypatch.setattr(os, "pwrite", lambda fd, b, off: write(fd, b[:137], off))
    monkeypatch.setattr(os, "preadv", lambda fd, b, off: read(fd, [b[0][:139]], off))
    t.issue(TierOpBatch(seq=1, offload=[(0, 0)], restore=[(0, 1)]))
    assert t.flush() == [1]
    assert torch.equal(backing.view(-1, STRIDE)[1].cpu(), expected.cpu())


def test_fragmented_slots_and_blocks_preserve_mapping(tier, monkeypatch):
    backing, t = tier
    rows = backing.view(-1, STRIDE)
    expected = torch.randint(-128, 127, (BLOCKS, STRIDE), dtype=torch.int8)
    rows.copy_(expected)
    slots = [0, 1, 7, 3, 4, 9, 11, 12]
    sources = [2, 3, 0, 5, 6, 4, 1, 7]
    targets = [7, 6, 5, 4, 3, 2, 1, 0]
    monkeypatch.setattr(nvme_mod, "_STAGING_BUDGET_BYTES", 3 * STRIDE)
    t.issue(TierOpBatch(1, list(zip(sources, slots)), []))
    assert t.flush() == [1]
    rows.zero_()
    t.issue(TierOpBatch(2, [], list(zip(slots, targets))))
    assert t.flush() == [2]
    for source, target in zip(sources, targets):
        assert torch.equal(rows[target].cpu(), expected[source])


def test_verification_uses_digest_from_read_before_slot_reuse(tier, monkeypatch):
    backing, t = tier
    rows = backing.view(-1, STRIDE)
    rows[0].fill_(7)
    rows[1].fill_(9)
    monkeypatch.setattr(nvme_mod, "_VERIFY", True)
    t.issue(TierOpBatch(1, [(0, 0)], []))
    t.issue(TierOpBatch(2, [], [(0, 4)]))
    t.issue(TierOpBatch(3, [(1, 0)], []))
    t.issue(TierOpBatch(4, [], [(0, 5)]))
    assert t.flush() == [1, 2, 3, 4]
    assert torch.equal(rows[4].cpu(), torch.full((STRIDE,), 7, dtype=torch.int8))
    assert torch.equal(rows[5].cpu(), torch.full((STRIDE,), 9, dtype=torch.int8))


def test_file_setup_failure_closes_descriptor(tmp_path, monkeypatch):
    import os

    descriptors = []

    def fail(t, size):
        descriptors.append(t.fd)
        raise OSError("preallocation failed")

    monkeypatch.setattr(KVTierNVMe, "_check_free_space", staticmethod(lambda *a: None))
    monkeypatch.setattr(KVTierNVMe, "_preallocate", fail)
    with pytest.raises(OSError, match="preallocation failed"):
        KVTierNVMe(
            torch.zeros(STRIDE, dtype=torch.int8),
            STRIDE,
            1,
            torch.device("cpu"),
            str(tmp_path),
        )
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert not list(tmp_path.iterdir())
