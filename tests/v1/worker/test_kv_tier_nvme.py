# SPDX-License-Identifier: Apache-2.0
"""NvmeTierFile: aligned O_DIRECT round trip through a temp tier file."""

import os
import time

import torch

from vllm.v1.worker.gpu.kv_tier_nvme import PAGE, DiskOp, NvmeTierFile


def _aligned_rows(rows: int, row_bytes: int) -> torch.Tensor:
    raw = torch.empty(rows * row_bytes + PAGE, dtype=torch.int8)
    off = (-raw.data_ptr()) % PAGE
    return raw[off : off + rows * row_bytes].view(rows, row_bytes)


def _wait(tier, ids, timeout=30.0):
    done = {}
    t0 = time.time()
    while len(done) < len(ids):
        for op_id, err in tier.poll_done():
            done[op_id] = err
        assert time.time() - t0 < timeout, "IO timed out"
        time.sleep(0.005)
    return done


def test_write_then_read_roundtrip(tmp_path):
    row = 3 * PAGE
    arena = _aligned_rows(4, row)
    src = torch.randint(-128, 127, (row,), dtype=torch.int8)
    arena[1].copy_(src)
    arena[2].zero_()
    tier = NvmeTierFile(str(tmp_path), num_slots=8, slot_bytes=row, threads=2)
    try:
        waited = []
        tier.submit(
            DiskOp(1, True, 5, memoryview(arena[1].numpy()), wait=lambda: waited.append(1))
        )
        assert _wait(tier, [1]) == {1: None}
        assert waited == [1]
        tier.submit(DiskOp(2, False, 5, memoryview(arena[2].numpy())))
        assert _wait(tier, [2]) == {2: None}
        assert torch.equal(arena[2], src)
        # The slot next door was never written: reads return the fallocated
        # zeros, not a neighbour's bytes.
        arena[3].fill_(7)
        tier.submit(DiskOp(3, False, 6, memoryview(arena[3].numpy())))
        _wait(tier, [3])
        assert int(arena[3].abs().sum()) == 0
    finally:
        tier.close()
    assert not os.listdir(tmp_path)  # unnamed (or unlinked) file: no residue


def test_many_concurrent_ops_complete(tmp_path):
    row = PAGE
    n = 64
    arena = _aligned_rows(2 * n, row)
    rows = torch.randint(-128, 127, (n, row), dtype=torch.int8)
    arena[:n].copy_(rows)
    tier = NvmeTierFile(str(tmp_path), num_slots=n, slot_bytes=row, threads=4)
    try:
        for i in range(n):
            tier.submit(DiskOp(i, True, (i * 7) % n, memoryview(arena[i].numpy())))
        assert all(e is None for e in _wait(tier, list(range(n))).values())
        for i in range(n):
            tier.submit(DiskOp(n + i, False, (i * 7) % n, memoryview(arena[n + i].numpy())))
        assert all(e is None for e in _wait(tier, list(range(n, 2 * n))).values())
        assert torch.equal(arena[n:], rows)
    finally:
        tier.close()

