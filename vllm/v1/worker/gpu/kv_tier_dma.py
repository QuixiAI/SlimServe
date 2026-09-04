# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side host KV tier: pinned arena + async DMA queues.

Each rank owns a pinned host arena shaped [num_slots, slot_bytes]. Slot ids
are global (the scheduler assigns them); every rank executes the same
operations on its own shard, so no cross-rank coordination is needed.

The GPU side is, per cache group, a list of segments: [num_blocks, width]
int8 views whose row `b` holds block `b`'s bytes for the layers that
segment covers. A packed cross-layer slab gives every group ONE segment (a
block-strided view of the slab, so each op is a single flat copy - the
measured fast path); per-layer paged tensors give a group one segment per
layer (L copies per op, the correctness fallback).

An operation is (slot, col, group, gpu_block): the group's block bytes are
copied between the arena row `slot` at column `col` and the group's
segments. The scheduler lays out what shares a slot row - a position's
sub-blocks of every full-attention group, or one tail block - and both
roles derive the layout from KVCacheConfig.group_block_bytes.

Full blocks are immutable, which shapes the whole design:
- Offload is write-once at block-fill time and never on the critical path.
- Restore repopulates GPU blocks whose contents will not change.
- There is no dirty state and no write-back.

Ordering contract:
- Offload ops for blocks filled at step N are enqueued at step N+1 on the
  copy stream, which first waits on the compute stream (the fill).
- Restore ops copy host->GPU on the copy stream; the scheduler only makes
  the waiting request schedulable after the worker reports the op batch
  complete, and the next step's compute stream waits on the restore event,
  so no kernel can read a half-restored block.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch

_VERIFY = os.environ.get("VLLM_KV_TIER_VERIFY", "0") == "1"

# (slot, col, group, gpu_block)
TierOp = tuple[int, int, int, int]


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:12]


@dataclass
class TierOpBatch:
    """One scheduler-issued batch of tier operations."""

    seq: int
    offload: list[TierOp]
    restore: list[TierOp]
    # (group, gpu_block) pairs to zero with the batch (a resumed request's
    # ring block: the framework never zeroes ring blocks, so this is
    # defensive determinism over the stale-claimed bytes internal hits
    # run on).
    zero: list[tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.zero is None:
            self.zero = []


@dataclass
class TierSegment:
    """One GPU allocation's contribution to a group's block.

    `blocks` is a [num_blocks, width] int8 view: row b holds block b's bytes
    for the layers this segment covers. Rows need not be contiguous with
    each other (a packed-slab view is block-strided), but each row is.
    """

    blocks: torch.Tensor

    @property
    def width(self) -> int:
        return self.blocks.shape[1]

    @classmethod
    def from_flat(cls, backing: torch.Tensor, row_bytes: int) -> TierSegment:
        """A flat int8 allocation whose blocks are consecutive rows."""
        assert backing.dtype == torch.int8 and backing.is_cuda
        assert backing.numel() % row_bytes == 0, (
            f"allocation of {backing.numel()} bytes is not a whole number "
            f"of {row_bytes}-byte blocks"
        )
        return cls(backing.view(-1, row_bytes))


class KVTierDMA:
    def __init__(
        self,
        group_segments: dict[int, list[TierSegment]],
        slot_bytes: int,
        num_slots: int,
        device: torch.device,
    ):
        assert group_segments, "host tier needs at least one KV group"
        self.segments = group_segments
        self.group_bytes: dict[int, int] = {}
        num_blocks: set[int] = set()
        for gid, segs in group_segments.items():
            assert segs, f"group {gid} has no segments"
            for seg in segs:
                assert seg.blocks.dtype == torch.int8 and seg.blocks.is_cuda
                assert seg.blocks.dim() == 2
                num_blocks.add(seg.blocks.shape[0])
            self.group_bytes[gid] = sum(seg.width for seg in segs)
            assert self.group_bytes[gid] <= slot_bytes, (
                f"group {gid} block ({self.group_bytes[gid]} B) exceeds the "
                f"{slot_bytes}-byte slot"
            )
        assert len(num_blocks) == 1, (
            f"segments disagree on block count: {sorted(num_blocks)}"
        )
        self.num_blocks = num_blocks.pop()
        self.slot_bytes = slot_bytes
        self.device = device
        # The arena is pinned in chunks with retry: one giant hipHostMalloc
        # racing other ranks' pins fails fast with hipErrorOutOfMemory while
        # the kernel is still reclaiming page cache (observed 2026-08-31:
        # 4/8 ranks pinned 256 GiB, the rest OOMed instantly, and a lone
        # process pinning the same size succeeded). Chunks small enough for
        # reclaim to keep pace, plus a bounded backoff, make the pin robust.
        from vllm.logger import init_logger

        logger = init_logger(__name__)
        chunk_bytes = 16 << 30
        self._rows_per_chunk = max(1, chunk_bytes // slot_bytes)
        self._chunks: list[torch.Tensor] = []
        remaining = num_slots
        while remaining > 0:
            rows = min(self._rows_per_chunk, remaining)
            for attempt in range(60):
                try:
                    self._chunks.append(
                        torch.empty(
                            (rows, slot_bytes), dtype=torch.int8, pin_memory=True
                        )
                    )
                    break
                except Exception as e:  # torch.AcceleratorError is not a RuntimeError
                    if attempt % 10 == 0:
                        pinned = sum(c.numel() for c in self._chunks) >> 30
                        host_stats = ""
                        try:
                            hs = torch.cuda.host_memory_stats()
                            h_alloc = hs.get("allocated_bytes.all.current", 0) >> 30
                            h_res = hs.get("reserved_bytes.all.current", 0) >> 30
                            host_stats = (
                                f" host_alloc={h_alloc}G host_reserved={h_res}G"
                            )
                        except Exception:
                            pass
                        logger.warning(
                            "host-tier: pin chunk %d (%d rows) attempt %d failed "
                            "(%s); %d GiB pinned so far%s",
                            len(self._chunks),
                            rows,
                            attempt,
                            str(e)[:80],
                            pinned,
                            host_stats,
                        )
                    if attempt == 59 or "out of memory" not in str(e).lower():
                        raise
                    import time

                    time.sleep(5.0)
            remaining -= rows
        self.num_slots = num_slots
        self.copy_stream = torch.cuda.Stream(device)
        # In-flight batches: (batch, event) in issue order; events complete
        # in stream order, so polling stops at the first incomplete one.
        self._inflight: list[tuple[TierOpBatch, torch.cuda.Event]] = []
        self._restore_events: list[torch.cuda.Event] = []
        self._digests: dict[tuple[int, int], str] = {}

    def row(self, slot: int) -> torch.Tensor:
        return self._chunks[slot // self._rows_per_chunk][slot % self._rows_per_chunk]

    def _copy(self, op: TierOp, to_host: bool) -> None:
        slot, col, gid, block = op
        row = self.row(slot)
        for seg in self.segments[gid]:
            w = seg.width
            if to_host:
                row[col : col + w].copy_(seg.blocks[block], non_blocking=True)
            else:
                seg.blocks[block].copy_(row[col : col + w], non_blocking=True)
            col += w

    def issue(self, batch: TierOpBatch) -> None:
        """Enqueue a batch of copies on the copy stream."""
        if not batch.offload and not batch.restore and not batch.zero:
            return
        main = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            # Offloaded blocks were produced on the compute stream.
            self.copy_stream.wait_stream(main)
            for op in batch.offload:
                self._copy(op, to_host=True)
            for gid, block in batch.zero:
                for seg in self.segments[gid]:
                    seg.blocks[block].zero_()
            for op in batch.restore:
                self._copy(op, to_host=False)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        self._inflight.append((batch, event))
        if batch.restore or batch.zero:
            self._restore_events.append(event)

    def fence_restores(self) -> None:
        """Make the compute stream wait on all outstanding restore copies.

        Called at the top of each step: restored blocks the scheduler has
        already handed to a request must be fully resident before any kernel
        can read them.
        """
        if not self._restore_events:
            return
        main = torch.cuda.current_stream(self.device)
        for event in self._restore_events:
            main.wait_event(event)
        self._restore_events.clear()

    def poll_done(self) -> list[int]:
        """Return seq ids of batches whose copies have completed."""
        done: list[int] = []
        while self._inflight and self._inflight[0][1].query():
            batch = self._inflight[0][0]
            done.append(batch.seq)
            self._inflight.pop(0)
            if _VERIFY:
                self._verify(batch)
        return done

    def _gpu_digest(self, gid: int, block: int) -> str:
        segs = self.segments[gid]
        if len(segs) == 1:
            return _digest(segs[0].blocks[block])
        return _digest(torch.cat([seg.blocks[block] for seg in segs]))

    def _host_digest(self, slot: int, col: int, gid: int) -> str:
        return _digest(self.row(slot)[col : col + self.group_bytes[gid]])

    def _verify(self, batch: TierOpBatch) -> None:
        from vllm.logger import init_logger

        logger = init_logger(__name__)
        for slot, col, gid, _ in batch.offload:
            self._digests[(slot, col)] = self._host_digest(slot, col, gid)
        bad = 0
        for slot, col, gid, block in batch.restore:
            expect = self._digests.get((slot, col))
            got = self._gpu_digest(gid, block)
            host = self._host_digest(slot, col, gid)
            if expect is not None and got != expect:
                bad += 1
                logger.warning(
                    "kv-tier VERIFY MISMATCH slot=%d col=%d group=%d block=%d "
                    "offloaded=%s host_now=%s gpu_after_restore=%s",
                    slot,
                    col,
                    gid,
                    block,
                    expect,
                    host,
                    got,
                )
        if batch.restore:
            logger.info(
                "kv-tier verify: batch %d: %d/%d restores mismatched",
                batch.seq,
                bad,
                len(batch.restore),
            )

    def flush(self) -> list[int]:
        """Synchronously drain all in-flight batches (shutdown/tests)."""
        self.copy_stream.synchronize()
        return self.poll_done()
