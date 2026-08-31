# SPDX-License-Identifier: Apache-2.0
"""Worker-side host KV tier: pinned arena + async DMA queues.

Each rank owns a pinned host arena shaped [num_slots, block_stride] that
mirrors its shard of the packed KV slab. Slot ids are global (the scheduler
assigns them); every rank executes the same (block, slot) operations on its
own shard, so no cross-rank coordination is needed.

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


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()[:12]


@dataclass
class TierOpBatch:
    """One scheduler-issued batch of tier operations."""

    seq: int
    offload: list[tuple[int, int]]  # (gpu_block, host_slot)
    restore: list[tuple[int, int]]  # (host_slot, gpu_block)
    # GPU blocks to zero with the batch (a resumed request's ring block:
    # the framework never zeroes ring blocks, so this is defensive
    # determinism over the stale-claimed bytes internal hits run on).
    zero: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.zero is None:
            self.zero = []


class KVTierDMA:
    def __init__(
        self,
        backing: torch.Tensor,
        block_stride: int,
        num_slots: int,
        device: torch.device,
    ):
        assert backing.dtype == torch.int8 and backing.is_cuda
        assert backing.numel() % block_stride == 0
        self.blocks = backing.view(-1, block_stride)
        self.device = device
        # One pinned arena; rows are DMA-contiguous with the GPU block rows.
        self.arena = torch.empty(
            (num_slots, block_stride), dtype=torch.int8, pin_memory=True
        )
        self.copy_stream = torch.cuda.Stream(device)
        # In-flight batches: (batch, event) in issue order; events complete
        # in stream order, so polling stops at the first incomplete one.
        self._inflight: list[tuple[TierOpBatch, torch.cuda.Event]] = []
        self._restore_events: list[torch.cuda.Event] = []
        self._slot_digests: dict[int, str] = {}
        # Cumulative op counters, logged periodically: the tier moves bytes
        # silently otherwise, which makes "acceptance passed" claims
        # unfalsifiable (a full re-prefill answers recall probes just as
        # well as a restore does).
        self._total_offloads = 0
        self._total_restores = 0
        self._batches_since_log = 0

    def issue(self, batch: TierOpBatch) -> None:
        """Enqueue a batch of copies on the copy stream."""
        if not batch.offload and not batch.restore and not batch.zero:
            return
        main = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            # Offloaded blocks were produced on the compute stream.
            self.copy_stream.wait_stream(main)
            for gpu_block, slot in batch.offload:
                self.arena[slot].copy_(self.blocks[gpu_block], non_blocking=True)
            for gpu_block in batch.zero:
                self.blocks[gpu_block].zero_()
            for slot, gpu_block in batch.restore:
                self.blocks[gpu_block].copy_(self.arena[slot], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        self._inflight.append((batch, event))
        if batch.restore or batch.zero:
            self._restore_events.append(event)
        self._total_offloads += len(batch.offload)
        self._total_restores += len(batch.restore)
        self._batches_since_log += 1
        if batch.restore or self._batches_since_log >= 200:
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "kv-tier dma: totals offload=%d restore=%d (+%d/+%d this batch)",
                self._total_offloads,
                self._total_restores,
                len(batch.offload),
                len(batch.restore),
            )
            self._batches_since_log = 0

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

    def _verify(self, batch: TierOpBatch) -> None:
        from vllm.logger import init_logger

        logger = init_logger(__name__)
        for gpu_block, slot in batch.offload:
            self._slot_digests[slot] = _digest(self.arena[slot])
        bad = 0
        for slot, gpu_block in batch.restore:
            expect = self._slot_digests.get(slot)
            got = _digest(self.blocks[gpu_block])
            host = _digest(self.arena[slot])
            if expect is not None and got != expect:
                bad += 1
                logger.warning(
                    "kv-tier VERIFY MISMATCH slot=%d block=%d offloaded=%s "
                    "host_now=%s gpu_after_restore=%s",
                    slot, gpu_block, expect, host, got,
                )
        if batch.restore:
            logger.info(
                "kv-tier verify: batch %d: %d/%d restores mismatched",
                batch.seq, bad, len(batch.restore),
            )

    def flush(self) -> list[int]:
        """Synchronously drain all in-flight batches (shutdown/tests)."""
        self.copy_stream.synchronize()
        return self.poll_done()
