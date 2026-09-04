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

from vllm.logger import init_logger
from vllm.v1.worker.gpu.kv_tier_nvme import PAGE, DiskOp, NvmeTierFile

logger = init_logger(__name__)

_VERIFY = os.environ.get("VLLM_KV_TIER_VERIFY", "0") == "1"


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()[:12]


@dataclass
class TierOpBatch:
    """One scheduler-issued batch of tier operations."""

    seq: int
    offload: list[tuple[int, int, int]]  # (gpu_block, host_slot, kv_group)
    restore: list[tuple[int, int, int]]  # (host_slot, gpu_block, kv_group)
    # GPU blocks to zero with the batch (a resumed request's ring block:
    # the framework never zeroes ring blocks, so this is defensive
    # determinism over the stale-claimed bytes internal hits run on).
    zero: list[int] = None  # type: ignore[assignment]
    # NVMe tier: write-through (host_slot, disk_slot) of rows whose GPU ->
    # host copy an EARLIER batch produced (the scheduler stages them one
    # confirmation later); promotion reads (disk_slot, host_slot) that must
    # land before this request's restore copies run.
    disk_writes: list[tuple[int, int]] = None  # type: ignore[assignment]
    disk_reads: list[tuple[int, int]] = None  # type: ignore[assignment]
    req_id: str | None = None

    def __post_init__(self) -> None:
        if self.zero is None:
            self.zero = []
        if self.disk_writes is None:
            self.disk_writes = []
        if self.disk_reads is None:
            self.disk_reads = []

    @property
    def is_empty(self) -> bool:
        return not (
            self.offload or self.restore or self.zero or self.disk_writes or self.disk_reads
        )


_HOST_REGISTER_PORTABLE = 0x01


def padded_stride(block_stride: int) -> int:
    """Arena row width: the block stride rounded up to a 4 KiB multiple so
    every row is an O_DIRECT-able buffer and file slot."""
    return -(-block_stride // PAGE) * PAGE


def _register_host_arena(
    num_slots: int, block_stride: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Exactly-sized, page-aligned page-locked host arena of padded rows.

    torch's caching pinned allocator (pin_memory=True) rounds every
    allocation up to the next power of two, which turned an 88 GiB/rank
    arena into 128 GiB/rank (1 TiB across 8 ranks) and got the workers
    OOM-killed on a 995 GB box, while 64 GiB (a power of two) fit. Measured
    2026-09-02: 3 GiB pin_memory -> 4,190 MiB of RAM; 3 GiB registered ->
    3,118 MiB.

    Returns (arena_view, registered_buffer). The whole backing buffer is
    registered from its own base (is_pinned() is answered for a view by
    its storage base, and cudaHostUnregister needs the registered
    pointer); the arena is the page-aligned window inside it. Falls back
    to the caching pinned allocator only if registration is refused, and
    says so, since that fallback can cost up to 2x the requested RAM.
    """
    row = padded_stride(block_stride)
    total = num_slots * row
    raw = torch.empty(total + PAGE, dtype=torch.int8)
    rc = torch.cuda.cudart().cudaHostRegister(
        raw.data_ptr(), raw.numel(), _HOST_REGISTER_PORTABLE
    )
    if rc == 0 and raw.is_pinned():
        off = (-raw.data_ptr()) % PAGE
        return raw[off : off + total].view(num_slots, row), raw
    logger.warning(
        "kv-tier: cudaHostRegister of the %.1f GiB arena failed (%s); "
        "falling back to pin_memory=True, which rounds up to the next "
        "power of two",
        total / (1 << 30),
        rc,
    )
    del raw
    return torch.empty((num_slots, row), dtype=torch.int8, pin_memory=True), None


class KVTierDMA:
    def __init__(
        self,
        backing: torch.Tensor,
        block_stride: int,
        num_slots: int,
        device: torch.device,
        group_nbytes: dict[int, int] | None = None,
        disk: NvmeTierFile | None = None,
    ):
        assert backing.dtype == torch.int8 and backing.is_cuda
        assert backing.numel() % block_stride == 0
        self.blocks = backing.view(-1, block_stride)
        self.disk = disk
        if disk is not None:
            assert disk.slot_bytes == padded_stride(block_stride)
        # Live bytes per KV-cache group within a row (each group's layers
        # occupy a contiguous [0, n) prefix of its own rows); copies stop
        # at the group's extent. Unknown group -> whole row.
        self._group_nbytes = group_nbytes or {}
        self._stride = block_stride
        self.device = device
        # One pinned arena (exact size, page-aligned padded rows); rows are
        # DMA-contiguous with the GPU block rows.
        self.arena, self._registered_buf = _register_host_arena(num_slots, block_stride)
        self._registered = self._registered_buf is not None
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
        # NVMe tier bookkeeping (all on the caller's thread; the IO threads
        # only see DiskOps and report op ids back through poll_done).
        # Last copy-stream event that wrote each host row (offload target):
        # a disk write of that row waits on it before reading the bytes.
        self._slot_event: dict[int, torch.cuda.Event] = {}
        self._disk_op_seq = 0
        # op id -> ("w", batch_seq) | ("r", req_id)
        self._disk_ops: dict[int, tuple[str, object]] = {}
        self._write_remaining: dict[int, int] = {}
        self._disk_done: list[int] = []
        self._read_remaining: dict[str, int] = {}
        self._read_failed: set[str] = set()
        # Restore batches waiting for their request's disk reads, in order.
        self._deferred: list[TierOpBatch] = []
        self._invalid_blocks: set[int] = set()
        # VERIFY: offload digests keyed by host slot travel with the bytes
        # through the disk tier, so a promoted restore (new host slot) is
        # checked against the ORIGINAL offload digest, not whatever block
        # last used that host slot.
        self._disk_digests: dict[int, str] = {}
        self._read_targets: dict[int, tuple[int, int]] = {}

    def issue(self, batch: TierOpBatch) -> None:
        """Enqueue a batch: copies on the copy stream, disk ops to the IO
        engine. A batch carrying (or following) disk reads for its request
        is held back until those reads have landed in the arena."""
        if batch.is_empty:
            return
        if batch.disk_reads or (
            batch.req_id is not None and self._read_remaining.get(batch.req_id, 0) > 0
        ):
            assert self.disk is not None and batch.req_id is not None
            for disk_slot, host_slot in batch.disk_reads:
                self._submit_disk(
                    write=False, disk_slot=disk_slot, host_slot=host_slot,
                    tag=("r", batch.req_id),
                )
            self._read_remaining[batch.req_id] = (
                self._read_remaining.get(batch.req_id, 0) + len(batch.disk_reads)
            )
            batch.disk_reads = []
            self._deferred.append(batch)
            return
        self._issue_copies(batch)

    def _submit_disk(
        self, write: bool, disk_slot: int, host_slot: int, tag: tuple[str, object]
    ) -> None:
        self._disk_op_seq += 1
        op_id = self._disk_op_seq
        self._disk_ops[op_id] = tag
        wait = None
        if write:
            ev = self._slot_event.get(host_slot)
            if ev is not None:
                wait = ev.synchronize
            if _VERIFY and host_slot in self._slot_digests:
                self._disk_digests[disk_slot] = self._slot_digests[host_slot]
        elif _VERIFY:
            self._read_targets[op_id] = (disk_slot, host_slot)
        self.disk.submit(
            DiskOp(
                op_id=op_id,
                write=write,
                disk_slot=disk_slot,
                buffer=memoryview(self.arena[host_slot].numpy()),
                wait=wait,
            )
        )

    def pump(self) -> None:
        """Absorb IO completions: account write-through batches, and issue
        restore batches whose disk reads have all landed."""
        if self.disk is None:
            return
        for op_id, err in self.disk.poll_done():
            kind, key = self._disk_ops.pop(op_id)
            if _VERIFY and kind == "r":
                target = self._read_targets.pop(op_id, None)
                if target is not None and err is None:
                    disk_slot, host_slot = target
                    d = self._disk_digests.get(disk_slot)
                    if d is not None:
                        self._slot_digests[host_slot] = d
            if kind == "w":
                left = self._write_remaining[key] - 1
                if left <= 0:
                    del self._write_remaining[key]
                    # A failed write leaves the disk slot unconfirmed for
                    # the scheduler (never reported done): the block stays
                    # host-resident and the trajectory is not demotable.
                    if err is None:
                        self._disk_done.append(key)
                else:
                    self._write_remaining[key] = left
                continue
            req_id = key
            if err is not None:
                self._read_failed.add(req_id)
            left = self._read_remaining.get(req_id, 0) - 1
            if left > 0:
                self._read_remaining[req_id] = left
                continue
            self._read_remaining.pop(req_id, None)
            failed = req_id in self._read_failed
            self._read_failed.discard(req_id)
            ready = [b for b in self._deferred if b.req_id == req_id]
            self._deferred = [b for b in self._deferred if b.req_id != req_id]
            for b in ready:
                if failed:
                    # Report the target blocks as failed loads so the
                    # scheduler recomputes them instead of reading garbage.
                    self._invalid_blocks.update(g for _, g, _gid in b.restore)
                    b.restore = []
                self._issue_copies(b)

    def take_disk_done(self) -> list[int]:
        done, self._disk_done = self._disk_done, []
        return done

    def take_invalid_blocks(self) -> set[int]:
        inv, self._invalid_blocks = self._invalid_blocks, set()
        return inv

    def _issue_copies(self, batch: TierOpBatch) -> None:
        """Enqueue a batch of copies on the copy stream (and its disk
        write-through, which waits on the producing copies' events)."""
        if batch.disk_writes:
            assert self.disk is not None
            self._write_remaining[batch.seq] = len(batch.disk_writes)
            for host_slot, disk_slot in batch.disk_writes:
                self._submit_disk(
                    write=True, disk_slot=disk_slot, host_slot=host_slot,
                    tag=("w", batch.seq),
                )
        if not batch.offload and not batch.restore and not batch.zero:
            return
        main = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            # Offloaded blocks were produced on the compute stream.
            self.copy_stream.wait_stream(main)
            for gpu_block, slot, gid in batch.offload:
                n = self._group_nbytes.get(gid, self._stride)
                self.arena[slot][:n].copy_(
                    self.blocks[gpu_block][:n], non_blocking=True
                )
            for gpu_block in batch.zero:
                self.blocks[gpu_block].zero_()
            for slot, gpu_block, gid in batch.restore:
                n = self._group_nbytes.get(gid, self._stride)
                self.blocks[gpu_block][:n].copy_(
                    self.arena[slot][:n], non_blocking=True
                )
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        for _, slot, _gid in batch.offload:
            self._slot_event[slot] = event
        self._inflight.append((batch, event))
        if batch.restore or batch.zero:
            self._restore_events.append(event)
        self._total_offloads += len(batch.offload)
        self._total_restores += len(batch.restore)
        self._batches_since_log += 1
        if batch.restore or self._batches_since_log >= 200:
            logger.info(
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
        self.pump()
        done: list[int] = []
        while self._inflight and self._inflight[0][1].query():
            batch = self._inflight[0][0]
            done.append(batch.seq)
            self._inflight.pop(0)
            if _VERIFY:
                self._verify(batch)
        return done

    def _verify(self, batch: TierOpBatch) -> None:
        for gpu_block, slot, gid in batch.offload:
            n = self._group_nbytes.get(gid, self._stride)
            self._slot_digests[slot] = _digest(self.arena[slot][:n])
        bad = 0
        for slot, gpu_block, gid in batch.restore:
            n = self._group_nbytes.get(gid, self._stride)
            expect = self._slot_digests.get(slot)
            got = _digest(self.blocks[gpu_block][:n])
            host = _digest(self.arena[slot][:n])
            if expect is not None and got != expect:
                bad += 1
                logger.warning(
                    "kv-tier VERIFY MISMATCH slot=%d block=%d gid=%d offloaded=%s "
                    "host_now=%s gpu_after_restore=%s",
                    slot, gpu_block, gid, expect, host, got,
                )
        if batch.restore:
            logger.info(
                "kv-tier verify: batch %d: %d/%d restores mismatched",
                batch.seq, bad, len(batch.restore),
            )

    def flush(self) -> list[int]:
        """Synchronously drain all in-flight batches (shutdown/tests)."""
        import time

        while self.disk is not None and (self._disk_ops or self._deferred):
            self.pump()
            if self._disk_ops:
                time.sleep(0.001)
        self.copy_stream.synchronize()
        return self.poll_done()

    def release(self) -> None:
        """Drain, close the disk tier, then unregister the arena BEFORE its
        storage can be freed (dropping page-locked memory while it is still
        cudaHostRegistered leaves dangling pinned-page state that corrupts
        later CUDA work in the process - the PLE shared-table lesson)."""
        if getattr(self, "_registered", False):
            self.flush()
            if self.disk is not None:
                self.disk.close()
                self.disk = None
            torch.cuda.cudart().cudaHostUnregister(self._registered_buf.data_ptr())
            self._registered = False

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass
