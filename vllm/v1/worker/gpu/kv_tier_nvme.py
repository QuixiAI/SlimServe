# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side NVMe KV tier for unified-memory (Metal) platforms.

Same slot/batch contract as KVTierDMA, backed by one unlinked file per rank
instead of a pinned arena. On Apple Silicon there is no host-RAM tier to
offload to - host RAM and the KV pool are the same physical bytes, and every
byte of staging is a byte the Metal working set loses - so the second tier is
the NVMe device, reached with explicit pread/pwrite (never mmap: file-backed
page cache competes with Metal buffers and feeds the VM compressor).

Threading model (MPS has exactly one stream, and it belongs to the main
thread):
- The main thread does every MPS operation: D2H staging copies at issue(),
  H2D apply copies and ring zeroes at pump time, and all event records.
- One daemon IO thread does pure file IO in strict batch-issue order. FIFO
  is the correctness backbone: the scheduler confirms offload slots one step
  after issue without waiting for durability, which is safe only because a
  later restore (or slot-reuse write) of the same slot queues behind the
  pending write.
- An offload job's pwrite waits for the main thread to observe its D2H event
  (events on the single stream signal in issue order, so head-blocking cannot
  deadlock). A restore job's pread runs as soon as it reaches the queue head;
  its bytes are then applied H2D by the main thread, and the batch is
  reported complete only after the apply event has signaled. The scheduler
  keeps the request in WAITING_FOR_REMOTE_KVS until then, so at least one
  full step separates the apply from the first dependent kernel.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.kv_tier_dma import TierOpBatch

logger = init_logger(__name__)

_VERIFY = os.environ.get("VLLM_KV_TIER_VERIFY", "0") == "1"

# Bound on CPU staging held by not-yet-written offloads plus not-yet-applied
# restores. Steady state is tens of MiB; the cap only matters under a
# pathological burst, where issue() waits for the IO thread to drain.
_STAGING_BUDGET_BYTES = 512 << 20


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()[:12]


def _make_event():
    """A recorded, queryable completion event - or None when there is no
    accelerator (pure-CPU tests), where every copy is already synchronous."""
    if torch.backends.mps.is_available():
        ev = torch.mps.Event()
        ev.record()
        return ev
    if torch.cuda.is_available():
        ev = torch.cuda.Event()
        ev.record()
        return ev
    return None


def _signaled(event) -> bool:
    return event is None or event.query()


def _contiguous_runs(ops: list[tuple[int, int]], column: int):
    """Adjacent staging rows whose source/destination rows are also adjacent."""
    start = 0
    for end in range(1, len(ops) + 1):
        if end == len(ops) or ops[end][column] != ops[end - 1][column] + 1:
            yield start, end, ops[start][column]
            start = end


@dataclass
class _Job:
    batch: TierOpBatch
    # Only the final chunk acknowledges the scheduler's original batch.
    final: bool = True
    # Offload phase: bytes leave the device before the file write.
    offload_staging: torch.Tensor | None = None
    d2h_event: object | None = None
    d2h_ready: threading.Event = field(default_factory=threading.Event)
    # Restore phase: bytes come off the file before the device write.
    restore_staging: torch.Tensor | None = None
    restore_digests: list[str | None] | None = None
    io_done: bool = False
    applied: bool = False
    apply_event: object | None = None


class KVTierNVMe:
    """File-backed tier with the KVTierDMA surface: issue / fence_restores /
    poll_done / flush."""

    def __init__(
        self,
        backing: torch.Tensor,
        block_stride: int,
        num_slots: int,
        device: torch.device,
        tier_dir: str,
    ):
        assert backing.dtype == torch.int8
        assert backing.numel() % block_stride == 0
        self.blocks = backing.view(-1, block_stride)
        self.device = device
        self.block_stride = block_stride
        self.num_slots = num_slots
        if block_stride > _STAGING_BUDGET_BYTES:
            raise ValueError("nvme-tier: one KV block exceeds the staging budget")

        size = num_slots * block_stride
        os.makedirs(tier_dir, exist_ok=True)
        self._check_free_space(tier_dir, size)
        path = os.path.join(tier_dir, f"kv_tier_{os.getpid()}.bin")
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        # Unlinked while open: the space reclaims itself on any exit,
        # including a crash. Nothing in the tier outlives the process - the
        # trajectory index is in-memory and dies with the scheduler anyway.
        try:
            os.unlink(path)
            self._set_nocache()
            self._preallocate(size)
        except BaseException:
            os.close(self.fd)
            raise

        self._lock = threading.Lock()
        self._have_work = threading.Condition(self._lock)
        self._budget_freed = threading.Condition(self._lock)
        self._jobs: deque[_Job] = deque()  # FIFO in issue order
        self._staged_bytes = 0
        self._done_seqs: list[int] = []
        self._shutdown = False
        self._io_error: BaseException | None = None
        self._slot_digests: dict[int, str] = {}
        self._io_thread = threading.Thread(
            target=self._io_loop, name="kv-tier-nvme-io", daemon=True
        )
        self._io_thread.start()

    # -- file setup ----------------------------------------------------

    @staticmethod
    def _check_free_space(tier_dir: str, size: int) -> None:
        st = os.statvfs(tier_dir)
        free = st.f_bavail * st.f_frsize
        # Leave 32 GiB of slack for the OS and everything else on the volume.
        if free < size + (32 << 30):
            raise RuntimeError(
                f"nvme-tier: {tier_dir} has {free / (1 << 30):.0f} GiB free; "
                f"the tier file needs {size / (1 << 30):.0f} GiB plus 32 GiB "
                "slack. Shrink nvme_tier_gb_per_rank or free disk space."
            )

    def _set_nocache(self) -> None:
        import fcntl

        # A silent fallback on Metal would compete with the GPU working set.
        if sys.platform == "darwin":
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)

    def _preallocate(self, size: int) -> None:
        import fcntl
        import struct

        if sys.platform == "darwin":
            # fstore_t: fst_flags u32, fst_posmode i32, fst_offset i64,
            # fst_length i64, fst_bytesalloc i64. F_ALLOCATEALL=4,
            # F_PEOFPOSMODE=3.
            fstore = struct.pack("=IiqqQ", 4, 3, 0, size, 0)
            # Darwin sys/fcntl.h defines F_PREALLOCATE=42; CPython does not
            # expose this constant. Reserve real space instead of silently
            # creating a sparse file when the Python attribute is absent.
            fcntl.fcntl(self.fd, 42, fstore)
        elif hasattr(os, "posix_fallocate"):
            os.posix_fallocate(self.fd, 0, size)
        os.ftruncate(self.fd, size)

    # -- main-thread API (KVTierDMA surface) ---------------------------

    def issue(self, batch: TierOpBatch) -> None:
        if not batch.offload and not batch.restore and not batch.zero:
            return
        self._raise_io_error()
        if self._shutdown:
            raise RuntimeError("nvme-tier is shut down")
        # Reserve only one bounded chunk at a time. Reserving an entire
        # oversized batch before queuing it deadlocks: no queued work can
        # release that reservation. Separate offloads from restores so a
        # mixed batch cannot hold its own unqueued reservation either.
        rows = _STAGING_BUDGET_BYTES // self.block_stride
        chunks = []
        for start in range(0, len(batch.offload), rows):
            chunks.append(
                TierOpBatch(batch.seq, batch.offload[start : start + rows], [])
            )
        for start in range(0, len(batch.restore), rows):
            chunks.append(
                TierOpBatch(
                    batch.seq,
                    [],
                    batch.restore[start : start + rows],
                    batch.zero if start == 0 else [],
                )
            )
        if batch.zero and not batch.restore:
            chunks.append(TierOpBatch(batch.seq, [], [], batch.zero))
        for i, chunk in enumerate(chunks):
            self._issue_chunk(chunk, final=i == len(chunks) - 1)

    def _issue_chunk(self, batch: TierOpBatch, *, final: bool) -> None:
        job = _Job(batch=batch, final=final)
        if batch.offload:
            n = len(batch.offload)
            self._reserve_staging(n * self.block_stride)
            job.offload_staging = torch.empty((n, self.block_stride), dtype=torch.int8)
            for start, end, block in _contiguous_runs(batch.offload, 0):
                job.offload_staging[start:end].copy_(
                    self.blocks[block : block + end - start], non_blocking=True
                )
            job.d2h_event = _make_event()
            if job.d2h_event is None:
                job.d2h_ready.set()
        if batch.restore:
            self._reserve_staging(len(batch.restore) * self.block_stride)
        with self._lock:
            self._jobs.append(job)
            self._have_work.notify()
        self._pump()

    def fence_restores(self) -> None:
        # No fence needed: apply copies are enqueued on the single MPS
        # stream from this (main) thread before the step's forward, so all
        # subsequent compute is already ordered after them; an apply whose
        # event has not signaled belongs to a batch not yet reported done,
        # whose request the scheduler cannot have made schedulable.
        # This hook is still the step-top pump point.
        self._pump()

    def poll_done(self) -> list[int]:
        self._raise_io_error()
        self._pump()
        self._retire_ready()
        with self._lock:
            done, self._done_seqs = self._done_seqs, []
        return done

    def _retire_ready(self) -> None:
        """Pop completed head jobs into the done buffer. Internal callers
        (the staging-budget drain) retire to free staging without stealing
        seq ids from poll_done - the connector must see every completion."""
        with self._lock:
            while self._jobs and self._job_complete(self._jobs[0]):
                job = self._jobs.popleft()
                if job.final:
                    self._done_seqs.append(job.batch.seq)
                if _VERIFY:
                    self._verify(job)
                self._release_staging(job)

    def flush(self) -> list[int]:
        """Synchronously drain everything in flight (shutdown/tests)."""
        done: list[int] = []
        while True:
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            done.extend(self.poll_done())
            with self._lock:
                if not self._jobs:
                    return done
                # Release the GIL while disk IO progresses. Device events
                # still need periodic main-thread pumping.
                self._budget_freed.wait(timeout=0.001)
            self._raise_io_error()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        try:
            self.flush()
        finally:
            with self._lock:
                self._shutdown = True
                self._have_work.notify_all()
            # Never close/reuse the descriptor while the IO thread owns it.
            self._io_thread.join()
            os.close(self.fd)

    # -- main-thread internals -----------------------------------------

    def _pump(self) -> None:
        """Advance every phase that needs the main thread: flip D2H
        readiness for the IO thread, and apply H2D for io-complete
        restores (in issue order)."""
        with self._lock:
            jobs = list(self._jobs)
        for job in jobs:
            if not job.d2h_ready.is_set() and _signaled(job.d2h_event):
                job.d2h_ready.set()
        for job in jobs:
            if (
                job.io_done
                and not job.applied
                and (job.batch.restore or job.batch.zero)
            ):
                self._apply(job)

    def _apply(self, job: _Job) -> None:
        batch = job.batch
        for gpu_block in batch.zero:
            self.blocks[gpu_block].zero_()
        if job.restore_staging is not None:
            for start, end, block in _contiguous_runs(batch.restore, 1):
                # Pageable-source H2D on MPS stages synchronously on the
                # CPU; device-side population is stream-ordered
                # (buffer_utils.copy_h2d_sync_free rationale). The staging
                # row is reusable as soon as copy_ returns.
                self.blocks[block : block + end - start].copy_(
                    job.restore_staging[start:end], non_blocking=True
                )
        job.apply_event = _make_event()
        job.applied = True

    def _job_complete(self, job: _Job) -> bool:
        batch = job.batch
        if batch.restore or batch.zero:
            return job.applied and _signaled(job.apply_event)
        return job.io_done

    def _reserve_staging(self, nbytes: int) -> None:
        if nbytes > _STAGING_BUDGET_BYTES:
            raise ValueError("nvme-tier: staging chunk exceeds budget")
        while True:
            self._raise_io_error()
            with self._lock:
                if self._staged_bytes + nbytes <= _STAGING_BUDGET_BYTES:
                    self._staged_bytes += nbytes
                    return
            # Over budget: the IO thread may be waiting on a D2H flip only
            # this thread can provide, and staging is only freed when a
            # head job retires - drive both (retired seq ids stay buffered
            # for poll_done), then let the drain progress.
            self._pump()
            self._retire_ready()
            with self._lock:
                if self._staged_bytes + nbytes > _STAGING_BUDGET_BYTES:
                    self._budget_freed.wait(timeout=0.001)

    def _release_staging(self, job: _Job) -> None:
        # Called with self._lock held.
        freed = 0
        if job.offload_staging is not None:
            freed += job.offload_staging.numel()
            job.offload_staging = None
        if job.restore_staging is not None:
            freed += job.restore_staging.numel()
            job.restore_staging = None
        if freed:
            self._staged_bytes -= freed
            self._budget_freed.notify_all()

    def _verify(self, job: _Job) -> None:
        batch = job.batch
        bad = 0
        for i, (slot, gpu_block) in enumerate(batch.restore):
            # Snapshot at pread time: later FIFO writes may already have
            # reused the slot before the main thread verifies this restore.
            expect = job.restore_digests[i] if job.restore_digests else None
            got = _digest(self.blocks[gpu_block])
            if expect is not None and got != expect:
                bad += 1
                logger.warning(
                    "nvme-tier VERIFY MISMATCH slot=%d block=%d "
                    "offloaded=%s gpu_after_restore=%s",
                    slot,
                    gpu_block,
                    expect,
                    got,
                )
        if batch.restore:
            logger.info(
                "nvme-tier verify: batch %d: %d/%d restores mismatched",
                batch.seq,
                bad,
                len(batch.restore),
            )
        if bad:
            raise RuntimeError("nvme-tier: restored bytes failed verification")

    def _raise_io_error(self) -> None:
        if self._io_error is not None:
            raise RuntimeError(
                "nvme-tier IO thread failed; the tier is unusable"
            ) from self._io_error

    # -- IO thread ------------------------------------------------------

    def _io_loop(self) -> None:
        try:
            while True:
                with self._lock:
                    while not self._jobs_pending_io() and not self._shutdown:
                        self._have_work.wait(timeout=0.1)
                    if self._shutdown:
                        return
                    job = self._next_io_job()
                if job is None:
                    continue
                self._do_io(job)
        except BaseException as e:  # noqa: BLE001 - surfaced on main thread
            logger.exception("nvme-tier IO thread died")
            with self._lock:
                self._io_error = e
                self._budget_freed.notify_all()

    def _jobs_pending_io(self) -> bool:
        # Called with self._lock held.
        return any(not j.io_done for j in self._jobs)

    def _next_io_job(self) -> _Job | None:
        # Called with self._lock held. Strict FIFO: the first job whose IO
        # has not run yet. Jobs ahead of it are all io_done (their apply
        # phases belong to the main thread), so ordering holds.
        for j in self._jobs:
            if not j.io_done:
                return j
        return None

    def _do_io(self, job: _Job) -> None:
        batch = job.batch
        if batch.offload:
            # Wait for the main thread to observe the D2H event. Events on
            # the single stream signal in issue order, so the head job's
            # event is always the next to fire.
            while not job.d2h_ready.wait(timeout=0.1):
                with self._lock:
                    if self._shutdown or self._io_error is not None:
                        return
            staging = job.offload_staging
            assert staging is not None
            arr = staging.numpy()
            for start, end, slot in _contiguous_runs(batch.offload, 1):
                view = memoryview(arr[start:end]).cast("B")
                off = slot * self.block_stride
                written = 0
                while written < len(view):
                    count = os.pwrite(self.fd, view[written:], off + written)
                    if count <= 0:
                        raise OSError("nvme-tier: write made no progress")
                    written += count
            if _VERIFY:
                for i, (gpu_block, slot) in enumerate(batch.offload):
                    digest = _digest(staging[i])
                    self._slot_digests[slot] = digest
                    logger.debug(
                        "nvme-tier snapshot: seq=%d block=%d slot=%d digest=%s",
                        batch.seq,
                        gpu_block,
                        slot,
                        digest,
                    )
        if batch.restore:
            staging = torch.empty(
                (len(batch.restore), self.block_stride), dtype=torch.int8
            )
            arr = staging.numpy()
            for start, end, slot in _contiguous_runs(batch.restore, 0):
                view = memoryview(arr[start:end]).cast("B")
                off = slot * self.block_stride
                read = 0
                while read < len(view):
                    count = os.preadv(self.fd, [view[read:]], off + read)
                    if count <= 0:
                        raise OSError("nvme-tier: unexpected end of tier file")
                    read += count
            job.restore_staging = staging
            if _VERIFY:
                job.restore_digests = [
                    self._slot_digests.get(slot) for slot, _ in batch.restore
                ]
        with self._lock:
            job.io_done = True
            self._budget_freed.notify_all()
