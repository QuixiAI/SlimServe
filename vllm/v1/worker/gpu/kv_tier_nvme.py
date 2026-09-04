# SPDX-License-Identifier: Apache-2.0
"""NVMe tier IO engine: pinned host arena rows <-> slots in a per-rank file.

Third KV tier below GPU and pinned host RAM. The engine owns one
preallocated file per rank on the tier device and moves whole packed
blocks between host arena rows and file slots with O_DIRECT positional IO
from a small thread pool. It never touches the GPU: the DMA engine
(kv_tier_dma) orders disk writes after the GPU -> host copy that produced
the row, and defers a restore's host -> GPU copies until the disk -> host
reads that fill its rows have completed.

Alignment: O_DIRECT needs page-aligned buffers, lengths and offsets. The
arena rows are 4 KiB-aligned and padded (kv_tier_dma._register_host_arena),
and a file slot is exactly one padded row, so every op is one contiguous
aligned transfer.

File lifetime: the file is an unnamed O_TMPFILE in the tier directory, so
it vanishes with the process (crash included) and two servers on one box
never collide. Named fallback (with immediate unlink) where O_TMPFILE is
unsupported; buffered fallback where O_DIRECT is refused (tmpfs).
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

PAGE = 4096


@dataclass
class DiskOp:
    op_id: int
    write: bool  # True: host row -> disk slot; False: disk slot -> host row
    disk_slot: int
    buffer: memoryview  # the arena row (padded, aligned)
    # Called on the IO thread before the transfer (e.g. wait on the CUDA
    # event of the copy that produced the row).
    wait: Callable[[], None] | None = None
    # VERIFY: live bytes of the row (a group's block may be shorter than
    # the slab row); the write-time digest of this prefix is what a later
    # restore of the promoted row is checked against.
    nbytes: int = 0


_VERIFY_ROWS = os.environ.get("VLLM_KV_TIER_VERIFY", "0") == "1"


def _row_digest(buf) -> str:
    import hashlib

    return hashlib.sha256(bytes(buf)).hexdigest()[:12]


class NvmeTierFile:
    def __init__(
        self,
        directory: str,
        num_slots: int,
        slot_bytes: int,
        threads: int = 4,
    ):
        assert num_slots > 0 and slot_bytes % PAGE == 0
        self.num_slots = num_slots
        self.slot_bytes = slot_bytes
        os.makedirs(directory, exist_ok=True)
        self.direct = True
        self.fd = self._open(directory)
        os.posix_fallocate(self.fd, 0, num_slots * slot_bytes)
        self._queue: deque[DiskOp | None] = deque()
        self._cv = threading.Condition()
        self._done: deque[tuple[int, str | None]] = deque()
        self._done_lock = threading.Lock()
        self._stop = False
        self.row_digests: dict[int, str] = {}
        self.slot_digests: dict[int, str] = {}
        self._row_lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._worker, name=f"kv-nvme-{i}", daemon=True)
            for i in range(max(1, threads))
        ]
        for t in self._threads:
            t.start()
        logger.info(
            "kv-nvme: %d slots x %d bytes (%.1f GiB) in %s, %s, %d IO threads",
            num_slots,
            slot_bytes,
            num_slots * slot_bytes / (1 << 30),
            directory,
            "O_DIRECT" if self.direct else "buffered",
            len(self._threads),
        )

    def _open(self, directory: str) -> int:
        direct = getattr(os, "O_DIRECT", 0)
        tmpfile = getattr(os, "O_TMPFILE", 0)
        attempts = []
        if tmpfile:
            attempts.append(("tmpfile+direct", os.O_RDWR | tmpfile | direct, True))
            attempts.append(("tmpfile", os.O_RDWR | tmpfile, False))
        for name, flags, is_direct in attempts:
            try:
                fd = os.open(directory, flags, 0o600)
                self.direct = is_direct
                return fd
            except OSError as exc:
                logger.debug("kv-nvme: open %s failed: %s", name, exc)
        path = os.path.join(directory, f"kv-tier-{os.getpid()}-{id(self)}.arena")
        for is_direct in (True, False):
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | (direct if is_direct else 0)
            try:
                fd = os.open(path, flags, 0o600)
            except OSError as exc:
                logger.debug("kv-nvme: open named (direct=%s) failed: %s", is_direct, exc)
                continue
            os.unlink(path)
            self.direct = is_direct
            return fd
        raise OSError(f"kv-nvme: cannot create the tier file in {directory}")

    # ------------------------------------------------------------ submit/poll

    def submit(self, op: DiskOp) -> None:
        assert 0 <= op.disk_slot < self.num_slots
        assert len(op.buffer) == self.slot_bytes, (len(op.buffer), self.slot_bytes)
        with self._cv:
            self._queue.append(op)
            self._cv.notify()

    def poll_done(self) -> list[tuple[int, str | None]]:
        """Completed (op_id, error) pairs since the last poll."""
        with self._done_lock:
            done = list(self._done)
            self._done.clear()
        return done

    def pending(self) -> int:
        with self._cv:
            return len(self._queue)

    # ------------------------------------------------------------- IO thread

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop and not self._queue:
                    return
                op = self._queue.popleft()
            err: str | None = None
            try:
                if op.wait is not None:
                    op.wait()
                if _VERIFY_ROWS and op.write:
                    # Fidelity check (VLLM_KV_TIER_VERIFY): hash the row AS
                    # WRITTEN, i.e. after the producing copy's event - a
                    # digest taken at submission can predate the copy. A
                    # later read of the slot must hash the same (row), and
                    # the restore verify checks the live prefix (slot).
                    with self._row_lock:
                        self.row_digests[op.disk_slot] = _row_digest(op.buffer)
                        if op.nbytes > 0:
                            self.slot_digests[op.disk_slot] = _row_digest(
                                op.buffer[: op.nbytes]
                            )
                self._transfer(op)
                if _VERIFY_ROWS and not op.write:
                    got = _row_digest(op.buffer)
                    with self._row_lock:
                        exp = self.row_digests.get(op.disk_slot)
                    if exp is not None and got != exp:
                        logger.warning(
                            "kv-nvme DISK ROUND-TRIP MISMATCH disk_slot=%d "
                            "written=%s read=%s", op.disk_slot, exp, got,
                        )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                err = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "kv-nvme: %s slot %d failed: %s",
                    "write" if op.write else "read",
                    op.disk_slot,
                    err,
                )
            with self._done_lock:
                self._done.append((op.op_id, err))

    def _transfer(self, op: DiskOp) -> None:
        offset = op.disk_slot * self.slot_bytes
        buf = op.buffer
        n = self.slot_bytes
        pos = 0
        while pos < n:
            if op.write:
                k = os.pwritev(self.fd, [buf[pos:]], offset + pos)
            else:
                k = os.preadv(self.fd, [buf[pos:]], offset + pos)
            if k <= 0:
                raise OSError(f"short {'write' if op.write else 'read'} at {pos}/{n}")
            pos += k

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=30)
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def default_tier_dir() -> str:
    """Operator-configured tier directory (never a path in the repo)."""
    d = os.environ.get("SLIMSERVE_KV_TIER_DIR")
    if d:
        return d
    cache = os.environ.get("SLIMSERVE_CACHE")
    base = cache if cache else os.path.expanduser("~/.cache/slimserve")
    return os.path.join(base, "kv-tier")
