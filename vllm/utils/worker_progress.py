# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, payload-free host phase recorder; this never queries a device.

Enable before worker startup with VLLM_WORKER_PROGRESS_DIR. Read externally:
  python vllm/utils/worker_progress.py --directory DIR --pids PID [PID ...]
Records describe host entry/exit, not GPU completion. Operation counters are
local to each thread; async output counters are not main-thread RPC IDs.
"""

import argparse
import json
import logging
import mmap
import os
import struct
import threading
import time
from contextlib import nullcontext
from enum import IntEnum
from pathlib import Path


class Phase(IntEnum):
    RPC_EXECUTE = 1
    RPC_SAMPLE = 2
    RPC_DRAFT_IDS = 3
    RPC_OTHER = 4
    SAMPLE = 5
    PRIOR_OUTPUT = 6
    PRIOR_DRAFT = 7
    MTP = 8
    DRAFT_COPY = 9
    ASYNC_OUTPUT = 10


_MAGIC = b"SVPROG01"
_HEADER = struct.Struct("<8sIIQQ")  # magic, version, rank, pid, monotonic creation
_U64 = struct.Struct("<Q")
_RECORD = struct.Struct("<QQQIIQ")  # seq, time, operation, phase, edge, native tid
_HEADER_SIZE, _SLOT_SIZE, _SLOTS, _CAPACITY = 64, 8192, 2, 128
_FILE_SIZE = _HEADER_SIZE + _SLOTS * _SLOT_SIZE
_ROOT_PHASES = {
    Phase.RPC_EXECUTE,
    Phase.RPC_SAMPLE,
    Phase.RPC_DRAFT_IDS,
    Phase.RPC_OTHER,
    Phase.ASYNC_OUTPUT,
}
_NOOP = nullcontext()
_recorder = None


class Recorder:
    def __init__(self, directory: str, rank: int):
        self.enabled = True
        self._local = threading.local()
        self._threads = {}
        self._register_lock = threading.Lock()
        Path(directory).mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = Path(directory) / f"worker-{os.getpid()}.bin"
        fd = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            os.ftruncate(fd, _FILE_SIZE)
            self._map = mmap.mmap(fd, _FILE_SIZE)
        finally:
            os.close(fd)
        _HEADER.pack_into(
            self._map, 0, _MAGIC, 1, rank, os.getpid(), time.monotonic_ns()
        )

    def close(self):
        self.enabled = False
        self._map.close()

    def write(self, phase: Phase, edge: int):
        if not self.enabled:
            return
        try:
            state = getattr(self._local, "state", None)
            if state is False:
                return
            if state is None:
                tid = threading.get_native_id()
                # Registration locks once per thread, never per event.
                with self._register_lock:
                    if len(self._threads) >= _SLOTS:
                        self._local.state = False
                        return
                    slot = self._threads[tid] = len(self._threads)
                state = self._local.state = [slot, 0, 0, tid]
            if edge == 0 and phase in _ROOT_PHASES:
                state[2] += 1
            state[1] += 1
            seq = state[1]
            base = _HEADER_SIZE + state[0] * _SLOT_SIZE
            offset = base + 64 + ((seq - 1) % _CAPACITY) * _RECORD.size
            _RECORD.pack_into(
                self._map,
                offset,
                seq,
                time.monotonic_ns(),
                state[2],
                int(phase),
                edge,
                state[3],
            )
            # Publish only after the fixed record is complete. Readers check
            # publication before/after copying and validate record sequences.
            _U64.pack_into(self._map, base, seq)
        except Exception:
            # Diagnostics must never replace a serving failure or change its
            # exception behavior. Do not stringify arbitrary exception data.
            self.enabled = False


class _Scope:
    def __init__(self, recorder: Recorder, phase: Phase):
        self.recorder, self.phase = recorder, phase

    def __enter__(self):
        self.recorder.write(self.phase, 0)

    def __exit__(self, exc_type, exc_value, tb):
        self.recorder.write(self.phase, 1 if exc_type is None else 2)
        return False


def initialize_worker_progress(rank: int):
    global _recorder
    if _recorder is not None:
        return
    directory = os.environ.get("VLLM_WORKER_PROGRESS_DIR")
    if not directory:
        return
    try:
        _recorder = Recorder(directory, rank)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Worker progress recorder unavailable: %s", type(exc).__name__
        )


def phase_scope(phase: Phase):
    if _recorder is None or not _recorder.enabled:
        return _NOOP
    if not isinstance(phase, Phase):
        raise TypeError("Worker progress phases must be fixed Phase values")
    return _Scope(_recorder, phase)


def rpc_scope(method: str | bytes):
    if _recorder is None or not _recorder.enabled:
        return _NOOP
    # Do not store arbitrary RPC names, serialized callables, or arguments.
    phase = {
        "execute_model": Phase.RPC_EXECUTE,
        "sample_tokens": Phase.RPC_SAMPLE,
        "take_draft_token_ids": Phase.RPC_DRAFT_IDS,
    }.get(method, Phase.RPC_OTHER)
    return phase_scope(phase)


def read_snapshot(path: str | Path) -> dict:
    with open(path, "rb") as source:
        if os.fstat(source.fileno()).st_size != _FILE_SIZE:
            raise ValueError("Invalid worker progress file size")
        with mmap.mmap(source.fileno(), _FILE_SIZE, access=mmap.ACCESS_READ) as view:
            magic, version, rank, pid, created = _HEADER.unpack_from(view)
            if magic != _MAGIC or version != 1:
                raise ValueError("Unknown worker progress format")
            threads = []
            for slot in range(_SLOTS):
                base = _HEADER_SIZE + slot * _SLOT_SIZE
                for _ in range(5):
                    before = _U64.unpack_from(view, base)[0]
                    data = view[base : base + _SLOT_SIZE]
                    if before == _U64.unpack_from(view, base)[0]:
                        break
                else:
                    threads.append({"slot": slot, "unstable": True})
                    continue
                if before == 0:
                    continue
                records, active = [], []
                for seq in range(max(1, before - _CAPACITY + 1), before + 1):
                    fields = _RECORD.unpack_from(
                        data, 64 + ((seq - 1) % _CAPACITY) * _RECORD.size
                    )
                    actual, when, operation, phase, edge, tid = fields
                    if actual != seq:
                        continue
                    label = Phase(phase).name
                    records.append(
                        {
                            "sequence": seq,
                            "monotonic_ns": when,
                            "operation": operation,
                            "phase": label,
                            "edge": ("begin", "end", "error")[edge],
                            "native_tid": tid,
                        }
                    )
                    if edge == 0:
                        active.append(label)
                    elif active and active[-1] == label:
                        active.pop()
                threads.append(
                    {
                        "slot": slot,
                        "published_sequence": before,
                        "active_phases": active,
                        "history_truncated": before > _CAPACITY,
                        "records": records,
                    }
                )
            return {
                "version": version,
                "pid": pid,
                "rank": rank,
                "created_monotonic_ns": created,
                "threads": threads,
            }


def snapshot_workers(directory: str, pids: list[int]) -> list[dict]:
    snapshots = []
    for pid in pids:
        try:
            result = read_snapshot(Path(directory) / f"worker-{int(pid)}.bin")
            if result["pid"] != pid:
                raise ValueError("Worker PID does not match file")
            snapshots.append(result)
        except (OSError, ValueError, struct.error, IndexError) as exc:
            snapshots.append({"pid": pid, "error": type(exc).__name__})
    return snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot_workers(args.directory, args.pids), indent=2))
