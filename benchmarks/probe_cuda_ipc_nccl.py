#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded, model-free CUDA IPC/P2P and NCCL qualification for a runtime.

Run with torchrun --standalone --nproc-per-node=4 on otherwise idle GPUs.
Use an external memory/time limit as well. This checks actual remote writes,
not just peer-access capability bits, and records loaded CUDA library paths.
It does NOT qualify B12X custom collectives or measure serving performance.
"""

import argparse
import ctypes as ct
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist


class Handle(ct.Structure):
    # CUDA 13 driver_types.h: CUDA_IPC_HANDLE_SIZE, not HIP's handle size.
    _fields_ = [("reserved", ct.c_byte * 64)]


def loaded_libraries():
    return sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/" in line
            and any(name in line for name in ("libcuda.", "libcudart.", "libnccl."))
        }
    )


class Runtime:
    def __init__(self):
        paths = [p for p in loaded_libraries() if "libcudart." in p]
        if len(paths) != 1:
            raise RuntimeError(f"expected one loaded libcudart, got {paths}")
        self.lib = ct.CDLL(paths[0])
        signatures = {
            "cudaMalloc": [ct.POINTER(ct.c_void_p), ct.c_size_t],
            "cudaFree": [ct.c_void_p],
            "cudaMemset": [ct.c_void_p, ct.c_int, ct.c_size_t],
            "cudaMemcpy": [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int],
            "cudaDeviceSynchronize": [],
            "cudaIpcGetMemHandle": [ct.POINTER(Handle), ct.c_void_p],
            "cudaIpcOpenMemHandle": [ct.POINTER(ct.c_void_p), Handle, ct.c_uint],
            "cudaIpcCloseMemHandle": [ct.c_void_p],
            "cudaDriverGetVersion": [ct.POINTER(ct.c_int)],
            "cudaRuntimeGetVersion": [ct.POINTER(ct.c_int)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ct.c_int
        self.lib.cudaGetErrorString.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.restype = ct.c_char_p

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            message = self.lib.cudaGetErrorString(result).decode()
            raise RuntimeError(f"{name}: CUDA error {result}: {message}")


def expected_bytes(rank, world, stripe):
    return b"".join(
        bytes([0 if peer == rank else peer + 1]) * stripe for peer in range(world)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = int(os.environ["LOCAL_RANK"])
    timeout = datetime.timedelta(seconds=args.timeout_seconds)
    dist.init_process_group("gloo", timeout=timeout)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    report = {
        "status": "running",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rank": rank,
        "device": device,
        "world": world,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NCCL_", "CUDA_")) or k == "LD_LIBRARY_PATH"
        },
        "phases": [],
    }
    output = args.output / f"rank-{rank}.json"

    def phase(name, operation):
        error, value = None, None
        try:
            value = operation()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        peers = [None] * world
        dist.all_gather_object(peers, error)
        if any(peers):
            report["status"] = "failed"
        report["phases"].append({"name": name, "errors": peers, "value": value})
        report["libraries"] = loaded_libraries()
        output.write_text(json.dumps(report, indent=2) + "\n")
        if any(peers):
            # No premature free of exports still mapped by peers on failure.
            # The external launcher must terminate the entire owned process set.
            raise RuntimeError(f"{name} failed: {peers}")
        return value

    def require_idle():
        if rank == 0:
            active = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            ).strip()
            if active:
                raise RuntimeError(f"GPUs already have compute processes: {active}")

    phase("idle_guard", require_idle)
    phase("cuda_context", lambda: torch.cuda.set_device(device))
    runtime = None

    def load_runtime():
        nonlocal runtime
        runtime = Runtime()

    phase("load_runtime", load_runtime)
    stripe, local, handle, imports = 1024, ct.c_void_p(), Handle(), []

    def export():
        runtime.call("cudaMalloc", ct.byref(local), stripe * world)
        runtime.call("cudaMemset", local, 0, stripe * world)
        runtime.call("cudaDeviceSynchronize")
        runtime.call("cudaIpcGetMemHandle", ct.byref(handle), local)
        versions = []
        for name in ("cudaDriverGetVersion", "cudaRuntimeGetVersion"):
            value = ct.c_int()
            runtime.call(name, ct.byref(value))
            versions.append(value.value)
        return {"driver_version": versions[0], "runtime_version": versions[1]}

    phase("ipc_export", export)
    handles = [None] * world
    dist.all_gather_object(handles, bytes(handle))

    def open_imports():
        for peer, encoded in enumerate(handles):
            if peer == rank:
                continue
            pointer = ct.c_void_p()
            runtime.call(
                "cudaIpcOpenMemHandle",
                ct.byref(pointer),
                Handle.from_buffer_copy(encoded),
                1,
            )
            imports.append(pointer)

    phase("ipc_import", open_imports)

    def remote_write():
        for pointer in imports:
            runtime.call(
                "cudaMemset",
                ct.c_void_p(pointer.value + rank * stripe),
                rank + 1,
                stripe,
            )
        runtime.call("cudaDeviceSynchronize")

    phase("remote_write", remote_write)

    def verify():
        host = (ct.c_byte * (stripe * world))()
        runtime.call("cudaMemcpy", host, local, len(host), 2)  # device -> host
        if bytes(host) != expected_bytes(rank, world, stripe):
            raise RuntimeError("remote-write bytes differ from every-peer pattern")
        return {"peers_verified": world - 1, "bytes_verified": len(host)}

    phase("remote_verify", verify)
    phase(
        "close_imports",
        lambda: [runtime.call("cudaIpcCloseMemHandle", p) for p in imports],
    )
    phase("free_export", lambda: runtime.call("cudaFree", local))
    group = dist.new_group(backend="nccl", timeout=timeout)
    for count in (1, 4096, 31195136):

        def reduce(count=count):
            tensor = torch.full((count,), rank + 1, device=device, dtype=torch.bfloat16)
            dist.all_reduce(tensor, group=group)
            torch.cuda.synchronize()
            expected = world * (world + 1) // 2
            if not bool(tensor.eq(expected).all()):
                raise RuntimeError(f"all-reduce differs from {expected}")
            return {
                "elements": count,
                "bytes": count * 2,
                "nccl_version": torch.cuda.nccl.version(),
            }

        phase(f"nccl_allreduce_{count}", reduce)
    dist.destroy_process_group(group)
    report["status"] = "complete"
    output.write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
