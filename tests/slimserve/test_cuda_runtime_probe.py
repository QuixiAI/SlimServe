# SPDX-License-Identifier: Apache-2.0
import ctypes
from types import SimpleNamespace

import pytest

from benchmarks.probe_cuda_ipc_nccl import (
    Handle,
    Runtime,
    expected_bytes,
    loaded_libraries,
)


def test_cuda_ipc_handle_and_each_peer_owns_a_distinct_stripe():
    assert ctypes.sizeof(Handle) == 64
    assert expected_bytes(1, 4, 2) == bytes([1, 1, 0, 0, 3, 3, 4, 4])


def test_loaded_libraries_records_real_paths_and_deduplicates(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.probe_cuda_ipc_nccl.Path.read_text",
        lambda _: "0 r-x /opt/compat/libcuda.so.610\n"
        "1 r-x /opt/compat/libcuda.so.610\n"
        "2 r-x /opt/lib/libcudart.so.13\n"
        "3 r-x /opt/lib/libnccl.so.2\n"
        "4 r-x /opt/lib/libc.so.6\n",
    )
    assert loaded_libraries() == [
        "/opt/compat/libcuda.so.610",
        "/opt/lib/libcudart.so.13",
        "/opt/lib/libnccl.so.2",
    ]


def test_runtime_error_identifies_the_first_failing_cuda_operation():
    runtime = Runtime.__new__(Runtime)
    runtime.lib = SimpleNamespace(
        cudaIpcOpenMemHandle=lambda *_: 217,
        cudaGetErrorString=lambda _: b"peer access is not supported",
    )
    with pytest.raises(RuntimeError, match="cudaIpcOpenMemHandle: CUDA error 217"):
        runtime.call("cudaIpcOpenMemHandle")
