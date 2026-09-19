# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.v1.worker import gpu_worker
from vllm.v1.worker.xpu_worker import XPUWorker


def test_hsa_staging_probe_is_rocm_only():
    with patch.object(gpu_worker.current_platform, "is_rocm", return_value=True):
        assert gpu_worker._should_warm_hsa_host_staging()
    with patch.object(gpu_worker.current_platform, "is_rocm", return_value=False):
        assert not gpu_worker._should_warm_hsa_host_staging()


def test_xpu_prepare_shutdown_drains_before_idempotent_cleanup():
    worker = object.__new__(XPUWorker)
    worker.rank = 2
    worker.local_rank = 2
    worker.device = "xpu:0"
    calls = []

    stream = MagicMock()
    stream.synchronize.side_effect = lambda: calls.append("stream")
    allocator = MagicMock()
    allocator.release_pools.side_effect = lambda: calls.append("pools")

    with (
        patch("vllm.v1.worker.gpu_worker.Worker.shutdown") as base_shutdown,
        patch("torch.xpu.current_stream", return_value=stream),
        patch("vllm.device_allocator.xpumem.XpuMemAllocator.instance", allocator),
    ):
        base_shutdown.side_effect = lambda: calls.append("base")
        worker.prepare_shutdown()
        worker.prepare_shutdown()
        worker.shutdown()
        worker.shutdown()

    assert calls == ["stream", "base", "pools"]


def test_xpu_shutdown_does_not_drain_rank_independently():
    worker = object.__new__(XPUWorker)
    worker.rank = 2
    worker.local_rank = 2
    calls = []
    allocator = MagicMock()
    allocator.release_pools.side_effect = lambda: calls.append("pools")

    with (
        patch("vllm.v1.worker.gpu_worker.Worker.shutdown") as base_shutdown,
        patch("torch.xpu.current_stream") as current_stream,
        patch("vllm.device_allocator.xpumem.XpuMemAllocator.instance", allocator),
    ):
        base_shutdown.side_effect = lambda: calls.append("base")
        worker.shutdown()

    current_stream.assert_not_called()
    assert calls == ["base", "pools"]
