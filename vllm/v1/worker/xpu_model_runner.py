# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V1 GPU model runners on Intel XPU.

The shared runners reach for ``torch.cuda`` streams/events/graphs by name.
Rather than fork them, alias those names to their ``torch.xpu`` counterparts
for the duration of construction. Two rules from the sibling XPU tree:

- every alias is its own ``functools.partial``: aliasing ``torch.cuda.X =
  torch.xpu.X`` directly makes Dynamo's ``_get_handlers()`` assert on a
  duplicate registration the first time it compiles anything;
- ``torch.xpu.Event`` has no ``blocking=`` kwarg.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        self.cascade_attn_enabled = False

    def _init_device_properties(self) -> None:
        # vLLM heuristics want an SM count; Xe cores are the honest analogue.
        self.num_sms = torch.xpu.get_device_properties(self.device).max_compute_units

    def _sync_device(self) -> None:
        # Current-stream wait, never torch.xpu.synchronize (device-wide queue
        # walk; see vllm/platforms/xpu.py).
        torch.xpu.current_stream(self.device).synchronize()


class XPUModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _xpu_graph(graph, pool=None, stream=None):
    # torch.xpu.graph synchronizes every device queue on entry; that queue
    # walk crashes once Triton has created transient queues. Record directly
    # on the current in-order stream instead.
    capture_stream = stream or torch.xpu.current_stream()
    previous_stream = torch.xpu.current_stream()
    torch.xpu.set_stream(capture_stream)
    try:
        graph.capture_begin(pool=pool)
        try:
            yield
        finally:
            graph.capture_end()
    finally:
        torch.xpu.set_stream(previous_stream)


def _xpu_event(*args, blocking=None, **kwargs):
    return torch.xpu.Event(*args, **kwargs)


def _xpu_empty_cache_noop():
    # A device-wide cache purge walks transient SYCL queues (UR fault) and,
    # under graph capture, would drop pool allocations; explicit callers that
    # need it use torch.xpu.empty_cache directly.
    return None


def _xpu_synchronize(device=None):
    torch.xpu.current_stream(device).synchronize()


_ALIASES_INSTALLED = False


def install_torch_cuda_aliases() -> None:
    """Point the torch.cuda names the shared GPU path uses at torch.xpu."""
    global _ALIASES_INSTALLED
    if _ALIASES_INSTALLED:
        return
    torch.cuda.Stream = torch.xpu.Stream  # type: ignore[misc]
    torch.cuda.default_stream = partial(torch.xpu.current_stream)  # type: ignore[assignment]
    torch.cuda.current_stream = partial(torch.xpu.current_stream)  # type: ignore[assignment]
    torch.cuda.stream = partial(torch.xpu.stream)  # type: ignore[assignment]
    torch.cuda.set_stream = partial(torch.xpu.set_stream)  # type: ignore[assignment]
    torch.cuda.Event = _xpu_event  # type: ignore[misc]
    # Current-stream wait, not torch.xpu.synchronize: the device-wide sync
    # walks transient SYCL queues and faults in urQueueRelease.
    torch.cuda.synchronize = _xpu_synchronize  # type: ignore[assignment]
    torch.cuda.memory_allocated = partial(torch.xpu.memory_allocated)  # type: ignore[assignment]
    torch.cuda.memory_reserved = partial(torch.xpu.memory_reserved)  # type: ignore[assignment]
    torch.cuda.max_memory_allocated = partial(torch.xpu.max_memory_allocated)  # type: ignore[assignment]
    torch.cuda.get_device_properties = partial(torch.xpu.get_device_properties)  # type: ignore[assignment]
    torch.cuda.OutOfMemoryError = torch.OutOfMemoryError  # type: ignore[misc]
    torch.cuda.is_current_stream_capturing = partial(  # type: ignore[assignment]
        torch.xpu.is_current_stream_capturing
    )
    torch.cuda.is_available = partial(torch.xpu.is_available)  # type: ignore[assignment]
    torch.cuda.device_count = partial(torch.xpu.device_count)  # type: ignore[assignment]
    torch.cuda.current_device = partial(torch.xpu.current_device)  # type: ignore[assignment]
    torch.cuda.empty_cache = _xpu_empty_cache_noop  # type: ignore[assignment]
    # Device-wide cache purges walk transient SYCL queues (UR fault) and would
    # drop graph-pool allocations between captures; the shared runners call
    # torch.accelerator.empty_cache() at several capture/profile points.
    torch.accelerator.empty_cache = _xpu_empty_cache_noop  # type: ignore[assignment]
    if supports_xpu_graph():
        import vllm._quixicore_C as qc

        torch.cuda.graph = partial(_xpu_graph)  # type: ignore[assignment]
        # The QuixiCore-XPU binding's graph class: at::xpu::XPUGraph with a
        # capture-stream-scoped synchronize() (needed by segment replay);
        # capture is driven by _xpu_graph on the current stream.
        torch.cuda.CUDAGraph = qc.XPUGraph  # type: ignore[misc]
        torch.cuda.graph_pool_handle = partial(torch.xpu.graph_pool_handle)  # type: ignore[assignment]
    _ALIASES_INSTALLED = True


@contextmanager
def _torch_cuda_wrapper():
    install_torch_cuda_aliases()
    yield
