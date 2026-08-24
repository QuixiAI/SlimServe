# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What has to be replaced to run the V1 GPU path on PyTorch-MPS.

Four unrelated problems, all of which surface as something other than a clean
failure, which is why they are collected and documented here rather than
patched at their call sites:

1. vLLM ships several hot batch-prep helpers as Triton kernels. Triton does not
   target Metal and is not installed on macOS, so those kernels fall back to
   plain Python functions that raise when launched with ``kernel[grid](...)``.
2. There is no usable ``torch.compile`` backend: no MPS inductor backend, and
   the CPU one needs a C++ toolchain we do not require at runtime.
3. A non-blocking host-to-device copy is not ordered against dependent MPS
   work. This one is the dangerous one -- see below.
4. Shared CUDA/ROCm serving code gates multi-stream and graph-registration
   work on ``torch.cuda.is_current_stream_capturing()``. On an MPS-only build
   that is a dummy stub that raises when called. Metal has no graph capture,
   so the truthful answer is a constant ``False``.

The async-output completion-event wedge (a boot's first multi-chunk prefill
parking the engine forever in MPSEvent::synchronize -- GPU idle, signal
never delivered) is addressed structurally in
``gpu/async_utils.py``: torch.Stream(mps) always returns the single
stream 0, so the CUDA cross-stream choreography (set_stream, wait_stream,
generic torch.Event record) was pure risk with no overlap to buy. On Metal
the copy paths (AsyncOutput, AsyncPoolingOutput, DraftTokensHandler)
enqueue on the producing stream via make_output_copy_stream and record a
native torch.mps.Event there via make_completion_event;
VLLM_QC_ASYNC_OUT_DRAIN=1 swaps that event for a full
torch.mps.synchronize() drain as an ops fallback. History and validation:
perf/optimization_status.md (2026-08-14 boot_v12 bisect, cleanup-phase
bisect, 2026-08-17 fix entry -- five M1 Ultra boots, both wedge trigger
protocols clean, all anchors bit-exact). The earlier naive attempt --
swapping only the host-side wait for a drain while leaving wait_stream's
GPU-side wait encoded -- moved the park and produced a
command-buffer-timeout variant; the structural fix removes both.

The race was timing-sensitive on the host side: stripping the phaseprof
brackets and marshalling-memo conditionals from
vllm/models/deepseek_v4/{compressor,metal}.py made even RAMPED multi-chunk
requests park at completion (boot-level bisect, cleanup-phase entry).
Those code paths keep their structure until a soak on the fixed event path
proves the retention unnecessary; the boot-ramp ops protocol likewise
stays recommended until then.

Applied once, from the platform's check_and_update_config.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_APPLIED = False


def _metal_compute_slot_mapping(self, num_reqs, query_start_loc, positions) -> None:
    """Torch replacement for BlockTable's Triton slot-mapping kernel.

    slot = block_table[req, pos // block_size] * block_size + pos % block_size
    """
    if self.pcp_world_size * self.dcp_world_size != 1:
        raise NotImplementedError(
            "context/decode parallelism needs more than one device; "
            "Apple exposes one GPU per machine"
        )

    slot_mapping = self.slot_mapping.gpu
    device = slot_mapping.device
    block_size = self.block_size
    num_tokens = positions.shape[0]

    # Per-token request index, from the query_start_loc offsets.
    starts = query_start_loc[: num_reqs + 1].to("cpu", dtype=torch.long)
    counts = (starts[1:] - starts[:-1]).clamp_min(0)
    req_idx = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.long), counts
    ).to(device)

    pos = positions[:num_tokens].to(device=device, dtype=torch.long)
    n = req_idx.shape[0]
    pos = pos[:n]

    block_ids = self.block_table.gpu[req_idx, pos // block_size].to(torch.long)
    slots = block_ids * block_size + (pos % block_size)

    slot_mapping[:n] = slots.to(slot_mapping.dtype)
    if slot_mapping.shape[0] > n:
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        slot_mapping[n:] = PAD_SLOT_ID


def _patch_cpu_gpu_buffer_blocking() -> None:
    """Force CpuGpuBuffer's host/device copies to block.

    vLLM issues ``copy_(..., non_blocking=True)`` for its input-prep buffers.
    On CUDA the following same-stream kernels wait on that copy. On MPS a
    non-blocking copy out of non-pinned host memory is *not* ordered against
    the dependent MPS graph, so a gather such as ``positions[req_indices]`` can
    read a stale buffer and index out of bounds -- a wrong-answer or crash bug
    that appears only under load. Pinned memory is unavailable here anyway, so
    making these copies synchronous costs nothing real.

    (This does not contradict the deliberate ``non_blocking=True`` copies in
    ``vllm/v1/worker/gpu/buffer_utils.py``: those are safe because torch MPS
    stages pageable sources synchronously on the CPU either way -- the flag
    only skips the stream drain -- while the hazard here is device-side reads
    of a buffer the staging has not populated yet.)
    """
    from vllm.v1.utils import CpuGpuBuffer

    def copy_to_gpu(self, n=None):
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=False)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=False)

    def copy_to_cpu(self, n=None):
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=False)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=False)

    CpuGpuBuffer.copy_to_gpu = copy_to_gpu  # type: ignore[method-assign]
    CpuGpuBuffer.copy_to_cpu = copy_to_cpu  # type: ignore[method-assign]


def apply_compat_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    try:
        import torch._dynamo

        torch._dynamo.config.disable = True
    except Exception:
        logger.warning("Could not disable torch._dynamo; compile paths may fail")

    from vllm.v1.worker.block_table import BlockTable

    BlockTable.compute_slot_mapping = (  # type: ignore[method-assign]
        _metal_compute_slot_mapping
    )

    _patch_cpu_gpu_buffer_blocking()

    torch.cuda.is_current_stream_capturing = (  # type: ignore[method-assign]
        lambda: False
    )

    _APPLIED = True
    logger.info(
        "Applied Metal compat patches (dynamo off, torch slot mapping, "
        "blocking host copies, stream-capture check stubbed False)"
    )
