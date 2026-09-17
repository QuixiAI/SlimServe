# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x PCIe DMA-ring all-reduce for large eager messages.

The ring moves each shard once over the copy engines (reduce-scatter, then
all-gather) instead of having every SM-driven peer read cross the bus, so
on PCIe-only boxes it wins once the message is tens of MiB: the prefill
chunk reductions of a tensor-parallel model. Decode-size messages stay on
the custom kernel, and nothing here runs inside a CUDA graph capture.
"""

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class B12xDmaAllReduce:
    """Size-gated wrapper around ``b12x.comm.pcie.pcie_dma.PCIeDmaAllReduce``.

    ``disabled`` is set when b12x is missing, the world size is not one the
    ring supports, or its IPC setup fails; the caller then skips it.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        min_bytes: int,
        max_bytes: int,
    ) -> None:
        self.disabled = True
        self._ring = None
        try:
            from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce
        except ImportError:
            logger.warning(
                "VLLM_B12X_DMA_AR_MIN_MB is set but b12x is not installed "
                "(pip install b12x); large all-reduces stay on the custom "
                "kernel / NCCL."
            )
            return
        try:
            ring = PCIeDmaAllReduce(
                exchange_group=group, device=device, max_bytes=max_bytes
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("b12x DMA-ring all-reduce unavailable: %s", exc)
            return
        ring.min_bytes = min_bytes
        self._ring = ring
        self.disabled = False
        logger.info(
            "b12x DMA-ring all-reduce enabled for eager messages of %d..%d MiB "
            "(wire %s)",
            min_bytes >> 20,
            max_bytes >> 20,
            ring.wire_mode,
        )

    def should_use(self, inp: torch.Tensor) -> bool:
        if self.disabled or torch.cuda.is_current_stream_capturing():
            return False
        assert self._ring is not None
        return self._ring.should_allreduce(inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._ring is not None
        return self._ring.all_reduce(inp)

    def close(self) -> None:
        if self._ring is not None:
            self._ring.close()
            self._ring = None
        self.disabled = True


def maybe_create_b12x_dma_all_reduce(
    group: ProcessGroup, device: torch.device
) -> B12xDmaAllReduce | None:
    min_mb = envs.VLLM_B12X_DMA_AR_MIN_MB
    if min_mb <= 0:
        return None
    max_bytes = max(min_mb, envs.VLLM_B12X_DMA_AR_MAX_MB) << 20
    comm = B12xDmaAllReduce(group, device, min_mb << 20, max_bytes)
    return None if comm.disabled else comm
