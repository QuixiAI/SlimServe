# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-worker single-device ``ZE_AFFINITY_MASK`` for Intel XPU tensor parallel.

In a multi-device Level Zero context every device allocation is shared to the
peer cards via dma-buf, and with no PCIe P2P path the ``xe`` driver backs each
shared buffer in system RAM: host RSS tracks total VRAM in use. Spawning each
TP worker with a *single-device* mask keeps its Level Zero context
single-device. Measured on 4x Intel Arc Pro B70 (2026-08-17, sibling XPU vLLM
tree on this host): host RSS 118 GiB -> 34 GiB and +18% tok/s.

The mask has to be set in the parent *before* ``Process.start()`` because
Level Zero enumerates devices as soon as ``vllm`` is imported in the child.
This module therefore imports no torch so ``multiproc_executor`` can import it
on every platform.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, VllmConfig

logger = init_logger(__name__)

# Set (to "1") in a worker's environment when it was spawned pinned to one
# device; the worker then addresses ``xpu:0`` and the communicator knows peer
# memory is not directly addressable.
XPU_WORKER_AFFINITY_PINNED_ENV = "VLLM_XPU_WORKER_AFFINITY_PINNED"


def xpu_dp_adjusted_local_rank(parallel_config: ParallelConfig, local_rank: int) -> int:
    """The physical device index a worker maps to (mirrors XPUWorker.init_device)."""
    if (
        parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
        and parallel_config.data_parallel_backend != "ray"
        and parallel_config.nnodes_within_dp == 1
    ):
        dp_local_rank = parallel_config.data_parallel_rank_local
        if dp_local_rank is None:
            dp_local_rank = parallel_config.data_parallel_index
        tp_pp = parallel_config.pipeline_parallel_size * parallel_config.tensor_parallel_size
        return local_rank + dp_local_rank * tp_pp
    return local_rank


def xpu_worker_affinity_pinned() -> bool:
    return os.environ.get(XPU_WORKER_AFFINITY_PINNED_ENV) == "1"


@contextlib.contextmanager
def xpu_worker_affinity_env(local_rank: int, vllm_config: VllmConfig):
    """Narrow ``ZE_AFFINITY_MASK`` to one device around a worker spawn."""
    from vllm.platforms import current_platform
    from vllm.utils.system_utils import get_mp_context

    if not current_platform.is_xpu() or not envs.VLLM_XPU_PER_WORKER_AFFINITY:
        yield
        return
    if get_mp_context().get_start_method() != "spawn":
        logger.warning_once(
            "VLLM_XPU_PER_WORKER_AFFINITY needs the spawn start method; "
            "workers will share a multi-device Level Zero context."
        )
        yield
        return

    physical = xpu_dp_adjusted_local_rank(vllm_config.parallel_config, local_rank)
    parent_mask = os.environ.get("ZE_AFFINITY_MASK")
    if parent_mask:
        entries = [e.strip() for e in parent_mask.split(",") if e.strip()]
        if physical >= len(entries):
            raise RuntimeError(
                f"worker local_rank {local_rank} (physical {physical}) is outside "
                f"the parent ZE_AFFINITY_MASK={parent_mask!r}"
            )
        # Index INTO the parent's mask, not the raw device id.
        child_mask = entries[physical]
    else:
        child_mask = str(physical)

    saved_mask = os.environ.get("ZE_AFFINITY_MASK")
    saved_pinned = os.environ.get(XPU_WORKER_AFFINITY_PINNED_ENV)
    os.environ["ZE_AFFINITY_MASK"] = child_mask
    os.environ[XPU_WORKER_AFFINITY_PINNED_ENV] = "1"
    logger.info("XPU worker local_rank=%d pinned to ZE_AFFINITY_MASK=%s", local_rank, child_mask)
    try:
        yield
    finally:
        for key, value in (
            ("ZE_AFFINITY_MASK", saved_mask),
            (XPU_WORKER_AFFINITY_PINNED_ENV, saved_pinned),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
