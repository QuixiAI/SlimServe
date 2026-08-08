# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V1 GPU worker on PyTorch-MPS.

Subclasses the stock ``Worker`` and replaces only what is hard-wired to CUDA:
device setup, the memory-pool context, kernel warmup, and the KV-cache budget.
Weight loading, ``execute_model`` and the KV plumbing are inherited.
"""

import gc
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger(__name__)


class MetalWorker(Worker):
    def init_device(self):
        assert self.device_config.device_type == "mps"

        self.device = torch.device("mps")
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # One process, one device. gloo is the only backend available and the
        # group never carries real traffic; it exists so the shared code paths
        # that assume an initialized group keep working.
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )

        set_random_seed(self.model_config.seed)

        gc.collect()
        current_platform.empty_cache()

        # No CUDA MemorySnapshot: determine_available_memory() budgets from
        # unified-memory totals instead.
        self.init_snapshot = None  # type: ignore[assignment]
        self.requested_memory = None  # type: ignore[assignment]

        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import GPUModelRunner

            self.model_runner = cast(Any, GPUModelRunner(self.vllm_config, self.device))
        else:
            from vllm.v1.worker.metal_model_runner import MetalModelRunner

            self.model_runner = MetalModelRunner(self.vllm_config, self.device)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # No CuMem sleep-mode allocator; weights load straight into the
        # unified-memory pool.
        return nullcontext()

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load weights without the CUDA-only staging warmup.

        The base GPU worker allocates pinned host memory and calls
        ``torch.cuda.synchronize()`` before constructing the model.  Neither
        operation exists on MPS, and unified memory does not need that staging
        path in the first place.  Keep the otherwise-useful vLLM loading
        context and communicator initialization.
        """
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                self._maybe_get_memory_pool_context(tag="weights"),
                set_current_vllm_config(self.vllm_config),
            ):
                tp_device_comm = get_tp_group().device_communicator
                self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
                if tp_device_comm is not None:
                    wait_for_comm_init = getattr(
                        tp_device_comm, "wait_for_comm_init", None
                    )
                    if wait_for_comm_init is not None:
                        wait_for_comm_init()
        finally:
            if gc_was_enabled:
                gc.enable()
        gc.freeze()

    def compile_or_warm_up_model(self):
        # kernel_warmup() eagerly imports CUDA-only JIT kernels (deep_gemm,
        # flashinfer, minimax MSA) that do not exist on macOS. The vendored
        # Metal kernels are compiled ahead of time into the extension's
        # metallib, so there is nothing to warm here.
        import vllm.v1.worker.gpu_worker as gpu_worker

        original = gpu_worker.kernel_warmup
        original_v2 = gpu_worker.warmup_kernels
        gpu_worker.kernel_warmup = lambda worker: None
        gpu_worker.warmup_kernels = lambda *args, **kwargs: None
        try:
            return super().compile_or_warm_up_model()
        finally:
            gpu_worker.kernel_warmup = original
            gpu_worker.warmup_kernels = original_v2

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """KV-cache budget, from unified memory rather than a device pool.

        There is no ``mem_get_info`` to profile against: weights, KV and
        activations all come out of one pool. Run the profile pass for shape
        warmup, then budget against what Metal will let the GPU hold.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            logger.info(
                "Using %.2f GiB fixed Metal KV-cache budget; skipping the "
                "memory-profiling dummy forward",
                kv_cache_memory_bytes / 2**30,
            )
            return kv_cache_memory_bytes

        self.model_runner.profile_run()

        total = current_platform.get_device_total_memory()
        util = self.cache_config.gpu_memory_utilization
        in_use = int(current_platform.get_current_memory_usage())
        available = int(total * util) - in_use
        if available <= 0:
            raise RuntimeError(
                "No unified memory left for the KV cache: "
                f"{total / 2**30:.1f} GiB usable * {util} utilization, "
                f"{in_use / 2**30:.1f} GiB already held by weights. "
                "Use a smaller quant or raise --gpu-memory-utilization."
            )
        logger.info(
            "Metal KV-cache budget: %.2f GiB (%.2f GiB usable * %.2f "
            "utilization - %.2f GiB in use)",
            available / 2**30,
            total / 2**30,
            util,
            in_use / 2**30,
        )
        return available
