# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V1 GPU worker on Intel XPU.

Subclasses the stock ``Worker`` and replaces only what is hard-wired to CUDA:
device selection (single-device affinity pinning aware), the oneCCL process
group environment, and profiling. Weight loading, memory profiling,
``execute_model`` and the KV plumbing are inherited.
"""

from __future__ import annotations

import gc
import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.xpu_affinity import (
    xpu_dp_adjusted_local_rank,
    xpu_worker_affinity_pinned,
)
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.worker.xpu_model_runner import (
    XPUModelRunner,
    XPUModelRunnerV2,
    install_torch_cuda_aliases,
)

from .utils import request_memory

logger = init_logger(__name__)


class XPUWorker(Worker):
    """A XPU worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        assert self.device_config.device_type == "xpu"
        assert current_platform.is_xpu()

    def init_device(self):
        # The shared worker/runner code touches torch.cuda.* by name well
        # before the model runner exists (streams in the KV connector, events
        # in the sampler); alias early.
        install_torch_cuda_aliases()
        current_platform.import_kernels()

        parallel_config = self.parallel_config
        affinity_pinned = xpu_worker_affinity_pinned()
        # DP offset applied to local_rank (mirrored by xpu_affinity so the
        # parent's mask index and the worker agree).
        self.local_rank = xpu_dp_adjusted_local_rank(parallel_config, self.local_rank)
        if affinity_pinned:
            # A single-device ZE_AFFINITY_MASK: this process sees one GPU.
            visible = torch.accelerator.device_count()
            assert visible == 1, (
                f"affinity-pinned XPU worker sees {visible} devices; expected 1"
            )
            self.device_index = 0
        else:
            visible = torch.accelerator.device_count()
            assert self.local_rank < visible, (
                f"DP adjusted local rank {self.local_rank} is out of bounds for "
                f"{visible} visible XPU devices."
            )
            self.device_index = self.local_rank

        device = self.device_config.device
        if not (isinstance(device, torch.device) and device.type == "xpu"):
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")
        self.device = torch.device(f"xpu:{self.device_index}")
        torch.accelerator.set_device_index(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)
        self.init_gpu_memory = torch.xpu.get_device_properties(
            self.device_index
        ).total_memory

        # oneCCL process-group environment (must precede PG init). Rank
        # bookkeeping uses local_rank; device addressing uses device_index.
        world = str(parallel_config.world_size)
        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        os.environ.setdefault("LOCAL_WORLD_SIZE", world)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ.setdefault("CCL_PROCESS_LAUNCHER", "none")
        os.environ.setdefault("CCL_LOCAL_SIZE", os.environ["LOCAL_WORLD_SIZE"])
        os.environ["CCL_LOCAL_RANK"] = str(self.local_rank)
        os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "sockets")
        os.environ.setdefault("FI_TCP_IFACE", "lo")

        # parallel_state derives torch.device(f"xpu:{local_rank}") from this
        # argument, so pass the VISIBLE index, not the CCL local rank.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.device_index,
            current_platform.dist_backend,
        )
        # No eager warmup all-reduce here: on multi-XPU hosts it can spin
        # before model load; real execution initializes collectives.

        if self.use_v2_model_runner:
            logger.info_once("Using V2 Model Runner")

        set_random_seed(self.model_config.seed)

        # Memory snapshot after the process group exists. No empty_cache:
        # device-wide cache purges walk transient SYCL queues.
        if self.cache_config.kv_cache_memory_bytes is None:
            gc.collect()
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        model_runner = XPUModelRunnerV2 if self.use_v2_model_runner else XPUModelRunner
        self.model_runner = model_runner(self.vllm_config, self.device)  # type: ignore

        if self.rank == 0:
            report_usage_stats(self.vllm_config)

    def compile_or_warm_up_model(self):
        # The shared kernel_warmup imports CUDA-only warmups at call time
        # (minimax msa -> fusion pass matchers -> torch.ops._C fp8 quant ops)
        # and warms FlashInfer/DeepGEMM/CuteDSL paths that do not exist here.
        # Skip it as Metal does; the DSV4 Triton kernels JIT on the profile
        # run and the first request. TODO(xpu): a targeted DSV4 sparse-MLA
        # Triton warmup once the serving path is measured.
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

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Set --profiler-config.profiler=torch "
                "and --profiler-config.torch_profiler_dir=DIR."
            )
        if is_start and self.profiler is None:
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
            trace_name = (
                f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix
            )
            self.profiler = TorchProfilerWrapper(
                self.profiler_config,
                worker_name=trace_name,
                local_rank=self.local_rank,
                activities=["CPU", "XPU"],
            )
        super().profile(is_start=is_start, profile_prefix=profile_prefix)
