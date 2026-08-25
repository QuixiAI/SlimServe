# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V1 GPU worker on PyTorch-MPS.

Subclasses the stock ``Worker`` and replaces only what is hard-wired to CUDA:
device setup, the memory-pool context, kernel warmup, and the KV-cache budget.
Weight loading, ``execute_model`` and the KV plumbing are inherited.
"""

import gc
import os
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

# Headroom kept outside the KV pool for activations, sampling, kernel scratch
# and allocator fragmentation, as a fraction of the Metal working set and as
# an absolute floor. See `_fit_kv_pool_to_working_set` for the measurements.
_KV_RESERVE_FRAC = 0.04
_KV_RESERVE_MIN_BYTES = 4 << 30
# Below this a pool cannot hold a useful request, so refuse instead of
# booting into a geometry that will fail on the first prompt.
_KV_POOL_FLOOR_BYTES = 1 << 29


class MetalWorker(Worker):
    def init_device(self):
        assert self.device_config.device_type == "mps"

        from vllm.v1.worker.metal_syncprof import install as install_syncprof

        install_syncprof()

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
        if not load_dummy_weights:
            self._make_weights_resident()
            self._pin_weights_resident()
            if os.environ.get("VLLM_QC_STEP_TAPE", "0") != "0":
                from vllm.models.deepseek_v4.metal_tape import (
                    maybe_install_tape,
                )

                maybe_install_tape(self.model_runner.model)

    @staticmethod
    def _compressor_bytes() -> int:
        """Physical bytes currently held by the macOS VM compressor.

        MTLBuffer pages that get compressed are attributed to the GPU
        subsystem, not this process, so per-process accounting misses them;
        vm_stat's global counter is the only reliable residency signal.
        """
        import subprocess

        import regex as re

        try:
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=10
            ).stdout
            m = re.search(r"page size of (\d+)", out)
            page = int(m.group(1)) if m else 16384
            m = re.search(r"Pages occupied by compressor:\s+(\d+)", out)
            return int(m.group(1)) * page if m else 0
        except (OSError, subprocess.SubprocessError, ValueError):
            return 0

    def _make_weights_resident(self) -> None:
        """Touch every MPS weight buffer once on the GPU.

        Load-time page-cache pressure can push freshly written weight buffers
        into the VM compressor; serving then decompresses on every touch at
        a tiny fraction of memory bandwidth. One GPU-side read promotes
        everything resident while nothing competes for memory — paid once at
        boot instead of every token.

        A single pass is not always enough: on a dirty box the sweep itself
        can rotate earlier tensors back out while repairing later ones.
        Verify against the global compressor occupancy and repeat until it
        drains or stops improving.
        """
        threshold = 4 << 30
        occupancy = 0
        for sweep_pass in range(3):
            self._sweep_pass_once()
            previous = occupancy
            occupancy = self._compressor_bytes()
            if occupancy < threshold:
                return
            logger.warning(
                "VM compressor still holds %.2f GiB after resident sweep "
                "pass %d; weights are not fully resident",
                occupancy / 2**30,
                sweep_pass + 1,
            )
            if sweep_pass > 0 and previous and occupancy > previous * 0.9:
                logger.warning(
                    "Resident sweep is not converging (%.2f -> %.2f GiB); "
                    "total memory demand likely exceeds RAM. Serving will "
                    "thrash the compressor.",
                    previous / 2**30,
                    occupancy / 2**30,
                )
                return

    def _weight_tensors(self) -> list:
        import itertools

        modules = [self.model_runner.model]
        speculator = getattr(self.model_runner, "speculator", None)
        spec_model = getattr(speculator, "model", None)
        if spec_model is not None:
            modules.append(spec_model)
        return [
            t
            for module in modules
            for t in itertools.chain(module.parameters(), module.buffers())
            if not torch.nn.parameter.is_lazy(t) and t.device.type == "mps"
        ]

    def _pin_weights_resident(self) -> None:
        """Pin weight allocations into an MTLResidencySet (macOS 15+).

        The resident sweep only decompresses; the pages stay pageable and
        macOS re-compresses them in tens-of-GiB waves during serving while
        the GPU stalls faulting them back. Pinning makes the weight heaps
        permanently GPU-resident so the compressor never takes them.
        ``VLLM_METAL_RESIDENCY=0`` disables.
        """
        if os.environ.get("VLLM_METAL_RESIDENCY", "1") != "1":
            logger.info("Metal residency pinning disabled by env")
            return
        try:
            from vllm import _quixicore_C as qc
        except ImportError as e:
            logger.warning("Metal residency pinning unavailable: %s", e)
            return
        if not hasattr(qc, "residency_pin"):
            logger.warning(
                "Metal residency pinning unavailable: extension predates residency_pin"
            )
            return
        added, nbytes = qc.residency_pin(self._weight_tensors())
        if added:
            logger.info(
                "Pinned %d Metal allocations (%.2f GiB) into the weight residency set",
                added,
                nbytes / 2**30,
            )
        else:
            logger.warning(
                "Metal residency pinning added no allocations "
                "(pre-macOS 15, or no MPS weights?)"
            )

    def _sweep_pass_once(self) -> None:
        import time

        start = time.perf_counter()
        total_bytes = 0
        # Torch's MPS reduction dispatch SIGSEGVs (nil compute pipeline state
        # in reduction_dispatch_mps) on GiB-scale uint8 sums when
        # PYTORCH_MPS_LOG_PROFILE_INFO is enabled, so sweep in slices small
        # enough for every dispatch path. Each sum also materializes a
        # same-size transient copy; the periodic synchronize keeps those from
        # accumulating in one command stream.
        chunk_elems = 128 << 20
        sync_window = 2 << 30
        unsynced_bytes = 0
        for t in self._weight_tensors():
            nbytes = t.numel() * t.element_size()
            total_bytes += nbytes
            try:
                flat = t.detach().view(-1).view(torch.uint8)
            except RuntimeError:
                flat = None
            if flat is None:
                t.detach().sum()
            else:
                for off in range(0, flat.numel(), chunk_elems):
                    flat[off : off + chunk_elems].sum()
            unsynced_bytes += nbytes
            if unsynced_bytes >= sync_window:
                torch.mps.synchronize()
                unsynced_bytes = 0
        torch.mps.synchronize()
        logger.info(
            "Resident sweep touched %.2f GiB of MPS weights in %.2f s",
            total_bytes / 2**30,
            time.perf_counter() - start,
        )

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
            result = super().compile_or_warm_up_model()
        finally:
            gpu_worker.kernel_warmup = original
            gpu_worker.warmup_kernels = original_v2
        # Warmup and KV-cache allocation run after the resident sweep and can
        # push weights back into the VM compressor on a tight box; re-verify
        # so serving never starts on a half-compressed model.
        if self._compressor_bytes() > (4 << 30):
            logger.warning(
                "VM compressor grew during warmup; re-running resident sweep"
            )
            self._make_weights_resident()
        return result

    def _fit_kv_pool_to_working_set(self, requested_bytes: int) -> int:
        """Largest KV pool that stays resident, given what is already held.

        Metal's ``recommendedMaxWorkingSetSize`` is not enforced by failing
        allocations. A pool that pushes weights plus KV past it allocates
        fine, and the driver then pages: since ``_pin_weights_resident`` pins
        the weight heaps, the KV pool is what gets evicted, so decode reads
        non-resident pages and returns token soup while the request still
        answers HTTP 200. There is no OOM to catch, which is why the pool has
        to be sized against the working set up front.

        Residency is measured, not declared: GGUF weights land anywhere from
        1.07x to 2x their file bytes once the Metal path is done with them,
        so a profile cannot compute this from artifact sizes.
        """
        free_bytes, total_bytes = torch.accelerator.get_memory_info(self.device)
        resident_bytes = total_bytes - free_bytes
        # Activations, sampling, kernel scratch and allocator fragmentation
        # all live outside the pool. Measured on an M5 Max serving
        # DeepSeek-V4-Flash IQ2_XXS (93.57 GiB resident, 107.52 GiB working
        # set): 4.9 GiB of headroom decoded at 26 tok/s, 1.9 GiB crawled at
        # 3.6 tok/s, and overrunning the working set produced token soup.
        reserve_bytes = max(_KV_RESERVE_MIN_BYTES, int(total_bytes * _KV_RESERVE_FRAC))
        budget_bytes = total_bytes - resident_bytes - reserve_bytes

        if budget_bytes < _KV_POOL_FLOOR_BYTES:
            raise RuntimeError(
                "No unified memory left for the KV cache: weights and runtime "
                f"hold {resident_bytes / 2**30:.2f} GiB of the "
                f"{total_bytes / 2**30:.2f} GiB Metal working set, leaving "
                f"{max(budget_bytes, 0) / 2**30:.2f} GiB after the "
                f"{reserve_bytes / 2**30:.2f} GiB activation reserve. Serve a "
                "smaller quant, drop the draft model, or use a machine with "
                "more unified memory."
            )

        if requested_bytes <= budget_bytes:
            return requested_bytes

        logger.warning(
            "Requested %.2f GiB KV pool does not fit this machine; using "
            "%.2f GiB. Weights and runtime hold %.2f GiB of the %.2f GiB "
            "Metal working set and %.2f GiB is reserved for activations. An "
            "oversized pool on unified memory is not an OOM: the driver "
            "evicts KV pages and decode silently degrades into garbage.",
            requested_bytes / 2**30,
            budget_bytes / 2**30,
            resident_bytes / 2**30,
            total_bytes / 2**30,
            reserve_bytes / 2**30,
        )
        return budget_bytes

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """KV-cache budget, from unified memory rather than a device pool.

        There is no ``mem_get_info`` to profile against: weights, KV and
        activations all come out of one pool. Run the profile pass for shape
        warmup, then budget against what Metal will let the GPU hold.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            granted = self._fit_kv_pool_to_working_set(kv_cache_memory_bytes)
            self.cache_config.kv_cache_memory_bytes = granted
            logger.info(
                "Using %.2f GiB fixed Metal KV-cache budget; skipping the "
                "memory-profiling dummy forward",
                granted / 2**30,
            )
            return granted

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
        # Utilization is a fraction of the working set, so this path cannot
        # overrun it by construction -- but at utilization near 1.0 it leaves
        # no room for activations, which the same reserve covers.
        return self._fit_kv_pool_to_working_set(available)
