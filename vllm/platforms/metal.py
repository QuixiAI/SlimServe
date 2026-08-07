# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple Silicon (Metal / PyTorch-MPS) platform.

The third serving platform in this fork, alongside CUDA and ROCm. It keeps
vLLM's native model executor and the V1 GPU worker and runner, and moves the
math onto ``mps`` tensors; the kernels behind it are the QuixiCore-Metal
sources vendored under ``csrc/quixicore/metal/``, reached through
``vllm._quixicore_C`` exactly as the CUDA path reaches its own.

Three facts shape everything here:

- A Mac is one GPU with one unified memory pool. There is no second device to
  shard onto and no separate VRAM to profile, so tensor parallelism is
  meaningless and the KV budget comes from the Metal driver's recommended
  working-set maximum rather than ``mem_get_info``.
- There is no CUDA-graph analogue, so capture is forced off. That is what gates
  out most of the CUDA-specific paths in the shared GPU runner.
- Triton does not target Metal and is not installed on macOS, so the Triton
  helpers vLLM uses for batch prep are replaced with torch equivalents in
  ``metal_compat``.
"""

import os
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)

try:
    import vllm._quixicore_C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._quixicore_C with %r", e)


@cache
def _gpu_core_count(default: int = 32) -> int:
    """Apple GPU core count, standing in for the CUDA SM count in heuristics."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return default
    for line in out.splitlines():
        if "Total Number of Cores" in line:
            try:
                return int(line.split(":")[1].strip())
            except (IndexError, ValueError):
                break
    return default


class MetalPlatform(Platform):
    _enum = PlatformEnum.METAL
    device_name: str = "metal"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    # No NCCL analogue on Apple; gloo is the only collective backend available,
    # and with one device it never carries real traffic.
    dist_backend: str = "gloo"
    # Neither inductor backend applies: there is no MPS backend, and the CPU
    # one needs a toolchain we do not require at runtime.
    simple_compile_backend: str = "eager"
    supported_quantization: list[str] = ["gguf"]

    @classmethod
    def is_available(cls) -> bool:
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_built() and mps.is_available())

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # bf16 first: every GGUF this fork serves keeps its norms, attention
        # and embeddings in bf16, and the vendored kernels are written against
        # that. fp16 stays available for the quantized GEMV path.
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Apple Silicon (Metal)"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        # vLLM gates model and quantization choices on CUDA compute capability
        # in many places. Report sm80, the floor this fork's kernels assume,
        # so those gates resolve to the same answers they do on A100.
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """What Metal will actually let the GPU hold, not what the box has.

        ``recommendedMaxWorkingSetSize`` is roughly 90% of physical (115.4 of
        128 GiB on an M5 Max). Sizing against physical RAM instead is how you
        get an allocation that succeeds and then stalls the machine.
        """
        try:
            return int(torch.mps.recommended_max_memory())
        except Exception:
            import psutil

            return int(psutil.virtual_memory().total)

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        try:
            return float(torch.mps.current_allocated_memory())
        except Exception:
            return 0.0

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """No-op: MPS is a single implicit device."""

    @classmethod
    def device_count(cls) -> int:
        return 1

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            torch.manual_seed(seed)

    @classmethod
    def inference_mode(cls):
        # inference_mode() interacts badly with non-CUDA backends; the CPU and
        # TPU platforms use no_grad() and Metal follows them.
        return torch.no_grad()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def is_integrated_gpu(cls, device_id: int = 0) -> bool:
        # Unified memory. vLLM already corrects its memory accounting for UMA
        # devices (GH200, Spark, Jetson); Apple gets that handling for free.
        return True

    @classmethod
    def import_kernels(cls) -> None:
        import vllm._quixicore_C  # noqa: F401

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return False

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                f"Apple Silicon does not support dtype {dtype}; "
                "use bfloat16, float16 or float32."
            )

    @classmethod
    def get_punica_wrapper(cls) -> str:
        raise NotImplementedError("LoRA (punica) has no Metal kernels.")

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # One device: the base communicator's no-op collectives are correct.
        return (
            "vllm.distributed.device_communicators."
            "base_device_communicator.DeviceCommunicatorBase"
        )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if getattr(attn_selector_config, "use_mla", False):
            # Resolves so the failure names the missing kernels rather than
            # surfacing as an ImportError; see that module for what is left.
            return (
                "vllm.v1.attention.backends.mla.metal_mla_sparse.MetalMLASparseBackend"
            )
        return "vllm.v1.attention.backends.metal_attn.MetalAttentionBackend"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config import CompilationMode

        # vLLM is fully imported by now, which is the only safe point to
        # install the Triton replacements and disable dynamo.
        from vllm.platforms.metal_compat import apply_compat_patches

        apply_compat_patches()

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.metal_worker.MetalWorker"
        parallel_config.disable_custom_all_reduce = True
        if getattr(parallel_config, "enable_dbo", False):
            parallel_config.enable_dbo = False
        if parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "Apple exposes one GPU per machine, so tensor parallelism is "
                f"not available; got tensor_parallel_size="
                f"{parallel_config.tensor_parallel_size}."
            )

        vllm_config.scheduler_config.async_scheduling = False

        # No CUDA graphs here. Forcing capture off is what keeps the shared GPU
        # runner off its torch.cuda.graph and Stream paths.
        compilation_config = vllm_config.compilation_config
        compilation_config.cudagraph_capture_sizes = []
        try:
            from vllm.config import CUDAGraphMode

            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        except Exception:
            pass
        compilation_config.mode = CompilationMode.NONE

        model_config = vllm_config.model_config
        if model_config is not None:
            model_config.disable_cascade_attn = True

        cache_config = vllm_config.cache_config
        if not getattr(cache_config, "user_specified_block_size", False):
            # The vendored paged-attention and KV-cache kernels take any
            # multiple of 16; 16 is vLLM's default and the best-tested.
            cache_config.block_size = 16

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return False

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        raise NotImplementedError("Graph capture has no Metal analogue.")


def gpu_core_count() -> int:
    """Apple GPU core count, the closest honest analogue to a CUDA SM count."""
    return _gpu_core_count()


def metal_is_available() -> bool:
    """Whether this host can serve on the Apple GPU.

    Kept module-level and import-light: it runs during platform resolution,
    where any exception is swallowed and would silently demote us.
    """
    if os.environ.get("VLLM_DISABLE_METAL") == "1":
        return False
    import platform as _platform

    if _platform.system() != "Darwin" or _platform.machine() != "arm64":
        return False
    return MetalPlatform.is_available()
