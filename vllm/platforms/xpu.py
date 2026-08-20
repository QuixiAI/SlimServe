# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU platform (Arc Pro B70 / Battlemage, SYCL + Level Zero + oneCCL).

Adapted from the upstream XPU platform this fork dropped in 75b5e0f60, minus
its ``vllm_xpu_kernels`` dependency: device kernels come from the vendored
QuixiCore-XPU SYCL library (``vllm._quixicore_C``, see csrc/quixicore/xpu/)
plus Triton-XPU, and everything hard-wired to CUDA in the shared V1 GPU path
is aliased in vllm/v1/worker/xpu_model_runner.py.

Operating rules learned on this class of hardware (see perf notebook
2026-08-18 and the sibling XPU tree's findings):
- one SYCL runtime per process (torch's bundled one; never source setvars.sh
  in the serving environment);
- each TP worker gets a single-device ZE_AFFINITY_MASK (host-RAM mirroring
  otherwise, see xpu_affinity.py);
- oneCCL over OFI/TCP loopback with socket-based Level Zero IPC exchange; the
  direct ze IPC P2P path hangs on PCIe-only Arc cards;
- no device-wide synchronize/empty_cache on hot paths: torch's device sync
  walks transient SYCL queues and trips a UR queue-release bug.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def _torch_runtime_lib_dir() -> str | None:
    """The wheel-bundled runtime dir (libsycl, libur_*, libccl*): <venv>/lib."""
    import pathlib

    # .../<venv>/lib/python3.X/site-packages/torch/__init__.py -> <venv>/lib
    candidate = pathlib.Path(torch.__file__).resolve().parents[3]
    if (candidate / "libccl.so.1").is_file() or (candidate / "libsycl.so.9").is_file():
        return str(candidate)
    return None


def ensure_xpu_runtime_env() -> None:
    """Make the pip oneCCL runtime loadable by spawned workers.

    oneCCL 2022 (torch's bundled libccl.so.2) is a plugin loader that
    dlopen()s ``libccl.so.1`` by soname; that file lives beside libsycl in
    <venv>/lib, which is on torch's RUNPATH but not on the dlopen search path,
    so PG init fails with "Could not load any plugin". The loader reads
    LD_LIBRARY_PATH once at process start, so this has to be set in the parent
    before the engine/worker processes are spawned (check_and_update_config).
    """
    lib_dir = _torch_runtime_lib_dir()
    if lib_dir is None:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir, *parts])


def xpu_is_available() -> bool:
    try:
        return torch.xpu.is_available() and torch.xpu.device_count() > 0
    except Exception:
        return False


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    # Intel XPU's device key is "GPU" for Ray.
    ray_device_key: str = "GPU"
    dist_backend: str = "xccl"
    device_control_env_var: str = "ZE_AFFINITY_MASK"
    # No inductor for bring-up; the DSV4 profile runs eager + native kernels.
    simple_compile_backend: str = "eager"

    @classmethod
    def import_kernels(cls) -> None:
        # No vllm._C on XPU: the CUDA/HIP stable-ABI extension does not exist
        # here. The elementwise ops the shared layers call through
        # torch.ops._C are registered in Python (spawned workers included).
        from vllm.platforms.xpu_c_ops import register_xpu_c_ops

        register_xpu_c_ops()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str:
        # DeepSeek-V4 sparse-MLA layers select their own backend through
        # AttentionLayerBase.get_attn_backend; this is for generic layers.
        kv_cache_dtype = attn_selector_config.kv_cache_dtype
        if kv_cache_dtype is not None and kv_cache_dtype.startswith("turboquant_"):
            logger.info_once("Using TurboQuant attention backend on XPU.")
            return AttentionBackendEnum.TURBOQUANT.get_path()
        if attn_selector_config.use_mla:
            logger.info_once("Using Triton MLA backend on XPU.")
            return AttentionBackendEnum.TRITON_MLA.get_path()
        if selected_backend not in (None, AttentionBackendEnum.TRITON_ATTN):
            logger.warning_once(
                "Attention backend %s is not available on XPU; using TRITON_ATTN.",
                selected_backend,
            )
        return AttentionBackendEnum.TRITON_ATTN.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list[AttentionBackendEnum]:
        return [AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.TORCH_SDPA]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: AttentionBackendEnum | None = None,
    ) -> AttentionBackendEnum:
        if backend is not None and backend in cls.get_supported_vit_attn_backends():
            return backend
        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.xpu.set_device(device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        # XPU capability format differs from CUDA's; comparisons against SM
        # numbers would misfire, so report none.
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.xpu.get_device_properties(device_id).total_memory

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CUDAGraphMode
        from vllm.config.compilation import CompilationMode
        from vllm.utils.torch_utils import supports_xpu_graph

        # vLLM is fully imported by now: the only safe point to define the
        # torch.ops._C elementwise ops the shared layers call by name.
        from vllm.platforms.xpu_c_ops import register_xpu_c_ops

        register_xpu_c_ops()

        compilation_config = vllm_config.compilation_config
        if compilation_config.compile_sizes is None:
            compilation_config.compile_sizes = []

        # XPU graphs (SYCL command graphs through torch.xpu.XPUGraph) are the
        # serving mode, as everywhere else in this fork. The capture is the
        # breakable one: attention/indexer ops that host-sync and the oneCCL
        # collectives (not capturable) run as eager segments between graph
        # segments (vllm/compilation/breakable_cudagraph.py); replay is
        # ordered with current-stream waits (device-wide sync is a UR fault).
        if not supports_xpu_graph():
            raise RuntimeError(
                "XPU graphs need torch >= 2.11 (torch.xpu.XPUGraph); found "
                f"{torch.__version__}. This fork does not serve eager."
            )
        os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
        # oneCCL collectives cannot be recorded into a SYCL command graph:
        # a FULL capture with TP > 1 replays garbage (measured 2026-08-18,
        # Qwen2.5-0.5B TP2). The breakable PIECEWISE capture runs them as
        # eager segments; FULL stays available for single-device serving.
        if (
            vllm_config.parallel_config.world_size > 1
            and compilation_config.cudagraph_mode is not None
            and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and compilation_config.cudagraph_mode != CUDAGraphMode.PIECEWISE
        ):
            logger.info_once(
                "XPU with tensor parallelism: cudagraph_mode %s -> PIECEWISE "
                "(collectives are graph breaks on oneCCL).",
                compilation_config.cudagraph_mode.name,
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        if compilation_config.mode != CompilationMode.NONE:
            # No inductor on XPU: graphs are recorded from the eager model
            # (CompilationMode.NONE + cudagraphs), the proven configuration.
            logger.info_once("Inductor compilation is not used on XPU; mode=NONE.")
            compilation_config.mode = CompilationMode.NONE

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.xpu_worker.XPUWorker"
        # The CUDA custom all-reduce path does not exist here; oneCCL does
        # the collectives (the sibling tree's SYCL P2P all-reduce hangs under
        # per-worker affinity pinning, which wins on net).
        parallel_config.disable_custom_all_reduce = True
        if getattr(parallel_config, "enable_dbo", False):
            parallel_config.enable_dbo = False
        if vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.enable_permute_local_kv = True

        model_config = vllm_config.model_config
        if model_config is not None:
            model_config.disable_cascade_attn = True

        ensure_xpu_runtime_env()
        # UCX can misdetect GPU memory as host memory (invalid access).
        os.environ["UCX_MEMTYPE_CACHE"] = "n"
        # spawn is the only supported multiprocessing method on XPU.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # oneCCL over OFI on a single host: pin libfabric to loopback (it
        # otherwise picks a random bridge NIC and hangs at PG init), exchange
        # Level Zero IPC handles over sockets (direct ze IPC P2P hangs on
        # PCIe-only Arc), and skip the fabric topology probe.
        os.environ.setdefault("FI_TCP_IFACE", "lo")
        os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "sockets")
        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        os.environ.setdefault("CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK", "0")
        # torch.distributed's flight recorder captures a Python traceback per
        # collective and segfaults in ProcessGroupXCCL::initWork under load.
        os.environ.setdefault("TORCH_FR_BUFFER_SIZE", "0")
        # oneCCL / Level Zero need a graceful release or the next server on
        # the same devices hangs in CCL init.
        if vllm_config.shutdown_timeout == 0:
            vllm_config.shutdown_timeout = 5

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return True

    @classmethod
    def get_current_memory_usage(cls, device: torch.types.Device | None = None) -> float:
        # Not the empty_cache/reset_peak/max_allocated triple: empty_cache is a
        # device-wide queue walk (see module docstring).
        return torch.xpu.memory_allocated(device)

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if not torch.distributed.is_xccl_available():
            logger.warning(
                "xccl is not enabled in this torch build; multi-XPU "
                "communication is not available."
            )
        return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"

    @classmethod
    def supports_fp8(cls) -> bool:
        return True

    @classmethod
    def device_count(cls) -> int:
        return torch.xpu.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        if dtype == torch.bfloat16 and "a770" in cls.get_device_name().lower():
            raise ValueError(
                "Intel Arc A770 has a known bfloat16 accuracy issue; use --dtype=half."
            )

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        dst_cache[dst_block_indices] = src_cache[src_block_indices].to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        dst_cache[dst_block_indices] = src_cache[src_block_indices].cpu()

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.xpu.get_device_properties(device_id).max_compute_units

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True
