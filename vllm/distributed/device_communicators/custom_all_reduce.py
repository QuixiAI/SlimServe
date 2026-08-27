# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

logger = init_logger(__name__)


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            # can_device_access_peer takes visible device ordinals, while
            # rank and i are logical local IDs.
            return torch.cuda.can_device_access_peer(
                current_platform.logical_device_id_to_visible_device_id(rank),
                current_platform.logical_device_id_to_visible_device_id(i),
            )
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


from vllm.distributed.utils import is_weak_contiguous  # noqa: E402


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=8192 * 1024,
        symm_mem_enabled=False,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES:
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        # device.index is a visible ordinal, not a logical local ID.
        physical_device_id = current_platform.visible_device_id_to_physical_device_id(
            device.index
        )
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        if world_size > 2 and not fully_connected:
            if not envs.VLLM_CUSTOM_AR_ALLOW_PCIE:
                logger.warning(
                    "Custom allreduce is disabled because it's not supported on"
                    " more than two PCIe-only GPUs. To silence this warning, "
                    "specify disable_custom_all_reduce=True explicitly. If this"
                    " box has working PCIe P2P with a full-size BAR1, set"
                    " VLLM_CUSTOM_AR_ALLOW_PCIE=1 to enable it."
                )
                return
            logger.info_once(
                "Custom allreduce enabled on a PCIe-only topology"
                " (VLLM_CUSTOM_AR_ALLOW_PCIE=1); P2P must be functional"
                " for this to be correct."
            )
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(
            ops.meta_size() + max_size, group=group, uncached=True
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            if not self.disabled:
                # Drain eager warmup work before capture. CUDA forbids a graph
                # from waiting on an event recorded by uncaptured work.
                ops.wait_dsv4_mhc(self._ptr, self.rank_data)
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL -- unless the platform has
        # real PCIe P2P (large-BAR1 patched driver) and the user opted in.
        if (
            self.world_size == 2
            or self.fully_connected
            or envs.VLLM_CUSTOM_AR_ALLOW_PCIE
        ):
            return inp_size < self.max_size
        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def should_fuse_ar_norm(self, inp: torch.Tensor):
        """Whether the fused allreduce + residual add + RMSNorm kernel
        applies. It is one-shot only, so restrict to message sizes where
        the one-shot algorithm would have been picked anyway."""
        if not self.should_custom_ar(inp):
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16):
            return False
        inp_size = inp.numel() * inp.element_size()
        if self.world_size == 2:
            return True
        # one-shot/two-shot crossover in custom_all_reduce.cuh
        limit = 512 * 1024 if self.world_size <= 4 else 256 * 1024
        return inp_size < limit

    def all_reduce_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        *,
        registered: bool = False,
    ):
        """Out-of-place fused op: residual += allreduce(inp);
        returns rmsnorm(residual) * weight."""
        out = torch.empty_like(inp)
        if registered:
            ops.all_reduce_add_rms_norm(
                self._ptr, inp, residual, weight, out, epsilon, 0, 0
            )
        else:
            ops.all_reduce_add_rms_norm(
                self._ptr,
                inp,
                residual,
                weight,
                out,
                epsilon,
                self.buffer_ptrs[self.rank],
                self.max_size,
            )
        return out

    def fused_all_reduce_add_rms_norm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor | None:
        """Cuda-graph-aware entry for the fused kernel. Returns None when
        the fused path does not apply and the caller must fall back to
        allreduce followed by fused_add_rms_norm."""
        if self.disabled or not self.should_fuse_ar_norm(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce_add_rms_norm(
                    input, residual, weight, epsilon, registered=True
                )
            else:
                # Warmup only mimics the allocation pattern.
                return torch.empty_like(input)
        else:
            return self.all_reduce_add_rms_norm(
                input, residual, weight, epsilon, registered=False
            )

    def should_fuse_dsv4_mhc(
        self, inp: torch.Tensor, residual: torch.Tensor
    ) -> bool:
        local_hidden = 4096 // self.world_size
        return (
            self.should_custom_ar(inp)
            and self.world_size in (2, 4, 8)
            and inp.dtype == torch.bfloat16
            and inp.shape in ((1, 4096), (1, local_hidden))
            and residual.dtype == torch.bfloat16
            and residual.shape == (1, 4, 4096)
        )

    def all_reduce_dsv4_mhc(
        self,
        inp: torch.Tensor,
        addend: torch.Tensor | None,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_multiplier: float,
        sinkhorn_repeat: int,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        input_prepared: bool = False,
        own_projections: bool = False,
        publish_prepared: bool = False,
        local_input_owned: bool = False,
        *,
        registered: bool = False,
        _buffers: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
        ]
        | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if local_input_owned and (norm_weight is None or not own_projections):
            raise ValueError(
                "DSV4 local input ownership requires fused norm and owned projections"
            )
        if _buffers is None:
            _buffers = self._allocate_dsv4_mhc_buffers(
                inp, residual, norm_weight, local_input_owned
            )
        residual_out, partial, next_post, next_comb, layer_input, quant_input = (
            _buffers
        )
        reg_buffer = 0 if registered else self.buffer_ptrs[self.rank]
        reg_buffer_size = 0 if registered else self.max_size
        ops.all_reduce_dsv4_mhc(
            self._ptr,
            inp,
            addend,
            residual,
            post_mix,
            comb_mix,
            fn,
            residual_out,
            partial,
            scale,
            base,
            next_post,
            next_comb,
            layer_input,
            norm_weight,
            quant_input,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_repeat,
            norm_eps,
            input_prepared,
            own_projections,
            publish_prepared,
            reg_buffer,
            reg_buffer_size,
        )
        if quant_input is not None:
            return residual_out, next_post, next_comb, layer_input, quant_input
        return residual_out, next_post, next_comb, layer_input

    def _allocate_dsv4_mhc_buffers(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor | None,
        local_input_owned: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        residual_out = torch.empty_like(residual)
        # next_post and next_comb retain the deferred projection workspace.
        partial = torch.empty((1, 32, 25), dtype=torch.float32, device=inp.device)
        partial_flat = partial.flatten()
        next_post = partial_flat[:4].view(1, 4, 1)
        next_comb = partial_flat[4:20].view(1, 4, 4)
        layer_hidden = 4096 // self.world_size if local_input_owned else 4096
        layer_input = torch.empty(
            (1, layer_hidden), dtype=inp.dtype, device=inp.device
        )
        quant_input = (
            torch.empty(
                (1, layer_hidden // 32 * 9),
                dtype=torch.int32,
                device=inp.device,
            )
            if norm_weight is not None
            else None
        )
        return (
            residual_out,
            partial,
            next_post,
            next_comb,
            layer_input,
            quant_input,
        )

    def _breakable_all_reduce_dsv4_mhc(
        self, args: tuple[object, ...]
    ) -> tuple[torch.Tensor, ...] | None:
        """Put async mHC outside each breakable graph segment.

        The deferred mHC stream can then overlap the following captured layer.
        The next eager mHC launch performs the existing event wait before it
        starts, preserving the original async schedule without leaving an
        auxiliary stream inside a CUDA capture at segment end.
        """
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.utils.torch_utils import weak_ref_tensor

        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return None

        inp = cast(torch.Tensor, args[0])
        residual = cast(torch.Tensor, args[2])
        norm_weight = cast(torch.Tensor | None, args[13])
        local_input_owned = cast(bool, args[18])
        buffers = self._allocate_dsv4_mhc_buffers(
            inp, residual, norm_weight, local_input_owned
        )
        weak_args = tuple(weak_ref_tensor(arg) for arg in args)
        weak_buffers = cast(
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ],
            tuple(weak_ref_tensor(buffer) for buffer in buffers),
        )
        capture.add_eager(
            lambda: self.all_reduce_dsv4_mhc(
                *weak_args, registered=False, _buffers=weak_buffers
            )
        )
        residual_out, _, next_post, next_comb, layer_input, quant_input = buffers
        if quant_input is not None:
            return residual_out, next_post, next_comb, layer_input, quant_input
        return residual_out, next_post, next_comb, layer_input

    def dsv4_channel_owned_mhc(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_multiplier: float,
        sinkhorn_repeat: int,
        norm_eps: float,
        addend: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Reduce into channel ownership and apply the DSV4 mHC transition."""
        if self.disabled or self.world_size not in (2, 4, 8):
            raise RuntimeError("DSV4 channel ownership requires TP2, TP4, or TP8")
        local_hidden = 4096 // self.world_size
        full_input = inp.shape == (1, 4096)
        if inp.shape not in ((1, local_hidden), (1, 4096)) or residual.shape != (
            1,
            4,
            local_hidden,
        ):
            raise ValueError("invalid DSV4 channel-owned activation shape")
        if addend is not None and (not full_input or addend.shape != inp.shape):
            raise ValueError("DSV4 ownership addend must match a full-H input")

        residual_out = torch.empty_like(residual)
        partial = torch.empty(320, dtype=torch.float32, device=inp.device)
        next_post = torch.empty((1, 4, 1), dtype=torch.float32, device=inp.device)
        next_comb = torch.empty((1, 4, 4), dtype=torch.float32, device=inp.device)
        layer_input = torch.empty(
            (1, local_hidden), dtype=inp.dtype, device=inp.device
        )
        quant_input = torch.empty(
            (1, local_hidden // 32 * 9), dtype=torch.int32, device=inp.device
        )
        stream_capturing = torch.cuda.is_current_stream_capturing()
        registered = self._IS_CAPTURING and stream_capturing
        if full_input and self._IS_CAPTURING and not stream_capturing:
            # Graph warmup establishes allocation and shape flow before the
            # producer tensors have graph-registered peer addresses. Match the
            # allocations above and let the immediately following capture run
            # execute the direct peer reduce-scatter.
            return (
                residual_out,
                next_post,
                next_comb,
                layer_input,
                quant_input,
                partial,
            )
        if full_input and not registered:
            raise RuntimeError(
                "full-H DSV4 ownership transition requires CUDA graph capture"
            )
        ops.dsv4_channel_owned_mhc(
            self._ptr,
            inp,
            addend,
            residual,
            post_mix,
            comb_mix,
            fn,
            residual_out,
            partial,
            scale,
            base,
            next_post,
            next_comb,
            layer_input,
            norm_weight,
            quant_input,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_repeat,
            norm_eps,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return residual_out, next_post, next_comb, layer_input, quant_input, partial

    def dsv4_channel_owned_q2_down(
        self,
        quant_mid: torch.Tensor,
        weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply output-stationary W2 to peer Q8_1 intermediate shards."""
        if self.disabled or self.world_size not in (2, 4):
            raise RuntimeError("DSV4 channel-owned Q2_K requires TP2 or TP4")
        output = torch.empty(
            (1, 4096 // self.world_size),
            dtype=torch.bfloat16,
            device=quant_mid.device,
        )
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        ops.dsv4_channel_owned_q2_down(
            self._ptr,
            quant_mid,
            weights,
            topk_ids,
            output,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return output

    def dsv4_owned_attention_projections(
        self,
        local_input: torch.Tensor,
        local_quant: torch.Tensor,
        aligned_q8_weight: torch.Tensor,
        bf16_weight0: torch.Tensor | None,
        bf16_weight1: torch.Tensor | None,
        bf16_weight2: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project an owned DSV4 input and reduce only compact outputs."""
        if self.disabled or self.world_size not in (2, 4, 8):
            raise RuntimeError("DSV4 input ownership requires TP2, TP4, or TP8")
        local_hidden = 4096 // self.world_size
        if local_input.shape != (1, local_hidden):
            raise ValueError("invalid DSV4 input-owned attention shape")
        empty_weight = torch.empty(
            (0, 4096), dtype=torch.bfloat16, device=local_input.device
        )
        weights = tuple(
            weight if weight is not None else empty_weight
            for weight in (bf16_weight0, bf16_weight1, bf16_weight2)
        )
        q8_rows = aligned_q8_weight.shape[0]
        rows = tuple(weight.shape[0] for weight in weights)
        q8_output = torch.empty(
            (1, q8_rows), dtype=torch.bfloat16, device=local_input.device
        )
        bf16_output0 = torch.empty(
            (1, rows[0]), dtype=torch.float32, device=local_input.device
        )
        bf16_output1 = torch.empty(
            (1, rows[1]), dtype=torch.bfloat16, device=local_input.device
        )
        bf16_output2 = torch.empty(
            (1, rows[2]), dtype=torch.float32, device=local_input.device
        )
        total = q8_rows + sum(rows)
        partial = torch.empty(total, dtype=torch.float32, device=local_input.device)
        reduced = torch.empty_like(partial)
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        ops.dsv4_owned_attention_projections(
            self._ptr,
            local_input,
            local_quant,
            aligned_q8_weight,
            *weights,
            q8_output,
            bf16_output0,
            bf16_output1,
            bf16_output2,
            partial,
            reduced,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return q8_output, bf16_output0, bf16_output1, bf16_output2

    def dsv4_gather_owned_q8(self, local_quant: torch.Tensor) -> torch.Tensor:
        """Gather one channel-owned Q8_1 shard from every TP rank."""
        if self.disabled or self.world_size not in (2, 4, 8):
            raise RuntimeError("DSV4 Q8 gather requires TP2, TP4, or TP8")
        output = torch.empty(
            (local_quant.shape[0], 4096 // 32 * 9),
            dtype=torch.int32,
            device=local_quant.device,
        )
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        ops.dsv4_gather_owned_q8(
            self._ptr,
            local_quant,
            output,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return output

    def dsv4_gather_owned_bf16(self, local_input: torch.Tensor) -> torch.Tensor:
        """Gather channel-owned BF16 rows for the final model boundary."""
        output_shape = (*local_input.shape[:-1], 4096)
        output = torch.empty(
            output_shape, dtype=torch.bfloat16, device=local_input.device
        )
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        ops.dsv4_gather_owned_bf16(
            self._ptr,
            local_input,
            output,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return output

    def dsv4_owned_router(
        self, local_input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Apply the DSV4 router to channel-owned input and reduce logits."""
        output = torch.empty(
            (local_input.shape[0], weight.shape[0]),
            dtype=torch.float32,
            device=local_input.device,
        )
        partial = torch.empty_like(output)
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        ops.dsv4_owned_router(
            self._ptr,
            local_input,
            weight,
            output,
            partial,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return output

    def dsv4_channel_owned_q2_down_pending(
        self,
        pending: torch.Tensor,
        addend: torch.Tensor,
    ) -> torch.Tensor:
        """Finish native W2 into owned hidden rows and fold shared TP parts."""
        if self.disabled or self.world_size not in (2, 4):
            raise RuntimeError("DSV4 channel-owned pending down requires TP2 or TP4")
        output = torch.empty(
            (1, 4096 // self.world_size),
            dtype=torch.bfloat16,
            device=pending.device,
        )
        # Six full Q8_1 routes followed by the FP32 sum of the owned shared
        # rows. This is small, graph-stable, and avoids per-output-CTA P2P loads.
        scratch = torch.empty(
            6 * (2048 // 32) * 36 + (4096 // self.world_size) * 4,
            dtype=torch.uint8,
            device=pending.device,
        )
        ops.dsv4_channel_owned_q2_down_pending(
            self._ptr,
            pending,
            addend,
            scratch,
            output,
            self.buffer_ptrs[self.rank],
            self.max_size,
        )
        return output

    def dsv4_channel_owned_moe(
        self,
        quant_input: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        addend: torch.Tensor,
        swiglu_limit: float = 7.0,
    ) -> torch.Tensor:
        """Run producer-owned IQ2_XXS/SwiGLU/Q2_K into owned hidden rows."""
        if self.disabled or self.world_size not in (2, 4):
            raise RuntimeError("DSV4 channel-owned MoE requires TP2 or TP4")
        local_rows = 4096 // self.world_size
        output = torch.empty(
            (1, local_rows), dtype=torch.bfloat16, device=quant_input.device
        )
        scratch = torch.empty(
            local_rows * 4, dtype=torch.uint8, device=quant_input.device
        )
        ops.dsv4_channel_owned_moe(
            self._ptr,
            quant_input,
            w1,
            w2,
            topk_weights,
            topk_ids,
            addend,
            scratch,
            output,
            swiglu_limit,
            self.buffer_ptrs[self.rank],
            self.max_size,
        )
        return output

    def dsv4_output_owned_moe(
        self,
        quant_input: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_quant: torch.Tensor,
        shared_w2: torch.Tensor,
        swiglu_limit: float = 7.0,
    ) -> torch.Tensor:
        """Run routed and shared experts directly into owned hidden rows."""
        if self.disabled or self.world_size not in (2, 4):
            raise RuntimeError("DSV4 output-owned MoE requires TP2 or TP4")
        output = torch.empty(
            (1, 4096 // self.world_size),
            dtype=torch.bfloat16,
            device=quant_input.device,
        )
        ops.dsv4_output_owned_moe(
            self._ptr,
            quant_input,
            w1,
            w2,
            topk_weights,
            topk_ids,
            shared_quant,
            shared_w2,
            output,
            swiglu_limit,
            self.buffer_ptrs[self.rank],
            self.max_size,
        )
        return output

    def dsv4_output_owned_q8(
        self,
        local_quant: torch.Tensor,
        aligned_weight: torch.Tensor,
        rows_per_cta: int = 2,
    ) -> torch.Tensor:
        """Publish compact Q8 K-shards and compute only this rank's H rows."""
        if self.disabled or self.world_size not in (2, 4, 8):
            raise RuntimeError("DSV4 output-owned Q8 requires TP2, TP4, or TP8")
        output = torch.empty(
            (local_quant.shape[0], 4096 // self.world_size),
            dtype=torch.bfloat16,
            device=local_quant.device,
        )
        ops.dsv4_output_owned_q8(
            self._ptr,
            local_quant,
            aligned_weight,
            output,
            rows_per_cta,
            self.buffer_ptrs[self.rank],
            self.max_size,
        )
        return output

    def dsv4_owned_reduce_scatter(
        self,
        input: torch.Tensor,
        addend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce full-H local partials into this rank's owned H/TP rows."""
        if self.disabled or self.world_size not in (2, 4, 8):
            raise RuntimeError(
                "DSV4 owned reduce-scatter requires TP2, TP4, or TP8"
            )
        output = torch.empty(
            (input.shape[0], 4096 // self.world_size),
            dtype=torch.bfloat16,
            device=input.device,
        )
        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        ops.dsv4_owned_reduce_scatter(
            self._ptr,
            input,
            addend,
            output,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return output

    def fused_all_reduce_dsv4_mhc(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_multiplier: float,
        sinkhorn_repeat: int,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        *,
        input_prepared: bool = False,
        own_projections: bool = False,
        publish_prepared: bool = False,
        local_input_owned: bool = False,
    ) -> tuple[torch.Tensor, ...] | None:
        if self.disabled or not self.should_fuse_dsv4_mhc(inp, residual):
            return None
        args = (
            inp,
            None,
            residual,
            post_mix,
            comb_mix,
            fn,
            scale,
            base,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
            input_prepared,
            own_projections,
            publish_prepared,
            local_input_owned,
        )
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                breakable_output = self._breakable_all_reduce_dsv4_mhc(args)
                if breakable_output is not None:
                    return breakable_output
                return self.all_reduce_dsv4_mhc(*args, registered=True)
            layer_hidden = 4096 // self.world_size if local_input_owned else 4096
            outputs = (
                torch.empty_like(residual),
                torch.empty((1, 4, 1), dtype=torch.float32, device=inp.device),
                torch.empty((1, 4, 4), dtype=torch.float32, device=inp.device),
                torch.empty((1, layer_hidden), dtype=inp.dtype, device=inp.device),
            )
            if norm_weight is not None:
                return outputs + (
                    torch.empty(
                        (1, layer_hidden // 32 * 9),
                        dtype=torch.int32,
                        device=inp.device,
                    ),
                )
            return outputs
        return self.all_reduce_dsv4_mhc(*args, registered=False)

    def all_reduce_dsv4_q2_mhc(
        self,
        pending: torch.Tensor,
        addend: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_multiplier: float,
        sinkhorn_repeat: int,
        norm_weight: torch.Tensor,
        norm_eps: float,
        *,
        registered: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual_out = torch.empty_like(residual)
        # 16 logical reduction slots plus 128 exact-order physical warp
        # partials. The final 160 values remain spare for norm scratch.
        partial_values = 48 * 25
        producer_values = 4096 * pending.element_size() // 4
        workspace = torch.empty(
            partial_values + producer_values,
            dtype=torch.float32,
            device=pending.device,
        )
        partial_flat = workspace[:partial_values]
        partial = partial_flat.view(1, 48, 25)
        producer_output = workspace[partial_values:].view(torch.bfloat16).view(
            1, 4096
        )
        next_post = partial_flat[:4].view(1, 4, 1)
        next_comb = partial_flat[4:20].view(1, 4, 4)
        layer_input = torch.empty_like(pending)
        quant_input = torch.empty(
            (1, 4096 // 32 * 9), dtype=torch.int32, device=pending.device
        )
        ops.all_reduce_dsv4_q2_mhc(
            self._ptr,
            pending,
            addend,
            producer_output,
            residual,
            post_mix,
            comb_mix,
            fn,
            residual_out,
            partial,
            scale,
            base,
            next_post,
            next_comb,
            layer_input,
            norm_weight,
            quant_input,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_repeat,
            norm_eps,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return residual_out, next_post, next_comb, layer_input, quant_input

    def fused_all_reduce_dsv4_q2_mhc(
        self,
        pending: torch.Tensor,
        addend: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_multiplier: float,
        sinkhorn_repeat: int,
        norm_weight: torch.Tensor | None,
        norm_eps: float,
    ) -> tuple[torch.Tensor, ...] | None:
        if (
            self.disabled
            or self.world_size not in (2, 4)
            or norm_weight is None
            or pending.dtype != torch.bfloat16
            or pending.shape != (1, 4096)
            or addend.dtype != pending.dtype
            or addend.shape != pending.shape
            or residual.dtype != pending.dtype
            or residual.shape != (1, 4, 4096)
        ):
            return None
        args = (
            pending,
            addend,
            residual,
            post_mix,
            comb_mix,
            fn,
            scale,
            base,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce_dsv4_q2_mhc(*args, registered=True)
            return (
                torch.empty_like(residual),
                torch.empty((1, 4, 1), dtype=torch.float32, device=pending.device),
                torch.empty((1, 4, 4), dtype=torch.float32, device=pending.device),
                torch.empty_like(pending),
                torch.empty(
                    (1, 4096 // 32 * 9),
                    dtype=torch.int32,
                    device=pending.device,
                ),
            )
        return self.all_reduce_dsv4_q2_mhc(*args, registered=False)

    def wait_dsv4_mhc(self, anchor: torch.Tensor) -> None:
        if self.disabled:
            return
        if self._IS_CAPTURING and torch.cuda.is_current_stream_capturing():
            from vllm.compilation.breakable_cudagraph import (
                BreakableCUDAGraphCapture,
            )
            from vllm.utils.torch_utils import weak_ref_tensor

            capture = BreakableCUDAGraphCapture.current()
            if capture is not None and capture._capturing:
                weak_anchor = weak_ref_tensor(anchor)
                capture.add_eager(lambda: ops.wait_dsv4_mhc(self._ptr, weak_anchor))
                return
        ops.wait_dsv4_mhc(self._ptr, anchor)

    def dsv4_indexer_topk(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor,
        k: int,
        max_seq_len: int,
    ) -> None:
        """Select top-k from rank-local head partials without materializing AR."""
        if self.disabled or self.world_size not in (2, 4, 8):
            raise RuntimeError("DSV4 peer top-k requires custom all-reduce")

        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        ops.dsv4_indexer_peer_topk(
            self._ptr,
            logits,
            lengths,
            output,
            workspace,
            k,
            max_seq_len,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )

    def dsv4_indexer_token_merge(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        local_indices: torch.Tensor,
        output: torch.Tensor,
        k: int,
    ) -> None:
        """Merge rank-local token-shard candidates directly over peer IPC."""
        if self.disabled or self.world_size not in (2, 4):
            raise RuntimeError("DSV4 token-shard merge requires TP2 or TP4 custom AR")

        registered = self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        if self._IS_CAPTURING and not registered:
            output.copy_(local_indices)
            return

        ops.dsv4_indexer_token_merge(
            self._ptr,
            logits,
            lengths,
            local_indices,
            output,
            k,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )

    def fused_all_reduce_dsv4_mhc_add(
        self,
        inp: torch.Tensor,
        addend: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        rms_eps: float,
        pre_eps: float,
        sinkhorn_eps: float,
        post_multiplier: float,
        sinkhorn_repeat: int,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        *,
        input_prepared: bool = False,
        own_projections: bool = False,
        publish_prepared: bool = False,
        local_input_owned: bool = False,
    ) -> tuple[torch.Tensor, ...] | None:
        """Graph-aware local BF16 add + all-reduce + DSV4 mHC transition."""
        if (
            self.disabled
            or not self.should_fuse_dsv4_mhc(inp, residual)
            or addend.dtype != inp.dtype
            or addend.shape != inp.shape
        ):
            return None
        args = (
            inp,
            addend,
            residual,
            post_mix,
            comb_mix,
            fn,
            scale,
            base,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
            input_prepared,
            own_projections,
            publish_prepared,
            local_input_owned,
        )
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                breakable_output = self._breakable_all_reduce_dsv4_mhc(args)
                if breakable_output is not None:
                    return breakable_output
                return self.all_reduce_dsv4_mhc(*args, registered=True)
            layer_hidden = 4096 // self.world_size if local_input_owned else 4096
            outputs = (
                torch.empty_like(residual),
                torch.empty((1, 4, 1), dtype=torch.float32, device=inp.device),
                torch.empty((1, 4, 4), dtype=torch.float32, device=inp.device),
                torch.empty((1, layer_hidden), dtype=inp.dtype, device=inp.device),
            )
            if norm_weight is not None:
                return outputs + (
                    torch.empty(
                        (1, layer_hidden // 32 * 9),
                        dtype=torch.int32,
                        device=inp.device,
                    ),
                )
            return outputs
        return self.all_reduce_dsv4_mhc(*args, registered=False)

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
