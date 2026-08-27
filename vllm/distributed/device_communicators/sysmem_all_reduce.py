# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-engine all-reduce over pinned host memory for no-P2P boxes.

GeForce-class multi-GPU hosts have no peer-to-peer, so NCCL falls back to
its SHM transport, which measures ~2.7 GB/s bus bandwidth on 8x RTX 3090
regardless of tuning, while each GPU's DMA engines sustain ~26 GB/s to
pinned host memory in both directions
(perf/results/2026-08-26/qwen38fn-3090-opt/).

This communicator stages a reduce-scatter + all-gather through one shared
pinned /dev/shm segment using only:

- copy-engine transfers (``Tensor.copy_`` between device tensors and pinned
  host views, graph-capturable, full PCIe bandwidth), and
- a one-block Triton barrier kernel (each rank publishes a system-scope
  arrival flag and sweep-polls the packed flag word of all ranks).

Per all-reduce of ``B`` bytes on ``n`` ranks:

  1. D2H: my full input -> my input slot.                     [B]
  2. barrier.
  3. H2D: peer stripes of my 1/n stripe -> staging.           [(n-1)B/n]
  4. local fp32 reduce of n stripes (device memory).
  5. D2H: reduced stripe -> my result slot.                   [B/n]
  6. barrier.
  7. H2D: the other n-1 reduced stripes -> output.            [(n-1)B/n]

DMA completion and stream ordering carry the data-visibility argument: the
barrier kernel launches after the D2H copy completes on the same stream,
and a peer's H2D copies launch after its barrier kernel observes every
rank's flag.

A third barrier closes each all-reduce so no rank can start the next one
(and overwrite its slot) while a straggler still reads this round's data;
the barrier round index is a cumulative device counter advanced by the
kernel itself, so an all-reduce is a fixed op sequence that replays
correctly inside CUDA graphs.

Latency floor is two barrier round-trips (~30-60 us), so small payloads
stay on NCCL; this class only accepts tensors above ``min_bytes``.
"""

import mmap
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

_CUDA_HOST_REGISTER_PORTABLE = 0x01
_CUDA_HOST_REGISTER_MAPPED = 0x02


if HAS_TRITON:

    @triton.jit
    def _sysmem_barrier_kernel(
        flags_ptr_int,  # packed u32[world] flag words (pinned host)
        round_ptr,  # device: cumulative barrier counter
        rank,
        WORLD_SIZE: tl.constexpr,
    ):
        flags = flags_ptr_int.to(tl.pointer_type(tl.uint32))
        rnd = tl.atomic_add(round_ptr, 1, sem="acq_rel") + 1
        tl.atomic_add(flags + rank, 1, sem="release", scope="sys")
        lanes = tl.arange(0, WORLD_SIZE)
        ready = 0
        while ready == 0:
            now = tl.load(flags + lanes, volatile=True)
            if tl.min(now, axis=0) >= rnd:
                ready = 1
        tl.atomic_add(flags + rank, 0, sem="acquire", scope="sys")


class SysmemAllreduce:
    """Reduce-scatter/all-gather all-reduce staged through pinned host RAM."""

    # Entry-data barrier, result barrier, and exit barrier.
    _BARRIERS_PER_AR = 3

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_bytes: int = 48 * 1024 * 1024,
        min_bytes: int = 128 * 1024,
        cpu_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.disabled = True
        if not HAS_TRITON:
            return
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size == 1 or self.world_size & (self.world_size - 1):
            # Power-of-two ranks keep the flag sweep a single vector load.
            return
        node_group = cpu_group if cpu_group is not None else group
        if not all(in_the_same_node_as(node_group, source_rank=0)):
            return
        self.device = device
        self.max_bytes = max_bytes
        self.min_bytes = min_bytes

        # Rank 0 names and creates the segment; everyone maps it.
        slot = max_bytes
        self.slot_bytes = slot
        self.result_slot_bytes = slot // self.world_size
        input_bank = self.world_size * slot
        result_bank = self.world_size * self.result_slot_bytes
        self.input_bank_bytes = input_bank
        self.result_bank_bytes = result_bank
        flags_off = input_bank + result_bank
        seg_bytes = flags_off + 4096

        if self.rank == 0:
            name = f"/dev/shm/vllm_sysmem_ar_{os.getpid()}"
            with open(name, "wb") as f:
                f.truncate(seg_bytes)
            payload = [name]
        else:
            payload = [None]
        dist.broadcast_object_list(payload, src=dist.get_global_rank(group, 0),
                                   group=group)
        self.shm_path = payload[0]
        fd = os.open(self.shm_path, os.O_RDWR)
        self._mmap = mmap.mmap(fd, seg_bytes)
        os.close(fd)
        host = torch.frombuffer(self._mmap, dtype=torch.uint8)
        rc = torch.cuda.cudart().cudaHostRegister(
            host.data_ptr(),
            seg_bytes,
            _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED,
        )
        if rc != 0:
            logger.warning("SysmemAllreduce: cudaHostRegister failed (%s)", rc)
            return
        self._host = host
        if self.rank == 0:
            host.zero_()
        dist.barrier(group=group)
        # The registration examined above keeps the segment alive; unlink so
        # it disappears with the processes.
        if self.rank == 0:
            try:
                os.unlink(self.shm_path)
            except OSError:
                pass

        self.flags_off = flags_off
        self.round = torch.zeros(1, dtype=torch.int32, device=device)
        # Device staging for peer stripes (input and result phases).
        self._staging = torch.empty(
            (self.world_size - 1) * (slot // self.world_size),
            dtype=torch.uint8,
            device=device,
        )
        self.disabled = False
        logger.info_once(
            "SysmemAllreduce enabled: world=%d, slot=%d MiB, min=%d KiB, "
            "segment=%s",
            self.world_size,
            slot >> 20,
            min_bytes >> 10,
            self.shm_path,
        )

    def _barrier(self) -> None:
        _sysmem_barrier_kernel[(1,)](
            self._host.data_ptr() + self.flags_off,
            self.round,
            self.rank,
            WORLD_SIZE=self.world_size,
            num_warps=1,
        )

    def _bank_views(self, nbytes: int, stripe: int):
        base = 0
        res_base = self.input_bank_bytes
        inp = [
            self._host[base + r * self.slot_bytes:
                       base + r * self.slot_bytes + nbytes]
            for r in range(self.world_size)
        ]
        res = [
            self._host[res_base + r * self.result_slot_bytes:
                       res_base + r * self.result_slot_bytes + stripe]
            for r in range(self.world_size)
        ]
        return inp, res

    def should_sysmem_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        nbytes = inp.numel() * inp.element_size()
        return (
            self.min_bytes <= nbytes <= self.max_bytes
            and inp.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and inp.is_contiguous()
            # Stripes must be element-aligned for every rank.
            and inp.numel() % self.world_size == 0
        )

    def all_reduce(
        self, inp: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(inp)
        world = self.world_size
        rank = self.rank
        flat_in = inp.view(-1).view(torch.uint8)
        flat_out = out.view(-1).view(torch.uint8)
        nbytes = flat_in.numel()
        stripe = nbytes // world

        inp_slots, res_slots = self._bank_views(nbytes, stripe)

        # 1. my input -> my slot (D2H copy engine)
        inp_slots[rank].copy_(flat_in, non_blocking=True)
        # 2. all ranks' inputs visible
        self._barrier()
        # 3. peer slices of my stripe -> staging (H2D)
        lo = rank * stripe
        staging = self._staging[: (world - 1) * stripe].view(world - 1, stripe)
        idx = 0
        for peer in range(world):
            if peer == rank:
                continue
            staging[idx].copy_(inp_slots[peer][lo: lo + stripe], non_blocking=True)
            idx += 1
        # 4. local fp32 reduce of my stripe
        elem = inp.element_size()
        st_lo = lo // elem
        st_hi = (lo + stripe) // elem
        mine = inp.view(-1)[st_lo:st_hi]
        peers = staging.view(inp.dtype).view(world - 1, st_hi - st_lo)
        reduced = (mine.to(torch.float32) + peers.to(torch.float32).sum(0)).to(
            inp.dtype
        )
        out.view(-1)[st_lo:st_hi].copy_(reduced)
        # 5. reduced stripe -> my result slot (D2H)
        res_slots[rank].copy_(reduced.view(torch.uint8), non_blocking=True)
        # 6. all reduced stripes visible
        self._barrier()
        # 7. other ranks' reduced stripes -> out (H2D)
        for peer in range(world):
            if peer == rank:
                continue
            p_lo = peer * stripe
            flat_out[p_lo: p_lo + stripe].copy_(res_slots[peer], non_blocking=True)
        # 8. exit barrier: nobody starts the next round (overwriting slots)
        #    until every rank has finished reading this one.
        self._barrier()
        return out

    def close(self) -> None:
        if getattr(self, "_host", None) is not None:
            try:
                torch.cuda.cudart().cudaHostUnregister(self._host.data_ptr())
            except Exception:
                pass
            self._host = None
