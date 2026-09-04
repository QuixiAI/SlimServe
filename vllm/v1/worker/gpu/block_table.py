# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from functools import cache

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter,
    StagedWriteTensor,
    UvaBackedTensor,
    _load_ptr,
)


def _quixicore_disabled() -> bool:
    """VLLM_QC_DISABLE_NATIVE=1 forces the Triton path.

    Exists so the native and Triton routes can be compared on an otherwise
    identical server; the kernels are bitwise-equal, so greedy output must
    match token for token.
    """
    import os

    return os.environ.get("VLLM_QC_DISABLE_NATIVE") == "1"


@cache
def _integrity_checks_enabled() -> bool:
    """VLLM_KV_INTEGRITY_CHECK=1 validates slot mappings before they are used.

    Paged attention is all bookkeeping: a slot mapping that addresses the wrong
    page returns another request's tokens, and the model stays fluent while
    doing it, so the failure reads as a quality problem rather than a bug. This
    turns that class into a loud abort. Off by default -- the check costs a
    sort per step, and correct bookkeeping is the normal case.
    """
    import os

    return os.environ.get("VLLM_KV_INTEGRITY_CHECK") == "1"


def _check_slot_mappings(slot_mappings: torch.Tensor, num_tokens: int) -> None:
    """Every token this step writes must own a distinct, real KV slot.

    Catches aliasing (two tokens handed the same physical slot, so one
    overwrites the other) and negative indices that are not the pad sentinel.
    It does not catch a slot that is individually plausible but stale -- that
    needs a generation tag on each page, which has to live in the allocator.
    """
    live = slot_mappings[:, :num_tokens]
    for group, mapping in enumerate(live):
        valid = mapping[mapping != PAD_SLOT_ID]
        if valid.numel() == 0:
            continue
        if bool((valid < 0).any()):
            bad = valid[valid < 0][:8].tolist()
            raise RuntimeError(
                f"KV integrity: group {group} produced negative slots {bad}, "
                f"which are not the pad sentinel ({PAD_SLOT_ID})"
            )
        unique = torch.unique(valid)
        if unique.numel() != valid.numel():
            counts = torch.bincount(valid - int(valid.min()))
            worst = int(torch.argmax(counts)) + int(valid.min())
            raise RuntimeError(
                f"KV integrity: group {group} mapped {valid.numel()} tokens "
                f"onto {unique.numel()} distinct slots; slot {worst} is "
                f"claimed {int(counts.max())} times in one step"
            )


@cache
def _use_native(op_name: str) -> bool:
    """Prefer the native block-table kernel over the Triton one."""
    if _quixicore_disabled():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return False
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.is_available() and quixicore_ops.has(op_name)


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int],
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
        slot_mapping_enabled: list[bool] | None = None,
    ):
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        if slot_mapping_enabled is None:
            slot_mapping_enabled = [True] * self.num_kv_cache_groups
        assert len(slot_mapping_enabled) == self.num_kv_cache_groups
        self._slot_mapping_enabled = slot_mapping_enabled

        self.blocks_per_kv_block = [
            bs // kbs for bs, kbs in zip(block_sizes, kernel_block_sizes)
        ]

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            max_num_blocks = max_num_blocks_per_group[i] * self.blocks_per_kv_block[i]
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)

        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )
        self.fused_writer: FusedStagedWriter | None = None
        if self.num_kv_cache_groups > 1:
            # Only the multi-group path uses the fused writer.
            self.fused_writer = FusedStagedWriter(
                self.device, self.num_kv_cache_groups * self.max_num_reqs
            )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        if self.device.type == "mps":
            # Metal paths address the tensors directly and never dereference
            # device-side pointer tables. Keep a shape-compatible placeholder.
            return torch.zeros(len(list(x)), dtype=torch.int64, device=self.device)
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        # Called at init and after a CuMem kv_cache wake-up. The ptr tensors
        # cache raw data_ptr() values that go stale once the underlying tensors
        # are reallocated on wake; block_sizes_tensor needs re-populating
        # because its storage lives under the kv_cache pool tag and comes back
        # with undefined contents.
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.slot_mapping_enabled = torch.tensor(
            self._slot_mapping_enabled, dtype=torch.bool, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            bpk = self.blocks_per_kv_block[i]
            if bpk > 1:
                block_ids = [b * bpk + k for b in block_ids for k in range(bpk)]
            self.block_tables[i].stage_write(req_index, start, block_ids)
            self.num_blocks.np[i, req_index] = start + len(block_ids)

    def apply_staged_writes(self) -> None:
        if self.num_kv_cache_groups == 1:
            # Single group: write directly, skipping the per-write group lookup.
            self.block_tables[0].apply_write()
        else:
            # Multiple groups: apply all block tables with one fused kernel.
            assert self.fused_writer is not None
            self.fused_writer.apply(
                self.block_tables, self.block_table_ptrs, self.block_table_strides
            )
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if out is None:
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups
        num_reqs = idx_mapping.shape[0]
        # Launch kernel with num_reqs_padded to fuse zeroing of padded rows.
        if self.device.type == "mps":
            mapped = idx_mapping.to(torch.int64)
            for src, dst in zip(self.block_tables, out):
                dst.zero_()
                dst[:num_reqs].copy_(src.gpu.index_select(0, mapped))
        elif _use_native("gather_block_tables"):
            from vllm.quixicore import quixicore_ops

            quixicore_ops.gather_block_tables(
                idx_mapping,
                self.block_table_ptrs,
                out_ptrs,
                self.block_table_strides,
                self.num_blocks.gpu,
                self.num_blocks.gpu.stride(0),
                num_reqs,
                num_reqs_padded,
            )
        else:
            _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs_padded)](
                idx_mapping,
                self.block_table_ptrs,
                out_ptrs,
                self.block_table_strides,
                self.num_blocks.gpu,
                self.num_blocks.gpu.stride(0),
                num_reqs,
                BLOCK_SIZE=1024,  # type: ignore
            )
        if _integrity_checks_enabled() and self.device.type != "mps":
            self._check_gathered_block_tables(idx_mapping, out, num_reqs)
        return tuple(bt[:num_reqs_padded] for bt in out)

    def _check_gathered_block_tables(self, idx_mapping, out, num_reqs) -> None:
        """VLLM_KV_INTEGRITY_CHECK=1: the gathered per-batch block tables must
        equal the per-request source rows for the live prefix, and the live
        prefix length the GPU used (UVA num_blocks) must match the CPU's.
        A stale prefix length silently truncates every downstream reader
        (indexer gather, slot mapping, attention) at a request-varying point
        - the 2026-09-02 high-block garbling signature."""
        torch.cuda.synchronize()
        idx = idx_mapping[:num_reqs].tolist()
        for g in range(self.num_kv_cache_groups):
            src = self.block_tables[g].gpu
            for b, ridx in enumerate(idx):
                if ridx < 0:
                    continue
                nb_cpu = int(self.num_blocks.np[g, ridx])
                nb_gpu = int(self.num_blocks.gpu[g, ridx])
                if nb_cpu != nb_gpu:
                    raise RuntimeError(
                        f"KV integrity: group {g} req_idx {ridx}: UVA num_blocks "
                        f"{nb_gpu} != CPU {nb_cpu} (stale UVA slot)"
                    )
                if nb_cpu == 0:
                    continue
                got = out[g][b, :nb_cpu]
                exp = src[ridx, :nb_cpu]
                if not torch.equal(got, exp):
                    bad = (got != exp).nonzero().flatten()[:8].tolist()
                    raise RuntimeError(
                        f"KV integrity: group {g} batch row {b} (req_idx {ridx}) "
                        f"gathered block table differs from source at entries "
                        f"{bad} of {nb_cpu}: got {got[bad].tolist()} exp "
                        f"{exp[bad].tolist()}"
                    )

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
        num_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        slot_mappings = self.slot_mappings if out is None else out
        if self.device.type == "mps":
            from vllm.v1.worker.gpu.input_batch import mps_segment_ids

            slot_mappings.fill_(PAD_SLOT_ID)
            actual_num_tokens = (
                num_tokens
                if num_tokens is not None
                else int(query_start_loc[num_reqs].cpu())
            )
            if actual_num_tokens:
                qsl = query_start_loc[: num_reqs + 1].to(torch.int64)
                batch_indices = mps_segment_ids(qsl, num_reqs, actual_num_tokens)
                req_indices = idx_mapping.to(torch.int64)[batch_indices]
                token_positions = positions[:actual_num_tokens]
                for group_id, (block_table, block_size) in enumerate(
                    zip(self.block_tables, self.kernel_block_sizes)
                ):
                    block_span = block_size * self.cp_size
                    block_indices = token_positions // block_span
                    if not self._slot_mapping_enabled[group_id]:
                        block_indices = torch.zeros_like(block_indices)
                    block_offsets = token_positions % block_span
                    block_numbers = block_table.gpu[
                        req_indices, block_indices.to(torch.int64)
                    ].to(torch.int64)
                    if self.cp_size == 1:
                        slots = block_numbers * block_size + block_offsets
                    else:
                        is_local = (
                            block_offsets // self.cp_interleave % self.cp_size
                            == self.cp_rank
                        )
                        rounds = block_offsets // (self.cp_interleave * self.cp_size)
                        remainders = block_offsets % self.cp_interleave
                        local_offsets = rounds * self.cp_interleave + remainders
                        slots = block_numbers * block_size + local_offsets
                        slots = torch.where(is_local, slots, PAD_SLOT_ID)
                    slot_mappings[group_id, :actual_num_tokens].copy_(slots)
            if _integrity_checks_enabled():
                _check_slot_mappings(slot_mappings, num_tokens_padded)
            return slot_mappings[:, :num_tokens_padded]
        if _use_native("compute_slot_mappings") and all(self._slot_mapping_enabled):
            from vllm.quixicore import quixicore_ops

            quixicore_ops.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                self.block_table_ptrs,
                self.block_table_strides,
                self.block_sizes_tensor,
                slot_mappings,
                slot_mappings.stride(0),
                slot_mappings.shape[1],
                self.cp_rank,
                self.cp_size,
                self.cp_interleave,
                PAD_SLOT_ID,
            )
            if _integrity_checks_enabled():
                _check_slot_mappings(slot_mappings, num_tokens_padded)
            return slot_mappings[:, :num_tokens_padded]
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            slot_mappings.shape[1],
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mapping_enabled,
            slot_mappings,
            slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        if _integrity_checks_enabled():
            _check_slot_mappings(slot_mappings, num_tokens_padded)
        return slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    max_num_blocks = stride  # stride equals max_num_blocks for this group.
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mapping_enabled,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)
    mapping_enabled = tl.load(slot_mapping_enabled + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        block_indices = positions // (block_size * CP_SIZE)
        # A disabled mapping is a circular buffer: one physical block per
        # request in column 0, addressed by position modulo the ring capacity.
        block_indices = tl.where(mapping_enabled, block_indices, 0)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        slot_ids = tl.where(mapping_enabled, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
