# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from typing import cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
    DeepseekV4FlashMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    max_index,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        # Bound as well as sign-check: see _pack_global_topk_ragged_kernel.
        count += tl.sum(
            ((local_idx >= 0) & (local_idx < max_index)).to(tl.int32), axis=0
        )

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    max_index,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    # `max_index` is the addressable extent of one block-table row. An index at
    # or beyond it cannot name a slot this request owns, so dereferencing it
    # reads past the block table -- which is exactly the illegal access this
    # kernel used to take under load. The AITER decode top-k leaves a few of the
    # `topk` output slots uninitialized when a request has fewer than `topk`
    # candidates, and uninitialized memory reads as large positive ints, so the
    # sign check alone does not reject them. The prefill path next door already
    # bounds its indices the same way (`topk_indices < N` in
    # _combine_topk_swa_indices_kernel); this is the decode half of that.
    valid = mask & (local_idx >= 0) & (local_idx < max_index)
    block_indices = tl.where(valid, local_idx // block_size, 0)
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def _debug_check_topk_bounds(
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> None:
    """Reproduce the pack kernel's addressing on the host and report overruns.

    The kernel's fault is asynchronous and names nothing; this turns it into a
    Python error carrying the offending values. Debug only, behind
    VLLM_DSV4_DEBUG_TOPK.

    Reading the values back synchronizes, which a capturing stream forbids, so
    this is a no-op during CUDA graph capture rather than a crash of its own.
    """
    if torch.cuda.is_current_stream_capturing():
        return
    num_tokens, topk = topk_indices.shape
    n_rows, row_width = block_table.shape
    max_index = row_width * block_size
    offs = torch.arange(topk, device=topk_indices.device).unsqueeze(0)
    in_window = offs < topk_lens[:num_tokens].unsqueeze(1).to(offs.dtype)
    valid = in_window & (topk_indices >= 0) & (topk_indices < max_index)
    req = token_to_req_indices[:num_tokens]
    bad_req = (req < 0) | (req >= n_rows)
    blk = torch.where(valid, topk_indices // block_size, torch.zeros_like(topk_indices))
    bad_blk = valid & (blk >= row_width)
    if bad_req.any() or bad_blk.any():
        rows = bad_blk.any(dim=1).nonzero().flatten().tolist()
        detail = []
        for r in rows[:4]:
            row = topk_indices[r]
            detail.append(
                f"row{r}: len={int(topk_lens[r])} req={int(req[r])} "
                f"first8={row[:8].tolist()} "
                f"n_neg1={int((row == -1).sum())} n_huge={int((row > 1 << 20).sum())}"
            )
        raise RuntimeError(
            "dsv4 topk pack would read out of bounds: "
            f"block_table={tuple(block_table.shape)} block_size={block_size} "
            f"num_tokens={num_tokens} topk={topk} "
            f"bad_req={int(bad_req.sum())} bad_blk={int(bad_blk.sum())} "
            f"bad_rows={rows} | " + " | ".join(detail)
        )


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    # Every position a block-table row can address. Both kernels reject indices
    # at or beyond it rather than dereferencing them.
    max_index = block_table.shape[1] * block_size

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        max_index,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if os.environ.get("VLLM_DSV4_DEBUG_TOPK"):
        _debug_check_topk_bounds(
            topk_indices, topk_lens, token_to_req_indices, block_table, block_size
        )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            max_index,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    nnz = ragged_indices.numel()
    if nnz > 0:
        ragged_out[:nnz].copy_(ragged_indices, non_blocking=True)
    return ragged_out, indptr_out


@dataclass
class DeepseekV4ROCMAiterMLASparseMetadata(DeepseekV4FlashMLAMetadata):
    """ROCm-specific DeepSeek V4 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4FlashMLAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices, ragged_indptr = self._update_c128a_ragged(
            base.c128a_global_decode_topk_indices,
            base.c128a_decode_topk_lens,
        )

        return DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )



    def _update_c128a_ragged(
        self,
        dense_decode: torch.Tensor | None,
        decode_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Repack decode topk into the persistent ragged buffers (device work
        shared by build() and steady_decode_update(); the returned views are
        deterministic slices of those buffers)."""
        if dense_decode is None or decode_lens is None:
            return None, None
        ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
            dense_decode.reshape(dense_decode.shape[0], -1),
            decode_lens,
        )
        assert self.c128a_decode_topk_ragged_indices_buffer is not None
        assert self.c128a_decode_topk_ragged_indptr_buffer is not None
        return _copy_ragged_to_graph_buffers(
            ragged_indices,
            ragged_indptr,
            self.c128a_decode_topk_ragged_indices_buffer,
            self.c128a_decode_topk_ragged_indptr_buffer,
            dense_decode.shape[0],
            self.c128a_max_compressed,
        )

    def steady_decode_update(
        self,
        metadata: "DeepseekV4ROCMAiterMLASparseMetadata",
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> "DeepseekV4ROCMAiterMLASparseMetadata":
        super().steady_decode_update(metadata, common_attn_metadata)
        self._update_c128a_ragged(
            metadata.c128a_global_decode_topk_indices,
            metadata.c128a_decode_topk_lens,
        )
        return metadata


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices, ragged_indptr = self._update_swa_ragged(base)

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )

    def _update_swa_ragged(
        self, metadata: DeepseekSparseSWAMetadata
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Repack decode SWA indices into the persistent ragged buffers
        (device work shared by build() and steady_decode_update())."""
        if (
            metadata.num_decode_tokens <= 0
            or metadata.decode_swa_indices is None
            or metadata.decode_swa_lens is None
        ):
            return None, None
        ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
            metadata.decode_swa_indices.reshape(metadata.num_decode_tokens, -1),
            metadata.decode_swa_lens,
        )
        return _copy_ragged_to_graph_buffers(
            ragged_indices,
            ragged_indptr,
            self.decode_swa_ragged_indices_buffer,
            self.decode_swa_ragged_indptr_buffer,
            metadata.num_decode_tokens,
            # Actual dense width for this build: window_size (causal) or
            # noncausal_index_width (DSpark non-causal draft).
            metadata.decode_swa_indices.shape[-1],
        )

    def steady_decode_update(
        self,
        metadata: "DeepseekV4ROCMAiterSparseSWAMetadata",
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> "DeepseekV4ROCMAiterSparseSWAMetadata":
        super().steady_decode_update(metadata, common_attn_metadata)
        self._update_swa_ragged(metadata)
        return metadata



class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder


class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )
        _attn_split_debug(
            self, q, output, num_decode_tokens, num_prefills, num_decodes
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
        )
        _decode_lens_debug(self, output, swa_metadata, topk_lens)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
            self.PREFILL_CHUNK_SIZE
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )


_ATTN_SPLIT_DEBUG_STATE: dict = {}


def _attn_split_debug(
    attn, q, output, num_decode_tokens, num_prefills, num_decodes
) -> None:
    """Diagnostic (VLLM_DSV4_ATTN_SPLIT_DEBUG=1): after the real A100 sparse
    attention (rocm.py forward_mqa), report which SEGMENT of the output holds
    NaN — decode rows ([:num_decode_tokens], rocm_sparse_attn_decode) or
    prefill-chunk rows ([num_decode_tokens:], _forward_prefill) — plus query
    health, per attention layer. Syncs; diagnostic boots only.
    """
    state = _ATTN_SPLIT_DEBUG_STATE
    if "on" not in state:
        state["on"] = os.getenv(
            "VLLM_DSV4_ATTN_SPLIT_DEBUG", "0").lower() in ("1", "true", "on")
        state["dumps"] = 0
    if not state["on"] or state["dumps"] >= 8:
        return
    flat = output.float().reshape(output.shape[0], -1)
    nan_rows = torch.isnan(flat).any(dim=-1)
    if not bool(nan_rows.any()):
        return
    state["dumps"] += 1
    rows = nan_rows.nonzero().flatten().tolist()
    q_nan_rows = torch.isnan(
        q.float().reshape(q.shape[0], -1)).any(dim=-1).nonzero().flatten().tolist()
    dec = [r for r in rows if r < num_decode_tokens]
    pre = [r for r in rows if r >= num_decode_tokens]
    from vllm.logger import init_logger
    init_logger(__name__).error(
        "ATTN_SPLIT_DEBUG dump %d: layer=%s compress_ratio=%s tokens=%d "
        "(decode=%d prefill_reqs=%d) q_nan_rows=%s | NAN decode_rows=%s "
        "prefill_rows=%s",
        state["dumps"], getattr(attn, "prefix", "?"),
        getattr(attn, "compress_ratio", "?"), output.shape[0],
        num_decode_tokens, num_prefills, q_nan_rows[:8], dec[:8], pre[:8])


_DECODE_LENS_DEBUG_STATE: dict = {}


def _decode_lens_debug(attn, output, swa_metadata, topk_lens) -> None:
    """Diagnostic (VLLM_DSV4_ATTN_SPLIT_DEBUG=1, shared gate): for NaN decode
    rows, dump the per-row attention extents — is_valid_token, topk_lens,
    decode_swa_lens — against a clean row. All-zero extents mean an empty
    reduction (0/0 = NaN from clean inputs)."""
    state = _DECODE_LENS_DEBUG_STATE
    if "on" not in state:
        state["on"] = os.getenv(
            "VLLM_DSV4_ATTN_SPLIT_DEBUG", "0").lower() in ("1", "true", "on")
        state["dumps"] = 0
    if not state["on"] or state["dumps"] >= 6:
        return
    flat = output.float().reshape(output.shape[0], -1)
    nan_rows = torch.isnan(flat).any(dim=-1)
    if not bool(nan_rows.any()):
        return
    state["dumps"] += 1
    rows = nan_rows.nonzero().flatten().tolist()
    clean = (~nan_rows).nonzero().flatten().tolist()
    n = output.shape[0]
    is_valid = swa_metadata.is_valid_token
    swa_lens = swa_metadata.decode_swa_lens

    def rep(r):
        iv = int(is_valid[r]) if is_valid is not None and is_valid.numel() > r else "?"
        tl_ = int(topk_lens[r]) if topk_lens is not None and topk_lens.numel() > r else "?"
        sl = int(swa_lens[r]) if swa_lens is not None and swa_lens.numel() > r else "?"
        return f"row{r}: valid={iv} topk_len={tl_} swa_len={sl}"

    from vllm.logger import init_logger
    init_logger(__name__).error(
        "DECODE_LENS_DEBUG dump %d: layer=%s n_decode_tokens=%d nan=%s | %s "
        "| CONTROL %s",
        state["dumps"], getattr(attn, "prefix", "?"), n, rows[:6],
        " | ".join(rep(r) for r in rows[:4]),
        rep(clean[0]) if clean else "none")
