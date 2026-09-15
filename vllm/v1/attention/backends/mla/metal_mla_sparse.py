# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse NoPE MLA (DSA) backend for Apple Metal, torch-native.

The Apple counterpart of `quixicore_mla_sparse` (Ampere) and
`rocm_aiter_mla_sparse` (gfx942), built for GLM-5.3-Flash (`glm5_next`):
`kv_lora_rank` 512, `qk_rope_head_dim` 0, bf16/f16 latent pages, and the
pooled indexer's top-k list in `topk_indices_buffer`. Like both of those it
has no dense-MHA prefill path; every query row (decode, spec verify and
prefill alike) is answered on the MQA path against the paged latent, so
serving keeps `sparse_mla_force_mqa` semantics whether or not the flag is
set (the platform has no MLA prefill backend, so `MLAAttention` already
routes all tokens through `forward_mqa`).

KV cache: `(num_blocks, block_size, head_size)` with head_size =
kv_lora_rank + qk_rope_head_dim, the same page geometry as the CUDA backend,
so the hybrid (KDA + MLA + indexer) pool, the packed cross-layer slab and the
KV tiers see identical pages. The block dim is 0 (pages already contiguous),
so the hybrid blocks-first restride is a no-op for this group.

Math, per query row, over its selected positions (`-1` = none):

    s_j = q . kv[j] * sm_scale       softmax over valid j
    o   = sum_j p_j kv[j][:kv_lora_rank]

which is what `mla_decode_bf16_sparse_nope` computes on CUDA (the caller's
`_v_up_proj` then applies W_UV, and W_UK was absorbed into `q` upstream).
Prefill requests whose whole context fits the selection width
(`seq_len <= index_topk + index_kpool - 1`, 2051 for GLM) instead run dense
causal attention over their latent prefix: with every pool and the tail
selectable the pooled indexer selects every token, so the dense result is
the sparse result without the 2080-wide gather (the ds4 `selected_limit`
behavior). Longer requests take the gathered path row by row.

Everything is fixed-shape and sync-free from python-int metadata; the only
host-side numbers are the per-request query/sequence bounds the builder
already holds on the CPU. Gathers go through `_gather_rows`, which windows
the cache when the page view exceeds the signed 32-bit element range that
MPS index kernels have been seen to use for strided sources (the
`kv_cache_gather_range` precedent in metal_attn.py).

fp8 latent pages are not implemented here (MPS has no fp8 dtype; the DSV4
Metal path decodes e4m3 through a LUT in a native kernel). Requesting an
fp8 KV cache with this backend raises at layer construction.
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Query rows per gathered-sparse launch and per dense-causal launch; both
# bound the fp32 score tile ([rows, heads, width]) and the gathered latent
# ([rows, width, 512] bf16) instead of scaling them with the batch.
_SPARSE_ROW_CHUNK = 32
_DENSE_ROW_CHUNK = 128
# Fresh-prefill rows under the dense limit attend through MPS SDPA in the
# per-head space (dense_causal_attend_sdpa); 0 pins the torch fp32 paths.
_PREFILL_SDPA = os.environ.get("VLLM_METAL_MLA_PREFILL_SDPA", "1") != "0"
_MLA_TLEN = os.environ.get("VLLM_METAL_MLA_TLEN", "1") != "0"
_INDEX32_LIMIT = 2**31 - 1


def _page_extent(kv_cache: torch.Tensor) -> int:
    if kv_cache.shape[0] == 0:
        return 0
    return (kv_cache.shape[0] - 1) * kv_cache.stride(0) + (
        kv_cache.shape[1] * kv_cache.shape[2]
    )


def _gather_rows(
    kv_cache: torch.Tensor, blk: torch.Tensor, off: torch.Tensor
) -> torch.Tensor:
    """rows[...] = kv_cache[blk[...], off[...]] with int64 indices; windowed
    when the strided page view exceeds the 32-bit element-offset range."""
    if kv_cache.device.type != "mps" or _page_extent(kv_cache) <= _INDEX32_LIMIT:
        return kv_cache[blk, off]
    win = max(1, _INDEX32_LIMIT // max(1, kv_cache.stride(0)))
    out = torch.zeros(
        (*blk.shape, kv_cache.shape[2]), dtype=kv_cache.dtype, device=kv_cache.device
    )
    for b0 in range(0, kv_cache.shape[0], win):
        b1 = min(kv_cache.shape[0], b0 + win)
        inwin = (blk >= b0) & (blk < b1)
        local = torch.where(inwin, blk - b0, 0)
        out = torch.where(inwin[..., None], kv_cache[b0:b1][local, off], out)
    return out


def _scatter_rows(
    kv_cache: torch.Tensor, blk: torch.Tensor, off: torch.Tensor, vals: torch.Tensor
) -> None:
    if kv_cache.device.type != "mps" or _page_extent(kv_cache) <= _INDEX32_LIMIT:
        kv_cache[blk, off] = vals
        return
    win = max(1, _INDEX32_LIMIT // max(1, kv_cache.stride(0)))
    for b0 in range(0, kv_cache.shape[0], win):
        b1 = min(kv_cache.shape[0], b0 + win)
        sub = kv_cache[b0:b1]
        inwin = (blk >= b0) & (blk < b1)
        local = torch.where(inwin, blk - b0, 0)
        loff = torch.where(inwin, off, 0)
        sub[local, loff] = torch.where(inwin[:, None], vals, sub[0, 0][None, :])


def insert_latent_rows(
    kv_cache: torch.Tensor, latent: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    """kv_cache[slot] = latent for every slot >= 0. PAD slots (-1) are
    clamped onto block 0 row 0, the KV manager's null block that no request
    reads (metal_attn's cache-update convention; a data-dependent mask
    would need a host sync)."""
    n = slot_mapping.shape[0]
    if n == 0:
        return
    if _row_insert_kernel_ok(kv_cache):
        from vllm.quixicore import quixicore_ops

        # The kernel takes int32 or int64 slots; the runner's int64 mapping
        # goes in as is (the per-layer cast was a launch per MLA layer).
        slots = slot_mapping
        if slots.dtype not in (torch.int32, torch.int64):
            slots = slots.to(torch.int64)
        latent_rows = latent[:n]
        if latent_rows.dtype != kv_cache.dtype:
            latent_rows = latent_rows.to(kv_cache.dtype)
        quixicore_ops.paged_row_insert(latent_rows, kv_cache, slots.contiguous())
        return
    bs = kv_cache.shape[1]
    slot = slot_mapping.to(torch.int64).clamp_min(0)
    blk = torch.div(slot, bs, rounding_mode="floor")
    off = slot - blk * bs
    _scatter_rows(kv_cache, blk, off, latent[:n].to(kv_cache.dtype))


_ROW_INSERT: bool | None = None


def _row_insert_kernel_ok(kv_cache: torch.Tensor) -> bool:
    """One Metal launch for the latent insert (the torch scatter is five
    small ops). VLLM_METAL_ROW_INSERT=0 pins torch."""
    global _ROW_INSERT
    if _ROW_INSERT is None:
        _ROW_INSERT = False
        if (
            current_platform.is_metal()
            and os.environ.get("VLLM_METAL_ROW_INSERT", "1") != "0"
        ):
            try:
                from vllm.quixicore import quixicore_ops

                _ROW_INSERT = quixicore_ops.is_available() and (
                    quixicore_ops.has("paged_row_insert")
                )
            except Exception:
                _ROW_INSERT = False
    return (
        _ROW_INSERT
        and kv_cache.device.type == "mps"
        and kv_cache.dim() == 3
        and kv_cache.stride(2) == 1
        and kv_cache.stride(1) == kv_cache.shape[2]
    )


_SPARSE_KERNEL: bool | None = None


def _sparse_kernel_ok(q: torch.Tensor, kv_cache: torch.Tensor, d_v: int) -> bool:
    """The Metal ``mla_sparse_latent_decode`` kernel applies: MPS, bf16/f16
    latent pages of width 512 with the value being the whole latent (NoPE),
    q in the cache dtype. ``VLLM_METAL_MLA_SPARSE_KERNEL=0`` pins torch."""
    global _SPARSE_KERNEL
    if q.device.type != "mps":
        return False
    if _SPARSE_KERNEL is None:
        import os

        if os.environ.get("VLLM_METAL_MLA_SPARSE_KERNEL", "1") == "0":
            _SPARSE_KERNEL = False
        else:
            try:
                from vllm.quixicore import quixicore_ops

                _SPARSE_KERNEL = quixicore_ops.has("mla_sparse_latent_decode")
            except Exception:
                _SPARSE_KERNEL = False
    return (
        _SPARSE_KERNEL
        and q.dtype in (torch.bfloat16, torch.float16)
        and kv_cache.dtype == q.dtype
        and q.shape[-1] == 512
        and d_v == 512
        and kv_cache.shape[-1] == 512
    )


def sparse_attend_rows(
    q: torch.Tensor,  # [R, H, Dk] (any float dtype)
    kv_cache: torch.Tensor,  # [num_blocks, BS, Dk]
    block_table: torch.Tensor,  # [R, max_blocks] int (one row per query row)
    indices: torch.Tensor,  # [R, W] int32 request-local positions, -1 pad
    block_size: int,
    sm_scale: float,
    d_v: int,
    tlen: torch.Tensor | None = None,  # [R] int32 valid prefix of `indices`
) -> torch.Tensor:
    """Softmax attention of each query row over its selected latent rows.
    Returns [R, H, d_v] in q.dtype. Rows with no valid index return 0."""
    R, H, _ = q.shape
    if _sparse_kernel_ok(q, kv_cache, d_v):
        from vllm.quixicore import quixicore_ops

        bt = block_table
        if bt.dtype != torch.int32:
            bt = bt.to(torch.int32)
        idx = indices if indices.dtype == torch.int32 else indices.to(torch.int32)
        return quixicore_ops.mla_sparse_latent_decode(
            q.contiguous(), kv_cache, bt.contiguous(), idx.contiguous(), sm_scale,
            0, tlen,
        )
    out = torch.empty((R, H, d_v), dtype=q.dtype, device=q.device)
    num_blocks = kv_cache.shape[0]
    max_col = max(block_table.shape[1] - 1, 0)
    bt64 = block_table.to(torch.int64)
    for r0 in range(0, R, _SPARSE_ROW_CHUNK):
        r1 = min(r0 + _SPARSE_ROW_CHUNK, R)
        idx = indices[r0:r1].to(torch.int64)
        valid = idx >= 0
        pos = torch.where(valid, idx, 0)
        col = torch.div(pos, block_size, rounding_mode="floor").clamp_(max=max_col)
        blk = bt64[r0:r1].gather(1, col).clamp_(0, num_blocks - 1)
        off = torch.remainder(pos, block_size)
        kv = _gather_rows(kv_cache, blk, off).float()  # [r, W, Dk]
        # PAD entries gather an arbitrary row (possibly a never-written
        # page holding NaN bit patterns); 0 * NaN would poison the weighted
        # sum, so zero them the way the CUDA kernel skips -1 entries.
        kv = torch.where(valid[:, :, None], kv, torch.zeros_like(kv))
        s = torch.einsum("rhd,rwd->rhw", q[r0:r1].float(), kv) * sm_scale
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        p = torch.softmax(s, dim=-1)
        # a row without any valid position is all -inf -> NaN; zero it.
        p = torch.where(valid.any(dim=1)[:, None, None], p, torch.zeros_like(p))
        out[r0:r1] = torch.einsum("rhw,rwv->rhv", p, kv[..., :d_v]).to(q.dtype)
        del kv, s, p
    return out


def dense_causal_attend(
    q: torch.Tensor,  # [Q, H, Dk] the request's query rows, in order
    kv_cache: torch.Tensor,
    blocks: torch.Tensor,  # [nblk] int block ids covering [0, seq_len)
    seq_len: int,
    block_size: int,
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    """Dense causal attention of Q query rows (absolute positions
    seq_len - Q .. seq_len - 1) over the request's latent prefix [0, seq_len).
    Returns [Q, H, d_v] in q.dtype."""
    Q, H, _ = q.shape
    dev = q.device
    nblk = blocks.shape[0]
    blk = (
        blocks.to(torch.int64)
        .clamp(0, kv_cache.shape[0] - 1)[:, None]
        .expand(nblk, block_size)
    )
    off = torch.arange(block_size, device=dev)[None, :].expand(nblk, block_size)
    kv = _gather_rows(kv_cache, blk, off).reshape(-1, kv_cache.shape[2])[:seq_len]
    kv = kv.float()
    ctx0 = seq_len - Q
    key_pos = torch.arange(seq_len, device=dev)
    out = torch.empty((Q, H, d_v), dtype=q.dtype, device=dev)
    for i0 in range(0, Q, _DENSE_ROW_CHUNK):
        i1 = min(i0 + _DENSE_ROW_CHUNK, Q)
        qpos = ctx0 + torch.arange(i0, i1, device=dev)
        s = torch.einsum("qhd,sd->qhs", q[i0:i1].float(), kv) * sm_scale
        s = s.masked_fill((key_pos[None, :] > qpos[:, None])[:, None, :], float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[i0:i1] = torch.einsum("qhs,sv->qhv", p, kv[:, :d_v]).to(q.dtype)
        del s, p
    return out


def dense_causal_attend_sdpa(
    q_nope: torch.Tensor,  # [H, Q, P] un-absorbed query rows, positions 0..Q-1
    kv: torch.Tensor,  # [Q, L] the request's latent rows 0..Q-1
    W_UK_T: torch.Tensor,  # [H, P, L] absorb weight (k_nope_h = kv @ W_UK_T[h].T)
    sm_scale: float,
) -> torch.Tensor:
    """Dense causal attention of a request's first Q rows over its own first
    Q latent rows, in the hybrid form: scores in the per-head 256-wide
    space (K decompressed once with kv_b, one bf16 GEMM), values in the
    512-wide latent space so the caller's v_up projection is unchanged.
    Mathematically the absorbed form (q_nope W_UK_T . kv == q_nope . (kv
    W_UK_T^T)); measured 2.3x the torch fp32 einsum chain at 2048 rows
    (w11-prefill/bench_mla_prefill.log). Returns [Q, H, L] in q dtype."""
    H, Q, _ = q_nope.shape
    S, L = kv.shape
    assert S == Q, (S, Q)
    k = torch.matmul(kv[None], W_UK_T.transpose(1, 2))  # [H, Q, P]
    v = kv[None, None].expand(1, H, Q, L)
    o = F.scaled_dot_product_attention(
        q_nope[None], k[None], v, is_causal=True, scale=sm_scale
    )
    return o[0].transpose(0, 1)


class MetalMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    # Latent pages in the model dtype only; see the module docstring on fp8.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The torch path takes the block size at runtime, so the kernel
        # block may equal the KV-manager block (mamba alignment raises it
        # for the hybrid KDA + MLA pool); 16 matches METAL_ATTN.
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "METAL_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["MetalMLASparseMetadata"]:
        return MetalMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["MetalMLASparseMetadataBuilder"]:
        return MetalMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MetalMLASparseImpl"]:
        return MetalMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


@dataclass
class MetalMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    # Per-token block-table gather (int32), one row per query token.
    bt_per_token: torch.Tensor
    attn_out_dtype: torch.dtype

    # Host copies for the per-request prefill loop (no device sync inside
    # the forward): query bounds and context lengths (int32 CPU).
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor

    block_size: int = 64
    topk_tokens: int = 2048
    # Contexts up to this length are fully selected by the pooled indexer
    # (index_topk + index_kpool - 1); prefill requests within it run dense.
    selected_limit: int = 2051

    # Same contract as the CUDA sparse backend: no dense-MHA prefill
    # metadata; everything runs on the MQA path.
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill_max_seq_len: int = 0
    prefill: None = None


class MetalMLASparseMetadataBuilder(AttentionMetadataBuilder[MetalMLASparseMetadata]):
    # No graph capture on Metal.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device
        self.vllm_config = vllm_config
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        hf = self.model_config.hf_text_config
        self.topk_tokens = int(hf.index_topk)
        kpool = int(getattr(hf, "index_kpool", 1) or 1)
        self.selected_limit = self.topk_tokens + kpool - 1
        # Exact host context lengths without a device pull: with serial
        # scheduling the CPU upper bound is exact (see metal_attn.py); under
        # async scheduling it can overshoot on spec-decode rows, which only
        # ever take the gathered path (the dense path reads prefill rows,
        # whose bound is exact in both modes).
        self._exact_cpu_seq_lens = not bool(
            vllm_config.scheduler_config.async_scheduling
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.zeros(
            (max_tokens,), dtype=torch.int64, device=device
        )
        self.bt_per_token_buffer: torch.Tensor | None = None
        self._arange_dev: torch.Tensor | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MetalMLASparseMetadata:
        m = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            m, decode_threshold=self.reorder_batch_threshold
        )
        num_tokens = m.num_actual_tokens
        qsl_cpu = m.query_start_loc_cpu.to(torch.int32)
        # Request id per token, built on the device: a CPU->MPS copy_ blocks
        # until every queued command buffer drains (measured 1.9 ms/step
        # of pipeline stall at decode), so the segment lengths come from
        # the device query_start_loc and the arange is a persistent buffer.
        num_reqs = m.num_reqs
        if (
            self._arange_dev is None
            or self._arange_dev.numel() < num_reqs
            or self._arange_dev.device != self.device
        ):
            self._arange_dev = torch.arange(
                max(num_reqs, 64), dtype=torch.int64, device=self.device
            )
        seg_lengths = torch.diff(m.query_start_loc[: num_reqs + 1])
        req_id_per_token = torch.repeat_interleave(
            self._arange_dev[:num_reqs], seg_lengths, output_size=num_tokens
        )
        self.req_id_per_token_buffer[:num_tokens].copy_(req_id_per_token)
        block_table = m.block_table_tensor
        if (
            self.bt_per_token_buffer is None
            or self.bt_per_token_buffer.shape[1] != block_table.shape[1]
        ):
            self.bt_per_token_buffer = torch.zeros(
                (self.req_id_per_token_buffer.shape[0], block_table.shape[1]),
                dtype=torch.int32,
                device=self.device,
            )
        self.bt_per_token_buffer[:num_tokens].copy_(
            block_table.to(torch.int32).index_select(
                0, self.req_id_per_token_buffer[:num_tokens]
            )
        )
        bound = m.seq_lens_cpu_upper_bound
        if bound is not None and bound.numel() >= m.num_reqs:
            seq_lens_cpu = bound[: m.num_reqs].to(torch.int32)
        else:
            seq_lens_cpu = m.seq_lens[: m.num_reqs].to("cpu", dtype=torch.int32)
        prefill_max = 0
        if num_prefills > 0 and seq_lens_cpu.numel() > num_decodes:
            prefill_max = int(seq_lens_cpu[num_decodes:].max())
        return MetalMLASparseMetadata(
            num_reqs=m.num_reqs,
            max_query_len=m.max_query_len,
            max_seq_len=m.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=m.query_start_loc,
            slot_mapping=m.slot_mapping,
            block_table=block_table,
            bt_per_token=self.bt_per_token_buffer[:num_tokens],
            attn_out_dtype=self.model_config.dtype,
            query_start_loc_cpu=qsl_cpu,
            seq_lens_cpu=seq_lens_cpu,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            selected_limit=self.selected_limit,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            prefill_max_seq_len=prefill_max,
        )


class MetalMLASparseImpl(MLAAttentionImpl[MetalMLASparseMetadata]):
    """Top-k gathered MQA (and dense-causal short prefill) over the paged
    latent, in torch. forward_mha is intentionally absent, as on CUDA/ROCm."""

    is_sparse = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer=None,
        **mla_args,
    ) -> None:
        if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
            raise NotImplementedError(
                "METAL_MLA_SPARSE keeps the latent pages in the model dtype "
                f"(bf16/f16); kv_cache_dtype={kv_cache_dtype!r} is not "
                "implemented on Apple Metal (MPS has no fp8 dtype)."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_rope_head_dim: int = mla_args.get("qk_rope_head_dim", 0)
        self.softmax_scale = float(scale)
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # The indexer carries the shared buffer for normal layers; the
        # explicit buffer covers backbone skip layers without an indexer.
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        # The GLM pooled indexer also publishes each row's valid prefix
        # length; the decode kernel partitions only that range
        # (VLLM_METAL_MLA_TLEN=0 restores the padded-width scan for
        # bisection: same values up to fp32 partition-merge rounding).
        self.topk_len_buffer: torch.Tensor | None = (
            getattr(indexer, "topk_len_buffer", None) if _MLA_TLEN else None
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Latent insert (torch; the CUDA op concat_and_cache_mla has no
        Metal binding). The page row is [kv_c | k_pe] like everywhere else;
        for NoPE GLM the k_pe half is empty."""
        if kv_cache.numel() == 0 or self.kv_sharing_target_layer_name is not None:
            return
        slots = slot_mapping.flatten()
        n = slots.shape[0]
        latent = kv_c_normed[:n]
        if k_pe is not None and k_pe.shape[-1] > 0:
            latent = torch.cat([latent, k_pe[:n].reshape(n, -1)], dim=-1)
        insert_latent_rows(kv_cache, latent, slots)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MetalMLASparseMetadata,
        layer=None,
    ) -> tuple[torch.Tensor, None]:
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = ql_nope if q_pe.shape[-1] == 0 else torch.cat([ql_nope, q_pe], dim=-1)
        num_tokens = attn_metadata.num_actual_tokens
        q = q[:num_tokens]
        assert self.topk_indices_buffer is not None
        cache = kv_c_and_k_pe_cache
        assert q.shape[-1] == cache.shape[-1], (q.shape, cache.shape)
        d_v = self.kv_lora_rank
        block_size = attn_metadata.block_size
        indices = self.topk_indices_buffer[:num_tokens]
        tlen = (
            self.topk_len_buffer[:num_tokens]
            if self.topk_len_buffer is not None
            else None
        )
        bt = attn_metadata.bt_per_token[:num_tokens]

        out = torch.empty(
            (num_tokens, self.num_heads, d_v), dtype=q.dtype, device=q.device
        )
        nd = attn_metadata.num_decode_tokens
        if nd == num_tokens:
            # Pure decode step: hand the kernel/torch result back directly
            # (no output slab + copy per layer).
            return (
                sparse_attend_rows(
                    q, cache, bt, indices, block_size, self.softmax_scale, d_v,
                    tlen,
                ),
                None,
            )
        if nd > 0:
            out[:nd] = sparse_attend_rows(
                q[:nd],
                cache,
                bt[:nd],
                indices[:nd],
                block_size,
                self.softmax_scale,
                d_v,
                None if tlen is None else tlen[:nd],
            )
        if attn_metadata.num_prefills > 0:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
            lens = attn_metadata.seq_lens_cpu.tolist()
            limit = attn_metadata.selected_limit
            block_table = attn_metadata.block_table
            q_nope = getattr(layer, "_metal_q_nope", None) if _PREFILL_SDPA else None
            W_UK_T = getattr(layer, "W_UK_T", None)
            for r in range(attn_metadata.num_decodes, attn_metadata.num_reqs):
                s, e = qsl[r], qsl[r + 1]
                if e <= s:
                    continue
                seq_len = int(lens[r])
                if (
                    q_nope is not None
                    and W_UK_T is not None
                    and seq_len == e - s
                    and q_nope.shape[1] >= e
                ):
                    # Fresh prefill from position 0: row i attends keys
                    # 0..i, all of which the indexer selects while i < limit,
                    # so those rows are dense-exact and take the SDPA form;
                    # rows past the limit keep the gathered top-k path.
                    nd = min(e - s, limit)
                    nblk = (nd + block_size - 1) // block_size
                    blk = (
                        block_table[r, :nblk]
                        .to(torch.int64)
                        .clamp(0, cache.shape[0] - 1)[:, None]
                        .expand(nblk, block_size)
                    )
                    off = torch.arange(block_size, device=q.device)[None, :].expand(
                        nblk, block_size
                    )
                    kv = _gather_rows(cache, blk, off).reshape(-1, cache.shape[2])[:nd]
                    out[s : s + nd] = dense_causal_attend_sdpa(
                        q_nope[:, s : s + nd], kv, W_UK_T, self.softmax_scale
                    ).to(out.dtype)
                    if nd < e - s:
                        out[s + nd : e] = sparse_attend_rows(
                            q[s + nd : e],
                            cache,
                            bt[s + nd : e],
                            indices[s + nd : e],
                            block_size,
                            self.softmax_scale,
                            d_v,
                        )
                    continue
                if seq_len <= limit and seq_len >= e - s:
                    nblk = (seq_len + block_size - 1) // block_size
                    out[s:e] = dense_causal_attend(
                        q[s:e],
                        cache,
                        block_table[r, :nblk],
                        seq_len,
                        block_size,
                        self.softmax_scale,
                        d_v,
                    )
                else:
                    out[s:e] = sparse_attend_rows(
                        q[s:e],
                        cache,
                        bt[s:e],
                        indices[s:e],
                        block_size,
                        self.softmax_scale,
                        d_v,
                    )
        return out, None
