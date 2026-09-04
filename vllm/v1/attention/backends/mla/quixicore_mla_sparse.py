# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA (DSA) backend for NVIDIA Ampere, backed by QuixiCore-CUDA.

The CUDA counterpart of `rocm_aiter_mla_sparse`: the top-k-gathered MQA
decode runs through the vendored `mla_decode_fp8_sparse` kernel
(csrc/quixicore/serving/mla_kernels.cuh) instead of AITER's `mla_decode_fwd`.
Like the ROCm backend it has no dense-MHA prefill path, so serving requires
`--attention-config '{"sparse_mla_force_mqa": true}'`.

Bring-up status: the class surface, metadata plumbing and kernel wiring are
in place; the pieces marked TODO(quixicore-cuda) below must be finished and
validated against `reference_mla_sparse_prefill` (the pure-torch oracle in
`rocm_aiter_mla_sparse.py`) before this backend serves traffic:

1. Cache layout adapter: vLLM stores the fp8 latent as one
   (num_blocks, block_size, 576[+scale]) page; the kernel takes separate
   packed data/scale tensors. Decide view vs. split at insert time.
2. `build()`: the top-k index tensors arrive req-local from the indexer;
   the kernel resolves them against the block table, so no aiter-style
   global-index conversion pass is needed — but the -1 padding convention
   and spec-decode (uniform-batch CUDA graph) paths need wiring.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.quixicore import quixicore_ops
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


_BF16_PARTITION = 128
_BF16_PARTITION_SCRATCH_CAP = 512 << 20  # bytes of fp32 partials


def _bf16_partition(q: torch.Tensor, idx: torch.Tensor) -> int:
    """Partition size for the bf16 sparse decode launches.

    Partitioning exists to expose parallelism at small token counts (a
    decode step is B x H warps, 8 at TP8/c1). Its scratch is
    B x H x P x 512 fp32; this MQA path also serves prefill chunks, where
    B is thousands of tokens and the scratch reached 3.4 GiB and OOMed
    glm52-q2k-8 (2026-09-03). Above the cap the unpartitioned launch is
    already B x H warps wide, so partition only while the scratch is small.
    """
    B, H = q.shape[0], q.shape[1]
    P = (idx.shape[1] + _BF16_PARTITION - 1) // _BF16_PARTITION
    if B * H * P * 512 * 4 > _BF16_PARTITION_SCRATCH_CAP:
        return 0
    return _BF16_PARTITION


class QuixiCoreMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # The sparse decode kernel reads the latent cache as packed e4m3 + scales
    # (software-decoded on sm80, which has no fp8 hardware).
    # sm80 has no native fp8e4nv, so vLLM's reshape-and-cache cannot store an
    # fp8 KV cache there -- bf16 is the geometry that actually runs on Ampere.
    # fp8 stays listed for sm89+; forward_mqa dispatches on the cache dtype.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "bfloat16", "fp8", "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "QUIXICORE_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["QuixiCoreMLASparseMetadata"]:
        return QuixiCoreMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["QuixiCoreMLASparseMetadataBuilder"]:
        return QuixiCoreMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["QuixiCoreMLASparseImpl"]:
        return QuixiCoreMLASparseImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
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

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major >= 8


@dataclass
class QuixiCoreMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    attn_out_dtype: torch.dtype

    # Per-token block-table gather, identical for every layer in a step.
    # Computed lazily by the first forward_mqa call and reused by the rest;
    # under CUDA graph capture the first layer's gather is captured once and
    # replays against the persistent block_table/req_id buffers.
    bt_per_token: torch.Tensor | None = None

    block_size: int = 64
    topk_tokens: int = 2048

    # Same contract as the ROCm sparse backend: no dense-MHA prefill
    # metadata; `sparse_mla_force_mqa` keeps everything on the MQA path.
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill_max_seq_len: int = 0
    prefill: None = None


@dataclass
class QuixiCoreMLASparseMetadataBuilder(
    AttentionMetadataBuilder[QuixiCoreMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device
        self.vllm_config = vllm_config
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        parallel_config = vllm_config.parallel_config
        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_text_config.index_topk

        self.req_id_per_token_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        # Per-token block-table gather, ROCm-parity persistence: computed in
        # build() into a persistent buffer instead of lazily inside the first
        # layer's forward. Lazy in-forward compute allocated the gather output
        # per step, which under FULL-graph capture baked a capture-pool
        # address into every layer's kernels while replay-time metadata was
        # rebuilt around it — the buffer-persistence gap the old TODO warned
        # about. Allocated on first build (block-table width is per-boot).
        self.bt_per_token_buffer: torch.Tensor | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QuixiCoreMLASparseMetadata:
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
        )

        num_tokens = common_attn_metadata.num_actual_tokens
        starts = common_attn_metadata.query_start_loc_cpu[:-1]
        seg_lengths = torch.diff(common_attn_metadata.query_start_loc_cpu)
        req_id_per_token = torch.repeat_interleave(
            torch.arange(len(seg_lengths), dtype=torch.int32),
            seg_lengths,
        )
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            req_id_per_token, non_blocking=True
        )
        self.req_id_per_token_buffer[req_id_per_token.shape[0] :].fill_(0)

        block_table = common_attn_metadata.block_table_tensor
        if (
            self.bt_per_token_buffer is None
            or self.bt_per_token_buffer.shape[1] != block_table.shape[1]
        ):
            self.bt_per_token_buffer = torch.zeros(
                (
                    self.vllm_config.scheduler_config.max_num_batched_tokens,
                    block_table.shape[1],
                ),
                dtype=torch.int32,
                device=self.device,
            )
        torch.index_select(
            block_table.to(torch.int32),
            0,
            self.req_id_per_token_buffer[:num_tokens],
            out=self.bt_per_token_buffer[:num_tokens],
        )

        return QuixiCoreMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            attn_out_dtype=self.model_config.dtype,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            bt_per_token=self.bt_per_token_buffer[:num_tokens],
        )


class QuixiCoreMLASparseImpl(MLAAttentionImpl[QuixiCoreMLASparseMetadata]):
    """Top-k MQA against the fp8 paged latent via QuixiCore-CUDA.

    forward_mha is intentionally absent (as on ROCm): serving requires
    sparse_mla_force_mqa, and prefill goes through the same gathered-MQA
    kernel one query block at a time.
    """

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
        # Mirrors ROCMAiterMLASparseImpl: the base MLAAttentionImpl does not
        # accept the MLA-specific kwargs, so fields are set directly rather than
        # forwarded through super().__init__.
        if not quixicore_ops.is_available():
            raise ImportError(
                "QUIXICORE_MLA_SPARSE requires the vllm._quixicore_C "
                "extension (built automatically for CUDA targets)."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = float(scale)
        # Host copy of layer._k_scale. Reading it per call would be a D2H sync,
        # which CUDA graph capture rejects; the scale is fixed once weights are
        # loaded, so it is cached on first use during eager warmup.
        self._k_scale_host: float | None = None
        # The indexer carries the shared buffer for normal layers; the explicit
        # buffer covers backbone skip layers whose indexer is not constructed.
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: QuixiCoreMLASparseMetadata,
        layer=None,
    ) -> tuple[torch.Tensor, None]:
        """Top-k gathered MQA over the paged fp8 latent.

        The kernel is `mla_decode_fp8_v<SPARSE, !PART, 576, 512, 576, 1>`: q and
        each cache slot are 576 fp8-decoded elements, the value is the leading
        512 (the trailing 64 rope dims score but do not accumulate), and the
        cache carries a single per-tensor `k_scale`.

        `indices` are request-local logical token positions that the kernel
        resolves through the block table, which is exactly what
        `topk_indices_buffer` already holds -- so unlike the AITER path this
        needs no global-index conversion pass. Entries of -1 are skipped by the
        kernel, so `topk_length` can stay at the padded width.
        """
        num_tokens = attn_metadata.num_actual_tokens
        splitq = None
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            # ql_nope arrives as the transpose view of a head-major bmm
            # output; when that and q_pe are directly readable and the cache
            # is fp8, skip the per-layer cat and read both buffers in-kernel.
            nope_hm = ql_nope.transpose(0, 1)
            if (
                ql_nope.shape[0] == num_tokens
                and nope_hm.is_contiguous()
                and ql_nope.shape[-1] == 512
                and q_pe.shape[-1] == 64
                and q_pe.stride(-1) == 1
                and q_pe.stride(0) == q_pe.shape[1] * q_pe.stride(1)
                and kv_c_and_k_pe_cache.dtype != torch.bfloat16
            ):
                splitq = (nope_hm, q_pe)
            else:
                q = torch.cat([ql_nope, q_pe], dim=-1)

        if splitq is None:
            q = q[:num_tokens].contiguous()
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        idx = topk_indices.to(torch.int32).contiguous()

        # One kernel "batch" entry per query token; each needs its request's
        # block table row. The gather is step-level metadata, so the first
        # layer computes it and the rest reuse it.
        bt = attn_metadata.bt_per_token
        if bt is None:
            bt = attn_metadata.block_table.index_select(
                0, attn_metadata.req_id_per_token[:num_tokens].to(torch.int32)
            ).to(torch.int32).contiguous()
            attn_metadata.bt_per_token = bt

        # Effective length per token, not the constant 2048: at short context
        # the indexer pads most of the index list with -1, and a constant tlen
        # makes the kernel walk every padded slot. Profiled at 48% of ALL
        # decode GPU time before this. last-valid-position+1 is exact even if
        # -1s were interleaved rather than trailing.
        tlen = quixicore_ops.sparse_topk_tlen(idx)

        if kv_c_and_k_pe_cache.dtype == torch.bfloat16:
            if q.shape[-1] == 512:
                # NoPE MLA (glm5_next): no rope segment, 512-wide latents.
                # Partitioned (128 -> 17 partitions at the 2080-wide list):
                # the one-warp-per-(head, token) walk was 155 ms/token at
                # TP8 with the 2048 top-k (8 warps per rank). Microbench
                # H=8 L=2048: unpartitioned 1138 us, P256 138 us, P128 74 us
                # per call with the vectorized bf16 row path.
                return quixicore_ops.mla_decode_bf16_sparse_nope(
                    q, kv_c_and_k_pe_cache.reshape(-1), bt, idx, tlen,
                    attn_metadata.block_size, self.softmax_scale,
                    partition_size=_bf16_partition(q, idx),
                ), None
            return quixicore_ops.mla_decode_bf16_sparse_glm(
                q, kv_c_and_k_pe_cache.reshape(-1), bt, idx, tlen,
                attn_metadata.block_size, self.softmax_scale,
                partition_size=_bf16_partition(q, idx),
            ), None

        if self._k_scale_host is None:
            ks = getattr(layer, "_k_scale", None) if layer is not None else None
            if ks is None:
                self._k_scale_host = 1.0
            else:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "k_scale not cached before CUDA graph capture; reading "
                        "it here would be a D2H sync. Run an eager warmup first."
                    )
                self._k_scale_host = float(ks)
        k_scale = self._k_scale_host

        if splitq is not None:
            out = quixicore_ops.mla_decode_fp8_sparse_glm_splitq(
                splitq[0],
                splitq[1],
                kv_c_and_k_pe_cache.view(torch.uint8).reshape(-1),
                bt,
                idx,
                tlen,
                attn_metadata.block_size,
                self.softmax_scale,
                k_scale,
                partition_size=256,
            )
            _sparse_nan_debug(
                out, splitq[0].transpose(0, 1), kv_c_and_k_pe_cache, bt, idx,
                tlen, attn_metadata,
            )
            return out, None

        out = quixicore_ops.mla_decode_fp8_sparse_glm(
            q,
            kv_c_and_k_pe_cache.view(torch.uint8).reshape(-1),
            bt,
            idx,
            tlen,
            attn_metadata.block_size,
            self.softmax_scale,
            k_scale,
            partition_size=256,
        )
        _sparse_nan_debug(out, q, kv_c_and_k_pe_cache, bt, idx, tlen,
                          attn_metadata)
        return out, None


_SPARSE_NAN_DEBUG_STATE: dict = {}


def _sparse_nan_debug(out, q, kv_cache, bt, idx, tlen, attn_metadata) -> None:
    """Diagnostic (VLLM_DSV4_SPARSE_NAN_DEBUG=1): when a NaN row leaves the
    sparse decode kernel, dump that row's inputs — query health, top-k index
    census, effective length, and a byte-level census of the fp8 KV entries
    it gathered (0x7F/0xFF encode NaN in e4m3) — with a clean row as control.
    Syncs; diagnostic boots only.
    """
    state = _SPARSE_NAN_DEBUG_STATE
    if "on" not in state:
        import os
        state["on"] = os.getenv(
            "VLLM_DSV4_SPARSE_NAN_DEBUG", "0").lower() in ("1", "true", "on")
        state["dumps"] = 0
    if not state["on"] or state["dumps"] >= 6:
        return
    nan_rows = torch.isnan(out.float().reshape(out.shape[0], -1)).any(dim=-1)
    if not bool(nan_rows.any()):
        return
    state["dumps"] += 1
    rows = nan_rows.nonzero().flatten().tolist()
    clean_rows = (~nan_rows).nonzero().flatten().tolist()
    control = clean_rows[0] if clean_rows else None
    blk = attn_metadata.block_size
    kv_bytes = kv_cache.view(torch.uint8)
    entry = kv_bytes.shape[-1]

    def row_report(r: int) -> str:
        q_nan = bool(torch.isnan(q[r].float()).any())
        row_idx = idx[r]
        valid = row_idx[row_idx >= 0]
        t = int(tlen[r]) if tlen.numel() > r else -1
        if valid.numel() == 0:
            return (f"row {r}: q_nan={q_nan} tlen={t} no valid indices")
        blocks = bt[r][(valid // blk).long()]
        flat = blocks.long() * blk + (valid % blk).long()
        gathered = kv_bytes.reshape(-1, entry)[flat]
        fp8_part = gathered[:, :512]
        nan_bytes = ((fp8_part == 0x7F) | (fp8_part == 0xFF)).sum().item()
        per_tok = ((fp8_part == 0x7F) | (fp8_part == 0xFF)).any(dim=-1)
        bad_tokens = int(per_tok.sum())
        worst = valid[per_tok.nonzero().flatten()[:8]].tolist() if bad_tokens else []
        return (f"row {r}: q_nan={q_nan} tlen={t} n_idx={int(valid.numel())} "
                f"idx_max={int(valid.max())} nan_bytes={nan_bytes} "
                f"tokens_with_nan_bytes={bad_tokens}/{int(valid.numel())} "
                f"first_bad_positions={worst}")

    reports = [row_report(r) for r in rows[:3]]
    ctl = row_report(control) if control is not None else "no clean row"
    logger.error(
        "SPARSE_NAN_DEBUG dump %d: batch=%d nan_rows=%s | %s | CONTROL %s",
        state["dumps"], out.shape[0], rows[:8], " | ".join(reports), ctl)
