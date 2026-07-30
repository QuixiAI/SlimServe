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

        # TODO(quixicore-cuda): spec-decode uniform-batch handling and the
        # CUDA-graph capture path need the same buffer-persistence treatment
        # as the ROCm builder before FULL_DECODE_ONLY graphs are enabled.
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
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = torch.cat([ql_nope, q_pe], dim=-1)

        num_tokens = attn_metadata.num_actual_tokens
        q = q[:num_tokens].contiguous()
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        topk = topk_indices.shape[-1]
        idx = topk_indices.to(torch.int32).contiguous()

        # One kernel "batch" entry per query token; each needs its request's
        # block table row.
        bt = attn_metadata.block_table.index_select(
            0, attn_metadata.req_id_per_token[:num_tokens].to(torch.int32)
        ).to(torch.int32).contiguous()

        tlen = torch.full(
            (num_tokens,), topk, dtype=torch.int32, device=q.device
        )

        if kv_c_and_k_pe_cache.dtype == torch.bfloat16:
            return quixicore_ops.mla_decode_bf16_sparse_glm(
                q, kv_c_and_k_pe_cache.reshape(-1), bt, idx, tlen,
                attn_metadata.block_size, self.softmax_scale,
            ), None

        k_scale = 1.0
        if layer is not None and getattr(layer, "_k_scale", None) is not None:
            k_scale = float(layer._k_scale)

        out = quixicore_ops.mla_decode_fp8_sparse_glm(
            q,
            kv_c_and_k_pe_cache.view(torch.uint8).reshape(-1),
            bt,
            idx,
            tlen,
            attn_metadata.block_size,
            self.softmax_scale,
            k_scale,
        )
        return out, None
