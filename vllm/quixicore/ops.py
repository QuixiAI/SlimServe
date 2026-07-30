# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrappers over the vendored QuixiCore-CUDA kernels (`vllm._quixicore_C`).

Mirrors the role of `rocm_aiter_ops` on ROCm: implemented ops are thin
pass-throughs to the compiled extension; ops the library does not provide yet
are stubs that delegate to a slower in-tree fallback so the serving path stays
functional on Ampere while the native kernel is written.

Stub inventory (TODO(quixicore-cuda), tracked for native implementation):
- ``fp8_mqa_logits``: DSA indexer prefill logits. DeepGEMM provides this on
  sm90+; on sm80 we fall back to the pure-torch reference.
- ``fp8_paged_mqa_logits``: DSA indexer decode logits against the paged fp8
  K cache. Same fallback situation.
- GGUF MoE grouped GEMM with sm80-tuned tiles: not a stub here — the in-tree
  ``torch.ops._C.ggml_moe_a8`` builds for CUDA with upstream (untuned) tiles;
  an Ampere-tuned QuixiCore kernel is a later perf project.
"""

from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


@cache
def _qc():
    import vllm._quixicore_C as qc

    return qc


class quixicore_ops:
    @staticmethod
    @cache
    def is_available() -> bool:
        try:
            _qc()
            return True
        except ImportError as e:
            logger.debug("QuixiCore-CUDA extension unavailable: %s", e)
            return False

    # ------------------------------------------------------------------
    # Sparse MLA decode (the AITER `mla_decode_fwd` counterpart on sm80)
    # ------------------------------------------------------------------

    @staticmethod
    def mla_decode_fp8_sparse(
        q: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor,
        block_size: int,
        sm_scale: float,
        partition_size: int = 0,
    ) -> torch.Tensor:
        """Top-k-gathered sparse MQA decode over the paged fp8 MLA latent.

        Args:
            q: [batch, heads, 576] bf16 queries (512 latent + 64 rope).
            kv_data: packed uint8 e4m3 paged latent cache.
            kv_scale: packed uint8 per-block scales.
            block_table: [batch, max_blocks] int32.
            topk_indices: [batch, topk] int32 global token indices (-1 padded).
            topk_length: [batch] int32 valid counts.
            block_size: KV block size (64 for this model).
            sm_scale: softmax scale.
            partition_size: >0 enables the split-K partitioned variant.

        Returns:
            [batch, heads, 512] bf16 attention output.
        """
        return _qc().mla_decode_fp8_sparse(
            q,
            kv_data,
            kv_scale,
            block_table,
            topk_indices,
            topk_length,
            block_size,
            sm_scale,
            partition_size,
        )

    @staticmethod
    def mla_decode_fp8_sparse_glm(
        q: torch.Tensor,
        kv_cache_u8: torch.Tensor,
        block_table: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        block_size: int,
        scale: float,
        kv_scale: float,
    ) -> torch.Tensor:
        """GLM-5.2-Vision sparse MLA decode (576 fp8 slot, value = leading 512).

        `indices` are request-local logical token positions resolved through
        `block_table`; -1 entries are skipped. Returns [tokens, heads, 512].
        """
        return _qc().mla_decode_fp8_sparse_glm(
            q, kv_cache_u8, block_table, indices, topk_length, block_size, scale,
            kv_scale
        )

    @staticmethod
    def mla_decode_bf16_sparse_glm(
        q: torch.Tensor,
        kv: torch.Tensor,
        block_table: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        """GLM sparse MLA decode over a bf16 latent cache (576 bf16 per slot).

        This is the geometry that runs on Ampere: sm80 has no native fp8e4nv, so
        vLLM cannot store an fp8 KV cache there. Returns [tokens, heads, 512].
        """
        return _qc().mla_decode_bf16_sparse_glm(
            q, kv, block_table, indices, topk_length, block_size, scale
        )

    @staticmethod
    def mla_decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        sm_scale: float,
    ) -> torch.Tensor:
        """Dense bf16 MLA decode (non-sparse fallback / draft-model use)."""
        return _qc().mla_decode(
            q, kv_cache, block_table, seq_lens, block_size, sm_scale
        )

    @staticmethod
    def mla_kv_insert(
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        _qc().mla_kv_insert(
            kv_c, k_pe, cos, sin, kv_cache, slot_mapping, block_size
        )

    # ------------------------------------------------------------------
    # DSA lightning indexer
    # ------------------------------------------------------------------

    @staticmethod
    def indexer_k_quant(
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_blocks: int,
        head_dim: int,
        quant_block_size: int,
        cache_block_size: int,
        ue8m0: bool = False,
    ) -> torch.Tensor:
        """fp8-quantize indexer K into the packed paged cache."""
        return _qc().indexer_k_quant(
            k,
            slot_mapping,
            num_blocks,
            head_dim,
            quant_block_size,
            cache_block_size,
            ue8m0,
        )

    @staticmethod
    def cp_gather_indexer(*args, **kwargs):
        return _qc().cp_gather_indexer(*args, **kwargs)

    # ------------------------------------------------------------------
    # Stubs: kernels QuixiCore-CUDA does not implement yet.
    # Each delegates to the in-tree pure-torch reference so the path works
    # (slowly) on sm80 today and can be swapped for the native kernel
    # without touching callers.
    # ------------------------------------------------------------------

    @staticmethod
    def fp8_mqa_logits(
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        """DSA indexer prefill logits.

        TODO(quixicore-cuda): native sm80 kernel (DeepGEMM `fp8_fp4_mqa_logits`
        equivalent; fp8 held as bytes, upconverted to bf16 in-register — no
        fp8 tensor cores on Ampere). Until then: pure-torch reference.
        """
        if hasattr(_qc(), "fp8_mqa_logits"):
            return _qc().fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            fp8_mqa_logits_torch,
        )

        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

    @staticmethod
    def fp8_paged_mqa_logits(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        max_model_len: int,
    ) -> torch.Tensor:
        """DSA indexer decode logits over the paged fp8 K cache.

        TODO(quixicore-cuda): native sm80 kernel. Until then: pure-torch
        reference.
        """
        if hasattr(_qc(), "fp8_paged_mqa_logits"):
            return _qc().fp8_paged_mqa_logits(
                q, kv_cache, weights, context_lens, block_tables, max_model_len
            )
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            fp8_paged_mqa_logits_torch,
        )

        return fp8_paged_mqa_logits_torch(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )
