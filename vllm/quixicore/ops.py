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
    def mla_decode_fp8_sparse_glm_splitq(
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache_u8: torch.Tensor,
        block_table: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        block_size: int,
        scale: float,
        kv_scale: float,
        partition_size: int = 0,
    ) -> torch.Tensor:
        """GLM sparse MLA decode reading q from its two source buffers.

        `q_nope` is head-major [heads, tokens, 512] (the natural bmm output,
        pre-transpose) and `q_pe` is [tokens, heads, 64]; this replaces the
        per-layer torch.cat with direct reads -- bitwise identical to the
        concatenated path. Returns [tokens, heads, 512].
        """
        return _qc().mla_decode_fp8_sparse_glm_splitq(
            q_nope, q_pe, kv_cache_u8, block_table, indices, topk_length,
            block_size, scale, kv_scale, partition_size,
        )

    @staticmethod
    def moe_weighted_sum(
        x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
    ) -> None:
        """out[t] = sum_k w[t, k] * x[t, k] with float accumulation.

        Fuses the out.mul_(topk_weights) + moe_sum pair into one launch; one
        fewer bf16 rounding than the pair (tolerance <= K ulp).
        """
        _qc().moe_weighted_sum(x, w, out)

    @staticmethod
    def sparse_topk_tlen(indices: torch.Tensor) -> torch.Tensor:
        """Effective top-k length per row: last valid (>= 0) index position + 1.

        One kernel launch replacing the (idx >= 0) * arange -> amax chain.
        `indices` is [rows, topk] int32; returns [rows] int32.
        """
        return _qc().sparse_topk_tlen(indices)

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
        partition_size: int = 0,
    ) -> torch.Tensor:
        """GLM-5.2-Vision sparse MLA decode (576 fp8 slot, value = leading 512).

        `indices` are request-local logical token positions resolved through
        `block_table`; -1 entries are skipped. Returns [tokens, heads, 512].
        partition_size > 0 splits the index list over blockIdx.z with an exact
        online-softmax merge; the unpartitioned form is one warp per
        (token, head) and starves the GPU at small batch.
        """
        return _qc().mla_decode_fp8_sparse_glm(
            q,
            kv_cache_u8,
            block_table,
            indices,
            topk_length,
            block_size,
            scale,
            kv_scale,
            partition_size,
        )

    @staticmethod
    def has(name: str) -> bool:
        """Whether the compiled extension exposes `name`."""
        try:
            return hasattr(_qc(), name)
        except ImportError:
            return False

    @staticmethod
    def indexer_metadata(
        query_start_loc: torch.Tensor,
        uncompressed_seq_lens: torch.Tensor,
        cu_compressed_seq_lens: torch.Tensor,
        row_start_cu: torch.Tensor,
        token_to_seq: torch.Tensor,
        cu_ks: torch.Tensor,
        cu_ke: torch.Tensor,
        query_slice_start: int,
        query_slice_stop: int,
        dcp_rank: int,
        dcp_world: int,
        dcp_interleave: int,
        compress_ratio: int,
    ) -> None:
        """Native CUDA DSA indexer metadata (replaces a Triton kernel)."""
        _qc().indexer_metadata(
            query_start_loc,
            uncompressed_seq_lens,
            cu_compressed_seq_lens,
            row_start_cu,
            token_to_seq,
            cu_ks,
            cu_ke,
            query_slice_start,
            query_slice_stop,
            dcp_rank,
            dcp_world,
            dcp_interleave,
            compress_ratio,
        )

    @staticmethod
    def compute_slot_mapping(
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_tokens: int,
        max_num_tokens: int,
        block_size: int,
        kv_cache_block_size: int,
        blocks_per_kv_block: int,
        cp_world: int,
        cp_rank: int,
        cp_interleave: int,
        pad_id: int,
    ) -> None:
        """Native CUDA token -> KV-slot mapping (replaces a Triton kernel)."""
        _qc().compute_slot_mapping(
            query_start_loc,
            positions,
            block_table,
            slot_mapping,
            num_tokens,
            max_num_tokens,
            block_size,
            kv_cache_block_size,
            blocks_per_kv_block,
            cp_world,
            cp_rank,
            cp_interleave,
            pad_id,
        )

    # ------------------------------------------------------------------
    # V2 model-runner batch-prep (native replacements for Triton kernels)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_write(
        out: torch.Tensor,
        row_stride: int,
        indices: torch.Tensor,
        starts: torch.Tensor,
        contents: torch.Tensor,
        cu_lens: torch.Tensor,
    ) -> None:
        """Native `_apply_write_kernel` (MULTI_GROUP=False)."""
        _qc().apply_write(out, row_stride, indices, starts, contents, cu_lens)

    @staticmethod
    def apply_write_multi(
        out_ptrs: torch.Tensor,
        out_strides: torch.Tensor,
        indices: torch.Tensor,
        starts: torch.Tensor,
        contents: torch.Tensor,
        cu_lens: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> None:
        """Native `_apply_write_kernel` (MULTI_GROUP=True, int32 elements)."""
        _qc().apply_write_multi(
            out_ptrs, out_strides, indices, starts, contents, cu_lens, group_ids
        )

    @staticmethod
    def prepare_pos_seq_lens(
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        max_num_reqs: int,
    ) -> None:
        """Native `_prepare_pos_seq_lens_kernel`."""
        _qc().prepare_pos_seq_lens(
            pos,
            seq_lens,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            max_num_reqs,
        )

    @staticmethod
    def prepare_prefill_inputs(
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        all_token_ids: torch.Tensor,
        all_token_ids_stride: int,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Native `_prepare_prefill_inputs_kernel` (input_batch.py)."""
        _qc().prepare_prefill_inputs(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            all_token_ids_stride,
            prefill_lens,
            num_computed_tokens,
        )

    @staticmethod
    def combine_sampled_and_draft_tokens(
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_tokens_stride: int,
        cu_num_logits: torch.Tensor,
        logits_indices: torch.Tensor,
        num_new_sampled_tokens: int,
    ) -> None:
        """Native `_combine_sampled_and_draft_tokens_kernel`."""
        _qc().combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens_stride,
            cu_num_logits,
            logits_indices,
            num_new_sampled_tokens,
        )

    @staticmethod
    def get_num_sampled_and_rejected(
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        prefill_len: torch.Tensor,
    ) -> None:
        """Native `_get_num_sampled_and_rejected_kernel`."""
        _qc().get_num_sampled_and_rejected(
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )

    @staticmethod
    def post_update(
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        output_bin_counts: torch.Tensor | None,
        output_bin_counts_stride: int,
        sampled_tokens: torch.Tensor,
        sampled_tokens_stride: int,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        all_token_ids: torch.Tensor,
        all_token_ids_stride: int,
        total_len: torch.Tensor,
    ) -> None:
        """Native `_post_update_kernel`."""
        _qc().post_update(
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            output_bin_counts_stride,
            sampled_tokens,
            sampled_tokens_stride,
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            all_token_ids_stride,
            total_len,
        )

    @staticmethod
    def post_update_num_computed_tokens(
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        """Native `_post_update_num_computed_tokens_kernel`."""
        _qc().post_update_num_computed_tokens(
            idx_mapping, num_computed_tokens, query_start_loc
        )

    @staticmethod
    def expand_idx_mapping(
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
    ) -> None:
        """Native `_expand_idx_mapping_kernel`."""
        _qc().expand_idx_mapping(
            idx_mapping, expanded_idx_mapping, expanded_local_pos, cu_num_logits
        )

    @staticmethod
    def gather_block_tables(
        idx_mapping: torch.Tensor,
        src_ptrs: torch.Tensor,
        dst_ptrs: torch.Tensor,
        strides: torch.Tensor,
        num_blocks: torch.Tensor,
        num_blocks_stride: int,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> None:
        """Native `_gather_block_tables_kernel` (V2 worker block tables)."""
        _qc().gather_block_tables(
            idx_mapping,
            src_ptrs,
            dst_ptrs,
            strides,
            num_blocks,
            num_blocks_stride,
            num_reqs,
            num_reqs_padded,
        )

    @staticmethod
    def compute_slot_mappings(
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        pos: torch.Tensor,
        block_table_ptrs: torch.Tensor,
        block_table_strides: torch.Tensor,
        block_sizes: torch.Tensor,
        slot_mappings: torch.Tensor,
        slot_mappings_stride: int,
        max_num_tokens: int,
        cp_rank: int,
        cp_size: int,
        cp_interleave: int,
        pad_id: int,
    ) -> None:
        """Native `_compute_slot_mappings_kernel` (V2 worker, multi-group)."""
        _qc().compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            pos,
            block_table_ptrs,
            block_table_strides,
            block_sizes,
            slot_mappings,
            slot_mappings_stride,
            max_num_tokens,
            cp_rank,
            cp_size,
            cp_interleave,
            pad_id,
        )

    @staticmethod
    def prepare_uniform_decode(
        seq_lens: torch.Tensor,
        decode_seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_table_stride: int,
        expanded_block_table: torch.Tensor,
        expanded_bt_stride: int,
        decode_lens: torch.Tensor,
        max_decode_len: int,
        num_decode_tokens: int,
    ) -> None:
        """Native `_prepare_uniform_decode_kernel` (DSA indexer)."""
        _qc().prepare_uniform_decode(
            seq_lens,
            decode_seq_lens,
            block_table,
            block_table_stride,
            expanded_block_table,
            expanded_bt_stride,
            decode_lens,
            max_decode_len,
            num_decode_tokens,
        )

    @staticmethod
    def prepare_dflash_inputs(*args, **kwargs) -> None:
        """Native `_prepare_dflash_inputs_kernel` (DFlash speculator)."""
        _qc().prepare_dflash_inputs(*args, **kwargs)

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
        _qc().mla_kv_insert(kv_c, k_pe, cos, sin, kv_cache, slot_mapping, block_size)

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
    # V2 model-runner sampler / spec-decode kernels (bit-exact ports of
    # the Triton kernels in vllm/v1/worker/gpu/sample and
    # spec_decode/rejection_sampler_utils.py).
    # ------------------------------------------------------------------

    @staticmethod
    def v2_apply_temperature(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        _qc().v2_apply_temperature(logits, expanded_idx_mapping, temperature)

    @staticmethod
    def v2_gumbel_sample(
        local_argmax: torch.Tensor,
        local_max: torch.Tensor,
        processed_logits: torch.Tensor | None,
        processed_logits_col: torch.Tensor | None,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        seeds: torch.Tensor,
        pos: torch.Tensor,
        temperature: torch.Tensor,
        apply_temperature: bool,
        per_token_col: bool,
    ) -> None:
        """Per-(token, vocab-block) Gumbel-max partials; the caller reduces."""
        _qc().v2_gumbel_sample(
            local_argmax,
            local_max,
            processed_logits,
            processed_logits_col,
            logits,
            expanded_idx_mapping,
            seeds,
            pos,
            temperature,
            apply_temperature,
            per_token_col,
        )

    @staticmethod
    def v2_topk_log_softmax(
        out: torch.Tensor, logits: torch.Tensor, topk_ids: torch.Tensor
    ) -> None:
        _qc().v2_topk_log_softmax(out, logits, topk_ids)

    @staticmethod
    def v2_ranks(
        out: torch.Tensor, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> None:
        _qc().v2_ranks(out, logits, token_ids)

    @staticmethod
    def v2_fill_logprob_token_ids(
        out_token_ids: torch.Tensor,
        out_valid_mask: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        topk_indices: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        num_per_req_token_ids: torch.Tensor,
        per_req_token_ids: torch.Tensor,
        num_topk: int,
    ) -> None:
        _qc().v2_fill_logprob_token_ids(
            out_token_ids,
            out_valid_mask,
            sampled_token_ids,
            topk_indices,
            expanded_idx_mapping,
            num_per_req_token_ids,
            per_req_token_ids,
            num_topk,
        )

    @staticmethod
    def v2_penalties(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        token_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        repetition_penalty: torch.Tensor,
        frequency_penalty: torch.Tensor,
        presence_penalty: torch.Tensor,
        prompt_bin_mask: torch.Tensor,
        output_bin_counts: torch.Tensor,
    ) -> None:
        _qc().v2_penalties(
            logits,
            expanded_idx_mapping,
            token_ids,
            expanded_local_pos,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            prompt_bin_mask,
            output_bin_counts,
        )

    @staticmethod
    def v2_bincount(
        expanded_idx_mapping: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        prefill_len: torch.Tensor,
        prompt_bin_mask: torch.Tensor,
        output_bin_counts: torch.Tensor,
        max_prefill_len: int,
    ) -> None:
        _qc().v2_bincount(
            expanded_idx_mapping,
            all_token_ids,
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            output_bin_counts,
            max_prefill_len,
        )

    @staticmethod
    def v2_prompt_logprobs_token_ids(
        out: torch.Tensor,
        query_start_loc: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        all_token_ids: torch.Tensor,
    ) -> None:
        _qc().v2_prompt_logprobs_token_ids(
            out,
            query_start_loc,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )

    @staticmethod
    def v2_rejection_sample(
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        target_rejected_lse: torch.Tensor,
        draft_rejected_lse: torch.Tensor,
        target_logits: torch.Tensor,
        t_local_argmax: torch.Tensor,
        t_local_max: torch.Tensor,
        t_local_sumexp: torch.Tensor,
        draft_sampled: torch.Tensor,
        draft_logits: torch.Tensor | None,
        d_local_max: torch.Tensor,
        d_local_sumexp: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        vocab_num_blocks: int,
    ) -> None:
        """Leviathan-style rejection loop (no block verification/synthetic)."""
        _qc().v2_rejection_sample(
            sampled,
            num_sampled,
            target_rejected_lse,
            draft_rejected_lse,
            target_logits,
            t_local_argmax,
            t_local_max,
            t_local_sumexp,
            draft_sampled,
            draft_logits,
            d_local_max,
            d_local_sumexp,
            cu_num_logits,
            idx_mapping,
            temperature,
            seed,
            pos,
            vocab_num_blocks,
        )

    @staticmethod
    def v2_resample(
        rl_argmax: torch.Tensor,
        rl_max: torch.Tensor,
        target_logits: torch.Tensor,
        target_rejected_lse: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_rejected_lse: torch.Tensor,
        rejected_step: torch.Tensor,
        cu_num_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        draft_sampled: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        vocab_size: int,
    ) -> None:
        _qc().v2_resample(
            rl_argmax,
            rl_max,
            target_logits,
            target_rejected_lse,
            draft_logits,
            draft_rejected_lse,
            rejected_step,
            cu_num_logits,
            expanded_idx_mapping,
            draft_sampled,
            temperature,
            seed,
            pos,
            vocab_size,
        )

    @staticmethod
    def v2_grammar_bitmask(
        logits: torch.Tensor,
        logits_indices: torch.Tensor,
        bitmask: torch.Tensor,
        num_masks: int,
    ) -> None:
        _qc().v2_grammar_bitmask(logits, logits_indices, bitmask, num_masks)

    @staticmethod
    def v2_min_p(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        min_p: torch.Tensor,
    ) -> None:
        _qc().v2_min_p(logits, expanded_idx_mapping, min_p)

    @staticmethod
    def v2_logit_bias(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        pos: torch.Tensor,
        num_allowed_token_ids: torch.Tensor,
        allowed_token_ids: torch.Tensor,
        num_logit_bias: torch.Tensor,
        logit_bias_token_ids: torch.Tensor,
        logit_bias: torch.Tensor,
        min_lens: torch.Tensor,
        num_stop_token_ids: torch.Tensor,
        stop_token_ids: torch.Tensor,
    ) -> None:
        _qc().v2_logit_bias(
            logits,
            expanded_idx_mapping,
            pos,
            num_allowed_token_ids,
            allowed_token_ids,
            num_logit_bias,
            logit_bias_token_ids,
            logit_bias,
            min_lens,
            num_stop_token_ids,
            stop_token_ids,
        )

    @staticmethod
    def v2_bad_words(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        bad_word_token_ids: torch.Tensor,
        bad_word_offsets: torch.Tensor,
        num_bad_words: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        total_len: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        _qc().v2_bad_words(
            logits,
            expanded_idx_mapping,
            bad_word_token_ids,
            bad_word_offsets,
            num_bad_words,
            all_token_ids,
            prompt_len,
            total_len,
            input_ids,
            expanded_local_pos,
        )

    @staticmethod
    def v2_local_logits_stats(
        t_local_argmax: torch.Tensor,
        t_local_max: torch.Tensor,
        t_local_sumexp: torch.Tensor,
        d_local_max: torch.Tensor,
        d_local_sumexp: torch.Tensor,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        vocab_size: int,
        num_speculative_steps: int,
    ) -> None:
        _qc().v2_local_logits_stats(
            t_local_argmax,
            t_local_max,
            t_local_sumexp,
            d_local_max,
            d_local_sumexp,
            target_logits,
            draft_logits,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_speculative_steps,
        )

    @staticmethod
    def v2_insert_resampled(
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        rl_argmax: torch.Tensor,
        rl_max: torch.Tensor,
        resample_num_blocks: int,
        cu_num_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        _qc().v2_insert_resampled(
            sampled,
            num_sampled,
            rl_argmax,
            rl_max,
            resample_num_blocks,
            cu_num_logits,
            expanded_idx_mapping,
            temperature,
        )

    @staticmethod
    def v2_flatten_sampled(
        flat_sampled: torch.Tensor,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
    ) -> None:
        _qc().v2_flatten_sampled(flat_sampled, sampled, num_sampled, cu_num_logits)

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
            k, kscale = kv
            return _qc().fp8_mqa_logits(
                q,
                k,
                kscale.view(torch.float32).reshape(-1).contiguous(),
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
            )
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

    # ------------------------------------------------------------------
    # ROCm sparse-MLA indexer index arithmetic (no CUDA counterpart)
    # ------------------------------------------------------------------

    @staticmethod
    def convert_req_index_to_global_index(
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        block_size: int,
        topk: int,
    ) -> None:
        """Map request-local top-k positions to global paged-KV slots.

        Writes `out` ragged, packed at `cu_seqlens`. An invalid token or an
        out-of-range block id yields 0, matching the Triton kernel body (whose
        docstring says -1).
        """
        _qc().convert_req_index_to_global_index(
            req_id, block_table, token_indices, cu_seqlens, out, block_size, topk
        )

    @staticmethod
    def generate_sparse_seqlen(
        seq_lens: torch.Tensor,
        cu_query_lens: torch.Tensor,
        out: torch.Tensor,
        topk_token: int,
    ) -> None:
        """Per-query-token sparse KV length, clamped to `topk_token`.

        `out` must be zero-initialized: rows with seq_len == 0 are skipped.
        """
        _qc().generate_sparse_seqlen(seq_lens, cu_query_lens, out, topk_token)

    @staticmethod
    def indexer_k_quant_and_cache(
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_scale: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        block_tile_size: int,
        head_tile_size: int,
        fp8_max: float,
        shuffle: bool,
    ) -> None:
        """Per-token fp8 quantize of the indexer K vector into the paged cache."""
        _qc().indexer_k_quant_and_cache(
            k,
            kv_cache,
            kv_cache_scale,
            slot_mapping,
            block_size,
            block_tile_size,
            head_tile_size,
            fp8_max,
            1 if shuffle else 0,
        )

    @staticmethod
    def cp_gather_indexer_quant_cache(
        kv_cache: torch.Tensor,
        kv_cache_scale: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlen: torch.Tensor,
        token_to_seq: torch.Tensor,
        block_size: int,
        block_tile_size: int,
        head_tile_size: int,
        num_batches: int,
        num_blocks: int,
        shuffle: bool,
    ) -> None:
        """Gather quantized indexer K and its scale out of the paged cache."""
        _qc().cp_gather_indexer_quant_cache(
            kv_cache,
            kv_cache_scale,
            k_fp8,
            k_scale,
            block_table,
            cu_seqlen,
            token_to_seq,
            block_size,
            block_tile_size,
            head_tile_size,
            num_batches,
            num_blocks,
            1 if shuffle else 0,
        )

    @staticmethod
    def mqa_logits_gfx942(
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_scales: torch.Tensor,
        weights: torch.Tensor,
        cu_start: torch.Tensor,
        cu_end: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """DSA indexer MQA logits, bitwise-equal to the Triton kernel.

        `logits` must arrive pre-filled with -inf: positions outside
        [cu_start, cu_end) are left untouched, matching AITER's semantics.
        """
        _qc().mqa_logits_gfx942(q, kv, kv_scales, weights, cu_start, cu_end, logits)
