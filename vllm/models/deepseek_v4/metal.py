# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple Metal implementation boundary for DeepSeek-V4 sparse MLA."""

import os
from typing import Any, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.quixicore.ops import quixicore_ops
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# Marshalling-memo kill switches (default on; =0 restores per-call
# marshalling — bit-exact either way). The guarded branches are retained
# as-is: removing them (together with the phaseprof brackets in
# compressor.py) shifts per-layer host timing enough to flip the
# async-output completion race documented in vllm/platforms/metal_compat.py.
_QC_MEMO_POS = os.environ.get("VLLM_QC_MEMO_POS", "1") != "0"
# Dense-causal prefill MMA FA (mla_prefill_fa_mma): replaces the decode-shape
# candidate walk at prefill widths on single-request prefill steps. ULP class
# (P tile rounds to half, block-level softmax reassociation).
# VLLM_QC_MLA_PREFILL_FA_MMA=0 restores the decode-shape walk bit-exactly —
# the retained ULP-reversion sentinel.
_QC_PREFILL_FA_MMA = os.environ.get("VLLM_QC_MLA_PREFILL_FA_MMA", "1") != "0"
_QC_MEMO_WOA = os.environ.get("VLLM_QC_MEMO_WOA", "1") != "0"


class DeepseekV4MetalSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "METAL_FLASHMLA_SPARSE_DSV4"


# Slot-table builders shared by forward_mqa (below) and the step tape
# (metal_tape.py). Verbatim extractions from forward_mqa so both paths are
# bit-identical by construction; forward_mqa's per-step pass cache still
# wraps them.


def build_swa_tables(swa_metadata, positions, num_tokens, window_size, device):
    token_to_req_indices = swa_metadata.token_to_req_indices
    valid_token = swa_metadata.is_valid_token
    assert token_to_req_indices is not None and valid_token is not None
    req_ids = token_to_req_indices[:num_tokens].to(torch.long)
    valid_tokens = valid_token[:num_tokens]

    swa_width = window_size
    swa_offsets = torch.arange(swa_width, device=device, dtype=positions.dtype)
    swa_lens = torch.minimum(
        positions + 1,
        positions.new_full(positions.shape, swa_width),
    )
    swa_start = positions + 1 - swa_lens
    swa_pos = swa_start.unsqueeze(1) + swa_offsets.unsqueeze(0)
    swa_valid = swa_offsets.unsqueeze(0) < swa_lens.unsqueeze(1)
    swa_blocks = swa_metadata.block_table.index_select(0, req_ids)
    swa_block_col = torch.div(
        swa_pos, swa_metadata.block_size, rounding_mode="floor"
    ).clamp(min=0, max=swa_blocks.shape[1] - 1)
    swa_block = swa_blocks.gather(1, swa_block_col.to(torch.long))
    swa_slots = swa_block * swa_metadata.block_size + torch.remainder(
        swa_pos, swa_metadata.block_size
    )
    swa_slots = (
        torch.where(swa_valid, swa_slots, -1).to(torch.int32).contiguous()
    )
    swa_lens = (
        torch.where(valid_tokens, swa_lens, 0).to(torch.int32).contiguous()
    )
    return (req_ids, valid_tokens, swa_slots, swa_lens)


def build_comp_tables(
    layer_metadata,
    positions,
    req_ids,
    valid_tokens,
    compress_ratio,
    compressed_width,
    num_tokens,
    device,
):
    compressed_offsets = torch.arange(
        compressed_width, device=device, dtype=positions.dtype
    )
    compressed_lens = torch.minimum(
        torch.div(
            positions + 1,
            compress_ratio,
            rounding_mode="floor",
        ),
        positions.new_full(positions.shape, compressed_width),
    )
    compressed_valid = compressed_offsets.unsqueeze(
        0
    ) < compressed_lens.unsqueeze(1)
    compressed_blocks = layer_metadata.block_table.index_select(0, req_ids)
    storage_block_size = layer_metadata.block_size // compress_ratio
    compressed_block_col = torch.div(
        compressed_offsets,
        storage_block_size,
        rounding_mode="floor",
    ).clamp(max=compressed_blocks.shape[1] - 1)
    compressed_block = compressed_blocks.gather(
        1, compressed_block_col.unsqueeze(0).expand(num_tokens, -1)
    )
    compressed_slots = (
        compressed_block * storage_block_size
        + torch.remainder(compressed_offsets, storage_block_size)
    )
    compressed_slots = (
        torch.where(compressed_valid, compressed_slots, -1)
        .to(torch.int32)
        .contiguous()
    )
    compressed_lens = (
        torch.where(valid_tokens, compressed_lens, 0)
        .to(torch.int32)
        .contiguous()
    )
    return (compressed_slots, compressed_lens)


def build_comp_none(num_tokens, device):
    return (
        torch.full((num_tokens, 1), -1, dtype=torch.int32, device=device),
        torch.zeros(num_tokens, dtype=torch.int32, device=device),
    )


class DeepseekV4MetalAttention(DeepseekV4Attention):
    """DeepSeek-V4 attention hosted by QuixiCore-Metal kernels."""

    backend_cls = DeepseekV4MetalSparseBackend

    def __init__(self, *args, **kwargs) -> None:
        # The shared attention and indexer allocate synchronization events.
        # They do not run concurrently on Metal, but real MPS events keep the
        # common serial execution helper type-compatible.
        original_event = torch.cuda.Event
        torch.cuda.Event = torch.mps.Event  # type: ignore[misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = original_event  # type: ignore[misc]

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if not isinstance(attn_metadata, dict):
            return q
        swa_metadata = cast(
            DeepseekSparseSWAMetadata,
            attn_metadata[self.swa_cache_layer.prefix],
        )
        # Per-step marshalling memo: every layer passes the same positions /
        # slot_mapping, but the host op wants int32 / int64. Convert once on
        # the per-step metadata object so the host's .to() calls are no-ops
        # for the remaining layers (identical values — bit-exact).
        if _QC_MEMO_POS:
            pos32 = getattr(swa_metadata, "_qc_pos32", None)
            if pos32 is None or swa_metadata._qc_pos32_src is not positions:
                pos32 = positions.to(torch.int32).contiguous()
                swa_metadata._qc_pos32 = pos32
                swa_metadata._qc_pos32_src = positions
            slots64 = getattr(swa_metadata, "_qc_slots64", None)
            if slots64 is None:
                slots64 = swa_metadata.slot_mapping.to(torch.int64).contiguous()
                swa_metadata._qc_slots64 = slots64
        else:
            pos32 = positions
            slots64 = swa_metadata.slot_mapping
        # The op accepts fp16 q/kv directly: the half-input kernel variants
        # round each element to bf16 in-register (bit-identical to eager
        # casts).
        return quixicore_ops.deepseek_v4_qnorm_rope_kv_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            slots64,
            pos32,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        prequant_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # vLLM's memory-profile pass has no attention metadata or allocated KV
        # pages.  CUDA/ROCm reserve workspace inside their sparse op; Metal has
        # no such workspace yet, so keep the profile pass moving and let the
        # worker budget unified memory from actual allocations.
        if get_forward_context().attn_metadata is None:
            return torch.zeros_like(hidden_states)
        return super().forward(
            positions, hidden_states, llama_4_scaling, prequant_input
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # The DSV4 A projection is eight independent matrices.  GGUF stores
        # them as consecutive row groups; apply each group to its matching
        # attention heads, then feed the concatenated low-rank result to WO_B.
        cs = self.rotary_emb.cos_sin_cache
        if (
            o.dtype in (torch.float16, torch.bfloat16)
            and o.shape[1] % 8 == 0
            and o.stride(2) == 1
            and (cs.dtype == torch.float32 or cs.dtype == o.dtype)
            and quixicore_ops.has("dsv4_o_inv_rope")
        ):
            # Single-dispatch inverse RoPE (mirrors forward_native
            # inverse=True to per-op rounding).
            o_flat = quixicore_ops.dsv4_o_inv_rope(o, positions, cs)
            grouped = o_flat.view(o.shape[0], self.n_local_groups, -1)
        else:
            o_ref, _ = self.rotary_emb.forward_native(positions, o, key=None, inverse=True)  # type: ignore[call-arg]
            grouped = o_ref.reshape(o.shape[0], self.n_local_groups, -1)

        if hasattr(self.wo_a, "qweight"):
            from vllm.model_executor.layers.quantization.gguf import (
                fused_mul_mat_gguf,
            )

            qweight_type = self.wo_a.qweight_type.weight_type
            rows_per_group = self.o_lora_rank
            # The per-group qweight slices are constant after load;
            # memoize the list so the slice+contiguous chain runs once.
            groups = (
                getattr(self, "_qc_wo_a_groups", None) if _QC_MEMO_WOA else None
            )
            if groups is None:
                qweight = self.wo_a.qweight
                groups = [
                    qweight[
                        g * rows_per_group : (g + 1) * rows_per_group
                    ].contiguous()
                    for g in range(self.n_local_groups)
                ]
                if _QC_MEMO_WOA:
                    self._qc_wo_a_groups = groups
            projected = [
                fused_mul_mat_gguf(grouped[:, g], groups[g], qweight_type)
                for g in range(self.n_local_groups)
            ]
            z = torch.cat(projected, dim=-1)
        else:
            weight = self.wo_a.weight.reshape(
                self.n_local_groups, self.o_lora_rank, grouped.shape[-1]
            )
            z = torch.einsum("tgd,grd->tgr", grouped, weight).flatten(1)
        return self.wo_b(z)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        ctx = get_forward_context()
        if ctx.attn_metadata is None:
            output.zero_()
            return
        metadata = ctx.attn_metadata
        assert isinstance(metadata, dict)
        swa_metadata = cast(
            DeepseekSparseSWAMetadata,
            metadata[self.swa_cache_layer.prefix],
        )
        layer_metadata = cast(Any, metadata.get(self.prefix))
        num_tokens = q.shape[0]

        # The slot tables depend only on step-level metadata that every layer
        # of a KV-cache group shares, so build them once per forward pass and
        # reuse them for the remaining layers. The cache hangs off the forward
        # context, which vLLM recreates for each model execution, so entries
        # can never leak across steps or between target and draft passes. The
        # sparse-attention kernel only reads the tables.
        pass_cache = getattr(ctx, "_metal_mqa_cache", None)
        if pass_cache is None:
            pass_cache = {}
            ctx._metal_mqa_cache = pass_cache

        swa_key = ("swa", id(swa_metadata), self.window_size, num_tokens)
        swa_entry = pass_cache.get(swa_key)
        if swa_entry is None:
            swa_entry = build_swa_tables(
                swa_metadata, positions, num_tokens, self.window_size, q.device
            )
            pass_cache[swa_key] = swa_entry
        req_ids, valid_tokens, swa_slots, swa_lens = swa_entry

        dense_tables = False
        if self.compress_ratio > 1 and (
            layer_metadata is not None
            and self.topk_indices_buffer is not None
            and layer_metadata.max_seq_len // self.compress_ratio
            > self.topk_indices_buffer.shape[1]
        ):
            # Long context: consume the Lightning-indexer top-k indices this
            # layer's metal_sparse_attn_indexer just wrote (request-local
            # compressed positions, -1 padded, valid entries first). Never
            # cached in pass_cache — the buffer contents are per-layer. The
            # gate mirrors the producer's short-context branch in
            # attention.py exactly (buffer width == index_topk).
            storage_block_size = layer_metadata.block_size // self.compress_ratio
            local_idx = self.topk_indices_buffer[:num_tokens].to(torch.long)
            topk_bt = layer_metadata.block_table.index_select(0, req_ids).to(
                torch.long
            )
            block_cols = torch.div(
                local_idx, storage_block_size, rounding_mode="floor"
            ).clamp(min=0, max=max(topk_bt.shape[1] - 1, 0))
            blocks = topk_bt.gather(1, block_cols)
            topk_valid = local_idx >= 0
            compressed_slots = (
                torch.where(
                    topk_valid,
                    blocks * storage_block_size
                    + torch.remainder(local_idx, storage_block_size),
                    -1,
                )
                .to(torch.int32)
                .contiguous()
            )
            # The sparse kernel skips slot < 0 entries inside [0, len), so the
            # count is an upper bound; -1 pads sit at the tail (top-k order).
            compressed_lens = (
                torch.where(
                    valid_tokens,
                    topk_valid.sum(dim=1, dtype=torch.int32),
                    torch.zeros((), dtype=torch.int32, device=q.device),
                )
                .to(torch.int32)
                .contiguous()
            )
            compressed_cache = self.kv_cache
        elif self.compress_ratio > 1:
            assert layer_metadata is not None
            assert self.topk_indices_buffer is not None
            compressed_width = min(
                self.topk_indices_buffer.shape[1],
                (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio,
            )
            comp_key = (
                "comp",
                id(layer_metadata),
                self.compress_ratio,
                compressed_width,
                num_tokens,
            )
            comp_entry = pass_cache.get(comp_key)
            if comp_entry is None:
                comp_entry = build_comp_tables(
                    layer_metadata,
                    positions,
                    req_ids,
                    valid_tokens,
                    self.compress_ratio,
                    compressed_width,
                    num_tokens,
                    q.device,
                )
                pass_cache[comp_key] = comp_entry
            compressed_slots, compressed_lens = comp_entry
            compressed_cache = self.kv_cache
            dense_tables = True
        else:
            comp_key = ("comp_none", num_tokens)
            comp_entry = pass_cache.get(comp_key)
            if comp_entry is None:
                comp_entry = build_comp_none(num_tokens, q.device)
                pass_cache[comp_key] = comp_entry
            compressed_slots, compressed_lens = comp_entry
            compressed_cache = self.swa_cache_layer.kv_cache

        # Dense-causal prefill MMA FA: on single-request prefill steps the
        # compressed table is a shared causal prefix and the SWA table is a
        # band over the raw-position axis, so attention runs as a tiled
        # simdgroup-MMA FA over pre-decoded half scratches. Eligibility
        # comes from CPU-side metadata only — no mid-encode device syncs.
        if (
            _QC_PREFILL_FA_MMA
            and dense_tables
            and num_tokens >= 64
            and swa_metadata.num_decode_tokens == 0
            and swa_metadata.num_prefills == 1
            and swa_metadata.prefill_seq_lens_cpu is not None
            and swa_metadata.prefill_query_lens_cpu is not None
            and int(swa_metadata.prefill_query_lens_cpu[0]) == num_tokens
        ):
            pos_last = int(swa_metadata.prefill_seq_lens_cpu[0]) - 1
            pos0 = pos_last + 1 - num_tokens
            window = self.window_size
            axis0 = max(0, pos0 + 1 - window)
            cr = self.compress_ratio
            nc = (pos_last + 1) // cr
            if nc > 0:
                req = swa_metadata.num_decodes  # decodes first in the batch
                dev = q.device

                def _pad_slots(t: torch.Tensor) -> torch.Tensor:
                    n = t.numel()
                    npad = ((n + 31) // 32) * 32 + 32
                    out_t = torch.full(
                        (npad,), -1, dtype=torch.int32, device=t.device
                    )
                    out_t[:n] = t
                    return out_t

                # Step-shared position tensors (per step, layer-independent).
                fa_pos_key = ("fa_pos", pos0, pos_last, window, cr, axis0)
                fa_pos = pass_cache.get(fa_pos_key)
                if fa_pos is None:
                    pos_t = torch.arange(
                        pos0, pos_last + 1, device=dev, dtype=torch.int32
                    )
                    lens_c_t = torch.div(
                        pos_t + 1, cr, rounding_mode="floor"
                    ).to(torch.int32)
                    swa_len_t = torch.minimum(
                        pos_t + 1,
                        torch.full_like(pos_t, window),
                    )
                    lo_rel = (pos_t + 1 - swa_len_t - axis0).to(torch.int32)
                    hi_rel = (pos_t + 1 - axis0).to(torch.int32)
                    j_c = torch.arange(nc, device=dev, dtype=torch.int32)
                    p_s = torch.arange(
                        axis0, pos_last + 1, device=dev, dtype=torch.int32
                    )
                    fa_pos = (lens_c_t, lo_rel, hi_rel, j_c, p_s)
                    pass_cache[fa_pos_key] = fa_pos
                lens_c_t, lo_rel, hi_rel, j_c, p_s = fa_pos

                # Per-layer axis slot tables (block tables are per layer).
                fa_axis_key = ("fa_axis", id(layer_metadata), id(swa_metadata))
                fa_axis = pass_cache.get(fa_axis_key)
                if fa_axis is None:
                    sbs = layer_metadata.block_size // cr
                    bt_c = layer_metadata.block_table[req].to(torch.long)
                    blk_c = bt_c[
                        torch.div(j_c, sbs, rounding_mode="floor").to(
                            torch.long
                        )
                    ]
                    slots_c = (
                        blk_c * sbs + torch.remainder(j_c, sbs).to(torch.long)
                    ).to(torch.int32)
                    bs_s = swa_metadata.block_size
                    bt_s = swa_metadata.block_table[req].to(torch.long)
                    blk_s = bt_s[
                        torch.div(p_s, bs_s, rounding_mode="floor").to(
                            torch.long
                        )
                    ]
                    slots_s = (
                        blk_s * bs_s
                        + torch.remainder(p_s, bs_s).to(torch.long)
                    ).to(torch.int32)
                    fa_axis = (_pad_slots(slots_c), _pad_slots(slots_s))
                    pass_cache[fa_axis_key] = fa_axis
                slots_c_pad, slots_s_pad = fa_axis

                kc = quixicore_ops.deepseek_v4_prefill_dequant(
                    self.kv_cache, slots_c_pad
                )
                ks = quixicore_ops.deepseek_v4_prefill_dequant(
                    self.swa_cache_layer.kv_cache, slots_s_pad
                )
                fa_direct = (
                    output.is_contiguous()
                    and output.shape == q.shape
                    and output.dtype in (torch.float16, torch.bfloat16)
                )
                result = quixicore_ops.deepseek_v4_prefill_fa(
                    q.to(torch.bfloat16).contiguous(),
                    kc,
                    ks,
                    lens_c_t,
                    lo_rel,
                    hi_rel,
                    self.attn_sink,
                    self.scale,
                    output if fa_direct else None,
                )
                if not fa_direct:
                    output.copy_(result)
                return

        # The kernel writes the caller's buffer directly (fp16 stores round
        # through bf16 in-register, bit-identical to a bf16 result +
        # .copy_() chain).
        direct = (
            output.is_contiguous()
            and output.shape == q.shape
            and output.dtype in (torch.float16, torch.bfloat16)
        )
        result = quixicore_ops.deepseek_v4_sparse_attention(
            q.to(torch.bfloat16).contiguous(),
            compressed_cache,
            compressed_slots,
            compressed_lens,
            self.swa_cache_layer.kv_cache,
            swa_slots,
            swa_lens,
            self.attn_sink,
            self.scale,
            output if direct else None,
        )
        if not direct:
            output.copy_(result)
