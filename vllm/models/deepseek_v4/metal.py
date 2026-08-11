# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple Metal implementation boundary for DeepSeek-V4 sparse MLA."""

from typing import Any, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.quixicore.ops import quixicore_ops
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4MetalSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "METAL_FLASHMLA_SPARSE_DSV4"


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
        return quixicore_ops.deepseek_v4_qnorm_rope_kv_insert(
            q.contiguous(),
            kv.contiguous(),
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
            positions,
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
        o_ref, _ = self.rotary_emb.forward_native(positions, o, key=None, inverse=True)  # type: ignore[call-arg]
        grouped = o_ref.reshape(o.shape[0], self.n_local_groups, -1)

        if hasattr(self.wo_a, "qweight"):
            from vllm.model_executor.layers.quantization.gguf import (
                fused_mul_mat_gguf,
            )

            qweight = self.wo_a.qweight
            qweight_type = self.wo_a.qweight_type.weight_type
            rows_per_group = self.o_lora_rank
            projected = [
                fused_mul_mat_gguf(
                    grouped[:, group],
                    qweight[
                        group * rows_per_group : (group + 1) * rows_per_group
                    ].contiguous(),
                    qweight_type,
                )
                for group in range(self.n_local_groups)
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
        if get_forward_context().attn_metadata is None:
            output.zero_()
            return
        metadata = get_forward_context().attn_metadata
        assert isinstance(metadata, dict)
        swa_metadata = cast(
            DeepseekSparseSWAMetadata,
            metadata[self.swa_cache_layer.prefix],
        )
        layer_metadata = cast(Any, metadata.get(self.prefix))
        num_tokens = q.shape[0]
        token_to_req_indices = swa_metadata.token_to_req_indices
        valid_token = swa_metadata.is_valid_token
        assert token_to_req_indices is not None and valid_token is not None
        req_ids = token_to_req_indices[:num_tokens].to(torch.long)
        valid_tokens = valid_token[:num_tokens]

        swa_width = self.window_size
        swa_offsets = torch.arange(swa_width, device=q.device, dtype=positions.dtype)
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
        swa_slots = torch.where(swa_valid, swa_slots, -1).to(torch.int32)
        swa_lens = torch.where(valid_tokens, swa_lens, 0).to(torch.int32)

        if self.compress_ratio > 1:
            assert layer_metadata is not None
            assert self.topk_indices_buffer is not None
            compressed_width = min(
                self.topk_indices_buffer.shape[1],
                (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio,
            )
            compressed_offsets = torch.arange(
                compressed_width, device=q.device, dtype=positions.dtype
            )
            compressed_lens = torch.minimum(
                torch.div(
                    positions + 1,
                    self.compress_ratio,
                    rounding_mode="floor",
                ),
                positions.new_full(positions.shape, compressed_width),
            )
            compressed_valid = compressed_offsets.unsqueeze(
                0
            ) < compressed_lens.unsqueeze(1)
            compressed_blocks = layer_metadata.block_table.index_select(0, req_ids)
            storage_block_size = layer_metadata.block_size // self.compress_ratio
            compressed_block_col = torch.div(
                compressed_offsets,
                storage_block_size,
                rounding_mode="floor",
            ).clamp(max=compressed_blocks.shape[1] - 1)
            compressed_block = compressed_blocks.gather(
                1, compressed_block_col.unsqueeze(0).expand(num_tokens, -1)
            )
            compressed_slots = compressed_block * storage_block_size + torch.remainder(
                compressed_offsets, storage_block_size
            )
            compressed_slots = torch.where(compressed_valid, compressed_slots, -1).to(
                torch.int32
            )
            compressed_lens = torch.where(valid_tokens, compressed_lens, 0).to(
                torch.int32
            )
            compressed_cache = self.kv_cache
        else:
            compressed_slots = torch.full(
                (num_tokens, 1), -1, dtype=torch.int32, device=q.device
            )
            compressed_lens = torch.zeros(
                num_tokens, dtype=torch.int32, device=q.device
            )
            compressed_cache = self.swa_cache_layer.kv_cache

        result = quixicore_ops.deepseek_v4_sparse_attention(
            q.contiguous(),
            compressed_cache,
            compressed_slots.contiguous(),
            compressed_lens.contiguous(),
            self.swa_cache_layer.kv_cache,
            swa_slots.contiguous(),
            swa_lens.contiguous(),
            self.attn_sink,
            self.scale,
        )
        output.copy_(result)
