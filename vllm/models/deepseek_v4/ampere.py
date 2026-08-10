# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 sparse MLA boundary for NVIDIA Ampere (A100)."""

from __future__ import annotations

import os

import torch

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseMetadata,
    DeepseekV4ROCMAiterMLASparseMetadataBuilder,
    DeepseekV4ROCMAiterSparseSWAMetadata,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.platforms.interface import DeviceCapability
from vllm.quixicore import quixicore_ops


# One warp serially walking a 512-entry source leaves nearly all 108 A100 SMs
# idle. The persistent source kernel's production-shape sweep selects 4 for
# both TP2 and TP4. Keep an explicit override so notebook experiments can
# sweep the dispatch without code edits.
_MLA_PARTITION_SIZE_OVERRIDE = os.getenv("VLLM_DSV4_MLA_PARTITION_SIZE")


def _mla_partition_size(num_local_heads: int) -> int:
    if _MLA_PARTITION_SIZE_OVERRIDE is not None:
        return int(_MLA_PARTITION_SIZE_OVERRIDE)
    return 4


class DeepseekV4AmpereMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "QUIXICORE_MLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV4AmpereMLAAttention(DeepseekV4ROCMAiterMLAAttention):
    """DeepSeek-V4 attention for A100.

    Decode uses the native QuixiCore DSV4 packed-page op. It reads the
    fp8_ds_mla cache in place (576B token data + 8B scales) and merges SWA plus
    compressed sparse sources in the reducer, matching the GLM-5.2 native path
    without materializing a bf16 KV workspace.
    """

    backend_cls = DeepseekV4AmpereMLASparseBackend
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.native_q8_o_proj = os.getenv(
            "VLLM_DSV4_NATIVE_Q8_O_PROJ", "1"
        ).lower() not in {"0", "false", "off", "no"}
        self.defer_tp_reduce = (
            get_tensor_model_parallel_world_size() > 1
            and os.getenv("VLLM_DSV4_DEFER_TP_REDUCE", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        if self.defer_tp_reduce:
            self.wo_b.reduce_results = False

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        qweight = getattr(self.wo_a, "qweight", None)
        qweight_type = getattr(self.wo_a, "qweight_type", None)
        weight_type = getattr(qweight_type, "weight_type", -1)
        if not (
            self.native_q8_o_proj
            and qweight is not None
            and weight_type == 8
        ):
            return super()._o_proj(o, positions)

        from vllm.model_executor.layers.quantization.gguf import ops

        num_tokens = o.shape[0]
        if num_tokens <= 32:
            z, quant_z = ops.ggml_dsv4_o_proj_q8_0(
                qweight,
                o.contiguous(),
                positions.contiguous(),
                self.rotary_emb.cos_sin_cache.contiguous(),
                self.n_local_groups,
                self.rope_head_dim,
            )
            down_quant_method = getattr(self.wo_b, "quant_method", None)
            apply_prequant = getattr(down_quant_method, "apply_prequant", None)
            down_qweight_type = getattr(self.wo_b, "qweight_type", None)
            if (
                apply_prequant is not None
                and getattr(down_qweight_type, "weight_type", -1) == 8
            ):
                return apply_prequant(self.wo_b, z, quant_z)
            return self.wo_b(z)

        # Prefill remains packed too. The grouped matrices use different
        # activation rows, so dispatch each local group through the native Q8
        # MMQ path instead of caching an expanded BF16 wo_a tensor.
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            _fused_inverse_rope_gptj,
        )

        inverse = _fused_inverse_rope_gptj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.rope_head_dim,
        )
        group_dim = inverse.shape[1] // self.n_local_groups * inverse.shape[2]
        rows_per_group = qweight.shape[0] // self.n_local_groups
        grouped = inverse.view(num_tokens, self.n_local_groups, group_dim)
        projected = []
        for group in range(self.n_local_groups):
            group_weight = qweight[
                group * rows_per_group : (group + 1) * rows_per_group
            ]
            projected.append(
                ops.ggml_mul_mat_a8(
                    group_weight,
                    grouped[:, group].contiguous(),
                    8,
                    rows_per_group,
                )
            )
        z = torch.stack(projected, dim=1).reshape(
            num_tokens, self.n_local_groups * rows_per_group
        )
        down_qweight = getattr(self.wo_b, "qweight", None)
        down_qweight_type = getattr(self.wo_b, "qweight_type", None)
        if (
            down_qweight is not None
            and getattr(down_qweight_type, "weight_type", -1) == 8
        ):
            return ops.ggml_mul_mat_a8(
                down_qweight, z, 8, down_qweight.shape[0]
            )
        return self.wo_b(z)

    def _empty_sparse_indices(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty((q.shape[0], 0), dtype=torch.int32, device=q.device),
            torch.zeros((q.shape[0],), dtype=torch.int32, device=q.device),
        )

    def _empty_block_table(self, q: torch.Tensor) -> torch.Tensor:
        return torch.empty((q.shape[0], 0), dtype=torch.int32, device=q.device)

    def _forward_decode_torch_fallback(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None

        swa_indices = (
            swa_metadata.decode_swa_indices[:num_decode_tokens]
            .reshape(num_decode_tokens, -1)
            .to(torch.int64)
        )
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens].to(torch.int64)
        flat_swa_cache = self.swa_cache_layer.kv_cache.reshape(-1, q.shape[-1])
        swa_arange = torch.arange(swa_indices.shape[1], device=q.device)[None, :]
        safe_swa_indices = swa_indices.clamp(
            min=0, max=max(flat_swa_cache.shape[0] - 1, 0)
        )
        gathered_parts = [
            flat_swa_cache.index_select(
                0, safe_swa_indices.reshape(-1)
            ).reshape(num_decode_tokens, swa_indices.shape[1], q.shape[-1])
        ]
        mask_parts = [(swa_indices >= 0) & (swa_arange < swa_lens[:, None])]

        if not swa_only:
            assert attn_metadata is not None
            assert kv_cache is not None
            comp_block_size = attn_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                assert swa_metadata.token_to_req_indices is not None
                local_idx = self.topk_indices_buffer[:num_decode_tokens].to(torch.int64)
                req_ids = swa_metadata.token_to_req_indices[:num_decode_tokens].to(
                    torch.long
                )
                bt = attn_metadata.block_table.index_select(0, req_ids).to(torch.long)
                block_cols = torch.div(local_idx, comp_block_size, rounding_mode="floor")
                block_offsets = local_idx % comp_block_size
                valid = local_idx >= 0
                safe_cols = block_cols.clamp(min=0, max=max(bt.shape[1] - 1, 0))
                blocks = bt.gather(1, safe_cols)
                comp_indices = torch.where(
                    valid, blocks * comp_block_size + block_offsets, -1
                )
                comp_lens = quixicore_ops.sparse_topk_tlen(
                    comp_indices.to(torch.int32).contiguous()
                ).to(torch.int64)
            else:
                assert attn_metadata.c128a_global_decode_topk_indices is not None
                assert attn_metadata.c128a_decode_topk_lens is not None
                comp_indices = (
                    attn_metadata.c128a_global_decode_topk_indices[
                        :num_decode_tokens
                    ]
                    .reshape(num_decode_tokens, -1)
                    .to(torch.int64)
                )
                comp_lens = attn_metadata.c128a_decode_topk_lens[
                    :num_decode_tokens
                ].to(torch.int64)
            flat_comp_cache = kv_cache.reshape(-1, q.shape[-1])
            comp_arange = torch.arange(comp_indices.shape[1], device=q.device)[None, :]
            safe_comp_indices = comp_indices.clamp(
                min=0, max=max(flat_comp_cache.shape[0] - 1, 0)
            )
            gathered_parts.append(
                flat_comp_cache.index_select(
                    0, safe_comp_indices.reshape(-1)
                ).reshape(num_decode_tokens, comp_indices.shape[1], q.shape[-1])
            )
            mask_parts.append((comp_indices >= 0) & (comp_arange < comp_lens[:, None]))

        gathered = torch.cat(gathered_parts, dim=1)
        mask = torch.cat(mask_parts, dim=1)
        scores = torch.einsum("bhd,bkd->bhk", q, gathered) * self.scale
        scores = scores.masked_fill(~mask[:, None, :], -float("inf"))
        value_count = gathered.shape[1]
        if self.attn_sink is not None:
            scores = torch.cat(
                [
                    scores,
                    self.attn_sink[: q.shape[1]].view(1, q.shape[1], 1).expand(
                        q.shape[0], -1, -1
                    ),
                ],
                dim=-1,
            )
        probs = torch.softmax(scores, dim=-1)
        value_probs = probs[..., :value_count].to(gathered.dtype)
        output.copy_(
            torch.einsum("bhk,bkd->bhd", value_probs, gathered).to(output.dtype)
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
        if self.swa_cache_layer.kv_cache.dtype != torch.uint8:
            return self._forward_decode_torch_fallback(
                q=q,
                kv_cache=kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=attn_metadata,
                swa_only=swa_only,
                output=output,
            )

        num_decode_tokens = swa_metadata.num_decode_tokens
        q = q.contiguous()

        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        swa_indices = (
            swa_metadata.decode_swa_indices[:num_decode_tokens]
            .reshape(num_decode_tokens, -1)
            .to(torch.int32)
            .contiguous()
        )
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens].to(
            torch.int32
        ).contiguous()
        empty_bt = self._empty_block_table(q)

        extra_cache = self.swa_cache_layer.kv_cache
        extra_bt = empty_bt
        extra_indices, extra_lens = self._empty_sparse_indices(q)
        extra_indices_are_slots = True
        extra_block_size = swa_metadata.block_size

        if not swa_only:
            assert attn_metadata is not None
            assert kv_cache is not None
            extra_cache = kv_cache
            extra_block_size = attn_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                assert swa_metadata.is_valid_token is not None
                local_idx = (
                    self.topk_indices_buffer[:num_decode_tokens]
                    .to(torch.int32)
                    .contiguous()
                )
                is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
                local_idx = local_idx.masked_fill(~is_valid[:, None], -1)
                extra_indices = local_idx
                extra_lens = quixicore_ops.sparse_topk_tlen(local_idx)
                req_ids = swa_metadata.token_to_req_indices
                assert req_ids is not None
                extra_bt = (
                    attn_metadata.block_table.index_select(
                        0, req_ids[:num_decode_tokens].to(torch.int32)
                    )
                    .to(torch.int32)
                    .contiguous()
                )
                extra_indices_are_slots = False
            else:
                assert attn_metadata.c128a_global_decode_topk_indices is not None
                assert attn_metadata.c128a_decode_topk_lens is not None
                extra_indices = (
                    attn_metadata.c128a_global_decode_topk_indices[
                        :num_decode_tokens
                    ]
                    .reshape(num_decode_tokens, -1)
                    .to(torch.int32)
                    .contiguous()
                )
                extra_lens = attn_metadata.c128a_decode_topk_lens[
                    :num_decode_tokens
                ].to(torch.int32).contiguous()
                extra_indices_are_slots = True

        out = quixicore_ops.mla_decode_fp8_sparse_dsv4(
            q,
            self.swa_cache_layer.kv_cache,
            empty_bt,
            swa_indices,
            swa_lens,
            True,
            extra_cache,
            extra_bt,
            extra_indices,
            extra_lens,
            extra_indices_are_slots,
            self.attn_sink.contiguous(),
            swa_metadata.block_size,
            extra_block_size,
            self.scale,
            _mla_partition_size(q.shape[1]),
        )
        output.copy_(out)
