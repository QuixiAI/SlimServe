# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant MLA attention for the DeepSeek-V4 0731 DSpark draft.

The target keeps its specialized sparse-MLA kernels and FP8 cache. The three
standalone 0731 draft layers are sliding-window-only and use ordinary paged
attention after their compressed MLA K/V row is projected and rotary-encoded.
This adapter preserves those trained projections while giving only the draft a
TurboQuant cache.
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum


class DeepseekV4TurboQuantDraftAttention(nn.Module):
    """DeepSeek MLA draft attention backed by TurboQuant paged KV."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        if self.n_heads % tp_size:
            raise ValueError(
                f"DeepSeek-V4 DSpark has {self.n_heads} attention heads, "
                f"which cannot be sharded across TP{tp_size}."
            )
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=1,
        )
        self.tq_attn = Attention(
            self.n_local_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.window_size,
            prefix=f"{prefix}.tq_attn",
            sinks=self.attn_sink,
        )

    @property
    def cache_layer_name(self) -> str:
        return self.tq_attn.layer_name

    def _rotate_kv(self, kv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        rotated, _ = self.rotary_emb(positions, kv.unsqueeze(1), None)
        return rotated

    def insert_context_kv(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        rotated = self._rotate_kv(kv, positions)
        assert hasattr(self.tq_attn.impl, "do_kv_cache_update")
        self.tq_attn.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            self.tq_attn,
            rotated,
            rotated,
            self.tq_attn.kv_cache,
            slot_mapping,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(
            q.dtype
        )
        kv = kv.unsqueeze(1)
        q, key = self.rotary_emb(positions, q, kv)
        assert key is not None
        output = self.tq_attn(q, key, key).view(-1, self.n_local_heads, self.head_dim)
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            output,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))
