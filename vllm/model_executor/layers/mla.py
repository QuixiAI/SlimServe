# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.v1.worker.metal_phaseprof import phase as _qc_phase


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    # Output gate (Kimi K3's mla_use_output_gate). Applied to the attention
    # output before o_proj; None for models without one.
    g_proj: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]


_METAL_DUAL_NORM: bool | None = None


def _metal_dual_norm_ok(mla, qkv_lora: torch.Tensor) -> bool:
    """Metal (opt-in VLLM_METAL_MLA_DUAL_NORM=1): the q_a and kv_a RMS norms
    of the fused projection output ride one dual-segment dispatch. Requires
    plain RMSNorm modules with fp32 (GGUF) weights sharing one epsilon and a
    row-strided fp16/bf16 [T, S] input covering both segments."""
    global _METAL_DUAL_NORM
    if _METAL_DUAL_NORM is None:
        _METAL_DUAL_NORM = False
        if (
            current_platform.is_metal()
            and os.environ.get("VLLM_METAL_MLA_DUAL_NORM", "0") == "1"
        ):
            try:
                from vllm.quixicore import quixicore_ops

                _METAL_DUAL_NORM = quixicore_ops.is_available() and (
                    quixicore_ops.has("rms_norm_dual")
                )
            except Exception:
                _METAL_DUAL_NORM = False
    if not _METAL_DUAL_NORM or qkv_lora.device.type != "mps":
        return False
    qn, kn = mla.q_a_layernorm, mla.kv_a_layernorm
    if type(qn).__name__ != "RMSNorm" or type(kn).__name__ != "RMSNorm":
        return False
    wq, wk = getattr(qn, "weight", None), getattr(kn, "weight", None)
    if wq is None or wk is None:
        return False
    if wq.dtype != torch.float32 or wk.dtype != torch.float32:
        return False
    eps_q = getattr(qn, "variance_epsilon", None)
    if eps_q is None or eps_q != getattr(kn, "variance_epsilon", None):
        return False
    d0, d1 = mla.q_lora_rank, mla.kv_lora_rank
    return (
        qkv_lora.dim() == 2
        and qkv_lora.dtype in (torch.float16, torch.bfloat16)
        and qkv_lora.stride(1) == 1
        and qkv_lora.stride(0) >= d0 + d1
        and qkv_lora.shape[1] >= d0 + d1
        and wq.numel() == d0
        and wk.numel() == d1
        and wq.is_contiguous()
        and wk.is_contiguous()
    )


@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.g_proj = mla_modules.g_proj

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        # qrep is active when the query projection is a DCP-group-sharded layer
        # that materializes the full group head set locally.
        q_proj_layer = self.q_b_proj if self.q_lora_rank is not None else self.q_proj
        self.dcp_q_replicate = getattr(q_proj_layer, "qrep_active", False)
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            dcp_q_replicate=self.dcp_q_replicate,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
        )

        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            with _qc_phase("mla_qkv_a"):
                qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            kv_c_pre = None
            if _metal_dual_norm_ok(self, qkv_lora):
                # One dispatch norms q_c and kv_c (both slices of qkv_lora)
                # with their own weights; per segment bit-exact to the two
                # qc_rms_norm launches it replaces (VLLM_METAL_MLA_DUAL_NORM=1,
                # the glm53f-q2-1 env block).
                from vllm.quixicore import quixicore_ops

                q_c, kv_c_pre = quixicore_ops.rms_norm_dual(
                    qkv_lora,
                    self.q_lora_rank,
                    self.q_a_layernorm.weight,
                    self.kv_lora_rank,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                )
            else:
                q_c = self.q_a_layernorm(q_c)
            q_proj_layer = self.q_b_proj
            q_proj_input = q_c
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states
            kv_c_pre = None

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = (
            kv_c_pre if kv_c_pre is not None else self.kv_a_layernorm(kv_c)
        )
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        with _qc_phase("mla_q_b"):
            q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            with _qc_phase("mla_indexer"):
                self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        with _qc_phase("mla_core"):
            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
                q_dcp_replicated=q_dcp_replicated,
            )

        if self.g_proj is not None:
            attn_out = attn_out * self.g_proj(hidden_states)[0].sigmoid()

        with _qc_phase("mla_o_proj"):
            out = self.o_proj(attn_out)[0]
        return out
