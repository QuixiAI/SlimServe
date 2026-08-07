# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apple Metal implementation boundary for DeepSeek-V4 sparse MLA."""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend


class DeepseekV4MetalSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "METAL_FLASHMLA_SPARSE_DSV4"


class DeepseekV4MetalAttention(DeepseekV4Attention):
    """DeepSeek-V4 attention hosted by QuixiCore-Metal kernels.

    Model construction is deliberately separated from the sparse-attention
    kernel implementation so Metal never falls through to CUDA.  The forward
    methods remain explicit failure boundaries until the packed DSV4 cache
    kernels are connected.
    """

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
        raise NotImplementedError(
            "DeepSeek-V4 Q/KV normalization, RoPE, and cache insertion "
            "are not connected to QuixiCore-Metal yet."
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # vLLM's memory-profile pass has no attention metadata or allocated KV
        # pages.  CUDA/ROCm reserve workspace inside their sparse op; Metal has
        # no such workspace yet, so keep the profile pass moving and let the
        # worker budget unified memory from actual allocations.
        if get_forward_context().attn_metadata is None:
            return torch.zeros_like(hidden_states)
        return super().forward(positions, hidden_states, llama_4_scaling)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # The DSV4 A projection is eight independent matrices.  GGUF stores
        # them as consecutive row groups; apply each group to its matching
        # attention heads, then feed the concatenated low-rank result to WO_B.
        o_ref, _ = self.rotary_emb.forward_native(
            positions, o, key=None, inverse=True
        )
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
        raise NotImplementedError(
            "DeepSeek-V4 two-cache sparse MLA is not connected to "
            "QuixiCore-Metal yet."
        )
