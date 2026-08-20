# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU boundary for DeepSeek-V4 sparse MLA.

First serving version. The ROCm-family attention path is Triton plus torch
fallbacks (aiter only when enabled), so with triton-xpu it is the shortest
route to a correct sparse-MLA forward on Arc Pro B70; the QuixiCore-XPU SYCL
decode kernel replaces ``_forward_decode`` once it exists (see the perf
notebook, 2026-08-18). What this class owns:

- the backend identity (``XPU_MLA_SPARSE_DSV4``);
- the profile-run guard (no attention metadata / KV pages yet, as Metal);
- CUDA-event-free construction (the shared layer allocates sync events).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseMetadataBuilder,
)
from vllm.models.deepseek_v4.common.ops.triton_qnorm_rope_kv_insert import (
    triton_qnorm_rope_kv_fp8_insert,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4XPUMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "XPU_MLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4ROCMAiterMLASparseMetadataBuilder]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # XPU reports no CUDA-style capability; the platform gates on device.
        return True


class DeepseekV4XPUAttention(DeepseekV4ROCMAiterMLAAttention):
    """DeepSeek-V4 attention on Intel XPU (Triton sparse-MLA path)."""

    backend_cls = DeepseekV4XPUMLASparseBackend

    def __init__(self, *args, **kwargs) -> None:
        # torch.cuda.Event is aliased to torch.xpu.Event by the XPU model
        # runner before layers are built; keep a local guard for direct
        # construction (tests, offline tools).
        original_event = torch.cuda.Event
        if not hasattr(original_event, "__xpu_alias__"):
            torch.cuda.Event = torch.xpu.Event  # type: ignore[misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = original_event  # type: ignore[misc]

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        # The CUDA/HIP fused op (_C.fused_deepseek_v4_qnorm_rope_kv_rope_*)
        # is not built here; the Triton equivalent does Q per-head RMSNorm +
        # GPT-J RoPE in place, KV RoPE, and the UE8M0 fp8 paged insert in the
        # same fp8_ds_mla row layout the sparse-MLA Triton kernels read.
        if not isinstance(attn_metadata, dict):
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q, (0, 0, 0, self.padded_heads - self.n_local_heads), value=0.0
                )
            return q
        if self._tq_impl is not None:
            raise NotImplementedError(
                "TurboQuant SWA cache is not wired on XPU yet (needs the "
                "in-place transform + TQ cache update kernels)."
            )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        swa_kv_cache = self.swa_cache_layer.kv_cache
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        if swa_kv_cache.dtype == torch.uint8:
            triton_qnorm_rope_kv_fp8_insert(
                q,
                kv,
                swa_kv_cache,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )
            return q
        raise NotImplementedError(
            f"XPU DSV4 SWA cache dtype {swa_kv_cache.dtype} is not supported; "
            "serve with kv_cache_dtype=fp8 (fp8_ds_mla layout)."
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        prequant_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Memory-profile pass: no attention metadata or allocated KV pages.
        if get_forward_context().attn_metadata is None:
            return torch.zeros_like(hidden_states)
        return super().forward(positions, hidden_states, llama_4_scaling, prequant_input)
