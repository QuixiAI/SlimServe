# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek sparse-attention indexer logits on Intel XPU.

``SparseAttnIndexer.forward_cuda`` routes XPU to ``torch.ops.vllm.
xpu_fp8_mqa_logits`` / ``xpu_fp8_paged_mqa_logits`` (upstream backed them
with vllm-xpu-kernels). This fork registers those two custom ops on top of
the platform-neutral torch references, so the indexer runs unchanged; the
QuixiCore-XPU ``mqa_logits`` XMX kernel (kernels/serving/mqa_logits) is the
native replacement once wired (perf notebook 2026-08-18).

The paged decode reference does one host sync per request; correct, and a
known bring-up cost.
"""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# The torch references live in the ROCm ops module; imported lazily because
# that module's import chain can re-enter platform init (register_xpu_c_ops
# imports this file).


def _xpu_fp8_mqa_logits_impl(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch

    return fp8_mqa_logits_torch(q, (kv, kv_scales), weights, cu_seqlen_ks, cu_seqlen_ke)


def _xpu_fp8_mqa_logits_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((q.shape[0], kv.shape[0]), dtype=torch.float32, device=q.device)


def _xpu_fp8_paged_mqa_logits_impl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    del schedule_metadata  # DeepGEMM scheduling hint; unused by the reference
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_paged_mqa_logits_torch

    if kv_cache.dim() == 3:
        kv_cache = kv_cache.unsqueeze(-2)
    return fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def _xpu_fp8_paged_mqa_logits_fake(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    batch_size, next_n = q.shape[0], q.shape[1]
    return torch.empty(
        (batch_size * next_n, max_model_len), dtype=torch.float32, device=q.device
    )


direct_register_custom_op(
    op_name="xpu_fp8_mqa_logits",
    op_func=_xpu_fp8_mqa_logits_impl,
    fake_impl=_xpu_fp8_mqa_logits_fake,
)
direct_register_custom_op(
    op_name="xpu_fp8_paged_mqa_logits",
    op_func=_xpu_fp8_paged_mqa_logits_impl,
    fake_impl=_xpu_fp8_paged_mqa_logits_fake,
)
