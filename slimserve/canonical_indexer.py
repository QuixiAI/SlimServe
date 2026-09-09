# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM pool-order and separate native cutoff-tie diagnostics.

The disabled factory returns the original callable. Enabled only in the GLM
pooled indexer, with model/selector observers and canonical MoE also required.
The two-kernel control retains its extra sorting launch. CANONICAL_INDEX_FUSED
instead emits ordered IDs inside the qualified native selector; it remains an
opt-in diagnostic until full-model qualification, not a production default.
CANONICAL_INDEX_TIES additionally chooses smaller pool IDs at exact equal-score
cutoffs in the native selector. It never changes scores or higher-score choices.
"""

import os
from functools import wraps

import torch


def enabled():
    value = os.getenv("SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER", "0")
    if value not in ("0", "1"):
        raise ValueError("canonical index order flag must be 0 or 1")
    if value == "1" and (
        os.getenv("SLIMSERVE_GLM53_MODEL_JOURNAL") != "1"
        or os.getenv("SLIMSERVE_GLM53_CANONICAL_MOE") != "1"
        or not os.getenv("SLIMSERVE_GLM53_INDEX_JOURNAL")
    ):
        raise ValueError(
            "canonical index order requires model/index traces and canonical MoE"
        )
    return value == "1"


def canonicalize(indices):
    if (
        indices.dtype != torch.int32
        or not indices.is_cuda
        or not indices.is_contiguous()
        or indices.dim() != 2
        or not 1 <= indices.shape[0] <= 8192
        or indices.shape[1] != 512
    ):
        raise ValueError(
            "canonical index order requires contiguous CUDA int32 [1..8192,512]"
        )
    from slimserve.canonical_indexer_kernel import sort_selected_pools

    sort_selected_pools(indices)


def ties_enabled():
    value = os.getenv("SLIMSERVE_GLM53_CANONICAL_INDEX_TIES", "0")
    if value not in ("0", "1"):
        raise ValueError("canonical index ties flag must be 0 or 1")
    if value == "1" and not enabled():
        raise ValueError("canonical index ties requires canonical index order")
    return value == "1"


def _native_tie_selector():
    from vllm import _custom_ops as ops

    if not hasattr(torch.ops._C, "glm53_top_k_per_row_prefill"):
        raise RuntimeError("canonical index ties requires a rebuilt native selector")
    return ops.glm53_top_k_per_row_prefill


def fused_enabled():
    value = os.getenv("SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED", "0")
    if value not in ("0", "1"):
        raise ValueError("canonical index fusion flag must be 0 or 1")
    if value == "1" and not ties_enabled():
        raise ValueError("canonical index fusion requires canonical index ties")
    return value == "1"


def _native_fused_selector():
    from vllm import _custom_ops as ops

    if not hasattr(torch.ops._C, "glm53_top_k_per_row_ordered"):
        raise RuntimeError("canonical index fusion requires a rebuilt native selector")
    return ops.glm53_top_k_per_row_ordered


def maybe_ordered_topk(function):
    canonical_ties = ties_enabled()
    fused = fused_enabled()
    if not enabled():
        return function
    selector = (
        _native_fused_selector()
        if fused
        else _native_tie_selector()
        if canonical_ties
        else function
    )

    @wraps(function)
    def ordered(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    ):
        if num_rows != raw_topk_indices.shape[0] or topk_tokens != 512:
            raise ValueError("canonical index order selector geometry changed")
        result = selector(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )
        if not fused:
            canonicalize(raw_topk_indices)
        return result

    return ordered
