# SPDX-License-Identifier: Apache-2.0
"""Opt-in causal diagnostic: order existing GLM pool selections, never reselect.

The disabled factory returns the original callable. Enabled only in the GLM
pooled indexer, with model/selector observers and canonical MoE also required.
This extra sorting launch is NOT the proposed final serving implementation.
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


def maybe_ordered_topk(function):
    if not enabled():
        return function

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
        result = function(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )
        canonicalize(raw_topk_indices)
        return result

    return ordered
