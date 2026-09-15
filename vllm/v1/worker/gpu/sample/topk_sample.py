# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused top-k / top-p / Gumbel-max sampling (native ``topk_sample``).

When every request in the batch samples with ``1 <= top_k <= 32``, one native
op replaces the Triton top-k/top-p mask pass over the full vocabulary plus the
Gumbel argmax pass. It keeps every tie at the k-th value, orders the nucleus
ascending by (logit, id), and draws the same (seed, pos, token)-keyed Philox
noise as ``gumbel_sample``, so seeded requests stay reproducible.
"""

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.states import SamplingStates

MAX_TOP_K = 32
# The candidate window (16 partitions x 32 candidates) must fit the vocabulary.
MIN_VOCAB = 512


@cache
def enabled() -> bool:
    from vllm.v1.worker.gpu.sample.gumbel import _use_native_sample_kernels

    if not _use_native_sample_kernels():
        return False
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.has_topk_sample()


def eligible(
    sampling_states: "SamplingStates",
    top_k: torch.Tensor | None,
    idx_mapping_np: np.ndarray,
    logits: torch.Tensor,
    pos: torch.Tensor,
    needs_processed_logits: bool,
) -> bool:
    """True when the batch can be sampled by the fused kernel.

    Greedy rows keep the argmax path, and a batch that must return the
    masked logits (processed logprobs) keeps the materialized mask.
    """
    if not enabled() or top_k is None or needs_processed_logits:
        return False
    if (
        logits.device.type != "cuda"
        or logits.dtype != torch.float32
        or logits.dim() != 2
        or logits.shape[1] < MIN_VOCAB
        or logits.stride(1) != 1
        or pos.dtype != torch.int64
    ):
        return False
    if sampling_states.any_greedy(idx_mapping_np):
        return False
    k_np = sampling_states.top_k.np[idx_mapping_np]
    return bool(np.all((k_np >= 1) & (k_np <= MAX_TOP_K)))


def sample(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor | None,
    expanded_idx_mapping: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    use_fp64: bool,
) -> torch.Tensor:
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.topk_sample(
        logits,
        top_k.contiguous(),
        None if top_p is None else top_p.contiguous(),
        expanded_idx_mapping.contiguous(),
        seeds,
        pos.contiguous(),
        use_fp64,
    )


@cache
def mask_enabled() -> bool:
    from vllm.v1.worker.gpu.sample.gumbel import _use_native_sample_kernels

    if not _use_native_sample_kernels():
        return False
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.has_topk_topp_mask()


def mask_eligible(
    sampling_states: "SamplingStates",
    idx_mapping_np: np.ndarray,
    logits: torch.Tensor,
) -> bool:
    """True when the batch's top-k/top-p mask can be written in place by the
    native cutoff (every request's ``1 <= top_k <= 32``); greedy rows are
    fine here, the mask does not draw."""
    if not mask_enabled():
        return False
    if (
        logits.device.type != "cuda"
        or logits.dtype != torch.float32
        or logits.dim() != 2
        or logits.shape[1] < MIN_VOCAB
        or logits.stride(1) != 1
    ):
        return False
    k_np = sampling_states.top_k.np[idx_mapping_np]
    return bool(np.all((k_np >= 1) & (k_np <= MAX_TOP_K)))


def mask(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor | None,
) -> None:
    """In place: -inf over every token ``sample`` would not draw."""
    from vllm.quixicore import quixicore_ops

    quixicore_ops.topk_topp_mask(
        logits,
        top_k.contiguous(),
        None if top_p is None else top_p.contiguous(),
    )
