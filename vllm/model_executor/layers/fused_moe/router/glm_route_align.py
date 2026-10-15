# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused small-M routing for the Marlin MoE path (QuixiCore glm_route_align).

One launch does the scored top-k with bias-only selection (DeepSeek-style
noaux_tc with a single expert group), renormalisation and scaling, and the
Marlin block alignment that fused_marlin_moe would otherwise recompute with
moe_align_block_size (two kernels plus the fill of sorted_token_ids). The
router returns the usual (topk_weights, topk_ids) and publishes the alignment
for fused_marlin_moe, which consumes it (matched on the topk_ids tensor) when
its own block size matches. Decode only: M <= 16 tokens."""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size_geometry,
)
from vllm.platforms import current_platform
from vllm.quixicore.ops import quixicore_ops
from vllm.utils.torch_utils import direct_register_custom_op

MAX_TOKENS = 16
# (num_experts, topk) instantiated in the kernel.
SUPPORTED_SHAPES = {(288, 8)}
SCORING = {"sigmoid": 0, "sqrtsoftplus": 1}


@dataclass
class RoutingAlignment:
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    block_size: int


# The router and the Marlin wrapper run back to back inside the moe_forward
# custom op on one thread; the router publishes the alignment here and
# fused_marlin_moe consumes it, matched on the identity of the topk_ids
# tensor so a stale alignment can never be applied to another batch.
_published: RoutingAlignment | None = None


@dataclass
class DeferredRouting:
    """A batch whose routing the NVFP4 decode pair computes inside its first
    kernel: `route` hands out the (still unwritten) topk_ids / topk_weights
    tensors and keeps the router inputs here; `fused_marlin_moe` consumes the
    entry (matched on the topk_ids identity) and either lets gemv1 write the
    tensors or, when the pair declines the batch, materializes the routing
    with the route_align kernel and publishes the alignment as usual."""

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    logits: torch.Tensor
    bias: torch.Tensor
    top_k: int
    scoring: int
    renormalize: bool
    scaling: float
    block_size: int
    max_padded: int
    max_blocks: int
    # The fixed expert appended to every token's slots at weight 1.0 (a shared
    # expert served as expert E of E + 1), or -1.
    fixed: int = -1


_deferred: DeferredRouting | None = None


def consume_deferred(topk_ids: torch.Tensor) -> DeferredRouting | None:
    global _deferred
    entry = _deferred
    if entry is not None and entry.topk_ids is topk_ids:
        _deferred = None
        return entry
    return None


def pending_deferred() -> bool:
    return _deferred is not None


def clear_deferred() -> None:
    """Drop a never-consumed entry (the runner's cleanup after a failed
    forward), so a later batch does not inherit it."""
    global _deferred
    _deferred = None


def materialize(entry: DeferredRouting) -> None:
    """Route the batch now (the pair is not serving it): fills the handed-out
    tensors in place and publishes the Marlin alignment."""
    weights, ids, sorted_ids, expert_ids, post_pad = torch.ops.vllm.glm_route_align(
        entry.logits,
        entry.bias,
        entry.top_k,
        entry.scoring,
        entry.renormalize,
        entry.scaling,
        entry.block_size,
        entry.max_padded,
        entry.max_blocks,
        entry.fixed,
    )
    entry.topk_ids.copy_(ids)
    entry.topk_weights.copy_(weights)
    publish(RoutingAlignment(entry.topk_ids, sorted_ids, expert_ids, post_pad, entry.block_size))


def publish(alignment: RoutingAlignment) -> None:
    global _published
    _published = alignment


def consume(topk_ids: torch.Tensor) -> RoutingAlignment | None:
    global _published
    alignment, _published = _published, None
    if alignment is not None and alignment.topk_ids is topk_ids:
        return alignment
    return None


def eligible(router, router_logits: torch.Tensor, indices_type) -> bool:
    bias = router.e_score_correction_bias
    return (
        current_platform.is_cuda()
        and quixicore_ops.has_glm_route_align()
        and router_logits.dim() == 2
        and router_logits.dtype == torch.float32
        and router_logits.is_contiguous()
        and 0 < router_logits.shape[0] <= MAX_TOKENS
        and (router_logits.shape[1], router.top_k) in SUPPORTED_SHAPES
        and router.num_expert_group == 1
        and router.topk_group == 1
        and router.scoring_func in SCORING
        and bias is not None
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and bias.shape == (router_logits.shape[1],)
        # At most one fused shared expert: the kernel appends it as a fixed slot.
        and getattr(router, "num_fused_shared_experts", 0) in (0, 1)
        and getattr(router, "shared_expert_weight", 1.0) == 1.0
        and indices_type in (None, torch.int32)
    )


def route(router, router_logits: torch.Tensor):
    num_tokens, num_experts = router_logits.shape
    # A fused shared expert is expert `num_experts` of the weights and the last
    # of every token's slots at weight 1.0 (the routed scaling stays on the
    # routed slots); the alignment then spans num_experts + 1 experts.
    n_fixed = getattr(router, "num_fused_shared_experts", 0)
    fixed = num_experts if n_fixed else -1
    slots = router.top_k + n_fixed
    aligned_experts = num_experts + n_fixed
    # fused_marlin_moe's block size for this batch (bf16 activations) and
    # moe_align_block_size's buffer geometry, from their own definitions.
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        marlin_moe_block_size_m,
    )

    block_size = marlin_moe_block_size_m(num_tokens, slots, aligned_experts, None)
    max_padded, max_blocks = moe_align_block_size_geometry(
        num_tokens * slots, aligned_experts, block_size
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        nvfp4_decode_pair_serves,
    )

    if nvfp4_decode_pair_serves(num_tokens * slots):
        # The pair's gemv1 routes the batch itself (one launch fewer on the
        # decode critical path); hand out the tensors it will write.
        global _deferred
        ids = torch.empty((num_tokens, slots), dtype=torch.int32, device=router_logits.device)
        weights = torch.empty((num_tokens, slots), dtype=torch.float32, device=router_logits.device)
        _deferred = DeferredRouting(
            ids, weights, router_logits, router.e_score_correction_bias.data, router.top_k,
            SCORING[router.scoring_func], bool(router.renormalize), float(router.routed_scaling_factor),
            block_size, max_padded, max_blocks, fixed,
        )
        return weights, ids
    weights, ids, sorted_ids, expert_ids, post_pad = torch.ops.vllm.glm_route_align(
        router_logits,
        router.e_score_correction_bias.data,
        router.top_k,
        SCORING[router.scoring_func],
        router.renormalize,
        float(router.routed_scaling_factor),
        block_size,
        max_padded,
        max_blocks,
        fixed,
    )
    publish(RoutingAlignment(ids, sorted_ids, expert_ids, post_pad, block_size))
    return weights, ids


def _glm_route_align_impl(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring: int,
    renormalize: bool,
    scaling: float,
    block_size: int,
    max_padded: int,
    max_blocks: int,
    fixed: int = -1,
) -> list[torch.Tensor]:
    return quixicore_ops.glm_route_align(
        logits,
        bias,
        topk,
        scoring,
        renormalize,
        scaling,
        block_size,
        max_padded,
        max_blocks,
        fixed,
    )


def _glm_route_align_fake(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring: int,
    renormalize: bool,
    scaling: float,
    block_size: int,
    max_padded: int,
    max_blocks: int,
    fixed: int = -1,
) -> list[torch.Tensor]:
    i32 = logits.new_empty(0, dtype=torch.int32)
    slots = topk + (1 if fixed >= 0 else 0)
    return [
        logits.new_empty((logits.shape[0], slots)),
        i32.new_empty((logits.shape[0], slots)),
        i32.new_empty((max_padded,)),
        i32.new_empty((max_blocks,)),
        i32.new_empty((1,)),
    ]


direct_register_custom_op(
    op_name="glm_route_align",
    op_func=_glm_route_align_impl,
    fake_impl=_glm_route_align_fake,
)
