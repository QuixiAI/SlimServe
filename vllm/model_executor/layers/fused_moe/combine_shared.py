# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-expert combine handoff for the Marlin path (QuixiCore moe_sum_add).

The runner launches the shared experts before the routed experts (on the aux
stream at decode) and publishes them here keyed on the batch's topk_ids. The
Marlin experts' final per-assignment sum consumes the publication and folds
the shared-expert output into the routed sum in the same launch, replacing
the moe_sum kernel, the finalize copy and the separate add. If nothing
consumed it (another kernel path, a shape mismatch), the runner adds the
shared output itself inside the op. Same single-thread, identity-matched
contract as the routing alignment handoff in router/glm_route_align.py."""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)


@dataclass
class _Pending:
    topk_ids: torch.Tensor
    shared_experts: SharedExperts
    consumed: bool = False


_pending: _Pending | None = None


def publish(topk_ids: torch.Tensor, shared_experts: SharedExperts) -> None:
    global _pending
    _pending = _Pending(topk_ids, shared_experts)


def consume(topk_ids: torch.Tensor, output: torch.Tensor) -> torch.Tensor | None:
    """The shared-expert output to fold into `output` for this batch, joined
    with the current stream, or None when there is nothing compatible."""
    pending = _pending
    if pending is None or pending.consumed or pending.topk_ids is not topk_ids:
        return None
    shared = pending.shared_experts.peek_output()
    if (
        shared is None
        or shared.shape != output.shape
        or shared.dtype != output.dtype
        or not shared.is_contiguous()
    ):
        return None
    pending.consumed = True
    return pending.shared_experts.output


def retire() -> bool:
    """Runner side, after the routed experts: drop the publication and report
    whether the kernel folded the shared output in."""
    global _pending
    pending, _pending = _pending, None
    return pending is not None and pending.consumed
