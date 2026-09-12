# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated routing diagnostic: how many distinct experts each MoE layer
touches per forward call.

A weight-streaming MoE kernel's cost is set by the number of DISTINCT
experts a batch routes to (every touched expert's full weights are read),
not by the token count. ``VLLM_MOE_EXPERT_STATS=N`` (calls per log line,
0 = off) accumulates, per MoE layer, the mean number of active experts and
the mean token count over N calls and logs them. The accumulation is a
few device ops with no host sync, so it is CUDA-graph safe; the log itself
is skipped while a graph is being captured.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_EVERY = int(os.getenv("VLLM_MOE_EXPERT_STATS", "0") or 0)


class ExpertStats:
    """Per-layer accumulator; ``record`` is a no-op unless enabled."""

    __slots__ = ("name", "num_experts", "hist", "sums", "calls", "ones")

    def __init__(self, name: str, num_experts: int, device: torch.device):
        self.name = name
        self.num_experts = num_experts
        self.hist = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # [active experts, tokens] accumulated on the device.
        self.sums = torch.zeros(2, dtype=torch.int64, device=device)
        self.calls = 0
        self.ones = torch.ones(1, dtype=torch.int32, device=device)

    def record(self, topk_ids: torch.Tensor) -> None:
        ids = topk_ids.reshape(-1).long()
        self.hist.zero_()
        self.hist.index_add_(0, ids, self.ones.expand(ids.numel()))
        self.sums[0] += (self.hist > 0).sum()
        self.sums[1] += topk_ids.shape[0]
        self.calls += 1
        if self.calls % _EVERY == 0 and not torch.cuda.is_current_stream_capturing():
            active, tokens = self.sums.tolist()
            logger.info(
                "moe-stats %s: %d calls, mean active experts %.1f of %d, mean tokens %.1f",
                self.name,
                _EVERY,
                active / _EVERY,
                self.num_experts,
                tokens / _EVERY,
            )
            self.sums.zero_()


def make_expert_stats(name: str, num_experts: int, device: torch.device) -> ExpertStats | None:
    return ExpertStats(name, num_experts, device) if _EVERY > 0 else None
