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


_BUCKETS = (1, 8, 16, 32, 64, 128, 256, 1 << 30)   # upper token bounds (inclusive)


class ExpertStats:
    """Per-layer accumulator; ``record`` is a no-op unless enabled.

    Everything is accumulated on the device, bucketed by the call's token
    count, and the call count lives on the device too: a CUDA-graph replay
    re-executes the captured accumulation with the replayed routing but
    never runs this Python, so host-side counting would miss every graphed
    decode step (2026-09-12: means above the expert count)."""

    __slots__ = ("name", "num_experts", "hist", "acc", "calls", "ones")

    def __init__(self, name: str, num_experts: int, device: torch.device):
        self.name = name
        self.num_experts = num_experts
        self.hist = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # [bucket, (active experts, tokens, calls)]
        self.acc = torch.zeros(len(_BUCKETS), 3, dtype=torch.int64, device=device)
        self.calls = 0
        self.ones = torch.ones(1, dtype=torch.int32, device=device)

    def record(self, topk_ids: torch.Tensor) -> None:
        tokens = topk_ids.shape[0]
        b = next(i for i, hi in enumerate(_BUCKETS) if tokens <= hi)
        ids = topk_ids.reshape(-1).long()
        self.hist.zero_()
        self.hist.index_add_(0, ids, self.ones.expand(ids.numel()))
        self.acc[b, 0] += (self.hist > 0).sum()
        self.acc[b, 1] += tokens
        self.acc[b, 2] += 1
        self.calls += 1
        if self.calls % _EVERY == 0 and not torch.cuda.is_current_stream_capturing():
            rows = self.acc.tolist()
            parts = []
            for hi, (active, toks, calls) in zip(_BUCKETS, rows):
                if calls:
                    parts.append(
                        f"<={hi if hi < (1 << 30) else 'inf'}tok: {calls} calls, "
                        f"tokens {toks / calls:.1f}, active {active / calls:.1f}"
                    )
            logger.info("moe-stats %s of %d experts: %s", self.name, self.num_experts, "; ".join(parts))
            self.acc.zero_()


def make_expert_stats(name: str, num_experts: int, device: torch.device) -> ExpertStats | None:
    return ExpertStats(name, num_experts, device) if _EVERY > 0 else None
