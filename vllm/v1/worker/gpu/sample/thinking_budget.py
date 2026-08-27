# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thinking-token budget enforcement for the V2 model runner.

Qwen3.8-class reasoning models can think nearly without bound even at low
reasoning effort, so the per-request ``thinking_token_budget`` is a hard
functional requirement, not a tuning convenience. The V1 sampler enforced it
with a host-side state holder that scans output token lists per step; that
design collides with the V2 runner's async scheduling (the host does not
hold step N's tokens when step N+1 is scheduled). This implementation is
GPU-resident and rides the same in-place logits-forcing seam as the grammar
bitmask, so it is async-safe and works unchanged under speculative decoding:
a verify row whose logits are masked to the reasoning-end token simply
rejects any draft token that disagrees.

Semantics (matching the V1 holder after PR #13's fixes):
- The budget counts GENERATED tokens strictly inside a reasoning block;
  the start/end markers themselves are not charged.
- A prompt that ends inside an open reasoning block starts generation
  in-think with the full budget (prompt tokens are never charged).
- When the budget is exhausted while in-think, the next logit row is forced
  to the reasoning-end token. Under speculative decoding the first draft
  position that would exceed the budget is forced instead, so a draft that
  closes the block naturally within budget is left for rejection sampling
  to validate.

Scope: single-token reasoning markers only (the qwen3 family and most
modern templates). Multi-token markers keep the V1-only rejection at the
input processor.
"""

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch

_NEG_INF = float("-inf")


def reasoning_markers_are_single_token(reasoning_config) -> bool:
    """V2 enforcement supports exactly one start and one end marker token."""
    if reasoning_config is None or not reasoning_config.enabled:
        return False
    starts = reasoning_config.reasoning_start_token_ids or []
    ends = reasoning_config.reasoning_end_token_ids or []
    return len(starts) == 1 and len(ends) == 1


class ThinkingBudgetState:
    """Per-request GPU state: remaining budget and in-think flag.

    Host-side bookkeeping is limited to which rows carry a budget at all
    (the fast skip check) and the one-time prompt scan at admission; all
    per-step accounting runs on GPU tensors so the async pipeline is never
    drained.
    """

    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
        reasoning_config,
    ) -> None:
        assert reasoning_markers_are_single_token(reasoning_config)
        self.start_id = int(reasoning_config.reasoning_start_token_ids[0])
        self.end_id = int(reasoning_config.reasoning_end_token_ids[0])
        self.device = device

        # -1 = no budget for this row (feature inert).
        self.budget_np = np.full(max_num_reqs, -1, dtype=np.int64)
        self.remaining = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.in_think = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        # GPU mirror of budget_np >= 0, so per-row gathers never touch host
        # state. Staged alongside the other admission writes.
        self.active = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        # Admission writes staged host-side, applied in one H2D per step.
        self._staged: list[tuple[int, int, bool, bool]] = []

    def add_request(
        self,
        req_idx: int,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams,
    ) -> None:
        budget = sampling_params.thinking_token_budget
        if budget is None or budget < 0:
            # None = unset; -1 = the explicit unlimited opt-out. The chat
            # protocol resolves -1 to None, but SamplingParams built
            # directly (offline LLM API) can carry it here -- staging it as
            # an active budget of -1 would force the end marker on the
            # first in-think token.
            self.budget_np[req_idx] = -1
            # Clear the GPU active flag so a previous occupant's budget
            # cannot leak into this row's masking.
            self._staged.append((req_idx, 0, False, False))
            return
        in_think0 = False
        if prompt_token_ids:
            arr = np.asarray(prompt_token_ids)
            starts = np.flatnonzero(arr == self.start_id)
            ends = np.flatnonzero(arr == self.end_id)
            last_start = int(starts[-1]) if starts.size else -1
            last_end = int(ends[-1]) if ends.size else -1
            in_think0 = last_start > last_end
        self.budget_np[req_idx] = budget
        self._staged.append((req_idx, budget, in_think0, True))

    def apply_staged_writes(self) -> None:
        if not self._staged:
            return
        idx = torch.tensor(
            [s[0] for s in self._staged], dtype=torch.long, device=self.device
        )
        rem = torch.tensor(
            [s[1] for s in self._staged], dtype=torch.int32, device=self.device
        )
        thk = torch.tensor(
            [s[2] for s in self._staged], dtype=torch.bool, device=self.device
        )
        act = torch.tensor(
            [s[3] for s in self._staged], dtype=torch.bool, device=self.device
        )
        self.remaining[idx] = rem
        self.in_think[idx] = thk
        self.active[idx] = act
        self._staged.clear()

    def _batch_is_inert(self, input_batch: InputBatch) -> bool:
        return not bool(np.any(self.budget_np[input_batch.idx_mapping_np] >= 0))

    def apply_mask(self, logits: torch.Tensor, input_batch: InputBatch) -> None:
        """Force exhausted rows to the reasoning-end token, in place.

        Runs where the grammar bitmask runs: on the target logits before the
        plain sampler or the rejection sampler consumes them. Row local
        position 0's input token is the previous step's last accepted token
        (already accounted by update_state); positions >= 1 carry this
        step's draft tokens, which are consumed positionally here.
        """
        if self._batch_is_inert(input_batch):
            return
        tok = input_batch.input_ids[input_batch.logits_indices]  # [rows]
        slots = input_batch.idx_mapping  # [num_reqs] -> req slot

        # Per-REQUEST running state; consumption accumulates across a
        # request's verify span, not per row.
        req_active = self.active[slots]
        req_in_think = self.in_think[slots]
        req_remaining = self.remaining[slots].to(torch.int64)
        req_consumed = torch.zeros_like(req_remaining)

        # Span geometry is host-known (cu_num_logits_np): row indices per
        # local position are computed on the host, so nothing here syncs.
        cu = input_batch.cu_num_logits_np[: input_batch.num_reqs + 1]
        spans = np.diff(cu)
        max_local = int(spans.max()) - 1 if spans.size else 0
        num_rows = tok.shape[0]
        force = torch.zeros(num_rows, dtype=torch.bool, device=self.device)

        # Local position 0: the input token is the previous step's last
        # accepted token, already folded into state by update_state; a row
        # is forced when the block is open with nothing remaining.
        rows0 = torch.from_numpy(cu[:-1].astype(np.int64)).to(self.device)
        force[rows0] = req_active & req_in_think & (req_remaining <= 0)

        for pos in range(1, max_local + 1):
            sel = np.flatnonzero(spans > pos)
            rows_p = torch.from_numpy((cu[sel] + pos).astype(np.int64)).to(self.device)
            reqs_p = torch.from_numpy(sel.astype(np.int64)).to(self.device)
            t_p = tok[rows_p]
            is_start = t_p == self.start_id
            is_end = t_p == self.end_id
            it = req_in_think[reqs_p]
            charge = it & ~is_start & ~is_end
            req_consumed[reqs_p] += charge.to(torch.int64)
            it = torch.where(is_start, True, it)
            it = torch.where(is_end, False, it)
            req_in_think[reqs_p] = it
            force[rows_p] = (
                req_active[reqs_p]
                & it
                & (req_consumed[reqs_p] >= req_remaining[reqs_p])
            )
        # Shape-static masked writes: boolean advanced indexing would give
        # data-dependent shapes, which drain the GPU queue per step on MPS.
        logits.masked_fill_(force.unsqueeze(1), _NEG_INF)
        end_col = logits[:, self.end_id]
        logits[:, self.end_id] = torch.where(force, torch.zeros_like(end_col), end_col)

    def update_state(
        self,
        input_batch: InputBatch,
        sampled_token_ids: torch.Tensor,  # [num_reqs, W]
        num_sampled: torch.Tensor,  # [num_reqs]
    ) -> None:
        """Fold this step's ACCEPTED tokens into the per-request state."""
        if self._batch_is_inert(input_batch):
            return
        slots = input_batch.idx_mapping  # [num_reqs] -> req slot
        in_think = self.in_think[slots]
        remaining = self.remaining[slots]
        width = sampled_token_ids.shape[1]
        for p in range(width):
            t = sampled_token_ids[:, p]
            valid = p < num_sampled
            is_start = valid & (t == self.start_id)
            is_end = valid & (t == self.end_id)
            charge = valid & in_think & ~is_start & ~is_end
            remaining = remaining - charge.to(torch.int32)
            in_think = torch.where(is_start, True, in_think)
            in_think = torch.where(is_end, False, in_think)
        self.in_think[slots] = in_think
        self.remaining[slots] = remaining
