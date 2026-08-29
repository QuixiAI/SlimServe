# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 thinking-budget enforcement (vllm/v1/worker/gpu/sample/thinking_budget.py).

Mirrors the V1 holder's semantics as pinned by PR #13: budgets count
generated in-think tokens only, prompt-open blocks start at zero consumed,
exhaustion forces the reasoning-end token, and under speculative decoding
the first violating draft position is forced while a natural in-budget
close is left to rejection sampling.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.sample.thinking_budget import (
    ThinkingBudgetState,
    reasoning_markers_are_single_token,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple Metal (MPS)",
)

DEV = "mps"
START, END = 100, 101
VOCAB = 128


def make_state(max_reqs=4):
    rc = SimpleNamespace(
        enabled=True,
        reasoning_start_token_ids=[START],
        reasoning_end_token_ids=[END],
    )
    assert reasoning_markers_are_single_token(rc)
    return ThinkingBudgetState(max_reqs, torch.device(DEV), rc)


def make_batch(slots, spans, input_tokens):
    """Minimal InputBatch stand-in: one row per (slot, local_pos)."""
    rows = sum(spans)
    local, mapping = [], []
    for slot, span in zip(slots, spans):
        for p in range(span):
            mapping.append(slot)
            local.append(p)
    cu = np.zeros(len(slots) + 1, dtype=np.int64)
    cu[1:] = np.cumsum(spans)
    toks = torch.tensor(input_tokens, dtype=torch.long, device=DEV)
    return SimpleNamespace(
        num_reqs=len(slots),
        idx_mapping=torch.tensor(slots, dtype=torch.long, device=DEV),
        idx_mapping_np=np.array(slots),
        expanded_idx_mapping=torch.tensor(mapping, dtype=torch.long, device=DEV),
        expanded_local_pos=torch.tensor(local, dtype=torch.long, device=DEV),
        logits_indices=torch.arange(rows, device=DEV),
        input_ids=toks,
        cu_num_logits_np=cu,
    )


def sp(budget):
    return SimpleNamespace(thinking_token_budget=budget)


def forced_rows(state, batch):
    rows = batch.expanded_idx_mapping.shape[0]
    logits = torch.zeros(rows, VOCAB, device=DEV)
    state.apply_mask(logits, batch)
    torch.mps.synchronize()
    return [
        i
        for i in range(rows)
        if torch.isinf(logits[i, 0]).item() and logits[i, END].item() == 0.0
    ]


def test_no_budget_rows_untouched():
    st = make_state()
    st.add_request(0, [1, 2, 3], sp(None))
    st.apply_staged_writes()
    batch = make_batch([0], [1], [7])
    assert forced_rows(st, batch) == []


def test_prompt_open_block_exhausts_after_budget_tokens():
    st = make_state()
    st.add_request(0, [1, START, 5, 6], sp(2))  # open block, budget 2
    st.apply_staged_writes()
    batch = make_batch([0], [1], [7])
    # Two in-think tokens accepted -> budget spent -> next logit forced.
    st.update_state(
        batch,
        torch.tensor([[8]], device=DEV),
        torch.tensor([1], device=DEV),
    )
    assert forced_rows(st, batch) == []
    st.update_state(
        batch,
        torch.tensor([[9]], device=DEV),
        torch.tensor([1], device=DEV),
    )
    assert forced_rows(st, batch) == [0]


def test_budget_zero_prompt_open_forces_immediately():
    st = make_state()
    st.add_request(0, [1, START], sp(0))
    st.apply_staged_writes()
    assert forced_rows(st, make_batch([0], [1], [7])) == [0]


def test_markers_are_not_charged():
    st = make_state()
    st.add_request(0, [1], sp(1))
    st.apply_staged_writes()
    batch = make_batch([0], [1], [7])
    # start marker accepted: enters think, charges nothing.
    st.update_state(
        batch, torch.tensor([[START]], device=DEV), torch.tensor([1], device=DEV)
    )
    assert forced_rows(st, batch) == []
    # end marker accepted: leaves think, charges nothing, no forcing after.
    st.update_state(
        batch, torch.tensor([[END]], device=DEV), torch.tensor([1], device=DEV)
    )
    assert forced_rows(st, batch) == []


def test_spec_draft_crossing_budget_forces_at_violation():
    st = make_state()
    st.add_request(0, [1, START], sp(2))  # open block, 2 tokens left
    st.apply_staged_writes()
    # Verify span of 4 rows; draft tokens at local 1..3 are plain think
    # tokens. Consumption: row1 -> 1, row2 -> 2 (== budget -> forced),
    # row3 also >= budget -> forced.
    batch = make_batch([0], [4], [7, 8, 9, 10])
    assert forced_rows(st, batch) == [2, 3]


def test_spec_natural_close_within_budget_is_not_forced():
    st = make_state()
    st.add_request(0, [1, START], sp(2))
    st.apply_staged_writes()
    # Draft closes the block at local 2 after one think token: no forcing.
    batch = make_batch([0], [4], [7, 8, END, 11])
    assert forced_rows(st, batch) == []


def test_rejected_draft_tokens_not_charged():
    st = make_state()
    st.add_request(0, [1, START], sp(5))
    st.apply_staged_writes()
    batch = make_batch([0], [1], [7])
    # Sampler emitted width 4 but only 2 accepted: charge 2, not 4.
    st.update_state(
        batch,
        torch.tensor([[8, 9, 10, 11]], device=DEV),
        torch.tensor([2], device=DEV),
    )
    # 3 tokens of budget left -> a 4-row span forces only at the row where
    # cumulative draft consumption reaches 3.
    span = make_batch([0], [4], [7, 8, 9, 10])
    assert forced_rows(st, span) == [3]


def test_slot_reuse_clears_previous_budget():
    st = make_state()
    st.add_request(0, [1, START], sp(0))
    st.apply_staged_writes()
    assert forced_rows(st, make_batch([0], [1], [7])) == [0]
    # New unbudgeted occupant of the same slot must not inherit forcing.
    st.add_request(0, [1, 2], sp(None))
    st.apply_staged_writes()
    assert forced_rows(st, make_batch([0], [1], [7])) == []


def test_mixed_batch_only_budgeted_rows_forced():
    st = make_state()
    st.add_request(0, [1, START], sp(0))
    st.add_request(1, [1, START], sp(1000))
    st.apply_staged_writes()
    batch = make_batch([0, 1], [1, 1], [7, 7])
    assert forced_rows(st, batch) == [0]


def test_explicit_minus_one_budget_is_inactive_even_in_mixed_batch():
    """-1 is the unlimited opt-out: a direct SamplingParams can carry it to
    add_request, and it must not become an active budget of -1 (which would
    force the end marker on the first in-think token)."""
    st = make_state()
    st.add_request(0, [1, START], sp(-1))  # opted out, in an open block
    st.add_request(1, [1, START], sp(0))  # exhausted budget
    st.apply_staged_writes()
    batch = make_batch([0, 1], [1, 1], [7, 7])
    assert forced_rows(st, batch) == [1]
