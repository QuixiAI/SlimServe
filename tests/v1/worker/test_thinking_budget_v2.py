# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 thinking-budget enforcement (vllm/v1/worker/gpu/sample/thinking_budget.py).

Reference parity with llama.cpp common/reasoning-budget.cpp (issue #20) plus
the operator-directed wrap-up nudge (2026-08-30): budgets count generated
in-think tokens only, a wrap-up message is injected uncharged at the nudge
fraction, exhaustion HARD-forces the end marker (held across a UTF-8
boundary), a natural close is never overridden, and a second reasoning
block re-arms a fresh budget and a fresh nudge. Under speculative decoding
the first violating draft position is forced while a natural in-budget
close is left to rejection sampling.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.sample.thinking_budget import (
    ThinkingBudgetState,
    _kmp_dfa,
    reasoning_budget_supported,
)

if torch.cuda.is_available():
    DEV = "cuda"
elif torch.backends.mps.is_available():
    DEV = "mps"
else:
    DEV = "cpu"

START, END = 100, 101
NUDGE = [110, 111, 112]  # 3-token wrap-up message
INCOMPLETE = 120  # token whose bytes end mid-codepoint
VOCAB = 128


def rc(
    starts=(START,),
    ends=(END,),
    nudge=(),
    fraction=0.85,
    incomplete=(),
):
    return SimpleNamespace(
        enabled=True,
        reasoning_start_token_ids=list(starts),
        reasoning_end_token_ids=list(ends),
        thinking_budget_message_token_ids=list(nudge),
        thinking_budget_nudge_fraction=fraction,
        incomplete_utf8_token_ids=list(incomplete),
    )


def make_state(max_reqs=4, **rc_kwargs):
    config = rc(**rc_kwargs)
    assert reasoning_budget_supported(config)
    return ThinkingBudgetState(max_reqs, torch.device(DEV), config, VOCAB)


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


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def forced_map(state, batch):
    """{row: forced_token} for rows whose logits were force-masked."""
    rows = batch.expanded_idx_mapping.shape[0]
    logits = torch.zeros(rows, VOCAB, device=DEV)
    state.apply_mask(logits, batch)
    sync()
    out = {}
    for i in range(rows):
        if torch.isinf(logits[i]).sum().item() >= VOCAB - 1:
            out[i] = int((logits[i] == 0).nonzero()[0].item())
    return out


def step(state, slot, token, budget_batch=None):
    """Fold one accepted token for one request occupying `slot`."""
    batch = budget_batch or make_batch([slot], [1], [0])
    sampled = torch.tensor([[token]], dtype=torch.long, device=DEV)
    n = torch.tensor([1], dtype=torch.long, device=DEV)
    state.update_state(batch, sampled, n)


def run_tokens(state, slot, tokens):
    for t in tokens:
        step(state, slot, t)


def forced_now(state, slot):
    batch = make_batch([slot], [1], [0])
    fm = forced_map(state, batch)
    return fm.get(0)


# ------------------------------------------------------------------ basics


def test_no_budget_rows_untouched():
    st = make_state()
    st.add_request(0, [1, 2], sp(None))
    st.apply_staged_writes()
    batch = make_batch([0], [1], [5])
    assert forced_map(st, batch) == {}


def test_prompt_open_block_hard_close_after_budget():
    st = make_state()
    st.add_request(0, [1, START, 2], sp(3))
    st.apply_staged_writes()
    run_tokens(st, 0, [10, 11])
    assert forced_now(st, 0) is None  # 2 of 3 consumed
    step(st, 0, 12)  # third: exhausted
    assert forced_now(st, 0) == END
    step(st, 0, END)  # forced token accepted -> DONE
    assert forced_now(st, 0) is None


def test_budget_zero_prompt_open_forces_immediately():
    st = make_state()
    st.add_request(0, [1, START], sp(0))
    st.apply_staged_writes()
    assert forced_now(st, 0) == END


def test_single_token_markers_are_not_charged():
    st = make_state()
    st.add_request(0, [1], sp(2))
    st.apply_staged_writes()
    run_tokens(st, 0, [START, 10, END, START, 20])
    # markers uncharged; each block spent 1 of its own budget of 2
    assert forced_now(st, 0) is None


def test_natural_close_within_budget_never_forced():
    st = make_state()
    st.add_request(0, [START], sp(100))
    st.apply_staged_writes()
    run_tokens(st, 0, [10, 11, END, 12, 13])
    assert forced_now(st, 0) is None


# ------------------------------------------------------------------ re-arm


def test_second_block_gets_fresh_budget():
    st = make_state()
    st.add_request(0, [1], sp(2))
    st.apply_staged_writes()
    run_tokens(st, 0, [START, 10, 11])  # exhausts block 1
    assert forced_now(st, 0) == END
    run_tokens(st, 0, [END])  # forced close accepted -> DONE
    assert forced_now(st, 0) is None
    run_tokens(st, 0, [START])  # re-arm
    assert forced_now(st, 0) is None  # fresh budget, not inherited zero
    run_tokens(st, 0, [20, 21])
    assert forced_now(st, 0) == END  # fresh budget also enforces


# ------------------------------------------------------------------- utf8


def test_utf8_hold_delays_close_until_codepoint_complete():
    st = make_state(incomplete=(INCOMPLETE,))
    st.add_request(0, [START], sp(2))
    st.apply_staged_writes()
    run_tokens(st, 0, [10, INCOMPLETE])  # exhausts ON an incomplete token
    assert forced_now(st, 0) is None  # held: waiting for the codepoint
    step(st, 0, INCOMPLETE)  # still incomplete: keep waiting
    assert forced_now(st, 0) is None
    step(st, 0, 11)  # codepoint closed
    assert forced_now(st, 0) == END


def test_utf8_wait_still_honors_natural_close():
    st = make_state(incomplete=(INCOMPLETE,))
    st.add_request(0, [START], sp(1))
    st.apply_staged_writes()
    run_tokens(st, 0, [INCOMPLETE])  # exhausted, held
    run_tokens(st, 0, [END])  # model closes naturally while held
    assert forced_now(st, 0) is None
    run_tokens(st, 0, [30])
    assert forced_now(st, 0) is None  # DONE: passthrough


# ------------------------------------------------------------------- nudge


def test_nudge_injected_at_fraction_then_counting_resumes():
    st = make_state(nudge=NUDGE, fraction=0.5)
    st.add_request(0, [START], sp(10))  # nudge at remaining <= 5
    st.apply_staged_writes()
    run_tokens(st, 0, [10, 11, 12, 13])
    assert forced_now(st, 0) is None  # remaining 6 > 5
    step(st, 0, 14)  # remaining 5: trigger
    assert forced_now(st, 0) == NUDGE[0]
    step(st, 0, NUDGE[0])
    assert forced_now(st, 0) == NUDGE[1]
    step(st, 0, NUDGE[1])
    assert forced_now(st, 0) == NUDGE[2]
    step(st, 0, NUDGE[2])
    # message complete: counting resumed, model free again
    assert forced_now(st, 0) is None
    # nudge tokens were NOT charged: 5 remaining for real tokens
    run_tokens(st, 0, [20, 21, 22, 23])
    assert forced_now(st, 0) is None
    step(st, 0, 24)  # remaining 0: hard cutoff
    assert forced_now(st, 0) == END


def test_nudge_fires_once_per_block_but_rearms():
    st = make_state(nudge=NUDGE, fraction=0.5)
    st.add_request(0, [START], sp(4))  # nudge at remaining <= 2
    st.apply_staged_writes()
    run_tokens(st, 0, [10, 11])  # trigger
    assert forced_now(st, 0) == NUDGE[0]
    run_tokens(st, 0, list(NUDGE))
    assert forced_now(st, 0) is None
    run_tokens(st, 0, [12])  # remaining 1; no second nudge
    assert forced_now(st, 0) is None
    run_tokens(st, 0, [13, END])  # exhaust -> forced close accepted
    run_tokens(st, 0, [START, 20, 21])  # new block: nudge re-armed
    assert forced_now(st, 0) == NUDGE[0]


def test_nudge_held_across_utf8_boundary():
    st = make_state(nudge=NUDGE, fraction=0.5, incomplete=(INCOMPLETE,))
    st.add_request(0, [START], sp(4))  # nudge at remaining <= 2
    st.apply_staged_writes()
    run_tokens(st, 0, [10, INCOMPLETE])  # trigger lands mid-codepoint
    assert forced_now(st, 0) is None  # held
    step(st, 0, 11)  # codepoint closed
    assert forced_now(st, 0) == NUDGE[0]


def test_no_message_means_no_nudge():
    st = make_state(nudge=(), fraction=0.85)
    st.add_request(0, [START], sp(10))
    st.apply_staged_writes()
    run_tokens(st, 0, [10] * 9)
    assert forced_now(st, 0) is None  # only the hard close at exhaustion
    step(st, 0, 11)
    assert forced_now(st, 0) == END


# ------------------------------------------------- multi-token markers


def test_multi_token_markers_match_and_do_not_charge_completion():
    st = make_state(starts=(80, 81), ends=(90, 91))
    st.add_request(0, [1], sp(3))
    st.apply_staged_writes()
    run_tokens(st, 0, [80, 81])  # start marker
    run_tokens(st, 0, [10])  # charged (1 of 3)
    # The end-marker PREFIX token is charged like a normal token (2 of 3,
    # reference semantics); the completing token matches before the charge.
    run_tokens(st, 0, [90, 91])
    assert forced_now(st, 0) is None


def test_multi_token_forced_close_emitted_in_order():
    st = make_state(starts=(80, 81), ends=(90, 91))
    st.add_request(0, [80, 81], sp(1))
    st.apply_staged_writes()
    run_tokens(st, 0, [10])  # exhausted
    assert forced_now(st, 0) == 90
    step(st, 0, 90)
    assert forced_now(st, 0) == 91
    step(st, 0, 91)
    assert forced_now(st, 0) is None  # DONE


def test_kmp_dfa_handles_self_overlap():
    dfa = _kmp_dfa([7, 7, 8], 16)
    s = 0
    hits = []
    for t in [7, 7, 7, 8]:
        s = int(dfa[s, t]) if s < 3 else 0
        if s == 3:
            hits.append(t)
            s = 0
    assert hits == [8]  # aab matched inside aaab


# ------------------------------------------------------- speculative spans


def test_spec_draft_crossing_budget_forces_at_violation():
    st = make_state()
    st.add_request(0, [START], sp(2))
    st.apply_staged_writes()
    step(st, 0, 10)  # 1 consumed, 1 remaining
    # verify span: pos0 row + 2 draft tokens [11, 12]
    batch = make_batch([0], [3], [0, 11, 12])
    fm = forced_map(st, batch)
    assert 0 not in fm  # remaining 1: the next token is still free
    # Draft token at pos1 consumes the last unit, so row 1 (predicting the
    # following token) forces the close; the draft proposed 12 there, so
    # rejection sampling truncates at that position.
    assert fm.get(1) == END


def test_spec_natural_close_within_budget_is_not_forced():
    st = make_state()
    st.add_request(0, [START], sp(2))
    st.apply_staged_writes()
    step(st, 0, 10)
    batch = make_batch([0], [3], [0, END, 12])
    fm = forced_map(st, batch)
    assert fm == {}  # draft closed the block naturally: passthrough


def test_spec_rejected_draft_tokens_not_charged():
    st = make_state()
    st.add_request(0, [START], sp(3))
    st.apply_staged_writes()
    # apply_mask simulates 2 draft tokens, but only 1 is accepted
    batch = make_batch([0], [3], [0, 11, 12])
    forced_map(st, batch)
    sampled = torch.tensor([[11, 0, 0]], dtype=torch.long, device=DEV)
    st.update_state(batch, sampled, torch.tensor([1], device=DEV))
    # only 1 of 3 consumed; two more free tokens before the close
    run_tokens(st, 0, [13])
    assert forced_now(st, 0) is None
    run_tokens(st, 0, [14])
    assert forced_now(st, 0) == END


def test_spec_nudge_positions_forced_in_span():
    st = make_state(nudge=NUDGE, fraction=0.5)
    st.add_request(0, [START], sp(4))  # nudge at remaining <= 2
    st.apply_staged_writes()
    step(st, 0, 10)  # remaining 3
    # span: pos0 + drafts [11, NUDGE0]; pos1 charges to remaining 2 ->
    # trigger, so pos1's row forces NUDGE[0]; the draft agreed, so pos2
    # forces NUDGE[1].
    batch = make_batch([0], [3], [0, 11, NUDGE[0]])
    fm = forced_map(st, batch)
    assert 0 not in fm
    assert fm.get(1) == NUDGE[0]
    assert fm.get(2) == NUDGE[1]


# ---------------------------------------------------------- housekeeping


def test_slot_reuse_clears_previous_budget():
    st = make_state()
    st.add_request(0, [START], sp(1))
    st.apply_staged_writes()
    run_tokens(st, 0, [10])
    assert forced_now(st, 0) == END
    st.add_request(0, [1, 2], sp(None))  # new occupant, no budget
    st.apply_staged_writes()
    assert forced_now(st, 0) is None


def test_mixed_batch_only_budgeted_rows_forced():
    st = make_state()
    st.add_request(0, [START], sp(1))
    st.add_request(1, [START], sp(None))
    st.apply_staged_writes()
    batch2 = make_batch([0, 1], [1, 1], [0, 0])
    sampled = torch.tensor([[10], [10]], dtype=torch.long, device=DEV)
    st.update_state(batch2, sampled, torch.tensor([1, 1], device=DEV))
    fm = forced_map(st, batch2)
    assert fm.get(0) == END
    assert 1 not in fm


def test_explicit_minus_one_budget_is_inactive():
    st = make_state()
    st.add_request(0, [START], sp(-1))
    st.apply_staged_writes()
    run_tokens(st, 0, [10, 11, 12])
    assert forced_now(st, 0) is None
