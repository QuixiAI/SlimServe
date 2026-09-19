# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thinking-budget forcing must respect grammar and speculative boundaries."""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample import thinking_budget_state as budget_module
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

START, END = 256, 257
VOCAB_SIZE = 258


def make_holder(budget, output, drafts, num_spec=3):
    holder = ThinkingBudgetStateHolder(
        SimpleNamespace(
            reasoning_start_token_ids=[START], reasoning_end_token_ids=[END]
        ),
        1,
        num_spec,
        torch.device("cpu"),
        False,
    )
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, SamplingParams(thinking_token_budget=budget), [START], [])],
            moved=[],
        )
    )
    holder.update_state([output], [drafts])
    return holder


def json_grammar_allowed():
    import xgrammar as xgr

    tokenizer = xgr.TokenizerInfo(
        encoded_vocab=[bytes([i]) for i in range(256)] + [b"<think>", b"</think>"],
        vocab_type=xgr.VocabType.RAW,
        vocab_size=VOCAB_SIZE,
        stop_token_ids=[],
    )
    compiler = xgr.GrammarCompiler(tokenizer, max_threads=1)
    matcher = xgr.GrammarMatcher(compiler.compile_grammar('root ::= "{}"'))
    mask = xgr.allocate_token_bitmask(1, VOCAB_SIZE)
    matcher.fill_next_token_bitmask(mask, 0)
    ids = torch.arange(VOCAB_SIZE)
    allowed = ((mask[0, ids // 32] >> (ids % 32)) & 1).bool()
    assert not allowed[END]
    assert allowed[ord("{")]
    return allowed


@pytest.mark.parametrize(
    "rocm,contiguous", [(False, True), (True, True), (True, False)]
)
@pytest.mark.parametrize("mode", ["plain", "target", "bonus"])
def test_budget_never_resurrects_a_grammar_masked_token(
    monkeypatch, rocm, contiguous, mode
):
    monkeypatch.setattr(
        budget_module, "current_platform", SimpleNamespace(is_rocm=lambda: rocm)
    )
    if mode == "plain":
        drafts, output, budget, nrows, target, nspec = [], [], 0, 1, 0, 0
    elif mode == "target":
        drafts, output, budget, nrows, target, nspec = [120] * 3, [120], 2, 3, 1, 3
    else:
        drafts, output, budget, nrows, target, nspec = [120] * 3, [120], 4, 1, 0, 3
    holder = make_holder(budget, output, drafts, nspec)
    if contiguous:
        logits = torch.zeros(nrows, VOCAB_SIZE)
    else:
        logits = torch.zeros(nrows, VOCAB_SIZE * 2)[:, ::2]
        assert not logits.is_contiguous()
    logits[target].masked_fill_(~json_grammar_allowed(), -torch.inf)
    before = logits.clone()
    holder.apply_to_logits(logits, mode == "bonus", [drafts])
    # Preserve the entire row, including allowed alternatives. Do not replace
    # the grammar mask with a forced closer or create an all-infinite row.
    assert torch.equal(logits, before)
    assert logits[target].argmax().item() == ord("{")
    assert torch.isfinite(logits[target]).any()


@pytest.mark.parametrize(
    "rocm,contiguous", [(False, True), (True, True), (True, False)]
)
def test_zero_budget_still_forces_allowed_closer(monkeypatch, rocm, contiguous):
    monkeypatch.setattr(
        budget_module, "current_platform", SimpleNamespace(is_rocm=lambda: rocm)
    )
    holder = make_holder(0, [], [], num_spec=0)
    logits = torch.zeros(1, VOCAB_SIZE * (1 if contiguous else 2))
    if not contiguous:
        logits = logits[:, ::2]
    holder.apply_to_logits(logits, False, [[]])
    assert logits.argmax().item() == END
    assert logits[0, END] == 1e9


@pytest.mark.parametrize("bonus", [False, True])
def test_natural_draft_closer_prevents_duplicate_forcing(bonus):
    drafts = [END, ord("{"), ord("}")]
    holder = make_holder(4 if bonus else 2, [120], drafts)
    logits = torch.zeros(1 if bonus else 3, VOCAB_SIZE)
    before = logits.clone()
    holder.apply_to_logits(logits, bonus, [drafts])
    assert torch.equal(logits, before)
    # Draft speculation must not permanently switch off the budget: a rejection
    # before the natural closer leaves reasoning active for the next step.
    holder.update_state([[120, 121]], [[120, 120, 120]])
    retry = torch.zeros(3, VOCAB_SIZE)
    holder.apply_to_logits(retry, False, [[120, 120, 120]])
    assert retry[2 if bonus else 0].argmax().item() == END


def test_end_marker_at_forced_position_is_not_mistaken_for_prior_end():
    drafts = [120, END, ord("{")]
    holder = make_holder(2, [120], drafts)
    logits = torch.zeros(3, VOCAB_SIZE)
    holder.apply_to_logits(logits, False, [drafts])
    assert logits[1].argmax().item() == END


def test_two_thousand_token_budget_still_forces_at_correct_draft_row():
    drafts = [120, 120, 120]
    holder = make_holder(2000, [120] * 1999, drafts)
    logits = torch.zeros(3, VOCAB_SIZE)
    holder.apply_to_logits(logits, False, [drafts])
    assert logits[1, END] == 1e9
    assert logits[0, END] == 0
    assert logits[2, END] == 0


def test_split_multitoken_end_marker_in_draft_prefix_does_not_force_again():
    second_end = END + 1
    drafts = [second_end, 120, 120]
    holder = ThinkingBudgetStateHolder(
        SimpleNamespace(
            reasoning_start_token_ids=[START],
            reasoning_end_token_ids=[END, second_end],
        ),
        1,
        3,
        torch.device("cpu"),
        False,
    )
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, SamplingParams(thinking_token_budget=3), [START], [])],
            moved=[],
        )
    )
    holder.update_state([[120, END]], [drafts])
    logits = torch.zeros(3, VOCAB_SIZE + 1)
    holder.apply_to_logits(logits, False, [drafts])
    assert torch.equal(logits, torch.zeros_like(logits))
