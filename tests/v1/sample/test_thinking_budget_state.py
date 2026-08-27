# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for the native reasoning-token budget state machine."""

from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import get_effective_thinking_token_budget
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

THINK_START = 248_068
THINK_END = 248_069

REASONING_CONFIG = SimpleNamespace(
    enabled=True,
    reasoning_start_token_ids=[THINK_START],
    reasoning_end_token_ids=[THINK_END],
)


@pytest.fixture(autouse=True)
def force_cpu_logits_path(monkeypatch):
    """Avoid platform-specific index writes; these are state-machine tests."""
    from vllm.v1.sample import thinking_budget_state

    monkeypatch.setattr(
        thinking_budget_state,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: False),
    )


def make_holder(
    budget: int,
    *,
    prompt: list[int] | None = None,
    output: list[int] | None = None,
    num_spec_tokens: int = 0,
) -> tuple[ThinkingBudgetStateHolder, list[int]]:
    holder = ThinkingBudgetStateHolder(
        REASONING_CONFIG,
        max_num_seqs=8,
        num_spec_tokens=num_spec_tokens,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )
    live_output = output if output is not None else []
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[
                (
                    0,
                    SamplingParams(thinking_token_budget=budget),
                    prompt,
                    live_output,
                )
            ],
            moved=(),
        )
    )
    return holder, live_output


def apply_one(holder: ThinkingBudgetStateHolder) -> torch.Tensor:
    logits = torch.zeros((1, THINK_END + 1), dtype=torch.float32)
    return holder.apply_to_logits(logits, False, None)


def test_natural_close_disables_forcing():
    holder, output = make_holder(4)
    output.extend([THINK_START, 11, 12, THINK_END])
    holder.update_state([output], None)

    assert not holder._state[0]["in_think"]
    assert not holder._state[0]["in_end"]
    assert torch.count_nonzero(apply_one(holder)) == 0


def test_cap_forces_only_native_end_token():
    holder, output = make_holder(3)
    output.extend([THINK_START, 11, 12, 13])

    logits = torch.zeros((1, THINK_END + 1), dtype=torch.float32)
    Sampler().apply_logits_processors(
        logits,
        make_sampling_metadata(holder, output, None),
        predict_bonus_token=False,
    )
    assert holder._state[0]["think_count"] == 3
    assert holder._state[0]["in_end"]
    assert logits.argmax(dim=-1).item() == THINK_END
    assert logits[0, THINK_END].item() == 1e9


def test_output_without_reasoning_is_passthrough():
    holder, output = make_holder(1)
    output.extend([31, 32, 33])
    holder.update_state([output], None)

    assert not holder._state[0]["in_think"]
    assert torch.count_nonzero(apply_one(holder)) == 0


def test_prompt_open_reasoning_counts_generated_tokens_only():
    holder, output = make_holder(2, prompt=[7, THINK_START, 8, 9])
    assert holder._state[0]["think_count"] == 0

    output.extend([41, 42])
    holder.update_state([output], None)

    assert holder._state[0]["think_count"] == 2
    assert apply_one(holder).argmax(dim=-1).item() == THINK_END


def test_forced_close_continues_into_final_answer():
    holder, output = make_holder(1)
    output.extend([THINK_START, 51])
    holder.update_state([output], None)
    assert apply_one(holder).argmax(dim=-1).item() == THINK_END

    output.append(THINK_END)
    holder.update_state([output], None)

    assert not holder._state[0]["in_end"]
    assert torch.count_nonzero(apply_one(holder)) == 0


def test_swap_moves_budget_state_to_plain_row():
    holder, _ = make_holder(3)
    holder.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[(1, SamplingParams(), None, [])],
            moved=(),
        )
    )
    holder.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )

    assert 0 not in holder._state
    assert holder._state[1]["thinking_token_budget"] == 3


def test_unidirectional_plain_move_clears_replaced_budget_state():
    holder, _ = make_holder(3)
    holder.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    assert not holder._state


def make_sampling_metadata(
    holder: ThinkingBudgetStateHolder,
    output: list[int],
    proposal: list[list[int]] | None,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=[output],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=proposal,
        thinking_budget_state_holder=holder,
    )


def test_spec_proposal_crossing_cap_forces_exact_target_row():
    holder, output = make_holder(
        2,
        prompt=[THINK_START],
        num_spec_tokens=3,
    )
    proposal = [[61, 62, 63]]
    holder.update_state([output], proposal)

    logits = torch.zeros((3, THINK_END + 1), dtype=torch.float32)
    rejection_sampler = RejectionSampler(Sampler())
    metadata = SpecDecodeMetadata.make_dummy(proposal, torch.device("cpu"))
    rejection_sampler.apply_logits_processors(
        logits,
        make_sampling_metadata(holder, output, proposal),
        metadata,
    )

    assert holder._state[0]["force_index"] == [2]
    assert torch.count_nonzero(logits[:2]) == 0
    assert logits[2].argmax().item() == THINK_END


def test_spec_natural_close_before_cap_is_not_forced():
    holder, output = make_holder(
        2,
        prompt=[THINK_START],
        num_spec_tokens=3,
    )
    proposal = [[61, THINK_END, 63]]
    holder.update_state([output], proposal)

    logits = torch.zeros((3, THINK_END + 1), dtype=torch.float32)
    holder.apply_to_logits(logits, False, proposal)

    assert not holder._state[0]["in_end"]
    assert torch.count_nonzero(logits) == 0


def test_spec_cap_at_bonus_row_forces_bonus_without_touching_targets():
    holder, output = make_holder(
        3,
        prompt=[THINK_START],
        num_spec_tokens=3,
    )
    proposal = [[71, 72, 73]]
    holder.update_state([output], proposal)

    target_logits = torch.zeros((3, THINK_END + 1), dtype=torch.float32)
    holder.apply_to_logits(target_logits, False, proposal)
    bonus_logits = torch.zeros((1, THINK_END + 1), dtype=torch.float32)
    holder.apply_to_logits(bonus_logits, True, proposal)

    assert torch.count_nonzero(target_logits) == 0
    assert bonus_logits.argmax(dim=-1).item() == THINK_END


@pytest.mark.parametrize(
    ("request_budget", "max_tokens", "defaults", "expected"),
    [
        (None, 32_768, {}, None),
        (None, 32_768, {"thinking_token_budget": 16_384}, 16_384),
        (None, 8_192, {"thinking_token_budget": 16_384}, 8_191),
        (None, 2_048, {"thinking_token_budget": 16_384}, 2_047),
        (1_024, 8_192, {"thinking_token_budget": 16_384}, 1_024),
    ],
)
def test_effective_budget_respects_completion_ceiling(
    request_budget, max_tokens, defaults, expected
):
    assert (
        get_effective_thinking_token_budget(request_budget, max_tokens, defaults)
        == expected
    )


def test_request_minus_one_opts_out_of_server_default():
    request = ChatCompletionRequest(
        model="Qwen3.8-27B",
        messages=[{"role": "user", "content": "hi"}],
        thinking_token_budget=-1,
    )
    assert request.thinking_token_budget == -1

    params = request.to_sampling_params(
        max_tokens=32_768,
        default_sampling_params={"thinking_token_budget": 16_384},
    )
    assert params.thinking_token_budget is None


LEVEL_BUDGETS = {
    "low": 4_096,
    "medium": 8_192,
    "high": 16_384,
    "xhigh": 32_768,
}


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    [
        ("low", 4_096),
        ("medium", 8_192),
        ("high", 16_384),
        ("xhigh", 32_767),
        # An omitted request level has a documented medium default.
        (None, 8_192),
        # Partial maps use medium for a valid level without its own entry.
        ("minimal", 8_192),
        ("none", None),
    ],
)
def test_effective_budget_resolves_reasoning_effort_map(reasoning_effort, expected):
    defaults = {
        "thinking_token_budget": LEVEL_BUDGETS,
    }
    assert (
        get_effective_thinking_token_budget(None, 32_768, defaults, reasoning_effort)
        == expected
    )


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"), [("low", 4_096), (None, 8_192)]
)
def test_chat_request_passes_reasoning_effort_to_budget_map(reasoning_effort, expected):
    request = ChatCompletionRequest(
        model="Qwen3.8-27B",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=reasoning_effort,
    )
    params = request.to_sampling_params(
        max_tokens=32_768,
        default_sampling_params={"thinking_token_budget": LEVEL_BUDGETS},
    )
    assert params.thinking_token_budget == expected


@pytest.mark.parametrize("request_budget", [1_024, -1])
def test_explicit_request_budget_overrides_reasoning_effort_map(request_budget):
    expected = None if request_budget == -1 else request_budget
    assert (
        get_effective_thinking_token_budget(
            request_budget,
            32_768,
            {"thinking_token_budget": LEVEL_BUDGETS},
            "xhigh",
        )
        == expected
    )


@pytest.mark.parametrize(
    "budget_map",
    [
        {"low": 4_096},
        {"medium": 8_192, "turbo": 16_384},
        {"medium": None},
        {"medium": 1.5},
    ],
)
def test_invalid_reasoning_effort_budget_map_is_rejected(budget_map):
    with pytest.raises(VLLMValidationError):
        get_effective_thinking_token_budget(
            None,
            32_768,
            {"thinking_token_budget": budget_map},
            "medium",
        )


def test_reasoning_effort_none_ignores_scalar_default_and_explicit_budget():
    """`none` disables thinking at the template layer, so no cutoff applies
    even against a scalar server default or an explicit request budget."""
    from vllm.sampling_params import get_effective_thinking_token_budget

    assert (
        get_effective_thinking_token_budget(
            None, 1000, {"thinking_token_budget": 256}, "none"
        )
        is None
    )
    assert get_effective_thinking_token_budget(64, 1000, {}, "none") is None
