from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.structured_output import StructuredOutputManager

EOS = 31  # Exercise the signed bit in an int32 mask word.
STOP = 40
END_THINK = 7
TEXT = 10


class Reasoner:
    def is_reasoning_end_streaming(self, history, delta):
        return END_THINK in delta


def build(required=True, ended=False, spec=3):
    manager = object.__new__(StructuredOutputManager)
    manager.vllm_config = SimpleNamespace(
        num_speculative_tokens=spec, model_config=SimpleNamespace(is_diffusion=False)
    )
    manager._grammar_bitmask = torch.full((8, 2), -1, dtype=torch.int32)
    manager._apply_rows = np.zeros(8, dtype=bool)
    manager._full_mask = torch.tensor(-1, dtype=torch.int32)
    manager.fill_bitmask_parallel_threshold = 128
    manager.enable_in_reasoning = False
    manager.reasoner_cls = Reasoner
    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.validate_tokens.side_effect = lambda tokens: tokens
    grammar.accept_tokens.return_value = True
    # Once reasoning ends, this fake grammar allows EOS. The test verifies that
    # the extra reasoning mask does not override the grammar after transition.
    grammar.fill_bitmask.side_effect = lambda mask, index: mask[index].fill_(-1)
    request = SimpleNamespace(
        structured_output_request=SimpleNamespace(
            grammar=grammar,
            reasoner=Reasoner(),
            reasoning_ended=ended,
            params=SimpleNamespace(_required_tool_call=required),
        ),
        sampling_params=SimpleNamespace(all_stop_token_ids={EOS, STOP}),
        all_token_ids=[1, TEXT],
        prompt_token_ids=[1],
    )
    return manager, request


def allowed(mask, row, token):
    return bool(int(mask[row, token // 32]) & (1 << (token % 32)))


@pytest.mark.parametrize("drafts", [[], [TEXT, TEXT, TEXT]])
def test_required_reasoning_masks_stop_in_all_draft_and_bonus_rows(drafts):
    manager, request = build()
    mask, applied = manager.grammar_bitmask({"r": request}, ["r"], {"r": drafts})
    for row in range(len(drafts) + 1):
        assert applied[row]
        assert not allowed(mask, row, EOS)
        assert not allowed(mask, row, STOP)
        assert allowed(mask, row, TEXT)
        assert allowed(mask, row, END_THINK)


def test_mid_window_reasoning_end_returns_eos_control_to_grammar():
    manager, request = build()
    mask, applied = manager.grammar_bitmask(
        {"r": request}, ["r"], {"r": [TEXT, END_THINK, TEXT]}
    )
    assert [allowed(mask, row, EOS) for row in range(4)] == [False, False, True, True]


@pytest.mark.parametrize("required,ended", [(False, False), (True, True)])
def test_auto_and_finished_reasoning_do_not_get_extra_stop_mask(required, ended):
    manager, request = build(required=required, ended=ended)
    mask, applied = manager.grammar_bitmask({"r": request}, ["r"], {"r": []})
    assert allowed(mask, 0, EOS)


def test_parallel_non_speculative_masking_and_reused_rows():
    manager, request = build(spec=0)
    manager.fill_bitmask_parallel_threshold = 0
    manager.fill_bitmask_parallel_batch_size = 1
    with ThreadPoolExecutor(1) as executor:
        manager.executor_for_fillmask = executor
        mask, applied = manager.grammar_bitmask({"r": request}, ["r"], {})
        assert applied[0]
        assert not allowed(mask, 0, EOS)
        request.structured_output_request.params._required_tool_call = False
        mask, applied = manager.grammar_bitmask({"r": request}, ["r"], {})
        assert not applied[0]
        assert allowed(mask, 0, EOS)


def test_required_metadata_survives_reasoning_tag_replacement():
    params = StructuredOutputsParams(structural_tag="{}", _required_tool_call=True)
    assert replace(params, structural_tag='{"format": {}}')._required_tool_call


@pytest.mark.parametrize(
    "choice,required",
    [
        ("required", True),
        ({"type": "function", "name": "lookup"}, True),
        (
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "lookup"}],
            },
            True,
        ),
        ("auto", False),
        (
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "lookup"}],
            },
            False,
        ),
    ],
)
def test_responses_tool_choice_reaches_required_metadata(choice, required):
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.parser.abstract_parser import DelegatingParser

    request = ResponsesRequest(
        input="Look it up",
        tool_choice=choice,
        tools=[{"type": "function", "name": "lookup", "parameters": {}}],
    )
    parser = object.__new__(DelegatingParser)
    parser._tool_parser = Mock()
    parser._tool_parser.get_structural_tag.return_value.model_dump.return_value = {}
    parser._apply_structural_tag(request)
    assert request.structured_outputs._required_tool_call is required
