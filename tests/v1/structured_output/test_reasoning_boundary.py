from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from vllm.v1.structured_output import StructuredOutputManager

MARKER = 248069
NEWLINE = 198
VALID = 5005


class MarkerReasoner:
    def is_reasoning_end_streaming(self, _input_ids, delta_ids):
        return MARKER in list(delta_ids)


class FakeBitmask:
    shape = (16, 1)

    def __getitem__(self, _index):
        return self

    def numpy(self):
        return self


def make_manager_and_request(grammar):
    manager = object.__new__(StructuredOutputManager)
    manager.vllm_config = SimpleNamespace(
        num_speculative_tokens=3,
        model_config=SimpleNamespace(is_diffusion=False),
    )
    manager._grammar_bitmask = FakeBitmask()
    manager._apply_rows = np.zeros(16, dtype=bool)
    manager._fill_bitmasks = Mock()
    manager.fill_bitmask_parallel_threshold = 128
    manager.enable_in_reasoning = False
    manager.reasoner_cls = MarkerReasoner
    manager.tokenizer = None

    structured = SimpleNamespace(
        grammar=grammar,
        params=SimpleNamespace(_required_tool_call=False),
        reasoner=MarkerReasoner(),
        reasoning_ended=False,
        reasoning_parser_kwargs=None,
    )
    request = SimpleNamespace(
        request_id="req",
        structured_output_request=structured,
        use_structured_output=True,
        prompt_token_ids=[1],
        all_token_ids=[1, 2],
    )
    return manager, request


def test_invalid_draft_after_reasoning_boundary_is_not_advanced():
    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.validate_tokens.side_effect = lambda tokens: (
        tokens if tokens == [VALID] else []
    )
    manager, request = make_manager_and_request(grammar)

    manager.grammar_bitmask(
        {"req": request},
        ["req"],
        {"req": [MARKER, NEWLINE, VALID]},
    )

    # The newline was drafted while reasoning was unconstrained. Its rejection
    # is expected, so the mutating/logging accept path must not see it. The
    # later draft is unreachable after this first rejection.
    grammar.validate_tokens.assert_called_once_with([NEWLINE])
    grammar.accept_tokens.assert_not_called()
    grammar.rollback.assert_not_called()


def test_valid_drafts_after_reasoning_boundary_are_rolled_back():
    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.validate_tokens.side_effect = lambda tokens: tokens
    grammar.accept_tokens.return_value = True
    manager, request = make_manager_and_request(grammar)

    manager.grammar_bitmask(
        {"req": request},
        ["req"],
        {"req": [MARKER, VALID, VALID]},
    )

    assert grammar.validate_tokens.call_count == 2
    assert grammar.accept_tokens.call_count == 2
    grammar.rollback.assert_called_once_with(2)


def test_established_grammar_path_does_not_double_validate():
    grammar = Mock()
    grammar.is_terminated.return_value = False
    grammar.accept_tokens.return_value = True
    manager, request = make_manager_and_request(grammar)
    request.structured_output_request.reasoning_ended = True

    manager.grammar_bitmask(
        {"req": request},
        ["req"],
        {"req": [VALID]},
    )

    grammar.validate_tokens.assert_not_called()
    grammar.accept_tokens.assert_called_once_with("req", [VALID])
    grammar.rollback.assert_called_once_with(1)
