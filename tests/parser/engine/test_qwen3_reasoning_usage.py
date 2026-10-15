# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning usage follows Qwen's initial/implicit reasoning transitions."""

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.qwen3 import Qwen3Parser

VOCAB = {"<think>": 200, "</think>": 201, "<tool_call>": 202, "</tool_call>": 203}


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], 0),
        ([1, 2, 3, 201, 4, 5], 3),  # Opening <think> is in the prompt.
        ([200, 1, 2, 3, 201, 4, 5], 3),  # Explicit opener is not a nested span.
        ([200, 200, 1, 201, 4], 1),  # Repeated opener is absorbed by the parser.
        ([1, 2, 3], 3),  # Budget exhausted while still reasoning.
        ([1, 2, 202, 4, 5, 203, 6], 2),  # Tool opener implicitly ends thinking.
        ([200, 1, 202, 4, 200, 5, 201], 1),  # Markers in tool args stay tool args.
        ([201, 201, 4], 0),  # Duplicate closing markers do not start reasoning.
        ([202, 4, 5], 0),
        ([1, 201, 4, 200, 5], 1),  # Content markers do not reopen Qwen reasoning.
    ],
)
def test_qwen_reasoning_usage(token_ids, expected):
    parser = Qwen3Parser(make_mock_tokenizer(VOCAB))
    assert parser.count_reasoning_tokens(token_ids) == expected


def test_disabled_thinking_does_not_count_content_as_reasoning():
    parser = Qwen3Parser(
        make_mock_tokenizer(VOCAB), chat_template_kwargs={"enable_thinking": False}
    )
    assert parser.count_reasoning_tokens([200, 1, 2, 201, 3]) == 0


def test_legacy_reasoning_adapter_preserves_initial_span_usage():
    reasoning_adapter, _ = make_adapters(Qwen3Parser)
    parser = reasoning_adapter(make_mock_tokenizer(VOCAB))
    assert parser.count_reasoning_tokens([1, 2, 202, 3]) == 2
