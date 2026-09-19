# SPDX-License-Identifier: Apache-2.0
"""Responses uses the same operator/request budget rules as Chat Completions."""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


@pytest.mark.parametrize(
    "kwargs,defaults,effort,expected",
    [
        ({}, {}, None, None),
        ({}, {"thinking_token_budget": 40}, None, 40),
        ({"thinking_token_budget": 7}, {"thinking_token_budget": 40}, None, 7),
        ({"thinking_token_budget": -1}, {"thinking_token_budget": 40}, None, None),
        ({"thinking_token_budget": 0}, {"thinking_token_budget": 40}, None, 0),
        ({"thinking_token_budget": None}, {"thinking_token_budget": 40}, None, 40),
        ({"reasoning": {"effort": "none"}}, {"thinking_token_budget": 40}, None, None),
        ({}, {"thinking_token_budget": {"low": 8, "medium": 20}}, "low", 8),
        (
            {"reasoning": {"effort": "medium"}},
            {"thinking_token_budget": {"low": 8, "medium": 20}},
            "low",
            20,
        ),
        ({"max_output_tokens": 4}, {"thinking_token_budget": 40}, None, 3),
    ],
)
def test_responses_thinking_budget(kwargs, defaults, effort, expected):
    request = ResponsesRequest(input="Summarize", **kwargs)
    params = request.to_sampling_params(64, defaults, default_reasoning_effort=effort)
    assert params.thinking_token_budget == expected


@pytest.mark.parametrize("budget", [True, -2, 1.5])
def test_invalid_request_budget_is_rejected(budget):
    with pytest.raises((ValidationError, ValueError)):
        ResponsesRequest(input="Summarize", thinking_token_budget=budget)
