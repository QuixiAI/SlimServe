# SPDX-License-Identifier: Apache-2.0
"""Thinking limits are deployment defaults; explicit API requests take priority."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config.model import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

EFFORTS = [None, "none", "minimal", "low", "medium", "high", "xhigh", "max"]


def sampling_params(protocol, defaults, effort=None, **kwargs):
    if protocol == "chat":
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Summarize."}],
            reasoning_effort=effort,
            **kwargs,
        )
    else:
        request = ResponsesRequest(
            model="test",
            input="Summarize.",
            reasoning=None if effort is None else {"effort": effort},
            **kwargs,
        )
    return request.to_sampling_params(4096, defaults)


@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize("configured", [False, True])
def test_omitted_budget_inherits_only_the_deployments_default(
    protocol, effort, configured
):
    defaults = {"thinking_token_budget": 2000} if configured else {}
    expected = 0 if effort == "none" else 2000 if configured else None
    assert sampling_params(protocol, defaults, effort).thinking_token_budget == expected


@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("effort", [None, "none", "high"])
@pytest.mark.parametrize(
    "override,expected", [(0, 0), (64, 64), (None, None), (-1, None)]
)
def test_explicit_budget_overrides_profile_and_reasoning_effort(
    protocol, effort, override, expected
):
    params = sampling_params(
        protocol,
        {"thinking_token_budget": 2000},
        effort,
        thinking_token_budget=override,
    )
    assert params.thinking_token_budget == expected


@pytest.mark.parametrize("source", ["vllm", "auto"])
@pytest.mark.parametrize(
    "override", [{}, {"thinking_token_budget": 2000}, {"thinking_token_budget": 0}]
)
def test_model_config_forwards_profile_budget_without_synthesizing_one(
    source, override
):
    model_config = SimpleNamespace(
        generation_config=source,
        override_generation_config=override,
        try_get_generation_config=Mock(return_value={}),
    )
    assert ModelConfig.get_diff_sampling_param(model_config) == override
    if source == "vllm":
        model_config.try_get_generation_config.assert_not_called()
    else:
        model_config.try_get_generation_config.assert_called_once_with()


def test_profile_budget_overrides_model_generation_budget_without_mutating_override():
    overrides = {"thinking_token_budget": 2000}
    model_config = SimpleNamespace(
        generation_config="auto",
        override_generation_config=overrides,
        try_get_generation_config=Mock(
            return_value={"thinking_token_budget": 6000, "top_p": 0.9}
        ),
    )
    assert ModelConfig.get_diff_sampling_param(model_config) == {
        "thinking_token_budget": 2000,
        "top_p": 0.9,
    }
    assert overrides == {"thinking_token_budget": 2000}
