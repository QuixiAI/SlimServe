import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {}},
    },
}
SECOND_FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {"type": "object", "properties": {}},
    },
}
CUSTOM_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Apply a patch.",
}


@pytest.fixture(autouse=True)
def _glm53_tool_calling_profile(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TOOL_CALLING_PROFILE", "glm53")


def _chat_params(request):
    return request.build_chat_params(None, "auto").with_defaults(
        {"thinking": True, "enable_thinking": True}
    )


def test_glm53_native_tools_default_to_non_thinking() -> None:
    request = ChatCompletionRequest(
        model="GLM-5.3-Flash-FP8",
        messages=[],
        tools=[FUNCTION_TOOL],
        tool_choice="required",
    )
    kwargs = _chat_params(request).chat_template_kwargs
    assert kwargs["thinking"] is False
    assert kwargs["enable_thinking"] is False


def test_chat_params_preserve_and_render_tool_choice_metadata() -> None:
    request = ChatCompletionRequest(
        model="GLM-5.3-Flash-FP8",
        messages=[],
        tools=[FUNCTION_TOOL],
        tool_choice="required",
        chat_template_kwargs={"tool_choice": "none"},
    )
    params = _chat_params(request)

    assert params.tool_choice == "required"
    assert params.get_apply_chat_template_kwargs()["tool_choice"] == "required"


def test_responses_chat_params_render_allowed_tool_choice_metadata() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "GLM-5.3-Flash-FP8",
            "input": "Use the tool.",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "get_weather"}],
            },
        }
    )
    rendered_choice = _chat_params(request).get_apply_chat_template_kwargs()[
        "tool_choice"
    ]

    assert rendered_choice["type"] == "allowed_tools"
    assert rendered_choice["mode"] == "required"


def test_glm53_native_tools_allow_explicit_thinking() -> None:
    request = ChatCompletionRequest(
        model="GLM-5.3-Flash-FP8",
        messages=[],
        tools=[FUNCTION_TOOL],
        tool_choice="required",
        chat_template_kwargs={"thinking": True, "enable_thinking": True},
    )
    kwargs = _chat_params(request).chat_template_kwargs
    assert kwargs["thinking"] is True
    assert kwargs["enable_thinking"] is True


def test_glm53_custom_tools_force_non_thinking() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "GLM-5.3-Flash-FP8",
            "input": "Patch the file.",
            "tools": [CUSTOM_TOOL],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "chat_template_kwargs": {
                "thinking": True,
                "enable_thinking": True,
            },
        }
    )
    kwargs = _chat_params(request).chat_template_kwargs
    assert kwargs["thinking"] is False
    assert kwargs["enable_thinking"] is False


def test_profile_policy_does_not_depend_on_the_served_model_alias() -> None:
    request = ChatCompletionRequest(
        model="production-model-alias",
        messages=[],
        tools=[FUNCTION_TOOL],
        tool_choice="required",
    )
    kwargs = _chat_params(request).chat_template_kwargs
    assert kwargs["thinking"] is False
    assert kwargs["enable_thinking"] is False


def test_model_name_does_not_enable_compatibility_without_profile(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_TOOL_CALLING_PROFILE")
    request = ChatCompletionRequest(
        model="GLM-5.3-Flash-FP8",
        messages=[],
        tools=[FUNCTION_TOOL],
        tool_choice="required",
    )
    kwargs = _chat_params(request).chat_template_kwargs
    assert kwargs["thinking"] is True
    assert kwargs["enable_thinking"] is True


def test_glm53_parallel_single_native_tool_uses_json_array_constraint() -> (
    None
):
    request = ChatCompletionRequest(
        model="GLM-5.3-Flash-FP8",
        messages=[],
        tools=[FUNCTION_TOOL],
        tool_choice="required",
        parallel_tool_calls=True,
    )
    parser = object.__new__(Glm47MoeModelToolParser)
    # Flash greedily repeats even a sole eligible XML tool call when parallel
    # calling is left at its protocol default.  The whole-array JSON path
    # terminates naturally after the requested call.
    assert parser.get_structural_tag(request) is None


def test_glm53_unbounded_multi_tool_chat_uses_json_array_constraint() -> None:
    request = ChatCompletionRequest(
        model="GLM-5.3-Flash-FP8",
        messages=[],
        tools=[FUNCTION_TOOL, SECOND_FUNCTION_TOOL],
        tool_choice="required",
        parallel_tool_calls=True,
    )
    parser = object.__new__(Glm47MoeModelToolParser)

    # Returning None here deliberately selects ToolParser.adjust_request's
    # whole-array JSON schema, which can terminate without an artificial call
    # limit and is decoded by the GLM-5.3 JSON-array fallback parser.
    assert parser.get_structural_tag(request) is None


def test_glm53_responses_max_tool_calls_does_not_bound_functions() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "GLM-5.3-Flash-FP8",
            "input": "Use both tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "max_tool_calls": 2,
        }
    )
    parser = object.__new__(Glm47MoeModelToolParser)

    # Responses max_tool_calls applies to processed built-ins, not generated
    # function calls, so it must not select a bounded native grammar.
    assert parser.get_structural_tag(request) is None


def test_glm53_unbounded_multi_tool_responses_uses_json_array_constraint() -> (
    None
):
    request = ResponsesRequest.model_validate(
        {
            "model": "GLM-5.3-Flash-FP8",
            "input": "Use both tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            "tool_choice": "required",
            "parallel_tool_calls": True,
        }
    )
    parser = object.__new__(Glm47MoeModelToolParser)

    assert parser.get_structural_tag(request) is None


def test_glm53_responses_required_defaults_use_json_array() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "GLM-5.3-Flash-FP8",
            "input": "Use the tool.",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": "required",
        }
    )
    parser = object.__new__(Glm47MoeModelToolParser)

    assert request.parallel_tool_calls is True
    assert request.max_tool_calls is None
    # The repeatable XML sequence makes Flash emit the correct call until the
    # token cap. The whole-array JSON path naturally closes and reaches EOS.
    assert parser.get_structural_tag(request) is None


def test_glm53_single_native_tool_keeps_structural_constraint() -> None:
    request = ChatCompletionRequest(
        model="GLM-5.3-Flash-FP8",
        messages=[],
        tools=[FUNCTION_TOOL],
        tool_choice="required",
        parallel_tool_calls=False,
    )
    parser = object.__new__(Glm47MoeModelToolParser)
    assert parser.get_structural_tag(request) is not None


def test_glm53_custom_tool_keeps_structural_constraint() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "GLM-5.3-Flash-FP8",
            "input": "Patch the file.",
            "tools": [CUSTOM_TOOL],
            "tool_choice": "required",
            "parallel_tool_calls": True,
        }
    )
    parser = object.__new__(Glm47MoeModelToolParser)
    assert parser.get_structural_tag(request) is not None
