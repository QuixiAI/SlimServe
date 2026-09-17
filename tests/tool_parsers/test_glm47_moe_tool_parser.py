# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the GLM-4.7 tool call parser."""

import json
import os
from unittest.mock import Mock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import build_response_output_items
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import Glm47MoeParserReasoningAdapter
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.7"


@pytest.fixture
def glm53_tool_profile(monkeypatch):
    monkeypatch.setenv("VLLM_TOOL_CALLING_PROFILE", "glm53")


class GlmDelegatingParser(DelegatingParser):
    tool_parser_cls = Glm47MoeModelToolParser


class GlmCombinedDelegatingParser(DelegatingParser):
    reasoning_parser_cls = Glm47MoeParserReasoningAdapter
    tool_parser_cls = Glm47MoeModelToolParser


@pytest.fixture(scope="module")
def glm47_tokenizer():
    tokenizer_name = os.environ.get("VLLM_TEST_GLM_TOKENIZER", MODEL)
    return get_tokenizer(tokenizer_name=tokenizer_name)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            ),
        ),
    ]


@pytest.fixture
def glm47_tool_parser(glm47_tokenizer, sample_tools):
    return Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    request.tool_choice = "auto"
    return request


@pytest.fixture
def namespace_tool_request() -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__computer_use",
                    "description": "Computer use tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_app_state",
                            "description": "Get app state.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "app": {"type": "string"},
                                },
                            },
                        }
                    ],
                }
            ],
        }
    )


class TestGlm47ExtractToolCalls:
    def test_responses_allowed_required_native_xml_is_a_function_call(
        self, glm47_tokenizer
    ):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Call get_time for Tokyo.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "parameters": {"type": "object"},
                    },
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{"type": "function", "name": "get_time"}],
                },
            }
        )
        parser = GlmDelegatingParser(glm47_tokenizer, tools=request.tools)
        out = (
            "<tool_call>get_time"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
            "</tool_call>"
        )

        reasoning, content, tool_calls = parser.parse(
            out, request=request, enable_auto_tools=True
        )

        assert reasoning is None
        assert content is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_time"
        assert json.loads(tool_calls[0].arguments) == {"city": "Tokyo"}

    def test_responses_allowed_required_native_xml_streams_as_function_call(
        self, glm47_tokenizer
    ):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Call get_time for Tokyo.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_time",
                        "parameters": {"type": "object"},
                    }
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{"type": "function", "name": "get_time"}],
                },
            }
        )
        parser = GlmDelegatingParser(glm47_tokenizer, tools=request.tools)
        chunks = [
            "<tool_call>",
            "get_time",
            "<arg_key>city</arg_key>",
            "<arg_value>Tokyo</arg_value>",
            "</tool_call>",
        ]
        deltas = []
        prompt_token_ids = []
        for chunk in chunks:
            delta = parser.parse_delta(
                chunk,
                [],
                request,
                prompt_token_ids=prompt_token_ids,
                finished=False,
            )
            prompt_token_ids = None
            if delta is not None:
                deltas.append(delta)

        calls = [call for delta in deltas for call in (delta.tool_calls or [])]
        names = [call.function.name for call in calls if call.function.name]
        arguments = "".join(
            call.function.arguments or "" for call in calls if call.function
        )
        assert names == ["get_time"]
        assert json.loads(arguments) == {"city": "Tokyo"}

    def test_responses_allowed_tools_rejects_generated_call_outside_subset(
        self, glm47_tokenizer
    ):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Call get_time for Tokyo.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "parameters": {"type": "object"},
                    },
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{"type": "function", "name": "get_time"}],
                },
            }
        )
        parser = GlmDelegatingParser(glm47_tokenizer, tools=request.tools)
        disallowed = "<tool_call>get_weather</tool_call>"

        with pytest.raises(ValueError, match="outside the allowed_tools subset"):
            parser.parse(disallowed, request=request, enable_auto_tools=True)

    def test_namespace_tool_call_round_trip_to_responses_output(
        self, glm47_tokenizer, namespace_tool_request
    ):
        parser = Glm47MoeModelToolParser(
            glm47_tokenizer, tools=namespace_tool_request.tools
        )
        out = (
            "<tool_call>mcp__computer_use__get_app_state"
            "<arg_key>app</arg_key>"
            "<arg_value>Google Chrome</arg_value>"
            "</tool_call>"
        )

        result = parser.extract_tool_calls(out, request=namespace_tool_request)

        assert result.tools_called
        tool_call = result.tool_calls[0].function
        assert tool_call == FunctionCall(
            name="mcp__computer_use__get_app_state",
            arguments='{"app": "Google Chrome"}',
        )

        output_items = build_response_output_items(
            reasoning=None,
            content=None,
            tool_calls=[tool_call],
            tools=namespace_tool_request.tools,
        )
        output_tool_call = output_items[0]
        assert isinstance(output_tool_call, ResponseFunctionToolCall)
        assert output_tool_call.name == "get_app_state"
        assert output_tool_call.namespace == "mcp__computer_use"

    def test_no_tool_call(self, glm47_tool_parser, mock_request):
        out = "This is a plain response."
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert not r.tools_called
        assert r.content == out

    def test_zero_arg_inline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_args_with_newlines(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_whitespace_preserved_in_arg_values(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "  Beijing  "}

    def test_content_before(self, glm47_tool_parser, mock_request):
        out = "Checking.<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.content == "Checking."

    def test_multiple(self, glm47_tool_parser, mock_request):
        out = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>"
        )
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert len(r.tool_calls) == 2

    @pytest.mark.usefixtures("glm53_tool_profile")
    @pytest.mark.parametrize(
        "model",
        ["GLM-5.3-FP8", "GLM-5.3-Flash-FP8"],
    )
    def test_glm53_parallel_json_array_fallback(
        self, glm47_tokenizer, sample_tools, model
    ):
        request = ChatCompletionRequest(
            model=model,
            messages=[],
            tools=sample_tools,
            tool_choice="required",
            parallel_tool_calls=True,
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)
        out = json.dumps(
            [
                {"name": "get_weather", "parameters": {"city": "Reykjavik"}},
                {"name": "get_current_date", "parameters": {}},
            ]
        )

        result = parser.extract_tool_calls(out, request=request)

        assert result.tools_called
        assert result.content is None
        assert [call.function.name for call in result.tool_calls] == [
            "get_weather",
            "get_current_date",
        ]
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "city": "Reykjavik"
        }
        assert json.loads(result.tool_calls[1].function.arguments) == {}

    @pytest.mark.parametrize(
        "model,out",
        [
            (
                "GLM-5.3-FP8",
                "[]",
            ),
            (
                "GLM-5.3-Flash-FP8",
                '[{"name":"undeclared","parameters":{"city":"Reykjavik"}}]',
            ),
            (
                "GLM-5.3-Flash-FP8",
                '[{"name":"get_weather","parameters":"not-an-object"}]',
            ),
        ],
    )
    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_glm53_json_array_fallback_rejects_invalid_calls(
        self, glm47_tokenizer, sample_tools, model, out
    ):
        request = ChatCompletionRequest(
            model=model,
            messages=[],
            tools=sample_tools,
            tool_choice="required",
            parallel_tool_calls=True,
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)

        result = parser.extract_tool_calls(out, request=request)

        assert not result.tools_called
        assert result.content == out

    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_glm53_parallel_json_array_streaming_fallback(
        self, glm47_tokenizer, sample_tools
    ):
        request = ChatCompletionRequest(
            model="GLM-5.3-Flash-FP8",
            messages=[],
            tools=sample_tools,
            tool_choice="required",
            parallel_tool_calls=True,
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)
        chunks = [
            "[",
            '{"name":"get_weather","parameters":{"city":"Reykjavik"}},',
            '{"name":"get_current_date","parameters":{}}',
            "]",
        ]
        current_text = ""
        deltas = []
        for chunk in chunks:
            previous_text = current_text
            current_text += chunk
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )
            if delta:
                deltas.append(delta)

        assert parser.finish_streaming() is None
        assert len(deltas) == 1
        assert [call.function.name for call in deltas[0].tool_calls] == [
            "get_weather",
            "get_current_date",
        ]
        assert json.loads(deltas[0].tool_calls[0].function.arguments) == {
            "city": "Reykjavik"
        }

    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_glm53_completed_json_array_is_not_reemitted_on_empty_finish(
        self, glm47_tokenizer, sample_tools
    ):
        """The engine's empty finish chunk must not replay completed calls."""
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Call the tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    }
                    for tool in sample_tools
                ],
                "tool_choice": "required",
                "parallel_tool_calls": True,
                "stream": True,
            }
        )
        parser = GlmCombinedDelegatingParser(
            glm47_tokenizer,
            tools=request.tools,
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        )
        output = json.dumps(
            [
                {"name": "get_weather", "parameters": {"city": "Reykjavik"}},
                {"name": "get_current_date", "parameters": {}},
            ]
        )

        deltas = [
            parser.parse_delta(
                output,
                glm47_tokenizer.encode(output, add_special_tokens=False),
                request,
                prompt_token_ids=[],
                finished=False,
            ),
            parser.parse_delta(
                "",
                [],
                request,
                finished=True,
            ),
        ]
        calls = [
            call
            for delta in deltas
            if delta is not None
            for call in (delta.tool_calls or [])
        ]

        assert [call.function.name for call in calls] == [
            "get_weather",
            "get_current_date",
        ]
        assert json.loads(calls[0].function.arguments) == {"city": "Reykjavik"}
        assert json.loads(calls[1].function.arguments) == {}

    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_glm53_invalid_json_array_streams_as_content_at_finish(
        self, glm47_tokenizer, sample_tools
    ):
        request = ChatCompletionRequest(
            model="GLM-5.3-Flash-FP8",
            messages=[],
            tools=sample_tools,
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)
        out = '[{"name":"undeclared","parameters":{}}]'

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=out,
            delta_text=out,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        assert delta is None
        finish = parser.finish_streaming()
        assert finish is not None
        assert finish.content == out
        assert not finish.tool_calls

    def test_custom_tool_returns_raw_input(self, glm47_tokenizer):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash",
                "input": "Patch the file.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": 'start: "*** Begin Patch"',
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)
        raw_input = "*** Begin Patch"
        out = (
            "<tool_call>apply_patch"
            f"<arg_key>input</arg_key><arg_value>{raw_input}</arg_value>"
            "</tool_call>"
        )

        result = parser.extract_tool_calls(out, request=request)

        assert result.tools_called
        assert result.tool_calls[0].function.name == "apply_patch"
        assert result.tool_calls[0].function.arguments == raw_input

    @pytest.mark.parametrize(
        "raw_input",
        [
            "*** Begin Patch",
            "café.txt",
            "文件.txt",
        ],
    )
    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_glm53_recovers_grammar_valid_raw_custom_input(
        self, glm47_tokenizer, raw_input
    ):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Use the custom tool.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": (
                                f"start: {json.dumps(raw_input, ensure_ascii=False)}"
                            ),
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)

        result = parser.extract_tool_calls(raw_input, request=request)

        assert result.tools_called
        assert result.content is None
        assert result.tool_calls[0].function.name == "apply_patch"
        assert result.tool_calls[0].function.arguments == raw_input

    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_glm53_recovers_regex_valid_raw_custom_input(self, glm47_tokenizer):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Use the digits tool.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "digits",
                        "format": {
                            "type": "grammar",
                            "syntax": "regex",
                            "definition": "[0-9]+",
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)

        result = parser.extract_tool_calls("12053", request=request)

        assert result.tools_called
        assert result.tool_calls[0].function.name == "digits"
        assert result.tool_calls[0].function.arguments == "12053"

    @pytest.mark.parametrize(
        ("model", "tool_choice", "extra_tool"),
        [
            ("GLM-5.3-Flash-FP8", "none", False),
            ("GLM-5.3-Flash-FP8", "auto", True),
        ],
    )
    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_raw_custom_recovery_is_fail_closed(
        self, glm47_tokenizer, model, tool_choice, extra_tool
    ):
        raw_input = "*** Begin Patch"
        tools = [
            {
                "type": "custom",
                "name": "apply_patch",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": 'start: "*** Begin Patch"',
                },
            }
        ]
        if extra_tool:
            tools.append(
                {
                    "type": "function",
                    "name": "other_tool",
                    "parameters": {"type": "object"},
                }
            )
        request = ResponsesRequest.model_validate(
            {
                "model": model,
                "input": "Use the custom tool.",
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)

        result = parser.extract_tool_calls(raw_input, request=request)

        assert not result.tools_called
        assert result.content == raw_input

    def test_raw_custom_recovery_honors_allowed_tools_subset(self, glm47_tokenizer):
        raw_input = "*** Begin Patch"
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Use apply_patch.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": 'start: "*** Begin Patch"',
                        },
                    },
                    {
                        "type": "function",
                        "name": "other_tool",
                        "parameters": {"type": "object"},
                    },
                ],
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{"type": "custom", "name": "apply_patch"}],
                },
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)

        result = parser.extract_tool_calls(raw_input, request=request)

        assert result.tools_called
        assert result.tool_calls[0].function.name == "apply_patch"
        assert result.tool_calls[0].function.arguments == raw_input

    def test_empty_content_none(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None

    def test_whitespace_content_none(self, glm47_tool_parser, mock_request):
        out = "  \n  <tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None


def _reset(parser):
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._sent_content_idx = 0


class TestGlm47Streaming:
    def test_responses_stream_does_not_duplicate_completed_function_arguments(
        self, glm47_tokenizer
    ):
        """A closing/final chunk must not replay already streamed JSON args."""
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Read the nearby lines.",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_nearby",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "line_end": {"type": "integer"},
                                "line_start": {"type": "integer"},
                            },
                            "required": ["line_end", "line_start"],
                            "additionalProperties": False,
                        },
                    }
                ],
                "tool_choice": "required",
                "stream": True,
            }
        )
        parser = GlmCombinedDelegatingParser(
            glm47_tokenizer,
            tools=request.tools,
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        )
        output = (
            "<tool_call>read_nearby"
            "<arg_key>line_end</arg_key><arg_value>43104</arg_value>"
            "<arg_key>line_start</arg_key><arg_value>42890</arg_value>"
            "</tool_call>"
        )
        token_ids = glm47_tokenizer.encode(output, add_special_tokens=False)
        deltas = []
        for index, token_id in enumerate(token_ids):
            delta = parser.parse_delta(
                glm47_tokenizer.decode([token_id]),
                [token_id],
                request,
                prompt_token_ids=[] if index == 0 else None,
                finished=index == len(token_ids) - 1,
            )
            if delta is not None:
                deltas.append(delta)

        calls = [call for delta in deltas for call in (delta.tool_calls or [])]
        assert [call.function.name for call in calls if call.function.name] == [
            "read_nearby"
        ]
        arguments = "".join(call.function.arguments or "" for call in calls)
        assert json.loads(arguments) == {"line_end": 43104, "line_start": 42890}
        assert arguments.count("{") == 1
        assert arguments.count("}") == 1

    def test_responses_single_final_chunk_does_not_replay_function_arguments(
        self, glm47_tokenizer
    ):
        """A whole response delivered with finish_reason must be emitted once."""
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Read the nearby lines.",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_nearby",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "line_end": {"type": "integer"},
                                "line_start": {"type": "integer"},
                            },
                            "required": ["line_end", "line_start"],
                            "additionalProperties": False,
                        },
                    }
                ],
                "tool_choice": "required",
                "stream": True,
            }
        )
        parser = GlmCombinedDelegatingParser(
            glm47_tokenizer,
            tools=request.tools,
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        )
        output = (
            "<tool_call>read_nearby"
            "<arg_key>line_end</arg_key><arg_value>43104</arg_value>"
            "<arg_key>line_start</arg_key><arg_value>42890</arg_value>"
            "</tool_call>"
        )
        delta = parser.parse_delta(
            output,
            glm47_tokenizer.encode(output, add_special_tokens=False),
            request,
            prompt_token_ids=[],
            finished=True,
        )

        assert delta is not None
        calls = delta.tool_calls or []
        assert [call.function.name for call in calls if call.function.name] == [
            "read_nearby"
        ]
        arguments = "".join(call.function.arguments or "" for call in calls)
        assert json.loads(arguments) == {"line_end": 43104, "line_start": 42890}

    def test_responses_empty_finish_chunk_does_not_replay_function_arguments(
        self, glm47_tokenizer
    ):
        """An empty finish notification must not replay a completed call."""
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Read the nearby lines.",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_nearby",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "line_end": {"type": "integer"},
                                "line_start": {"type": "integer"},
                            },
                            "required": ["line_end", "line_start"],
                            "additionalProperties": False,
                        },
                    }
                ],
                "tool_choice": "required",
                "stream": True,
            }
        )
        parser = GlmCombinedDelegatingParser(
            glm47_tokenizer,
            tools=request.tools,
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        )
        output = (
            "<tool_call>read_nearby"
            "<arg_key>line_end</arg_key><arg_value>43104</arg_value>"
            "<arg_key>line_start</arg_key><arg_value>42890</arg_value>"
            "</tool_call>"
        )
        deltas = [
            parser.parse_delta(
                output,
                glm47_tokenizer.encode(output, add_special_tokens=False),
                request,
                prompt_token_ids=[],
                finished=False,
            ),
            parser.parse_delta(
                "",
                [],
                request,
                finished=True,
            ),
        ]

        calls = [
            call
            for delta in deltas
            if delta is not None
            for call in (delta.tool_calls or [])
        ]
        assert [call.function.name for call in calls if call.function.name] == [
            "read_nearby"
        ]
        arguments = "".join(call.function.arguments or "" for call in calls)
        assert json.loads(arguments) == {"line_end": 43104, "line_start": 42890}

    def test_no_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = ["<tool_call>", "get_current_date", "</tool_call>"]
        current_text = ""
        deltas = []
        for chunk in chunks:
            current_text += chunk
            delta = glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if delta:
                deltas.append(delta)
        tool_calls = [
            tool_call for delta in deltas for tool_call in (delta.tool_calls or [])
        ]
        names = [
            tool_call.function.name
            for tool_call in tool_calls
            if tool_call.function and tool_call.function.name
        ]
        arguments = [
            tool_call.function.arguments
            for tool_call in tool_calls
            if tool_call.function and tool_call.function.arguments
        ]
        assert names == ["get_current_date"]
        assert "".join(arguments) == "{}"

    def test_with_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Beijing",
            "</arg_value>",
            "</tool_call>",
        ]
        current_text = ""
        deltas = []
        for chunk in chunks:
            current_text += chunk
            delta = glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
            if delta:
                deltas.append(delta)
        arguments = [
            tool_call.function.arguments
            for delta in deltas
            for tool_call in (delta.tool_calls or [])
            if tool_call.function and tool_call.function.arguments
        ]
        args = json.loads("".join(arguments))
        assert args["city"] == "Beijing"

    @pytest.mark.usefixtures("glm53_tool_profile")
    def test_glm53_raw_custom_input_streams_as_call_at_finish(
        self, glm47_tokenizer
    ):
        raw_input = "*** Begin Patch"
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Patch the file.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": 'start: "*** Begin Patch"',
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)
        current_text = ""

        for chunk in ("*** ", "Begin ", "Patch"):
            previous_text = current_text
            current_text += chunk
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )
            assert delta is None

        finish = parser.finish_streaming()
        assert finish is not None
        assert finish.content is None
        assert len(finish.tool_calls) == 1
        assert finish.tool_calls[0].function.name == "apply_patch"
        assert finish.tool_calls[0].function.arguments == raw_input

    def test_delegating_parser_buffers_delta_only_raw_custom_stream(
        self, glm47_tokenizer
    ):
        raw_input = "*** Begin Patch"
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Patch the file.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": 'start: "*** Begin Patch"',
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
        parser = GlmCombinedDelegatingParser(
            glm47_tokenizer,
            tools=request.tools,
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        )
        deltas = []
        chunks = ("*** ", "Begin ", "Patch")
        for index, chunk in enumerate(chunks):
            delta = parser.parse_delta(
                chunk,
                [],
                request,
                prompt_token_ids=[] if index == 0 else None,
                finished=index == len(chunks) - 1,
            )
            if delta is not None:
                deltas.append(delta)

        assert len(deltas) == 1
        assert deltas[0].content is None
        assert len(deltas[0].tool_calls) == 1
        assert deltas[0].tool_calls[0].function.name == "apply_patch"
        assert deltas[0].tool_calls[0].function.arguments == raw_input

    def test_delegating_parser_custom_native_xml_with_ordinary_token_ids(
        self, glm47_tokenizer
    ):
        raw_input = "*** Begin Patch"
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Patch the file.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": 'start: "*** Begin Patch"',
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
        parser = GlmCombinedDelegatingParser(
            glm47_tokenizer,
            tools=request.tools,
            chat_template_kwargs={"thinking": False, "enable_thinking": False},
        )
        ordinary_token_id = glm47_tokenizer.encode(
            "x", add_special_tokens=False
        )[0]
        chunks = (
            "<tool_call>apply_patch",
            "<arg_key>input",
            "</arg_key><arg_value>",
            raw_input,
            "</arg_value>",
            "</tool_call>",
        )
        deltas = []
        for index, chunk in enumerate(chunks):
            delta = parser.parse_delta(
                chunk,
                [ordinary_token_id],
                request,
                prompt_token_ids=[] if index == 0 else None,
                finished=index == len(chunks) - 1,
            )
            if delta is not None:
                deltas.append(delta)

        assert all(delta.content is None for delta in deltas)
        calls = [call for delta in deltas for call in (delta.tool_calls or [])]
        assert [call.function.name for call in calls if call.function.name] == [
            "apply_patch"
        ]
        assert "".join(call.function.arguments or "" for call in calls) == raw_input

    def test_incomplete_raw_custom_input_streams_as_content_at_finish(
        self, glm47_tokenizer
    ):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Patch the file.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": 'start: "*** Begin Patch"',
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)
        partial = "*** Begin"

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=partial,
            delta_text=partial,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )

        assert delta is None
        finish = parser.finish_streaming()
        assert finish is not None
        assert finish.content == partial
        assert not finish.tool_calls

    def test_native_custom_envelope_still_streams_normally(self, glm47_tokenizer):
        request = ResponsesRequest.model_validate(
            {
                "model": "GLM-5.3-Flash-FP8",
                "input": "Patch the file.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": 'start: "*** Begin Patch"',
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
        parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=request.tools)
        chunks = [
            "<tool_",
            "call>apply_patch",
            "<arg_key>input</arg_key><arg_value>",
            "*** Begin Patch",
            "</arg_value></tool_call>",
        ]
        current_text = ""
        deltas = []
        for chunk in chunks:
            previous_text = current_text
            current_text += chunk
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )
            if delta is not None:
                deltas.append(delta)

        finish = parser.finish_streaming()
        if finish is not None:
            deltas.append(finish)
        calls = [call for delta in deltas for call in (delta.tool_calls or [])]
        assert [call.function.name for call in calls if call.function.name] == [
            "apply_patch"
        ]
        assert (
            "".join(call.function.arguments or "" for call in calls)
            == "*** Begin Patch"
        )
