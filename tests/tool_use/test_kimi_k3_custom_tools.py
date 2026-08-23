# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import ResponseCustomToolCall
from xgrammar import Grammar

from vllm.entrypoints.chat_utils import (
    adapt_custom_tool_calls_for_chat_template,
    adapt_custom_tools_for_chat_template,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import (
    construct_chat_messages_with_tool_call,
    construct_tool_dicts,
)
from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
from vllm.transformers_utils.tokenizers.encoding_k3 import build_chat_segments


class _DummyTokenizer:
    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def _request(*, tool_choice="required", syntax="lark") -> ResponsesRequest:
    definition = (
        'start: INT " + " INT\n%import common.INT'
        if syntax == "lark"
        else r"[0-9]+ \+ [0-9]+"
    )
    return ResponsesRequest.model_validate(
        {
            "model": "kimi-k3",
            "input": "Use the math tool.",
            "tools": [
                {
                    "type": "custom",
                    "name": "math_exp",
                    "description": "Build an arithmetic expression.",
                    "format": {
                        "type": "grammar",
                        "syntax": syntax,
                        "definition": definition,
                    },
                }
            ],
            "tool_choice": tool_choice,
        }
    )


def _xtml_call(raw_input: str) -> str:
    return (
        "<|open|>tools<|sep|>"
        '<|open|>call tool="math_exp" index="1"<|sep|>'
        '<|open|>argument key="input" type="string"<|sep|>'
        f"{raw_input}"
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
    )


def _render_segments(segments) -> str:
    return "".join(segment.text for segment in segments)


def test_kimi_k3_lark_structural_tag_constrains_only_raw_input():
    request = _request()

    structural_tag = get_model_structural_tag(
        "kimi_k3", request.tools, request.tool_choice, reasoning=False
    )

    assert structural_tag is not None
    dumped = structural_tag.model_dump_json()
    assert '<|open|>call tool=\\"math_exp\\" index=\\"' in dumped
    assert '<|open|>argument key=\\"input\\" type=\\"string\\"<|sep|>' in dumped
    assert "root ::= start" in dumped
    Grammar.from_structural_tag(structural_tag)


def test_kimi_k3_regex_structural_tag_compiles():
    request = _request(syntax="regex")

    structural_tag = get_model_structural_tag(
        "kimi_k3", request.tools, request.tool_choice, reasoning=False
    )

    assert structural_tag is not None
    assert r"[0-9]+ \\+ [0-9]+" in structural_tag.model_dump_json()
    Grammar.from_structural_tag(structural_tag)


def test_kimi_k3_non_streaming_custom_call_returns_raw_input():
    request = _request()
    parser = KimiK3ToolParser(_DummyTokenizer(), request.tools)

    result = parser.extract_tool_calls(_xtml_call("4 + 4"), request)

    assert result.tools_called
    assert result.tool_calls[0].function.name == "math_exp"
    assert result.tool_calls[0].function.arguments == "4 + 4"


def test_kimi_k3_streaming_custom_call_emits_raw_input_when_call_closes():
    request = _request()
    parser = KimiK3ToolParser(_DummyTokenizer(), request.tools)
    output = _xtml_call("4 + 4")

    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=output,
        delta_text=output,
        previous_token_ids=[],
        current_token_ids=[1],
        delta_token_ids=[1],
        request=request,
    )

    assert delta is not None
    assert delta.tool_calls is not None
    assert delta.tool_calls[0].function.name == "math_exp"
    assert delta.tool_calls[0].function.arguments == "4 + 4"


def test_kimi_k3_custom_call_continuation_renders_as_xtml():
    request = _request()
    messages = construct_chat_messages_with_tool_call(
        [
            ResponseCustomToolCall(
                type="custom_tool_call",
                id="item_1",
                call_id="call_1",
                name="math_exp",
                input="4 + 4",
            ),
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "8",
            },
        ]
    )
    tools = construct_tool_dicts(request.tools, request.tool_choice)
    adapted_messages = adapt_custom_tool_calls_for_chat_template(messages)
    adapted_tools = adapt_custom_tools_for_chat_template(tools)

    rendered = _render_segments(
        build_chat_segments(
            adapted_messages,
            adapted_tools,
            add_generation_prompt=False,
            thinking=False,
        )
    )

    assert _xtml_call("4 + 4") in rendered
    assert (
        '<|open|>message role="tool" tool="math_exp" index="1"<|sep|>'
        "8<|close|>message<|sep|>"
    ) in rendered


def test_kimi_k3_named_custom_choice_builds_forced_xtml_tag():
    request = _request(tool_choice={"type": "custom", "name": "math_exp"})

    structural_tag = get_model_structural_tag(
        "kimi_k3", request.tools, request.tool_choice, reasoning=False
    )

    assert structural_tag is not None
    assert 'tool=\\"math_exp\\"' in structural_tag.model_dump_json()
    Grammar.from_structural_tag(structural_tag)
