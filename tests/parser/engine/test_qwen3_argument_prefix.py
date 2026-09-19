# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen XML argument streams must agree with final JSON reconstruction."""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.qwen3 import Qwen3Parser


def _parse_arguments(body, schema, chunk_size):
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {"value": schema, "tail": {"type": "string"}},
            },
        },
    )
    request = ChatCompletionRequest(
        model="test", messages=[], tools=[tool], tool_choice="required"
    )

    def parser():
        return Qwen3Parser(
            make_mock_tokenizer(
                {
                    "<think>": 200,
                    "</think>": 201,
                    "<tool_call>": 202,
                    "</tool_call>": 203,
                }
            ),
            tools=[tool],
            chat_template_kwargs={"enable_thinking": False},
        )

    text = f"<tool_call>\n<function=f>\n{body}\n</function>\n</tool_call>"
    stream_parser = parser()
    fragments = []
    for start in range(0, len(text), chunk_size):
        delta = stream_parser.parse_delta(
            text[start : start + chunk_size],
            [],
            request,
            finished=start + chunk_size >= len(text),
        )
        if delta:
            fragments.extend(
                call.function.arguments
                for call in delta.tool_calls
                if call.function and call.function.arguments
            )
    final = parser().extract_tool_calls(text, request).tool_calls[0].function.arguments
    return "".join(fragments), final, text


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
@pytest.mark.parametrize(
    ("value", "schema"),
    [
        ("", {"type": "string"}),
        ("  hello \t", {"type": "string"}),
        ("\n\nhello\n\n", {"type": "string"}),
        ('def f():\n    return "hi"\n', {"type": "string"}),
        ('a\\"b\\nc', {"type": "string"}),
        ("snowman ☃\r\nnext", {"type": "string"}),
        ("4e2", {"type": "number"}),
        ("42", {"type": "integer"}),
        ("true", {"type": "boolean"}),
        ("null", {"type": ["string", "null"]}),
        ('{"a":[1,2]}', {"type": "object"}),
        ("[1,2]", {"type": "array"}),
    ],
)
def test_ordinary_arguments_match_final_json(value, schema, chunk_size):
    body = f"<parameter=value>{value}</parameter>\n<parameter=tail>ok</parameter>"
    streamed, final, _ = _parse_arguments(body, schema, chunk_size)
    assert streamed == final
    assert isinstance(json.loads(streamed), dict)


@pytest.mark.parametrize("chunk_size", [1, 1024])
def test_grammar_accepted_repeated_parameter_matches_final_json(chunk_size):
    import xgrammar as xgr
    from openai.types.responses import FunctionTool

    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

    body = (
        "<parameter=value>first</parameter>\n"
        "<parameter=value>second</parameter>\n"
        "<parameter=tail>ok</parameter>"
    )
    streamed, final, text = _parse_arguments(body, {"type": "string"}, chunk_size)
    tag = get_model_structural_tag(
        model="qwen_3_coder",
        tools=[
            FunctionTool(
                type="function",
                name="f",
                parameters={"type": "object"},
                strict=False,
            )
        ],
        tool_choice="required",
        reasoning=False,
    )
    tokenizer = xgr.TokenizerInfo(
        encoded_vocab=[bytes([i]) for i in range(256)],
        vocab_type=xgr.VocabType.RAW,
        vocab_size=256,
        stop_token_ids=[],
    )
    compiler = xgr.GrammarCompiler(tokenizer, max_threads=1)
    matcher = xgr.GrammarMatcher(
        compiler.compile_structural_tag(tag.model_dump_json()),
        terminate_without_stop_token=True,
    )
    assert all(matcher.accept_token(token) for token in text.encode())
    assert matcher.is_completed()
    assert json.loads(final) == {"value": "second", "tail": "ok"}
    assert streamed == final


def test_large_arguments_wait_for_close_but_other_channels_stream():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    )
    request = ChatCompletionRequest(
        model="test", messages=[], tools=[tool], tool_choice="required"
    )
    parser = Qwen3Parser(
        make_mock_tokenizer(
            {"<think>": 200, "</think>": 201, "<tool_call>": 202, "</tool_call>": 203}
        ),
        tools=[tool],
    )

    def feed(text):
        return parser.parse_delta(text, [], request, finished=False)

    reasoning = feed("Prepare the source code.")
    assert reasoning.reasoning == "Prepare the source code."
    content = feed("</think>I will submit the source code.")
    assert content.content == "I will submit the source code."
    header = feed("<tool_call>\n<function=f>\n<parameter=value>")
    assert header.tool_calls[0].function.name == "f"
    assert not header.tool_calls[0].function.arguments

    # A substantial multiline argument with JSON escapes and Unicode. Argument
    # buffering must not postpone earlier reasoning, content, or call identity.
    source = "".join(f'print("row {i}: ☃")\n' for i in range(400)) + "# done"
    for start in range(0, len(source), 257):
        delta = feed(source[start : start + 257])
        assert delta is None or not any(
            call.function and call.function.arguments for call in delta.tool_calls
        )
    parameter_end = feed("</parameter>\n")
    assert parameter_end is None or not any(
        call.function and call.function.arguments for call in parameter_end.tool_calls
    )
    closed = feed("</function>\n</tool_call>")
    arguments = "".join(
        call.function.arguments
        for call in closed.tool_calls
        if call.function and call.function.arguments
    )
    assert json.loads(arguments) == {"value": source}
