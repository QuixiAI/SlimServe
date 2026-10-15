# SPDX-License-Identifier: Apache-2.0
"""An array-valued argument must not be mistaken for a JSON-array reply.

GLM-5.3 can answer with its tool calls as a bare JSON array, so the streaming
parser switches to that mode when the text starts with "[". An <arg_value>
holding an array makes the text the parser sees start with "[" while a native
<tool_call> envelope is already open; the parser then latched into array mode
and buffered the rest of the stream forever, so the value, the closing
delimiters and every later call were dumped into message content with no
tool_calls at all. That was ~10% of real agent turns, with an array argument
in every captured case (HexisPro, 2026-10-15) - sources=["memory"],
chunk_ids=[...] - and only ever in streaming, because the non-streaming path
reaches the array branch solely as a fallback after the native parse fails.
"""
import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.registered_adapters import Glm47MoeParserReasoningAdapter
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

TOOLS = [
    {"type": "function", "function": {"name": "search_history", "parameters": {"type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                       "sources": {"type": "array", "items": {"type": "string"}}},
        "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_document_chunk", "parameters": {"type": "object",
        "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["chunk_ids"]}}},
]


class _Combined(DelegatingParser):
    """What the server runs: reasoning and tools in one parser."""
    reasoning_parser_cls = Glm47MoeParserReasoningAdapter
    tool_parser_cls = Glm47MoeModelToolParser


def _request():
    req = ChatCompletionRequest.model_validate(
        {"model": "zai-org/GLM-4.7", "messages": [{"role": "user", "content": "x"}],
         "tools": TOOLS, "tool_choice": "auto", "stream": True})
    req.include_reasoning = True
    return req


def _stream(tokenizer, raw):
    req = _request()
    parser = _Combined(tokenizer, tools=req.tools,
                       chat_template_kwargs={"thinking": False, "enable_thinking": False})
    ids = tokenizer.encode(raw, add_special_tokens=False)
    content, arguments, names = "", "", []
    for i, token_id in enumerate(ids):
        delta = parser.parse_delta(
            tokenizer.decode([token_id]), [token_id], req,
            prompt_token_ids=[] if i == 0 else None, finished=(i == len(ids) - 1))
        if delta is None:
            continue
        content += delta.content or ""
        for call in (delta.tool_calls or []):
            if call.function and call.function.name:
                names.append(call.function.name)
            if call.function and call.function.arguments:
                arguments += call.function.arguments
    return names, arguments, content


def _no_markup(text):
    return not any(t in text for t in ("<tool_call", "<arg_key>", "<arg_value>", "</arg_value>"))


@pytest.mark.usefixtures("glm53_tool_profile")
def test_array_argument_streams_as_a_tool_call(glm47_tokenizer):
    names, arguments, content = _stream(glm47_tokenizer,
        "Sure.<tool_call>search_history<arg_key>query</arg_key><arg_value>Q3</arg_value>"
        '<arg_key>sources</arg_key><arg_value>["history", "memory"]</arg_value></tool_call>')
    assert names == ["search_history"]
    assert arguments == '{"query": "Q3", "sources": ["history", "memory"]}'
    assert content == "Sure." and _no_markup(content)


@pytest.mark.usefixtures("glm53_tool_profile")
def test_a_call_after_an_array_argument_is_not_swallowed(glm47_tokenizer):
    # The array used to end parsing, so later calls leaked as raw markup.
    names, arguments, content = _stream(glm47_tokenizer,
        '<tool_call>search_history<arg_key>sources</arg_key><arg_value>["memory"]</arg_value></tool_call>'
        '<tool_call>open_document_chunk<arg_key>chunk_ids</arg_key>'
        '<arg_value>["644cca64"]</arg_value></tool_call>')
    assert names == ["search_history", "open_document_chunk"]
    assert arguments == '{"sources": ["memory"]}{"chunk_ids": ["644cca64"]}'
    assert _no_markup(content)


@pytest.mark.usefixtures("glm53_tool_profile")
def test_string_only_arguments_are_unaffected(glm47_tokenizer):
    names, arguments, content = _stream(glm47_tokenizer,
        "<tool_call>search_history<arg_key>query</arg_key><arg_value>Q3</arg_value></tool_call>")
    assert names == ["search_history"]
    assert arguments == '{"query": "Q3"}'
    assert _no_markup(content)
