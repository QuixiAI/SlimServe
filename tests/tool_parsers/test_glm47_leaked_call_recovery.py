# SPDX-License-Identifier: Apache-2.0
"""Recovery of GLM tool calls the state machine does not claim.

The engine matches GLM's grammar exactly. When a reply deviates - and under
sampling it sometimes does - every call in it used to reach the client as XML
in `content` with no `tool_calls`, which an agent loop cannot see at all
(HexisPro measured ~10% of real turns on 43-tool requests, 2026-10-15, with
array-valued arguments in every captured case). The parser now rebuilds such
calls, using the same argument regex and schema coercion as the engine, so a
recovered call is indistinguishable from a parsed one.
"""
import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.7"
TOOLS = [
    {"type": "function", "function": {"name": "search_history", "parameters": {"type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"},
                       "sources": {"type": "array", "items": {"type": "string"}}},
        "required": ["query"]}}},
    {"type": "function", "function": {"name": "recall", "parameters": {"type": "object",
        "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
ARGS = ("<arg_key>query</arg_key><arg_value>Q3 budget</arg_value>"
        "<arg_key>limit</arg_key><arg_value>5</arg_value>"
        "<arg_key>sources</arg_key><arg_value>[\"memory\"]</arg_value>")
EXPECTED = {"query": "Q3 budget", "limit": 5, "sources": ["memory"]}


@pytest.fixture
def parser(glm47_tokenizer):
    return lambda: Glm47MoeModelToolParser(glm47_tokenizer, tools=request_obj().tools)


def request_obj():
    return ChatCompletionRequest.model_validate(
        {"model": MODEL, "messages": [{"role": "user", "content": "x"}],
         "tools": TOOLS, "tool_choice": "auto"})


def _extract(glm47_tokenizer, raw):
    req = request_obj()
    p = Glm47MoeModelToolParser(glm47_tokenizer, tools=req.tools)
    return p.extract_tool_calls(raw, request=req)


def _no_markup(text):
    return not any(t in (text or "") for t in ("<tool_call", "<arg_key>", "<arg_value>"))


@pytest.mark.usefixtures("glm53_tool_profile")
def test_grammar_conforming_call_is_unaffected(glm47_tokenizer):
    r = _extract(glm47_tokenizer, "<tool_call>search_history" + ARGS + "</tool_call>")
    assert r.tools_called and len(r.tool_calls) == 1
    assert json.loads(r.tool_calls[0].function.arguments) == EXPECTED


@pytest.mark.usefixtures("glm53_tool_profile")
@pytest.mark.parametrize("raw, label", [
    ('<tool_call name="search_history">' + ARGS + "</tool_call>", "attribute spelling"),
    ('Let me look.\n<tool_call name="search_history">' + ARGS + "</tool_call>", "prose then call"),
])
def test_deviant_opening_tag_is_recovered(glm47_tokenizer, raw, label):
    r = _extract(glm47_tokenizer, raw)
    assert r.tools_called, label
    assert r.tool_calls[0].function.name == "search_history"
    # Arguments keep their schema types, exactly as the engine would produce.
    assert json.loads(r.tool_calls[0].function.arguments) == EXPECTED
    assert _no_markup(r.content)


@pytest.mark.usefixtures("glm53_tool_profile")
def test_several_recovered_calls_keep_their_order(glm47_tokenizer):
    raw = ('<tool_call name="search_history">' + ARGS + "</tool_call>\n"
           '<tool_call name="recall"><arg_key>query</arg_key><arg_value>y</arg_value></tool_call>')
    r = _extract(glm47_tokenizer, raw)
    assert [c.function.name for c in r.tool_calls] == ["search_history", "recall"]
    assert _no_markup(r.content)


@pytest.mark.usefixtures("glm53_tool_profile")
def test_an_unoffered_tool_name_is_never_invented(glm47_tokenizer):
    # Better to hand back text the client can see than to fabricate a call
    # for a tool the request never offered.
    r = _extract(glm47_tokenizer, '<tool_call name="search_hist">' + ARGS + "</tool_call>")
    assert not r.tools_called and not r.tool_calls


@pytest.mark.usefixtures("glm53_tool_profile")
def test_a_reply_with_no_call_is_untouched(glm47_tokenizer):
    r = _extract(glm47_tokenizer, "Just a plain answer.")
    assert not r.tools_called
    assert r.content == "Just a plain answer."
