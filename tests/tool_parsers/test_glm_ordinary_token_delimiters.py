# SPDX-License-Identifier: Apache-2.0
"""A mid-call delimiter spelled with ordinary tokens is still structure.

Once token IDs are available, a terminal configured as a token ID is only
structure when it arrives as that token, so a model discussing "<tool_call>"
in prose cannot open a call. That protection used to apply inside an open
call too, where there is no prose to protect: GLM-5.3 sometimes spells a
mid-call delimiter with ordinary tokens - most often right after an
array-valued argument - and the machine then dropped out of the call, so the
held-back argument value, the remaining delimiters and every later call
arrived as raw XML in message content with no tool_calls at all (HexisPro:
~10% of real agent turns, 2026-10-15).
"""
import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

TOOLS = [
    {"type": "function", "function": {"name": "search_history", "parameters": {"type": "object",
        "properties": {"query": {"type": "string"},
                       "sources": {"type": "array", "items": {"type": "string"}}},
        "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_document_chunk", "parameters": {"type": "object",
        "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["chunk_ids"]}}},
]
EXPECTED_ARGS = '{"query": "Q3 budget", "sources": ["history", "memory"]}{"chunk_ids": ["644cca64"]}'


def _request():
    req = ChatCompletionRequest.model_validate(
        {"model": "zai-org/GLM-4.7", "messages": [{"role": "user", "content": "x"}],
         "tools": TOOLS, "tool_choice": "auto", "stream": True})
    req.include_reasoning = True
    return req


def _stream(tokenizer, segments):
    """Feed one (text, spelling) script token by token, as the server does."""
    req = _request()
    parser = Glm47MoeModelToolParser(tokenizer, tools=req.tools)

    def ids_for(text, spelling):
        if spelling != "ordinary":
            return tokenizer.encode(text, add_special_tokens=False)
        opening = "</" if text.startswith("</") else "<"
        pieces = (opening, text.strip("</>"), ">")
        return [t for piece in pieces for t in tokenizer.encode(piece, add_special_tokens=False)]

    deltas = [(tokenizer.decode([t]), t)
              for text, spelling in segments for t in ids_for(text, spelling)]
    prev, content, args, names, seen = "", "", "", [], []
    for piece, tid in deltas:
        delta = parser.extract_tool_calls_streaming(
            previous_text=prev, current_text=prev + piece, delta_text=piece,
            previous_token_ids=list(seen), current_token_ids=list(seen) + [tid],
            delta_token_ids=[tid], request=req)
        prev += piece
        seen.append(tid)
        if delta is None:
            continue
        content += delta.content or ""
        for call in (delta.tool_calls or []):
            if call.function and call.function.name:
                names.append(call.function.name)
            if call.function and call.function.arguments:
                args += call.function.arguments
    return names, args, content


def _script(closing_spelling):
    """Two calls; the first one's closing delimiters use `closing_spelling`."""
    return (
        [("I'll search.", "text")]
        + [("<tool_call>", "special"), ("search_history", "text"),
           ("<arg_key>", "special"), ("query", "text"), ("</arg_key>", "special"),
           ("<arg_value>", "special"), ("Q3 budget", "text"), ("</arg_value>", "special"),
           ("<arg_key>", "special"), ("sources", "text"), ("</arg_key>", "special"),
           ("<arg_value>", "special"), ('["history", "memory"]', "text")]
        + [("</arg_value>", closing_spelling), ("</tool_call>", closing_spelling)]
        + [("<tool_call>", "special"), ("open_document_chunk", "text"),
           ("<arg_key>", "special"), ("chunk_ids", "text"), ("</arg_key>", "special"),
           ("<arg_value>", "special"), ('["644cca64"]', "text"),
           ("</arg_value>", "special"), ("</tool_call>", "special")]
    )


@pytest.mark.usefixtures("glm53_tool_profile")
@pytest.mark.parametrize("spelling", ["special", "ordinary"])
def test_both_spellings_yield_the_same_calls(glm47_tokenizer, spelling):
    names, args, content = _stream(glm47_tokenizer, _script(spelling))
    assert names == ["search_history", "open_document_chunk"]
    # The array argument survives whole, and nothing structural reaches text.
    assert args == EXPECTED_ARGS
    assert content == "I'll search."


@pytest.mark.usefixtures("glm53_tool_profile")
def test_prose_outside_a_call_still_cannot_open_one(glm47_tokenizer):
    # The protection this rule exists for: a model quoting the markup in
    # ordinary tokens, with no call ever opened by its special token.
    names, args, content = _stream(glm47_tokenizer, [
        ("To call it, emit ", "text"), ("<tool_call>", "ordinary"),
        ("search_history", "text"), ("</tool_call>", "ordinary"),
    ])
    assert names == [] and args == ""
    assert "<tool_call>" in content
