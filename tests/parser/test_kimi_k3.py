# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's XTML frame must not reach the caller.

The strings below are real completions captured from k3-6, not invented: K3
closes its channels in-band, so without a parser every served answer trails
"<|close|> response <|sep|> <|close|> message <|sep|>".
"""

import pytest

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.kimi_k3 import kimi_k3_config

# Captured from k3-6, thinking off.
PLAIN = (
    " The capital of France is **Paris**. <|close|> response <|sep|> "
    "<|close|> message <|sep|> <|end_of_msg|>"
)
# The same shape with a think channel in front, as thinking mode emits.
THINKING = (
    " The user asks for a capital. Simple. <|close|> think <|sep|> "
    "<|open|> response <|sep|> The capital of France is **Paris**. "
    "<|close|> response <|sep|> <|close|> message <|sep|> <|end_of_msg|>"
)
FRAME_MARKERS = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")


def _run(text: str, thinking: bool, chunk: int | None = None):
    """Feed `text` through the engine and return (content, reasoning)."""
    engine = StreamingParserEngine(kimi_k3_config(thinking=thinking), tokenizer=None)
    events = []
    if chunk is None:
        events.extend(engine.parse_complete(text))
    else:
        for i in range(0, len(text), chunk):
            events.extend(engine.feed(text[i : i + chunk], []))
        events.extend(engine.finish())
    content = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
    reasoning = "".join(e.value for e in events if e.type == EventType.REASONING_CHUNK)
    return content, reasoning


def test_non_thinking_answer_has_no_frame():
    content, reasoning = _run(PLAIN, thinking=False)
    assert content.strip() == "The capital of France is **Paris**."
    assert reasoning == ""
    assert not any(m in content for m in FRAME_MARKERS)


def test_tag_names_are_not_leaked_as_content():
    """'response' and 'message' are ordinary text inside the frame."""
    content, _ = _run(PLAIN, thinking=False)
    assert "response" not in content
    assert "message" not in content


def test_thinking_splits_reasoning_from_answer():
    content, reasoning = _run(THINKING, thinking=True)
    assert content.strip() == "The capital of France is **Paris**."
    assert reasoning.strip() == "The user asks for a capital. Simple."
    assert not any(m in content or m in reasoning for m in FRAME_MARKERS)


def test_reasoning_end_is_announced():
    """The serving layer needs this to know the think channel closed."""
    engine = StreamingParserEngine(kimi_k3_config(thinking=True), tokenizer=None)
    types = [e.type for e in engine.parse_complete(THINKING)]
    assert EventType.REASONING_END in types
    assert types.index(EventType.REASONING_END) < len(types) - 1


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 13, 64])
@pytest.mark.parametrize(("text", "thinking"), [(PLAIN, False), (THINKING, True)])
def test_chunking_does_not_change_the_result(text, thinking, chunk):
    """A delimiter split across deltas must not leak or drop text."""
    whole = _run(text, thinking)
    assert _run(text, thinking, chunk=chunk) == whole


def test_unterminated_reasoning_still_closes():
    """A truncated think channel must not swallow the whole reply."""
    content, reasoning = _run(" thinking with no close tag", thinking=True)
    assert reasoning.strip() == "thinking with no close tag"
    assert content == ""


@pytest.mark.parametrize(("text", "thinking"), [(PLAIN, False), (THINKING, True)])
def test_no_frame_leaks_on_the_reasoning_only_pass(text, thinking):
    """The reasoning adapter runs with skip_tool_parsing, and that path re-emits
    any terminal owned by a tool state as literal text. Routing K3's delimiters
    through TOOL_PREAMBLE/TOOL_BETWEEN therefore put the whole frame back into
    the reply while every direct-engine test still passed.
    """
    engine = StreamingParserEngine(kimi_k3_config(thinking=thinking), tokenizer=None)
    engine.skip_tool_parsing = True  # what the reasoning adapter sets
    events = engine.parse_complete(text)
    content = "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)
    assert not any(m in content for m in FRAME_MARKERS)
    assert "The capital of France is **Paris**." in content


def test_no_terminal_belongs_to_a_tool_state():
    """Guards the above at the config level, where the cause actually lives."""
    tool_states = {
        ParserState.TOOL_PREAMBLE,
        ParserState.TOOL_NAME,
        ParserState.TOOL_ARGS,
        ParserState.TOOL_BETWEEN,
    }
    for thinking in (True, False):
        for (state, _), transition in kimi_k3_config(thinking).transitions.items():
            assert state not in tool_states
            assert transition.next_state not in tool_states


def test_initial_state_follows_the_thinking_flag():
    """The prompt opens the channel, so the parser must start inside it."""
    assert kimi_k3_config(thinking=True).initial_state is ParserState.REASONING
    assert kimi_k3_config(thinking=False).initial_state is ParserState.CONTENT
