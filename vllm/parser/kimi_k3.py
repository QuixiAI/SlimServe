# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 parser: reasoning and content out of K3's XTML frame.

K3 does not use ``<think>``. It frames every channel with three special tokens,
rendered by ``vllm/transformers_utils/tokenizers/encoding_k3.py``:

    open   <|open|>  tag ( k="v")* <|sep|>
    close  <|close|> tag           <|sep|>

and an assistant turn is a ``think`` channel (thinking mode only) followed by a
``response`` channel, both inside a ``message``. The generation prompt already
opens the first channel, so generation *starts inside it* -- with thinking on
the model resumes mid-``think``, otherwise mid-``response``:

    thinking   REASONING <|close|>think<|sep|> <|open|>response<|sep|>
               ANSWER <|close|>response<|sep|> <|close|>message<|sep|>
    otherwise  ANSWER <|close|>response<|sep|> <|close|>message<|sep|>

Only the three delimiters are special tokens; a tag name is ordinary text. So
the machine cannot branch on which tag it is looking at -- it only knows whether
a name follows an open or a close. That is enough, because the two lead to
different places: the name after ``<|open|>`` ends at a ``<|sep|>`` that starts a
channel body, while the name after ``<|close|>`` ends at a ``<|sep|>`` that
leaves one. Those are the two header states below, and both discard their text,
which is what keeps tag names and attributes out of the reply.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

# The one state whose text the engine drops and that is not part of a tool
# transition. That last part is load-bearing: the reasoning-only pass runs with
# skip_tool_parsing, which re-emits any terminal belonging to a tool state as
# literal text -- so routing K3's delimiters through TOOL_PREAMBLE/TOOL_BETWEEN
# put the whole frame back in the reply, which is the bug this parser exists to
# fix. Every tag name goes here instead, whether it opened or closed a channel.
_TAG_NAME = ParserState.MESSAGE_HEADER


@functools.cache
def kimi_k3_config(thinking: bool = True) -> ParserEngineConfig:
    terminals = {
        "OPEN": OPEN_TOKEN,
        "CLOSE": CLOSE_TOKEN,
        "SEP": SEP_TOKEN,
        # Claimed rather than left to the engine's drop-unknown-special-tokens
        # pass: requests run with skip_special_tokens=False, and K3's tokenizer
        # is a vendored TikToken one, so end-of-message was surviving that pass
        # and arriving at the caller.
        "END": END_OF_MSG_TOKEN,
    }
    # Both delimiters lead to a tag name, and the <|sep|> that ends the name
    # hands back to a body. Which body does not need distinguishing: a close is
    # always followed by another tag, so anything landing in CONTENT between
    # channels is the whitespace separating them.
    to_tag_name = Transition(_TAG_NAME)
    return ParserEngineConfig(
        name="kimi_k3",
        # Generation resumes inside whichever channel the prompt opened.
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals=terminals,
        # All three are real special tokens, so match them by id too: that is
        # both split-proof across deltas and unspoofable by a model that writes
        # the literal text in prose.
        token_id_terminals=dict(terminals),
        transitions={
            # Leaving the think channel is the end of reasoning, however it
            # ends -- a close tag normally, an open tag if the close was lost.
            (ParserState.REASONING, "CLOSE"): Transition(
                _TAG_NAME, (EventType.REASONING_END,)
            ),
            (ParserState.REASONING, "OPEN"): Transition(
                _TAG_NAME, (EventType.REASONING_END,)
            ),
            (ParserState.CONTENT, "CLOSE"): to_tag_name,
            (ParserState.CONTENT, "OPEN"): to_tag_name,
            (_TAG_NAME, "SEP"): Transition(ParserState.CONTENT),
            # Nothing follows end-of-message, so park in the state that drops
            # text and let anything after it go there too.
            (ParserState.CONTENT, "END"): to_tag_name,
            (ParserState.REASONING, "END"): Transition(
                _TAG_NAME, (EventType.REASONING_END,)
            ),
            (_TAG_NAME, "END"): to_tag_name,
        },
    )


class KimiK3Parser(ParserEngine):
    """K3 renders thinking as a channel, so the mode changes where we start."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        thinking = chat_kwargs.get("thinking")
        if thinking is None:
            thinking = chat_kwargs.get("enable_thinking")
        # K3's own template defaults thinking on, so match it when unspecified.
        self.thinking_enabled = True if thinking is None else bool(thinking)
        kwargs.setdefault(
            "parser_engine_config", kimi_k3_config(thinking=self.thinking_enabled)
        )
        super().__init__(tokenizer, tools, **kwargs)
