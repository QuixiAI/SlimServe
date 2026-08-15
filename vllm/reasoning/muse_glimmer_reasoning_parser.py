# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Muse-Glimmer-30B.

The chat template renders turns as ``<|start|>ROLE<|message|>BODY<|eot|>``.
Generation begins after a bare ``<|start|>assistant``; the model may first
emit a self-directed reasoning segment::

     to=self<|message|>REASONING<|eom|><|start|>assistant<|message|>CONTENT<|eot|>

or reply directly::

    <|message|>CONTENT<|eot|>

``to=self`` segments become ``reasoning_content``; ordinary assistant
segments (no recipient, or ``to=user``) become ``content``. Tool-call
segments (``to=<function>``) are left in the content for the tool parser.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_SEGMENT_RE = re.compile(
    r"(?:<\|start\|>assistant)?(?P<recipient> to=[^<]*)?<\|message\|>"
    r"(?P<body>.*?)(?:<\|eom\|>|<\|eot\|>|$)",
    re.DOTALL,
)


def _parse(text: str) -> tuple[str | None, str | None]:
    """Split raw generation text into (reasoning_content, content)."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    matched = False
    for match in _SEGMENT_RE.finditer(text):
        matched = True
        recipient = (match.group("recipient") or "").strip()
        body = match.group("body")
        if recipient == "to=self":
            reasoning_parts.append(body)
        elif recipient in ("", "to=user"):
            content_parts.append(body)
        else:
            # Tool-call segment: keep the full segment text for the tool
            # parser, recipient marker included.
            content_parts.append(match.group(0))
    if not matched:
        # No markers at all (e.g. plain-text continuation): all content.
        return None, text or None
    reasoning = "".join(reasoning_parts) or None
    content = "".join(content_parts) or None
    return reasoning, content


class MuseGlimmerReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        vocab = self.vocab
        self.eom_token_id = vocab.get("<|eom|>")
        self.message_token_id = vocab.get("<|message|>")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Reasoning is over once a non-self <|message|> body has begun: either
        # an <|eom|> closed the self segment, or the first segment was not
        # self-directed at all.
        ids = list(input_ids)
        if self.eom_token_id is not None and self.eom_token_id in ids:
            return True
        if self.message_token_id is None or self.message_token_id not in ids:
            return False
        first_message = ids.index(self.message_token_id)
        prefix = self.model_tokenizer.decode(ids[:first_message])
        return "to=self" not in prefix

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content ids follow the last <|eom|>-terminated self segment.
        if self.eom_token_id is not None and self.eom_token_id in input_ids:
            last = len(input_ids) - 1 - input_ids[::-1].index(self.eom_token_id)
            return input_ids[last + 1 :]
        return input_ids

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        return _parse(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        # Reparse-and-diff: robust against markers splitting across deltas.
        prev_reasoning, prev_content = _parse(previous_text)
        cur_reasoning, cur_content = _parse(current_text)
        delta_reasoning = (cur_reasoning or "")[len(prev_reasoning or "") :]
        delta_content = (cur_content or "")[len(prev_content or "") :]
        if not delta_reasoning and not delta_content:
            return None
        return DeltaMessage(
            reasoning_content=delta_reasoning or None,
            content=delta_content or None,
        )
