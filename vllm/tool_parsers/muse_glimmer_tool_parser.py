# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse-Glimmer ATEM tool-call parser and structural decoding format.

The released model's chat template uses recipient-framed ATEM calls::

    <|start|>assistant to=apply_patch<|message|>
    <atem:function_calls>
    <atem:invoke name="apply_patch">
    <atem:parameter name="input">*** Begin Patch ...</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eot|>

Responses custom tools use one synthetic ``input`` parameter internally, but
the API-facing parser returns that parameter body as an unwrapped string.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import regex as re
from openai.types.responses import CustomTool
from xgrammar import StructuralTag
from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.structural_tag_registry import (
    SimplifiedToolChoice,
    custom_tool_as_function_schema,
    get_custom_tool_input_format,
    register_custom_tool_format,
    register_vllm_structural_tag,
    replace_custom_tool_payloads,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    find_tool_properties,
    partial_tag_overlap,
    responses_custom_tool_names,
)

ATEM_CALLS_START = "<atem:function_calls>"
ATEM_CALLS_END = "</atem:function_calls>"
ATEM_INVOKE_PREFIX = '<atem:invoke name="'
ATEM_INVOKE_NAME_END = '">\n'
ATEM_INVOKE_END = "</atem:invoke>"
ATEM_PARAM_PREFIX = '<atem:parameter name="'
ATEM_PARAM_NAME_END = '">'
ATEM_PARAM_END = "</atem:parameter>"

_CALLS_RE = re.compile(
    re.escape(ATEM_CALLS_START) + r"(?P<body>.*?)" + re.escape(ATEM_CALLS_END),
    re.DOTALL,
)
_INVOKE_RE = re.compile(
    r'<atem:invoke\s+name="(?P<name>[^"]+)">\s*'
    r"(?P<body>.*?)</atem:invoke>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r'<atem:parameter\s+name="(?P<name>[^"]+)">'
    r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)
_RECIPIENT_CALL_RE = re.compile(
    r"(?:<\|start\|>assistant)?\s*to=[^<]*<\|message\|>\s*"
    + re.escape(ATEM_CALLS_START)
    + r".*?"
    + re.escape(ATEM_CALLS_END)
    + r"(?:<\|eom\|>|<\|eot\|>)?",
    re.DOTALL,
)
_ASSISTANT_SEGMENT_RE = re.compile(
    r"(?:<\|start\|>assistant)?(?P<recipient> to=[^<]*)?<\|message\|>"
    r"(?P<body>.*?)(?:<\|eom\|>|<\|eot\|>|\Z)",
    re.DOTALL,
)
_UNFRAMED_CONTROL_RE = re.compile(
    r"<\|(?:eom|eot|begin_of_text|end_of_text)\|>"
    r"|<\|start\|>assistant(?: to=user)?"
    r"|<\|message\|>"
)

_ASSISTANT_START = "<|start|>assistant"
_MESSAGE_START = "<|message|>"
_SEGMENT_ENDS = ("<|eom|>", "<|eot|>")
_STREAM_CONTROL_MARKERS = (
    _ASSISTANT_START,
    _MESSAGE_START,
    *_SEGMENT_ENDS,
    "<|begin_of_text|>",
    "<|end_of_text|>",
    ATEM_CALLS_START,
    ATEM_CALLS_END,
)
_RECIPIENT_PREFIXES = (" to=", "to=")


def _atem_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return [
        TagFormat(
            begin=(
                ATEM_INVOKE_PREFIX
                + tool.function.name.replace("&", "&amp;").replace('"', "&quot;")
                + ATEM_INVOKE_NAME_END
            ),
            content=AnyTextFormat(excludes=[ATEM_INVOKE_END]),
            end=ATEM_INVOKE_END,
        )
        for tool in tools
    ]


@register_vllm_structural_tag("muse_glimmer")
def get_muse_glimmer_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    """Preserve Muse reasoning/recipient text and constrain its ATEM block."""
    del builtin_tools

    calls_tag = TagFormat(
        begin=ATEM_CALLS_START + "\n",
        content=TagsWithSeparatorFormat(
            tags=_atem_tool_tags(tools),
            separator="\n",
            at_least_one=True,
            stop_after_first=tool_choice == "forced",
        ),
        end="\n" + ATEM_CALLS_END,
    )
    if tool_choice == "auto":
        output_format = TriggeredTagsFormat(
            triggers=[ATEM_CALLS_START],
            tags=[calls_tag],
        )
    elif tool_choice == "forced" and not reasoning:
        # A named choice leaves exactly one tool after xgrammar normalizes the
        # request.  Muse selects the recipient before opening the ATEM block;
        # force that model-native prefix as well as the block itself.  An
        # unconstrained text prefix would let the model emit an ordinary
        # assistant message and stop before ever reaching the required tag.
        tool_name = tools[0].function.name
        output_format = SequenceFormat(
            elements=[
                ConstStringFormat(value=f" to={tool_name}<|message|>"),
                calls_tag,
            ]
        )
    else:
        prefix_excludes = [ATEM_CALLS_START]
        if not reasoning:
            # Muse's reasoning channel is a recipient segment beginning with
            # ``to=self``. For a forced tool call with reasoning effort none,
            # exclude that recipient at the grammar level while still allowing
            # the model-native ``to=<tool><|message|>`` prefix before ATEM.
            prefix_excludes.append("to=self")
        output_format = SequenceFormat(
            elements=[
                AnyTextFormat(excludes=prefix_excludes),
                calls_tag,
                AnyTextFormat(),
            ]
        )
    return StructuralTag(format=output_format)


@register_custom_tool_format(
    "muse_glimmer",
    as_function_tool=custom_tool_as_function_schema,
)
def _apply_muse_glimmer_custom_tool_format(
    structural_tag: StructuralTag,
    custom_tools: dict[str, CustomTool],
) -> None:
    """Apply a custom grammar only to Muse's synthetic ATEM input body."""

    def match_tool_name(tag: TagFormat) -> str | None:
        begin = tag.begin
        if not isinstance(begin, str) or not begin.startswith(ATEM_INVOKE_PREFIX):
            return None
        escaped = begin[len(ATEM_INVOKE_PREFIX) :].split('"', 1)[0]
        return escaped.replace("&quot;", '"').replace("&amp;", "&")

    def make_content(custom_tool: CustomTool) -> SequenceFormat:
        return SequenceFormat(
            elements=[
                ConstStringFormat(
                    value=ATEM_PARAM_PREFIX + "input" + ATEM_PARAM_NAME_END
                ),
                get_custom_tool_input_format(custom_tool),
                ConstStringFormat(value=ATEM_PARAM_END + "\n"),
            ]
        )

    replace_custom_tool_payloads(
        structural_tag,
        custom_tools,
        match_tool_name=match_tool_name,
        make_content=make_content,
    )


class MuseGlimmerToolParser(ToolParser):
    supports_required_and_named = False
    structural_tag_model = "muse_glimmer"
    handles_tool_choice_none = True

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._sent_tool_call_count = 0
        # Streaming content is consumed monotonically from raw deltas. Do not
        # index into a repeatedly filtered version of cumulative text: when a
        # split protocol marker becomes complete, that filtered text can
        # shrink retroactively and corrupt the content cursor.
        self._content_pending = ""
        self._content_visible = True
        self._content_expects_header = True

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
    ):
        # The shared parser historically passes ``reasoning=False`` because
        # most tool grammars do not own the model's reasoning envelope. Muse
        # does: its self-recipient segment precedes the ATEM block. Preserve
        # that segment unless the API explicitly requests effort ``none``.
        if isinstance(request, ResponsesRequest):
            effort = request.reasoning.effort if request.reasoning else None
        else:
            effort = request.reasoning_effort
        return super().get_structural_tag(
            request,
            reasoning=effort != "none",
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        # ATEM and recipient markers are tokenizer special tokens and must
        # survive detokenization as contiguous strings for parsing.
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    @staticmethod
    def _unescape_attr(value: str) -> str:
        return value.replace("&quot;", '"').replace("&amp;", "&")

    def _decode_invoke(
        self,
        name: str,
        body: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ToolCall:
        name = self._unescape_attr(name)
        params = {
            self._unescape_attr(match["name"]): match["value"]
            for match in _PARAM_RE.finditer(body)
        }
        if name in responses_custom_tool_names(request.tools):
            return ToolCall(
                function=FunctionCall(name=name, arguments=params.get("input", ""))
            )

        properties = find_tool_properties(request.tools, name)
        typed: dict[str, object] = {}
        for key, value in params.items():
            schema_type = properties.get(key, {}).get("type", "string")
            typed[key] = coerce_to_schema_type(value, schema_type)
        return ToolCall(
            function=FunctionCall(
                name=name,
                arguments=json.dumps(typed, ensure_ascii=False),
            )
        )

    def _calls(
        self,
        text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> list[ToolCall]:
        return [
            self._decode_invoke(invoke["name"], invoke["body"], request)
            for block in _CALLS_RE.finditer(text)
            for invoke in _INVOKE_RE.finditer(block["body"])
        ]

    @staticmethod
    def _visible_assistant_content(text: str) -> str | None:
        """Return user-directed bodies without Muse protocol delimiters."""
        content: list[str] = []
        cursor = 0
        for match in _ASSISTANT_SEGMENT_RE.finditer(text):
            prefix = _UNFRAMED_CONTROL_RE.sub("", text[cursor : match.start()])
            if prefix:
                content.append(prefix)
            recipient = (match.group("recipient") or "").strip()
            if recipient in ("", "to=user"):
                content.append(match.group("body"))
            cursor = match.end()
        suffix = _UNFRAMED_CONTROL_RE.sub("", text[cursor:])
        if suffix:
            content.append(suffix)
        return "".join(content) or None

    @classmethod
    def _content_without_calls(cls, text: str) -> str | None:
        text = _RECIPIENT_CALL_RE.sub("", text)
        text = _CALLS_RE.sub("", text)
        return cls._visible_assistant_content(text)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ExtractedToolCallInformation:
        calls = self._calls(model_output, request)
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content=self._content_without_calls(model_output),
        )

    def _consume_expected_recipient(self) -> bool | None:
        """Consume a recipient header, or report that it is incomplete.

        ``True`` means a complete header was consumed, ``False`` means the
        pending text is ordinary visible content, and ``None`` means more raw
        input is required before making that decision.
        """
        pending = self._content_pending
        for prefix in _RECIPIENT_PREFIXES:
            if pending.startswith(prefix):
                message_at = pending.find(_MESSAGE_START, len(prefix))
                if message_at < 0:
                    return None
                recipient = pending[len(prefix) : message_at].strip()
                self._content_visible = recipient in ("", "user")
                self._content_expects_header = False
                self._content_pending = pending[message_at + len(_MESSAGE_START) :]
                return True
            if len(pending) < len(prefix) and prefix.startswith(pending):
                return None

        # Hold split control markers until their complete spelling is known.
        # Complete markers are handled by the main scanner below.
        if any(
            len(pending) < len(marker) and marker.startswith(pending)
            for marker in _STREAM_CONTROL_MARKERS
        ):
            return None

        self._content_visible = True
        self._content_expects_header = False
        return False

    def _content_delta(self, delta_text: str) -> str | None:
        """Return stable visible content from one raw Muse stream delta."""
        self._content_pending += delta_text
        visible: list[str] = []

        while self._content_pending:
            if self._content_expects_header:
                consumed = self._consume_expected_recipient()
                if consumed is None:
                    break
                if consumed:
                    continue

            matches = [
                (index, marker)
                for marker in _STREAM_CONTROL_MARKERS
                if (index := self._content_pending.find(marker)) >= 0
            ]
            if not matches:
                overlap = max(
                    partial_tag_overlap(self._content_pending, marker)
                    for marker in _STREAM_CONTROL_MARKERS
                )
                sendable = len(self._content_pending) - overlap
                if sendable:
                    if self._content_visible:
                        visible.append(self._content_pending[:sendable])
                    self._content_pending = self._content_pending[sendable:]
                break

            marker_at, marker = min(matches, key=lambda item: item[0])
            if marker_at:
                if self._content_visible:
                    visible.append(self._content_pending[:marker_at])
                self._content_pending = self._content_pending[marker_at:]
                continue

            self._content_pending = self._content_pending[len(marker) :]
            if marker == _ASSISTANT_START:
                self._content_visible = False
                self._content_expects_header = True
            elif marker == _MESSAGE_START:
                self._content_visible = True
                self._content_expects_header = False
            elif marker in _SEGMENT_ENDS:
                self._content_visible = True
                self._content_expects_header = True
            elif marker == ATEM_CALLS_START:
                self._content_visible = False
                self._content_expects_header = False
            elif marker == ATEM_CALLS_END:
                # A tool recipient remains hidden through its closing EOT.
                self._content_visible = False

        return "".join(visible) or None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        del previous_text, previous_token_ids, current_token_ids, delta_token_ids

        content = self._content_delta(delta_text)
        calls = self._calls(current_text, request)
        if len(calls) <= self._sent_tool_call_count:
            return DeltaMessage(content=content) if content else None

        new_calls = calls[self._sent_tool_call_count :]
        deltas = [
            DeltaToolCall(
                index=self._sent_tool_call_count + index,
                id=call.id,
                type="function",
                function=DeltaFunctionCall(
                    name=call.function.name,
                    arguments=call.function.arguments,
                ),
            )
            for index, call in enumerate(new_calls)
        ]
        self._sent_tool_call_count = len(calls)
        return DeltaMessage(content=content, tool_calls=deltas)
