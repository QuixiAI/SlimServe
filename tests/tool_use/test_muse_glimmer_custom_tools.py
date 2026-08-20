# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from textwrap import dedent

from openai.types.responses import ResponseCustomToolCall
from xgrammar import Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo

from vllm.entrypoints.chat_utils import (
    adapt_custom_tool_calls_for_chat_template,
    adapt_custom_tools_for_chat_template,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    split_delta,
)
from vllm.entrypoints.openai.responses.utils import (
    build_response_output_items,
    construct_chat_messages_with_tool_call,
    construct_tool_dicts,
)
from vllm.parser.parser_manager import ParserManager
from vllm.renderers.online_renderer import _adapt_responses_custom_tool_rendering
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

APPLY_PATCH_LARK = dedent(
    """\
    start: begin_patch hunk+ end_patch
    begin_patch: "*** Begin Patch" LF
    end_patch: "*** End Patch" LF?

    hunk: add_hunk | delete_hunk | update_hunk
    add_hunk: "*** Add File: " filename LF add_line+
    delete_hunk: "*** Delete File: " filename LF
    update_hunk: "*** Update File: " filename LF change_move? change?

    filename: /(.+)/
    add_line: "+" /(.*)/ LF -> line

    change_move: "*** Move to: " filename LF
    change: (change_context | change_line)+ eof_line?
    change_context: ("@@" | "@@ " /(.+)/) LF
    change_line: ("+" | "-" | " ") /(.*)/ LF
    eof_line: "*** End of File" LF

    %import common.LF
    """
)
RAW_PATCH = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n"
APPLY_PATCH_REGEX = r"\*\*\* Begin Patch\n[\s\S]*\n\*\*\* End Patch\n?"


class _DummyTokenizer:
    def get_vocab(self):
        return {"<|message|>": 1, "<|eom|>": 2}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def _apply_patch_request(
    *,
    tool_choice="required",
    syntax: str = "lark",
    stream: bool = False,
) -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "model": "Muse-Glimmer-30B",
            "input": "Use apply_patch to add hello.txt containing hello.",
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": (
                        "Use the `apply_patch` tool to edit files. This is a "
                        "FREEFORM tool, so do not wrap the patch in JSON."
                    ),
                    "format": {
                        "type": "grammar",
                        "syntax": syntax,
                        "definition": (
                            APPLY_PATCH_LARK if syntax == "lark" else APPLY_PATCH_REGEX
                        ),
                    },
                }
            ],
            "tool_choice": tool_choice,
            "stream": stream,
        }
    )


def _atem_call(name: str, parameters: dict[str, str]) -> str:
    args = "".join(
        f'<atem:parameter name="{key}">{value}</atem:parameter>\n'
        for key, value in parameters.items()
    )
    return (
        f"<|start|>assistant to={name}<|message|>"
        "<atem:function_calls>\n"
        f'<atem:invoke name="{name}">\n'
        f"{args}"
        "</atem:invoke>\n"
        "</atem:function_calls><|eot|>"
    )


def test_muse_custom_lark_structural_tag_constrains_only_input_body():
    request = _apply_patch_request()

    tag = get_model_structural_tag(
        "muse_glimmer", request.tools, request.tool_choice, reasoning=False
    )

    assert tag is not None
    dumped = tag.model_dump_json()
    assert '<atem:invoke name=\\"apply_patch\\">' in dumped
    assert '<atem:parameter name=\\"input\\">' in dumped
    assert "root ::= start" in dumped
    Grammar.from_structural_tag(tag)


def test_muse_codex_apply_patch_lark_contract_compiles():
    request = _apply_patch_request()

    tag = get_model_structural_tag(
        "muse_glimmer", request.tools, request.tool_choice, reasoning=False
    )

    assert tag is not None
    dumped = tag.model_dump_json()
    assert '<atem:invoke name=\\"apply_patch\\">' in dumped
    assert '<atem:parameter name=\\"input\\">' in dumped
    assert "begin_patch ::=" in dumped
    assert "end_patch ::=" in dumped
    Grammar.from_structural_tag(tag)

    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)
    result = parser.extract_tool_calls(
        _atem_call("apply_patch", {"input": RAW_PATCH}), request
    )
    assert result.tool_calls[0].function.name == "apply_patch"
    assert result.tool_calls[0].function.arguments == RAW_PATCH


def test_codex_apply_patch_lark_accepts_patch_but_not_multiline_filename():
    request = _apply_patch_request()
    tag = get_model_structural_tag(
        "muse_glimmer", request.tools, request.tool_choice, reasoning=False
    )
    assert tag is not None
    grammar = tag.format.elements[1].content.tags[0].content.elements[1].grammar
    compiler = GrammarCompiler(TokenizerInfo([chr(i) for i in range(128)]))
    compiled = compiler.compile_grammar(Grammar.from_ebnf(grammar))

    matcher = GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(RAW_PATCH)
    assert matcher.is_completed()

    matcher.reset()
    invalid = "*** Begin Patch\n*** Add File: hello\n.txt\n+hello\n*** End Patch\n"
    assert not matcher.accept_string(invalid)


def test_muse_custom_regex_structural_tag_compiles():
    request = _apply_patch_request(syntax="regex")

    tag = get_model_structural_tag(
        "muse_glimmer", request.tools, request.tool_choice, reasoning=False
    )

    assert tag is not None
    assert r"\\*\\*\\* Begin Patch" in tag.model_dump_json()
    Grammar.from_structural_tag(tag)


def test_muse_named_custom_choice_builds_forced_atem_tag():
    request = _apply_patch_request(
        tool_choice={"type": "custom", "name": "apply_patch"}
    )

    tag = get_model_structural_tag(
        "muse_glimmer", request.tools, request.tool_choice, reasoning=False
    )

    assert tag is not None
    assert 'invoke name=\\"apply_patch\\"' in tag.model_dump_json()
    Grammar.from_structural_tag(tag)


def test_muse_non_streaming_custom_call_returns_raw_input():
    request = _apply_patch_request()
    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)

    result = parser.extract_tool_calls(
        _atem_call("apply_patch", {"input": RAW_PATCH}), request
    )

    assert result.tools_called
    assert result.content is None
    assert result.tool_calls[0].function.name == "apply_patch"
    assert result.tool_calls[0].function.arguments == RAW_PATCH


def test_muse_streaming_custom_call_emits_raw_input_after_close():
    request = _apply_patch_request()
    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)
    output = _atem_call("apply_patch", {"input": RAW_PATCH})

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
    assert delta.content is None
    assert delta.tool_calls[0].function.name == "apply_patch"
    assert delta.tool_calls[0].function.arguments == RAW_PATCH


def test_muse_streaming_plain_text_containing_tool_name_is_not_withheld():
    request = _apply_patch_request(tool_choice="auto")
    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)
    output = "I decided to=apply_patch is not a tool call."

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
    assert delta.content == output
    assert not delta.tool_calls


def test_muse_streaming_split_atem_markers_do_not_leak():
    request = _apply_patch_request()
    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)
    previous_text = ""
    previous_ids: list[int] = []
    messages = []
    chunks = [
        " to=apply_patch",
        "<|message|>",
        "<atem:",
        "function_calls>\n",
        '<atem:invoke name="apply_patch">\n',
        f'<atem:parameter name="input">{RAW_PATCH}</atem:parameter>\n',
        "</atem:invoke>\n",
        "</atem:function_calls>",
    ]

    for index, chunk in enumerate(chunks, start=1):
        current_text = previous_text + chunk
        current_ids = previous_ids + [index]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=[index],
            request=request,
        )
        if delta is not None:
            messages.append(delta)
        previous_text = current_text
        previous_ids = current_ids

    content = "".join(message.content or "" for message in messages)
    tool_deltas = [
        tool_call for message in messages for tool_call in message.tool_calls
    ]
    assert content == ""
    assert len(tool_deltas) == 1
    assert tool_deltas[0].function is not None
    assert tool_deltas[0].function.name == "apply_patch"
    assert tool_deltas[0].function.arguments == RAW_PATCH


def test_muse_composed_parser_preserves_reasoning_and_raw_custom_call():
    request = _apply_patch_request()
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(_DummyTokenizer(), request.tools)
    output = " to=self<|message|>Prepare the requested patch.<|eom|>" + _atem_call(
        "apply_patch", {"input": RAW_PATCH}
    )

    reasoning, content, tool_calls = parser.parse(
        output,
        request,
        enable_auto_tools=True,
    )
    response_output = build_response_output_items(
        reasoning=reasoning,
        content=content,
        tool_calls=tool_calls,
        tools=request.tools,
    )

    assert reasoning == "Prepare the requested patch."
    assert content is None
    assert len(response_output) == 2
    assert response_output[0].type == "reasoning"
    custom_call = response_output[1]
    assert isinstance(custom_call, ResponseCustomToolCall)
    assert custom_call.name == "apply_patch"
    assert custom_call.input == RAW_PATCH


def test_muse_composed_streaming_emits_responses_custom_tool_events():
    request = _apply_patch_request()
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(_DummyTokenizer(), request.tools)
    processor = SimpleStreamingEventProcessor(tools=request.tools)
    output = " to=self<|message|>Prepare the requested patch.<|eom|>" + _atem_call(
        "apply_patch", {"input": RAW_PATCH}
    )

    delta = parser.parse_delta(
        delta_text=output,
        delta_token_ids=[2],
        request=request,
        prompt_token_ids=[],
        finished=True,
    )

    assert delta is not None
    assert delta.reasoning == "Prepare the requested patch."
    assert delta.content is None
    assert delta.tool_calls[0].function is not None
    assert delta.tool_calls[0].function.name == "apply_patch"
    assert delta.tool_calls[0].function.arguments == RAW_PATCH

    events = []
    for atomic_delta in split_delta(delta):
        target, tool_call = processor.resolve_target_state(atomic_delta)
        if processor.needs_transition(target, tool_call):
            events.extend(processor.close_current())
            events.extend(processor.open(target, tool_call))
        events.extend(
            processor.emit_delta(atomic_delta, output=None)  # type: ignore[arg-type]
        )
    events.extend(processor.close_current())

    event_types = [event.type for event in events]
    assert "response.reasoning_text.delta" in event_types
    assert "response.custom_tool_call_input.delta" in event_types
    assert "response.custom_tool_call_input.done" in event_types
    custom_deltas = [
        event.delta
        for event in events
        if event.type == "response.custom_tool_call_input.delta"
    ]
    assert custom_deltas == [RAW_PATCH]


def test_muse_composed_streaming_holds_partial_recipient_prefix():
    request = _apply_patch_request()
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(_DummyTokenizer(), request.tools)

    assert (
        parser.parse_delta(
            delta_text=" to=apply_patch",
            delta_token_ids=[7],
            request=request,
            prompt_token_ids=[],
            finished=False,
        )
        is None
    )

    # A truncated recipient prefix is protocol framing, not fallback text.
    assert (
        parser.parse_delta(
            delta_text="",
            delta_token_ids=[],
            request=request,
            finished=True,
        )
        is None
    )

    parser = parser_cls(_DummyTokenizer(), request.tools)
    assert (
        parser.parse_delta(
            delta_text=" to=apply_patch",
            delta_token_ids=[7],
            request=request,
            prompt_token_ids=[],
            finished=False,
        )
        is None
    )

    output = _atem_call("apply_patch", {"input": RAW_PATCH})
    delta = parser.parse_delta(
        delta_text=output[len("<|start|>assistant to=apply_patch") :],
        delta_token_ids=[1, 2],
        request=request,
        finished=True,
    )

    assert delta is not None
    assert delta.content is None
    assert delta.reasoning is None
    assert delta.tool_calls[0].function is not None
    assert delta.tool_calls[0].function.arguments == RAW_PATCH


def test_muse_function_call_coerces_atem_scalars_from_schema():
    request = ResponsesRequest.model_validate(
        {
            "model": "Muse-Glimmer-30B",
            "input": "Call it.",
            "tools": [
                {
                    "type": "function",
                    "name": "configure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                            "labels": {"type": "array"},
                            "note": {"type": "string"},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)

    result = parser.extract_tool_calls(
        _atem_call(
            "configure",
            {
                "count": "3",
                "enabled": "true",
                "labels": '["a", "b"]',
                "note": " spaced value ",
            },
        ),
        request,
    )

    assert json.loads(result.tool_calls[0].function.arguments) == {
        "count": 3,
        "enabled": True,
        "labels": ["a", "b"],
        "note": " spaced value ",
    }


def test_muse_custom_continuation_stays_function_shaped_for_chat_template():
    request = _apply_patch_request()
    messages = construct_chat_messages_with_tool_call(
        [
            ResponseCustomToolCall(
                type="custom_tool_call",
                id="item_1",
                call_id="call_1",
                name="apply_patch",
                input=RAW_PATCH,
            ),
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "Done!",
            },
        ]
    )
    tools = adapt_custom_tools_for_chat_template(
        construct_tool_dicts(request.tools, request.tool_choice)
    )
    messages = adapt_custom_tool_calls_for_chat_template(messages)

    assert tools is not None
    assert tools[0]["function"]["parameters"]["required"] == ["input"]
    call = messages[0]["tool_calls"][0]
    assert call["function"] == {
        "name": "apply_patch",
        "arguments": {"input": RAW_PATCH},
    }
    assert messages[1]["tool_call_id"] == "call_1"


def test_custom_continuation_adapts_history_when_tool_choice_is_none():
    request = _apply_patch_request(tool_choice="none")
    messages = construct_chat_messages_with_tool_call(
        [
            ResponseCustomToolCall(
                type="custom_tool_call",
                id="item_1",
                call_id="call_1",
                name="apply_patch",
                input=RAW_PATCH,
            ),
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "Done!",
            },
        ]
    )
    tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
    assert tool_dicts is None

    adapted_messages, adapted_tools = _adapt_responses_custom_tool_rendering(
        request,
        messages,
        tool_dicts,
    )

    assert adapted_tools is None
    call = adapted_messages[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"] == {
        "name": "apply_patch",
        "arguments": {"input": RAW_PATCH},
    }


def test_muse_tool_choice_none_strips_continuation_envelopes():
    request = _apply_patch_request(tool_choice="none", stream=True)
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(_DummyTokenizer(), request.tools)
    output = (
        "We applied patch. Done.<|eom|>"
        "<|start|>assistant to=user<|message|>"
        "Patch applied successfully.<|eot|>"
    )

    delta = parser.parse_delta(
        delta_text=output,
        delta_token_ids=[2],
        request=request,
        # A continuation prompt already contains completed assistant segments,
        # so the unified parser starts in its post-reasoning/tool phase.
        prompt_token_ids=[1, 2],
        finished=True,
    )

    assert delta is not None
    assert delta.tool_calls == []
    assert delta.content == "We applied patch. Done.Patch applied successfully."
    assert "<|" not in delta.content


def test_muse_tool_choice_none_strips_split_continuation_envelopes():
    request = _apply_patch_request(tool_choice="none", stream=True)
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse_glimmer",
        reasoning_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    cases = [
        (
            (
                "We applied patch. Done.<|eom|>"
                "<|start|>assistant to=user<|message|>"
                "Patch applied successfully.<|eot|>"
            ),
            "We applied patch. Done.Patch applied successfully.",
        ),
        (
            (
                " to=self<|message|>Check the result.<|eom|>"
                "<|start|>assistant to=user<|message|>"
                "Patch applied successfully.<|eot|>"
            ),
            "Patch applied successfully.",
        ),
    ]

    for output, expected in cases:
        for chunk_size in range(1, len(output) + 1):
            parser = parser_cls(_DummyTokenizer(), request.tools)
            content: list[str] = []
            for offset in range(0, len(output), chunk_size):
                chunk = output[offset : offset + chunk_size]
                delta = parser.parse_delta(
                    delta_text=chunk,
                    delta_token_ids=[3],
                    request=request,
                    prompt_token_ids=[1, 2] if offset == 0 else None,
                    finished=offset + len(chunk) == len(output),
                )
                if delta is not None:
                    assert delta.tool_calls == []
                    if delta.content:
                        content.append(delta.content)

            streamed = "".join(content)
            assert streamed == expected, (chunk_size, streamed)
            assert "<|" not in streamed


def test_muse_adjust_request_keeps_atem_markers_visible():
    request = _apply_patch_request()
    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)

    adjusted = parser.adjust_request(request)

    assert adjusted.skip_special_tokens is False
    if hasattr(adjusted, "spaces_between_special_tokens"):
        assert adjusted.spaces_between_special_tokens is False


def test_responses_reasoning_none_disables_both_template_thinking_switches():
    request = ResponsesRequest.model_validate(
        {
            **_apply_patch_request().model_dump(),
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "reasoning": {"effort": "none"},
        }
    )

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    ).with_defaults({"thinking": True, "enable_thinking": True})

    assert chat_params.chat_template_kwargs["reasoning_effort"] == "none"
    assert chat_params.chat_template_kwargs["reasoning_strength"] == "none"
    assert chat_params.chat_template_kwargs["thinking"] is False
    assert chat_params.chat_template_kwargs["enable_thinking"] is False

    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)
    tag = parser.get_structural_tag(request)
    assert tag is not None
    assert tag.format.elements[0].value == " to=apply_patch<|message|>"
    Grammar.from_structural_tag(tag)


def test_muse_default_reasoning_keeps_self_recipient_available():
    request = _apply_patch_request()
    parser = MuseGlimmerToolParser(_DummyTokenizer(), request.tools)

    tag = parser.get_structural_tag(request)

    assert tag is not None
    assert "to=self" not in tag.model_dump_json()
    Grammar.from_structural_tag(tag)
