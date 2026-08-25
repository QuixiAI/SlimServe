# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from openai.types.responses import ResponseCustomToolCall
from xgrammar import Grammar

from vllm.entrypoints.chat_utils import (
    adapt_custom_tool_calls_for_chat_template,
    adapt_custom_tools_for_chat_template,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
)
from vllm.entrypoints.openai.responses.utils import (
    build_response_output_items,
    construct_chat_messages_with_tool_call,
    construct_tool_dicts,
)
from vllm.tokenizers.deepseek_v4_encoding import (
    encode_arguments_to_dsml,
    tool_calls_from_openai_format,
    tools_from_openai_format,
)
from vllm.tool_parsers.deepseekv4_engine_tool_parser import (
    DeepSeekV4EngineToolParser,
)
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

APPLY_PATCH_LARK = r"""start: begin_patch hunk+ end_patch
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
RAW_PATCH = (
    "*** Begin Patch\n"
    "*** Update File: greeting.txt\n"
    "@@\n"
    "-Hello, world!\n"
    "+Hello, SlimServe!\n"
    "*** End Patch\n"
)


def custom_request(**updates) -> ResponsesRequest:
    data = {
        "model": "DeepSeek-V4-Flash",
        "input": (
            "greeting.txt currently contains exactly `Hello, world!\\n`. "
            "Use apply_patch to make it contain exactly "
            "`Hello, SlimServe!\\n`."
        ),
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
                    "syntax": "lark",
                    "definition": APPLY_PATCH_LARK,
                },
            }
        ],
        "tool_choice": "required",
    }
    data.update(updates)
    return ResponsesRequest.model_validate(data)


def _walk_format(format_dict):
    yield format_dict
    for key in ("content", "elements", "tags"):
        child = format_dict.get(key)
        if isinstance(child, dict):
            yield from _walk_format(child)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, dict):
                    yield from _walk_format(item)


def test_protocol_accepts_named_custom_tool_choice():
    request = custom_request(tool_choice={"type": "custom", "name": "apply_patch"})
    assert request.tool_choice.type == "custom"
    assert request.tool_choice.name == "apply_patch"


def test_protocol_rejects_missing_named_custom_tool():
    with pytest.raises(Exception, match="Named tool choice not found"):
        custom_request(tool_choice={"type": "custom", "name": "missing"})


def test_generic_tool_construction_preserves_custom_declaration():
    request = custom_request()
    tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)

    assert tool_dicts == [
        {
            "type": "custom",
            "name": "apply_patch",
            "description": (
                "Use the `apply_patch` tool to edit files. This is a "
                "FREEFORM tool, so do not wrap the patch in JSON."
            ),
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": APPLY_PATCH_LARK,
            },
        }
    ]


def test_deepseek_encoder_adapts_custom_tool_to_dsml_input_parameter():
    request = custom_request()
    tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
    assert tool_dicts is not None

    deepseek_tools = tools_from_openai_format(tool_dicts)
    assert deepseek_tools[0]["name"] == "apply_patch"
    assert deepseek_tools[0]["parameters"] == {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
        "additionalProperties": False,
    }
    assert "lark grammar" in deepseek_tools[0]["description"]


def test_deepseek_encoder_adapts_custom_tool_continuation_to_dsml():
    calls = tool_calls_from_openai_format(
        [
            {
                "id": "call_1",
                "type": "custom",
                "name": "apply_patch",
                "input": RAW_PATCH,
            }
        ]
    )

    assert calls == [{"name": "apply_patch", "arguments": {"input": RAW_PATCH}}]
    assert encode_arguments_to_dsml(calls[0]) == (
        f'<｜DSML｜parameter name="input" string="true">{RAW_PATCH}</｜DSML｜parameter>'
    )


def test_deepseek_lark_grammar_only_replaces_dsml_input_region():
    request = custom_request()
    assert DeepSeekV4EngineToolParser.structural_tag_model == "deepseek_v4"

    structural_tag = get_model_structural_tag(
        "deepseek_v4", request.tools, request.tool_choice, reasoning=False
    )
    assert structural_tag is not None
    formats = list(_walk_format(structural_tag.model_dump()["format"]))

    constants = [item["value"] for item in formats if item["type"] == "const_string"]
    grammars = [item["grammar"] for item in formats if item["type"] == "grammar"]
    assert '<｜DSML｜parameter name="input" string="true">' in constants
    assert "</｜DSML｜parameter>\n" in constants
    assert len(grammars) == 1
    assert "root ::= start" in grammars[0]
    assert "begin_patch ::=" in grammars[0]
    assert 'LF ::= "\\n"' in grammars[0]
    Grammar.from_structural_tag(structural_tag)


def test_deepseek_regex_grammar_only_replaces_dsml_input_region():
    request = custom_request(
        tools=[
            {
                "type": "custom",
                "name": "digits",
                "format": {
                    "type": "grammar",
                    "syntax": "regex",
                    "definition": r"[0-9]+",
                },
            }
        ]
    )

    structural_tag = get_model_structural_tag(
        "deepseek_v4", request.tools, request.tool_choice, reasoning=False
    )
    assert structural_tag is not None
    formats = list(_walk_format(structural_tag.model_dump()["format"]))
    regexes = [item["pattern"] for item in formats if item["type"] == "regex"]
    assert regexes == [r"[0-9]+"]


@pytest.mark.parametrize(
    ("model", "parser_module", "expected_constants"),
    [
        (
            "qwen_3_coder",
            "vllm.tool_parsers.qwen3_engine_tool_parser",
            ("<parameter=input>", "</parameter>"),
        ),
    ],
)
def test_other_model_formats_constrain_only_custom_payload(
    model, parser_module, expected_constants
):
    __import__(parser_module)
    request = custom_request()

    structural_tag = get_model_structural_tag(
        model, request.tools, request.tool_choice, reasoning=False
    )
    assert structural_tag is not None
    dumped = structural_tag.model_dump()["format"]
    formats = list(_walk_format(dumped))

    constants = [item["value"] for item in formats if item["type"] == "const_string"]
    grammars = [item["grammar"] for item in formats if item["type"] == "grammar"]
    for expected in expected_constants:
        assert expected in constants
    assert len(grammars) == 1
    Grammar.from_structural_tag(structural_tag)


def test_generic_chat_template_adapter_function_shapes_custom_tools_and_history():
    request = custom_request()
    tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)
    adapted_tools = adapt_custom_tools_for_chat_template(tool_dicts)
    assert adapted_tools is not None
    assert adapted_tools[0]["type"] == "function"
    assert adapted_tools[0]["function"]["parameters"]["required"] == ["input"]

    adapted_messages = adapt_custom_tool_calls_for_chat_template(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "custom",
                        "name": "apply_patch",
                        "input": RAW_PATCH,
                    }
                ],
            }
        ]
    )
    call = adapted_messages[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"] == {
        "name": "apply_patch",
        "arguments": {"input": RAW_PATCH},
    }


def test_non_streaming_custom_tool_output_and_continuation_round_trip():
    request = custom_request()
    output = build_response_output_items(
        reasoning=None,
        content=None,
        tool_calls=[FunctionCall(name="apply_patch", arguments=RAW_PATCH)],
        tools=request.tools,
    )

    assert len(output) == 1
    call = output[0]
    assert isinstance(call, ResponseCustomToolCall)
    assert call.type == "custom_tool_call"
    assert call.input == RAW_PATCH

    messages = construct_chat_messages_with_tool_call(
        [
            call,
            {
                "type": "custom_tool_call_output",
                "call_id": call.call_id,
                "output": "8",
            },
        ]
    )
    assert messages[0]["tool_calls"][0] == {
        "id": call.call_id,
        "type": "custom",
        "name": "apply_patch",
        "input": RAW_PATCH,
    }
    assert messages[1]["role"] == "tool"
    assert messages[1]["content"] == "8"


def test_streaming_custom_tool_events_use_raw_input_event_types():
    request = custom_request()
    processor = SimpleStreamingEventProcessor(tools=request.tools)
    start = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                type="function",
                function=DeltaFunctionCall(name="apply_patch"),
            )
        ]
    )
    target, tool_call = processor.resolve_target_state(start)
    opened = processor.open(target, tool_call)
    assert opened[0].item.type == "custom_tool_call"

    delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments=RAW_PATCH),
            )
        ]
    )
    events = processor.emit_delta(delta, output=None)  # type: ignore[arg-type]
    assert events[0].type == "response.custom_tool_call_input.delta"
    assert events[0].delta == RAW_PATCH

    done = processor.close_current()
    assert done[0].type == "response.custom_tool_call_input.done"
    assert done[0].input == RAW_PATCH
    assert done[1].item.type == "custom_tool_call"
    assert done[1].item.input == RAW_PATCH


def test_streaming_empty_custom_input_still_emits_done_event():
    request = custom_request()
    processor = SimpleStreamingEventProcessor(tools=request.tools)
    start = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                type="function",
                function=DeltaFunctionCall(name="apply_patch"),
            )
        ]
    )
    target, tool_call = processor.resolve_target_state(start)
    processor.open(target, tool_call)

    done = processor.close_current()
    assert done[0].type == "response.custom_tool_call_input.done"
    assert done[0].input == ""
