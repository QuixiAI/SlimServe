# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from openai.types.responses import FunctionTool

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

pytestmark = pytest.mark.skip_global_cleanup

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer"}},
    "required": ["count"],
    "additionalProperties": False,
}


def _qwen_content_schema(tool) -> tuple[dict, dict]:
    tag = get_model_structural_tag(
        model="qwen_3_coder",
        tools=[tool],
        tool_choice="required",
        reasoning=False,
    )
    assert tag is not None
    format_ = tag.model_dump()["format"]
    return format_, format_["tags"][0]["content"]["json_schema"]


@pytest.mark.parametrize("strict", [None, False])
@pytest.mark.parametrize("protocol", ["responses", "chat"])
def test_qwen_non_strict_tools_require_a_json_object_but_not_the_schema(
    strict: bool | None,
    protocol: str,
):
    if protocol == "responses":
        tool = FunctionTool(
            type="function",
            name="record_count",
            parameters=TOOL_SCHEMA,
            strict=strict,
        )
    else:
        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "record_count",
                "parameters": TOOL_SCHEMA,
                "strict": strict,
            },
        )

    format_, content_schema = _qwen_content_schema(tool)

    assert content_schema == {"type": "object", "additionalProperties": True}
    assert format_["at_least_one"] is True
    assert format_["stop_after_first"] is False


@pytest.mark.parametrize("protocol", ["responses", "chat"])
def test_qwen_explicit_strict_tool_enforces_its_schema(protocol: str):
    if protocol == "responses":
        tool = FunctionTool(
            type="function",
            name="record_count",
            parameters=TOOL_SCHEMA,
            strict=True,
        )
    else:
        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "record_count",
                "parameters": TOOL_SCHEMA,
                "strict": True,
            },
        )

    format_, content_schema = _qwen_content_schema(tool)

    assert content_schema == TOOL_SCHEMA
    assert format_["at_least_one"] is True


def test_qwen_auto_non_strict_tool_constrains_triggered_arguments():
    tool = FunctionTool(
        type="function",
        name="record_count",
        parameters=TOOL_SCHEMA,
        strict=None,
    )

    tag = get_model_structural_tag(
        model="qwen_3_coder",
        tools=[tool],
        tool_choice="auto",
        reasoning=False,
    )

    assert tag is not None
    format_ = tag.model_dump()["format"]
    assert format_["type"] == "triggered_tags"
    assert format_["tags"][0]["content"]["json_schema"] == {
        "type": "object",
        "additionalProperties": True,
    }
    assert format_["at_least_one"] is False


def test_qwen_generic_object_grammar_emits_xml_parameters_the_parser_accepts():
    import json
    from unittest.mock import MagicMock

    import xgrammar as xgr

    from vllm.parser.qwen3 import Qwen3Parser

    response_tool = FunctionTool(
        type="function",
        name="record_count",
        parameters=TOOL_SCHEMA,
        strict=None,
    )
    tag = get_model_structural_tag(
        model="qwen_3_coder",
        tools=[response_tool],
        tool_choice="required",
        reasoning=False,
    )
    assert tag is not None

    # Compile with a portable byte-level tokenizer and feed every token
    # through the real matcher. This catches XGrammar's default behavior for
    # ``additionalProperties``: without an explicit true value, the generic
    # object compiles to an empty-only Qwen argument body.
    tokenizer_info = xgr.TokenizerInfo(
        encoded_vocab=[bytes([i]) for i in range(256)],
        vocab_type=xgr.VocabType.RAW,
        vocab_size=256,
        stop_token_ids=[],
    )
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
    compiled = compiler.compile_structural_tag(tag.model_dump_json())
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    xml = (
        "<tool_call>\n<function=record_count>\n"
        "<parameter=count>7</parameter>\n"
        '<parameter=metadata>{"source":"fixture"}</parameter>\n'
        "<parameter=enabled>true</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert all(matcher.accept_token(token) for token in xml.encode())
    assert matcher.is_completed()

    chat_tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "record_count",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "metadata": {"type": "object", "additionalProperties": True},
                    "enabled": {"type": "boolean"},
                },
            },
        },
    )
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 50,
        "</think>": 51,
        "<tool_call>": 60,
        "</tool_call>": 61,
    }
    parser = Qwen3Parser(
        tokenizer,
        [chat_tool],
        chat_template_kwargs={"enable_thinking": False},
    )
    request = MagicMock(tools=[chat_tool], tool_choice="required")
    output = parser.extract_tool_calls_from_content(xml, request)

    assert output.tools_called is True
    assert json.loads(output.tool_calls[0].function.arguments) == {
        "count": 7,
        "metadata": {"source": "fixture"},
        "enabled": True,
    }
