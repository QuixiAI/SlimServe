# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import CustomTool
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    ConstStringFormat,
    SequenceFormat,
    TagFormat,
)

from vllm.parser.engine.registered_adapters import DeepSeekV4ParserToolAdapter
from vllm.tool_parsers.structural_tag_registry import (
    custom_tool_as_function_schema,
    get_custom_tool_input_format,
    register_custom_tool_format,
    replace_custom_tool_payloads,
)


@register_custom_tool_format(
    "deepseek_v4",
    as_function_tool=custom_tool_as_function_schema,
)
def _apply_deepseek_v4_custom_tool_format(
    structural_tag: StructuralTag,
    custom_tools: dict[str, CustomTool],
) -> None:
    """Map raw custom input into DeepSeek V4's one-parameter DSML call."""
    invoke_prefix = '<｜DSML｜invoke name="'
    parameter_begin = '<｜DSML｜parameter name="input" string="true">'
    parameter_end = "</｜DSML｜parameter>\n"

    def match_tool_name(tag: TagFormat) -> str | None:
        begin = tag.begin
        if isinstance(begin, str) and begin.startswith(invoke_prefix):
            return begin[len(invoke_prefix) :].split('"', 1)[0]
        return None

    def make_content(custom_tool: CustomTool) -> SequenceFormat:
        return SequenceFormat(
            elements=[
                ConstStringFormat(value=parameter_begin),
                get_custom_tool_input_format(custom_tool),
                ConstStringFormat(value=parameter_end),
            ]
        )

    replace_custom_tool_payloads(
        structural_tag,
        custom_tools,
        match_tool_name=match_tool_name,
        make_content=make_content,
    )


class DeepSeekV4EngineToolParser(DeepSeekV4ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "deepseek_v4"
