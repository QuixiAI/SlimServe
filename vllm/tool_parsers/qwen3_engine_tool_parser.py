# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import CustomTool
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    ConstStringFormat,
    SequenceFormat,
    TagFormat,
)

from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter
from vllm.tool_parsers.structural_tag_registry import (
    custom_tool_as_function_schema,
    get_custom_tool_input_format,
    register_custom_tool_format,
    replace_custom_tool_payloads,
)


def _register_qwen_custom_tool_format(model: str) -> None:
    @register_custom_tool_format(
        model,
        as_function_tool=custom_tool_as_function_schema,
    )
    def apply_format(
        structural_tag: StructuralTag,
        custom_tools: dict[str, CustomTool],
    ) -> None:
        function_prefix = "<tool_call>\n<function="
        parameter_begin = "<parameter=input>"
        parameter_end = "</parameter>"

        def match_tool_name(tag: TagFormat) -> str | None:
            begin = tag.begin
            if isinstance(begin, str) and begin.startswith(function_prefix):
                return begin[len(function_prefix) :].split(">", 1)[0]
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


for _structural_tag_model in ("qwen_3_coder", "qwen_3_5"):
    _register_qwen_custom_tool_format(_structural_tag_model)


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"
