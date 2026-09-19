# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool grammar must not silently discard assistant output constraints."""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import DelegatingParser
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def make_request(protocol, tool_choice, constraint):
    function = {"name": "lookup", "parameters": {"type": "object"}}
    kwargs = {"model": "test", "tool_choice": tool_choice}
    if protocol == "chat":
        kwargs.update(
            messages=[{"role": "user", "content": "Check inventory"}],
            tools=[{"type": "function", "function": function}],
        )
        request_class = ChatCompletionRequest
    else:
        kwargs.update(input="Check inventory", tools=[{"type": "function", **function}])
        request_class = ResponsesRequest
    if constraint == "structured_outputs":
        kwargs[constraint] = {"regex": "yes|no"}
    elif constraint is not None:
        format_ = {"type": constraint}
        if constraint == "json_schema":
            schema = {"name": "answer", "schema": SCHEMA, "strict": True}
            if protocol == "chat":
                format_["json_schema"] = schema
            else:
                format_.update(schema)
        if protocol == "chat":
            kwargs["response_format"] = format_
        else:
            kwargs["text"] = {"format": format_, "verbosity": "low"}
    return request_class(**kwargs)


def make_parser():
    parser = object.__new__(DelegatingParser)
    parser._tool_parser = MagicMock()
    parser._tool_parser.structural_tag_model = "qwen_3_coder"
    parser._tool_parser.get_structural_tag.side_effect = lambda request, reasoning: (
        get_model_structural_tag(
            "qwen_3_coder", request.tools, request.tool_choice, reasoning
        )
    )
    return parser


@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("tool_choice", ["auto", "required"])
@pytest.mark.parametrize(
    "constraint", ["json_object", "json_schema", "structured_outputs"]
)
def test_conflicting_output_constraint_is_rejected_without_mutating_request(
    protocol, tool_choice, constraint
):
    request = make_request(protocol, tool_choice, constraint)
    before = request.model_dump()
    with pytest.raises(VLLMValidationError, match="cannot be combined") as exc:
        make_parser()._apply_structural_tag(request)
    assert request.model_dump() == before
    error = create_error_response(exc.value)
    assert error.error.code == 400
    assert error.error.param == (
        "structured_outputs"
        if constraint == "structured_outputs"
        else "response_format"
        if protocol == "chat"
        else "text.format"
    )


@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize(
    "constraint", ["json_object", "json_schema", "structured_outputs"]
)
def test_tool_choice_none_preserves_assistant_output_constraint(protocol, constraint):
    request = make_request(protocol, "none", constraint)
    before = request.model_dump()
    assert make_parser()._apply_structural_tag(request) is request
    assert request.model_dump() == before
    assert request.extract_structured_outputs() is not None


@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("constraint", [None, "text"])
def test_unconstrained_assistant_text_can_use_auto_tool_grammar(protocol, constraint):
    request = make_request(protocol, "auto", constraint)
    before_text = (
        request.text.model_dump() if protocol == "responses" and request.text else None
    )
    make_parser()._apply_structural_tag(request)
    assert request.structured_outputs.structural_tag is not None
    assert not request.structured_outputs._required_tool_call
    assert request.extract_structured_outputs() is request.structured_outputs
    if before_text is not None:
        assert request.text.model_dump() == before_text
