# SPDX-License-Identifier: Apache-2.0
"""End-to-end replay regression with the optional Qwen tool template asset."""

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from tests.entrypoints.test_responses_assistant_history import (
    history_with_trailing_item,
    output_message,
)
from vllm.entrypoints.chat_utils import _postprocess_messages
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages

TEMPLATE = (
    Path(__file__).parents[2] / "slimserve/chat_templates/qwen38_tool_calling.jinja"
)


def render_history(messages):
    # This is the same argument decoding stage used by the chat renderer.
    _postprocess_messages(messages)
    rendered, _ = render_jinja_template(
        conversations=[messages],
        tools=[
            {
                "type": "function",
                "function": {"name": "inventory", "parameters": {"type": "object"}},
            }
        ],
        chat_template=TEMPLATE.read_text(),
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return rendered[0]


pytestmark = pytest.mark.skipif(
    not TEMPLATE.is_file(), reason="Requires the Qwen tool template asset from PR #61"
)


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "output_text", "text": "", "annotations": []}],
        [{"type": "refusal", "refusal": ""}],
    ],
)
@pytest.mark.parametrize("status", ["in_progress", "completed", "incomplete"])
def test_empty_trailing_item_renders_with_its_tool_turn(content, status):
    item = {**output_message(content), "id": "msg_empty", "status": status}
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": history_with_trailing_item(item),
        }
    )
    messages = construct_input_messages(request_input=request.input)
    prompt = render_history(messages)
    assert prompt.count("<function=inventory>") == 2
    assert "Three books." in prompt and "Four books." in prompt
    assert "Checking inventory. I cannot edit stock." in prompt


@pytest.mark.parametrize("content", ["", [], [{"type": "output_text", "text": ""}]])
def test_raw_empty_trailing_item_renders_with_its_tool_turn(content):
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": history_with_trailing_item(output_message([])),
        }
    )
    request.input[4] = {"role": "assistant", "content": content}
    messages = construct_input_messages(request_input=request.input)
    assert render_history(messages).count("<function=inventory>") == 2


def test_orphan_tool_result_is_still_rejected_by_template():
    messages = construct_input_messages(
        request_input=[
            {"role": "user", "content": "Check inventory."},
            {"role": "assistant", "content": ""},
            {
                "type": "function_call_output",
                "call_id": "missing",
                "output": "Three books.",
            },
        ]
    )
    with pytest.raises(Exception, match="must immediately follow"):
        render_history(messages)
