# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Responses output must remain usable as next-turn assistant history."""

import pytest
from openai.types.responses import ResponseOutputMessage

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages

HISTORY_CONTENT = [
    pytest.param([], "", id="empty"),
    pytest.param(
        [{"type": "output_text", "text": "", "annotations": []}],
        "",
        id="empty-text",
    ),
    pytest.param(
        [
            {"type": "output_text", "text": "First. ", "annotations": []},
            {"type": "output_text", "text": "Second.", "annotations": []},
        ],
        "First. Second.",
        id="multipart-text",
    ),
    pytest.param(
        [{"type": "refusal", "refusal": "I cannot perform that action."}],
        "I cannot perform that action.",
        id="refusal",
    ),
    pytest.param(
        [
            {"type": "output_text", "text": "First. ", "annotations": []},
            {"type": "refusal", "refusal": "I cannot do that. "},
            {"type": "output_text", "text": "Alternative.", "annotations": []},
        ],
        "First. I cannot do that. Alternative.",
        id="mixed-text-refusal",
    ),
]


def output_message(content):
    return {
        "id": "msg_history",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": content,
    }


@pytest.mark.parametrize("content, expected", HISTORY_CONTENT)
def test_next_turn_replays_all_assistant_content(content, expected):
    # Go through the request schema, as the HTTP endpoint does, rather than
    # handing a private converter a specially constructed object.
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Start."},
                output_message(content),
                {"role": "user", "content": "Continue."},
            ],
        }
    )
    assert isinstance(request.input[1], ResponseOutputMessage)
    assert construct_input_messages(request_input=request.input) == [
        {"role": "user", "content": "Start."},
        {"role": "assistant", "content": expected},
        {"role": "user", "content": "Continue."},
    ]


@pytest.mark.parametrize("content, expected", HISTORY_CONTENT)
def test_stored_response_preserves_all_assistant_content(content, expected):
    previous = ResponseOutputMessage.model_validate(output_message(content))
    assert construct_input_messages(
        request_input="Continue.",
        prev_msg=[{"role": "user", "content": "Start."}],
        prev_response_output=[previous],
    ) == [
        {"role": "user", "content": "Start."},
        {"role": "assistant", "content": expected},
        {"role": "user", "content": "Continue."},
    ]


@pytest.mark.parametrize("content, expected", HISTORY_CONTENT)
@pytest.mark.parametrize("message_before_call", [True, False])
def test_assistant_content_merges_with_reasoning_and_tool_calls(
    content, expected, message_before_call
):
    reasoning = {
        "id": "rs_history",
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Check the inventory."}],
    }
    call = {
        "id": "fc_history",
        "type": "function_call",
        "call_id": "call_inventory",
        "name": "inventory",
        "arguments": '{"section":"books"}',
        "status": "completed",
    }
    message = output_message(content)
    output_items = [message, call] if message_before_call else [call, message]
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                reasoning,
                *output_items,
                {
                    "type": "function_call_output",
                    "call_id": "call_inventory",
                    "output": '{"count":3}',
                },
                {"role": "user", "content": "Summarize."},
            ],
        }
    )
    messages = construct_input_messages(request_input=request.input)
    assert messages == [
        {
            "role": "assistant",
            "reasoning": "Check the inventory.",
            "content": expected,
            "tool_calls": [
                {
                    "id": "call_inventory",
                    "type": "function",
                    "function": {
                        "name": "inventory",
                        "arguments": '{"section":"books"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_inventory",
            "content": '{"count":3}',
        },
        {"role": "user", "content": "Summarize."},
    ]


@pytest.mark.parametrize(
    "parts, expected",
    [
        pytest.param(
            [
                {"type": "input_text", "text": "First. "},
                {"type": "input_text", "text": "Second."},
            ],
            "First. Second.",
            id="two-part",
        ),
        pytest.param([], "", id="empty"),
    ],
)
def test_untyped_assistant_message_replays_all_parts(parts, expected):
    # An easy-input assistant message stays a dict after request validation;
    # it must replay every part, exactly like a typed output message.
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Start."},
                {"role": "assistant", "content": parts},
                {"role": "user", "content": "Continue."},
            ],
        }
    )
    assert construct_input_messages(request_input=request.input) == [
        {"role": "user", "content": "Start."},
        {"role": "assistant", "content": expected},
        {"role": "user", "content": "Continue."},
    ]


def test_untyped_assistant_message_replays_refusal_parts():
    # Callers that hand construct_input_messages raw dicts (no schema pass)
    # may carry output_text/refusal parts; those join like the typed path.
    messages = construct_input_messages(
        request_input=[
            {"role": "user", "content": "Start."},
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "First. "},
                    {"type": "refusal", "refusal": "I cannot do that."},
                ],
            },
        ]
    )
    assert messages[1] == {
        "role": "assistant",
        "content": "First. I cannot do that.",
    }


def test_untyped_assistant_part_without_text_falls_through_unchanged():
    # A part carrying neither text nor refusal (here an image) cannot be
    # flattened to a string without losing it. The whole message falls
    # through to the generic handling and is passed on unmodified, rather
    # than being truncated to the text of its first part.
    parts = [
        {"type": "output_text", "text": "See the chart: "},
        {"type": "input_image", "image_url": "https://example.com/chart.png"},
    ]
    assistant = {"role": "assistant", "content": parts}
    messages = construct_input_messages(
        request_input=[
            {"role": "user", "content": "Start."},
            assistant,
            {"role": "user", "content": "Continue."},
        ]
    )
    assert messages == [
        {"role": "user", "content": "Start."},
        {"role": "assistant", "content": parts},
        {"role": "user", "content": "Continue."},
    ]
    assert messages[1] is assistant
