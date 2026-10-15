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


def history_call(index):
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": "inventory",
        "arguments": '{"section":"books"}',
        "status": "completed",
    }


def history_with_trailing_item(item):
    return [
        {"role": "user", "content": "Check inventory."},
        history_call(1),
        output_message(
            [
                {
                    "type": "output_text",
                    "text": "Checking inventory. ",
                    "annotations": [],
                },
                {"type": "refusal", "refusal": "I cannot edit stock."},
            ]
        ),
        history_call(2),
        item,
        {"type": "function_call_output", "call_id": "call_1", "output": "Three books."},
        {"type": "function_call_output", "call_id": "call_2", "output": "Four books."},
        {"role": "user", "content": "Summarize."},
    ]


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "output_text", "text": "", "annotations": []}],
        [{"type": "refusal", "refusal": ""}],
    ],
)
@pytest.mark.parametrize("status", ["in_progress", "completed", "incomplete"])
def test_empty_trailing_output_keeps_tool_results_attached(content, status):
    trailing = {**output_message(content), "id": "msg_empty", "status": status}
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": history_with_trailing_item(trailing),
        }
    )
    assert isinstance(request.input[4], ResponseOutputMessage)
    messages = construct_input_messages(request_input=request.input)
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert messages[1]["content"] == "Checking inventory. I cannot edit stock."
    assert [call["id"] for call in messages[1]["tool_calls"]] == ["call_1", "call_2"]


@pytest.mark.parametrize("content", ["", [], [{"type": "output_text", "text": ""}]])
def test_raw_dict_empty_trailing_message_does_not_detach_tools(content):
    # The converter also accepts dictionary history; keep this fallback aligned
    # with the typed Responses API path without changing nonempty boundaries.
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": history_with_trailing_item(output_message([])),
        }
    )
    request.input[4] = {"role": "assistant", "content": content}
    messages = construct_input_messages(request_input=request.input)
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            [{"type": "output_text", "text": "Another turn.", "annotations": []}],
            "Another turn.",
        ),
        ([{"type": "refusal", "refusal": "Cannot edit."}], "Cannot edit."),
        ([{"type": "output_text", "text": " ", "annotations": []}], " "),
    ],
)
def test_nonempty_followup_retains_separate_assistant_boundary(content, expected):
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                output_message(
                    [{"type": "output_text", "text": "First.", "annotations": []}]
                ),
                {**output_message(content), "id": "msg_next"},
            ],
        }
    )
    assert construct_input_messages(request_input=request.input) == [
        {"role": "assistant", "content": "First."},
        {"role": "assistant", "content": expected},
    ]


def test_empty_message_does_not_fabricate_a_call_for_orphan_tool_result():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Check inventory."},
                output_message([]),
                {
                    "type": "function_call_output",
                    "call_id": "missing_call",
                    "output": "Three books.",
                },
            ],
        }
    )
    messages = construct_input_messages(request_input=request.input)
    assert messages[1] == {"role": "assistant", "content": ""}
    assert "tool_calls" not in messages[1]
    assert messages[2]["tool_call_id"] == "missing_call"


# Content renders before tool calls, as in the real templates, so a
# continuation cut placed inside a tool-call turn would drop its calls.
CONTINUATION_TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.reasoning %}<think>{{ m.reasoning }}</think>{% endif %}"
    "{{ m.content or '' }}"
    "{% for c in m.tool_calls or [] %}<call>{{ c.function.name }}</call>"
    "{% endfor %}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def render_request(request):
    from transformers.utils.chat_template_utils import render_jinja_template

    params = request.build_chat_params(None, "auto")
    kwargs = params.chat_template_kwargs
    rendered, _ = render_jinja_template(
        conversations=[construct_input_messages(request_input=request.input)],
        chat_template=CONTINUATION_TEMPLATE,
        add_generation_prompt=kwargs["add_generation_prompt"],
        continue_final_message=kwargs["continue_final_message"],
    )
    return kwargs["continue_final_message"], rendered[0]


@pytest.mark.parametrize("content", [[], [{"type": "output_text", "text": ""}]])
def test_empty_in_progress_final_message_does_not_resume_tool_turn(content):
    # The empty item is the continuation anchor, not a stray trailing item:
    # it must not be absorbed into the tool-call turn, or the continuation
    # would resume that turn and cut its calls out of the prompt.
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Check inventory."},
                history_call(1),
                {
                    **output_message(content),
                    "id": "msg_partial",
                    "status": "in_progress",
                },
            ],
        }
    )
    messages = construct_input_messages(request_input=request.input)
    assert [m["role"] for m in messages] == ["user", "assistant", "assistant"]
    assert [c["id"] for c in messages[1]["tool_calls"]] == ["call_1"]
    assert messages[2] == {"role": "assistant", "content": ""}

    continued, prompt = render_request(request)
    assert continued
    assert prompt == (
        "<|user|>Check inventory.<|end|>"
        "<|assistant|><call>inventory</call><|end|>"
        "<|assistant|>"
    )


def test_nonempty_in_progress_final_message_after_tool_call_keeps_call():
    # A partial assistant message after a function call is a new turn being
    # continued; merging it into the tool-call turn would place the
    # continuation cut before the calls and drop them from the prompt.
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Check inventory."},
                history_call(1),
                {
                    **output_message(
                        [{"type": "output_text", "text": "Partial", "annotations": []}]
                    ),
                    "id": "msg_partial",
                    "status": "in_progress",
                },
            ],
        }
    )
    messages = construct_input_messages(request_input=request.input)
    assert [m["role"] for m in messages] == ["user", "assistant", "assistant"]
    assert [c["id"] for c in messages[1]["tool_calls"]] == ["call_1"]
    assert messages[2] == {"role": "assistant", "content": "Partial"}

    continued, prompt = render_request(request)
    assert continued
    assert prompt == (
        "<|user|>Check inventory.<|end|>"
        "<|assistant|><call>inventory</call><|end|>"
        "<|assistant|>Partial"
    )


def test_empty_in_progress_final_message_does_not_resume_finished_message():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Hello."},
                output_message(
                    [{"type": "output_text", "text": "Done.", "annotations": []}]
                ),
                {**output_message([]), "id": "msg_partial", "status": "in_progress"},
            ],
        }
    )
    continued, prompt = render_request(request)
    assert continued
    assert prompt == "<|user|>Hello.<|end|><|assistant|>Done.<|end|><|assistant|>"


def test_in_progress_final_message_still_joins_its_reasoning():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Hello."},
                {
                    "id": "rs_partial",
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Greet."}],
                },
                {
                    **output_message(
                        [{"type": "output_text", "text": "Hi", "annotations": []}]
                    ),
                    "id": "msg_partial",
                    "status": "in_progress",
                },
            ],
        }
    )
    continued, prompt = render_request(request)
    assert continued
    assert prompt == "<|user|>Hello.<|end|><|assistant|><think>Greet.</think>Hi"


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
