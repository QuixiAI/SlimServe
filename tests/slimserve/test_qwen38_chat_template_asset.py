# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

ASSET = (
    Path(__file__).parents[2]
    / "slimserve"
    / "chat_templates"
    / "qwen38_tool_calling.jinja"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up an identifier.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    }
]


def _render(messages, **kwargs) -> str:
    rendered, _ = render_jinja_template(
        conversations=[messages],
        tools=TOOLS,
        chat_template=ASSET.read_text(encoding="utf-8"),
        add_generation_prompt=True,
        **kwargs,
    )
    return rendered[0]


@pytest.mark.parametrize(
    ("choice", "instruction"),
    [
        ("none", "Do not call any function."),
        ("required", "must call at least one provided function"),
        (
            {"type": "function", "function": {"name": "lookup"}},
            "must call the function lookup",
        ),
    ],
)
def test_qwen38_template_renders_tool_choice_policy(choice, instruction):
    prompt = _render(
        [{"role": "user", "content": "Find it"}],
        tool_choice=choice,
        enable_thinking=True,
        reasoning_effort="low",
    )

    assert instruction in prompt
    assert "when strict is true, do not add undeclared parameters" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_qwen38_template_renders_valid_tool_history_and_results():
    prompt = _render(
        [
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "Need the record.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"id": "CVE-1"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found"},
            {"role": "user", "content": "Summarize"},
        ],
        tool_choice="auto",
        enable_thinking=True,
        reasoning_effort="high",
    )

    assert "<think>\nNeed the record.\n</think>" in prompt
    assert "<function=lookup>\n<parameter=id>\nCVE-1" in prompt
    assert (
        "<|im_start|>user\n<tool_response>\nfound\n</tool_response><|im_end|>" in prompt
    )
    assert "Reasoning effort is set to high." in prompt


def test_qwen38_template_disables_thinking_cleanly():
    prompt = _render(
        [{"role": "user", "content": "Answer"}],
        tool_choice="auto",
        enable_thinking=False,
        reasoning_effort="none",
    )

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen38_template_rejects_orphan_tool_result():
    with pytest.raises(Exception, match="must immediately follow"):
        _render(
            [
                {"role": "user", "content": "Find it"},
                {"role": "tool", "tool_call_id": "call_1", "content": "found"},
            ],
            tool_choice="auto",
            enable_thinking=True,
            reasoning_effort="low",
        )


def test_qwen38_template_preserves_image_placeholders_with_tools():
    prompt = _render(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe it"},
                    {"type": "image_url", "image_url": {"url": "fixture"}},
                ],
            }
        ],
        tool_choice="auto",
        enable_thinking=True,
        reasoning_effort="low",
    )
    assert "Describe it<|vision_start|><|image_pad|><|vision_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.parametrize(
    ("effort", "hint"),
    [
        ("minimal", "Reasoning effort is set to low."),
        ("max", "Reasoning effort is set to high."),
        ("medium", "Finish your reasoning within the available thinking budget."),
    ],
)
def test_qwen38_template_accepts_every_api_reasoning_effort(effort, hint):
    # The chat-completions protocol accepts minimal and max; rendering them
    # must map to a hint, never fail the request.
    prompt = _render(
        [{"role": "user", "content": "Answer"}],
        tool_choice="auto",
        enable_thinking=True,
        reasoning_effort=effort,
    )

    assert hint in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_qwen38_template_unknown_reasoning_effort_falls_back():
    prompt = _render(
        [{"role": "user", "content": "Answer"}],
        tool_choice="auto",
        reasoning_effort="extreme",
    )

    assert "Reasoning effort is set to high." in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_qwen38_template_explicit_enable_thinking_beats_effort_none():
    # The Qwen3 parser reads enable_thinking and expects reasoning when it is
    # true, so the template must open the think block too; the effort hint
    # is dropped.
    prompt = _render(
        [{"role": "user", "content": "Answer"}],
        tool_choice="auto",
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert "Reasoning effort is set" not in prompt
    assert "thinking budget" not in prompt


def test_qwen38_template_effort_none_alone_disables_thinking():
    prompt = _render(
        [{"role": "user", "content": "Answer"}],
        tool_choice="auto",
        reasoning_effort="none",
    )

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen38_template_lifts_inline_think_from_history_content():
    prompt = _render(
        [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "<think>\nGreet back.\n</think>\n\nHello!",
            },
            {"role": "user", "content": "Again"},
        ],
        tool_choice="auto",
        enable_thinking=True,
    )

    assert prompt.count("Greet back.") == 1
    assert "<|im_start|>assistant\n<think>\nGreet back.\n</think>\n\nHello!<|im_end|>" in (
        prompt
    )


def test_qwen38_template_explicit_reasoning_keeps_content_verbatim():
    # As stock Qwen: a message with a reasoning field is never split, so the
    # explicit reasoning renders in the think block and content is as sent.
    prompt = _render(
        [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "<think>stale</think>Hello!",
                "reasoning_content": "Greet back.",
            },
            {"role": "user", "content": "Again"},
        ],
        tool_choice="auto",
        enable_thinking=True,
    )

    assert (
        "<think>\nGreet back.\n</think>\n\n<think>stale</think>Hello!<|im_end|>"
        in prompt
    )


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_qwen38_template_empty_reasoning_keeps_literal_think_tag(field):
    prompt = _render(
        [
            {"role": "user", "content": "How do I close a block?"},
            {
                "role": "assistant",
                "content": "Use the </think> tag to close.",
                field: "",
            },
            {"role": "user", "content": "Thanks"},
        ],
        tool_choice="auto",
        enable_thinking=True,
    )

    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        "Use the </think> tag to close.<|im_end|>" in prompt
    )
