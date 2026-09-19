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
    assert "<|im_start|>user\n<tool_response>\nfound\n</tool_response><|im_end|>" in prompt
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
