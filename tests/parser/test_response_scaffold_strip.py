# SPDX-License-Identifier: Apache-2.0
"""The template's post-reasoning scaffold is stripped from returned content.

Qwen3 templates render assistant turns as ``'</think>\n\n' + content``: the
``\n\n`` is formatting the model reproduces at inference, not response. The
parser strips exactly one occurrence at the head of content — and only that:
a model that skips the scaffold, or emits a single newline, is passed
through untouched.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.qwen3 import Qwen3Parser


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


def make_request(**overrides):
    base = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest.model_validate(base)


def stream(parser, tokenizer, text, request):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    results = []
    for i, tid in enumerate(token_ids):
        results.append(
            parser.parse_delta(
                tokenizer.decode([tid]),
                [tid],
                request,
                prompt_token_ids=[] if i == 0 else None,
                finished=i == len(token_ids) - 1,
            )
        )
    return results


def collect(results):
    reasoning = "".join(r.reasoning for r in results if r and r.reasoning)
    content = "".join(r.content for r in results if r and r.content)
    return reasoning, content


# Generation starts inside <think> (the template opens it in the prompt).
OUTPUT_WITH_SCAFFOLD = "Some thoughts.\n</think>\n\nThe answer is 42."
OUTPUT_NO_SCAFFOLD = "Some thoughts.\n</think>The answer is 42."
OUTPUT_SINGLE_NEWLINE = "Some thoughts.\n</think>\nThe answer is 42."
OUTPUT_EXTRA_NEWLINE = "Some thoughts.\n</think>\n\n\nThe answer is 42."


def test_nonstreaming_strips_scaffold(tokenizer):
    parser = Qwen3Parser(tokenizer)
    reasoning, content = parser.extract_reasoning(
        OUTPUT_WITH_SCAFFOLD, make_request()
    )
    assert reasoning is not None and reasoning.strip() == "Some thoughts."
    assert content == "The answer is 42."


def test_nonstreaming_no_scaffold_untouched(tokenizer):
    parser = Qwen3Parser(tokenizer)
    _, content = parser.extract_reasoning(OUTPUT_NO_SCAFFOLD, make_request())
    assert content == "The answer is 42."


def test_nonstreaming_single_newline_preserved(tokenizer):
    parser = Qwen3Parser(tokenizer)
    _, content = parser.extract_reasoning(OUTPUT_SINGLE_NEWLINE, make_request())
    assert content == "\nThe answer is 42."


def test_nonstreaming_strips_only_one_scaffold(tokenizer):
    parser = Qwen3Parser(tokenizer)
    _, content = parser.extract_reasoning(OUTPUT_EXTRA_NEWLINE, make_request())
    assert content == "\nThe answer is 42."


def test_streaming_strips_scaffold_across_deltas(tokenizer):
    parser = Qwen3Parser(tokenizer)
    reasoning, content = collect(
        stream(parser, tokenizer, OUTPUT_WITH_SCAFFOLD, make_request())
    )
    assert "Some thoughts." in reasoning
    assert content == "The answer is 42."


def test_streaming_single_newline_preserved(tokenizer):
    parser = Qwen3Parser(tokenizer)
    _, content = collect(
        stream(parser, tokenizer, OUTPUT_SINGLE_NEWLINE, make_request())
    )
    assert content == "\nThe answer is 42."


def test_streaming_whitespace_only_content_stripped(tokenizer):
    """A response that is ONLY the scaffold yields empty content, not a
    stray whitespace flush at finish."""
    parser = Qwen3Parser(tokenizer)
    _, content = collect(
        stream(parser, tokenizer, "Some thoughts.\n</think>\n\n", make_request())
    )
    assert content == ""
