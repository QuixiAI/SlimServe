# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.muse_glimmer_reasoning_parser import (
    MuseGlimmerReasoningParser,
    _parse,
)


def test_muse_parser_accepts_recipient_with_or_without_leading_space():
    suffix = "to=self<|message|>reason<|eom|><|start|>assistant<|message|>4<|eot|>"

    assert _parse(suffix) == ("reason", "4")
    assert _parse(" " + suffix) == ("reason", "4")


def test_muse_streaming_parser_holds_partial_recipient_header():
    parser = MuseGlimmerReasoningParser.__new__(MuseGlimmerReasoningParser)

    partial = parser.extract_reasoning_streaming("", " to", " to", [], [], [])
    reasoning = parser.extract_reasoning_streaming(
        " to",
        " to=self<|message|>reason",
        "=self<|message|>reason",
        [],
        [],
        [],
    )
    content = parser.extract_reasoning_streaming(
        " to=self<|message|>reason<|eom|>",
        (" to=self<|message|>reason<|eom|><|start|>assistant<|message|>4"),
        "<|start|>assistant<|message|>4",
        [],
        [],
        [],
    )

    assert partial is None
    assert reasoning is not None
    assert reasoning.reasoning == "reason"
    assert reasoning.content is None
    assert content is not None
    assert content.reasoning is None
    assert content.content == "4"


def test_muse_streaming_parser_restores_disambiguated_plain_prefix():
    parser = MuseGlimmerReasoningParser.__new__(MuseGlimmerReasoningParser)

    held = parser.extract_reasoning_streaming("", "t", "t", [], [], [])
    plain = parser.extract_reasoning_streaming("t", "today", "oday", [], [], [])

    assert held is None
    assert plain is not None
    assert plain.reasoning is None
    assert plain.content == "today"


def test_muse_parser_treats_bare_assistant_prompt_suffix_as_new_reasoning():
    class _Tokenizer:
        @staticmethod
        def decode(token_ids):
            if token_ids == []:
                return ""
            if token_ids == [10, 20, 30]:
                return "<|start|>user<|message|>question<|eot|><|start|>assistant"
            if token_ids == [40]:
                return "to=self"
            if token_ids == [50]:
                return ""
            if token_ids == [40, 50]:
                return "to=self<|message|>"
            if token_ids == [40, 50, 60]:
                return "to=self<|message|>reason<|eom|>"
            raise AssertionError(f"unexpected token ids: {token_ids}")

    parser = MuseGlimmerReasoningParser.__new__(MuseGlimmerReasoningParser)
    parser.model_tokenizer = _Tokenizer()
    parser.message_token_id = 50
    parser.eom_token_id = 60

    assert not parser.is_reasoning_end([10, 20, 30])
    assert not parser.is_reasoning_end([40, 50])
    assert parser.is_reasoning_end([50])
    assert parser.is_reasoning_end([40, 50, 60])
