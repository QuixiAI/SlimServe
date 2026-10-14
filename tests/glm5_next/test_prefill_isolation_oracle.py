# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.validate_glm5_next_prefill_isolation import (
    aligned_prime_ids,
    check_reply,
)


@pytest.mark.parametrize("content", ["cedar-08-001-violet", "cedar-08-001-violet\n"])
def test_expected_record(content):
    check_reply({"choices": [{"message": {"content": content}}]},
                "cedar-08-001-violet", ["cedar-08-002-violet"])


@pytest.mark.parametrize("content", [None, "", "cedar-08-002-violet",
                                     "cedar-08-001-violet cedar-08-002-violet"])
def test_wrong_empty_or_mixed_record(content):
    with pytest.raises(AssertionError):
        check_reply({"choices": [{"message": {"content": content}}]},
                    "cedar-08-001-violet", ["cedar-08-002-violet"])


@pytest.mark.parametrize("length", [9217, 9218, 12000])
def test_prime_template_encoding_and_boundary(length):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            assert kwargs["add_generation_prompt"] and kwargs["thinking"]
            assert messages == [{"role": "user", "content": "question"}]
            return "rendered-chat"

        def encode(self, text, **kwargs):
            assert text == "rendered-chat" and kwargs == {"add_special_tokens": False}
            return list(range(length))

    if length <= 9217:
        with pytest.raises(AssertionError):
            aligned_prime_ids(Tokenizer(), "question")
    else:
        assert aligned_prime_ids(Tokenizer(), "question") == list(range(9217))
