# SPDX-License-Identifier: Apache-2.0
"""CPU regression for repeated long prompts at serving concurrency."""

import pytest

from benchmarks.benchmark_dsv4_exact import exact_prompts


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids):
        return "".join(map(chr, ids))


@pytest.mark.parametrize("count", [1, 8, 16, 32, 128])
@pytest.mark.parametrize("offset", [0, 23])
def test_repeated_source_has_concurrent_windows(count, offset):
    source = "".join(chr(0x100 + i) for i in range(256))
    prompts = exact_prompts(CharacterTokenizer(), source, count, 1024, offset, True)
    assert len(prompts) == count
    assert len(set(prompts)) == count
    assert all(len(p) == 1024 for p in prompts)
    assert prompts[0] == (source * 6)[offset : offset + 1024]


def test_short_protocol_unchanged_when_source_is_sufficient():
    tokenizer = CharacterTokenizer()
    source = "abcdefghijklmno"
    expected = ["abcde", "fghij", "klmno"]
    assert exact_prompts(tokenizer, source, 3, 5, 0, False) == expected
    assert exact_prompts(tokenizer, source, 3, 5, 0, True) == expected


@pytest.mark.parametrize("repeat", [False, True])
def test_empty_source_has_clear_error(repeat):
    with pytest.raises(ValueError, match="at least one token"):
        exact_prompts(CharacterTokenizer(), "", 8, 1024, 0, repeat)


def test_no_repeat_does_not_silently_extend_source():
    with pytest.raises(ValueError, match="need at least"):
        exact_prompts(CharacterTokenizer(), "abc", 8, 1024, 0, False)


@pytest.mark.parametrize("count,tokens,offset", [(0, 1, 0), (1, 0, 0), (1, 1, -1)])
def test_invalid_shapes_rejected(count, tokens, offset):
    with pytest.raises(ValueError, match="must be positive"):
        exact_prompts(CharacterTokenizer(), "abc", count, tokens, offset, True)
