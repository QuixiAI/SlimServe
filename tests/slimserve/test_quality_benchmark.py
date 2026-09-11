# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from benchmarks import benchmark_glm53_quality as quality
from benchmarks.benchmark_glm53_quality import (
    CODES,
    continuation_windows,
    needle_prefix,
    scored_tail,
)


def test_windows_keep_explicit_token_boundaries():
    windows = continuation_windows(list(range(30)), 3, 5, 3)
    assert windows == [
        (0, list(range(8))),
        (11, list(range(11, 19))),
        (22, list(range(22, 30))),
    ]
    with pytest.raises(ValueError, match="distinct"):
        continuation_windows(list(range(8)), 2, 5, 3)


def response(ids):
    return {
        "usage": {"prompt_tokens": len(ids)},
        "choices": [
            {
                "prompt_logprobs": [None]
                + [
                    {"999": {"logprob": -1}, str(t): {"logprob": -t / 10}}
                    for t in ids[1:]
                ]
            }
        ],
    }


def test_scores_actual_ids_not_first_dictionary_entry():
    ids = [10, 20, 30, 40]
    assert scored_tail(response(ids), ids, 2) == [-3, -4]


@pytest.mark.parametrize("problem", ["count", "alignment", "missing", "nonfinite"])
def test_invalid_server_scores_are_rejected(problem):
    ids = [10, 20, 30]
    data = response(ids)
    if problem == "count":
        data["usage"]["prompt_tokens"] = 4
    elif problem == "alignment":
        data["choices"][0]["prompt_logprobs"].pop()
    elif problem == "missing":
        data["choices"][0]["prompt_logprobs"][-1] = {"999": {"logprob": -1}}
    else:
        data["choices"][0]["prompt_logprobs"][-1]["30"]["logprob"] = float("nan")
    with pytest.raises(ValueError):
        scored_tail(data, ids, 1)


def test_needle_context_is_exact_even_when_source_repeats():
    encode = lambda text: list(text.encode())
    for fraction in (0.0, 0.25, 0.75, 1.0):
        prefix = needle_prefix(encode, [1, 2, 3], 1024, fraction)
        assert len(prefix) == 1024
        assert bytes(prefix).count(CODES[0].encode()) == 1


def test_failed_response_is_retained(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("a record with enough source tokens")
    args = SimpleNamespace(
        source=source,
        output=tmp_path / "quality.json",
        model="test",
        url="http://unused",
        pairs=1,
        prefix_tokens=5,
        score_tokens=3,
        needle_contexts=[256],
        needle_positions=[0.5],
    )

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    bad = {"usage": {"prompt_tokens": -1}, "choices": []}
    monkeypatch.setattr(quality, "request_score", lambda *args: bad)
    with pytest.raises(ValueError, match="count"):
        quality.run(args, Tokenizer())
    saved = json.loads(args.output.read_text())
    assert saved["status"] == "failed"
    assert saved["text"][0]["response"] == bad
