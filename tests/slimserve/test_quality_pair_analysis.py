# SPDX-License-Identifier: Apache-2.0
import copy
import statistics

import pytest

from benchmarks.analyze_glm53_quality_pair import (
    compare,
    compare_observations,
    held_out_controls,
)
from benchmarks.benchmark_glm53_quality import CODES


def response(ids, prefix, values):
    return {
        "usage": {"prompt_tokens": len(ids)},
        "choices": [
            {
                "prompt_logprobs": [None] * prefix
                + [
                    {str(token): {"logprob": value}}
                    for token, value in zip(ids[prefix:], values)
                ]
            }
        ],
    }


def update_window(doc, index, value):
    row = doc["text"][index]
    row["per_token"] = [value] * 128
    row["response"] = response(row["prompt_ids"], 512, row["per_token"])
    doc["summary"]["mean_text_logprob"] = statistics.mean(
        v for r in doc["text"] for v in r["per_token"]
    )


def build_docs():
    document = {
        "status": "complete",
        "source_sha256": "test-source",
        "settings": {"prefix_tokens": 512, "score_tokens": 128},
        "text": [],
        "needles": [],
        "summary": {
            "scored_text_tokens": 4096,
            "mean_text_logprob": -2.0,
            "needle_margins": [7.0] * 6,
            "all_needles_rank_first": True,
        },
    }
    for i in range(32):
        ids = [i + 1] * 640
        document["text"].append(
            {
                "offset": i,
                "prompt_ids": ids,
                "per_token": [-2.0] * 128,
                "response": response(ids, 512, [-2.0] * 128),
            }
        )
    for context in (1024, 8192, 32768):
        for position in (0.25, 0.75):
            prefix = [1] * context
            row = {
                "context_tokens": context,
                "position": position,
                "prefix_ids": prefix,
                "candidates": [],
                "margin": 7.0,
            }
            for i, code in enumerate(CODES):
                suffix, scores = [i + 2] * 7, [-1.0 if i == 0 else -2.0] * 7
                row["candidates"].append(
                    {
                        "code": code,
                        "suffix_ids": suffix,
                        "per_token": scores,
                        "response": response(prefix + suffix, context, scores),
                    }
                )
            document["needles"].append(row)
    return [copy.deepcopy(document) for _ in range(5)]


@pytest.fixture
def docs():
    return build_docs()


def test_fixed_identical_runs_pass(docs):
    result = compare(docs[:2], docs[2:])
    assert result["passed"]
    assert result["control_per_token_variation"]["changed_tokens"] == 0


def test_lower_of_both_controls_is_the_predeclared_reference(docs):
    for i in range(32):
        update_window(docs[1], i, -2.02)
        for candidate in docs[2:]:
            update_window(candidate, i, -2.025)
    assert compare(docs[:2], docs[2:])["passed"]


def test_one_bad_window_fails_even_if_aggregate_passes(docs):
    update_window(docs[2], 5, -2.011)
    result = compare(docs[:2], docs[2:])
    assert not result["passed"]
    assert result["candidates"][0]["aggregate_passed"]
    assert not result["candidates"][0]["windows"][5]["passed"]


def test_inclusive_predeclared_boundary(docs):
    for i in range(32):
        update_window(docs[2], i, -2.01)
    assert compare(docs[:2], docs[2:])["passed"]


@pytest.mark.parametrize(
    "mutation", ["incomplete", "ids", "scores", "summary", "count", "needle"]
)
def test_rejects_incomplete_or_inconsistent_evidence(docs, mutation):
    candidate = docs[2]
    if mutation == "incomplete":
        candidate["status"] = "running"
    elif mutation == "ids":
        candidate["text"][0]["offset"] = 99
    elif mutation == "scores":
        candidate["text"][0]["per_token"][0] = -5.0
    elif mutation == "summary":
        candidate["summary"]["mean_text_logprob"] = -1.0
    elif mutation == "count":
        candidate["text"].pop()
    elif mutation == "needle":
        candidate["needles"][0]["margin"] = 99
    with pytest.raises(ValueError):
        compare(docs[:2], docs[2:])


def test_rejects_missing_candidate(docs):
    with pytest.raises(ValueError, match="fixed two controls"):
        compare(docs[:2], docs[2:4])


def test_actual_single_observation_uses_unchanged_envelope(docs):
    one = compare_observations(docs[:2], docs[2:3])
    three = compare(docs[:2], docs[2:])
    assert one["candidates"] == three["candidates"][:1]
    assert one["tolerance_nat_per_token"] == three["tolerance_nat_per_token"] == 0.01
    update_window(docs[2], 3, -2.011)
    assert not compare_observations(docs[:2], docs[2:3])["passed"]
    with pytest.raises(ValueError, match="observations"):
        compare_observations(docs[:2], [])


def test_negative_needle_cannot_pass_with_consistent_summary(docs):
    candidate = docs[2]
    row = candidate["needles"][0]
    item = row["candidates"][0]
    item["per_token"] = [-3.0] * 7
    item["response"] = response(
        row["prefix_ids"] + item["suffix_ids"], 1024, item["per_token"]
    )
    row["margin"] = -7.0
    candidate["summary"]["needle_margins"][0] = -7.0
    candidate["summary"]["all_needles_rank_first"] = False
    assert not compare(docs[:2], docs[2:])["passed"]


def test_held_out_controls_use_each_observation_once(docs):
    original = copy.deepcopy(docs[:3])
    report = held_out_controls(docs[:3])
    assert report["failed_folds"] == 0
    assert not report["campaign_verdict_changed"]
    for i, fold in enumerate(report["folds"]):
        assert fold["held_out"] == i
        assert fold["controls"] == [j for j in range(3) if j != i]
        assert fold["comparison"]["tolerance_nat_per_token"] == 0.01
    assert docs[:3] == original


def test_held_out_control_failure_is_not_excused_by_aggregate(docs):
    for i in range(3):
        update_window(docs[i], i, -2.03)
    report = held_out_controls(docs[:3])
    assert report["failed_folds"] == 3
    for i, fold in enumerate(report["folds"]):
        candidate = fold["comparison"]["candidates"][0]
        assert candidate["aggregate_passed"]
        assert [w["window"] for w in candidate["windows"] if not w["passed"]] == [i]


def test_held_out_controls_reject_incomplete_or_mismatched_evidence(docs):
    with pytest.raises(ValueError, match="three actual"):
        held_out_controls(docs[:2])
    docs[2]["text"][0]["offset"] = 99
    with pytest.raises(ValueError, match="prompt IDs or source"):
        held_out_controls(docs[:3])
