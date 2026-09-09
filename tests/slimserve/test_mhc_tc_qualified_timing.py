# SPDX-License-Identifier: Apache-2.0
"""CPU-only fail-closed checks for the independent-census timing prerequisite."""

import copy
import json
import math

import pytest

from benchmarks.kernels.benchmark_glm53_mhc_prefill_tc import oracle_rows
from benchmarks.kernels.benchmark_glm53_mhc_tc_qualified import (
    BATCHES,
    qualified_census,
    validate_accuracy,
    validate_partial,
)
from benchmarks.kernels.check_glm53_mhc_tc_accuracy import PROBE_SHA, case_plan, sha


def accuracy(rows):
    arm = {
        "passed": True,
        "elements": rows * 4096,
        "violations": 0,
        "post_violations": 0,
        "comb_violations": 0,
        "squared_error": 0,
        "ideal_squared_sum": 1,
        "normalized_rms": 0,
    }
    return {
        "passed": True,
        "rows": rows,
        "candidate_rms_noninferior": True,
        "reference": dict(arm),
        "candidate": dict(arm),
    }


def partial(batch):
    return {
        "passed": True,
        "finite": True,
        "rows": oracle_rows(batch),
        "dot_normalized_rms": 0,
        "dot_row_peak_error": 0,
        "square_relative_error": 0,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("violations", 1),
        ("post_violations", 1),
        ("comb_violations", 1),
        ("elements", 4096),
        ("normalized_rms", math.nan),
        ("squared_error", -1),
        ("normalized_rms", 0.1),
    ],
)
def test_accuracy_validator_recomputes_gates_despite_passed_flags(key, value):
    data = accuracy(64)
    data["candidate"][key] = value
    with pytest.raises(ValueError):
        validate_accuracy(data, 64)


def test_accuracy_validator_recomputes_rms_noninferiority():
    data = accuracy(64)
    data["candidate"].update(squared_error=1e-6, normalized_rms=0.001)
    with pytest.raises(ValueError, match="RMS regression"):
        validate_accuracy(data, 64)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dot_normalized_rms", 2.01e-6),
        ("dot_row_peak_error", 2.01e-5),
        ("square_relative_error", 1.01e-5),
        ("finite", False),
        ("rows", [0]),
        ("square_relative_error", math.nan),
    ],
)
def test_partial_validator_preserves_original_bounds(key, value):
    data = partial(65)
    data[key] = value
    with pytest.raises(ValueError):
        validate_partial(data, 65)


def write_census(directory, change=None):
    plan = case_plan(BATCHES, list(range(90)))
    summary = {
        "status": "complete",
        "contract": "glm53-mhc-tc-accuracy-v1",
        "planned_cases": 2700,
        "completed_cases": 2700,
        "completed_graph_phases": 8100,
        "plan": plan,
        "extension_sha256": PROBE_SHA,
        "strict_parity_failed_comparisons": 1,
    }
    rows = []
    for item in plan:
        t = item["batch"]
        original_failure = (
            t == 7616
            and item["site"] == 79
            and item["magnitude"] == 1
            and item["fused"]
        )
        row = {
            **item,
            "status": "complete",
            "eager_fp64_all_rows": accuracy(t),
            "partial_oracle": partial(t),
            "strict_parity_diagnostic": {"passed": not original_failure},
            "graphs": [],
        }
        for seed in item["graph_seeds"]:
            row["graphs"].append(
                {
                    "status": "complete",
                    "seed": seed,
                    "all_output_bits_match_eager": True,
                    "oracle_rows": oracle_rows(t),
                    "fp64_sampled_rows": accuracy(len(oracle_rows(t))),
                    "partial_oracle": partial(t),
                    "strict_parity_diagnostic": {"passed": True},
                }
            )
        rows.append(row)
    if change:
        change(summary, rows)
    journal = directory / "checks.jsonl"
    journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary["journal_sha256"] = sha(journal)
    (directory / "summary.json").write_text(json.dumps(summary))


def test_full_census_accepts_preserved_old_parity_failure(tmp_path):
    write_census(tmp_path)
    _, receipt = qualified_census(tmp_path)
    assert receipt["completed_cases"] == 2700
    assert receipt["strict_parity_failed_comparisons"] == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda s, r: s.update(status="running"),
        lambda s, r: s.update(completed_cases=6),
        lambda s, r: s.update(strict_parity_failed_comparisons=0),
        lambda s, r: r.pop(),
        lambda s, r: r.append(copy.deepcopy(r[-1])),
        lambda s, r: r[-1].update(eager_seed=1),
        lambda s, r: r[-1]["graphs"][-1].update(seed=1),
        lambda s, r: r[-1]["graphs"][-1].update(all_output_bits_match_eager=False),
        lambda s, r: r[-1]["eager_fp64_all_rows"]["candidate"].update(violations=1),
    ],
)
def test_census_rejects_partial_or_inconsistent_receipts(tmp_path, change):
    write_census(tmp_path, change)
    with pytest.raises(ValueError):
        qualified_census(tmp_path)


def test_census_rejects_changed_journal(tmp_path):
    write_census(tmp_path)
    with (tmp_path / "checks.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="digest"):
        qualified_census(tmp_path)
