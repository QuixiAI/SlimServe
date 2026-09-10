#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only fixed 1/3/1 quality comparison; not general quality certification.

Applies perf/glm53-mhc-tc-accuracy-contract.md without changing its tolerance.
Checks original token IDs/scores, not just each run's aggregate summary.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from benchmarks.benchmark_glm53_quality import CODES, scored_tail

TOLERANCE = 0.01


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_quality(document):
    require(document["status"] == "complete", "incomplete quality run")
    require(len(document["text"]) == 32, "32 continuation windows required")
    require(document["settings"]["prefix_tokens"] == 512, "prefix changed")
    require(document["settings"]["score_tokens"] == 128, "score length changed")
    scores = []
    for row in document["text"]:
        require(len(row["prompt_ids"]) == 640, "prompt length changed")
        actual = scored_tail(row["response"], row["prompt_ids"], 512)
        require(actual == row["per_token"], "recorded scores differ from response")
        scores.extend(actual)
    require(len(scores) == 4096, "4096 scored tokens required")
    require(
        document["summary"]["scored_text_tokens"] == len(scores),
        "token summary mismatch",
    )
    require(
        math.isclose(
            statistics.mean(scores),
            document["summary"]["mean_text_logprob"],
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "score summary mismatch",
    )
    expected = [(t, p) for t in (1024, 8192, 32768) for p in (0.25, 0.75)]
    require(
        [(r["context_tokens"], r["position"]) for r in document["needles"]] == expected,
        "needle plan changed",
    )
    margins = []
    for row in document["needles"]:
        require(
            len(row["prefix_ids"]) == row["context_tokens"],
            "needle prefix length changed",
        )
        require(
            [c["code"] for c in row["candidates"]] == list(CODES),
            "needle codes changed",
        )
        totals = []
        for candidate in row["candidates"]:
            require(len(candidate["suffix_ids"]) == 7, "needle suffix length changed")
            actual = scored_tail(
                candidate["response"],
                row["prefix_ids"] + candidate["suffix_ids"],
                len(row["prefix_ids"]),
            )
            require(
                actual == candidate["per_token"], "needle scores differ from response"
            )
            totals.append(sum(actual))
        margin = totals[0] - max(totals[1:])
        require(margin == row["margin"], "needle margin mismatch")
        margins.append(margin)
    require(margins == document["summary"]["needle_margins"], "needle summary mismatch")
    require(
        all(m > 0 for m in margins) == document["summary"]["all_needles_rank_first"],
        "needle pass flag mismatch",
    )
    return scores


def input_identity(document):
    return {
        "source_sha256": document["source_sha256"],
        "text": [(r["offset"], r["prompt_ids"]) for r in document["text"]],
        "needles": [
            (r["prefix_ids"], [c["suffix_ids"] for c in r["candidates"]])
            for r in document["needles"]
        ],
    }


def delta_summary(a, b):
    delta = [y - x for x, y in zip(a, b)]
    return {
        "changed_tokens": sum(d != 0 for d in delta),
        "mean_delta": statistics.mean(delta),
        "mean_abs_delta": statistics.mean(abs(d) for d in delta),
        "rms_delta": math.sqrt(statistics.mean(d * d for d in delta)),
        "max_abs_delta": max(abs(d) for d in delta),
    }


def compare(controls, candidates):
    require(
        len(controls) == 2 and len(candidates) == 3,
        "fixed two controls and three candidates required",
    )
    return compare_observations(controls, candidates)


def compare_observations(controls, candidates):
    """Apply the same quality envelope to actual observations, without replicas.

    The historical 1/3/1 CLI and ``compare`` retain their exact cardinality. This
    entry point also checks each fresh production start against pinned historical
    controls before proceeding. It does not expose a tolerance override.
    """
    require(
        len(controls) == 2 and len(candidates) > 0,
        "two controls and observations required",
    )
    documents = [*controls, *candidates]
    scores = [verified_quality(d) for d in documents]
    identity = input_identity(controls[0])
    require(
        all(input_identity(d) == identity for d in documents),
        "quality prompt IDs or source changed",
    )
    control_means = [statistics.mean(s) for s in scores[:2]]
    windows = [
        [statistics.mean(s[i : i + 128]) for i in range(0, 4096, 128)] for s in scores
    ]
    results = []
    for index, values in enumerate(scores[2:]):
        mean = statistics.mean(values)
        per_window = []
        for i in range(32):
            reference = [windows[c][i] for c in (0, 1)]
            floor = min(reference) - TOLERANCE
            per_window.append(
                {
                    "window": i,
                    "offset": controls[0]["text"][i]["offset"],
                    "controls": reference,
                    "candidate": windows[index + 2][i],
                    "minimum_allowed": floor,
                    "passed": windows[index + 2][i] >= floor,
                }
            )
        aggregate_pass = mean >= min(control_means) - TOLERANCE
        worst = sorted(
            range(4096), key=lambda i: abs(values[i] - scores[0][i]), reverse=True
        )[:16]
        results.append(
            {
                "candidate": index + 1,
                "mean_logprob": mean,
                "aggregate_passed": aggregate_pass,
                "all_windows_passed": all(w["passed"] for w in per_window),
                "all_needles_passed": all(
                    m > 0 for m in candidates[index]["summary"]["needle_margins"]
                ),
                "windows": per_window,
                "per_token_delta_to_controls": [
                    delta_summary(s, values) for s in scores[:2]
                ],
                "largest_changes_from_first_control": [
                    {
                        "window": i // 128,
                        "scored_position": i % 128,
                        "token_id": controls[0]["text"][i // 128]["prompt_ids"][
                            512 + i % 128
                        ],
                        "controls": [s[i] for s in scores[:2]],
                        "candidate": values[i],
                    }
                    for i in worst
                ],
            }
        )
    controls_pass = all(d["summary"]["all_needles_rank_first"] for d in controls)
    return {
        "passed": controls_pass
        and all(
            r["aggregate_passed"]
            and r["all_windows_passed"]
            and r["all_needles_passed"]
            for r in results
        ),
        "method": __doc__,
        "tolerance_nat_per_token": TOLERANCE,
        "controls_mean_logprob": control_means,
        "control_per_token_variation": delta_summary(*scores[:2]),
        "controls_needles_passed": controls_pass,
        "candidates": results,
    }


def held_out_controls(documents):
    """Diagnose the fixed gate on three actual same-policy control observations.

    Each observation is held out once against the other two. This does not
    estimate a false-positive rate, enlarge the envelope, select references,
    or change a campaign verdict. Callers must establish policy/receipt identity.
    """
    require(len(documents) == 3, "three actual control observations required")
    folds = []
    for held_out in range(3):
        controls = [i for i in range(3) if i != held_out]
        comparison = compare_observations(
            [documents[i] for i in controls], [documents[held_out]]
        )
        folds.append(
            {
                "held_out": held_out,
                "controls": controls,
                "comparison": comparison,
            }
        )
    return {
        "method": held_out_controls.__doc__,
        "folds": folds,
        "failed_folds": sum(not f["comparison"]["passed"] for f in folds),
        "campaign_verdict_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs=3, required=True)
    parser.add_argument("--return-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "new output file required")
    paths = [args.control, args.return_control, *args.candidate]
    blobs = [p.read_bytes() for p in paths]
    documents = [json.loads(b) for b in blobs]
    result = compare(documents[:2], documents[2:])
    result["sources"] = [
        {"path": str(p), "sha256": hashlib.sha256(b).hexdigest()}
        for p, b in zip(paths, blobs)
    ]
    result["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "controls_mean_logprob": result["controls_mean_logprob"],
            }
        )
    )
    if not result["passed"]:
        raise SystemExit("predeclared quality gate failed; all comparisons retained")


if __name__ == "__main__":
    main()
