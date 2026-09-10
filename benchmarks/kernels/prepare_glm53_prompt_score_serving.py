# SPDX-License-Identifier: Apache-2.0
"""Prepare a memory-only intervention using the qualified unchanged AOT loader."""

import argparse
import sys
from pathlib import Path

from benchmarks.kernels import prepare_glm53_indexer_correction_serving as original
from benchmarks.kernels import prepare_glm53_kv_serving as common
from benchmarks.kernels.prepare_glm53_geometry_serving import integration_sources
from slimserve.prompt_score_diagnostic import AOT_PAIR_SHA as AOT_PAIR_SHA
from slimserve.prompt_score_diagnostic import CASES as CASES
from slimserve.prompt_score_diagnostic import SERVING_SCHEMA as SERVING_SCHEMA

ROOT, PAIR = original.ROOT, original.PAIR
ORDER, COMMON_FIELDS, EXTRA_KEY = (
    original.ORDER,
    original.COMMON_FIELDS,
    original.EXTRA_KEY,
)
check_completed_report = original.check_completed_report
INTEGRATION_SITES = original.INTEGRATION_SITES | {
    ROOT / p
    for p in (
        "slimserve/cli.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "benchmarks/benchmark_glm53_campaign.py",
        "slimserve/campaign_sources.py",
        "slimserve/score_journal.py",
        "slimserve/glm53_serving_diagnostic.py",
        "benchmarks/kernels/glm53_geometry_workload.py",
        "benchmarks/kernels/audit_glm53_geometry_serving.py",
    )
}


def completed_evidence(path):
    from benchmarks.kernels.check_glm53_attention_norms import load_checked, require
    from slimserve.rmsnorm_diagnostic import sha

    result = common.completed_evidence(path, workflow=sys.modules[__name__])
    probe = ROOT / "perf/results/2026-09-10/prompt-scores-gpu-v1/result.json"
    report = load_checked(
        probe, "68b82a83a8a7bc0ba8cc9b4ae349e6de6a0882f280e51edfec41aa99993cc271"
    )
    require(
        report["status"] == "passed"
        and len(report["results"]) == 60
        and all(r["exact"] for r in report["results"]),
        "isolated score qualification missing",
    )
    for name in (
        "vllm/v1/sample/prompt_logprobs.py",
        "vllm/v1/sample/sampler.py",
        "vllm/v1/sample/ops/logprobs.py",
    ):
        require(
            sha(ROOT / name) == report["sources"][str(ROOT / name)],
            "qualified sampler/helper changed",
        )
    result[3][str(probe)] = sha(probe)
    return result


def serving_sources(previous):
    from slimserve.campaign_sources import PATHS

    return integration_sources(
        previous,
        [
            Path(__file__),
            Path(common.__file__),
            Path(original.__file__),
            *(ROOT / p for p in PATHS),
            *sorted(INTEGRATION_SITES),
            *(
                ROOT / p
                for p in (
                    "benchmarks/kernels/prepare_glm53_geometry_serving.py",
                    "benchmarks/kernels/run_glm53_geometry_serving.py",
                    "benchmarks/analyze_glm53_deterministic_serving.py",
                    "slimserve/registry.py",
                    "slimserve/hardware.py",
                    "slimserve/profiles.json",
                    "vllm/compilation/cuda_graph.py",
                    "vllm/distributed/parallel_state.py",
                    "vllm/utils/torch_utils.py",
                    "perf/glm53-prompt-score-protocol.md",
                    "benchmarks/kernels/check_glm53_prompt_scores.py",
                    "tests/slimserve/test_prompt_score_runner.py",
                )
            ),
        ],
        allowed_changes=INTEGRATION_SITES,
    )


def prepare(path, output, *, inspect_only=False):
    return common.prepare(
        path, output, inspect_only=inspect_only, workflow=sys.modules[__name__]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, default=PAIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    prepare(args.qualification.resolve(), args.output, inspect_only=args.inspect_only)


if __name__ == "__main__":
    main()
