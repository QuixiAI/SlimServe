# SPDX-License-Identifier: Apache-2.0
"""Join completed AOT/leaf receipts into one control/correction/return series."""

import argparse
import sys
from pathlib import Path

from benchmarks.kernels import prepare_glm53_kv_serving as common
from benchmarks.kernels.check_glm53_attention_norms import require
from benchmarks.kernels.check_glm53_indexer_correction_loader import ORDER as ORDER
from benchmarks.kernels.prepare_glm53_geometry_serving import integration_sources
from slimserve.indexer_correction_diagnostic import (
    AOT_PAIR_SHA as AOT_PAIR_SHA,
)
from slimserve.indexer_correction_diagnostic import (
    CASES as CASES,
)
from slimserve.indexer_correction_diagnostic import (
    SERVING_SCHEMA as SERVING_SCHEMA,
)

ROOT = common.ROOT
PAIR = ROOT / "perf/results/2026-09-10/indexer-aot-v1/pair-analysis.json"
EXTRA_KEY = "correction"
COMMON_FIELDS = (
    *common.COMMON_FIELDS,
    "selection_capacity",
    "weight_sha256",
    "qualified_leaf_cases",
)
# No qualified kernel/loader/graph inspector bytes may change. Only this closed
# stage's notebook changed among the AOT source freeze; serving files are new.
INTEGRATION_SITES = {ROOT / "perf/glm53-indexer-loader-protocol.md"}


def check_completed_report(report, row, manifest):
    require(
        row["leaf_cases"] == report["leaf_cases"] == 60
        and report["leaf_phase_observations"] == 300
        and manifest["selection_capacity"] == 8192
        and len(manifest["qualified_leaf_cases"]) == 120,
        "completed bound-leaf/arena qualification required",
    )


def completed_evidence(path):
    return common.completed_evidence(path, workflow=sys.modules[__name__])


def serving_sources(previous):
    from slimserve.campaign_sources import PATHS

    return integration_sources(
        previous,
        [
            Path(__file__),
            Path(common.__file__),
            *(ROOT / p for p in PATHS),
            *(
                ROOT / f"benchmarks/kernels/{name}.py"
                for name in (
                    "prepare_glm53_geometry_serving",
                    "glm53_geometry_workload",
                    "audit_glm53_geometry_serving",
                    "run_glm53_geometry_serving",
                    "glm53_indexer_correction_serving",
                )
            ),
            ROOT / "benchmarks/analyze_glm53_deterministic_serving.py",
            ROOT / "slimserve/registry.py",
            ROOT / "slimserve/hardware.py",
            ROOT / "slimserve/profiles.json",
            ROOT / "vllm/compilation/cuda_graph.py",
            ROOT / "vllm/distributed/parallel_state.py",
            ROOT / "vllm/utils/torch_utils.py",
            ROOT / "perf/glm53-indexer-serving-protocol.md",
            *sorted(INTEGRATION_SITES),
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
