# SPDX-License-Identifier: Apache-2.0
"""Independent correction AOT graph/driver/leaf joins; CPU only."""

import argparse
import sys
from pathlib import Path

from benchmarks.kernels import audit_glm53_kv_loader as common
from benchmarks.kernels.check_glm53_attention_norms import require
from benchmarks.kernels.check_glm53_indexer_correction_loader import (
    ORDER as ORDER,
)
from benchmarks.kernels.check_glm53_indexer_correction_loader import (
    prior_receipts as prior_receipts,
)
from benchmarks.kernels.check_glm53_indexer_correction_loader import (
    read_manifest as read_manifest,
)
from benchmarks.kernels.glm53_indexer_bound_leaves import (
    audit_leaf,
    expected_bindings,
    expected_cases,
)
from slimserve.rmsnorm_diagnostic import sha

CANDIDATE_MODE = EXTRA_KEY = "correction"
EVENT_PREFIX = "indexer_correction"
DISPATCH = "combo_then_indexer_correction"
LOADER_SOURCE = Path(__file__).with_name("glm53_indexer_correction_loader.py")
RUNNER_SOURCE = Path(__file__).with_name("check_glm53_indexer_correction_loader.py")
PAIR_FIELDS = ("leaf_cases",)
EXPECTED_WEIGHTS = 4
SCOPE = (
    "Actual AOT/driver/binding and synthetic leaf replay qualification; "
    "not model/TPS qualification."
)


def check_extra(extra, manifest):
    capacity = manifest["selection_capacity"]
    require(
        extra["selection_capacity"] == capacity
        and extra["selection_bytes"] == (capacity + 2) * 128
        and extra["selection_dtype"] == "uint8"
        and extra["selection_device"] == f"cuda:{manifest['rank']}",
        "selection arena receipt differs",
    )


def check_leaf(manifest, graphs, summary, read):
    output = Path(manifest["cache_root"]).parent / "run"
    path = output / "bound-leaves.json"
    leaves = read(path)
    bindings, cases = expected_bindings(graphs), expected_cases(manifest)
    require(
        leaves["status"] == "complete"
        and leaves["bindings"] == bindings
        and len(bindings) == 2
        and len(cases) == 30
        and leaves["cases"] == len(leaves["records"]) == 60
        and summary["leaf_qualification"] == dict(cases=60, sha256=sha(path)),
        "bound-leaf qualification incomplete",
    )
    for index, (binding, case) in enumerate((b, c) for b in bindings for c in cases):
        receipt = leaves["records"][index]
        name = f"leaf-{index:03d}.json"
        require(
            receipt["path"] == name and sha(output / name) == receipt["sha256"],
            "leaf receipt changed/reordered",
        )
        audit_leaf(read(output / name), binding, case, manifest["mode"])
    return dict(leaf_cases=60, leaf_phase_observations=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "compare"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        raise SystemExit(
            0 if common.audit(args.path, workflow=sys.modules[__name__]) else 1
        )
    common.compare(args.path.resolve(), workflow=sys.modules[__name__])


if __name__ == "__main__":
    main()
