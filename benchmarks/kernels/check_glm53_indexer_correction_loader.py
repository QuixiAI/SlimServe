# SPDX-License-Identifier: Apache-2.0
"""One-attempt actual AOT load plus bound-leaf numerics; no model forward."""

import argparse
import subprocess
import sys
from pathlib import Path

from benchmarks.kernels import check_glm53_geometry_loader as common
from benchmarks.kernels.audit_glm53_indexer_correction_graphs import (
    inventory,  # noqa: F401
)
from benchmarks.kernels.check_glm53_attention_norms import require, verify
from benchmarks.kernels.glm53_indexer_bound_leaves import qualify_bindings  # noqa: F401
from benchmarks.kernels.glm53_indexer_correction_loader import (
    SCHEMA,
    IndexerCorrectionLoader,
)
from benchmarks.kernels.prepare_glm53_indexer_correction_loader import (
    CONTRACTS_SHA,
    ORDER,
    QUALIFICATION_SHA,
    SELECTION_CAPACITY,
)
from slimserve.rmsnorm_diagnostic import NAMESPACE, sha

MODULE = "benchmarks.kernels.check_glm53_indexer_correction_loader"
AUDITOR_MODULE = "benchmarks.kernels.audit_glm53_indexer_correction_loader"
UNIT_PREFIX = "glm53-indexer-aot-v1"
LOADER = IndexerCorrectionLoader
LEAF_WEIGHT_COUNT = 4


def read_manifest(path):
    path = path.resolve()
    data = common.read_json(path)
    prep = common.read_json(path.parent.parent / "preparation.json")
    require(
        data["schema"] == SCHEMA
        and data["namespace"] == NAMESPACE
        and data["qualification_sha256"] == QUALIFICATION_SHA
        and data["contracts_sha256"] == CONTRACTS_SHA
        and data["selection_capacity"] == SELECTION_CAPACITY
        and prep["status"] == "prepared"
        and prep["sources"] == data["sources"]
        and [(r["mode"], r["rank"]) for r in prep["runs"]] == list(ORDER),
        "wrong prepared correction series",
    )
    for row, (mode, rank) in zip(prep["runs"], ORDER):
        expected = path.parent.parent / f"{mode}-rank{rank}/manifest.json"
        require(
            row["manifest"] == str(expected)
            and row["manifest_sha256"] == sha(expected),
            "prepared manifest changed",
        )
    require(
        (data["mode"], data["rank"]) in ORDER
        and path.parent.name == f"{data['mode']}-rank{data['rank']}"
        and data["cache_root"] == str(path.parent / "cache")
        and data["private_namespace"]
        == str(path.parent / "cache/torch_compile_cache/torch_aot_compile" / NAMESPACE)
        and data["receipts"] == str(path.parent / "receipts"),
        "wrong rank/mode/cache paths",
    )
    require(
        set(data["targets"]) == {str(r) for r in range(4)}
        and all(
            len(data["targets"][str(r)]) == 1
            and len(data["targets"][str(r)][0]["static_graph_uses"]) == 2
            and len(data["expected_graphs"][str(r)])
            == len(data["artifact_roots"][str(r)])
            == 7
            and sum(len(a["submodules"]) for a in data["artifact_roots"][str(r)]) == 46
            for r in range(4)
        ),
        "incomplete target/AOT matrix",
    )
    require(
        data["private_sources"]
        == {
            t["correction"]["relative"]: t["correction"]["source_sha256"]
            for ts in data["targets"].values()
            for t in ts
        },
        "private correction source inventory differs",
    )
    from benchmarks.kernels.check_glm53_attention_overwrite import matrix

    require(
        [r["case"] for r in data["qualified_leaf_cases"]] == matrix(),
        "qualified leaf order changed",
    )
    require(
        data["git_commit"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source commit changed",
    )
    verify(data)
    return data, prep


def prior_receipts(path, manifest, preparation):
    return common.prior_receipts(path, manifest, preparation, order=ORDER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "launch"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    (common.run if args.action == "run" else common.launch)(
        args.manifest, workflow=sys.modules[__name__]
    )


if __name__ == "__main__":
    main()
