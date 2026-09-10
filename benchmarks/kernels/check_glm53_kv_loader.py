# SPDX-License-Identifier: Apache-2.0
"""Bounded no-weights KV-only AOT load. No model forward or timing."""

import argparse
import subprocess
import sys
from pathlib import Path

from benchmarks.kernels import check_glm53_geometry_loader as common
from benchmarks.kernels.audit_glm53_kv_graphs import inventory  # noqa: F401
from benchmarks.kernels.check_glm53_attention_norms import require, verify
from benchmarks.kernels.glm53_kv_loader import SCHEMA, KVLoader
from benchmarks.kernels.prepare_glm53_kv_loader import (
    CONTRACTS_SHA,
    ORDER,
    QUALIFICATION_SHA,
)
from slimserve.rmsnorm_diagnostic import NAMESPACE, sha

MODULE = "benchmarks.kernels.check_glm53_kv_loader"
AUDITOR_MODULE = "benchmarks.kernels.audit_glm53_kv_loader"
UNIT_PREFIX = "glm53-kv-aot-v2"
LOADER = KVLoader


def read_manifest(path):
    path = path.resolve()
    manifest = common.read_json(path)
    preparation = common.read_json(path.parent.parent / "preparation.json")
    require(
        manifest["schema"] == SCHEMA
        and manifest["namespace"] == NAMESPACE
        and manifest["qualification_sha256"] == QUALIFICATION_SHA
        and manifest["contracts_sha256"] == CONTRACTS_SHA
        and manifest["sources"] == preparation["sources"]
        and preparation["status"] == "prepared"
        and [(r["mode"], r["rank"]) for r in preparation["runs"]] == list(ORDER),
        "wrong prepared KV series",
    )
    for index, (mode, rank) in enumerate(ORDER):
        row = preparation["runs"][index]
        expected = path.parent.parent / f"{mode}-rank{rank}/manifest.json"
        require(
            row["manifest"] == str(expected)
            and sha(expected) == row["manifest_sha256"],
            "prepared manifest changed",
        )
    require(
        path.parent.name == f"{manifest['mode']}-rank{manifest['rank']}"
        and (manifest["mode"], manifest["rank"]) in ORDER
        and manifest["cache_root"] == str(path.parent / "cache")
        and manifest["private_namespace"]
        == str(path.parent / "cache/torch_compile_cache/torch_aot_compile" / NAMESPACE)
        and manifest["receipts"] == str(path.parent / "receipts")
        and set(manifest["targets"]) == {str(r) for r in range(4)}
        and all(
            len(manifest["targets"][str(r)]) == 1
            and len(manifest["targets"][str(r)][0]["static_graph_uses"]) == 2
            and len(manifest["expected_graphs"][str(r)]) == 7
            and len(manifest["artifact_roots"][str(r)]) == 7
            and sum(len(a["submodules"]) for a in manifest["artifact_roots"][str(r)])
            == 46
            for r in range(4)
        ),
        "invalid rank/mode/private paths or AOT coverage",
    )
    require(
        manifest["private_sources"]
        == {
            t["kv"]["relative"]: t["kv"]["source_sha256"]
            for targets in manifest["targets"].values()
            for t in targets
        },
        "private KV source inventory differs",
    )
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        == manifest["git_commit"],
        "source commit changed",
    )
    verify(manifest)
    return manifest, preparation


def prior_receipts(path, manifest, preparation):
    return common.prior_receipts(path, manifest, preparation, order=ORDER)


def run(path):
    return common.run(path, workflow=sys.modules[__name__])


def launch(path):
    return common.launch(path, workflow=sys.modules[__name__])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "launch"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    (run if args.action == "run" else launch)(args.manifest)


if __name__ == "__main__":
    main()
