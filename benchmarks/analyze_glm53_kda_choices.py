# SPDX-License-Identifier: Apache-2.0
"""Inventory retained KDA disk tuning choices without importing GPU code.

Disk minima are not live launch receipts, especially for the failed start's
shared cache: concurrent ranks can select different in-memory winners. The
completed geometry series has per-rank, closure-pinned copies instead.
"""

import argparse
import json
import math
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.analyze_glm53_reduction_receipts import sha

KERNELS = {
    "recompute_w_u_fwd_kernel",
    "chunk_kda_fwd_kernel_inter_solve_fused",
    "chunk_kda_fwd_kernel_intra_sub_chunk",
    "chunk_gla_fwd_kernel_o",
    "kda_gate_chunk_cumsum_vector_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
}
SUFFIX = ".autotune.json"
CLOSURE_SHA = "c25674eab322b7334c694e1354f0a2ab3c8b7f3b8bb9be4e8d1e6f92091cf597"
FAILED_AUDIT_SHA = "eb333fb93c2ac28218d46ca7799d95ce2378f73ebc0ccfe5d49138b6e2530c87"


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def disk_choice(document):
    """Match Triton's lexicographic minimum and first-entry tie behavior."""
    require(set(document) == {"key", "configs_timings"}, "unknown tuning schema")
    require(isinstance(document["key"], list) and document["key"], "empty tuning key")
    canonical(document["key"])
    rows = document["configs_timings"]
    require(isinstance(rows, list) and bool(rows), "empty config matrix")
    seen = set()
    for config, timing in rows:
        require(isinstance(config, dict), "invalid config")
        require(isinstance(config.get("kwargs"), dict), "missing config kwargs")
        require(config.get("pre_hook") is None, "unsupported cached pre-hook")
        for name in ("num_warps", "num_stages", "num_ctas"):
            require(
                type(config.get(name)) is int and config[name] > 0,
                "invalid config geometry",
            )
        identity = canonical(config)
        require(identity not in seen, "duplicate cached config")
        seen.add(identity)
        require(
            isinstance(timing, list) and len(timing) == 3, "invalid timing quantiles"
        )
        require(
            all(
                type(t) in (int, float) and not math.isnan(t) and t >= 0 for t in timing
            ),
            "invalid timing value",
        )
    # Do not sort by only the first timing or reorder exact ties. This is the
    # installed Autotuner.check_disk_cache selection rule, not a TPS estimate.
    config, timing = min(rows, key=lambda row: row[1])
    require(all(math.isfinite(t) for t in timing), "no finite winner")
    return dict(
        key=document["key"],
        candidates=[row[0] for row in rows],
        selected=config,
        selected_timing_ms=timing,
    )


def read_pinned(path, expected):
    require(sha(path) == expected, f"receipt changed: {path}")
    return json.loads(Path(path).read_text())


def within(root, relative):
    root = Path(root).resolve()
    path = root / relative
    require(
        not Path(relative).is_absolute() and ".." not in Path(relative).parts,
        "unbounded path",
    )
    require(path.resolve().is_relative_to(root), "path escapes cache")
    return path


def scan_choices(root, *, rank_private, frozen_files=None):
    """Read only the six known KDA tuning files, rejecting incomplete matrices."""
    root = Path(root).resolve()
    files = sorted(
        root.glob("*/*/*.autotune.json" if rank_private else "*/*.autotune.json")
    )
    result = {}
    for path in files:
        name = path.name.removesuffix(SUFFIX)
        require(name in KERNELS, f"unexpected tuned kernel: {name}")
        relative = str(path.relative_to(root))
        within(root, relative)
        rank = path.parts[-3] if rank_private else "shared"
        require(
            rank in ({"0", "1", "2", "3"} if rank_private else {"shared"}),
            "unknown rank",
        )
        require((rank, name) not in result, "ambiguous per-kernel tuning key")
        digest = sha(path)
        if frozen_files is not None:
            require(
                frozen_files.get(relative) == digest, "unpinned/changed tuning file"
            )
        result[rank, name] = dict(
            path=str(path),
            sha256=digest,
            disk_key=path.parent.name,
            **disk_choice(json.loads(path.read_text())),
        )
    ranks = ("0", "1", "2", "3") if rank_private else ("shared",)
    require(
        set(result) == {(r, k) for r in ranks for k in KERNELS}, "incomplete KDA matrix"
    )
    if frozen_files is not None:
        require(
            set(frozen_files)
            == {str(Path(row["path"]).relative_to(root)) for row in result.values()},
            "frozen tuning matrix differs",
        )
    return result


def compare_choices(control, fresh):
    rows = []
    for (rank, name), old in sorted(control.items()):
        new = fresh["shared", name]
        for field in ("disk_key", "key", "candidates"):
            require(old[field] == new[field], f"unmatched {field}: {name}")
        rows.append(
            dict(
                rank=int(rank),
                kernel=name,
                disk_key=old["disk_key"],
                key=old["key"],
                original=old,
                fresh_shared=new,
                changed=old["selected"] != new["selected"],
            )
        )
    return rows


def analyze(series, failed_audit):
    series = Path(series).resolve()
    closure = read_pinned(series / "closure.json", CLOSURE_SHA)
    require(closure["status"] == "complete", "series incomplete")
    require(
        [c["label"] for c in closure["cases"]]
        == ["control", "geometry", "return-control"],
        "wrong series",
    )
    arms, receipts = {}, {str(series / "closure.json"): CLOSURE_SHA}
    for case in closure["cases"]:
        folder = series / case["label"]
        documents = {}
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("launch.json", "launch_sha256"),
            ("worker-analysis.json", "worker_analysis_sha256"),
            ("workload-analysis.json", "workload_analysis_sha256"),
        ):
            documents[filename] = read_pinned(folder / filename, case[field])
            receipts[str(folder / filename)] = case[field]
        manifest = documents["manifest.json"]
        launch = documents["launch.json"]
        require(launch["status"] == "complete", "launch incomplete")
        private = Path(manifest["private_namespace"])
        require(
            private.resolve().is_relative_to(folder), "private namespace escapes case"
        )
        root = private / "inductor_cache/triton"
        frozen = {}
        for relative, digest in manifest["original_files"].items():
            if not relative.endswith(SUFFIX):
                continue
            path = within(private, relative)
            require(path.is_relative_to(root), "tuning file outside rank cache")
            require(
                launch["files"].get(str(path.relative_to(folder))) == digest,
                "series changed original tuning bytes",
            )
            frozen[str(path.relative_to(root))] = digest
        launch_tuning = {p for p in launch["files"] if p.endswith(SUFFIX)}
        require(
            launch_tuning == {str((root / p).relative_to(folder)) for p in frozen},
            "extra series tuning files",
        )
        arms[case["label"]] = scan_choices(root, rank_private=True, frozen_files=frozen)
    control = arms["control"]
    for label in ("geometry", "return-control"):
        require(control.keys() == arms[label].keys(), "arm matrix mismatch")
        require(
            all(control[k]["sha256"] == arms[label][k]["sha256"] for k in control),
            "KDA tuning bytes changed across arms",
        )
    audit = read_pinned(failed_audit, FAILED_AUDIT_SHA)
    receipts[str(Path(failed_audit).resolve())] = FAILED_AUDIT_SHA
    # Historical audit bounds the cache root, but did NOT record these six file
    # hashes or rank-local in-memory winners. Be explicit about that lower tier.
    fresh = scan_choices(audit["binary_cache_directory"], rank_private=False)
    rows = compare_choices(control, fresh)
    return dict(
        status="complete",
        analyzer_sha256=sha(__file__),
        receipts=receipts,
        original_rank_files=24,
        completed_series_files_verified=72,
        series_tuning_bytes_exact=True,
        fresh_shared_files=len(fresh),
        changed_rank_kernel_pairs=sum(row["changed"] for row in rows),
        changed_kernel_names=sorted({row["kernel"] for row in rows if row["changed"]}),
        comparisons=rows,
        limitation=(
            "Original per-rank disk choices are pinned by completed series receipts. "
            "Fresh shared-cache choices are hashed now, not historically pinned "
            "per rank. Neither is a live KDA launch/config receipt; shared races may "
            "hide rank-local winners. No model causality, accuracy or speed claim."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--failed-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    result = analyze(args.series, args.failed_audit)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("comparisons", "receipts")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
