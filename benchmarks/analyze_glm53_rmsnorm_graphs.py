# SPDX-License-Identifier: Apache-2.0
"""Offline receipt gates for the source-bound GLM53 RMSNorm experiment.

Graph inventory comes from the separately qualified real AOT loader. Neither
static callback counts nor Python object counts establish graph coverage.
"""

from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require


def expected(saved):
    return [
        dict(
            hash=saved["triton_cache_hash"],
            config={
                key: saved[key]
                for key in ("XBLOCK", "R0_BLOCK", "num_warps", "num_stages")
            },
        )
    ]


def audit_rank(
    records,
    *,
    rank,
    arm,
    manifest,
    manifest_sha256,
    source_sha256,
    expected_graphs,
    hash_file,
):
    """Check one complete rank, allowing aliases and non-static graph resolutions.

    expected_graphs maps (private-relative module path, symbol) to source SHA256.
    hash_file must read the actual files; tests provide an isolated fixture.
    The live verifier checks object identity; this audit binds its coverage
    receipt to the independently scanned graph inventory and exact binaries.
    """
    require(arm in ("control", "legacy"), "unknown arm")
    require(bool(records), "empty receipts")
    events = {"begin", "launcher", "graph_binding", "graph_coverage", "sealed"}
    require(
        all(r["event"] in events and r["rank"] == rank for r in records),
        "unknown event or wrong rank",
    )
    begins = [r for r in records if r["event"] == "begin"]
    seals = [r for r in records if r["event"] == "sealed"]
    coverages = [r for r in records if r["event"] == "graph_coverage"]
    require(
        len(begins) == len(seals) == len(coverages) == 1,
        "one begin, graph coverage and seal required",
    )
    begin, sealed, coverage = begins[0], seals[0], coverages[0]
    require(records[0] is begin and begin["mode"] == arm, "invalid begin")
    require(begin["manifest_sha256"] == manifest_sha256, "manifest hash changed")
    require(begin["source_sha256"] == source_sha256, "diagnostic source changed")
    seal_at = records.index(sealed)
    coverage_at = records.index(coverage)
    require(coverage_at == seal_at - 1, "coverage must immediately precede seal")
    require(sealed["sources"] == 1, "source coverage changed")
    private = Path(manifest["private_namespace"])
    target = manifest["targets"][str(rank)]
    native = expected(target["configs"][1])
    selected = expected(target["configs"][int(arm == "control")])
    bindings, graphs, targets, entries = set(), {}, [], []

    def relative(filename):
        path = Path(filename)
        require(path.is_relative_to(private), "source outside private namespace")
        name = str(path.relative_to(private))
        require(".." not in path.parts, "noncanonical source path")
        require(name in manifest["original_files"], "unrecorded generated source")
        require(
            hash_file(path) == manifest["original_files"][name],
            "generated source hash changed",
        )
        return name

    for index, row in enumerate(records):
        if row["event"] == "launcher":
            filename = relative(row["filename"])
            is_target = Path(filename).name == target["filename"]
            require(row["target"] == is_target, "wrong target classification")
            require(row["sealed"] == (index > seal_at), "wrong seal state")
            if not is_target:
                require(row["before"] == row["after"], "unrelated launcher changed")
            else:
                require(
                    hash_file(Path(row["filename"])) == target["source_sha256"],
                    "target source changed",
                )
                require(
                    row["resolution_index"] == len(targets) + 1,
                    "resolution sequence changed",
                )
                require(row["after"] == selected, "wrong selected binary/config")
                binding = row["binding_index"]
                if row["repeated"]:
                    require(binding in bindings, "unknown repeated binding")
                    allowed = (selected,) if row["sealed"] else (selected, native)
                    require(row["before"] in allowed, "repeat source binary changed")
                    require(
                        row["resolved_by"] in ("reuse", "graph"),
                        "repeat re-resolved upstream",
                    )
                else:
                    require(
                        binding == len(bindings) + 1 and not row["sealed"],
                        "new late or nonsequential binding",
                    )
                    require(row["before"] == native, "initial binary/config changed")
                    require(
                        row["resolved_by"] in ("upstream", "graph"),
                        "unknown initial resolution",
                    )
                    bindings.add(binding)
                targets.append(row)
            entries.append({**row, "filename": filename})
        elif row["event"] == "graph_binding":
            module = relative(row["module"])
            relative(row["filename"])
            key = (module, row["symbol"])
            require(key in expected_graphs, "unexpected graph/symbol")
            require(
                row["module_sha256"] == expected_graphs[key],
                "graph module hash changed",
            )
            require(row["selected"] == selected, "graph holds wrong binary/config")
            previous = records[index - 1]
            require(
                previous["event"] == "launcher"
                and previous["target"]
                and previous["resolved_by"] == "graph"
                and previous["filename"] == row["filename"]
                and previous["binding_index"] == row["binding_index"],
                "graph binding has no matching resolution",
            )
            require(index < coverage_at or key in graphs, "late unseen graph")
            graphs[key] = row
        elif row["event"] == "graph_coverage":
            require(
                set(graphs) == set(expected_graphs) and bool(graphs),
                "incomplete independently qualified graph inventory",
            )
            require(
                row["bindings"] == len(expected_graphs),
                "live graph coverage count differs from qualified inventory",
            )

    require(
        bindings == set(range(1, sealed["targets"] + 1)) and bool(bindings),
        "target binding count changed",
    )
    require(
        sealed["resolutions"] == sum(not r["sealed"] for r in targets),
        "sealed resolution count changed",
    )
    return dict(
        rank=rank,
        launchers=entries,
        targets=targets,
        graph_bindings=list(graphs.values()),
        graph_coverage=coverage,
        distinct_target_objects=len(bindings),
    )
