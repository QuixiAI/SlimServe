# SPDX-License-Identifier: Apache-2.0
"""Inventory the thirteen original-source geometry targets without GPU work.

This is discovery evidence, NOT a qualified intervention manifest. In particular,
the new geometry's source-specific binary keys/bytes must be established by a
separate source-exact probe before constructing MultiIntervention.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.analyze_glm53_norm_graph_roles import pair_graphs
from benchmarks.kernels.check_glm53_attention_norms import (
    checked_debug_source,
    load_checked,
    matching_function_provenance,
    require,
    source_info,
    verify,
)
from benchmarks.kernels.check_glm53_cached_rmsnorm import source_layout
from benchmarks.kernels.glm53_rmsnorm_geometry import CONTROL, GEOMETRY
from slimserve.rmsnorm_diagnostic import sha

PAIR_SHA = "320a6f39c4f92cd78459f21e23653e1e900870620da29bed923f5cba5dbe315e"


def select_changes(mapping, inventory):
    changes = mapping["correspondence"]["unique_changed_sources"]
    require(
        Counter(c["rank"] for c in changes) == {0: 3, 1: 3, 2: 3, 3: 4},
        "expected thirteen sources across all ranks",
    )
    require(
        Counter(tuple(c["roles"]) for c in changes)
        == {("input_layernorm",): 6, ("post_attention_layernorm",): 7},
        "unexpected model roles",
    )
    seen, targets = set(), []
    for change in sorted(changes, key=lambda c: (c["rank"], c["old_source"])):
        key = (change["rank"], change["old_source"])
        require(key not in seen, "duplicate geometry source")
        seen.add(key)
        matched = [
            row
            for row in inventory
            if (row["rank"], row["source"]) == key and row["arm"] == "combo"
        ]
        require(len(matched) == 1, "one original source receipt required")
        (source,) = matched
        require(
            change["old_config"] == source["selected"]["config"] == CONTROL
            and change["new_config"] == GEOMETRY
            and change["body"] == source["info"]["body_sha256"]
            and source["info"]["widths"] == [4096],
            "source/body/config correspondence changed",
        )
        targets.append((change, source))
    return targets


def prepare(pair_path, manifest_path, output):
    require(not output.exists(), "new discovery output required")
    mapping = load_checked(pair_path, PAIR_SHA)
    require(
        sha(manifest_path) == mapping["manifest_sha256"], "inventory manifest changed"
    )
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    require(
        pair_graphs(mapping["graphs"]) == mapping["correspondence"],
        "graph pairing changed",
    )
    for graph in mapping["graphs"]:
        require(sha(graph["path"]) == graph["sha256"], "graph source changed")
    original = Path(manifest["original_namespace"])
    sources = dict(manifest["sources"])
    for path in (Path(__file__), pair_path, manifest_path):
        sources[str(path.resolve())] = sha(path)
    from benchmarks.kernels import glm53_rmsnorm_geometry

    sources[glm53_rmsnorm_geometry.__file__] = sha(glm53_rmsnorm_geometry.__file__)
    targets = []
    for change, row in select_changes(mapping, manifest["inventory"]):
        source = Path(row["source"])
        text = source.read_text()
        require(source_info(text) == row["info"], "source semantics changed")
        kernel = row["info"]["kernel"]
        layout = source_layout(text, kernel)
        relative = source.relative_to(original)
        selected = row["selected"]
        binary = original / f"inductor_cache/triton/{row['rank']}/{selected['hash']}"
        paths = {
            suffix: binary / f"{kernel}.{suffix}" for suffix in ("cubin", "ptx", "json")
        }
        for path in paths.values():
            digest = manifest["original_files"][str(path.relative_to(original))]
            require(sha(path) == digest, "original binary receipt changed")
            sources[str(path)] = digest
        debug = checked_debug_source(paths["ptx"], original)
        matching_function_provenance(text, debug.read_text(), kernel)
        sources[str(debug)] = sha(debug)
        metadata = json.loads(paths["json"].read_text())
        require(
            metadata["name"] == kernel
            and metadata["num_warps"] == 16
            and metadata["num_stages"] == 1,
            "original binary metadata changed",
        )
        uses = [
            dict(
                graph=g["path"],
                graph_sha256=g["sha256"],
                symbol=n["symbol"],
                uses=n["uses"],
            )
            for g in mapping["graphs"]
            if g["arm"] == "old"
            for n in g["norms"]
            if (n["rank"], n["source"]) == (row["rank"], str(source))
        ]
        require(bool(uses) and all(u["uses"] for u in uses), "no static source uses")
        targets.append(
            dict(
                rank=row["rank"],
                relative=str(relative),
                source=str(source),
                source_sha256=row["source_sha256"],
                kernel=kernel,
                layout=layout,
                body_sha256=change["body"],
                roles=change["roles"],
                debug_source=str(debug),
                debug_source_sha256=sha(debug),
                control=dict(
                    CONTROL,
                    triton_cache_hash=selected["hash"],
                    cubin_sha256=sha(paths["cubin"]),
                ),
                geometry_config=GEOMETRY,
                static_graph_uses=uses,
            )
        )
    result = dict(
        schema="glm53-rmsnorm-geometry-discovery-v1",
        status="discovery-only",
        qualified_for_intervention=False,
        limitation=__doc__,
        targets=targets,
        sources=sources,
        original_namespace=str(original),
        original_files=manifest["original_files"],
    )
    verify(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(dict(output=str(output), sha256=sha(output), targets=len(targets)))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.pairs, args.manifest, args.output)


if __name__ == "__main__":
    main()
