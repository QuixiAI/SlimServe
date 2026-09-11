# SPDX-License-Identifier: Apache-2.0
"""Offline checks of actual graph bindings for deterministic GLM53 reductions.

No generated code is imported. Graph source independently supplies the expected
Triton symbols; live receipts supply their objects and selected configurations.
Cache keys identify compiled artifacts, not their byte digests: hash both.
"""

import ast
import base64
import hashlib
import json
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def declared_symbols(source):
    tree = ast.parse(source)
    assigned = {
        target.id
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name) and target.id.startswith("triton_")
    }
    called = {
        node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "run"
        and isinstance(node.value, ast.Name)
        and node.value.id.startswith("triton_")
    }
    return assigned | called


def reduction_metadata(source):
    fields = {
        "deterministic",
        "batch_invariant",
        "are_deterministic_algorithms_enabled",
        "num_reduction",
        "kernel_name",
    }
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.keyword) or node.arg != "inductor_meta":
            continue
        require(isinstance(node.value, ast.Dict), "nonliteral compiler metadata")
        found.append(
            {
                key.value: ast.literal_eval(value)
                for key, value in zip(node.value.keys, node.value.values)
                if isinstance(key, ast.Constant) and key.value in fields
            }
        )
    require(len(found) == 1, "one emitted compiler metadata block required")
    return found[0]


def audit_receipt(
    document, *, cache_root, triton_cache_root, rank, source_sha256, heuristic_sha256
):
    root = Path(cache_root).resolve()
    # AOT decorators redirect only Inductor; the non-AOT adaptor redirects both.
    # Require the caller's recorded Triton directory. A source-file sibling is
    # not proof of where its binary was compiled, and no cache search is allowed.
    binary_root = Path(triton_cache_root).resolve()
    require(binary_root.is_relative_to(root), "binary cache outside private root")
    require(document["status"] == "complete", "incomplete graph capture")
    require(document["rank"] == rank, "wrong rank")
    require(document["cache_root"] == str(root), "wrong private cache")
    require(document["source_sha256"] == source_sha256, "recorder changed")
    require(document["heuristic_source_sha256"] == heuristic_sha256, "compiler changed")
    require(document["compiler_options"]["deterministic"] is True, "wrong policy")
    require(set(document["snapshots"]) == {"before", "after"}, "missing capture phase")
    phases = {}
    for phase, snapshot in document["snapshots"].items():
        require(not snapshot["violations"], "live graph inspection failed")
        files = snapshot["files"]
        for filename, digests in files.items():
            path = Path(filename)
            require(path.resolve().is_relative_to(root), "source outside private cache")
            require(sha(path) == digests["sha256"], "generated source changed")
            normalized = path.read_bytes().replace(str(root).encode(), b"<CACHE>")
            require(
                hashlib.sha256(normalized).hexdigest() == digests["normalized_sha256"],
                "normalized source mismatch",
            )
        graphs = {g["module_id"]: g for g in snapshot["graphs"]}
        require(
            bool(graphs) and len(graphs) == len(snapshot["graphs"]),
            "missing/duplicate graph",
        )
        rows = snapshot["bindings"]
        require(
            len({(r["module_id"], r["symbol"]) for r in rows}) == len(rows),
            "duplicate graph binding",
        )
        require(all(r["module_id"] in graphs for r in rows), "binding without graph")
        for module_id, graph in graphs.items():
            require(
                files[graph["module"]]
                == {k: graph[k] for k in ("sha256", "normalized_sha256")},
                "graph hash mismatch",
            )
            expected = declared_symbols(Path(graph["module"]).read_text())
            actual = {r["symbol"] for r in rows if r["module_id"] == module_id}
            require(expected == actual, "incomplete graph source/global coverage")
        checked = []
        for row in rows:
            require(
                row["module"] == graphs[row["module_id"]]["module"],
                "binding module mismatch",
            )
            require(
                files[row["filename"]]
                == {k: row[k] for k in ("sha256", "normalized_sha256")},
                "kernel hash mismatch",
            )
            metadata = reduction_metadata(Path(row["filename"]).read_text())
            norm = "rms_norm" in row["symbol"] or "rms_norm" in metadata["kernel_name"]
            reduction = norm or bool(metadata.get("num_reduction", 0))
            require(
                row["rmsnorm"] == norm and row["reduction"] == reduction,
                "wrong reduction classification",
            )
            if not reduction:
                continue
            require(metadata["kernel_name"] == row["kernel"], "kernel name mismatch")
            require(
                metadata["deterministic"]
                is row["deterministic"]
                is row["runtime_deterministic"]
                is True,
                "nondeterministic reduction",
            )
            require(
                not any(
                    (
                        metadata.get("batch_invariant"),
                        metadata.get("are_deterministic_algorithms_enabled"),
                        row["batch_invariant"],
                        row["global_deterministic"],
                        row["dynamic_rblock_cached"],
                    )
                ),
                "unexpected numerical mode",
            )
            require(len(row["selected"]) == 1, "one selected reduction required")
            selected = row["selected"][0]
            config = selected["config"]
            if norm:
                require(
                    config["kwargs"] == {"XBLOCK": 1, "R0_BLOCK": 1024},
                    "unqualified RMSNorm tree",
                )
                require(
                    (config["num_warps"], config["num_stages"]) == (8, 1),
                    "unqualified RMSNorm launch",
                )
            folder = binary_root / selected["hash"]
            require(
                folder.resolve().is_relative_to(root), "binary outside private cache"
            )
            binary_metadata = folder / (row["kernel"] + ".json")
            binary = json.loads(binary_metadata.read_text())
            cache_key = (
                base64.b32encode(bytes.fromhex(binary["hash"])).decode().rstrip("=")
            )
            require(cache_key == selected["hash"], "compiled cache key mismatch")
            require(
                binary["target"]["backend"] == "cuda"
                and binary["target"]["arch"] == 120,
                "wrong binary target",
            )
            require(
                all(
                    binary[k] == config[k]
                    for k in ("num_warps", "num_ctas", "num_stages", "maxnreg")
                ),
                "compiled config mismatch",
            )
            cubin = folder / (row["kernel"] + ".cubin")
            require(cubin.stat().st_size > 0, "empty compiled binary")
            checked.append(
                dict(
                    rank=rank,
                    module=graphs[row["module_id"]]["normalized_sha256"],
                    symbol=row["symbol"],
                    source=row["normalized_sha256"],
                    selected=selected,
                    cubin_sha256=sha(cubin),
                    metadata_sha256=sha(binary_metadata),
                    rmsnorm=norm,
                )
            )
        require(any(r["rmsnorm"] for r in checked), "missing RMSNorm bindings")
        phases[phase] = sorted(checked, key=lambda r: (r["module"], r["symbol"]))
    require(phases["before"] == phases["after"], "reductions changed during capture")
    return phases["after"]
