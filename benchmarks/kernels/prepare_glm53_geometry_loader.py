# SPDX-License-Identifier: Apache-2.0
"""Prepare private AOT caches from completed, pinned RMSNorm qualification.

No GPU work and no unpickling. Old helper freezes have ended: consume their
pinned reports, verify the actual source/binary artifacts, then freeze current
helpers explicitly. Never silently refresh serving, compiler or native hashes.
"""

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import (
    load_checked,
    require,
    verify,
)
from benchmarks.kernels.check_glm53_rmsnorm_geometry import (
    verify_binaries,
    write_new,
)
from benchmarks.kernels.glm53_rmsnorm_geometry import CONTROL, GEOMETRY, SCHEMA
from benchmarks.kernels.prepare_glm53_rmsnorm_geometry import PAIR_SHA
from slimserve.rmsnorm_diagnostic import NAMESPACE, sha

QUALIFICATION_SHA = "3b5bcff2c1d23b4b8a2ffa7c648267d5ee74003f67f6ce6495528d1c69516f69"


def qualified_targets(manifest, binaries):
    """Join ordered source/config receipts; never infer a candidate binary key."""
    require(len(manifest["targets"]) == 13, "thirteen qualified sources required")
    require(len(binaries) == 26, "twenty-six qualified binaries required")
    original = Path(manifest["original_namespace"])
    targets = {str(rank): [] for rank in range(4)}
    seen = set()
    for index, source in enumerate(manifest["targets"]):
        require(source["rank"] in range(4), "invalid qualified source rank")
        require(source["relative"] not in seen, "duplicate qualified source")
        seen.add(source["relative"])
        require(
            str(original / source["relative"]) == source["source"]
            and sha(source["source"]) == source["source_sha256"]
            and sha(source["debug_source"]) == source["debug_source_sha256"],
            "qualified source/debug identity changed",
        )
        configs = {}
        for offset, (mode, config) in enumerate(
            (("control", CONTROL), ("geometry", GEOMETRY))
        ):
            row = binaries[2 * index + offset]
            require(
                row["source_index"] == index
                and row["rank"] == source["rank"]
                and row["arm"] == mode
                and row["source_sha256"] == source["source_sha256"]
                and row["selected"]["config"] == config,
                "qualified source/binary/config join changed",
            )
            key = row["selected"]["hash"]
            relative = (
                f"inductor_cache/triton/{source['rank']}/{key}/{source['kernel']}.cubin"
            )
            require(
                sha(original / relative)
                == manifest["original_files"][relative]
                == row["cubin_sha256"],
                "qualified image absent from original rank-local cache",
            )
            configs[mode] = dict(
                config, triton_cache_hash=key, cubin_sha256=row["cubin_sha256"]
            )
        require(configs["control"] == source["control"], "control receipt changed")
        target = copy.deepcopy(source)
        target["configs"] = configs
        targets[str(source["rank"])].append(target)
    require([len(t) for t in targets.values()] == [3, 3, 3, 4], "rank coverage changed")
    return targets


def graph_inventory(mapping, original):
    graphs = {str(rank): {} for rank in range(4)}
    for graph in mapping["graphs"]:
        if graph["arm"] != "old":
            continue
        ranks = {row["rank"] for row in graph["norms"]}
        require(len(ranks) == 1, "graph rank must be unambiguous")
        (rank,) = ranks
        relative = str(Path(graph["path"]).relative_to(original))
        require(sha(graph["path"]) == graph["sha256"], "original graph changed")
        require(relative not in graphs[str(rank)], "duplicate original graph")
        graphs[str(rank)][relative] = graph["sha256"]
    require(all(len(g) == 7 for g in graphs.values()), "seven original graphs per rank")
    return graphs


def evidence(pair_path, manifest_path, mapping_path):
    pair = load_checked(pair_path, QUALIFICATION_SHA)
    require(
        pair["status"] == "complete"
        and pair["numerical_pass"] is True
        and pair["cross_process_exact"] is True
        and pair["pairs"] == 624,
        "completed independent numerical qualification required",
    )
    manifest = load_checked(manifest_path, pair["manifest_sha256"])
    mapping = load_checked(mapping_path, PAIR_SHA)
    receipts = {
        str(p.resolve()): sha(p) for p in (pair_path, manifest_path, mapping_path)
    }
    summaries = []
    require(
        [r["arm"] for r in pair["receipts"]] == ["a", "b"], "paired receipts required"
    )
    for receipt in pair["receipts"]:
        folder = pair_path.parent / receipt["arm"]
        for filename, key in (
            ("summary.json", "summary_sha256"),
            ("analysis.json", "analysis_sha256"),
        ):
            path = folder / filename
            document = load_checked(path, receipt[key])
            receipts[str(path.resolve())] = receipt[key]
            if filename == "summary.json":
                summaries.append(document)
        # Deliberately no read_manifest(): its former helper-source freeze ended.
        verify_binaries(manifest, folder, summaries[-1])
    require(summaries[0]["binaries"] == summaries[1]["binaries"], "binary pair differs")
    targets = qualified_targets(manifest, summaries[0]["binaries"])
    original = Path(manifest["original_namespace"])
    graphs = graph_inventory(mapping, original)
    require(original.name == NAMESPACE, "wrong original AOT namespace")
    return manifest, targets, graphs, receipts


def current_sources(previous, additions):
    sources, refreshed = {}, {}
    helpers = Path(__file__).resolve().parents[1]
    for name, digest in previous.items():
        current = sha(name)
        if current != digest:
            require(
                Path(name).is_relative_to(helpers),
                f"non-helper artifact changed: {name}",
            )
            refreshed[name] = dict(previous=digest, current=current)
        sources[name] = current
    for path in additions:
        sources[str(Path(path).resolve())] = sha(path)
    return sources, refreshed


def prepare(pair_path, manifest_path, mapping_path, output):
    require(not output.exists(), "new series required; preserve previous attempts")
    old, targets, graphs, receipts = evidence(pair_path, manifest_path, mapping_path)
    original = Path(old["original_namespace"])
    output = output.resolve()
    require(
        not output.is_relative_to(original) and not original.is_relative_to(output),
        "private series overlaps original cache",
    )
    import torch._dynamo.aot_compile as aot_compile
    import torch._inductor.async_compile as async_compile
    import torch._inductor.runtime.cache_dir_utils as cache_dir_utils
    import torch._inductor.standalone_compile as standalone_compile
    import torch._inductor.triton_bundler as triton_bundler

    from benchmarks.kernels import (
        audit_glm53_geometry_graphs,
        audit_glm53_geometry_loader,
        check_glm53_geometry_loader,
        glm53_binary_observer,
        glm53_geometry_loader,
    )

    additions = [
        Path(__file__),
        Path(glm53_binary_observer.__file__),
        Path(glm53_geometry_loader.__file__),
        Path(audit_glm53_geometry_graphs.__file__),
        Path(check_glm53_geometry_loader.__file__),
        Path(audit_glm53_geometry_loader.__file__),
        *(
            Path(module.__file__)
            for module in (
                aot_compile,
                async_compile,
                standalone_compile,
                triton_bundler,
                cache_dir_utils,
            )
        ),
        Path(__file__).resolve().parents[2] / "vllm/compilation/caching.py",
    ]
    sources, refreshed = current_sources(old["sources"], additions)
    sources.update(receipts)
    base = dict(
        schema=SCHEMA,
        qualification_sha256=QUALIFICATION_SHA,
        namespace=NAMESPACE,
        original_namespace=str(original),
        original_files=old["original_files"],
        targets=targets,
        expected_graphs=graphs,
        sources=sources,
        released_helper_changes=refreshed,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    verify(base)
    require(
        not any(p.is_symlink() for p in original.rglob("*")),
        "original cache contains aliases",
    )
    output.mkdir(parents=True)
    runs = []
    for mode in ("control", "geometry"):
        for rank in range(4):
            folder = output / f"{mode}-rank{rank}"
            cache = folder / "cache"
            private = cache / "torch_compile_cache/torch_aot_compile" / NAMESPACE
            shutil.copytree(original, private)
            require(
                all(sha(private / p) == h for p, h in old["original_files"].items()),
                "private copy differs",
            )
            manifest = dict(
                base,
                rank=rank,
                mode=mode,
                cache_root=str(cache),
                private_namespace=str(private),
                receipts=str(folder / "receipts"),
            )
            path = folder / "manifest.json"
            write_new(path, manifest)
            runs.append(
                dict(
                    rank=rank, mode=mode, manifest=str(path), manifest_sha256=sha(path)
                )
            )
    verify(base)
    record = dict(
        status="prepared",
        runs=runs,
        sources=sources,
        original_files=len(old["original_files"]),
        qualification_sha256=QUALIFICATION_SHA,
    )
    write_new(output / "preparation.json", record)
    print(
        json.dumps(
            dict(
                output=str(output),
                runs=len(runs),
                preparation_sha256=sha(output / "preparation.json"),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.pair, args.manifest, args.mapping, args.output)


if __name__ == "__main__":
    main()
