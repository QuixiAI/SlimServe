#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare two identical private AOT cache copies, never edit the originals."""

import argparse
import json
import shutil
from pathlib import Path

from slimserve.rmsnorm_diagnostic import AUDIT_SHA, NAMESPACE, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required; preserve prior attempts")
    if sha(args.audit) != AUDIT_SHA:
        parser.error("requires the exact source-bound cache audit")
    audit = json.loads(args.audit.read_text())
    original = Path(audit["paths"][1]).parent
    if original.name != NAMESPACE:
        parser.error("unexpected AOT namespace")
    args.output.mkdir(parents=True)
    snapshot = {
        str(p.relative_to(original)): sha(p)
        for p in sorted(original.rglob("*"))
        if p.is_file()
    }
    if any(p.is_symlink() for p in original.rglob("*")):
        raise ValueError("cache must contain regular files, not aliases to originals")
    for arm in ("control", "legacy"):
        folder = (args.output / arm).resolve()
        cache_root = folder / "cache"
        private = cache_root / "torch_compile_cache/torch_aot_compile" / NAMESPACE
        shutil.copytree(original, private)
        if any(
            sha(private / relative) != expected
            for relative, expected in snapshot.items()
        ):
            raise ValueError("private cache copy differs")
        targets = {}
        for entry in audit["reduction_width_changes"]:
            (source,) = entry["sources"]
            relative = Path(source["relative"])
            (rank,) = source["device_indices"]
            copied = private / "inductor_cache" / relative
            if sha(copied) != source["source_sha256"]:
                raise ValueError("source changed since audit")
            injection = folder / "injection" / f"rank-{rank}" / relative.name
            injection.parent.mkdir(parents=True)
            shutil.copyfile(copied, injection)
            targets[rank] = dict(
                filename=relative.name,
                kernel=source["kernels"][0],
                source_sha256=source["source_sha256"],
                configs=entry["configs"],
                injection_source=str(injection),
            )
        manifest = dict(
            schema=1,
            namespace=NAMESPACE,
            audit_sha256=AUDIT_SHA,
            original_namespace=str(original),
            private_namespace=str(private),
            cache_root=str(cache_root),
            receipts=str(folder / "receipts"),
            targets=targets,
            original_files=snapshot,
            preparer_sha256=sha(__file__),
        )
        with (folder / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        print(
            json.dumps(
                dict(
                    arm=arm,
                    manifest=str(folder / "manifest.json"),
                    files=len(snapshot),
                    cache_root=str(cache_root),
                )
            ),
            flush=True,
        )
    if any(
        sha(original / relative) != expected for relative, expected in snapshot.items()
    ):
        raise ValueError("original cache changed during copying")


if __name__ == "__main__":
    main()
