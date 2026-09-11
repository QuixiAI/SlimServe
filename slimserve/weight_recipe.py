# SPDX-License-Identifier: Apache-2.0
"""Prepare the exact derived weights named by a platform record.

The original checkpoint stays intact. A separate directory links its registered
files and holds the recipe's verified sidecars. Builders run in a child process
so preparing weights never leaves a CUDA context in the serving launcher.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from slimserve import term
from slimserve.registry import Plan, cache_root, files_for

RECEIPT = "slimserve-weight-recipe.json"
SIDECARS = (
    "f32-overrides.safetensors",
    "fp8-swapset.safetensors",
    "fp8-swapset.json",
)


def artifact_digest(path: Path) -> str:
    """Hash tensor layout and bytes, excluding machine-dependent provenance.

    Safetensors metadata can contain a local path and its key order is not stable.
    The tensor descriptors (including offsets) and every data byte are included.
    The swap-set manifest similarly excludes only its informational source path.
    """
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        manifest.pop("source", None)
        return hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 100_000_000:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(size))
        header.pop("__metadata__", None)
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory: Path, recipe: dict) -> None:
    for name in SIDECARS:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"{recipe['id']}: missing required artifact {path}")
        if artifact_digest(path) != recipe["artifact_digests"][name]:
            raise ValueError(f"{recipe['id']}: artifact digest mismatch: {path}")


def _build(plan: Plan, staging: Path) -> None:
    recipe = plan.weight_recipe
    assert recipe is not None
    from huggingface_hub import snapshot_download

    native = recipe["native"]
    native_dir = snapshot_download(
        repo_id=native["repo"],
        revision=native["revision"],
        local_dir=cache_root() / native["local_dir"],
        allow_patterns=["*.safetensors", "config.json", "model.safetensors.index.json"],
        max_workers=2,
    )
    # Both builders read the original shard index; existing sidecars cannot
    # masquerade as original weights or suppress a rebuild.
    common = ["--native", str(native_dir), "--model", str(plan.model_dir)]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "slimserve.f32_overrides",
            *common,
            "--out",
            str(staging / SIDECARS[0]),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "slimserve.fp8_swapset",
            *common,
            "--out",
            str(staging / SIDECARS[1]),
            "--self-quant-kda",
            "--tp-size",
            str(recipe["tp_size"]),
        ],
        check=True,
    )


def ensure(plan: Plan) -> None:
    recipe = plan.weight_recipe
    assert recipe is not None
    if plan.source.get("revision") != recipe["target_revision"]:
        raise ValueError(f"{recipe['id']}: target revision differs from the recipe")
    for key in ("SLIMSERVE_F32_OVERRIDES", "SLIMSERVE_FP8_SWAPSET"):
        if os.environ.get(key, "1") != "1":
            raise ValueError(
                f"{recipe['id']} requires {key}=1; unset the conflicting value"
            )
    if plan.engine["tensor_parallel_size"] != recipe["tp_size"]:
        raise ValueError(
            f"{recipe['id']}: tensor parallel size differs from the recipe"
        )
    destination = plan.entry_file
    if destination.exists():
        receipt = destination / RECEIPT
        if not receipt.is_file() or json.loads(receipt.read_text()) != recipe:
            raise ValueError(
                f"{destination}: existing directory has a different weight recipe"
            )
        validate(destination, recipe)
        for entry in files_for(plan):
            if entry["role"] != "model" and entry["role"] != "shared":
                continue
            linked = destination / entry["path"]
            if (
                not linked.is_symlink()
                or linked.resolve() != (plan.model_dir / entry["path"]).resolve()
            ):
                raise ValueError(
                    f"{destination}: original checkpoint link differs: {entry['path']}"
                )
        term.ok(f"verified weight recipe: {recipe['id']}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    term.step(f"preparing {recipe['id']} in {staging}")
    try:
        for entry in files_for(plan):
            if entry["role"] in {"model", "shared"}:
                link = staging / entry["path"]
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to((plan.model_dir / entry["path"]).resolve())
        try:
            validate(plan.model_dir, recipe)
        except ValueError:
            _build(plan, staging)
        else:
            # Adopt only byte-verified existing sidecars, without sharing a
            # mutable inode with the experiment's originals.
            for name in SIDECARS:
                shutil.copyfile(plan.model_dir / name, staging / name)
        validate(staging, recipe)
        (staging / RECEIPT).write_text(json.dumps(recipe, indent=2) + "\n")
        staging.rename(destination)
    except BaseException:
        term.warn(f"incomplete recipe retained for diagnosis: {staging}")
        raise
    term.ok(f"prepared weight recipe: {recipe['id']}")
