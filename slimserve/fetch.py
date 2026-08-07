# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Get a plan's model files onto disk.

Every download is resumable and idempotent: a file whose registry constraints
already match is left alone. Exact draft artifacts are fetched alongside their
target and checksum-verified. Kimi additionally needs its five parts
concatenated and one header byte corrected; profiles.json describes that here.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from slimserve import term
from slimserve.registry import Plan, cache_root, files_for

_CHUNK = 32 << 20


def _complete(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def _destination(entry: dict[str, Any]) -> Path:
    return cache_root() / entry["local_dir"] / entry["path"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _valid(path: Path, entry: dict[str, Any]) -> bool:
    if not _complete(path, entry["bytes"]):
        return False
    expected = entry.get("sha256")
    return expected is None or _sha256(path) == expected


def _pending(plan: Plan) -> list[tuple[dict[str, Any], Path]]:
    entries = files_for(plan)
    if plan.quant.assembly and _complete(plan.entry_file, plan.quant.assembly["bytes"]):
        # The assembled file is what gets served; the model parts are scaffolding.
        entries = [entry for entry in entries if entry["role"] != "model"]
    return [
        (entry, destination)
        for entry in entries
        if not _valid(destination := _destination(entry), entry)
    ]


def missing(plan: Plan) -> list[dict[str, Any]]:
    """Registry entries whose local copy is absent, damaged, or wrong-sized."""
    return [entry for entry, _destination in _pending(plan)]


def total_bytes(entries: list[dict[str, Any]]) -> int:
    return sum(entry["bytes"] for entry in entries)


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _download(entry: dict[str, Any], dest: Path) -> None:
    """Fetch one file, resuming a partial `.part` if there is one."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > entry["bytes"]:
        part.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    label = Path(entry["path"]).name
    with requests.get(
        entry["url"], headers=headers, stream=True, timeout=60
    ) as response:
        if have and response.status_code == 200:
            # Server ignored the range; start over rather than corrupt the file.
            have = 0
        response.raise_for_status()
        mode = "ab" if have else "wb"
        done = have
        with part.open(mode) as fp:
            for chunk in response.iter_content(_CHUNK):
                fp.write(chunk)
                done += len(chunk)
                pct = 100.0 * done / entry["bytes"] if entry["bytes"] else 100.0
                term.progress(
                    f"{label}: {term.human_bytes(done)}"
                    f"/{term.human_bytes(entry['bytes'])} ({pct:.1f}%)"
                )
    term.progress(f"{label}: {term.human_bytes(done)} complete", done=True)

    if done != entry["bytes"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{label}: got {done} bytes, registry says {entry['bytes']}")
    expected_hash = entry.get("sha256")
    if expected_hash is not None:
        actual_hash = _sha256(part)
        if actual_hash != expected_hash:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"{label}: SHA-256 is {actual_hash}, registry says {expected_hash}"
            )
    part.replace(dest)


def _assemble(plan: Plan) -> None:
    """Concatenate the parts in order, then apply the header patch."""
    spec = plan.quant.assembly
    assert spec is not None
    out = plan.model_dir / spec["output"]
    if _complete(out, spec["bytes"]):
        return

    parts = [plan.model_dir / entry["path"] for entry in plan.quant.files]
    term.step(
        f"assembling {spec['output']} from {len(parts)} parts "
        f"({term.human_bytes(spec['bytes'])})"
    )
    staging = out.with_suffix(out.suffix + ".building")
    written = 0
    with staging.open("wb") as sink:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(_CHUNK):
                    sink.write(chunk)
                    written += len(chunk)
                    term.progress(
                        f"{spec['output']}: {term.human_bytes(written)}"
                        f"/{term.human_bytes(spec['bytes'])}"
                    )
    term.progress(f"{spec['output']}: {term.human_bytes(written)} written", done=True)
    if written != spec["bytes"]:
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"assembled {written} bytes, registry says {spec['bytes']}")

    with staging.open("r+b") as fp:
        for patch in spec.get("patch", []):
            fp.seek(patch["offset"])
            current = fp.read(1)[0]
            if current == patch["to"]:
                continue
            if current != patch["from"]:
                staging.unlink(missing_ok=True)
                raise RuntimeError(
                    f"byte {patch['offset']} is {current}, expected "
                    f"{patch['from']} or {patch['to']} -- refusing to patch"
                )
            fp.seek(patch["offset"])
            fp.write(bytes([patch["to"]]))
            term.ok(f"patched byte {patch['offset']}: {patch['why']}")
    staging.replace(out)

    if not spec.get("keep_parts", True):
        for part in parts:
            part.unlink(missing_ok=True)
        term.info(f"removed {len(parts)} source parts")


def ensure(plan: Plan, assume_yes: bool = False) -> None:
    """Make sure every file the plan needs is present, downloading if not."""
    outstanding = _pending(plan)
    if outstanding:
        entries = [entry for entry, _destination in outstanding]
        need = total_bytes(entries)
        plan.model_dir.mkdir(parents=True, exist_ok=True)
        available = free_bytes(plan.model_dir)
        # An assembled model is written alongside its parts, so the transient
        # peak is both. Say so before starting, not 8 hours in.
        peak = need
        if plan.quant.assembly and any(
            entry["role"] == "model" for entry in outstanding
        ):
            peak += plan.quant.assembly["bytes"]

        term.note(
            f"{len(outstanding)} file(s) to fetch, "
            f"{term.human_bytes(need)} into {plan.model_dir}"
        )
        if peak > available:
            term.die(
                f"needs up to {term.human_bytes(peak)} free and "
                f"{plan.model_dir} has {term.human_bytes(available)}"
            )
        if not assume_yes and not _confirm():
            term.die("cancelled", code=1)

        for entry, destination in outstanding:
            _download(entry, destination)

    if plan.quant.assembly:
        _assemble(plan)


def _confirm() -> bool:
    try:
        answer = input("proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")
