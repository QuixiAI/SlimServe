# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The legal-configuration registry, read from profiles.json.

profiles.json is the authority. Nothing here invents a configuration; this
module only resolves (profile, platform, quant) into a concrete plan and
explains, in the user's terms, why a combination is not allowed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("profiles.json")


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    with REGISTRY_PATH.open(encoding="utf-8") as fp:
        return json.load(fp)


def profile_ids() -> list[str]:
    return list(_registry()["profiles"])


def platform_title(platform: str) -> str:
    entry = _registry()["platforms"].get(platform)
    return entry["title"] if entry else platform


@dataclass(frozen=True)
class Quant:
    name: str
    title: str
    bytes: int
    summary: str
    files: list[dict[str, Any]]
    min_gpus: dict[str, int]
    assembly: dict[str, Any] | None

    def allowed_on(self, platform: str, gpus: int) -> bool:
        minimum = self.min_gpus.get(platform)
        return minimum is not None and gpus >= minimum


@dataclass(frozen=True)
class Plan:
    """One fully resolved, legal run."""

    profile_id: str
    title: str
    summary: str
    platform: str
    gpus: int
    source_key: str
    source: dict[str, Any]
    quant: Quant
    engine: dict[str, Any]
    env: dict[str, str]
    speculative: bool
    chat_template_kwargs: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    @property
    def model_dir(self) -> Path:
        return cache_root() / self.source["local_dir"]

    @property
    def entry_file(self) -> Path:
        """The path handed to the engine as --model."""
        if self.quant.assembly:
            return self.model_dir / self.quant.assembly["output"]
        return self.model_dir / self.quant.files[0]["path"]


class ProfileError(Exception):
    """A configuration that is not in the registry, with a readable reason."""


def cache_root() -> Path:
    """Where model files live. SLIMSERVE_CACHE wins, then ~/models."""
    override = os.environ.get("SLIMSERVE_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / "models"


def _quant(source: dict[str, Any], name: str) -> Quant:
    raw = source["quants"][name]
    return Quant(
        name=name,
        title=raw["title"],
        bytes=raw["bytes"],
        summary=raw["summary"],
        files=raw["files"],
        min_gpus=raw["min_gpus"],
        assembly=raw.get("assembly"),
    )


def describe(profile_id: str) -> dict[str, Any]:
    """The raw registry entry, for listing without resolving a platform."""
    profiles = _registry()["profiles"]
    if profile_id not in profiles:
        raise ProfileError(
            f"unknown profile {profile_id!r}; available: {', '.join(profiles)}"
        )
    return profiles[profile_id]


def quants_for(profile_id: str, platform: str) -> list[Quant]:
    """Every quant this profile can legally serve on this platform."""
    profile = describe(profile_id)
    source = _registry()["sources"][profile["source"]]
    gpus = profile["gpus"]
    return [
        quant
        for quant in (_quant(source, name) for name in source["quants"])
        if quant.allowed_on(platform, gpus)
    ]


def _merge_platform(profile: dict[str, Any], platform: str) -> dict[str, Any]:
    """Apply a platform's overrides over the profile's base settings."""
    override = (profile.get("platform_overrides") or {}).get(platform)
    engine = dict(profile["engine"])
    env = dict(profile.get("env") or {})
    default_quant = profile["default_quant"]
    if override:
        for key, value in (override.get("engine") or {}).items():
            if value is None:
                engine.pop(key, None)
            else:
                engine[key] = value
        if "env" in override:
            env = dict(override["env"])
        default_quant = override.get("default_quant", default_quant)
    return {"engine": engine, "env": env, "default_quant": default_quant}


def resolve(profile_id: str, platform: str, gpus: int, quant: str | None) -> Plan:
    """Turn a request into a Plan, or explain why it is not legal.

    `gpus` is what the machine has; the profile decides how many it uses.
    """
    profile = describe(profile_id)
    supported = profile["platforms"]
    if platform not in supported:
        raise ProfileError(
            f"{profile_id} is not supported on {platform_title(platform)}; "
            f"it runs on {', '.join(platform_title(p) for p in supported)}"
        )
    needed = profile["gpus"]
    if gpus < needed:
        raise ProfileError(
            f"{profile_id} needs {needed} GPUs and this machine shows {gpus}"
        )

    merged = _merge_platform(profile, platform)
    source_key = profile["source"]
    source = _registry()["sources"][source_key]
    name = quant or merged["default_quant"]
    if name not in source["quants"]:
        raise ProfileError(
            f"unknown quant {name!r} for {source['title']}; "
            f"available: {', '.join(source['quants'])}"
        )
    chosen = _quant(source, name)
    if not chosen.allowed_on(platform, needed):
        minimum = chosen.min_gpus.get(platform)
        if minimum is None:
            raise ProfileError(
                f"{chosen.title} is not supported on {platform_title(platform)}"
            )
        raise ProfileError(
            f"{chosen.title} needs at least {minimum} "
            f"{platform_title(platform)} GPUs and {profile_id} uses {needed}; "
            f"try {_suggest(profile_id, platform, minimum)}"
        )

    return Plan(
        profile_id=profile_id,
        title=profile["title"],
        summary=profile["summary"],
        platform=platform,
        gpus=needed,
        source_key=source_key,
        source=source,
        quant=chosen,
        engine=merged["engine"],
        env=merged["env"],
        speculative=bool(profile.get("speculative")),
        chat_template_kwargs=dict(profile.get("chat_template_kwargs") or {}),
        notes=list(profile.get("notes") or []),
    )


def _suggest(profile_id: str, platform: str, minimum: int) -> str:
    """Name a profile that would satisfy the quant, for the error message."""
    source = describe(profile_id)["source"]
    for other, entry in _registry()["profiles"].items():
        if (
            entry["source"] == source
            and platform in entry["platforms"]
            and entry["gpus"] >= minimum
        ):
            return other
    return "a larger machine"


def files_for(plan: Plan) -> list[dict[str, Any]]:
    """Every file that must be present locally, download URL included."""
    base = plan.source["base_url"]
    wanted: list[dict[str, Any]] = []
    for entry in plan.quant.files:
        wanted.append(
            {
                **entry,
                "url": f"{base}/{entry['path']}",
                "local_dir": plan.source["local_dir"],
                "role": "model",
            }
        )
    for entry in plan.source.get("shared") or []:
        entry_base = entry.get("base_url", base)
        wanted.append(
            {
                **entry,
                "url": f"{entry_base}/{entry['path']}",
                "local_dir": plan.source["local_dir"],
                "role": "shared",
            }
        )
    spec = plan.source.get("speculator") if plan.speculative else None
    if spec and (entry := spec.get("file")):
        wanted.append(
            {
                **entry,
                "url": f"{spec['base_url']}/{entry['path']}",
                "local_dir": spec["local_dir"],
                "role": "speculator",
            }
        )
    return wanted
