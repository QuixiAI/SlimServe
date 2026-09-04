# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The legal-configuration registry, read from profiles.json.

profiles.json is the authority. Nothing here invents a configuration; this
module only resolves (profile, platform, quant) into a concrete plan and
explains, in the user's terms, why a combination is not allowed.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from slimserve.term import human_bytes

REGISTRY_PATH = Path(__file__).with_name("profiles.json")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in profiles.json: {key!r}")
        result[key] = value
    return result


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    with REGISTRY_PATH.open(encoding="utf-8") as fp:
        return json.load(fp, object_pairs_hook=_unique_object)


def profile_ids() -> list[str]:
    return list(_registry()["profiles"])


def platform_title(platform: str) -> str:
    entry = _registry()["platforms"].get(platform)
    return entry["title"] if entry else platform


def platform_gate(platform: str) -> str:
    """What limits this platform: "gpus" (discrete cards) or "memory" (unified)."""
    entry = _registry()["platforms"].get(platform) or {}
    return entry.get("gate", "gpus")


def platform_blocked(platform: str) -> str | None:
    """Why this platform cannot serve yet, or None when it can."""
    entry = _registry()["platforms"].get(platform) or {}
    if entry.get("status", "supported") == "supported":
        return None
    return entry.get("status_reason", "not supported yet")


def platform_blocked_detail(platform: str) -> str:
    """The long form of `platform_blocked`, for the message that stops a run."""
    entry = _registry()["platforms"].get(platform) or {}
    return entry.get("status_detail") or platform_blocked(platform) or ""


def profile_blocked(profile_id: str, platform: str) -> str | None:
    """Why one profile cannot serve on a platform, or None when it can."""
    blocked = platform_blocked(platform)
    if blocked:
        return blocked
    profile = describe(profile_id)
    if platform not in profile["platforms"]:
        return None
    record = profile["variants"][platform]
    status = record.get("status", profile.get("status", "supported"))
    if status == "supported":
        return None
    return record.get(
        "status_reason", profile.get("status_reason", "not supported yet")
    )


def profile_blocked_detail(profile_id: str, platform: str) -> str:
    """Long-form explanation for a profile-specific serving gate."""
    profile = describe(profile_id)
    record = profile["variants"].get(platform) or {}
    return (
        record.get("status_detail")
        or profile.get("status_detail")
        or profile_blocked(profile_id, platform)
        or ""
    )


@dataclass(frozen=True)
class Quant:
    name: str
    title: str
    bytes: int
    summary: str
    files: list[dict[str, Any]]
    min_gpus: dict[str, int]
    min_memory_bytes: dict[str, int]
    min_host_ram_bytes: dict[str, int]
    assembly: dict[str, Any] | None

    def requirement(self, platform: str) -> int | None:
        """The gating figure for this platform: GPU count or bytes of memory."""
        if platform_gate(platform) == "memory":
            return self.min_memory_bytes.get(platform)
        return self.min_gpus.get(platform)

    def allowed_on(
        self,
        platform: str,
        gpus: int,
        memory_bytes: int = 0,
        host_ram_bytes: int = 0,
    ) -> bool:
        minimum = self.requirement(platform)
        if minimum is None:
            return False
        have = memory_bytes if platform_gate(platform) == "memory" else gpus
        if have < minimum:
            return False
        # Host-offload profiles (e.g. a pinned PLE table per rank) need
        # system RAM the GPU gate cannot see. Only enforced when the caller
        # provides a detected figure; 0 means unknown, not zero RAM.
        ram_minimum = self.min_host_ram_bytes.get(platform)
        return not (
            ram_minimum is not None and host_ram_bytes and host_ram_bytes < ram_minimum
        )


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
    speculative_overrides: dict[str, Any]
    chat_template_kwargs: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    # Variant-level drafter when platforms diverge; falls back to the
    # source-level speculator.
    variant_speculator: dict[str, Any] | None = None

    @property
    def speculator(self) -> dict[str, Any] | None:
        return self.variant_speculator or self.source.get("speculator")

    @property
    def model_dir(self) -> Path:
        return cache_root() / self.source["local_dir"]

    @property
    def entry_file(self) -> Path:
        """The path handed to the engine as --model."""
        if self.source.get("format") == "safetensors":
            # A safetensors checkpoint is a directory of shards plus config
            # and tokenizer files; the engine takes the directory itself.
            return self.model_dir
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
        min_memory_bytes=raw.get("min_memory_bytes") or {},
        min_host_ram_bytes=raw.get("min_host_ram_bytes") or {},
        assembly=raw.get("assembly"),
    )


def describe(profile_id: str) -> dict[str, Any]:
    """The registry entry, for listing without resolving a platform.

    A profile is model x quant x platform x config, so the stored entry holds
    one record per platform under `variants`. The id a user types carries no
    platform -- the CLI detects it -- so this returns the shared fields plus
    the platform list, and `variant()` returns the record for one platform.
    """
    profiles = _registry()["profiles"]
    if profile_id not in profiles:
        raise ProfileError(
            f"unknown profile {profile_id!r}; available: {', '.join(profiles)}"
        )
    entry = profiles[profile_id]
    return {**entry, "platforms": list(entry["variants"])}


def variant(profile_id: str, platform: str) -> dict[str, Any]:
    """The record for one platform, or an error naming the ones that exist."""
    entry = describe(profile_id)
    record = entry["variants"].get(platform)
    if record is None:
        raise ProfileError(
            f"{profile_id} is not supported on {platform_title(platform)}; "
            f"it runs on {', '.join(platform_title(p) for p in entry['platforms'])}"
        )
    return record


def quants_for(
    profile_id: str,
    platform: str,
    memory_bytes: int = 0,
    host_ram_bytes: int = 0,
) -> list[Quant]:
    """Every quant this profile can legally serve on this platform."""
    profile = describe(profile_id)
    source = _registry()["sources"][profile["source"]]
    gpus = profile["gpus"]
    return [
        quant
        for quant in (_quant(source, name) for name in source["quants"])
        if quant.allowed_on(platform, gpus, memory_bytes, host_ram_bytes)
    ]


def _merge_platform(profile: dict[str, Any], platform: str) -> dict[str, Any]:
    """The platform's own settings.

    Kept as a function because callers read it as one shape; there is nothing
    left to merge now that each platform has its own record.
    """
    record = profile["variants"][platform]
    return {
        "engine": dict(record["engine"]),
        "env": dict(record.get("env") or {}),
        "notes": list(record.get("notes") or []),
        "default_quant": record["default_quant"],
        "speculative_overrides": dict(record.get("speculative_overrides") or {}),
        # A variant may carry its own drafter when platforms diverge (e.g.
        # qwen38-nvfp4-1: the MI300X variant reuses the checkpoint's MTP
        # head, the Metal variant serves the measured DFlash2 drafter).
        "speculator": record.get("speculator"),
    }


# Serving behavior every profile gets unless it overrides the key itself.
#
# STANDING POLICY (2026-08-28): SlimServe serving ALWAYS has automatic
# prefix caching, automatic tool calling, and thinking enabled, and NEVER
# uses greedy sampling. Prefix caching admits NO opt-out: every platform
# record states enable_prefix_caching true and the registry test rejects
# anything else. For the other keys a profile may override only for a
# model-level impossibility, with a note naming the reason.
# Do not rely on engine defaults for any of these:
# vLLM silently defaults prefix caching OFF for hybrid (mamba/GDN) models,
# which shipped qwen38fn-fp8-8 with a 0.0% cache hit rate and full-history
# re-prefill on every chat turn. Benchmarks and diagnostics use the model's
# recommended sampling (temperature 1.0 / top_p 0.95 / top_k 20, seeded for
# reproducibility), never temperature 0.
#
# Main KV precision is a per-profile, on-box-validated choice (operator
# 2026-09-02, reversing the 2026-08-29 rtx3090 bf16 mandate): a record may
# quantize its main KV when a note names the format and the validation.
# qwen38fn-fp8-8/rtx3090 runs fp8 (e4m3) main KV through kv_cache_dtype;
# TurboQuant is not used for main KV there. Draft-model KV (DSpark
# TurboQuant) is always allowed: rejection sampling verifies drafts against
# the target, so draft precision affects speed, never output content.
# Enforced by test_quantized_main_kv_is_an_explicit_validated_choice.
#
# "thinking" is the DeepSeek/Kimi template switch, "enable_thinking" the
# GLM/Qwen one; templates ignore the name they do not use.
_SERVING_DEFAULTS: dict[str, Any] = {
    "enable_auto_tool_choice": True,
    "default_chat_template_kwargs": {"thinking": True, "enable_thinking": True},
    "enable_prefix_caching": True,
}


def resolve(
    profile_id: str,
    platform: str,
    gpus: int,
    quant: str | None,
    memory_bytes: int = 0,
    host_ram_bytes: int = 0,
) -> Plan:
    """Turn a request into a Plan, or explain why it is not legal.

    `gpus` is what the machine has; the profile decides how many it uses. On a
    unified-memory platform `memory_bytes` is what gates instead.
    """
    profile = describe(profile_id)
    supported = profile["platforms"]
    if platform not in supported:
        raise ProfileError(
            f"{profile_id} is not supported on {platform_title(platform)}; "
            f"it runs on {', '.join(platform_title(p) for p in supported)}"
        )
    needed = profile["gpus"]
    if platform_gate(platform) != "memory" and gpus < needed:
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
    ram_minimum = chosen.min_host_ram_bytes.get(platform)
    if ram_minimum is not None and host_ram_bytes and host_ram_bytes < ram_minimum:
        raise ProfileError(
            f"{chosen.title} pins host-offloaded tables in system RAM and "
            f"needs at least {human_bytes(ram_minimum)}; this machine has "
            f"{human_bytes(host_ram_bytes)}"
        )
    if not chosen.allowed_on(platform, needed, memory_bytes):
        minimum = chosen.requirement(platform)
        if minimum is None:
            raise ProfileError(
                f"{chosen.title} is not supported on {platform_title(platform)}"
            )
        if platform_gate(platform) == "memory":
            smaller = _suggest_quant(profile_id, platform, memory_bytes)
            advice = (
                f"try {smaller}"
                if smaller
                else "no quant of this model fits a machine that size"
            )
            raise ProfileError(
                f"{chosen.title} needs {human_bytes(minimum)} of unified "
                f"memory and this machine has {human_bytes(memory_bytes)}; "
                f"{advice}"
            )
        raise ProfileError(
            f"{chosen.title} needs at least {minimum} "
            f"{platform_title(platform)} GPUs and {profile_id} uses {needed}; "
            f"try {_suggest(profile_id, platform, minimum)}"
        )

    # Quant-specific engine/env adjustments (e.g. a KV byte budget measured
    # for one artifact does not transfer to a larger one). Applied after the
    # platform merge so they win over both base and platform settings.
    quant_override = profile["variants"][platform].get("quant_overrides", {}).get(name)
    if quant_override:
        for key, value in (quant_override.get("engine") or {}).items():
            if value is None:
                merged["engine"].pop(key, None)
            else:
                merged["engine"][key] = value
        for key, value in (quant_override.get("env") or {}).items():
            merged["env"][key] = value

    # Thinking and tool calling are on by default for every profile on every
    # platform. A profile sets these keys itself only to opt out.
    for key, value in _SERVING_DEFAULTS.items():
        merged["engine"].setdefault(key, copy.deepcopy(value))

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
        speculative_overrides=merged["speculative_overrides"],
        chat_template_kwargs=dict(profile.get("chat_template_kwargs") or {}),
        notes=merged["notes"],
        variant_speculator=merged["speculator"],
    )


def _suggest_quant(profile_id: str, platform: str, memory_bytes: int) -> str | None:
    """The largest quant this machine could actually hold, or None if it holds none.

    More memory is not a profile the user can pick, so on a unified-memory
    platform the useful advice is a smaller quant.
    """
    fits = quants_for(profile_id, platform, memory_bytes)
    if not fits:
        return None
    return max(fits, key=lambda quant: quant.bytes).name


def _suggest(profile_id: str, platform: str, minimum: int) -> str:
    """Name a profile that would satisfy the quant, for the error message."""
    source = describe(profile_id)["source"]
    for other, entry in _registry()["profiles"].items():
        if (
            entry["source"] == source
            and platform in entry["variants"]
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
    spec = plan.speculator if plan.speculative else None
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
