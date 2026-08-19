# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live smoke-test every SlimServe profile compatible with this machine.

Profiles are discovered from the registry instead of copied into this script.
Each compatible profile is loaded in isolation, checked for the required
DSpark/TurboQuant configuration, given a text request, and—when its registered
modalities include images—a deterministic image request. The complete matrix
is attempted by default so one failed profile cannot hide later omissions.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pybase64 as base64
import regex as re

from slimserve import fetch, hardware, registry
from slimserve.engine import engine_kwargs
from slimserve.registry import Plan, ProfileError
from slimserve.server import Server
from slimserve.stream import chat_completion, visible_text

_TEXT_PROMPT = "What is 2 + 2? Reply with only the number."
_IMAGE_PROMPT = "What is the dominant color in this image? Reply with one word."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help=(
            "test only this profile; repeat to select several (default: all compatible)"
        ),
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="download missing registered artifacts without prompting",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failure instead of completing the matrix",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=3600.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="engine log directory (default: a new directory under /tmp)",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON result here")
    return parser.parse_args()


def compatible_profile_ids(machine: hardware.Machine) -> list[str]:
    """Return every profile the registry says can run on ``machine``."""
    if not machine.known or machine.platform is None:
        return []

    compatible: list[str] = []
    for profile_id in registry.profile_ids():
        try:
            registry.resolve(
                profile_id,
                machine.platform,
                machine.count,
                None,
                machine.memory_bytes,
            )
        except ProfileError:
            continue
        compatible.append(profile_id)
    return compatible


def resolve_profiles(
    machine: hardware.Machine, requested: list[str] | None
) -> list[Plan]:
    """Resolve an explicit selection or the complete compatible matrix."""
    if not machine.known or machine.platform is None:
        raise RuntimeError(f"unsupported machine: {machine.device_name}")
    if blocked := registry.platform_blocked(machine.platform):
        raise RuntimeError(f"{registry.platform_title(machine.platform)}: {blocked}")

    compatible = compatible_profile_ids(machine)
    selected = requested or compatible
    if not selected:
        raise RuntimeError("no compatible profiles found")

    unknown = set(selected) - set(registry.profile_ids())
    if unknown:
        raise RuntimeError(f"unknown profile(s): {', '.join(sorted(unknown))}")
    incompatible = set(selected) - set(compatible)
    if incompatible:
        raise RuntimeError(
            f"profile(s) not compatible with this machine: "
            f"{', '.join(sorted(incompatible))}"
        )

    return [
        registry.resolve(
            profile_id,
            machine.platform,
            machine.count,
            None,
            machine.memory_bytes,
        )
        for profile_id in selected
    ]


def validate_acceleration(plan: Plan) -> dict[str, Any]:
    """Require the resolved, executable plan to use DSpark and TurboQuant."""
    speculative = engine_kwargs(plan).get("speculative_config")
    if not isinstance(speculative, dict):
        raise RuntimeError("resolved plan has no speculative configuration")
    required = {
        "method": "dspark",
        "attention_backend": "TURBOQUANT",
        "kv_cache_dtype": "turboquant_k8v4",
    }
    mismatches = {
        key: speculative.get(key)
        for key, expected in required.items()
        if speculative.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"resolved acceleration mismatch: {mismatches}")
    return speculative


def _red_image_data_url() -> str:
    """Build a deterministic image without a network or repository fixture."""
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _request(
    plan: Plan,
    base_url: str,
    prompt: str,
    *,
    image_url: str | None,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    content: str | list[dict[str, Any]] = prompt
    if image_url is not None:
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
    messages = [{"role": "user", "content": content}]
    started = time.perf_counter()
    raw = "".join(
        chat_completion(
            base_url,
            plan.engine.get("served_model_name", "model"),
            messages,
            max_tokens=max_tokens,
            # No sampling overrides: the model's shipped defaults apply.
            # Seeded for repeatable smoke answers; greedy is never used.
            seed=42,
            chat_template_kwargs=plan.chat_template_kwargs or None,
            timeout=timeout,
        )
    )
    answer = visible_text(raw)
    return {
        "answer": answer[:500],
        "seconds": time.perf_counter() - started,
    }


def _require_match(result: dict[str, Any], pattern: str, label: str) -> None:
    answer = str(result["answer"])
    if not re.search(pattern, answer, flags=re.IGNORECASE):
        raise RuntimeError(f"{label} check failed; answer was {answer!r}")


def run_profile(
    plan: Plan,
    *,
    log_dir: Path,
    download_missing: bool,
    max_tokens: int,
    startup_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    """Load and smoke-test one profile, always stopping it before returning."""
    speculative = validate_acceleration(plan)
    missing = fetch.missing(plan)
    if missing and not download_missing:
        names = ", ".join(entry["path"] for entry in missing)
        raise RuntimeError(
            f"missing registered artifacts: {names}; use --download-missing"
        )
    fetch.ensure(plan, assume_yes=True)

    log_path = log_dir / f"{plan.profile_id}.log"
    started = time.perf_counter()
    with Server(plan) as server:
        server.start(log_path=str(log_path))
        server.wait_until_ready(timeout=startup_timeout)
        loaded = time.perf_counter()

        text_result = _request(
            plan,
            server.base_url,
            _TEXT_PROMPT,
            image_url=None,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )
        _require_match(text_result, r"(?<!\d)4(?!\d)", "text")

        image_result = None
        if "image" in plan.source["modalities"]:
            image_result = _request(
                plan,
                server.base_url,
                _IMAGE_PROMPT,
                image_url=_red_image_data_url(),
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            _require_match(image_result, r"\bred\b", "image")

    return {
        "profile": plan.profile_id,
        "source": plan.source_key,
        "quant": plan.quant.name,
        "gpus": plan.gpus,
        "modalities": plan.source["modalities"],
        "dspark_tokens": speculative["num_speculative_tokens"],
        "draft_attention_backend": speculative["attention_backend"],
        "draft_kv_cache_dtype": speculative["kv_cache_dtype"],
        "load_seconds": loaded - started,
        "text": text_result,
        "image": image_result,
        "log": str(log_path),
        "passed": True,
    }


def main() -> int:
    args = parse_args()
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1")

    machine = hardware.detect()
    try:
        plans = resolve_profiles(machine, args.profiles)
    except RuntimeError as error:
        print(f"smoke_profiles: {error}", file=sys.stderr)
        return 2

    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="slimserve-smoke-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, plan in enumerate(plans, start=1):
        print(
            f"[{index}/{len(plans)}] {plan.profile_id}: {plan.quant.name} "
            f"on {plan.gpus} GPU(s)",
            file=sys.stderr,
            flush=True,
        )
        try:
            result = run_profile(
                plan,
                log_dir=log_dir,
                download_missing=args.download_missing,
                max_tokens=args.max_tokens,
                startup_timeout=args.startup_timeout,
                request_timeout=args.request_timeout,
            )
        except Exception as error:
            result = {
                "profile": plan.profile_id,
                "source": plan.source_key,
                "quant": plan.quant.name,
                "gpus": plan.gpus,
                "modalities": plan.source["modalities"],
                "log": str(log_dir / f"{plan.profile_id}.log"),
                "passed": False,
                "error": str(error),
            }
        results.append(result)
        if not result["passed"] and args.fail_fast:
            break

    compatible = compatible_profile_ids(machine)
    payload = {
        "machine": {
            "platform": machine.platform,
            "device_name": machine.device_name,
            "visible_devices": machine.count,
        },
        "compatible_profiles": compatible,
        "selected_profiles": [plan.profile_id for plan in plans],
        "all_passed": len(results) == len(plans)
        and all(result["passed"] for result in results),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
