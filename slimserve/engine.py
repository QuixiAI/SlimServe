# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn a Plan into a running engine, or into a server command line.

The same resolved settings drive both modes, so an answer from the chat REPL
and an answer from the served endpoint come from an identically configured
engine.
"""

from __future__ import annotations

import json
import os
from typing import Any

from slimserve.registry import (
    Plan,
    ProfileError,
    validate_cache_policy,
    validate_vision_policy,
)

# Settings the offline LLM class does not take as keyword arguments; they are
# server-side concepts and get dropped when building an in-process engine.
_SERVE_ONLY = frozenset(
    {
        "served_model_name",
        "default_chat_template_kwargs",
        "enable_auto_tool_choice",
        # An api_server flag, not an EngineArgs field; serve_argv reads
        # plan.engine directly so serving still gets it.
        "tool_call_parser",
    }
)


def validate_plan(plan: Plan) -> None:
    validate_cache_policy(plan.engine)
    spec = _speculative_config(plan)
    if spec:
        validate_cache_policy(spec)
    validate_vision_policy(plan)
    env = {**plan.env, **os.environ}
    if env.get("VLLM_QWEN4_EXP_TQ_MAIN_KV", "0").lower() in ("1", "true"):
        raise ProfileError("VLLM_QWEN4_EXP_TQ_MAIN_KV enables prohibited TurboQuant")
    if "turboquant" in env.get("VLLM_ATTENTION_BACKEND", "").lower():
        raise ProfileError("VLLM_ATTENTION_BACKEND cannot enable TurboQuant")


def apply_env(plan: Plan) -> None:
    """Export the profile's environment. Anything already set by the user wins."""
    validate_plan(plan)
    for key, value in plan.env.items():
        os.environ.setdefault(key, value)


def _speculative_config(plan: Plan) -> dict[str, Any] | None:
    if not plan.speculative:
        return None
    spec = plan.speculator
    if not spec:
        return None
    from slimserve.registry import cache_root

    local = cache_root() / spec["local_dir"]
    config: dict[str, Any] = {}
    if file := spec.get("file"):
        draft = str(local / file["path"])
    elif local.is_dir():
        draft = str(local)
    else:
        # Loading straight from the hub: carry the pinned revision so the
        # drafter cannot drift from the validated snapshot.
        draft = spec["repo"]
        if revision := spec.get("revision"):
            config["revision"] = revision
    return {
        "model": draft,
        **config,
        **spec["engine"],
        **plan.speculative_overrides,
    }


def engine_kwargs(plan: Plan) -> dict[str, Any]:
    """Keyword arguments for an in-process `LLM`, for tests and one-off scripts.

    Serving does not use this: both `--serve` and the chat prompt go through
    `serve_argv` so there is only one configured path.
    """
    validate_plan(plan)
    kwargs = {
        key: value for key, value in plan.engine.items() if key not in _SERVE_ONLY
    }
    kwargs["model"] = str(plan.entry_file)
    spec = _speculative_config(plan)
    if spec is not None:
        kwargs["speculative_config"] = spec
    return kwargs


def serve_argv(plan: Plan, host: str, port: int) -> list[str]:
    """Command line for `vllm.entrypoints.openai.api_server`."""
    validate_plan(plan)
    argv = ["--model", str(plan.entry_file), "--host", host, "--port", str(port)]
    for key, value in plan.engine.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key == "async_scheduling":
                # Omission means auto, not false. The bounded routing journal
                # requires an explicit synchronous scheduler for step metadata.
                argv.append("--no-async-scheduling")
        elif isinstance(value, (dict, list)):
            argv += [flag, json.dumps(value)]
        else:
            argv += [flag, str(value)]
    spec = _speculative_config(plan)
    if spec is not None:
        argv += ["--speculative-config", json.dumps(spec)]
    return argv
