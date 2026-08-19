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

from slimserve.registry import Plan

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


def apply_env(plan: Plan) -> None:
    """Export the profile's environment. Anything already set by the user wins."""
    for key, value in plan.env.items():
        os.environ.setdefault(key, value)


def _speculative_config(plan: Plan) -> dict[str, Any] | None:
    if not plan.speculative:
        return None
    spec = plan.source.get("speculator")
    if not spec:
        return None
    from slimserve.registry import cache_root

    local = cache_root() / spec["local_dir"]
    if file := spec.get("file"):
        draft = str(local / file["path"])
    else:
        draft = str(local) if local.is_dir() else spec["repo"]
    return {
        "model": draft,
        **spec["engine"],
        **plan.speculative_overrides,
    }


def engine_kwargs(plan: Plan) -> dict[str, Any]:
    """Keyword arguments for an in-process `LLM`, for tests and one-off scripts.

    Serving does not use this: both `--serve` and the chat prompt go through
    `serve_argv` so there is only one configured path.
    """
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
    argv = ["--model", str(plan.entry_file), "--host", host, "--port", str(port)]
    for key, value in plan.engine.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (dict, list)):
            argv += [flag, json.dumps(value)]
        else:
            argv += [flag, str(value)]
    spec = _speculative_config(plan)
    if spec is not None:
        argv += ["--speculative-config", json.dumps(spec)]
    return argv
