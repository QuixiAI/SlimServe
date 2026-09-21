# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from typing import Any

import vllm.envs as envs


def glm53_tool_compat_enabled() -> bool:
    """Whether the active SlimServe profile selected GLM-5.3 tool semantics."""
    return envs.VLLM_TOOL_CALLING_PROFILE == "glm53"


def apply_tool_thinking_compat(
    *,
    tools: Sequence[Any] | None,
    user_kwargs: Mapping[str, Any],
    extra_kwargs: dict[str, Any],
) -> None:
    """Force non-thinking for requests that carry custom/free-form tools.

    A custom tool's structural grammar produces empty or incomplete calls
    behind GLM-5.3's thinking preamble, so those requests render without it.
    Native function tools keep the serving default (thinking on): whether a
    request OFFERS tools says nothing about whether the model will call one,
    and agentic traffic offers tools on every turn, so defaulting those
    requests to non-thinking would switch reasoning off for all of it.
    """
    if not tools or not glm53_tool_compat_enabled():
        return

    def tool_type(tool: Any) -> str | None:
        value = (
            tool.get("type")
            if isinstance(tool, Mapping)
            else getattr(tool, "type", None)
        )
        return getattr(value, "value", value)

    if any(tool_type(tool) == "custom" for tool in tools):
        # Set both common switches: enable_thinking is canonical for GLM, and
        # thinking prevents a stale serving default from re-enabling it in a
        # downstream template or compatibility layer.
        extra_kwargs["enable_thinking"] = False
        extra_kwargs["thinking"] = False
