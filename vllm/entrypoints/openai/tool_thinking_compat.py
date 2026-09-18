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
    """Apply profile-selected thinking defaults for structured tool calls.

    GLM-5.3's thinking preamble can consume or corrupt the structural
    prefix expected by tool parsers. Default native function calls to
    non-thinking, while still permitting an explicit native-tool opt-in.
    Custom/free-form tools are always forced to non-thinking because their
    structural grammar otherwise produces empty or incomplete calls.
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

    force_disable = any(tool_type(tool) == "custom" for tool in tools)
    if force_disable or "enable_thinking" not in user_kwargs:
        # Set both common switches: enable_thinking is canonical for GLM, and
        # thinking prevents a stale serving default from re-enabling it in a
        # downstream template or compatibility layer.
        extra_kwargs["enable_thinking"] = False
        extra_kwargs["thinking"] = False
