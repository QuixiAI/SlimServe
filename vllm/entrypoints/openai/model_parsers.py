# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which reasoning and tool parsers a served model needs.

Each architecture here frames its thinking and its tool calls differently, and
picking the wrong parser fails quietly rather than loudly: the reply arrives
with the frame still attached, or lands entirely in `reasoning` with `content`
null, or a tool call is returned as prose. None of that looks like a
misconfiguration to the caller.

So the model chooses, not the command line. The GGUF already states its own
architecture and reading it costs a metadata parse, which is cached. An
explicit `--reasoning-parser` / `--tool-call-parser` still wins.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

# architecture -> (reasoning parser, tool-call parser)
_PARSERS_BY_ARCHITECTURE: dict[str, tuple[str, str]] = {
    "glm-dsa": ("glm45", "glm47"),
    "deepseek4": ("deepseek_v4", "deepseek_v4"),
    "kimi-k3": ("kimi_k3", "kimi_k3"),
}


def parsers_for_model(model: str | None) -> tuple[str, str] | None:
    """The (reasoning, tool) parser pair for a model path, if it is a GGUF we
    recognise. Returns None when the architecture is unknown or unreadable, so
    the caller keeps whatever default it had.
    """
    if not model:
        return None
    try:
        from vllm.transformers_utils.gguf_utils import gguf_architecture

        architecture = gguf_architecture(model)
    except Exception:
        # Not a GGUF, not readable yet, or an architecture the reader does not
        # know. Either way this is a hint, not a gate.
        return None
    return _PARSERS_BY_ARCHITECTURE.get(architecture)


def apply_default_parsers(args) -> None:
    """Fill in `reasoning_parser`/`tool_call_parser` from the model.

    Only fills what the caller left at its default, so an explicit flag is
    never overridden.
    """
    chosen = parsers_for_model(getattr(args, "model", None))
    if chosen is None:
        return
    reasoning, tool = chosen
    for attribute, value in (
        ("reasoning_parser", reasoning),
        ("tool_call_parser", tool),
    ):
        current = getattr(args, attribute, None)
        if current and current not in (_DEFAULTS.get(attribute), value):
            # The operator asked for something specific; leave it alone.
            continue
        if current != value:
            setattr(args, attribute, value)
            logger.info_once(
                "Selected --%s %s for this model", attribute.replace("_", "-"), value
            )


# The values the CLI carries when nobody chose: both are GLM's, because this
# fork grew up serving GLM. Treat them as "unset" for a non-GLM checkpoint.
_DEFAULTS = {"reasoning_parser": "glm45", "tool_call_parser": "glm47"}
