# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parser selection is the model's business, not the caller's.

Picking the wrong reasoning or tool parser fails quietly -- the frame stays
attached, or the whole reply lands in `reasoning`, or a tool call comes back as
prose -- so serving a model must not depend on remembering two flags.
"""

import argparse

import pytest

from vllm.entrypoints.openai.model_parsers import (
    _PARSERS_BY_ARCHITECTURE,
    apply_default_parsers,
    parsers_for_model,
)

GLM_DEFAULTS = {"reasoning_parser": "glm45", "tool_call_parser": "glm47"}


def _args(**kwargs):
    return argparse.Namespace(**{**GLM_DEFAULTS, "model": "m.gguf", **kwargs})


@pytest.fixture
def architecture(monkeypatch):
    """Pretend the GGUF at any path reports the given architecture."""

    def use(name):
        monkeypatch.setattr(
            "vllm.transformers_utils.gguf_utils.gguf_architecture",
            lambda _path: name,
        )

    return use


@pytest.mark.parametrize(
    ("arch", "reasoning", "tool"),
    [
        ("glm-dsa", "glm45", "glm47"),
        ("deepseek4", "deepseek_v4", "deepseek_v4"),
        ("kimi-k3", "kimi_k3", "kimi_k3"),
        ("muse-glimmer", "muse_glimmer", "muse_glimmer"),
    ],
)
def test_each_architecture_selects_its_own_parsers(architecture, arch, reasoning, tool):
    architecture(arch)
    args = _args()
    apply_default_parsers(args)
    assert (args.reasoning_parser, args.tool_call_parser) == (reasoning, tool)


def test_an_explicit_choice_is_never_overridden(architecture):
    architecture("kimi-k3")
    args = _args(reasoning_parser="deepseek_v4", tool_call_parser="glm47")
    apply_default_parsers(args)
    assert args.reasoning_parser == "deepseek_v4"
    # glm47 is the CLI default, so it is treated as unset and corrected.
    assert args.tool_call_parser == "kimi_k3"


def test_an_unreadable_model_leaves_the_defaults_alone(monkeypatch):
    """Not a GGUF, or a path that does not exist yet: this is a hint, not a gate."""

    def boom(_path):
        raise OSError("not a gguf")

    monkeypatch.setattr("vllm.transformers_utils.gguf_utils.gguf_architecture", boom)
    args = _args()
    apply_default_parsers(args)
    assert (args.reasoning_parser, args.tool_call_parser) == (
        GLM_DEFAULTS["reasoning_parser"],
        GLM_DEFAULTS["tool_call_parser"],
    )


def test_an_unknown_architecture_leaves_the_defaults_alone(architecture):
    architecture("something-new")
    args = _args()
    apply_default_parsers(args)
    assert args.tool_call_parser == GLM_DEFAULTS["tool_call_parser"]


def test_no_model_is_not_an_error():
    assert parsers_for_model(None) is None
    assert parsers_for_model("") is None


def test_every_mapped_parser_is_registered():
    """A name here that no manager knows would fail only at serve time."""
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    for reasoning, tool in _PARSERS_BY_ARCHITECTURE.values():
        assert ReasoningParserManager.get_reasoning_parser(reasoning) is not None
        assert ToolParserManager.get_tool_parser(tool) is not None
