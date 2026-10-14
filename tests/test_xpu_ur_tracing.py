# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the per-process XPU UR/XPTI tracing workaround."""

import os
import sys
import types
from types import ModuleType

import pytest

from vllm.platforms.xpu import XPUPlatform
from vllm.platforms.xpu_env import disable_ur_tracing

XPTI_DISABLED = {
    "XPTI_TRACE_ENABLE": "0",
    "XPTI_SUBSCRIBERS": "",
    "XPTI_FRAMEWORK_DISPATCHER": "",
}
TRACKED = ("UR_ENABLE_LAYERS", "VLLM_XPU_KEEP_UR_TRACING", *XPTI_DISABLED)


@pytest.fixture(autouse=True)
def isolate_tracing_environment(monkeypatch):
    for name in TRACKED:
        # Track absent variables too, so direct writes are undone after the test.
        monkeypatch.setenv(name, "fixture")
        monkeypatch.delenv(name)


def _xpti():
    return {name: os.environ.get(name) for name in XPTI_DISABLED}


@pytest.mark.parametrize("populated", [False, True])
def test_xpti_tracing_is_disabled_without_changing_unrelated_environment(
    monkeypatch, populated
):
    if populated:
        for name in XPTI_DISABLED:
            monkeypatch.setenv(name, "enabled-by-runtime")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1,2,3")
    monkeypatch.setenv("UR_LOG_LOADER", "level:warning")
    disable_ur_tracing()
    disable_ur_tracing()
    assert _xpti() == XPTI_DISABLED
    assert "UR_ENABLE_LAYERS" not in os.environ
    assert os.environ["ZE_AFFINITY_MASK"] == "0,1,2,3"
    assert os.environ["UR_LOG_LOADER"] == "level:warning"


@pytest.mark.parametrize(
    ("layers", "expected"),
    [
        ("UR_LAYER_TRACING", None),
        (
            "UR_LAYER_PARAMETER_VALIDATION;UR_LAYER_TRACING;UR_LAYER_LEAK_CHECKING",
            "UR_LAYER_PARAMETER_VALIDATION;UR_LAYER_LEAK_CHECKING",
        ),
        ("UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION", "UR_LAYER_PARAMETER_VALIDATION"),
        ("UR_LAYER_PARAMETER_VALIDATION", "UR_LAYER_PARAMETER_VALIDATION"),
    ],
)
def test_only_the_tracing_layer_is_removed(monkeypatch, layers, expected):
    monkeypatch.setenv("UR_ENABLE_LAYERS", layers)
    disable_ur_tracing()
    assert os.environ.get("UR_ENABLE_LAYERS") == expected


def test_escape_hatch_keeps_tracing(monkeypatch):
    monkeypatch.setenv("VLLM_XPU_KEEP_UR_TRACING", "1")
    monkeypatch.setenv("UR_ENABLE_LAYERS", "UR_LAYER_TRACING")
    monkeypatch.setenv("XPTI_TRACE_ENABLE", "1")
    disable_ur_tracing()
    assert os.environ["UR_ENABLE_LAYERS"] == "UR_LAYER_TRACING"
    assert os.environ["XPTI_TRACE_ENABLE"] == "1"


def test_import_kernels_clears_runtime_tracing_before_registration(monkeypatch):
    observed = []
    module = ModuleType("vllm.platforms.xpu_c_ops")
    module.register_xpu_c_ops = lambda: observed.append(
        (os.environ.get("UR_ENABLE_LAYERS"), _xpti())
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for _ in range(2):
        # Model runtime imports enabling tracing anew in each worker.
        monkeypatch.setenv("UR_ENABLE_LAYERS", "UR_LAYER_TRACING")
        for name in XPTI_DISABLED:
            monkeypatch.setenv(name, "runtime-setting")
        XPUPlatform.import_kernels()
    assert observed == [(None, XPTI_DISABLED), (None, XPTI_DISABLED)]


def test_platform_detection_scrubs_before_the_first_torch_xpu_call(monkeypatch):
    """xpu_platform_plugin must clear tracing before torch.xpu initialises
    the UR loader, which reads UR_ENABLE_LAYERS once."""
    import torch

    import vllm.platforms as platforms

    seen = []

    def is_available():
        seen.append(os.environ.get("UR_ENABLE_LAYERS"))
        return False

    monkeypatch.setattr(torch._C, "_has_xpu", True, raising=False)
    monkeypatch.setattr(
        torch, "xpu", types.SimpleNamespace(is_available=is_available)
    )
    monkeypatch.setenv("UR_ENABLE_LAYERS", "UR_LAYER_TRACING;UR_LAYER_LEAK_CHECKING")

    assert platforms.xpu_platform_plugin() is None
    assert seen == ["UR_LAYER_LEAK_CHECKING"]


def test_platform_detection_leaves_non_xpu_torch_alone(monkeypatch):
    import torch

    import vllm.platforms as platforms

    monkeypatch.setattr(torch._C, "_has_xpu", False, raising=False)
    monkeypatch.setenv("UR_ENABLE_LAYERS", "UR_LAYER_TRACING")

    assert platforms.xpu_platform_plugin() is None
    assert os.environ["UR_ENABLE_LAYERS"] == "UR_LAYER_TRACING"
