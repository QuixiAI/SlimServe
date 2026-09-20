# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the per-process XPU tracing workaround."""

import os
import sys
from types import ModuleType

import pytest

from vllm.platforms.xpu import XPUPlatform, disable_ur_tracing

DISABLED = {
    "UR_ENABLE_LAYERS": "",
    "XPTI_TRACE_ENABLE": "0",
    "XPTI_SUBSCRIBERS": "",
    "XPTI_FRAMEWORK_DISPATCHER": "",
}


@pytest.fixture(autouse=True)
def isolate_tracing_environment(monkeypatch):
    for name in DISABLED:
        # Track absent variables too, so direct writes are undone after the test.
        monkeypatch.setenv(name, "fixture")
        monkeypatch.delenv(name)


@pytest.mark.parametrize("populated", [False, True])
def test_tracing_is_disabled_without_changing_unrelated_environment(
    monkeypatch, populated
):
    if populated:
        for name in DISABLED:
            monkeypatch.setenv(name, "enabled-by-runtime")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1,2,3")
    monkeypatch.setenv("UR_LOG_LOADER", "level:warning")
    disable_ur_tracing()
    disable_ur_tracing()
    assert {name: os.environ[name] for name in DISABLED} == DISABLED
    assert os.environ["ZE_AFFINITY_MASK"] == "0,1,2,3"
    assert os.environ["UR_LOG_LOADER"] == "level:warning"


def test_import_kernels_clears_runtime_tracing_before_registration(monkeypatch):
    observed = []
    module = ModuleType("vllm.platforms.xpu_c_ops")
    module.register_xpu_c_ops = lambda: observed.append(
        {name: os.environ.get(name) for name in DISABLED}
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for _ in range(2):
        # Model runtime imports enabling tracing anew in each worker.
        for name in DISABLED:
            monkeypatch.setenv(name, "runtime-setting")
        XPUPlatform.import_kernels()
    assert observed == [DISABLED, DISABLED]
