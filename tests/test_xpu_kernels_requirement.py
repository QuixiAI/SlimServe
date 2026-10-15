# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup warning when the required vllm-xpu-kernels build is missing.

Without torch.ops._xpu_C.nvfp4_gemm / gdn_attention the XPU paths fall back
silently; the platform must say so once at startup. Runs on CPU with the
kernel module and op namespace stubbed.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.platforms import xpu_c_ops


def _ops(*names):
    return SimpleNamespace(
        ops=SimpleNamespace(_xpu_C=SimpleNamespace(**dict.fromkeys(names, object())))
    )


@pytest.fixture
def kernels_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels._xpu_C", SimpleNamespace())


def test_warns_when_package_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels._xpu_C", None)
    with patch.object(xpu_c_ops.logger, "warning_once") as warn:
        missing = xpu_c_ops.warn_if_xpu_kernels_missing()
    assert missing == ["nvfp4_gemm", "gdn_attention"]
    warn.assert_called_once()
    assert "b60-serving-port" in warn.call_args.args[-1]


def test_warns_when_build_lacks_nvfp4_gemm(kernels_importable):
    with (
        patch.object(xpu_c_ops, "torch", _ops("gdn_attention")),
        patch.object(xpu_c_ops.logger, "warning_once") as warn,
    ):
        assert xpu_c_ops.warn_if_xpu_kernels_missing() == ["nvfp4_gemm"]
    warn.assert_called_once()


def test_silent_with_required_build(kernels_importable):
    with (
        patch.object(xpu_c_ops, "torch", _ops("nvfp4_gemm", "gdn_attention")),
        patch.object(xpu_c_ops.logger, "warning_once") as warn,
    ):
        assert xpu_c_ops.warn_if_xpu_kernels_missing() == []
    warn.assert_not_called()


def test_check_and_update_config_runs_the_check():
    import inspect

    from vllm.platforms.xpu import XPUPlatform

    src = inspect.getsource(XPUPlatform.check_and_update_config)
    assert "warn_if_xpu_kernels_missing()" in src
