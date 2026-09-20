# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization discovery must not load unrelated hardware model modules."""

import subprocess
import sys


def test_quant_override_discovery_does_not_import_device_models():
    source = r"""
import importlib.abc
import sys

class RejectDeviceModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith((
            'vllm.models.deepseek_v4.xpu',
            'vllm.models.deepseek_v4.amd',
            'vllm.models.deepseek_v4.nvidia',
        )):
            raise AssertionError('unrelated device model imported: ' + fullname)

sys.meta_path.insert(0, RejectDeviceModels())
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.models import deepseek_v4
config = get_quantization_config('deepseek_v4_fp8')
assert config is deepseek_v4.DeepseekV4FP8Config
try:
    deepseek_v4.unknown_name
except AttributeError:
    pass
else:
    raise AssertionError('unknown module attribute should fail')
"""
    result = subprocess.run(
        [sys.executable, "-c", source], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
