# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-offloading a UVA parameter keeps its strides off XPU.

`process_weights_after_loading` may replace a UVA-offloaded weight with a
strided view (per-tensor FP8 and CUTLASS scaled_mm store `weight.t()`).
CUDA/ROCm kernels rely on that column-major layout, so the re-offload in
`device_loading_context` must not make it contiguous there; only the XPU
UVA view op needs a contiguous CPU tensor. Runs on CPU with the accelerator
view stubbed out.
"""

from unittest.mock import patch

import torch
from torch import nn

from vllm.model_executor.model_loader import utils as loader_utils

_MOD = "vllm.model_executor.model_loader.utils"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        # Already offloaded: lives off the CPU, flagged UVA.
        self.weight = nn.Parameter(torch.empty(128, 256, device="meta"))
        self.weight._vllm_is_uva_offloaded = True


def _reoffload(is_xpu: bool) -> torch.Tensor:
    layer = _Layer()
    with (
        patch(f"{_MOD}.current_platform") as platform,
        patch(f"{_MOD}.is_pin_memory_available", return_value=False),
        patch(f"{_MOD}.get_accelerator_view_from_cpu_tensor", side_effect=lambda t: t),
    ):
        platform.is_xpu.return_value = is_xpu
        with loader_utils.device_loading_context(layer, torch.device("meta")):
            # What fp8.py / cutlass.py do: replace with a transposed view.
            layer.weight = nn.Parameter(torch.randn(256, 128).t(), requires_grad=False)
    assert layer.weight._vllm_is_uva_offloaded
    return layer.weight.data


def test_reoffload_preserves_column_major_strides_off_xpu():
    w = _reoffload(is_xpu=False)
    assert w.shape == (128, 256)
    assert w.stride() == (1, 128)
    assert w.stride(0) == 1  # CUTLASS scaled_mm requirement


def test_reoffload_makes_contiguous_on_xpu():
    w = _reoffload(is_xpu=True)
    assert w.is_contiguous()
