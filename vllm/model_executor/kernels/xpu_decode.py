# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapters for XPU decode operators provided by vllm-xpu-kernels.

Only the NVFP4 W4A16 GEMM is on a SlimServe serving path today
(vllm/model_executor/kernels/linear/nvfp4/xpu.py); everything here is inert
unless the ``_xpu_C`` op library registered the op.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


def has_xpu_decode_op(name: str) -> bool:
    return hasattr(torch.ops._xpu_C, name)


if has_xpu_decode_op("nvfp4_gemm"):

    @register_fake("_xpu_C::nvfp4_gemm")
    def _nvfp4_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        block_scales: torch.Tensor,
        global_scale: float,
    ) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], weight.shape[0]))


def nvfp4_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: float,
) -> torch.Tensor:
    return torch.ops._xpu_C.nvfp4_gemm(x, weight, block_scales, global_scale)
