# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuixiCore Metal IR op implementations (Apple Silicon serving path)."""

import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform


def _has_metal_rms_norm() -> bool:
    if not current_platform.is_metal():
        return False
    try:
        from vllm.quixicore.ops import quixicore_ops

        return quixicore_ops.is_available() and quixicore_ops.has("rms_norm")
    except ImportError:
        return False


METAL_RMS_NORM = _has_metal_rms_norm()

rms_metal_args = lambda x, weight, epsilon, variance_size=None: (  # noqa: E731
    variance_size is None
    and weight is not None
    and (weight.dtype == x.dtype or weight.dtype == torch.float32)
    and x.dtype in (torch.float16, torch.bfloat16)
)
"""Metal kernel mirrors ir.ops.rms_norm for the weighted, no-variance-override
case with matching-dtype or fp32 (GGUF) weights; everything else falls through
to native."""


@ir.ops.rms_norm.register_impl(
    "quixicore_metal", supports_args=rms_metal_args, supported=METAL_RMS_NORM
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    from vllm.quixicore.ops import quixicore_ops

    return quixicore_ops.rms_norm(x, weight, epsilon)
