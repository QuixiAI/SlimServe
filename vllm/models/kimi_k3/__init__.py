# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model — hardware-isolated entry point.

Unlike ``vllm.models.deepseek_v4``, only the ``amd/`` implementation is
carried here. Upstream's ``nvidia/`` branch depends on cute_dsl, DeepGEMM and
the FA4 vision warmup, none of which build on MI300X, so carrying it would
mean ~12k lines that can never run. Serving Kimi K3 on NVIDIA means porting
that branch back from upstream, not flipping a flag.
"""

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    raise ImportError(
        "vllm.models.kimi_k3 carries only the ROCm implementation in this "
        "fork; see the module docstring."
    )

from .amd.linear import KimiLinearForCausalLM
from .amd.model import KimiK3ForConditionalGeneration
from .amd.mtp import KimiK3MTP

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiK3MTP",
    "KimiLinearForCausalLM",
]
