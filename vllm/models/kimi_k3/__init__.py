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

# The MODEL classes are ROCm-only (see docstring), but the vendored KDA
# Triton kernels under ops/third_party/kda are platform-neutral and are
# consumed on CUDA by glm5_next (GLM-5.3-Flash) through the shared
# KimiGatedDeltaNetAttention layer, so the package itself must import
# everywhere; only the class re-exports stay gated.
if current_platform.is_rocm():
    from .amd.linear import KimiLinearForCausalLM
    from .amd.model import KimiK3ForConditionalGeneration
    from .amd.mtp import KimiK3MTP

    __all__ = [
        "KimiK3ForConditionalGeneration",
        "KimiK3MTP",
        "KimiLinearForCausalLM",
    ]
else:
    __all__ = []


def __getattr__(name: str):
    if name in (
        "KimiK3ForConditionalGeneration",
        "KimiK3MTP",
        "KimiLinearForCausalLM",
    ):
        raise ImportError(
            "vllm.models.kimi_k3 carries only the ROCm model implementation "
            "in this fork; see the module docstring."
        )
    raise AttributeError(name)
