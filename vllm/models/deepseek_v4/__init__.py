# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.
"""

from vllm.platforms import current_platform

from .quant_config import DeepseekV4FP8Config


def _is_cuda_ampere() -> bool:
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 8


# Pick the per-platform implementation. The NVIDIA branch is the static
# default that mypy sees; the ROCm/XPU/Ampere branches override at runtime and
# are kept type-compatible via ``# type: ignore[assignment]``. A100 takes the
# AMD-family implementation because the Hopper/Blackwell NVIDIA DSv4 attention
# classes require sparse-MLA kernels that do not support sm80.
if current_platform.is_rocm() or current_platform.is_metal() or _is_cuda_ampere():
    from .amd.dspark import (  # type: ignore[assignment]
        DSparkDeepseekV4ForCausalLM,
    )
    from .amd.model import DeepseekV4ForCausalLM
    from .amd.mtp import DeepSeekV4MTP
elif current_platform.is_xpu():
    from .xpu.dspark import DSparkDeepseekV4ForCausalLM  # type: ignore[assignment]
    from .xpu.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .xpu.mtp import DeepSeekV4MTP  # type: ignore[assignment]
else:
    from .nvidia.dspark import (  # type: ignore[assignment]
        DSparkDeepseekV4ForCausalLM,
    )
    from .nvidia.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .nvidia.mtp import DeepSeekV4MTP  # type: ignore[assignment]

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
