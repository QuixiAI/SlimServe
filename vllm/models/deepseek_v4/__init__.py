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


def __getattr__(name: str):
    """Load model code only when the registry requests a model class.

    Quantization override discovery imports this package for its FP8 config
    while resolving other model families. It must not import a hardware model
    implementation or its optional native dependencies as a side effect.
    """
    if name not in {
        "DSparkDeepseekV4ForCausalLM",
        "DeepseekV4ForCausalLM",
        "DeepSeekV4MTP",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Preserve platform selection. A100 uses the AMD-family implementation
    # because NVIDIA DSv4 sparse-MLA kernels do not support sm80.
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

    globals().update(
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
        DeepseekV4ForCausalLM=DeepseekV4ForCausalLM,
        DeepSeekV4MTP=DeepSeekV4MTP,
    )
    return globals()[name]


__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
