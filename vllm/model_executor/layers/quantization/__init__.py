# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "auto_awq",
    "fp8",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "gguf",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "online",
    # Below are online quant shorthand names (see vllm.config.quantization).
    # Listed here as strings to avoid a circular import; kept in sync with
    # _ONLINE_SHORTHANDS by the assertion in get_quantization_config().
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "mxfp8",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.debug(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


# Method -> (module, class name). Resolved one at a time in
# `get_quantization_config`: importing the whole table eagerly pulled every
# quantization backend into a process that only ever resolves one of them.
_ONLINE_CONFIG = (".online.base", "OnlineQuantizationConfig")
_METHOD_TO_QUANT_CONFIG: dict[str, tuple[str, str]] = {
    "awq": (".auto_awq", "AutoAWQConfig"),
    "awq_marlin": (".auto_awq", "AutoAWQConfig"),
    "auto_awq": (".auto_awq", "AutoAWQConfig"),
    "fp8": (".fp8", "Fp8Config"),
    "modelopt": (".modelopt", "ModelOptFp8Config"),
    "modelopt_fp4": (".modelopt", "ModelOptNvFp4Config"),
    "modelopt_mxfp8": (".modelopt", "ModelOptMxFp8Config"),
    "modelopt_mixed": (".modelopt", "ModelOptMixedPrecisionConfig"),
    "gguf": (".gguf", "GGUFConfig"),
    "auto_gptq": (".auto_gptq", "AutoGPTQConfig"),
    "gptq": (".auto_gptq", "AutoGPTQConfig"),
    "gptq_marlin": (".auto_gptq", "AutoGPTQConfig"),
    "compressed-tensors": (
        ".compressed_tensors.compressed_tensors",
        "CompressedTensorsConfig",
    ),
    "bitsandbytes": (".bitsandbytes", "BitsAndBytesConfig"),
    "experts_int8": (".experts_int8", "ExpertsInt8Config"),
    "quark": ("vllm.model_executor.layers.quantization.quark.quark", "QuarkConfig"),
    "moe_wna16": (".moe_wna16", "MoeWNA16Config"),
    "torchao": (".torchao", "TorchAOConfig"),
    "inc": (".inc", "INCConfig"),
    "mxfp4": (".mxfp4", "Mxfp4Config"),
    "gpt_oss_mxfp4": (".mxfp4", "GptOssMxfp4Config"),
    "deepseek_v4_fp8": ("vllm.models.deepseek_v4", "DeepseekV4FP8Config"),
    "online": _ONLINE_CONFIG,
    # MiniMax-style checkpoints tag `quant_method: "mxfp8"`; load with the
    # ModelOpt MXFP8 config (same format). The "mxfp8" online shorthand only
    # applies to the `--quantization mxfp8` CLI path.
    "mxfp8": (".modelopt", "ModelOptMxFp8Config"),
}


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # Customized methods are registered last and so win over the builtins.
    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    target = _METHOD_TO_QUANT_CONFIG.get(quantization)
    if target is None:
        # Online shorthands (e.g. "fp8_per_tensor") resolve to the online
        # config, but only when they are not already a checkpoint method.
        from vllm.config.quantization import _ONLINE_SHORTHANDS

        if quantization in _ONLINE_SHORTHANDS:
            target = _ONLINE_CONFIG
    if target is None:
        raise KeyError(quantization)

    module, cls_name = target
    # lazy import to avoid triggering `torch.compile` too early
    mod = importlib.import_module(module, __name__ if module.startswith(".") else None)
    return getattr(mod, cls_name)


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]
