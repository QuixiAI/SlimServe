# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    kE2M1ToFloat_handle,
    ref_nvfp4_quant_dequant,
    run_nvfp4_emulations,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class EmulationNvFp4LinearKernel(NvFp4LinearKernel):
    """Software emulation fallback for NVFP4 (dequant → BF16 matmul).

    By default the weight is dequantized once at load time and kept resident
    in the model dtype. Per-call dequantization writes and re-reads the full
    high-precision weight on every forward, which dominates decode step time;
    caching trades VRAM (~4x the packed weight) for that traffic. Set
    ``VLLM_NVFP4_EMULATION_CACHE_WEIGHTS=0`` to keep the packed weight and
    dequantize per call (the memory-constrained behavior). Both modes produce
    bit-identical outputs: the cached tensor is exactly the per-call
    dequantization result. Activations keep the NVFP4 quantize-dequantize
    round trip, preserving w4a4 numerics.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Always available as a last-resort fallback.
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Move the E2M1 lookup table to the device now, because
        # `.to(device)` is not allowed during CUDA graph capture.
        kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(layer.weight.device)

        if not envs.VLLM_NVFP4_EMULATION_CACHE_WEIGHTS:
            return

        dequant_dtype = getattr(layer, "params_dtype", torch.bfloat16)
        weight_dq = dequantize_to_dtype(
            layer.weight.data.view(torch.uint8),
            layer.weight_scale.data,
            layer.weight_global_scale,
            dequant_dtype,
            block_size=16,
            swizzle=False,
        )
        logger.info_once(
            "NVFP4 emulation: caching load-time dequantized weights in %s "
            "(VLLM_NVFP4_EMULATION_CACHE_WEIGHTS=0 disables).",
            dequant_dtype,
        )
        # Replace the packed weight and its block scales so their storage is
        # freed; only the input global scale is still needed at runtime.
        layer.weight = torch.nn.Parameter(weight_dq, requires_grad=False)
        layer.weight_scale = None
        self._cached = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_cached", False):
            x_dq = ref_nvfp4_quant_dequant(
                x, layer.input_global_scale_inv, block_size=16
            )
            out = torch.matmul(x_dq, layer.weight.t())
        else:
            out = run_nvfp4_emulations(
                x=x,
                input_global_scale=layer.input_global_scale_inv,
                weight=layer.weight,
                weight_scale_swizzled=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                swizzle=False,
            )
        if bias is not None:
            out = out + bias
        return out
