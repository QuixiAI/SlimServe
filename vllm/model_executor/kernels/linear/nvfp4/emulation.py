# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    kE2M1ToFloat_handle,
    run_nvfp4_emulations,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class EmulationNvFp4LinearKernel(NvFp4LinearKernel):
    """Software emulation fallback for NVFP4 (dequant → BF16 matmul)."""

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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
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


class EmulationNvFp4W4A16LinearKernel(NvFp4LinearKernel):
    """Weight-only emulation for W4A16 NVFP4 (dequant -> matmul, no act quant).

    Consumes the same layer layout as MarlinNvFp4LinearKernel: packed uint8
    weight, linear (non-swizzled) fp8-e4m3 group scales, and
    weight_global_scale in ModelOpt's stored form (amax/2688), which is the
    direct dequant multiplier.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        kE2M1ToFloat_handle.val = kE2M1ToFloat_handle.val.to(layer.weight.device)

    # Dequantize at most this many output rows at a time. Materializing the
    # full dequantized weight (plus its fp32 intermediates) is prohibitive for
    # large layers such as lm_head (vocab x hidden) and can exhaust VRAM.
    _OUT_CHUNK = 8192

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight.data.view(torch.uint8)
        weight_scale = layer.weight_scale.data
        out_features = weight.shape[0]

        out = x.new_empty((*x.shape[:-1], out_features))
        for start in range(0, out_features, self._OUT_CHUNK):
            end = min(start + self._OUT_CHUNK, out_features)
            w_dq = dequantize_to_dtype(
                weight[start:end],
                weight_scale[start:end],
                layer.weight_global_scale,
                x.dtype,
                block_size=16,
                swizzle=False,
            )
            out[..., start:end] = torch.matmul(x, w_dq.t())
        if bias is not None:
            out = out + bias
        return out
