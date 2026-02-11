# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    repack_weight_int4_mfma_tiled,
    repack_weight_int4_w4a8_tiled,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


def _is_rocm_cdna3() -> bool:
    if not current_platform.is_rocm():
        return False
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    return arch.startswith("gfx94") or arch.startswith("gfx95")


def _use_w4a8() -> bool:
    return os.getenv("VLLM_W4A8", "0").strip().lower() in ("1", "true")


class RocmMfmaLinearKernel(MPLinearKernel):
    """INT4 MFMA kernel for ROCm CDNA3+ (MI300X).

    Supports two modes:
    - W4A16 (default): BF16/FP16 activations, 2× FP16 MFMA per K=32 tile
    - W4A8 (VLLM_W4A8=1): FP8 activations, 1× FP8 MFMA per K=32 tile
    Symmetric INT4 only (no zero-points), no act_order.
    """

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90  # gfx90x+

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not _is_rocm_cdna3():
            return False, "RocmMfma requires CDNA3+ GPU (gfx94x/gfx95x)"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "RocmMfma, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return (
                False,
                "RocmMfma only supports float16/bfloat16 activations",
            )

        if c.has_g_idx:
            return False, "RocmMfma does not support act_order (desc_act)"

        if c.zero_points:
            return False, "RocmMfma only supports symmetric quantization"

        if c.partition_weight_shape[0] % 32 != 0:
            return (
                False,
                f"K ({c.partition_weight_shape[0]}) must be divisible by 32",
            )

        if c.partition_weight_shape[1] % 16 != 0:
            return (
                False,
                f"N ({c.partition_weight_shape[1]}) must be divisible by 16",
            )

        if c.group_size not in (-1, 32, 64, 128):
            return (
                False,
                f"Group size ({c.group_size}) not supported, "
                "must be -1, 32, 64, or 128",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config

        # Get weight and transform layout: [out, in/pack] -> [in/pack, out]
        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            return x.data.contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)

        qweight = getattr(layer, self.w_q_name)
        size_k = c.partition_weight_shape[0]
        size_n = c.partition_weight_shape[1]

        # Choose tile format based on mode
        use_w4a8 = _use_w4a8()
        if use_w4a8:
            w_mfma = repack_weight_int4_w4a8_tiled(qweight, size_k, size_n)
        else:
            w_mfma = repack_weight_int4_mfma_tiled(qweight, size_k, size_n)

        from vllm.model_executor.layers.quantization.utils import (
            replace_parameter,
        )

        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(w_mfma, requires_grad=False),
        )

        # Store mode flag on layer for apply_weights
        layer.use_w4a8 = use_w4a8

        # Scales: transpose to [num_groups, N] linear layout
        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q = getattr(layer, self.w_q_name)
        w_s = getattr(layer, self.w_s_name)

        m = x_2d.shape[0]
        n = c.partition_weight_shape[1]
        k = c.partition_weight_shape[0]

        use_w4a8 = getattr(layer, 'use_w4a8', False)

        if use_w4a8:
            # W4A8: quantize activations to FP8 per-token
            a_fp8, a_scales = ops.scaled_fp8_quant(
                x_2d, use_per_token_if_dynamic=True
            )
            output = ops.int4_mfma_marlin_gemm(
                a=a_fp8,
                b_q_weight=w_q,
                b_scales=w_s,
                a_scales=a_scales,
                size_m=m,
                size_n=n,
                size_k=k,
            )
        else:
            # W4A16: standard BF16/FP16 path
            output = ops.int4_mfma_marlin_gemm(
                a=x_2d,
                b_q_weight=w_q,
                b_scales=w_s,
                a_scales=None,
                size_m=m,
                size_n=n,
                size_k=k,
            )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
