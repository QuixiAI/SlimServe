# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native NVFP4 linear kernel for ROCm (gfx942), backed by the vendored
QuixiCore HIP kernels in csrc/quixicore/tm_rocm/qc_rocm_nvfp4.cu.

Decode-sized calls (M <= 4, the spec-verify width at low concurrency) run the
packed path end to end: activations are NVFP4 QDQ'd (preserving the
checkpoint's w4a4 semantics), quantized to Q8_1, and multiplied against the
planar-repacked E2M1 weight with dp4a — the weight is read in its 4-bit form,
measured 1.3-1.8x the resident-bf16 hipBLASLt matmul at these widths.

Wider calls fall back to a load-time-dequantized bf16 copy through hipBLASLt:
its MFMA path is unbeatable by the dp4a MMQ at M >= 8 (measured; see
perf/optimization_status.md 2026-08-18). Retiring that copy needs an
MFMA-based packed MMQ (Marlin-on-CDNA3), tracked as follow-up work.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    ref_nvfp4_quant_dequant,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

# Above this many rows hipBLASLt on the bf16 copy wins (measured crossover
# between M=4 and M=8 on MI300X).
NATIVE_M_THRESHOLD = 4


def _qc():
    import vllm._quixicore_C as qc

    return qc


def repack_planar(weight_packed: torch.Tensor) -> torch.Tensor:
    """Reorder vLLM's interleaved nvfp4 rows into the planar chunk layout the
    HIP kernels consume: per 32-value chunk, byte j holds value j in the low
    nibble and value j+16 in the high nibble. Layout-only; still 4 bits."""
    n, half_k = weight_packed.shape
    lo = weight_packed & 0xF
    hi = weight_packed >> 4
    vals = torch.stack([lo, hi], dim=-1).reshape(n, half_k * 2)
    chunks = vals.reshape(n, -1, 32)
    return (chunks[..., :16] | (chunks[..., 16:] << 4)).reshape(n, half_k).contiguous()


def _nvfp4_native_gemm_impl(
    x: torch.Tensor,
    weight_planar: torch.Tensor,
    scales_f16: torch.Tensor,
    weight_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
    global_scale: float,
    input_global_scale_f: float,
) -> torch.Tensor:
    m = x.shape[0]
    if m <= NATIVE_M_THRESHOLD:
        # Fully fused decode path: NVFP4 activation QDQ + Q8_1 in one kernel,
        # then the packed-E2M1 GEMV with a bf16 epilogue.
        qc = _qc()
        y_q8 = qc.nvfp4_qdq_quantize_q8_1(x, input_global_scale_f)
        out = torch.empty((m, weight_planar.shape[0]), dtype=x.dtype, device=x.device)
        qc.nvfp4_gemv_q8(
            y_q8, weight_planar, scales_f16, global_scale, out, m, x.shape[1]
        )
        return out
    x_dq = ref_nvfp4_quant_dequant(x, input_global_scale, block_size=16)
    return torch.matmul(x_dq, weight_bf16.t())


def _nvfp4_native_gemm_fake(
    x: torch.Tensor,
    weight_planar: torch.Tensor,
    scales_f16: torch.Tensor,
    weight_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
    global_scale: float,
    input_global_scale_f: float,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight_planar.shape[0]), dtype=x.dtype, device=x.device
    )


direct_register_custom_op(
    "nvfp4_native_gemm",
    _nvfp4_native_gemm_impl,
    fake_impl=_nvfp4_native_gemm_fake,
)


class RocmNativeNvFp4LinearKernel(NvFp4LinearKernel):
    """Packed-E2M1 NVFP4 kernel for gfx942 (QuixiCore HIP GEMV + bf16-copy
    fallback for wide batches)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        if not envs.VLLM_NVFP4_TRITON_GEMM:
            return False, "disabled via VLLM_NVFP4_TRITON_GEMM=0."
        try:
            _qc()
        except ImportError:
            return False, "requires the vllm._quixicore_C extension."
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dequant_dtype = getattr(layer, "params_dtype", torch.bfloat16)
        # bf16 copy for the wide-batch hipBLASLt path, produced from the
        # original interleaved layout (bit-identical to the emulation path).
        weight_bf16 = dequantize_to_dtype(
            layer.weight.data.view(torch.uint8),
            layer.weight_scale.data,
            layer.weight_global_scale,
            dequant_dtype,
            block_size=16,
            swizzle=False,
        )
        scales_f16 = (
            layer.weight_scale.data.view(torch.float8_e4m3fn)
            .to(torch.float16)
            .contiguous()
        )
        planar = repack_planar(layer.weight.data.view(torch.uint8))
        layer.weight = torch.nn.Parameter(planar, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales_f16, requires_grad=False)
        layer.weight_bf16 = torch.nn.Parameter(weight_bf16, requires_grad=False)
        layer._nvfp4_global_scale = float(layer.weight_global_scale)
        layer._nvfp4_input_global_scale = float(layer.input_global_scale_inv)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.ops.vllm.nvfp4_native_gemm(
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_bf16,
            layer.input_global_scale_inv,
            layer._nvfp4_global_scale,
            layer._nvfp4_input_global_scale,
        )
        if bias is not None:
            out = out + bias
        return out
