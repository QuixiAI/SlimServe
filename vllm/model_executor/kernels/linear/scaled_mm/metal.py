# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 weight-only (W8A16) linear on Apple Silicon (Metal / torch-MPS).

Carries the FP8-per-channel side of mixed compressed-tensors checkpoints:
E4M3 weight bytes (stored as uint8 — torch-MPS has no fp8 dtype) with one
scale per output channel. Activations stay in the model dtype; the base
class's QuantFP8 activation quantizer is deliberately never constructed,
mirroring XPUW8A16FP8LinearKernel.

Apply routes:
- Decode widths (M <= 8): the QuixiCore Metal GEMV `qgemv_fp8ch` over the
  raw checkpoint bytes (planar (N, K) e4m3 rows + per-row float scale).
  VLLM_QC_FP8CH=0 is the kill switch back to the dense matmul.
- Prefill / larger M: dense matmul against the bf16 weights materialized at
  load. VLLM_METAL_CT_DEQUANT=call skips materialization (per-call dequant,
  low-memory bring-up fallback).
"""

import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from ..metal_dequant import dequant_fp8_channel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)

# Batch-1 always rides the GEMV. Even M up to the mb bound ride the
# weight-stationary column-pair _mb twin; everything else stays on the
# dense matmul. All bounds are measured on M1 Ultra at serving shapes:
# looping batch-1 launches re-read the weight bytes M times and landed
# below the dense bf16 GEMM at M=4/8 (c4 14.37 -> 13.00, c8 15.93 ->
# 14.00). With the original select-chain E4M3 decode the mb twin only
# reached dense parity at M=8 (qkv 0.466 vs 0.456 ms), so the bound was
# 4; the select-free bit-pattern decode halved the per-byte ALU and the
# twin now wins outright at M=8 (qkv 0.285 vs 0.455 ms, qkvz 0.237 vs
# 0.379), so the bound is 8, matching the 4-bit NVFP4 twin. The host op
# additionally routes contiguous batches 3..8 (odd M included) to the
# mv_ext multi-row twin (qkv M=8 0.165 ms, ~2x dense at odd M) behind
# VLLM_QC_FP8CH_MV4R; use_gemv_mv below admits those batches to the op.
_GEMV_MAX_ROWS = 1
_GEMV_MB_MAX_ROWS = 8


def _gemv_available() -> bool:
    try:
        import vllm._quixicore_C as qc

        return hasattr(qc, "fp8ch_mul_mat_vec")
    except ImportError:
        return False


class MetalWFp8A16LinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_metal():
            return False, "requires the Metal platform"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        # Fused per-tensor checkpoints are expanded to per-channel by the
        # scheme before weights are processed, so both keys land here.
        if c.weight_quant_key not in (kFp8StaticChannelSym, kFp8StaticTensorSym):
            return False, "only per-channel and per-tensor FP8 weights"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        # Skip FP8ScaledMMLinearKernel.__init__: it builds a QuantFP8
        # activation quantizer this W8A16 kernel must never run.
        ok, why = self.can_implement(c)
        if not ok:
            raise ValueError(f"MetalFP8W8A16LinearKernel: {why}")
        ok, why = self.is_supported()
        if not ok:
            raise ValueError(f"MetalFP8W8A16LinearKernel: {why}")
        self.config = c
        self.layer_param_names = layer_param_names
        self.materialize = os.environ.get("VLLM_METAL_CT_DEQUANT", "once") != "call"
        self.use_gemv = os.environ.get("VLLM_QC_FP8CH", "1") != "0"
        if self.use_gemv and not _gemv_available():
            logger.warning_once(
                "qgemv_fp8ch requested but vllm._quixicore_C has no "
                "fp8ch_mul_mat_vec (stale .so?); decode falls back to the "
                "dense matmul path."
            )
            self.use_gemv = False
        self.use_gemv_mb = (
            self.use_gemv and os.environ.get("VLLM_QC_FP8CH_MB", "1") != "0"
        )
        # mv_ext route for M in [3, 8] incl. odd M (the lm_head candidate
        # pass runs M = R*7); the host op reads the same env at boot to
        # pick mv4r over mb for batches 3..8.
        self.use_gemv_mv = (
            self.use_gemv and os.environ.get("VLLM_QC_FP8CH_MV4R", "1") != "0"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # On Metal the scheme skips its (K, N) canonicalization, so weight is
        # the checkpoint's row-major uint8 [N, K] and weight_scale is fp32
        # [N, 1] — exactly the qgemv_fp8ch operand layout.
        n, k = layer.weight.shape
        if self.use_gemv and n % 4 == 0 and k % 16 == 0:
            # Raw bytes + flat fp32 scale for the GEMV (kept alongside any
            # materialized dense weight; decode never touches the bf16 copy).
            layer.fp8ch_weight = layer.weight.data
            layer.fp8ch_scale = layer.weight_scale.data.reshape(-1).contiguous()
            layer.metal_fp8ch = True
        if not self.materialize:
            return
        dense = dequant_fp8_channel(
            layer.weight, layer.weight_scale, self.config.input_dtype
        )
        replace_parameter(layer, "weight", dense)
        layer.metal_ct_materialized = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        m = x_2d.shape[0]
        if (
            getattr(layer, "metal_fp8ch", False)
            and (
                m <= _GEMV_MAX_ROWS
                or (self.use_gemv_mb and m % 2 == 0 and m <= _GEMV_MB_MAX_ROWS)
                or (self.use_gemv_mv and 3 <= m <= _GEMV_MB_MAX_ROWS)
            )
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from vllm.quixicore import quixicore_ops

            out = quixicore_ops.fp8ch_mul_mat_vec(
                layer.fp8ch_weight, x_2d.contiguous(), layer.fp8ch_scale
            )
            out = out.view(*x.shape[:-1], out.shape[-1])
            if bias is not None:
                out = out + bias
            return out
        if getattr(layer, "metal_ct_materialized", False):
            return F.linear(x, layer.weight, bias)
        weight = dequant_fp8_channel(layer.weight, layer.weight_scale, x.dtype)
        return F.linear(x, weight, bias)

    def apply_scaled_mm(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "MetalWFp8A16LinearKernel overrides apply_weights directly"
        )
