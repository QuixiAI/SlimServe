# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 linear on Apple Silicon (Metal / torch-MPS).

Serves NVFP4 checkpoints with W4A16 semantics: activations stay in the model
dtype and only the weights are quantized. The checkpoint's input-activation
scales (input_global_scale / alpha, registered by the W4A4 scheme) are loaded
but deliberately unused — quantizing activations to FP4 on the fly buys
nothing on this hardware and torch-MPS has no fp8/fp4 dtypes to do it with.

Apply routes:
- Decode widths (M <= 8): the QuixiCore Metal GEMVs over the raw planar
  checkpoint buffers (packed e2m1 + e4m3 group scales + fp32 global) —
  `qgemv_nvfp4_planar` at M == 1, the weight-stationary `_mb` /
  `mv_ext` batch twins for M in [2, 8]. VLLM_QC_NVFP4=0 is the kill
  switch back to the dense matmul; VLLM_QC_NVFP4_MB=0 kills only the
  batch twins.
- Prefill / larger M: dense matmul against the bf16 weights materialized at
  load (VLLM_METAL_CT_DEQUANT=once, default). =call skips materialization
  (per-call dequant, low-memory bring-up fallback).
"""

import os

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from ..metal_dequant import dequant_nvfp4
from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

# Batch-1 always rides the GEMV; M in [2, 8] ride the batch twins (the
# host op picks the mv_ext route for 3..8 and the column-pair _mb twin for
# 2). Odd M in [3, 8] need the mv route (VLLM_QC_NVFP4_MV4R, default on);
# with it disabled they stay dense as before (looped batch-1 measured
# below the dense GEMM at M=4/8 — see scaled_mm/metal.py).
_GEMV_MAX_ROWS = 1
_GEMV_MB_MAX_ROWS = 8


def _gemv_available() -> bool:
    try:
        import vllm._quixicore_C as qc

        return hasattr(qc, "nvfp4_mul_mat_vec")
    except ImportError:
        return False


class MetalNvFp4LinearKernel(NvFp4LinearKernel):
    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.materialize = os.environ.get("VLLM_METAL_CT_DEQUANT", "once") != "call"
        self.use_gemv = os.environ.get("VLLM_QC_NVFP4", "1") != "0"
        if self.use_gemv and not _gemv_available():
            logger.warning_once(
                "qgemv_nvfp4_planar requested but vllm._quixicore_C has no "
                "nvfp4_mul_mat_vec (stale .so?); decode falls back to the "
                "dense matmul path."
            )
            self.use_gemv = False
        self.use_gemv_mb = (
            self.use_gemv and os.environ.get("VLLM_QC_NVFP4_MB", "1") != "0"
        )
        # mv_ext route for M in [3, 8] incl. odd M; the host op reads the
        # same env at boot to pick mv4r over mb for batches 3..8.
        self.use_gemv_mv = (
            self.use_gemv and os.environ.get("VLLM_QC_NVFP4_MV4R", "1") != "0"
        )
        vllm_config = get_current_vllm_config_or_none()
        self.out_dtype = (
            vllm_config.model_config.dtype
            if vllm_config is not None and vllm_config.model_config is not None
            else torch.bfloat16
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_metal():
            return False, "requires the Metal platform"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        # Every geometry the W4A4 scheme produces is servable: the GEMV
        # enforces its own N/K alignment at route time and everything else
        # falls back to the dense matmul.
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The scheme has already renamed weight_packed -> weight (uint8
        # [N, K/2]) and inverted the CT divisor, so weight_global_scale is the
        # true multiplier. weight_scale holds raw E4M3 bytes as uint8 (Metal
        # cannot allocate fp8; see the scheme's create_weights).
        n, k_half = layer.weight.shape  # type: ignore[misc]
        if self.use_gemv and n % 4 == 0 and (k_half * 2) % 16 == 0:
            layer.nvfp4_weight = layer.weight.data
            layer.nvfp4_scale = layer.weight_scale.data
            layer.nvfp4_global = (
                layer.weight_global_scale.data.to(torch.float32)  # type: ignore[operator, union-attr]
                .reshape(1)
                .contiguous()
            )
            layer.metal_nvfp4 = True  # type: ignore[assignment]
        if not self.materialize:
            return
        dense = dequant_nvfp4(
            layer.weight,  # type: ignore[arg-type]
            layer.weight_scale,  # type: ignore[arg-type]
            layer.weight_global_scale,  # type: ignore[arg-type]
            self.out_dtype,  # type: ignore[arg-type]
        )
        replace_parameter(layer, "weight", dense)
        layer.metal_ct_materialized = True  # type: ignore[assignment]

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        m = x_2d.shape[0]
        if (
            getattr(layer, "metal_nvfp4", False)
            and (
                m <= _GEMV_MAX_ROWS
                or (self.use_gemv_mb and m % 2 == 0 and m <= _GEMV_MB_MAX_ROWS)
                or (self.use_gemv_mv and 3 <= m <= _GEMV_MB_MAX_ROWS)
            )
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from vllm.quixicore import quixicore_ops

            out = quixicore_ops.nvfp4_mul_mat_vec(
                layer.nvfp4_weight,  # type: ignore[arg-type]
                x_2d.contiguous(),
                layer.nvfp4_scale,  # type: ignore[arg-type]
                layer.nvfp4_global,  # type: ignore[arg-type]
            )
            out = out.view(*x.shape[:-1], out.shape[-1])
            if bias is not None:
                out = out + bias
            return out
        if getattr(layer, "metal_ct_materialized", False):
            return F.linear(x, layer.weight, bias)  # type: ignore[arg-type]
        weight = dequant_nvfp4(
            layer.weight,  # type: ignore[arg-type]
            layer.weight_scale,  # type: ignore[arg-type]
            layer.weight_global_scale,  # type: ignore[arg-type]
            x.dtype,
        )
        return F.linear(x, weight, bias)
