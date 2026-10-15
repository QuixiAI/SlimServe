# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native NVFP4 weight-only linear kernel for Intel XPU."""

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels import xpu_decode

from .emulation import EmulationNvFp4W4A16LinearKernel

logger = init_logger(__name__)

# Above this M the kernel falls back to emulation (dequantize the full weight,
# then oneDNN matmul). The dequant cost is M-independent, so it only pays once M
# is large enough to amortize it -- true for real prefill (M in the thousands),
# but catastrophic for decode: crossing 16 roughly doubles decode step time.
# Decode M is bounded by max-num-seqs, so keep this at or above that cap.
_NATIVE_M_MAX = int(os.environ.get("VLLM_NVFP4_LINEAR_NATIVE_M_MAX", "16"))

# For 2 <= M <= this, run M single-row GEMV launches instead of the DPAS grouped
# GEMM (or the M-tiled kernel). Measured on B70 (2026-08-18, opt-cycle H-E7): the
# DPAS path has a ~150 us per-K-step-latency floor regardless of M, while
# back-to-back GEMV rows re-hit L2, so 2-4 rows cost 62/94/122 us on gate_up
# (N=8704 K=5120) vs 150 us DPAS, and 35/52/70 vs 123 us on down_proj. Very wide
# layers (lm_head, N=62080) only win at M=2, hence the N gate below. 0 = off.
_ROWLOOP_M_MAX = int(os.environ.get("VLLM_NVFP4_LINEAR_ROWLOOP_M_MAX", "0"))
_ROWLOOP_WIDE_N = 16384

# Memoized single-group rows_per_expert tensors keyed by (m, device). Prefill
# chunk sizes recur (one full-chunk value + the tail remainder), so caching
# collapses the ~1440 per-prefill bytes=4 Memcpy M2D copies to a handful.
_ROWS_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _rows_per_expert(m: int, dev: torch.device) -> torch.Tensor:
    """Build the [m] int32 rows_per_expert for the single-group grouped GEMM.

    Default path (`torch.tensor([m], device=dev)`) stages the scalar
    host->device -- a blocking bytes=4 Memcpy M2D per call on XPU. ROWS_FULL
    builds it on-device via a fill kernel (no H2D); ROWS_CACHE memoizes the
    tensor so the H2D pays once. The tensor is read-only input to the kernel,
    so sharing a cached instance is safe.
    """
    if envs.VLLM_NVFP4_DPAS_ROWS_FULL:
        return torch.full((1,), m, dtype=torch.int32, device=dev)
    if envs.VLLM_NVFP4_DPAS_ROWS_CACHE:
        key = (m, dev)
        cached = _ROWS_CACHE.get(key)
        if cached is None:
            cached = torch.tensor([m], dtype=torch.int32, device=dev)
            _ROWS_CACHE[key] = cached
        return cached
    return torch.tensor([m], dtype=torch.int32, device=dev)


class XPUNvFp4W4A16LinearKernel(EmulationNvFp4W4A16LinearKernel):
    """Use the native SYCL NVFP4 kernel for decode-sized batches."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.platforms import current_platform

        if not current_platform.is_xpu():
            return False, "Native NVFP4 kernel is XPU-only"
        if not xpu_decode.has_xpu_decode_op("nvfp4_gemm"):
            return False, "vllm-xpu-kernels lacks nvfp4_gemm"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer._xpu_global_scale = float(layer.weight_global_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = x.shape[-1]
        m = x.numel() // k
        if (
            k % 32 == 0
            and 2 <= m <= _ROWLOOP_M_MAX
            and (layer.weight.shape[0] <= _ROWLOOP_WIDE_N or m == 2)
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            x_2d = x.reshape(-1, k).contiguous()
            w = layer.weight.data.view(torch.uint8)
            ws = layer.weight_scale.data
            gs = layer._xpu_global_scale
            out = torch.cat(
                [xpu_decode.nvfp4_gemm(x_2d[i : i + 1], w, ws, gs) for i in range(m)],
                dim=0,
            )
            out = out.reshape(*x.shape[:-1], out.shape[-1])
            if bias is not None:
                out = out + bias
            return out
        if (
            k % 32 == 0
            and m <= _NATIVE_M_MAX
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            x_2d = x.reshape(-1, k).contiguous()
            out = xpu_decode.nvfp4_gemm(
                x_2d,
                layer.weight.data.view(torch.uint8),
                layer.weight_scale.data,
                layer._xpu_global_scale,
            )
            out = out.reshape(*x.shape[:-1], out.shape[-1])
            if bias is not None:
                out = out + bias
            return out
        if (
            envs.VLLM_NVFP4_LINEAR_PREFILL_DPAS
            and m > _NATIVE_M_MAX
            # The SYCL-TLA grouped GEMM is numerically wrong for K == 16 (mod 32)
            # (relerr ~1.3 on K=4368/1552; opt-cycle H-E4 oddshapes) — the K=16
            # policy's scale mapping, pre-existing and unrelated to the K=32 tile.
            # No Qwen3.x weight has such a K, but a future model could; require
            # K % 32 == 0 here so those fall back to emulation instead of garbage.
            and k % 32 == 0
            and x.dtype == torch.bfloat16
        ):
            out = self._grouped_prefill(layer, x, bias)
            if out is not None:
                return out
        return super().apply_weights(layer, x, bias)

    def _grouped_prefill(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Prefill path: dense NVFP4 linear via the SYCL-TLA DPAS grouped GEMM.

        Runs the [M, K] x [N, K] matmul as a single-group grouped GEMM
        (num_experts=1, rows_per_expert=[M]) that feeds XMX/DPAS, instead of
        dequantizing the whole weight to bf16 every forward (emulation). The
        kernel applies only e2m1 * per-block fp8 scale, so the scalar
        weight_global_scale is folded onto the bf16 output -- exact, since a
        per-tensor scalar factors out of the contraction. Matches the emulation
        W4A16 reference (bf16 x @ dequant(w).t()). Returns None on any failure
        so the caller falls back to emulation.
        """
        try:
            import vllm_xpu_kernels._xpu_C  # noqa: F401
        except Exception:
            logger.warning_once(
                "VLLM_NVFP4_LINEAR_PREFILL_DPAS set but vllm_xpu_kernels._xpu_C "
                "is unavailable; falling back to NVFP4 emulation."
            )
            return None

        k = x.shape[-1]
        x_2d = x.reshape(-1, k).contiguous()
        m = x_2d.shape[0]
        n = layer.weight.shape[0]
        dev = x_2d.device

        # weight: uint8 [N, K/2] -> float4_e2m1fn_x2 [1, N, K/2] (2 fp4/byte).
        # weight_scale: fp8-e4m3 [N, K/16], linear (non-swizzled) 16-block.
        w_v = layer.weight.data.view(torch.float4_e2m1fn_x2).unsqueeze(0)
        w_scale = layer.weight_scale.data.unsqueeze(0)
        rows_per_expert = _rows_per_expert(m, dev)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        try:
            torch.ops._xpu_C.cutlass_grouped_gemm_interface(
                ptr_A=x_2d,
                ptr_A_scale=None,
                ptr_B=w_v,
                ptr_B_scale=w_scale,
                ptr_bias=None,
                ptr_D=out,
                rows_per_expert=rows_per_expert,
                N=n,
                K=k,
                num_experts=1,
            )
        except Exception:
            logger.warning_once(
                "NVFP4 grouped-GEMM prefill failed (N=%d K=%d M=%d); "
                "falling back to emulation.",
                n,
                k,
                m,
            )
            return None

        if envs.VLLM_NVFP4_LINEAR_DPAS_BF16_SCALE:
            out = out * layer._xpu_global_scale
        else:
            out = (out.float() * layer._xpu_global_scale).to(x.dtype)
        out = out.reshape(*x.shape[:-1], n)
        if bias is not None:
            out = out + bias
        return out
