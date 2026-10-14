# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 weight-only (W8A16) linear method on the QuixiCore skinny kernel.

Applies to the dense layers a compressed-tensors NVFP4 checkpoint leaves in
bf16 (attention, KDA and indexer projections of GLM-5.3-Flash): at load the
bf16 weight is quantized per output channel to e4m3, packed once into the
kernel's mma fragment order and the bf16 copy is dropped (half the weight
bytes streamed per decode step, 1.25 GB less per TP4 rank). Batches of up
to 128 rows run the W8A16 skinny GEMM; larger batches (prefill) dequantize
to a bf16 temporary and use cuBLAS.

Enabled by VLLM_QC_DENSE_W8A16=1 (A/B switch) or additional_config
glm5_next_dense_w8a16. The e4m3 rounding of the weights is a numerics
change: canaries, recall probes and the exact medians are the gates.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_MAX_SKINNY_ROWS = 128
_FP8_MAX = 448.0
_EXP_FOLD = 256.0  # the kernel's e4m3 -> fp16 relocation yields value * 2^-8


def dense_w8a16_enabled(additional_config: dict | None) -> bool:
    if os.getenv("VLLM_QC_DENSE_W8A16", "0") == "1":
        return True
    return bool((additional_config or {}).get("glm5_next_dense_w8a16", False))


def _qc():
    import vllm._quixicore_C as qc

    return qc


def qc_w8a16_linear(
    x: torch.Tensor, wp: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, n: int, k: int
) -> torch.Tensor:
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if m <= _MAX_SKINNY_ROWS:
        out = _qc().w8a16_gemm(x2.contiguous(), wp, scale, bias, n, 256, 0)
    else:
        w = _qc().w8a16_dequant(wp, scale, n, k)
        out = torch.nn.functional.linear(x2, w, bias)
    return out.reshape(*x.shape[:-1], n)


def _qc_w8a16_linear_fake(x, wp, scale, bias, n, k):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    op_name="qc_w8a16_linear",
    op_func=qc_w8a16_linear,
    mutates_args=[],
    fake_impl=_qc_w8a16_linear_fake,
)


class QcW8A16LinearMethod(UnquantizedLinearMethod):
    """bf16 loading (so every existing weight loader works), fp8 serving."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        n, k = w.shape
        if w.dtype != torch.bfloat16 or n % 32 != 0 or k % 64 != 0 or not w.is_cuda:
            logger.warning_once(
                "qc-w8a16: layer %s (%d x %d, %s) stays bf16 (shape or dtype outside the kernel)",
                getattr(layer, "prefix", "?"), n, k, w.dtype,
            )
            layer._qc_w8a16 = False
            return super().process_weights_after_loading(layer)
        wf = w.float()
        amax = wf.abs().amax(dim=1).clamp(min=1e-8)
        scale = amax / _FP8_MAX
        wq = (wf / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        packed = _qc().w8a16_pack(wq.contiguous())
        del wf, wq
        layer.register_parameter("weight", None)
        layer.qc_w8a16_weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.qc_w8a16_scale = torch.nn.Parameter((scale * _EXP_FOLD).float().contiguous(), requires_grad=False)
        set_weight_attrs(layer.qc_w8a16_weight, {"qc_w8a16": True})
        layer._qc_w8a16 = True
        layer._qc_w8a16_n = n
        layer._qc_w8a16_k = k

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not getattr(layer, "_qc_w8a16", False):
            return super().apply(layer, x, bias)
        return torch.ops.vllm.qc_w8a16_linear(
            x, layer.qc_w8a16_weight, layer.qc_w8a16_scale, bias, layer._qc_w8a16_n, layer._qc_w8a16_k
        )


_EXCLUDED = ("visual", "vision", "lm_head", "embed", "mtp", "draft")


def dense_layer_wants_w8a16(prefix: str, additional_config: dict | None) -> bool:
    if not dense_w8a16_enabled(additional_config):
        return False
    return not any(tok in prefix for tok in _EXCLUDED)
