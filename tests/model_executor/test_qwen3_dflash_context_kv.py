# SPDX-License-Identifier: Apache-2.0
"""The DFlash context-KV precompute stacks per-layer K/V projection rows into
one GEMM only for dense bf16/fp16 weights.

Quantized projections (GGUF qweight, or FP8 online-quant weights that
``Fp8LinearMethod.process_weights_after_loading`` stores transposed with
their scales) keep the layer modules and run their quantized methods one
layer at a time: slicing an FP8 weight's rows past q_size cut the wrong
dimension and the fused GEMM failed at boot.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

HIDDEN = 16
Q_SIZE = 8
KV_SIZE = 4
HEAD_DIM = 2


def _layer(weight: torch.Tensor, bias: torch.Tensor | None = None):
    qkv = SimpleNamespace(weight=weight)
    if bias is not None:
        qkv.bias = bias
    return SimpleNamespace(
        qkv_proj=qkv,
        q_size=Q_SIZE,
        k_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(HEAD_DIM))),
    )


def _model() -> DFlashQwen3Model:
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    model.hidden_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(HIDDEN)))
    return model


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_weights_stack_into_one_kv_gemm(dtype: torch.dtype) -> None:
    layers = [
        _layer(torch.randn(Q_SIZE + 2 * KV_SIZE, HIDDEN, dtype=dtype)) for _ in range(3)
    ]
    model = _model()
    model._build_context_kv_buffers(layers, has_bias=False)
    assert model._context_qkv_projections is None
    assert model._fused_kv_bias is None
    assert model._fused_kv_weight.shape == (3 * 2 * KV_SIZE, HIDDEN)
    for i, layer in enumerate(layers):
        rows = slice(i * 2 * KV_SIZE, (i + 1) * 2 * KV_SIZE)
        torch.testing.assert_close(
            model._fused_kv_weight[rows], layer.qkv_proj.weight[Q_SIZE:]
        )
    assert model._k_norm_weights.shape == (3, HEAD_DIM)


def test_fp8_weights_keep_the_per_layer_quantized_path() -> None:
    fp8 = torch.randn(HIDDEN, Q_SIZE + 2 * KV_SIZE).to(torch.float8_e4m3fn)
    layers = [_layer(fp8.clone()) for _ in range(2)]
    model = _model()
    model._build_context_kv_buffers(layers, has_bias=False)
    assert model._fused_kv_weight is None
    assert model._fused_kv_bias is None
    assert model._context_qkv_projections is layers


def test_packed_qweight_layers_keep_the_per_layer_quantized_path() -> None:
    layers = [
        SimpleNamespace(
            qkv_proj=SimpleNamespace(qweight=torch.zeros(4, dtype=torch.uint8)),
            q_size=Q_SIZE,
            k_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(HEAD_DIM))),
        )
    ]
    model = _model()
    model._build_context_kv_buffers(layers, has_bias=False)
    assert model._fused_kv_weight is None
    assert model._context_qkv_projections is layers
