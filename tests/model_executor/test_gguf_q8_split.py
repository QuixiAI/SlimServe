# SPDX-License-Identifier: Apache-2.0
"""Contracts for the XPU byte-neutral Q8_0 split layout."""

from types import SimpleNamespace
from unittest import mock

import torch

from vllm.model_executor.layers.quantization.gguf import linear


def _layer(weight_types):
    qweight = torch.nn.Parameter(
        torch.arange(8 * 34, dtype=torch.int16).to(torch.uint8).reshape(8, 34),
        requires_grad=False,
    )
    qweight.loader_marker = object()
    return SimpleNamespace(
        qweight=qweight,
        qweight_type=SimpleNamespace(
            weight_type=8,
            shard_weight_type=dict(weight_types),
        ),
    )


def test_xpu_q8_split_replaces_only_storage_and_preserves_metadata(monkeypatch):
    layer = _layer([("q", 8), ("k", 8), ("v", 8)])
    parameter = layer.qweight
    marker = parameter.loader_marker
    expected = parameter.detach().clone().bitwise_xor_(1)
    method = linear.GGUFLinearMethod.__new__(linear.GGUFLinearMethod)

    monkeypatch.setattr(linear.current_platform, "is_xpu", lambda: True)
    repack = mock.Mock(return_value=expected)
    monkeypatch.setattr(linear.ops, "ggml_repack_q8_0_split", repack)

    method._create_xpu_q8_split_weight(layer)

    repack.assert_called_once_with(parameter, 8, 32)
    assert layer.qweight is parameter
    assert layer.qweight.loader_marker is marker
    assert layer.qweight.gguf_layout == "q8_0_split"
    assert layer._xpu_q8_split is True
    torch.testing.assert_close(layer.qweight, expected)


def test_xpu_q8_split_excludes_heterogeneous_merged_weight(monkeypatch):
    layer = _layer([("q", 8), ("k", 1)])
    method = linear.GGUFLinearMethod.__new__(linear.GGUFLinearMethod)
    monkeypatch.setattr(linear.current_platform, "is_xpu", lambda: True)
    repack = mock.Mock()
    monkeypatch.setattr(linear.ops, "ggml_repack_q8_0_split", repack)

    method._create_xpu_q8_split_weight(layer)

    repack.assert_not_called()
    assert not hasattr(layer, "_xpu_q8_split")
    assert not hasattr(layer.qweight, "gguf_layout")


def test_split_q8_dispatches_batches_through_16_to_native(monkeypatch):
    qweight = torch.zeros((3, 34), dtype=torch.uint8)
    native = mock.Mock(
        side_effect=lambda weight, x, rows: torch.full(
            (x.shape[0], rows), 7, dtype=x.dtype
        )
    )
    monkeypatch.setattr(linear.ops, "ggml_mul_mat_vec_split_q8", native)

    for batch in (1, 2, 8, 16):
        x = torch.zeros((batch, 32), dtype=torch.float32)
        out = linear._fused_mul_mat_gguf_split_q8(x, qweight)
        assert out.shape == (batch, 3)
        assert torch.all(out == 7)

    assert native.call_count == 4


def test_split_q8_large_batch_dequantizes_from_split_layout(monkeypatch):
    linear._q8_0_scratch.clear()
    qweight = torch.zeros((3, 34), dtype=torch.uint8)
    native = mock.Mock()
    monkeypatch.setattr(linear.ops, "ggml_mul_mat_vec_split_q8", native)

    def dequant(weight, rows, cols, out):
        assert weight is qweight
        assert (rows, cols) == (3, 32)
        out.fill_(0.5)

    split_dequant = mock.Mock(side_effect=dequant)
    monkeypatch.setattr(linear.ops, "ggml_dequantize_split_q8_into", split_dequant)
    x = torch.ones((17, 32), dtype=torch.float32)

    out = linear._fused_mul_mat_gguf_split_q8(x, qweight)

    native.assert_not_called()
    split_dequant.assert_called_once()
    torch.testing.assert_close(out, torch.full((17, 3), 16.0))


def test_split_q8_empty_batch_avoids_native_kernel(monkeypatch):
    qweight = torch.zeros((3, 34), dtype=torch.uint8)
    native = mock.Mock()
    monkeypatch.setattr(linear.ops, "ggml_mul_mat_vec_split_q8", native)

    out = linear._fused_mul_mat_gguf_split_q8(
        torch.empty((0, 32), dtype=torch.float32), qweight
    )

    assert out.shape == (0, 3)
    native.assert_not_called()
