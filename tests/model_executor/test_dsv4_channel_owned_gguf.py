# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.gguf import params as gguf_params


def _dsv4_layer(tp_size: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4096,
        moe_config=SimpleNamespace(
            intermediate_size=2048,
            moe_parallel_config=SimpleNamespace(tp_size=tp_size),
        ),
    )


def test_channel_owned_w2_materializes_output_rows(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DSV4_CHANNEL_OWNED", "1")
    monkeypatch.setattr(gguf_params, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(
        gguf_params, "get_tensor_model_parallel_world_size", lambda: 4
    )
    layer = _dsv4_layer()
    loaded = torch.zeros((2, 4096, 672), dtype=torch.uint8)
    loaded[:, :, 0] = torch.arange(4096, dtype=torch.int64).to(torch.uint8)
    param = gguf_params.GGUFUninitializedWeightParameter(requires_grad=False)

    def unexpected_base_loader(*args, **kwargs):
        raise AssertionError("output-stationary W2 must bypass the TP K-shard loader")

    result = gguf_params._gguf_moe_weight_loader(
        layer,
        unexpected_base_loader,
        param,
        loaded,
        "layers.0.ffn.experts.w2_qweight",
        "w2",
        0,
        return_success=True,
    )

    assert result is True
    assert param.shape == (2, 1024, 672)
    assert layer._dsv4_w2_output_sharded is True
    torch.testing.assert_close(param[:, :, 0], loaded[:, 2048:3072, 0])


def test_channel_owned_w2_is_restricted_to_dsv4_geometry(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DSV4_CHANNEL_OWNED", "1")
    layer = _dsv4_layer()
    wrong_hidden = torch.empty((2, 3072, 672), dtype=torch.uint8)
    wrong_quant_width = torch.empty((2, 4096, 336), dtype=torch.uint8)

    assert not gguf_params._use_dsv4_channel_owned_w2(
        layer, wrong_hidden, "w2"
    )
    assert not gguf_params._use_dsv4_channel_owned_w2(
        layer, wrong_quant_width, "w2"
    )
    assert not gguf_params._use_dsv4_channel_owned_w2(
        layer, torch.empty((2, 4096, 672), dtype=torch.uint8), "w1"
    )


def test_output_owned_row_parameter_shards_output_rows() -> None:
    loaded = torch.zeros((4096, 2176), dtype=torch.uint8)
    loaded[:, 0] = torch.arange(4096, dtype=torch.int64).to(torch.uint8)
    param = gguf_params.GGUFUninitializedWeightParameter(requires_grad=False)
    param.tp_rank = 2
    param.tp_size = 4
    param.dsv4_output_owned = True

    param.load_row_parallel_weight(loaded)

    assert param.shape == (1024, 2176)
    torch.testing.assert_close(param[:, 0], loaded[2048:3072, 0])


def test_standard_row_parameter_still_shards_input_columns() -> None:
    loaded = torch.zeros((4096, 2176), dtype=torch.uint8)
    param = gguf_params.GGUFUninitializedWeightParameter(requires_grad=False)
    param.tp_rank = 2
    param.tp_size = 4
    param.dsv4_output_owned = False

    param.load_row_parallel_weight(loaded)

    assert param.shape == (4096, 544)
