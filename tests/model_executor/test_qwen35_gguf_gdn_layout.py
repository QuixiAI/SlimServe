# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused contracts for the Qwen3.5 GGUF fused GDN projections."""

from types import MethodType, SimpleNamespace
from unittest import mock

import gguf
import numpy as np
import pytest
import torch
from gguf.quants import dequantize

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    out_proj_v_head_order,
)
from vllm.model_executor.model_loader.gguf_adapters.base import GGUFLoadSpec
from vllm.model_executor.model_loader.gguf_adapters.default import (
    GGUFWeightsAdapter,
)
from vllm.model_executor.model_loader.gguf_adapters.qwen35 import (
    Qwen35GGUFAdapter,
    _out_proj_v_head_group,
    _untile_packed_v_head_columns,
    _untile_v_head_axis,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5Model,
)


def _model_config():
    text_config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_key_head_dim=1,
        linear_num_value_heads=6,
        linear_value_head_dim=1,
    )
    return SimpleNamespace(
        hf_config=SimpleNamespace(get_text_config=lambda: text_config)
    )


def test_qwen35_gguf_qkv_quant_bytes_reach_fused_parameter_as_scalar_shards():
    """A fused GGUF QKV matrix must become three loadable merged shards.

    Real Qwen3.8 has 2048 Q rows, 2048 K rows and 6144 V rows. These small
    dimensions preserve the same 2:2:6 split while making row identity easy
    to assert.
    """
    adapter = Qwen35GGUFAdapter(SimpleNamespace())
    adapter._dequant_stems = set()
    adapter.load_spec = GGUFLoadSpec([], [], {})
    packed = torch.arange(10 * 4, dtype=torch.uint8).reshape(10, 4)
    qtype = torch.tensor(8)
    source = [
        ("model.layers.0.linear_attn.in_proj_qkv.qweight_type", qtype),
        ("model.layers.0.linear_attn.in_proj_qkv.qweight", packed),
    ]

    with mock.patch.object(
        GGUFWeightsAdapter, "prepare_weights", return_value=iter(source)
    ):
        split = list(adapter.prepare_weights(_model_config()))

    mapped = list(Qwen3_5Model.hf_to_vllm_mapper.apply(split))

    assert [name for name, _ in mapped] == [
        "model.layers.0.linear_attn.in_proj_qkvz.qweight_type",
        "model.layers.0.linear_attn.in_proj_qkvz.qweight_type",
        "model.layers.0.linear_attn.in_proj_qkvz.qweight_type",
        "model.layers.0.linear_attn.in_proj_qkvz.qweight",
        "model.layers.0.linear_attn.in_proj_qkvz.qweight",
        "model.layers.0.linear_attn.in_proj_qkvz.qweight",
    ]
    assert [tensor.shard_id for _, tensor in mapped] == [0, 1, 2, 0, 1, 2]
    assert [tensor.shape for _, tensor in mapped[3:]] == [
        torch.Size([2, 4]),
        torch.Size([2, 4]),
        torch.Size([6, 4]),
    ]
    torch.testing.assert_close(mapped[3][1], packed[:2])
    torch.testing.assert_close(mapped[4][1], packed[2:4])
    torch.testing.assert_close(mapped[5][1], packed[4:])


def test_qwen35_gguf_ba_shards_keep_existing_fused_mapping():
    weights = [
        ("model.layers.0.linear_attn.in_proj_b.qweight", torch.ones(2, 4)),
        ("model.layers.0.linear_attn.in_proj_a.qweight", torch.zeros(2, 4)),
    ]

    mapped = list(Qwen3_5Model.hf_to_vllm_mapper.apply(weights))

    assert [name for name, _ in mapped] == [
        "model.layers.0.linear_attn.in_proj_ba.qweight",
        "model.layers.0.linear_attn.in_proj_ba.qweight",
    ]
    assert [tensor.shard_id for _, tensor in mapped] == [0, 1]


def test_qwen35_gguf_untile_v_heads_restores_grouped_order():
    # Tiled source order is [r0g0, r0g1, r1g0, r1g1, r2g0, r2g1].
    tiled = torch.tensor(
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        dtype=torch.float32,
    )

    grouped = _untile_v_head_axis(
        tiled,
        axis=0,
        num_k_heads=2,
        num_v_heads=6,
        values_per_head=1,
    )

    assert grouped[:, 0].tolist() == [0, 2, 4, 1, 3, 5]


def test_qwen35_gguf_untile_q8_columns_moves_whole_quant_blocks():
    # head_dim=64 is two Q8_0 blocks = 68 raw bytes per value head.
    bytes_per_head = 68
    tiled = torch.cat(
        [torch.full((2, bytes_per_head), head, dtype=torch.uint8) for head in range(6)],
        dim=1,
    )

    grouped = _untile_packed_v_head_columns(
        tiled,
        num_k_heads=2,
        num_v_heads=6,
        head_dim=64,
        weight_type="Q8_0",
    )

    chunks = grouped.reshape(2, 6, bytes_per_head)
    assert chunks[0, :, 0].tolist() == [0, 2, 4, 1, 3, 5]
    assert chunks[1, :, -1].tolist() == [0, 2, 4, 1, 3, 5]


def test_qwen35_gguf_untile_rejects_a_run_that_splits_quant_blocks():
    with pytest.raises(ValueError, match="not aligned"):
        _untile_packed_v_head_columns(
            torch.empty(1, 1, dtype=torch.uint8),
            num_k_heads=2,
            num_v_heads=6,
            head_dim=128,
            weight_type="Q4_K",
            group=1,
        )


def test_qwen35_gguf_out_proj_group_covers_block_256_formats():
    # Every ssm_out type in the published UD-Q2_K_XL file is a 256 block.
    ud_q2kxl = ["Q4_K", "IQ3_S", "IQ4_XS", "IQ3_XXS", "IQ2_S", "IQ2_XS", "Q3_K"]
    assert _out_proj_v_head_group(ud_q2kxl, 128) == 2
    assert _out_proj_v_head_group(["Q8_0", "F16", "BF16"], 128) == 1
    assert _out_proj_v_head_group([], 128) == 1


def _random_q4k_blocks(rows: int, blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (rows, blocks, 144), dtype=np.uint8)
    scale = rng.uniform(0.005, 0.02, (rows, blocks, 2)).astype(np.float16)
    raw[..., :4] = scale.view(np.uint8)  # sane d / dmin
    return raw.reshape(rows, blocks * 144)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_qwen35_gguf_k_quant_out_proj_loads_quantized_and_matches_hf(tp_size):
    """A Q4_K out_proj (block 256 spans two 128-wide heads) loads quantized.

    The columns move in whole blocks to the run-tiled order, the GDN module
    gathers its grouped activations into that order per rank, and the
    projection must equal the HF grouped reference at TP1 and TP2.
    """
    num_k, num_v, head_dim, rows = 4, 12, 128, 3
    qtype = gguf.GGMLQuantizationType.Q4_K
    tiled_raw = _random_q4k_blocks(rows, num_v * head_dim // 256, seed=0)
    tiled_deq = torch.from_numpy(dequantize(tiled_raw, qtype)).double()
    reference = _untile_v_head_axis(
        tiled_deq,
        axis=1,
        num_k_heads=num_k,
        num_v_heads=num_v,
        values_per_head=head_dim,
    )

    group = _out_proj_v_head_group(["Q4_K"], head_dim)
    packed = _untile_packed_v_head_columns(
        torch.from_numpy(tiled_raw),
        num_k_heads=num_k,
        num_v_heads=num_v,
        head_dim=head_dim,
        weight_type="Q4_K",
        group=group,
    )
    assert packed.dtype == torch.uint8 and packed.shape == tiled_raw.shape
    loaded = torch.from_numpy(dequantize(packed.numpy(), qtype)).double()

    x = torch.randn(5, num_v * head_dim, dtype=torch.float64)
    expected = x @ reference.T
    local_k = num_k // tp_size
    order = out_proj_v_head_order(local_k, num_v // num_k, group)
    width = num_v * head_dim // tp_size
    out = torch.zeros_like(expected)
    for rank in range(tp_size):
        # RowParallelLinear: each rank owns a contiguous column slice and
        # the grouped core output for its own key heads.
        cols = slice(rank * width, (rank + 1) * width)
        x_local = x[:, cols].reshape(5, -1, head_dim)
        x_local = x_local.index_select(1, order).reshape(5, -1)
        out += x_local @ loaded[:, cols].T
    torch.testing.assert_close(out, expected)


def test_qwen35_gguf_out_proj_order_is_identity_without_runs():
    torch.testing.assert_close(out_proj_v_head_order(4, 3, 1), torch.arange(12))
    # Runs of two key heads, tiled inside: (k0j0 k1j0 k0j1 k1j1 ...).
    assert out_proj_v_head_order(2, 3, 2).tolist() == [0, 3, 1, 4, 2, 5]


def test_qwen35_gguf_untile_covers_all_per_v_head_tensors():
    config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=1,
        linear_value_head_dim=2,
    )
    adapter = Qwen35GGUFAdapter(SimpleNamespace())
    adapter._weight_type_map = {"model.layers.0.linear_attn.out_proj.weight": "Q8_0"}
    tiled_heads = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    expected_heads = torch.tensor([0, 0, 2, 2, 4, 4, 1, 1, 3, 3, 5, 5])

    result = adapter._untile_gdn_weight(
        "model.layers.0.linear_attn.in_proj_z.weight", tiled_heads, config
    )
    torch.testing.assert_close(result, expected_heads)
    for leaf in ("in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias"):
        result = adapter._untile_gdn_weight(
            f"model.layers.0.linear_attn.{leaf}", torch.arange(6), config
        )
        torch.testing.assert_close(result, torch.tensor([0, 2, 4, 1, 3, 5]))

    # Conv Q/K channels remain fixed; only its V tail is untiled.
    conv = torch.cat((torch.tensor([90, 91, 92, 93]), tiled_heads)).view(-1, 1, 1)
    conv_result = adapter._untile_gdn_weight(
        "model.layers.0.linear_attn.conv1d.weight", conv, config
    )
    torch.testing.assert_close(conv_result[:4], conv[:4])
    torch.testing.assert_close(conv_result[4:, 0, 0], expected_heads)


def test_qwen35_gguf_qkv_v_rows_are_untiled_before_fused_loading():
    adapter = Qwen35GGUFAdapter(SimpleNamespace())
    adapter._dequant_stems = set()
    adapter._untile_v_heads = True
    adapter.load_spec = GGUFLoadSpec([], [], {})
    # Q rows, K rows, then tiled V heads [r0g0,r0g1,r1g0,...].
    packed = torch.tensor([90, 91, 80, 81, 0, 1, 2, 3, 4, 5]).view(10, 1)
    source = [("model.layers.0.linear_attn.in_proj_qkv.qweight", packed)]

    with mock.patch.object(
        GGUFWeightsAdapter, "prepare_weights", return_value=iter(source)
    ):
        split = list(adapter.prepare_weights(_model_config()))

    assert [part[:, 0].tolist() for _, part in split] == [
        [90, 91],
        [80, 81],
        [0, 2, 4, 1, 3, 5],
    ]


def test_qwen35_gguf_prepare_loading_disables_runtime_tiled_layout():
    text_config = SimpleNamespace(
        gdn_tiled_v_head_layout=True,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_value_head_dim=128,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(get_text_config=lambda: text_config)
    )
    adapter = Qwen35GGUFAdapter(SimpleNamespace())
    base_spec = GGUFLoadSpec([], [], {})

    with (
        mock.patch.object(
            GGUFWeightsAdapter, "prepare_loading", return_value=base_spec
        ),
        mock.patch.object(
            adapter,
            "get_weight_type_map",
            return_value={
                "model.layers.0.linear_attn.out_proj.weight": "Q4_K",
                "model.layers.1.linear_attn.out_proj.weight": "Q8_0",
            },
        ),
    ):
        adapter.prepare_loading("target.gguf", model_config)

    assert adapter._untile_v_heads is True
    assert text_config.gdn_tiled_v_head_layout is False
    # Q4_K blocks span two value heads; the module must gather in runs of 2.
    assert text_config.gdn_out_proj_v_head_group == 2


def test_qwen35_gguf_adapter_untiles_k_quant_out_proj_in_runs():
    config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    adapter = Qwen35GGUFAdapter(SimpleNamespace())
    adapter._weight_type_map = {"model.layers.0.linear_attn.out_proj.weight": "Q4_K"}
    adapter._out_proj_group = 2
    # Each Q4_K block holds two adjacent tiled heads (r, 2b) and (r, 2b + 1):
    # tiled block order is (r0 run0, r0 run1, r1 run0, ...). Tag each block.
    tiled = torch.cat(
        [torch.full((2, 144), block, dtype=torch.uint8) for block in range(6)],
        dim=1,
    )
    result = adapter._untile_gdn_weight(
        "model.layers.0.linear_attn.out_proj.qweight", tiled, config
    )
    blocks = result.reshape(2, 6, 144)
    assert blocks[:, :, 0].tolist() == [[0, 2, 4, 1, 3, 5]] * 2
    assert (blocks == blocks[:, :, :1]).all()  # blocks moved whole


def test_qwen35_gdn_project_out_gathers_into_run_order():
    """The module's out_proj input gather matches the block-moved columns."""
    num_k, rep, head_dim, hidden = 4, 3, 2, 5
    num_v = num_k * rep
    tiled = torch.randn(hidden, num_v * head_dim, dtype=torch.float64)
    reference = _untile_v_head_axis(
        tiled, axis=1, num_k_heads=num_k, num_v_heads=num_v, values_per_head=2
    )
    run_ordered = _untile_v_head_axis(
        tiled,
        axis=1,
        num_k_heads=num_k,
        num_v_heads=num_v,
        values_per_head=2,
        group=2,
    )
    layer = SimpleNamespace(
        head_v_dim=head_dim,
        _out_proj_head_order=out_proj_v_head_order(num_k, rep, 2),
        out_proj=lambda x: (x @ run_ordered.T, None),
    )
    for name in ("_project_out", "_out_proj_input"):
        setattr(
            layer,
            name,
            MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    x = torch.randn(7, num_v * head_dim, dtype=torch.float64)
    torch.testing.assert_close(layer._project_out(x), x @ reference.T)
