# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused contracts for the Qwen3.5 GGUF fused GDN projections."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.model_executor.model_loader.gguf_adapters.base import GGUFLoadSpec
from vllm.model_executor.model_loader.gguf_adapters.default import (
    GGUFWeightsAdapter,
)
from vllm.model_executor.model_loader.gguf_adapters.qwen35 import (
    Qwen35GGUFAdapter,
    _untile_packed_v_head_columns,
    _untile_v_head_axis,
)
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE, is_hybrid
from vllm.model_executor.models.qwen3_5_mtp import (
    _clear_stale_mtp_attention_context,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForCausalLMBase,
    Qwen3_5Model,
    _qwen35_mlp_kind,
)
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig
from vllm.transformers_utils.gguf_config_parser import GGUFConfigParser


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


def test_qwen35_gguf_causal_model_implements_text_mrope_positions():
    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)

    assert isinstance(model, SupportsMRoPE)
    positions, delta = model.get_mrope_input_positions([17, 23, 42, 9], [])

    assert delta == 0
    torch.testing.assert_close(
        positions,
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
    )


def test_qwen35_gguf_causal_model_rejects_multimodal_mrope_features():
    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)

    with pytest.raises(NotImplementedError, match="text-only"):
        model.get_mrope_input_positions([17], [SimpleNamespace()])


def test_qwen35_gguf_causal_model_implements_hybrid_cache_hooks():
    model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    hf_config = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=hf_config,
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        speculative_config=None,
    )

    assert isinstance(model, IsHybrid)
    assert is_hybrid(Qwen3_5ForCausalLM)
    # Protocol stubs are callable too, so ensure the causal class owns real
    # implementations and execute every hook consumed by hybrid cache setup.
    for hook in (
        "get_mamba_state_dtype_from_config",
        "get_mamba_state_shape_from_config",
        "get_mamba_state_copy_func",
    ):
        assert hook in Qwen3_5ForCausalLMBase.__dict__

    assert model.get_mamba_state_dtype_from_config(vllm_config) == (
        torch.bfloat16,
        torch.float32,
    )
    conv_shape, temporal_shape = model.get_mamba_state_shape_from_config(vllm_config)
    assert sorted(conv_shape) == [3, 2560]
    assert temporal_shape == (12, 128, 128)
    copy_funcs = model.get_mamba_state_copy_func()
    assert [func.__name__ for func in copy_funcs] == [
        "get_conv_copy_spec",
        "get_temporal_copy_spec",
    ]


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


def test_qwen35_gguf_untile_rejects_split_quant_blocks():
    with pytest.raises(ValueError, match="not aligned"):
        _untile_packed_v_head_columns(
            torch.empty(1, 1, dtype=torch.uint8),
            num_k_heads=2,
            num_v_heads=6,
            head_dim=128,
            weight_type="Q4_K",
        )


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
        mock.patch.object(adapter, "get_weight_type_map", return_value={}),
    ):
        adapter.prepare_loading("target.gguf", model_config)

    assert adapter._untile_v_heads is True
    assert text_config.gdn_tiled_v_head_layout is False


def test_qwen35_gguf_mtp_maps_only_embedded_nextn_and_shared_heads():
    text_config = SimpleNamespace(num_hidden_layers=64)
    hf_config = SimpleNamespace(
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MTP"],
        get_text_config=lambda: text_config,
    )
    adapter = Qwen35GGUFAdapter(hf_config)
    name_map = adapter.build_name_map(SimpleNamespace(hf_config=hf_config))

    assert name_map["token_embd.weight"] == "model.embed_tokens.weight"
    assert name_map["output.weight"] == "lm_head.weight"
    assert "output_norm.weight" not in name_map
    assert name_map["blk.64.nextn.eh_proj.weight"] == "mtp.fc.weight"
    assert name_map["blk.64.nextn.enorm.weight"] == (
        "mtp.pre_fc_norm_embedding.weight"
    )
    assert name_map["blk.64.nextn.hnorm.weight"] == "mtp.pre_fc_norm_hidden.weight"
    assert name_map["blk.64.nextn.shared_head_norm.weight"] == "mtp.norm.weight"
    assert name_map["blk.64.attn_q.weight"] == (
        "mtp.layers.0.self_attn.q_proj.weight"
    )
    assert name_map["blk.64.ffn_down.weight"] == (
        "mtp.layers.0.mlp.down_proj.weight"
    )
    assert not any(name.startswith("blk.63.") for name in name_map)


def test_qwen35_mtp_model_type_preserves_dense_and_moe_mlp_dispatch():
    dense = Qwen3_5TextConfig(num_hidden_layers=1)
    dense.model_type = "qwen3_5_mtp"
    dense.architectures = ["Qwen3_5MTP"]
    assert _qwen35_mlp_kind(dense) == "dense"

    moe = Qwen3_5MoeTextConfig(num_hidden_layers=1)
    moe.model_type = "qwen3_5_mtp"
    moe.architectures = ["Qwen3_5MoeMTP"]
    assert _qwen35_mlp_kind(moe) == "moe"


def test_qwen35_mtp_replaces_only_its_stale_attention_registrations():
    target_layer = object()
    stale_draft_layer = object()
    context = {
        "model.layers.0.self_attn.attn": target_layer,
        "mtp.layers.0.self_attn.attn": stale_draft_layer,
    }
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context)
    )

    removed = _clear_stale_mtp_attention_context(config, "mtp")

    assert removed == ("mtp.layers.0.self_attn.attn",)
    assert context == {"model.layers.0.self_attn.attn": target_layer}


def test_gguf_parser_detaches_cached_config_for_embedded_mtp_override():
    cached = Qwen3_5TextConfig(num_hidden_layers=64)
    cached.architectures = ["Qwen3_5ForCausalLM"]
    parser = GGUFConfigParser()

    with (
        mock.patch(
            "vllm.transformers_utils.gguf_config_parser.gguf_architecture",
            return_value="qwen35",
        ),
        mock.patch(
            "vllm.transformers_utils.gguf_qwen35.build_qwen35_config_from_gguf",
            return_value=cached,
        ),
    ):
        _, target = parser.parse("model.gguf", trust_remote_code=False)
        _, draft = parser.parse("model.gguf", trust_remote_code=False)

    draft.model_type = "qwen3_5_mtp"
    draft.architectures = ["Qwen3_5MTP"]

    assert target is not draft
    assert target is not cached
    assert target.model_type == "qwen3_5_text"
    assert target.architectures == ["Qwen3_5ForCausalLM"]
