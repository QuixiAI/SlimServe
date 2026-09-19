# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused contracts for the Qwen3.5 GGUF fused GDN projections."""

from types import SimpleNamespace
from unittest import mock

from vllm.model_executor.model_loader.gguf_adapters.qwen35 import (
    Qwen35GGUFAdapter,
)
from vllm.model_executor.models.qwen3_5 import (
    _qwen35_mlp_kind,
)
from vllm.model_executor.models.qwen3_5_mtp import (
    _clear_stale_mtp_attention_context,
)
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig
from vllm.transformers_utils.gguf_config_parser import GGUFConfigParser


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
    assert name_map["blk.64.nextn.enorm.weight"] == ("mtp.pre_fc_norm_embedding.weight")
    assert name_map["blk.64.nextn.hnorm.weight"] == "mtp.pre_fc_norm_hidden.weight"
    assert name_map["blk.64.nextn.shared_head_norm.weight"] == "mtp.norm.weight"
    assert name_map["blk.64.attn_q.weight"] == ("mtp.layers.0.self_attn.q_proj.weight")
    assert name_map["blk.64.ffn_down.weight"] == ("mtp.layers.0.mlp.down_proj.weight")
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
