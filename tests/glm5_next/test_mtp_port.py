"""GLM-5.3-Flash MTP port: draft config, checkpoint-name rewrite, sidecar loading."""

import json
import os

import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

from vllm.config.load import LoadConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.glm5_next_sm120_mtp import Glm5NextMTP


def _vl_config():
    text = PretrainedConfig(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        index_share_for_mtp_iteration=True,
    )
    cfg = PretrainedConfig(
        model_type="glm5_next",
        architectures=["Glm5NextForConditionalGeneration"],
        text_config=text,
        quantization_config={"quant_method": "compressed-tensors", "config_groups": {}},
    )
    return cfg


def test_hf_config_override_promotes_text_config_and_quant():
    draft = SpeculativeConfig.hf_config_override(_vl_config())
    assert draft.model_type == "glm5_next_mtp"
    assert draft.architectures == ["Glm5NextMTPModel"]
    assert draft.n_predict == 1
    assert draft.num_hidden_layers == 45
    assert draft.quantization_config["quant_method"] == "compressed-tensors"
    # Shared config defaults remain conservative; the SM120 profile explicitly
    # enables the older adapter's index sharing through SpeculativeConfig.
    assert draft.index_share_for_mtp_iteration is False


def test_spec_layer_index_is_found_after_prefix_strip():
    from types import SimpleNamespace

    from vllm.model_executor.models.utils import get_spec_layer_idx_from_weight_name

    cfg = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1)
    raw = "model.language_model.layers.45.enorm.weight"
    assert get_spec_layer_idx_from_weight_name(cfg, raw) is None  # the trap
    assert (
        get_spec_layer_idx_from_weight_name(cfg, Glm5NextMTP.strip_hf_prefix(raw)) == 45
    )
    assert (
        get_spec_layer_idx_from_weight_name(
            cfg, Glm5NextMTP.strip_hf_prefix("model.language_model.layers.44.x")
        )
        is None
    )


def test_rewrite_spec_layer_name_targets_the_mtp_block():
    r = Glm5NextMTP.rewrite_spec_layer_name
    base = "model.language_model.layers.45."
    assert (
        r(45, base + "self_attn.q_a_proj.weight")
        == "model.layers.45.mtp_block.self_attn.q_a_proj.weight"
    )
    assert (
        r(45, base + "mlp.experts.7.gate_proj.weight_scale")
        == "model.layers.45.mtp_block.mlp.experts.7.gate_proj.weight_scale"
    )
    assert (
        r(45, base + "mlp.gate.e_score_correction_bias")
        == "model.layers.45.mtp_block.mlp.gate.e_score_correction_bias"
    )
    assert r(45, base + "enorm.weight") == "model.layers.45.enorm.weight"
    assert r(45, base + "eh_proj.weight") == "model.layers.45.eh_proj.weight"
    assert (
        r(45, base + "shared_head.norm.weight")
        == "model.layers.45.shared_head.norm.weight"
    )
    assert r(45, base + "embed_tokens.weight") == "model.embed_tokens.weight"
    assert r(45, "lm_head.weight") == "lm_head.weight"


def _fake_checkpoint(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"model.language_model.layers.0.x": torch.zeros(2)}, str(shard))
    save_file(
        {"model.language_model.layers.45.enorm.weight": torch.ones(2)},
        str(tmp_path / "model_mtp.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.layers.0.x": shard.name}})
    )


def test_loader_index_filter_keeps_only_indexed_shards_by_default(tmp_path):
    _fake_checkpoint(tmp_path)
    loader = DefaultModelLoader(LoadConfig())
    _, files, use_st = loader._prepare_weights(str(tmp_path), None, None, True, None)
    assert use_st and [os.path.basename(f) for f in files] == [
        "model-00001-of-00001.safetensors"
    ]


def test_loader_explicit_file_override_bypasses_the_index_filter(tmp_path):
    _fake_checkpoint(tmp_path)
    loader = DefaultModelLoader(LoadConfig())
    _, files, use_st = loader._prepare_weights(
        str(tmp_path), None, None, True, ["model_mtp.safetensors"]
    )
    assert use_st and [os.path.basename(f) for f in files] == ["model_mtp.safetensors"]


def test_loader_glob_override_still_filters_by_index(tmp_path):
    _fake_checkpoint(tmp_path)
    loader = DefaultModelLoader(LoadConfig())
    _, files, _ = loader._prepare_weights(
        str(tmp_path), None, None, True, ["*.safetensors"]
    )
    assert [os.path.basename(f) for f in files] == ["model-00001-of-00001.safetensors"]
