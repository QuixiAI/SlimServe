# SPDX-License-Identifier: Apache-2.0
"""CPU contract gates for the opt-in adapter; not serving acceptance."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Glm5NextConfig, Glm5NextTextConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.glm5_next_mtp import Glm5NextMTP, _draft_config
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)


@pytest.mark.parametrize("multimodal", [False, True])
def test_mtp_config_preserves_quantization_and_mla(multimodal):
    text = Glm5NextTextConfig(num_nextn_predict_layers=1)
    text.index_share_for_mtp_iteration = True
    quant = {"quant_method": "compressed-tensors", "config_groups": {"experts": {}}}
    if multimodal:
        root = Glm5NextConfig(text_config=text.to_dict())
        root.architectures = ["Glm5NextForConditionalGeneration"]
        root.quantization_config = quant
    else:
        root = text
        root.quantization_config = quant
    draft = SpeculativeConfig.hf_config_override(root)
    assert draft.model_type == "glm5_next_mtp"
    assert draft.architectures == ["Glm5NextMTPModel"]
    assert draft.n_predict == 1
    assert draft.quantization_config == quant
    assert not draft.index_share_for_mtp_iteration
    assert ModelArchConfigConvertorBase(draft, draft).is_deepseek_mla()
    converter = MODEL_ARCH_CONFIG_CONVERTORS[draft.model_type](draft, draft)
    assert converter.get_num_hidden_layers() == 1
    if multimodal:
        assert root.text_config.model_type == "glm5_next_text"
        assert root.text_config.index_share_for_mtp_iteration
        assert draft.quantization_config is not root.quantization_config


def _loader_fixture():
    model = Glm5NextMTP.__new__(Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1)
    indexer = nn.Module()
    indexer.wk = nn.Linear(8, 4, bias=False)
    indexer.weights_proj = nn.Linear(8, 2, bias=False)
    indexer.index_kpool_compress_ape = nn.Parameter(torch.empty(4, 4))
    attention, block, layer, predictor = (nn.Module() for _ in range(4))
    attention.indexer = indexer
    block.self_attn = attention
    layer.mtp_block = block
    predictor.layers = nn.ModuleDict({"45": layer})
    model.model = predictor
    return model, indexer


def test_constructor_uses_draft_not_target_vlm_config():
    target, draft = object(), object()
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=target),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=draft)
        ),
    )
    assert _draft_config(config) is draft


def test_mtp_loader_keeps_glm_indexer_separate(monkeypatch):
    model, indexer = _loader_fixture()
    inherited = []

    def collect(self, weights):
        inherited.extend(weights)
        return {name for name, _ in inherited}

    monkeypatch.setattr(DeepSeekMTP, "load_weights", collect)
    prefix = "model.language_model.layers.45."
    inputs = [
        (prefix + "self_attn.indexer.wk.weight", torch.randn_like(indexer.wk.weight)),
        (
            prefix + "self_attn.indexer.weights_proj.weight",
            torch.randn_like(indexer.weights_proj.weight),
        ),
        (
            prefix + "self_attn.indexer.index_kpool_compress_ape",
            torch.randn_like(indexer.index_kpool_compress_ape),
        ),
        (prefix + "eh_proj.weight", torch.randn(8, 16)),
        ("model.language_model.layers.44.self_attn.q_a_proj.weight", torch.empty(0)),
        ("model.visual.patch_embed.proj.weight", torch.empty(0)),
    ]
    loaded = model.load_weights(iter(inputs))
    assert [name for name, _ in inherited] == ["model.layers.45.eh_proj.weight"]
    assert len(loaded) == 4
    for suffix, value in inputs[:3]:
        target = suffix.replace("model.language_model.", "model.").replace(
            "layers.45.", "layers.45.mtp_block."
        )
        assert target in loaded
        torch.testing.assert_close(dict(model.named_parameters())[target], value)


def test_mtp_loader_refuses_unknown_indexer_parameter(monkeypatch):
    model, _ = _loader_fixture()
    monkeypatch.setattr(DeepSeekMTP, "load_weights", lambda self, weights: set(weights))
    with pytest.raises(KeyError, match="unknown"):
        name = "model.language_model.layers.45.self_attn.indexer.unknown"
        model.load_weights([(name, torch.empty(0))])


def test_proposer_load_uses_glm_image_token_and_language_model(monkeypatch):
    from vllm.v1.spec_decode import llm_base_proposer as proposer_module

    proposer = proposer_module.SpecDecodeBaseProposer.__new__(
        proposer_module.SpecDecodeBaseProposer
    )
    draft = nn.Module()
    draft.config = SimpleNamespace()
    language_model = nn.Module()
    target_cls = type(
        "Glm5NextForConditionalGeneration",
        (nn.Module,),
        {"get_language_model": lambda self: language_model},
    )
    target = target_cls()
    target.config = SimpleNamespace(image_token_id=260001)
    proposer.vllm_config = object()
    proposer.supports_mm_inputs = False
    proposer.parallel_drafting = False
    proposer._get_model = lambda: draft
    shared = []
    proposer._maybe_share_embeddings = lambda model: shared.append(model)
    proposer._maybe_share_lm_head = lambda model: shared.append(model)
    monkeypatch.setattr(proposer_module, "get_layers_from_vllm_config", lambda *_: {})
    monkeypatch.setattr(proposer_module, "supports_multimodal", lambda _: True)
    proposer.load_model(target)
    assert draft.config.image_token_index == 260001
    assert shared == [language_model, language_model]
