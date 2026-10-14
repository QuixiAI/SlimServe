# SPDX-License-Identifier: Apache-2.0
"""ModelOpt NVFP4 exports (nvidia/GLM-5.3-Flash-NVFP4) keep attention, shared
experts, routers, embeddings and the vision tower in bf16 ONLY through their
exclude list, written in checkpoint names (model.language_model.layers.N...).
The multimodal wrapper builds its modules as language_model.model.layers.N...,
so without the class's hf_to_vllm_mapper nothing matched and bf16 attention
would have been pushed through the NVFP4 path. Compressed-tensors exports never
exposed this: their positive target regex names the routed experts.
"""
import pytest

from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.models.glm5_next import Glm5NextForConditionalGeneration

# The shape of nvidia's export, one layer of each kind.
EXCLUDE = [
    "lm_head",
    "model.language_model.embed_tokens",
    "model.language_model.layers.0.self_attn*",
    "model.language_model.layers.5.self_attn*",
    "model.language_model.layers.5.mlp.gate",
    "model.language_model.layers.5.mlp.shared_experts*",
    "model.visual*",
]


@pytest.fixture
def config():
    qc = ModelOptNvFp4Config.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "kv_cache_scheme": None,
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["Linear"],
                }
            },
            "ignore": EXCLUDE,
        }
    )
    qc.apply_vllm_mapper(Glm5NextForConditionalGeneration.hf_to_vllm_mapper.get_unstacked_mapper())
    return qc


@pytest.mark.parametrize(
    "prefix",
    [
        "language_model.model.layers.5.self_attn.fused_qkv_a_proj",
        "language_model.model.layers.5.self_attn.o_proj",
        "language_model.model.layers.5.self_attn.indexer.wq_b",
        "language_model.model.layers.0.self_attn.in_proj_qkvgfab",
        "language_model.model.layers.5.mlp.shared_experts.down_proj",
        "language_model.model.layers.5.mlp.gate",
        "language_model.model.embed_tokens",
        "language_model.lm_head",
        "visual.blocks.0.attn.qkv",
    ],
)
def test_bf16_modules_are_excluded(config, prefix):
    assert config.is_layer_excluded(prefix), prefix


@pytest.mark.parametrize(
    "prefix",
    [
        "language_model.model.layers.5.mlp.experts",   # routed experts
        "language_model.model.layers.0.mlp.down_proj", # dense MLP, quantized by the export
    ],
)
def test_quantized_modules_are_not_excluded(config, prefix):
    assert not config.is_layer_excluded(prefix), prefix


def test_without_the_mapper_nothing_matches():
    """The failure this guards against: exclusions in checkpoint terms."""
    qc = ModelOptNvFp4Config.from_config(
        {"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_scheme": None,
         "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16},
                                       "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                                       "targets": ["Linear"]}},
         "ignore": EXCLUDE}
    )
    assert not qc.is_layer_excluded("language_model.model.layers.5.self_attn.o_proj")
