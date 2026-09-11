# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    _remap_qsa_cache_scale_name,
)
from vllm.models.qwen4_exp.nvidia.mtp import _remap_mtp_weight_name


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "layers.1.linear_attn.in_proj_qkv.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            (0, 1, 2),
        ),
        (
            "layers.1.linear_attn.in_proj_z.weight",
            "layers.1.linear_attn.in_proj_qkvz.weight",
            3,
        ),
        (
            "layers.1.linear_attn.in_proj_b.weight",
            "layers.1.linear_attn.in_proj_ba.weight",
            0,
        ),
        (
            "layers.1.mlp.gate_proj.weight",
            "layers.1.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.1.mlp.experts.0.gate_proj.weight",
            "layers.1.mlp.experts.0.gate_proj.weight",
            None,
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            "layers.0.self_attn.indexer.index_qk_proj.weight",
            None,
        ),
        # HC merged down-and-injection projection.
        (
            "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        (
            "layers.0.attn_hyper_connection.block_inject_weight.weight",
            "layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
        (
            "hyper_connection_mixer.input_mix_weight_down.weight",
            "hyper_connection_mixer.input_mix_weight_down.weight",
            None,
        ),
        (
            "layers.1.ple.ple_embedding.layer_multipliers",
            "layers.1.ple.ple_embedding.layer_multipliers",
            None,
        ),
    ],
)
def test_text_checkpoint_mapper_preserves_qwen4_exp_specific_weights(
    checkpoint_name: str,
    model_name: str,
    shard_id: str | int | tuple[int, ...] | None,
) -> None:
    assert Qwen4ExpModel.hf_to_vllm_mapper._map_name_with_shard(checkpoint_name) == (
        model_name,
        shard_id,
    )


def test_vl_checkpoint_mapper_composes_language_and_vision_paths() -> None:
    outer_mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper
    assert outer_mapper._map_name(
        "model.language_model.layers.0.ple.key_proj.weight"
    ) == ("language_model.model.layers.0.ple.key_proj.weight")
    assert outer_mapper._map_name("lm_head.weight") == ("language_model.lm_head.weight")

    visual_name = outer_mapper._map_name("model.visual.blocks.0.attn.q.weight")
    assert visual_name == "visual.blocks.0.attn.q.weight"
    child_name = visual_name.removeprefix("visual.")
    assert Qwen3_VisionTransformer.hf_to_vllm_mapper._map_name_with_shard(
        child_name
    ) == ("blocks.0.attn.qkv.weight", "q")

    packed_visual_name = outer_mapper._map_name("model.visual.blocks.0.attn.qkv.weight")
    assert packed_visual_name == "visual.blocks.0.attn.qkv.weight"
    assert Qwen3_VisionTransformer.hf_to_vllm_mapper._map_name_with_shard(
        packed_visual_name.removeprefix("visual.")
    ) == ("blocks.0.attn.qkv.weight", None)


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name"),
    [
        (
            "layers.0.self_attn.k_proj.k_scale",
            "layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.v_proj.output_scale",
            "layers.0.self_attn._v_scale",
        ),
        (
            "language_model.model.layers.0.self_attn.attn.k_scale",
            "language_model.model.layers.0.self_attn._k_scale",
        ),
        (
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
            "layers.0.self_attn.indexer.index_qk_proj.weight_scale",
        ),
        (
            "layers.1.self_attn.k_proj.k_scale",
            "layers.1.self_attn.k_proj.k_scale",
        ),
    ],
)
def test_only_qsa_main_cache_scales_move_to_the_merged_owner(
    checkpoint_name: str,
    model_name: str,
) -> None:
    assert _remap_qsa_cache_scale_name(checkpoint_name, frozenset({0})) == model_name


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name"),
    [
        ("mtp.fc_embedding.weight", "model.fc_embedding.weight"),
        (
            "model.mtp.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "model.language_model.mtp.pre_fc_norm_hidden.weight",
            "model.pre_fc_norm_hidden.weight",
        ),
        ("model.shared_head.head.weight", "lm_head.weight"),
        (
            "language_model.model.shared_head.head.weight",
            "lm_head.weight",
        ),
        (
            "model.language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        (
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        (
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        (
            "model.language_model.shared_head.head.weight",
            "lm_head.weight",
        ),
        (
            "model.language_model.model.lm_head.weight",
            "lm_head.weight",
        ),
        (
            "model.language_model.mtp.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.gate_up_proj",
        ),
        ("model.language_model.layers.0.mlp.down_proj.weight", None),
    ],
)
def test_qwen4_exp_mtp_weight_name_mapping(
    checkpoint_name: str,
    model_name: str | None,
) -> None:
    assert _remap_mtp_weight_name(checkpoint_name) == model_name


# --- ModelOpt MIXED_PRECISION release (nvidia/Qwen3.8-Flash-Next-NVFP4) -----


def _mixed_config(quantized_layers):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": ["lm_head", "model.language_model.layers.3.self_attn*"],
            "quantized_layers": quantized_layers,
        }
    )


def test_modelopt_mixed_ple_tables_take_the_fp8_host_path() -> None:
    from vllm.models.qwen4_exp.nvidia.ple_layer import (
        Qwen4ExpPLEFp8EmbeddingMethod,
        _get_ple_embedding_quant_method,
    )

    cfg = _mixed_config(
        {
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {
                "quant_algo": "FP8"
            },
            "model.language_model.layers.3.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        }
    )
    # The vLLM-side prefix roots the language model differently from the
    # checkpoint; the lookup must bridge that.
    method = _get_ple_embedding_quant_method(
        cfg, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    )
    assert isinstance(method, Qwen4ExpPLEFp8EmbeddingMethod)
    # An unlisted table (bf16 in the checkpoint) stays unquantized.
    assert (
        _get_ple_embedding_quant_method(
            cfg, "language_model.model.layers.5.ple.ple_embedding.ngram_embedding"
        )
        is None
    )


def test_modelopt_mixed_qsa_sees_no_quant_config() -> None:
    from vllm.models.qwen4_exp.nvidia.model import without_modelopt_fp4

    cfg = _mixed_config(
        {"model.language_model.layers.3.mlp.experts": {"quant_algo": "NVFP4"}}
    )
    assert without_modelopt_fp4(cfg) is None


def test_modelopt_mixed_drafter_experts_become_block_fp8() -> None:
    from vllm.models.qwen4_exp.nvidia.mtp import (
        Qwen4ExpDraftBlockFp8Config,
        _draft_quant_config_from_modelopt_mixed,
    )

    cfg = _mixed_config(
        {
            "model.language_model.layers.3.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_PB_WO", "group_size": 128},
        }
    )
    draft = _draft_quant_config_from_modelopt_mixed(cfg, mtp_start_layer_idx=48)
    assert isinstance(draft, Qwen4ExpDraftBlockFp8Config)
    assert draft.expert_prefixes == {"mtp.layers.48.mlp.experts"}
    assert draft.weight_block_size == [128, 128]
    assert draft.is_checkpoint_fp8_serialized and draft.activation_scheme == "dynamic"

    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        UnquantizedLinearMethod,
    )

    # Every non-expert draft module is bf16 in the release.
    linear = ColumnParallelLinear.__new__(ColumnParallelLinear)
    assert isinstance(
        draft.get_quant_method(linear, "mtp.layers.48.self_attn.qkv_proj"),
        UnquantizedLinearMethod,
    )
    assert draft.get_quant_method(object(), "mtp.layers.48.self_attn.attn") is None

    # A release with no drafter experts listed needs no draft quant config.
    assert (
        _draft_quant_config_from_modelopt_mixed(
            _mixed_config({"model.language_model.layers.3.mlp.experts": {"quant_algo": "NVFP4"}}),
            48,
        )
        is None
    )
