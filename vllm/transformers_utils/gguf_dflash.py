# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a DeepSeek-V4 DSpark draft config from standardized dflash GGUF."""

from __future__ import annotations

import struct
from collections import Counter
from functools import cache
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_native import _field
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)

ARCH = "dflash"
BLOCK_COUNT = 3
BLOCK_SIZE = 5
HIDDEN_SIZE = 4096
MARKOV_RANK = 256
MASK_TOKEN_ID = 128_799
TARGET_LAYERS = (41, 42, 43)
VOCAB_SIZE = 129_280

EXPECTED_TYPE_COUNTS = {
    "F32": 45,
    "F16": 2,
    "Q8_0": 25,
    "Q2_K": 9,
}


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


# Exact metadata emitted by the published standardized conversion.  Keeping
# this alongside the tensor inventory prevents a custom/legacy ``dspark.*``
# artifact from reaching a loader that expects llama.cpp's canonical dflash
# contract.
EXPECTED_METADATA: dict[str, object] = {
    "general.architecture": ARCH,
    "general.name": "DeepSeek-V4-Flash-0731-DSpark-Drafter",
    "general.source.url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
    "general.source.revision": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
    "general.license": "MIT",
    "general.quantization_version": 2,
    "dflash.block_count": BLOCK_COUNT,
    "dflash.context_length": 1_048_576,
    "dflash.embedding_length": HIDDEN_SIZE,
    "dflash.attention.head_count": 64,
    "dflash.attention.head_count_kv": 1,
    "dflash.rope.scaling.type": "yarn",
    "dflash.rope.scaling.factor": _f32(16.0),
    "dflash.rope.scaling.original_context_length": 65_536,
    "dflash.rope.scaling.yarn_beta_fast": _f32(32.0),
    "dflash.rope.scaling.yarn_beta_slow": _f32(1.0),
    "dflash.rope.freq_base": _f32(10_000.0),
    "dflash.attention.layer_norm_rms_epsilon": _f32(1e-6),
    "dflash.expert_count": 256,
    "dflash.expert_used_count": 6,
    "dflash.expert_gating_func": 4,
    "dflash.attention.key_length": 512,
    "dflash.attention.value_length": 512,
    "dflash.rope.dimension_count": 64,
    "dflash.attention.q_lora_rank": 1024,
    "dflash.attention.sliding_window": 128,
    "dflash.expert_feed_forward_length": 2048,
    "dflash.expert_shared_count": 1,
    "dflash.expert_weights_scale": _f32(1.5),
    "dflash.expert_weights_norm": True,
    "dflash.swiglu_clamp_exp": (_f32(10.0),) * BLOCK_COUNT,
    "dflash.swiglu_clamp_shexp": (_f32(10.0),) * BLOCK_COUNT,
    "dflash.attention.output_group_count": 8,
    "dflash.attention.output_lora_rank": 1024,
    "dflash.attention.compress_ratios": (0, 0, 0),
    "dflash.hyper_connection.count": 4,
    "dflash.hyper_connection.sinkhorn_iterations": 20,
    "dflash.hyper_connection.epsilon": _f32(1e-6),
    "dflash.hash_layer_count": 0,
    "dflash.block_size": BLOCK_SIZE,
    "dflash.target_layers": TARGET_LAYERS,
    "dflash.vocab_size": VOCAB_SIZE,
    "tokenizer.ggml.model": "no_vocab",
    "tokenizer.ggml.mask_token_id": MASK_TOKEN_ID,
}


def _layer_tensor_specs(layer: int) -> dict[str, tuple[tuple[int, ...], str]]:
    prefix = f"blk.{layer}"
    return {
        f"{prefix}.attn_sinks.weight": ((64,), "F32"),
        f"{prefix}.attn_norm.weight": ((HIDDEN_SIZE,), "F32"),
        f"{prefix}.ffn_norm.weight": ((HIDDEN_SIZE,), "F32"),
        f"{prefix}.attn_kv_a_norm.weight": ((512,), "F32"),
        f"{prefix}.attn_q_a_norm.weight": ((1024,), "F32"),
        f"{prefix}.attn_kv.weight": ((HIDDEN_SIZE, 512), "Q8_0"),
        f"{prefix}.attn_q_a.weight": ((HIDDEN_SIZE, 1024), "Q8_0"),
        f"{prefix}.attn_q_b.weight": ((1024, 32768), "Q8_0"),
        f"{prefix}.attn_output_a.weight": ((HIDDEN_SIZE, 8192), "Q8_0"),
        f"{prefix}.attn_output_b.weight": ((8192, HIDDEN_SIZE), "Q8_0"),
        f"{prefix}.ffn_gate_shexp.weight": ((HIDDEN_SIZE, 2048), "Q8_0"),
        f"{prefix}.ffn_up_shexp.weight": ((HIDDEN_SIZE, 2048), "Q8_0"),
        f"{prefix}.ffn_down_shexp.weight": ((2048, HIDDEN_SIZE), "Q8_0"),
        f"{prefix}.ffn_gate_inp.weight": ((HIDDEN_SIZE, 256), "F32"),
        f"{prefix}.exp_probs_b.bias": ((256,), "F32"),
        f"{prefix}.ffn_gate_exps.weight": (
            (HIDDEN_SIZE, 2048, 256),
            "Q2_K",
        ),
        f"{prefix}.ffn_up_exps.weight": (
            (HIDDEN_SIZE, 2048, 256),
            "Q2_K",
        ),
        f"{prefix}.ffn_down_exps.weight": (
            (2048, HIDDEN_SIZE, 256),
            "Q2_K",
        ),
        f"{prefix}.hc_attn_fn.weight": ((16384, 24), "F32"),
        f"{prefix}.hc_ffn_fn.weight": ((16384, 24), "F32"),
        f"{prefix}.hc_attn_base.weight": ((24,), "F32"),
        f"{prefix}.hc_ffn_base.weight": ((24,), "F32"),
        f"{prefix}.hc_attn_scale.weight": ((3,), "F32"),
        f"{prefix}.hc_ffn_scale.weight": ((3,), "F32"),
    }


def dflash_tensor_specs() -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for layer in range(BLOCK_COUNT):
        specs.update(_layer_tensor_specs(layer))
    specs.update(
        {
            "fc.weight": ((12288, HIDDEN_SIZE), "Q8_0"),
            "enc.output_norm.weight": ((HIDDEN_SIZE,), "F32"),
            "output_norm.weight": ((HIDDEN_SIZE,), "F32"),
            "markov_w1.weight": ((MARKOV_RANK, VOCAB_SIZE), "F16"),
            "markov_w2.weight": ((MARKOV_RANK, VOCAB_SIZE), "F16"),
            "output_hc_fn.weight": ((16384, 4), "F32"),
            "output_hc_base.weight": ((4,), "F32"),
            "output_hc_scale.weight": ((1,), "F32"),
            "conf_proj.weight": ((4352,), "F32"),
        }
    )
    return specs


def validate_dflash_reader(reader: Any) -> None:
    """Validate the published standardized 0731 DSpark schema."""
    for key, expected in EXPECTED_METADATA.items():
        if key not in reader.fields:
            raise ValueError(f"dflash GGUF is missing required metadata {key!r}")
        observed = _field(reader, key)
        if isinstance(expected, tuple):
            observed = tuple(type(expected[0])(value) for value in observed)
        elif isinstance(expected, bool):
            observed = bool(observed)
        elif isinstance(expected, int):
            observed = int(observed)
        elif isinstance(expected, float):
            observed = float(observed)
        elif isinstance(expected, str):
            observed = str(observed)
        if observed != expected:
            raise ValueError(
                f"dflash GGUF metadata {key!r} is {observed!r}, expected {expected!r}"
            )

    expected_specs = dflash_tensor_specs()
    observed = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(expected_specs.keys() - observed.keys())
    extra = sorted(observed.keys() - expected_specs.keys())
    if missing or extra:
        raise ValueError(
            "dflash GGUF tensor inventory differs from the published 81-tensor "
            f"schema: missing={missing}, extra={extra}"
        )

    for name, (expected_shape, expected_type) in expected_specs.items():
        tensor = observed[name]
        shape = tuple(int(dim) for dim in tensor.shape)
        tensor_type = tensor.tensor_type.name
        if shape != expected_shape or tensor_type != expected_type:
            raise ValueError(
                f"dflash tensor {name!r} is {shape}/{tensor_type}, "
                f"expected {expected_shape}/{expected_type}"
            )

    type_counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)
    if dict(type_counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(
            f"dflash GGUF type inventory is {dict(type_counts)}, "
            f"expected {EXPECTED_TYPE_COUNTS}"
        )


@cache
def build_dflash_config_from_gguf(gguf_path: str) -> Any:
    """Assemble the draft config from canonical llama.cpp dflash metadata."""
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    reader = gguf_reader(str(gguf_path))
    validate_dflash_reader(reader)
    g = lambda key, default=None: _field(  # noqa: E731
        reader, f"{ARCH}.{key}", default
    )

    one_based_layers = tuple(int(layer) for layer in g("target_layers"))
    target_layer_ids = tuple(layer - 1 for layer in one_based_layers)
    target_num_hidden_layers = max(one_based_layers)
    swiglu = tuple(float(value) for value in g("swiglu_clamp_exp"))
    if len(set(swiglu)) != 1:
        raise ValueError(f"dflash has per-layer SwiGLU limits: {swiglu!r}")
    swiglu_limit = swiglu[0]

    rope_parameters: dict[str, Any] = {"rope_type": "default"}
    if str(g("rope.scaling.type", "")) == "yarn":
        rope_parameters = {
            "rope_type": "yarn",
            "factor": float(g("rope.scaling.factor")),
            "original_max_position_embeddings": int(
                g("rope.scaling.original_context_length")
            ),
            "beta_fast": float(g("rope.scaling.yarn_beta_fast", 32.0)),
            "beta_slow": float(g("rope.scaling.yarn_beta_slow", 1.0)),
        }

    cfg = DeepseekV4Config(
        model_type="deepseek_v4",
        architectures=["DSparkDraftModel"],
        hidden_size=int(g("embedding_length")),
        num_hidden_layers=target_num_hidden_layers,
        n_mtp_layers=int(g("block_count")),
        vocab_size=int(g("vocab_size")),
        draft_vocab_size=int(g("vocab_size")),
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=int(g("attention.key_length")),
        qk_rope_head_dim=int(g("rope.dimension_count")),
        q_lora_rank=int(g("attention.q_lora_rank")),
        o_lora_rank=int(g("attention.output_lora_rank")),
        o_groups=int(g("attention.output_group_count")),
        sliding_window=int(g("attention.sliding_window")),
        compress_ratios=[0] * target_num_hidden_layers,
        n_routed_experts=int(g("expert_count")),
        n_shared_experts=int(g("expert_shared_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        norm_topk_prob=bool(g("expert_weights_norm")),
        routed_scaling_factor=float(g("expert_weights_scale")),
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        num_hash_layers=int(g("hash_layer_count")),
        hc_mult=int(g("hyper_connection.count")),
        hc_eps=round(float(g("hyper_connection.epsilon")), 12),
        hc_sinkhorn_iters=int(g("hyper_connection.sinkhorn_iterations")),
        index_topk=0,
        swiglu_limit=swiglu_limit,
        rms_norm_eps=round(float(g("attention.layer_norm_rms_epsilon")), 12),
        rope_theta=float(g("rope.freq_base")),
        rope_parameters=rope_parameters,
        max_position_embeddings=int(g("context_length")),
        dspark_block_size=int(g("block_size")),
        dspark_markov_rank=MARKOV_RANK,
        dspark_noise_token_id=int(_field(reader, "tokenizer.ggml.mask_token_id")),
        dspark_target_layer_ids=list(target_layer_ids),
        n_predict=int(g("block_size")),
        sample_from_anchor=True,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    cfg.architectures = ["DSparkDraftModel"]
    logger.info(
        "Built DeepSeek-V4 DSpark config from dflash GGUF: %d draft layers, "
        "block %d, target layers %s",
        cfg.n_mtp_layers,
        cfg.dspark_block_size,
        cfg.dspark_target_layer_ids,
    )
    return cfg
