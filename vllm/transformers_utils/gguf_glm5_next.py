# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the GLM-5.3-Flash config and tokenizer from a ``glm5-next`` GGUF.

Same reasoning as `gguf_kimi_k3`: transformers' GGUF reader has an
architecture whitelist and ``glm5-next`` (antirez's ds4-layout conversion of
zai-org/GLM-5.3-Flash) is not on it, so the metadata is read directly.

Unlike the Kimi file this one is metadata-rich: every hyperparameter the text
model consumes is present under ``glm5-next.*``, including the per-layer
``layer_types`` list, the KDA gate floor, the indexer pool size and the mHC
constants. Tensor shapes are used only to *verify* the metadata (see
``gguf_adapters/glm5_next.py``), never to guess a value.

The result is a ``Glm5NextConfig`` (text + default vision sub-config) with
``architectures=["Glm5NextForConditionalGeneration"]``: that is the only
registered entry for the family, and the campaign profile serves it with
``language_model_only=True`` so the tower is never constructed. The GGUF
carries no vision tensors.

Text-config field names follow the native transformers ``Glm5NextTextConfig``
plus the raw checkpoint keys vLLM's model reads directly (``linear_attn_config``,
``index_kpool_compress``, ``first_k_dense_replace``, ``num_nextn_predict_layers``,
``scoring_func``, ``topk_method``).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_native import (
    GLM4_PRETOKENIZER_REGEX,
    _field,
    build_bpe_tokenizer,
    stop_token_ids_from_gguf,
)
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)

ARCH = "glm5-next"

# `glm5-next.layer_types` element -> transformers layer type.
LAYER_TYPE_NAMES = {0: "linear_attention", 1: "deepseek_sparse_attention"}

# Not carried by the file and not derivable from a shape. GLM-5.3-Flash
# reference config.json values.
_ARCH_CONSTANTS: dict[str, Any] = {
    "hidden_act": "silu",
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    # Route in fp32 like ds4 (the GGUF keeps ffn_gate_inp in F32); the Metal
    # single-group router kernel also requires fp32 logits.
    "moe_router_dtype": "float32",
    "n_group": 1,
    "topk_group": 1,
    "moe_layer_freq": 1,
    "attention_bias": False,
    "tie_word_embeddings": False,
    # Pooled indexer: the file carries the pool size; the compress / tail
    # flags are what the pooled path asserts (glm5_next_indexer.py).
    "index_kpool_compress": True,
    "index_kpool_always_select_tail": True,
}


def is_glm5_next_gguf(gguf_path: str) -> bool:
    return str(_field(gguf_reader(str(gguf_path)), "general.architecture")) == ARCH


def _config_classes():
    """Native transformers classes when present, else the vendored mirrors."""
    try:
        from transformers import Glm5NextConfig, Glm5NextTextConfig
    except ImportError:
        from vllm.transformers_utils.configs.glm5_next import (
            Glm5NextConfig,
            Glm5NextTextConfig,
        )
    return Glm5NextConfig, Glm5NextTextConfig


def _construct(cls, fields: dict[str, Any]):
    """Build ``cls(**fields)``; if the class rejects extras, attach them after."""
    try:
        return cls(**fields)
    except TypeError:
        known = {
            k: v
            for k, v in fields.items()
            if k in getattr(cls, "__dataclass_fields__", {})
        }
        cfg = cls(**known)
        for key, value in fields.items():
            if key not in known:
                setattr(cfg, key, value)
        return cfg


def _optional_int(value: Any) -> int | None:
    """Token-id keys are optional in GGUF; keep an absent one as None."""
    return None if value is None else int(value)


def text_config_fields_from_gguf(gguf_path: str) -> dict[str, Any]:
    """The text-config dict, straight from ``glm5-next.*`` metadata."""
    r = gguf_reader(str(gguf_path))

    def g(key: str, default: Any = None) -> Any:
        return _field(r, f"{ARCH}.{key}", default)

    def req(key: str) -> Any:
        value = g(key)
        if value is None:
            raise ValueError(f"{gguf_path}: missing required key {ARCH}.{key}")
        return value

    block_count = int(req("block_count"))
    n_nextn = int(g("nextn_predict_layers", 0) or 0)
    trunk = g("trunk_block_count")
    num_layers = int(trunk) if trunk is not None else block_count - n_nextn
    if num_layers + n_nextn != block_count:
        raise ValueError(
            f"{gguf_path}: block_count {block_count} != trunk {num_layers} + "
            f"nextn {n_nextn}"
        )

    raw_types = list(req("layer_types"))
    if len(raw_types) < num_layers:
        raise ValueError(
            f"{gguf_path}: layer_types has {len(raw_types)} entries for "
            f"{num_layers} trunk layers"
        )
    layer_types = []
    for i, t in enumerate(raw_types[:num_layers]):
        name = LAYER_TYPE_NAMES.get(int(t))
        if name is None:
            raise ValueError(f"{gguf_path}: unknown layer_types[{i}] = {t}")
        layer_types.append(name)
    for i, t in enumerate(raw_types[num_layers:]):
        if int(t) != 1:
            raise ValueError(
                f"{gguf_path}: nextn block {num_layers + i} is not sparse MLA"
            )

    first_dense = int(req("leading_dense_block_count"))
    mlp_layer_types = ["dense"] * first_dense + ["sparse"] * (num_layers - first_dense)

    rope_dim = int(g("attention.rope_dimension_count", 0) or 0)
    qk_nope = int(req("attention.key_length")) - rope_dim
    v_head = int(req("attention.value_length"))
    heads = int(req("attention.head_count"))

    kda_heads = int(req("linear_attention.head_count"))
    kda_head_dim = int(req("linear_attention.head_dimension"))
    conv = int(req("linear_attention.conv_kernel"))
    gate_lower_bound = g("linear_attention.gate_lower_bound")
    gate_lower_bound = None if gate_lower_bound is None else float(gate_lower_bound)

    # Kimi's convention for the layer lists is 1-based; vLLM's shared KDA
    # layer reads only num_heads / head_dim / short_conv_kernel_size /
    # gate_lower_bound from this dict, the model itself reads `layer_types`.
    linear_attn_config = {
        "kda_layers": [
            i + 1 for i, t in enumerate(layer_types) if t == "linear_attention"
        ],
        "full_attn_layers": [
            i + 1 for i, t in enumerate(layer_types) if t != "linear_attention"
        ],
        "num_heads": kda_heads,
        "head_dim": kda_head_dim,
        "short_conv_kernel_size": conv,
        "gate_lower_bound": gate_lower_bound,
        "safe_gate": gate_lower_bound is not None,
        "use_full_rank_gate": False,
    }

    fields: dict[str, Any] = {
        "vocab_size": int(req("vocab_size")),
        "hidden_size": int(req("embedding_length")),
        "intermediate_size": int(req("feed_forward_length")),
        "moe_intermediate_size": int(req("expert_feed_forward_length")),
        "num_hidden_layers": num_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "max_position_embeddings": int(req("context_length")),
        # Stored as fp32; round so it matches the reference exactly.
        "rms_norm_eps": round(float(req("attention.layer_norm_rms_epsilon")), 12),
        # MLA (NoPE: rope dim 0)
        "q_lora_rank": int(req("attention.q_lora_rank")),
        "kv_lora_rank": int(req("attention.kv_lora_rank")),
        "qk_rope_head_dim": rope_dim,
        "qk_nope_head_dim": qk_nope,
        "v_head_dim": v_head,
        # MoE
        "n_routed_experts": int(req("expert_count")),
        "num_experts_per_tok": int(req("expert_used_count")),
        "n_shared_experts": int(req("expert_shared_count")),
        "routed_scaling_factor": float(req("expert_weights_scale")),
        "norm_topk_prob": bool(req("expert_weights_norm")),
        "first_k_dense_replace": first_dense,
        "mlp_layer_types": mlp_layer_types,
        "swiglu_limit": float(req("swiglu_limit")),
        # Hybrid layout
        "layer_types": layer_types,
        "linear_attn_config": linear_attn_config,
        "linear_num_heads": kda_heads,
        "linear_head_dim": kda_head_dim,
        "linear_conv_kernel_dim": conv,
        "linear_lower_bound": gate_lower_bound,
        # Pooled DSA indexer
        "index_n_heads": int(req("attention.indexer.head_count")),
        "index_head_dim": int(req("attention.indexer.key_length")),
        "index_topk": int(req("attention.indexer.top_k")),
        "index_kpool": int(req("attention.indexer.pool_size")),
        # mHC
        "hc_mult": int(req("hyper_connection.count")),
        "hc_sinkhorn_iters": int(req("hyper_connection.sinkhorn_iterations")),
        "hc_eps": round(float(req("hyper_connection.epsilon")), 12),
        # MTP head
        "num_nextn_predict_layers": n_nextn,
        # Tokens. eos is the full end-of-generation set (eos/eot/eom), not the
        # single `<|endoftext|>` the model never emits.
        "eos_token_id": stop_token_ids_from_gguf(r),
        "bos_token_id": _optional_int(_field(r, "tokenizer.ggml.bos_token_id")),
        "pad_token_id": _optional_int(_field(r, "tokenizer.ggml.padding_token_id")),
        **_ARCH_CONSTANTS,
    }
    return fields


@cache
def build_glm5_next_config_from_gguf(gguf_path: str) -> Any:
    """Assemble a ``Glm5NextConfig`` from ``glm5-next.*`` metadata."""
    Glm5NextConfig, Glm5NextTextConfig = _config_classes()
    fields = text_config_fields_from_gguf(gguf_path)
    text_config = _construct(Glm5NextTextConfig, fields)

    logger.info(
        "GLM-5.3-Flash GGUF: %d layers (%d KDA, %d sparse MLA), %d experts "
        "top-%d, first %d dense, indexer %dx%d top-%d pool %d, hc %d, nextn %d",
        fields["num_hidden_layers"],
        fields["layer_types"].count("linear_attention"),
        fields["layer_types"].count("deepseek_sparse_attention"),
        fields["n_routed_experts"],
        fields["num_experts_per_tok"],
        fields["first_k_dense_replace"],
        fields["index_n_heads"],
        fields["index_head_dim"],
        fields["index_topk"],
        fields["index_kpool"],
        fields["hc_mult"],
        fields["num_nextn_predict_layers"],
    )

    # A GGUF has no `architectures`; without it ModelConfig rejects the model
    # outright. The multimodal wrapper is the registered class; with
    # language_model_only the tower is a StageMissingLayer and the default
    # vision sub-config is never consulted.
    cfg = Glm5NextConfig(
        text_config=text_config,
        architectures=["Glm5NextForConditionalGeneration"],
        tie_word_embeddings=False,
    )
    cfg.architectures = ["Glm5NextForConditionalGeneration"]
    return cfg


def build_glm5_next_tokenizer_from_gguf(gguf_path: str):
    """GLM-5.3-Flash's byte-level BPE from ``tokenizer.ggml.*``.

    ``tokenizer.ggml.model`` is ``gpt2`` with the ``glm4`` pre-tokenizer, the
    same split GLM-5.2 uses. Unlike the GLM-5.2 builder this keeps the GGUF's
    own ``tokenizer.chat_template``: it is the GLM-5.3 template (``[gMASK]<sop>``
    opener, reasoning-effort system line, ``<think>`` blocks), and the model is
    served text-only, so nothing is gained by swapping in the vision one.
    """
    return build_bpe_tokenizer(
        str(Path(gguf_path)),
        regexes=(GLM4_PRETOKENIZER_REGEX,),
        chat_template_path=None,
    )
