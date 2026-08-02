# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the DeepSeek-V4-Flash config and tokenizer from its GGUF alone.

Same reasoning as `gguf_native` does for GLM-5.2: `transformers`' GGUF reader
has an architecture whitelist and `deepseek4` is not on it, so `get_config` and
`get_tokenizer` both die before anything downstream runs. Read the metadata
directly instead.

Every value below is read from the file. Where a dimension could plausibly be
read two ways, it was checked against the tensor shapes rather than inferred --
`attn_q_b` is [1024, 32768] = q_lora_rank x (64 heads * 512), which is what
pins `head_dim` to 512, and `attn_output_a` is [4096, 8192] = (64*512/8) x
(8*1024), which pins `o_groups` to 8 and `o_lora_rank` to 1024.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_native import (
    _field,
    build_bpe_tokenizer,
    stop_token_ids_from_gguf,
)
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)

ARCH = "deepseek4"

# llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_*; the release stores 4.
_GATING_FUNCS = {1: "softmax", 2: "sigmoid", 4: "sqrtsoftplus"}

# llama.cpp's LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM split (`llama-vocab.cpp`), which
# `tokenizer.ggml.pre` names as "joyai-llm". Applied in order, before
# byte-level. Plain ByteLevel would group digits and CJK runs wrongly, which
# silently shifts every id after the first number in a prompt.
JOYAI_LLM_PRETOKENIZER_REGEXES = (
    r"\p{N}{1,3}",
    r"[\x{4E00}-\x{9FA5}\x{3040}-\x{309F}\x{30A0}-\x{30FF}]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"
    r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
)


def is_deepseek4_gguf(gguf_path: str) -> bool:
    return str(_field(gguf_reader(str(gguf_path)), "general.architecture")) == ARCH


@cache
def build_deepseek4_config_from_gguf(gguf_path: str) -> Any:
    """Assemble a `DeepseekV4Config` from `deepseek4.*` metadata."""
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    r = gguf_reader(str(gguf_path))
    g = lambda k, d=None: _field(r, f"{ARCH}.{k}", d)  # noqa: E731

    # `attention.key_length` is the MLA latent width, not a per-head qk size:
    # attn_q_b is [q_lora_rank, num_heads * this].
    head_dim = int(g("attention.key_length"))

    # Stored per layer and uniform in this release; the model wants a scalar.
    swiglu = g("swiglu_clamp_exp")
    swiglu_limit = float(swiglu[0]) if isinstance(swiglu, list) else float(swiglu)

    # `nextn_predict_layers` is 1, but the file carries no MTP tensors at all --
    # layer 42 is structurally identical to layer 20, with no enorm/hnorm/eh_proj
    # anywhere. The nextn head ships as a separate DSpark drafter GGUF, so all 43
    # blocks here are ordinary transformer layers and enabling nextn would send
    # the loader looking for weights that do not exist.
    num_layers = int(g("block_count"))

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
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=int(g("embedding_length")),
        num_hidden_layers=num_layers,
        vocab_size=int(g("vocab_size")),
        num_attention_heads=int(g("attention.head_count")),
        head_dim=head_dim,
        qk_rope_head_dim=int(g("rope.dimension_count")),
        q_lora_rank=int(g("attention.q_lora_rank")),
        o_lora_rank=int(g("attention.output_lora_rank")),
        o_groups=int(g("attention.output_group_count")),
        sliding_window=int(g("attention.sliding_window")),
        compress_ratios=[int(x) for x in g("attention.compress_ratios")],
        compress_rope_theta=float(g("attention.compress_rope_freq_base")),
        n_routed_experts=int(g("expert_count")),
        n_shared_experts=int(g("expert_shared_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        norm_topk_prob=bool(g("expert_weights_norm")),
        routed_scaling_factor=float(g("expert_weights_scale")),
        scoring_func=_GATING_FUNCS.get(int(g("expert_gating_func", 4)), "sqrtsoftplus"),
        # GGUF has no key for this, but it is not a guess: the model allocates
        # `gate.e_score_correction_bias` only under "noaux_tc", and the file
        # ships `exp_probs_b.bias` for every non-hash layer. Leaving it unset
        # means those 40 tensors have no parameter to load into.
        topk_method="noaux_tc",
        num_hash_layers=int(g("hash_layer_count")),
        hc_mult=int(g("hyper_connection.count")),
        hc_eps=float(g("hyper_connection.epsilon")),
        hc_sinkhorn_iters=int(g("hyper_connection.sinkhorn_iterations")),
        index_n_heads=int(g("attention.indexer.head_count")),
        index_head_dim=int(g("attention.indexer.key_length")),
        index_topk=int(g("attention.indexer.top_k")),
        swiglu_limit=swiglu_limit,
        # GGUF stores this as fp32; round so it reads as the intended 1e-6.
        rms_norm_eps=round(float(g("attention.layer_norm_rms_epsilon")), 12),
        rope_theta=float(g("rope.freq_base")),
        rope_parameters=rope_parameters,
        max_position_embeddings=int(g("context_length")),
        num_nextn_predict_layers=0,
        hidden_act="silu",
        tie_word_embeddings=False,
        eos_token_id=stop_token_ids_from_gguf(r),
        pad_token_id=int(_field(r, "tokenizer.ggml.padding_token_id")),
    )
    cfg.architectures = ["DeepseekV4ForCausalLM"]
    logger.info(
        "Built DeepseekV4Config from GGUF: %d layers, %d experts (%d routed), "
        "head_dim %d, index_topk %d",
        cfg.num_hidden_layers,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        cfg.head_dim,
        cfg.index_topk,
    )
    return cfg


def build_deepseek4_tokenizer_from_gguf(gguf_path: str):
    """Fast BPE tokenizer from `tokenizer.ggml.*`.

    `tokenizer.ggml.model` is `gpt2` with a `joyai-llm` pre-tokenizer, so the
    vocab and merges reconstruct it.

    No chat template is installed. DeepSeek-V4 does not render one: the caller
    wraps this in `get_deepseek_v4_tokenizer`, whose `apply_chat_template` goes
    through `deepseek_v4_encoding.encode_messages` instead. Leaving the GGUF's
    own template in place would be worse than useless -- it binds `messages`
    itself, so rendering it raises "got multiple values for keyword argument".
    """
    return build_bpe_tokenizer(
        gguf_path,
        regexes=JOYAI_LLM_PRETOKENIZER_REGEXES,
        chat_template_path=None,
    )
