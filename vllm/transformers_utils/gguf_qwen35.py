# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build Qwen3.5 (Qwen3.8-27B) configs and tokenizers from the GGUF.

Same reasoning as `gguf_muse_glimmer`: the config is assembled from GGUF
metadata alone, validated field-for-field against the reference checkpoint
(Qwen3_5ForConditionalGeneration, 64 text layers as 48 Gated DeltaNet + 16
gated full attention). Text-only serving: the vision tower ships as a
separate mmproj GGUF that is never loaded, so the built config is the plain
text config with architectures=[Qwen3_5ForCausalLM].

Facts the metadata cannot express, from the reference config.json:
- attn_output_gate (sigmoid output gate fused per-head into q_proj) is
  always on for this family; the model code defaults it to True.
- The recurrent (ssm) state is fp32 (`mamba_ssm_dtype`), matching the HF
  checkpoint and both reference implementations; bf16 state accumulation is
  a known shipped-bug failure mode in this family.
- IMROPE sections [11, 11, 10, 0] degenerate to standard partial NeoX RoPE
  for text-only inputs (every position component equal), so rope_parameters
  stays `default` with partial_rotary_factor = rope_dims / head_dim.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_native import _field, build_bpe_tokenizer
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)

ARCH = "qwen35"

# llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 split (`tokenizer.ggml.pre` =
# "qwen35"): the qwen2 split with combining marks (\p{M}) treated as letter
# characters. Canonical tokenizer.json form.
QWEN35_PRETOKENIZER_REGEX = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def build_qwen35_tokenizer_from_gguf(gguf_path: str):
    """The 248k-token byte-level BPE with the qwen35 pre-tokenizer split."""
    return build_bpe_tokenizer(
        gguf_path,
        regexes=(QWEN35_PRETOKENIZER_REGEX,),
        chat_template_path=None,
    )


def build_qwen35_config_from_gguf(gguf_path: str):
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig

    reader = gguf_reader(gguf_path)
    if str(_field(reader, "general.architecture")) != ARCH:
        raise ValueError(f"{gguf_path} is not a {ARCH} GGUF")

    def field(key: str, default=None):
        value = _field(reader, f"{ARCH}.{key}", default)
        if value is None:
            raise ValueError(f"{gguf_path} missing metadata key {ARCH}.{key}")
        return value

    num_layers = int(field("block_count"))
    hidden_size = int(field("embedding_length"))
    head_dim = int(field("attention.key_length"))
    rope_dims = int(field("rope.dimension_count"))
    full_attn_interval = int(field("full_attention_interval"))

    # GDN geometry. llama.cpp's metadata reuses mamba key names:
    # group_count = K heads, time_step_rank = V heads, state_size = K head
    # dim, inner_size = V heads * V head dim.
    linear_num_key_heads = int(field("ssm.group_count"))
    linear_num_value_heads = int(field("ssm.time_step_rank"))
    linear_key_head_dim = int(field("ssm.state_size"))
    inner_size = int(field("ssm.inner_size"))
    if inner_size % linear_num_value_heads != 0:
        raise ValueError(
            f"ssm.inner_size {inner_size} not divisible by "
            f"ssm.time_step_rank {linear_num_value_heads}"
        )
    linear_value_head_dim = inner_size // linear_num_value_heads

    # Vocab is not in the metadata; read it from the embedding table.
    vocab_size = None
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            vocab_size = int(tensor.shape[1])
            break
    if vocab_size is None:
        raise ValueError(f"{gguf_path} has no token_embd.weight")

    layer_types = [
        "full_attention" if (i + 1) % full_attn_interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]

    config = Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=int(field("feed_forward_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(field("attention.head_count")),
        num_key_value_heads=int(field("attention.head_count_kv")),
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=int(field("context_length")),
        rms_norm_eps=float(field("attention.layer_norm_rms_epsilon")),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(field("rope.freq_base")),
            "partial_rotary_factor": rope_dims / head_dim,
        },
        linear_conv_kernel_dim=int(field("ssm.conv_kernel")),
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        layer_types=layer_types,
        tie_word_embeddings=False,
        bos_token_id=int(_field(reader, "tokenizer.ggml.bos_token_id")),
        eos_token_id=int(_field(reader, "tokenizer.ggml.eos_token_id")),
        pad_token_id=int(_field(reader, "tokenizer.ggml.padding_token_id")),
        dtype="bfloat16",
    )
    config.full_attention_interval = full_attn_interval
    # fp32 recurrent state; picked up by Qwen3_5ForConditionalGenerationConfig
    # when --mamba-ssm-cache-dtype is auto.
    config.mamba_ssm_dtype = "float32"
    logger.info(
        "qwen35 GGUF config: %d layers (%d linear / %d full), hidden %d, "
        "vocab %d, GDN %dx%d K / %dx%d V",
        num_layers,
        sum(t == "linear_attention" for t in layer_types),
        sum(t == "full_attention" for t in layer_types),
        hidden_size,
        vocab_size,
        linear_num_key_heads,
        linear_key_head_dim,
        linear_num_value_heads,
        linear_value_head_dim,
    )
    return config
