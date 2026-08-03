# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the Kimi K3 config from its GGUF.

Same reasoning as `gguf_deepseek4`: `transformers`' GGUF reader has an
architecture whitelist and `kimi-k3` is not on it, so reading the metadata
directly is the only way in.

Unlike the DeepSeek-V4 file, this one is metadata-poor -- fourteen keys, none
of which describe attention. Every dimension below is therefore recovered from
*tensor shapes*, which is both the only option and the more trustworthy one:
a shape cannot drift from the weights it describes. The derivations, all
checked against the released `config.json`:

    q_b_proj      [1536, 18432]  -> q_lora_rank 1536, and 18432 / 96 heads
                                    = 192 = qk_nope 128 + qk_rope 64
    kv_b_proj     [512, 24576]   -> kv_lora_rank 512, 24576 / 96
                                    = 256 = qk_nope 128 + v_head_dim 128
    kv_a_proj     [7168, 576]    -> 576 = kv_lora_rank 512 + qk_rope 64
    q_proj (KDA)  [7168, 12288]  -> 12288 = 96 heads x head_dim 128
    q_conv1d      [4, 1, 12288]  -> short_conv_kernel_size 4

Layer types come from which tensors a layer actually has -- `A_log` means KDA,
`kv_a_proj_with_mqa` means full MLA -- rather than from a hardcoded index list,
so a repack with a different interleave still loads correctly.

Four more have real GGUF keys upstream (unslothai/llama.cpp#48) that this
build simply omits -- the SITU betas, attn_res.block_size and the expert
latent length. Those go through `_KEYED_HPARAMS`, which reads the file first
and warns loudly when it has to fall back, so a fixed converter needs no code
change here. `_ARCH_CONSTANTS` is what is left: values with no key even
upstream and no trace in any shape.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import regex as re

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_native import _field
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)

ARCH = "kimi-k3"

# Values that a conformant Kimi-K3 GGUF carries as keys. Unsloth's llama.cpp
# fork (unslothai/llama.cpp#48) defines LLM_KV_ACTIVATION_SITU_BETA,
# LLM_KV_ACTIVATION_SITU_LINEAR_BETA, LLM_KV_ATTN_RES_BLOCK_SIZE and
# LLM_KV_EXPERT_LATENT_LENGTH, and hard-fails on the last two ("Kimi-K3
# requires attn_res.block_size" / "... expert_latent_length").
#
# The antirez build carries none of them and spells the latent length
# `routed_hidden_length`, so each entry lists the spellings to try. The PR
# diff does not include the key table itself, so the exact format strings are
# inferred from those assertion messages and llama.cpp's `%s.` convention --
# hence several candidates rather than one.
#
# A missing key falls back to the released config.json value, but *loudly*:
# the point of this project is that the GGUF is the sole authority, so an
# external value has to announce itself rather than hide. When the converter
# is fixed these reads start succeeding with no code change.
_KEYED_HPARAMS: dict[str, tuple[tuple[str, ...], Any]] = {
    "activation_situ_beta": (
        ("activation.situ_beta", "activation_situ_beta", "situ_beta"),
        4.0,
    ),
    "activation_situ_linear_beta": (
        (
            "activation.situ_linear_beta",
            "activation_situ_linear_beta",
            "situ_linear_beta",
        ),
        25.0,
    ),
    "attn_res_block_size": (
        ("attn_res.block_size", "attn_res_block_size"),
        12,
    ),
    "routed_expert_hidden_size": (
        ("expert_latent_length", "routed_hidden_length"),
        3584,
    ),
}

# Not derivable from any tensor shape, and with no key defined for them even
# upstream. Taken from the released moonshotai/Kimi-K3 `config.json`.
_ARCH_CONSTANTS = {
    "hidden_act": "situ",
    "rms_norm_eps": 1e-5,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "latent_moe_use_norm": True,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "topk_method": "noaux_tc",
    "use_grouped_topk": True,
    "num_expert_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "moe_layer_freq": 1,
    "num_nextn_predict_layers": 0,
    "tie_word_embeddings": False,
    "bos_token_id": 163584,
    "eos_token_id": 163586,
    "pad_token_id": 163839,
}

# Likewise not in the file; the KDA gate floor and the full-rank gate flag.
_LINEAR_ATTN_CONSTANTS = {"gate_lower_bound": -5.0, "use_full_rank_gate": True}

# The GGUF stores only the 163584 base ranks. The 256 ids above them are
# special, and the named ones come from the release's `added_tokens_decoder`;
# anything unnamed falls back to `<|reserved_token_N|>` inside the tokenizer.
# `<|media_pad|>` is the one the multimodal path substitutes image features
# into, so it has to keep id 163605.
_SPECIAL_TOKEN_NAMES = {
    163584: "[BOS]",
    163585: "[EOS]",
    163586: "<|end_of_msg|>",
    163587: "<|open|>",
    163588: "<|close|>",
    163589: "<|sep|>",
    163590: "[start_header_id]",
    163591: "[end_header_id]",
    163593: "[EOT]",
    163602: "<|media_begin|>",
    163603: "<|media_content|>",
    163604: "<|media_end|>",
    163605: "<|media_pad|>",
    163649: "<osagent_mode>",
    163838: "[UNK]",
    163839: "[PAD]",
}

_LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")


def is_kimi_k3_gguf(gguf_path: str) -> bool:
    return str(_field(gguf_reader(str(gguf_path)), "general.architecture")) == ARCH


def _shapes(reader) -> dict[str, list[int]]:
    return {t.name: [int(d) for d in t.shape] for t in reader.tensors}


def _layer_suffixes(shapes: dict[str, list[int]]) -> dict[int, set[str]]:
    """Map layer index -> the set of tensor suffixes that layer carries."""
    out: dict[int, set[str]] = {}
    for name in shapes:
        m = _LAYER_RE.match(name)
        if m:
            out.setdefault(int(m.group(1)), set()).add(m.group(2))
    return out


@cache
def build_kimi_k3_config_from_gguf(gguf_path: str) -> Any:
    """Assemble a `KimiK3Config` from `kimi-k3.*` metadata and tensor shapes."""
    from vllm.transformers_utils.configs.kimi_k3 import (
        KimiK3Config,
        KimiK3VisionConfig,
    )

    reader = gguf_reader(str(gguf_path))
    shapes = _shapes(reader)
    per_layer = _layer_suffixes(shapes)

    def meta(key: str, default: Any = None) -> Any:
        return _field(reader, f"{ARCH}.{key}", default)

    def keyed(field: str) -> Any:
        """Read a hparam the GGUF is supposed to carry, or say so out loud."""
        candidates, fallback = _KEYED_HPARAMS[field]
        for suffix in candidates:
            value = meta(suffix)
            if value is not None:
                return value
        logger.warning(
            "kimi-k3 GGUF: none of %s present; falling back to the released "
            "config.json value %r for %s. The file is supposed to carry this "
            "-- see unslothai/llama.cpp#48, which hard-fails without it.",
            [f"{ARCH}.{s}" for s in candidates],
            fallback,
            field,
        )
        return fallback

    hidden_size = int(meta("embedding_length"))
    num_layers = int(meta("block_count"))

    # Layer types from tensor presence, not an index list. The config's lists
    # are 1-based, matching the released config.json.
    kda_layers, full_attn_layers = [], []
    for idx in sorted(per_layer):
        suffixes = per_layer[idx]
        if any(s.startswith("self_attn.A_log") for s in suffixes):
            kda_layers.append(idx + 1)
        elif any(s.startswith("self_attn.kv_a_proj_with_mqa") for s in suffixes):
            full_attn_layers.append(idx + 1)
    if len(kda_layers) + len(full_attn_layers) != num_layers:
        raise ValueError(
            f"classified {len(kda_layers)} KDA + {len(full_attn_layers)} full "
            f"attention layers but block_count is {num_layers}"
        )

    def find(suffix: str) -> list[int]:
        for name, shape in shapes.items():
            if name.endswith(suffix):
                return shape
        raise ValueError(f"no tensor ending in {suffix!r}; not a Kimi K3 GGUF?")

    # KDA: q_proj is [hidden, num_heads * head_dim].
    num_heads = int(meta("attention.head_count", 0)) or None
    kda_head_dim = int(find("self_attn.o_norm.weight")[0])
    if num_heads is None:
        num_heads = find("self_attn.q_proj.weight")[1] // kda_head_dim

    q_lora_rank = find("self_attn.q_a_proj.weight")[1]
    kv_lora_rank = find("self_attn.kv_a_layernorm.weight")[0]
    # kv_a_proj packs the latent plus the rope half of the shared key.
    qk_rope_head_dim = find("self_attn.kv_a_proj_with_mqa.weight")[1] - kv_lora_rank
    qk_nope_head_dim = (
        find("self_attn.q_b_proj.weight")[1] // num_heads - qk_rope_head_dim
    )
    v_head_dim = find("self_attn.kv_b_proj.weight")[1] // num_heads - qk_nope_head_dim
    conv_kernel = find("self_attn.q_conv1d.weight")[0]

    # Dense-vs-MoE split: the dense prefix is the layers carrying `mlp.*`.
    dense_layers = sum(
        1 for s in per_layer.values() if any(x.startswith("mlp.") for x in s)
    )
    moe_intermediate = int(meta("expert_feed_forward_length"))
    shared_intermediate = find("shared_experts.gate_proj.weight")[1]

    text_config = {
        "model_type": "kimi_linear",
        "vocab_size": int(meta("vocabulary_size")),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "head_dim": kda_head_dim,
        "intermediate_size": find("mlp.gate_proj.weight")[1],
        "max_position_embeddings": int(meta("context_length")),
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "num_experts": int(meta("expert_count")),
        "num_experts_per_token": int(meta("expert_used_count")),
        "moe_intermediate_size": moe_intermediate,
        "num_shared_experts": shared_intermediate // moe_intermediate,
        "routed_expert_hidden_size": int(keyed("routed_expert_hidden_size")),
        "first_k_dense_replace": dense_layers,
        "linear_attn_config": {
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "num_heads": num_heads,
            "head_dim": kda_head_dim,
            "short_conv_kernel_size": conv_kernel,
            **_LINEAR_ATTN_CONSTANTS,
        },
        "activation_situ_beta": float(keyed("activation_situ_beta")),
        "activation_situ_linear_beta": float(keyed("activation_situ_linear_beta")),
        "attn_res_block_size": int(keyed("attn_res_block_size")),
        **_ARCH_CONSTANTS,
    }

    vision_config = KimiK3VisionConfig(text_hidden_size=hidden_size)

    logger.info(
        "Kimi K3 GGUF: %d layers (%d KDA, %d MLA), %d experts top-%d, "
        "%d heads, kv_lora %d, q_lora %d",
        num_layers,
        len(kda_layers),
        len(full_attn_layers),
        text_config["num_experts"],
        text_config["num_experts_per_token"],
        num_heads,
        kv_lora_rank,
        q_lora_rank,
    )

    # A GGUF has no `architectures`; without it ModelConfig rejects the model
    # outright ("No model architectures are specified") before any loader runs.
    return KimiK3Config(
        text_config=text_config,
        vision_config=vision_config,
        architectures=["KimiK3ForConditionalGeneration"],
    )


def _tiktoken_vocab_path(gguf_path: str) -> str:
    """Materialise `tokenizer.kimi-k3.tiktoken` as a tiktoken.model file.

    `tiktoken.load.load_tiktoken_bpe` reads a path, and the GGUF stores the
    identical content -- 163584 `base64 rank` lines, byte-for-byte the same as
    the `tiktoken.model` shipped in the model repo. Writing it out beside the
    HF cache keeps the vendored tokenizer unmodified.
    """
    import hashlib
    import os
    import tempfile

    reader = gguf_reader(str(gguf_path))
    field = reader.fields.get(f"tokenizer.{ARCH}.tiktoken")
    if field is None:
        raise ValueError(
            f"no tokenizer.{ARCH}.tiktoken in {gguf_path}; cannot build a vocab"
        )
    blob = bytes(field.parts[-1])

    digest = hashlib.sha256(blob).hexdigest()[:16]
    out = os.path.join(tempfile.gettempdir(), f"kimi-k3-{digest}.tiktoken.model")
    if not os.path.exists(out):
        tmp = f"{out}.{os.getpid()}"
        with open(tmp, "wb") as handle:
            handle.write(blob)
        os.replace(tmp, out)
    return out


@cache
def build_kimi_k3_tokenizer_from_gguf(gguf_path: str):
    """Kimi K3's own tiktoken tokenizer, fed from the GGUF's vocabulary.

    The 256 special ids above the 163584 base ranks are not in the file; the
    named ones below come from the release's `added_tokens_decoder` and the
    rest fall back to `<|reserved_token_N|>`, matching the reference.
    """
    from tokenizers import AddedToken

    from vllm.transformers_utils.tokenizers.kimi_k3 import TikTokenTokenizer

    added = {
        i: AddedToken(name, special=True) for i, name in _SPECIAL_TOKEN_NAMES.items()
    }
    tokenizer = TikTokenTokenizer(
        vocab_file=_tiktoken_vocab_path(gguf_path),
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        # Not optional despite the signature: __init__ looks unk_token up in
        # special_tokens unconditionally, so None becomes KeyError('None').
        unk_token="[UNK]",
        added_tokens_decoder=added,
    )
    # vLLM resolves a template before calling the tokenizer. K3 renders XTML
    # in its apply_chat_template override, so provide a non-empty sentinel;
    # the override intentionally ignores the resolved Jinja string.
    tokenizer.chat_template = "{# Rendered by TikTokenTokenizer.apply_chat_template. #}"
    logger.info(
        "Kimi K3 tokenizer: %d tokens (%d base + %d special)",
        tokenizer.vocab_size,
        tokenizer.vocab_size - TikTokenTokenizer.num_reserved_special_tokens,
        TikTokenTokenizer.num_reserved_special_tokens,
    )
    return tokenizer
