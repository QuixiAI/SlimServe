# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the GLM-5.2-Vision config and tokenizer from the GGUF pair alone.

llama.cpp serves this model from `<backbone>.gguf` + `mmproj-*.gguf` and
nothing else, because GGUF carries the hyperparameters, the tokenizer and (via
mmproj) the vision tower and its preprocessing constants. vLLM could not,
for one reason: `transformers`' GGUF reader has an architecture whitelist and
`glm-dsa` is not on it, so `get_config` and `get_tokenizer` both die with
"GGUF model with architecture glm-dsa is not supported yet" and everything
downstream cascades from that.

So do not route through transformers' GGUF reader. Read the metadata directly.

A handful of GLM-5.2 architectural constants have no GGUF key (the indexer
layer pattern, rope interleave flags). GGUF simply does not model them. This
repo serves exactly one model, so they are written here as named constants
with the reason attached, rather than being guessed at load time.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import gguf

from vllm.logger import init_logger

logger = init_logger(__name__)

# --- constants GGUF has no key for -------------------------------------------
# GLM-5.2 alternates DSA indexer types: the first `first_k_dense_replace`
# layers are "full", then every `INDEX_TOPK_FREQ`-th layer is "full" and the
# rest "shared". Confirmed against the reference config's `indexer_types`.
INDEX_TOPK_FREQ = 4
INDEX_SKIP_TOPK_OFFSET = 3
INDEXER_ROPE_INTERLEAVE = True
ROPE_INTERLEAVE = True
SCORING_FUNC = "sigmoid"
TOPK_METHOD = "noaux_tc"
# GLM `<|image|>`.
MEDIA_PLACEHOLDER_TOKEN_ID = 154854

# Video/limit knobs the mmproj has no key for; reference values.
PATCH_LIMIT_ON_ONE_SIDE = 512
SAMPLE_FPS = 2.0
TEMPORAL_MERGE_KERNEL_SIZE = 4
TIMESTAMP_MODE = "hh:mm:ss.fff"

# llama.cpp's LLAMA_VOCAB_PRE_TYPE_CHATGLM4 split, which `tokenizer.ggml.pre`
# names as "glm4".
GLM4_PRETOKENIZER_REGEX = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _field(reader: gguf.GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return default
    return field.contents()


def find_mmproj(gguf_path: str | Path) -> Path | None:
    """Locate the vision projector beside (or one level above) the backbone."""
    p = Path(gguf_path)
    for directory in (p.parent, p.parent.parent):
        for pattern in ("mmproj*.gguf", "*mmproj*.gguf"):
            hits = sorted(directory.glob(pattern))
            if hits:
                return hits[0]
    return None


def _indexer_types(num_layers: int, first_dense: int) -> list[str]:
    """Reference pattern: full at 0,1,2 then every layer == 2 (mod 4).

    i.e. the leading dense block is all "full", and thereafter one "full" per
    INDEX_TOPK_FREQ layers, phased so it lands on the last leading-dense index.
    Validated element-wise against the reference config's `indexer_types`.
    """
    phase = (first_dense - 1) % INDEX_TOPK_FREQ
    return [
        "full" if i < first_dense or i % INDEX_TOPK_FREQ == phase else "shared"
        for i in range(num_layers)
    ]


@cache
def build_config_from_gguf(gguf_path: str) -> Any:
    """Assemble a `Glm5vConfig` from backbone + mmproj metadata."""
    from vllm.transformers_utils.configs.glm5v import Glm5vConfig

    r = gguf.GGUFReader(str(gguf_path))
    g = lambda k, d=None: _field(r, f"glm-dsa.{k}", d)  # noqa: E731

    # GGUF stores `block_count` including the MTP/nextn layer; the runtime
    # model wants only the transformer layers.
    n_nextn = int(g("nextn_predict_layers", 0) or 0)
    num_layers = int(g("block_count")) - n_nextn
    first_dense = int(g("leading_dense_block_count"))
    kv_lora = int(g("attention.kv_lora_rank"))
    rope_dim = int(g("rope.dimension_count"))
    # key_length_mla == qk_nope + qk_rope; value_length_mla == v_head_dim.
    qk_nope = int(g("attention.key_length_mla")) - rope_dim
    v_head_dim = int(g("attention.value_length_mla"))

    text_config = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": int(g("embedding_length")),
        "intermediate_size": int(g("feed_forward_length")),
        "moe_intermediate_size": int(g("expert_feed_forward_length")),
        "num_hidden_layers": num_layers,
        "num_attention_heads": int(g("attention.head_count")),
        "num_key_value_heads": int(g("attention.head_count")),
        "vocab_size": int(g("vocab_size")),
        "max_position_embeddings": int(g("context_length")),
        # GGUF stores this as fp32; round so it matches the reference exactly.
        "rms_norm_eps": round(float(g("attention.layer_norm_rms_epsilon")), 12),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(g("rope.freq_base")),
        },
        "q_lora_rank": int(g("attention.q_lora_rank")),
        "kv_lora_rank": kv_lora,
        "qk_rope_head_dim": rope_dim,
        "qk_nope_head_dim": qk_nope,
        "qk_head_dim": qk_nope + rope_dim,
        "head_dim": qk_nope,
        "v_head_dim": v_head_dim,
        "n_routed_experts": int(g("expert_count")),
        "num_experts_per_tok": int(g("expert_used_count")),
        "n_shared_experts": int(g("expert_shared_count")),
        "n_group": int(g("expert_group_count")),
        "topk_group": int(g("expert_group_used_count")),
        "routed_scaling_factor": float(g("expert_weights_scale")),
        "norm_topk_prob": bool(g("expert_weights_norm")),
        "first_k_dense_replace": first_dense,
        "num_nextn_predict_layers": n_nextn,
        "index_n_heads": int(g("attention.indexer.head_count")),
        "index_head_dim": int(g("attention.indexer.key_length")),
        "index_topk": int(g("attention.indexer.top_k")),
        "index_topk_freq": INDEX_TOPK_FREQ,
        "index_skip_topk_offset": INDEX_SKIP_TOPK_OFFSET,
        "index_topk_pattern": None,
        "index_share_for_mtp_iteration": True,
        "indexer_types": _indexer_types(num_layers, first_dense),
        "indexer_rope_interleave": INDEXER_ROPE_INTERLEAVE,
        "rope_interleave": ROPE_INTERLEAVE,
        "scoring_func": SCORING_FUNC,
        "topk_method": TOPK_METHOD,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "ep_size": 1,
        "moe_layer_freq": 1,
        "pretraining_tp": 1,
        # Three stop tokens, not one: eos, eot and eom. With only `eos` the
        # model runs past <|user|>/<|observation|> turn ends.
        "eos_token_id": [
            int(_field(r, f"tokenizer.ggml.{k}_token_id"))
            for k in ("eos", "eot", "eom")
            if _field(r, f"tokenizer.ggml.{k}_token_id") is not None
        ],
        # GGUF declares 154821 here but the reference checkpoint pads with the
        # eos id; padding is unused by vLLM's ragged batching either way.
        "pad_token_id": int(_field(r, "tokenizer.ggml.eos_token_id")),
    }

    vision_config: dict[str, Any] = {}
    mmproj = find_mmproj(gguf_path)
    if mmproj is not None:
        m = gguf.GGUFReader(str(mmproj))
        c = lambda k, d=None: _field(m, f"clip.vision.{k}", d)  # noqa: E731
        hidden = int(c("embedding_length"))
        layers = int(c("block_count"))
        heads = int(c("attention.head_count"))
        inter = int(c("feed_forward_length"))
        scale = int(c("projector.scale_factor", 2))
        vision_config = {
            "patch_size": int(c("patch_size")),
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "intermediate_size": inter,
            # The projector reads the vt_* family; the tower reads the plain
            # names. Both are the same numbers.
            "vt_hidden_size": hidden,
            "vt_num_hidden_layers": layers,
            "vt_num_attention_heads": heads,
            "vt_intermediate_size": inter,
            "mm_hidden_size": hidden,
            "merge_kernel_size": [scale, scale],
            "projector_ln_eps": float(c("attention.layer_norm_epsilon", 1e-5)),
            # Projector output width is the text hidden size.
            "text_hidden_size": text_config["hidden_size"],
            "image_mean": list(c("image_mean", [0.5, 0.5, 0.5])),
            "image_std": list(c("image_std", [0.5, 0.5, 0.5])),
            "image_min_pixels": int(c("image_min_pixels", 1568)),
            "image_max_pixels": int(c("image_max_pixels", 3211264)),
        }
        logger.info("Built vision config from %s", mmproj.name)
    else:
        logger.warning(
            "No mmproj GGUF found beside %s; the model will load text-only.",
            gguf_path,
        )

    cfg = Glm5vConfig(
        text_config=text_config,
        vision_config=vision_config or None,
        media_placeholder_token_id=MEDIA_PLACEHOLDER_TOKEN_ID,
        tie_word_embeddings=False,
    )
    cfg.architectures = ["Glm5vForConditionalGeneration"]
    return cfg


def build_media_proc_cfg_from_gguf(gguf_path: str) -> dict[str, Any]:
    """The image-processor config, from mmproj metadata.

    Everything spatial is in the GGUF: `in_patch_limit` is
    `image_max_pixels / patch_size**2` (3211264 / 196 = 16384, matching the
    reference preprocessor_config.json exactly). The video-side knobs have no
    GGUF key, so they are pinned here to the reference values.
    """
    mmproj = find_mmproj(gguf_path)
    if mmproj is None:
        raise RuntimeError(f"no mmproj GGUF beside {gguf_path}")
    m = gguf.GGUFReader(str(mmproj))
    c = lambda k, d=None: _field(m, f"clip.vision.{k}", d)  # noqa: E731

    patch = int(c("patch_size"))
    max_pixels = int(c("image_max_pixels", 3211264))
    return {
        "in_patch_limit": max_pixels // (patch * patch),
        "patch_size": patch,
        "image_mean": [float(x) for x in c("image_mean", [0.5, 0.5, 0.5])],
        "image_std": [float(x) for x in c("image_std", [0.5, 0.5, 0.5])],
        "merge_kernel_size": int(c("projector.scale_factor", 2)),
        "fixed_output_tokens": None,
        "patch_limit_on_one_side": PATCH_LIMIT_ON_ONE_SIDE,
        "in_patch_limit_each_frame": max_pixels // (patch * patch),
        "in_patch_limit_video": None,
        "sample_fps": SAMPLE_FPS,
        "max_num_frames_each_video": None,
        "temporal_merge_kernel_size": TEMPORAL_MERGE_KERNEL_SIZE,
        "timestamp_mode": TIMESTAMP_MODE,
        "config_type": "media_proc.processors.moonvit.MoonViTMediaProcessorConfig",
    }


def build_tokenizer_from_gguf(gguf_path: str):
    """Build a fast BPE tokenizer from `tokenizer.ggml.*`.

    `tokenizer.ggml.model` is `gpt2` with a `glm4` pre-tokenizer, i.e. plain
    byte-level BPE, so the vocab and merge list are enough to reconstruct it
    without transformers' GGUF reader.
    """
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        pre_tokenizers,
        processors,
    )
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    r = gguf.GGUFReader(str(gguf_path))
    tokens = _field(r, "tokenizer.ggml.tokens")
    merges = _field(r, "tokenizer.ggml.merges")
    if tokens is None or merges is None:
        raise ValueError(f"{gguf_path} carries no tokenizer metadata")

    vocab = {str(t): i for i, t in enumerate(tokens)}
    merge_pairs = []
    for m in merges:
        parts = str(m).split(" ")
        if len(parts) == 2:
            merge_pairs.append((parts[0], parts[1]))

    bpe = BPE(vocab=vocab, merges=merge_pairs, fuse_unk=False)
    tk = Tokenizer(bpe)
    # `tokenizer.ggml.pre` is "glm4", not plain byte-level: it is the GPT-4
    # style split. Plain ByteLevel mis-groups digits ("7742" -> one token
    # instead of "77"+"42") and runs of spaces, which silently shifts every
    # id after the first number in a prompt. `\p{N}{1,3}` is the part that
    # matters most here.
    tk.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(GLM4_PRETOKENIZER_REGEX),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.ByteLevel(trim_offsets=False)

    # Register CONTROL (3) and USER_DEFINED (4) tokens as special, otherwise
    # BPE splits them: `<|image|>` came out as 5 pieces instead of id 154854,
    # which silently corrupts every multimodal prompt.
    token_types = _field(r, "tokenizer.ggml.token_type") or []
    specials = [
        AddedToken(str(tokens[i]), special=True, normalized=False)
        for i, tt in enumerate(token_types)
        if int(tt) in (3, 4)
    ]
    if specials:
        tk.add_special_tokens(specials)

    def _tok(key):
        i = _field(r, f"tokenizer.ggml.{key}")
        return str(tokens[int(i)]) if i is not None else None

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token=_tok("bos_token_id"),
        eos_token=_tok("eos_token_id"),
        pad_token=_tok("padding_token_id"),
        unk_token=_tok("unknown_token_id"),
    )
    # The GGUF's own `tokenizer.chat_template` is the TEXT-ONLY one: given
    # image content it emits "You are unable to process this image" instead of
    # `<|begin_of_image|><|image|><|end_of_image|>`, so the placeholder never
    # appears and multimodal prompt replacement fails outright. The
    # vision-capable template is a static asset of this model, vendored beside
    # this file.
    template_path = Path(__file__).parent / "chat_templates" / "glm5v.jinja"
    if template_path.is_file():
        fast.chat_template = template_path.read_text()
    else:
        chat_template = _field(r, "tokenizer.chat_template")
        if chat_template:
            fast.chat_template = str(chat_template)
    return fast
