# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build Muse-Glimmer configs from the published GGUF set.

Same reasoning as `gguf_deepseek4`: transformers' GGUF reader has an
architecture whitelist and `muse-glimmer` is not on it, so the whole config is
assembled from GGUF metadata here. The text backbone reads `muse-glimmer.*`
keys, the vision tower reads the mmproj's `clip.vision.*` keys, and the
DFlash drafter reads `dflash.*` keys from its own file.

Facts the metadata cannot express come from the model card of
meta-models/Muse-Glimmer-30B-GGUF: SwiGLU FFN, sigmoid-gated attention
output, sandwich norms, and RoPE on local (sliding) layers only.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_native import _field, find_mmproj
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)

ARCH = "muse-glimmer"


def build_muse_glimmer_tokenizer_from_gguf(gguf_path: str):
    """The 202k-token byte-level BPE with the llama4 (GPT-4o) split.

    The GGUF's own chat template is used as shipped; unlike GLM-5.2 it
    renders image parts natively.
    """
    from vllm.transformers_utils.gguf_native import (
        LLAMA4_PRETOKENIZER_REGEX,
        build_bpe_tokenizer,
    )

    return build_bpe_tokenizer(
        gguf_path,
        regexes=(LLAMA4_PRETOKENIZER_REGEX,),
        chat_template_path=None,
    )


def _stop_token_ids(reader: Any) -> list[int]:
    ids = {int(_field(reader, "tokenizer.ggml.eos_token_id"))}
    eot = _field(reader, "tokenizer.ggml.eot_token_id")
    if eot is not None:
        ids.add(int(eot))
    return sorted(ids)


def _vision_config_from_mmproj(gguf_path: str, text_hidden: int) -> dict | None:
    mmproj = find_mmproj(gguf_path)
    if mmproj is None:
        logger.warning(
            "No mmproj GGUF found beside %s; Muse-Glimmer loads text-only.",
            gguf_path,
        )
        return None
    m = gguf_reader(str(mmproj))
    projector = str(_field(m, "clip.projector_type", ""))
    if projector != ARCH:
        raise ValueError(
            f"mmproj {mmproj.name} has projector_type {projector!r}, expected {ARCH!r}"
        )
    c = lambda k, d=None: _field(m, f"clip.vision.{k}", d)  # noqa: E731
    # The published tower carries 1024 learned positions (a 32x32 grid);
    # 896px/14 needs 64x64, so the runtime interpolates. Read the actual
    # count from the tensor so a different export stays consistent.
    num_positions = 1024
    for tensor in m.tensors:
        if tensor.name == "v.position_embd.weight":
            num_positions = int(tensor.shape[-1])
            break
    if int(_field(m, "clip.vision.projection_dim")) != text_hidden:
        raise ValueError(
            "mmproj projection_dim "
            f"{int(_field(m, 'clip.vision.projection_dim'))} does not match "
            f"the text hidden size {text_hidden}"
        )
    return {
        "hidden_size": int(c("embedding_length")),
        "num_hidden_layers": int(c("block_count")),
        "num_attention_heads": int(c("attention.head_count")),
        "intermediate_size": int(c("feed_forward_length")),
        "image_size": int(c("image_size")),
        "patch_size": int(c("patch_size")),
        "num_positions": num_positions,
        "layer_norm_eps": round(float(c("attention.layer_norm_epsilon", 1e-5)), 12),
        "spatial_merge_size": int(c("spatial_merge_size", 2)),
        "image_mean": [float(v) for v in c("image_mean", [0.5, 0.5, 0.5])],
        "image_std": [float(v) for v in c("image_std", [0.5, 0.5, 0.5])],
    }


@cache
def build_muse_glimmer_config_from_gguf(gguf_path: str) -> Any:
    from vllm.transformers_utils.configs.muse_glimmer import MuseGlimmerConfig

    r = gguf_reader(str(gguf_path))
    g = lambda k, d=None: _field(r, f"{ARCH}.{k}", d)  # noqa: E731

    hidden = int(g("embedding_length"))
    pattern = [bool(v) for v in g("attention.sliding_window_pattern")]
    num_layers = int(g("block_count"))
    if len(pattern) != num_layers:
        raise ValueError(
            f"sliding_window_pattern has {len(pattern)} entries for {num_layers} layers"
        )

    vocab_size = None
    for tensor in r.tensors:
        if tensor.name == "token_embd.weight":
            vocab_size = int(tensor.shape[-1])
            break
    if vocab_size is None:
        raise ValueError("muse-glimmer GGUF has no token_embd.weight")

    cfg = MuseGlimmerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=int(g("feed_forward_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=int(g("attention.key_length")),
        hidden_act="silu",
        max_position_embeddings=int(g("context_length")),
        rms_norm_eps=round(float(g("attention.layer_norm_rms_epsilon")), 12),
        rope_theta=float(g("rope.freq_base")),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(g("rope.freq_base")),
        },
        sliding_window=int(g("attention.sliding_window")),
        sliding_window_pattern=pattern,
        final_logit_softcapping=float(g("final_logit_softcapping")),
        logit_scale=float(g("logit_scale")),
        attention_bias=False,
        tie_word_embeddings=False,
        vision_config=_vision_config_from_mmproj(gguf_path, hidden),
        bos_token_id=int(_field(r, "tokenizer.ggml.bos_token_id")),
        eos_token_id=_stop_token_ids(r),
        pad_token_id=int(_field(r, "tokenizer.ggml.padding_token_id")),
    )
    cfg.architectures = (
        ["MuseGlimmerForConditionalGeneration"]
        if cfg.vision_config is not None
        else ["MuseGlimmerForCausalLM"]
    )
    logger.info(
        "Built Muse-Glimmer config from GGUF: %d layers, %d global, vision=%s",
        num_layers,
        sum(1 for p in pattern if not p),
        cfg.vision_config is not None,
    )
    return cfg


@cache
def build_muse_glimmer_dflash_config_from_gguf(gguf_path: str) -> Any:
    """The Muse-Glimmer DFlash drafter: a plain 5-layer GQA transformer.

    No gated attention, no sandwich norms, sliding window with RoPE on every
    layer. It carries no embedding or output head; both are shared with the
    target at runtime. `target_layers` in the GGUF are 1-based.
    """
    from vllm.transformers_utils.configs.muse_glimmer import MuseGlimmerConfig

    reader = gguf_reader(str(gguf_path))
    g = lambda k, d=None: _field(reader, f"dflash.{k}", d)  # noqa: E731

    one_based = [int(v) for v in g("target_layers")]
    num_draft_layers = int(g("block_count"))
    pattern_raw = g("attention.sliding_window_pattern")
    pattern = (
        [bool(v) for v in pattern_raw]
        if pattern_raw is not None
        else [True] * num_draft_layers
    )

    cfg = MuseGlimmerConfig(
        vocab_size=202_048,
        hidden_size=int(g("embedding_length")),
        intermediate_size=int(g("feed_forward_length")),
        num_hidden_layers=num_draft_layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=int(g("attention.key_length")),
        hidden_act="silu",
        max_position_embeddings=int(g("context_length")),
        rms_norm_eps=round(float(g("attention.layer_norm_rms_epsilon")), 12),
        rope_theta=float(g("rope.freq_base")),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(g("rope.freq_base")),
        },
        sliding_window=int(g("attention.sliding_window")),
        sliding_window_pattern=pattern,
        final_logit_softcapping=None,
        logit_scale=1.0,
        tie_word_embeddings=False,
    )
    cfg.model_type = "muse_glimmer_dflash"
    # Must start with "DFlash": EAGLEConfig prepends the method name to any
    # architecture that does not already carry it.
    cfg.architectures = ["DFlashMuseGlimmerDraftModel"]
    # The drafter has neither gated attention nor sandwich norms.
    cfg.gated_attention = False
    cfg.sandwich_norms = False
    # DFlash contract, in the conventions qwen3_dflash and the generic dflash
    # proposer consume: sliding+causal layers via `layer_types`, aux target
    # layers and the mask token via `dflash_config`, and the shared-head vocab
    # via `draft_vocab_size`.
    cfg.block_size = int(g("block_size"))
    cfg.n_predict = int(g("block_size"))
    cfg.draft_vocab_size = cfg.vocab_size
    cfg.layer_types = [
        "sliding_attention" if sliding else "full_attention" for sliding in pattern
    ]
    cfg.target_layer_ids = [layer - 1 for layer in one_based]
    cfg.mask_token_id = int(_field(reader, "tokenizer.ggml.mask_token_id"))
    cfg.dflash_config = {
        "target_layer_ids": cfg.target_layer_ids,
        "mask_token_id": cfg.mask_token_id,
        "swa_window_size": cfg.sliding_window,
        "use_aux_hidden_state": True,
    }
    logger.info(
        "Built Muse-Glimmer DFlash config: %d draft layers, block %d, "
        "target layers %s, mask token %d",
        num_draft_layers,
        cfg.block_size,
        cfg.target_layer_ids,
        cfg.mask_token_id,
    )
    return cfg
