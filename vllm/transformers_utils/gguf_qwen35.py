# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build Qwen3.5-family configs from GGUF metadata.

Covers the two artifacts of the Qwen3.8-27B campaign: the target model
(unsloth GGUF, `general.architecture == "qwen35"`, a 3:1 hybrid of
gated-deltanet linear-attention layers and full-attention layers) and the
DFlash 2 drafter (z-lab GGUF, `general.architecture == "dflash"` carrying
`dflash.selector_rank`). Key names follow llama.cpp's qwen35 loader
(src/models/qwen35.cpp) and the DFlash 2 PR (#27342); both were verified
against the downloaded files -- see perf/qwen38_metal_design.md.

Facts the metadata cannot express come from the published HF config of
Qwen/Qwen3.8-27B: the attention output gate on full-attention layers,
interleaved MRoPE, and fp32 recurrent-state dtype.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.gguf_native import _field
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)

ARCH = "qwen35"


def build_qwen35_tokenizer_from_gguf(gguf_path: str):
    """The 248k-token byte-level BPE with the qwen35 split.

    Used by both the qwen35 target and the DFlash 2 drafter (whose GGUF
    carries the full tokenizer and chat template despite sharing weights
    with the target at runtime). The GGUF's own chat template is used as
    shipped; it renders image/video parts natively.
    """
    from vllm.transformers_utils.gguf_native import (
        QWEN35_PRETOKENIZER_REGEX,
        build_bpe_tokenizer,
    )

    return build_bpe_tokenizer(
        gguf_path,
        regexes=(QWEN35_PRETOKENIZER_REGEX,),
        chat_template_path=None,
    )


def _vocab_size(reader: Any) -> int:
    tokens = _field(reader, "tokenizer.ggml.tokens")
    if tokens is not None:
        return len(tokens)
    return int(_field(reader, f"{ARCH}.vocab_size"))


def _vision_config_from_mmproj(gguf_path: str) -> Any | None:
    """Build the Qwen3.5 vision config from the mmproj beside the target.

    Keys verified against the real dump (perf/qwen38_metal_design.md,
    "Vision: mmproj-F16.gguf"): 27 blocks, hidden 1152, 16 heads, ffn 4304
    (gelu, no gate), patch 16, temporal 2, spatial merge 2, LayerNorm eps
    1e-6, position table 2304 (48x48 at image_size 768), projection 5120,
    image mean/std 0.5, `is_deepstack_layers` all-false.
    """
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5VisionConfig
    from vllm.transformers_utils.gguf_native import find_mmproj

    mmproj = find_mmproj(gguf_path)
    if mmproj is None:
        logger.warning(
            "No mmproj GGUF found beside %s; Qwen3.5 loads text-only.",
            gguf_path,
        )
        return None
    m = gguf_reader(str(mmproj))
    projector = str(_field(m, "clip.projector_type", ""))
    if projector != "qwen3vl_merger":
        raise ValueError(
            f"mmproj {mmproj.name} has projector_type {projector!r}, "
            "expected 'qwen3vl_merger'"
        )
    c = lambda k, d=None: _field(m, f"clip.vision.{k}", d)  # noqa: E731
    deepstack = c("is_deepstack_layers")
    if deepstack is not None and any(bool(v) for v in deepstack):
        raise ValueError(
            f"mmproj {mmproj.name} carries deepstack layers; the vendored "
            "tower has no deepstack path"
        )
    # Read the actual position count from the tensor so a different export
    # stays consistent (the muse builder does the same).
    num_positions = 2304
    for tensor in m.tensors:
        if tensor.name == "v.position_embd.weight":
            num_positions = int(tensor.shape[-1])
            break
    image_size = int(c("image_size", 768))
    return Qwen3_5VisionConfig(
        depth=int(c("block_count")),
        hidden_size=int(c("embedding_length")),
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=int(c("feed_forward_length")),
        num_heads=int(c("attention.head_count")),
        in_channels=3,
        patch_size=int(c("patch_size")),
        spatial_merge_size=int(c("spatial_merge_size", 2)),
        temporal_patch_size=int(c("temporal_patch_size", 2)),
        out_hidden_size=int(c("projection_dim")),
        num_position_embeddings=num_positions,
        # Extra facts the class has no named params for; PretrainedConfig
        # keeps unknown kwargs as attributes.
        layer_norm_eps=round(float(c("attention.layer_norm_epsilon", 1e-6)), 12),
        image_size=image_size,
        image_mean=[float(v) for v in c("image_mean", [0.5, 0.5, 0.5])],
        image_std=[float(v) for v in c("image_std", [0.5, 0.5, 0.5])],
    )


@cache
def build_qwen35_config_from_gguf(gguf_path: str) -> Any:
    """The Qwen3.8-27B text backbone: 64 layers, 3:1 hybrid interleave.

    48 gated-deltanet linear-attention layers (16 K-heads / 48 V-heads at
    128 dim, short conv kernel 4) and 16 full-attention layers (GQA 24/4,
    head_dim 256, gated attention output, interleaved MRoPE over the first
    quarter of the head). The checkpoint appends `nextn_predict_layers`
    MTP block(s) beyond the main stack; those are excluded from
    `num_hidden_layers` and never loaded -- DFlash 2 replaces MTP here.

    Vision: when an mmproj (`qwen3vl_merger`) sits beside the text GGUF the
    builder emits the composite Qwen3_5Config (text_config + vision_config,
    architectures ["Qwen3_5ForConditionalGeneration"]); without one it
    emits the bare text config exactly as before.
    """
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig

    reader = gguf_reader(str(gguf_path))
    g = lambda k, d=None: _field(reader, f"{ARCH}.{k}", d)  # noqa: E731

    n_layer_all = int(g("block_count"))
    nextn = int(g("nextn_predict_layers", 0) or 0)
    num_layers = n_layer_all - nextn

    recurrent_raw = g("attention.recurrent_layers")
    if recurrent_raw is not None:
        recurrent = [bool(v) for v in recurrent_raw][:num_layers]
    else:
        interval = int(g("full_attention_interval", 4) or 4)
        recurrent = [(i + 1) % interval != 0 for i in range(num_layers)]

    head_dim = int(g("attention.key_length"))
    sections_raw = g("rope.dimension_sections")
    sections = [int(v) for v in sections_raw if int(v) > 0] if sections_raw else None
    if sections:
        # Interleaved MRoPE rotates 2 * sum(sections) of the head dims;
        # 11+11+10 = 32 pairs = 64 of 256 -> partial_rotary_factor 0.25.
        partial_rotary_factor = 2.0 * sum(sections) / head_dim
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": float(g("rope.freq_base")),
            "mrope_section": sections,
            "mrope_interleaved": True,
            "partial_rotary_factor": partial_rotary_factor,
        }
    else:
        partial_rotary_factor = 0.25
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": float(g("rope.freq_base")),
        }

    cfg = Qwen3_5TextConfig(
        vocab_size=_vocab_size(reader),
        hidden_size=int(g("embedding_length")),
        intermediate_size=int(g("feed_forward_length")),
        num_hidden_layers=num_layers,
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=int(g("context_length")),
        rms_norm_eps=round(float(g("attention.layer_norm_rms_epsilon")), 12),
        rope_parameters=rope_parameters,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")),
        linear_key_head_dim=int(g("ssm.state_size")),
        linear_value_head_dim=int(g("ssm.state_size")),
        linear_num_key_heads=int(g("ssm.group_count")),
        linear_num_value_heads=int(g("ssm.time_step_rank")),
        layer_types=["linear_attention" if r else "full_attention" for r in recurrent],
        tie_word_embeddings=False,
        partial_rotary_factor=partial_rotary_factor,
    )
    # Facts from the published HF config that GGUF metadata cannot express.
    cfg.attn_output_gate = True
    cfg.output_gate_type = "swish"
    cfg.mamba_ssm_dtype = "float32"
    cfg.architectures = ["Qwen3_5ForCausalLM"]
    # llama.cpp's converter (conversion/qwen.py,
    # _LinearAttentionVReorderBase) reorders every per-V-head GDN tensor
    # (in_proj_qkv V rows, in_proj_z/b/a rows, A_log, dt_bias, conv1d V
    # channels, out_proj columns) from HF grouped order
    # [G0v0 G0v1 G0v2, G1v0, ...] to ggml tiled-broadcast order
    # [G0v0 G1v0 ... G15v0, G0v1, ...] whenever num_k_heads != num_v_heads.
    # The layout is self-consistent, but the q/k -> v-head pairing becomes
    # i_k = i_hv % num_k_heads (tile) instead of HF's
    # i_k = i_hv // (num_v_heads // num_k_heads) (repeat_interleave).
    # The GDN core must expand q/k with tile semantics for these weights.
    cfg.gdn_tiled_v_head_layout = True
    logger.info(
        "Built qwen35 config from GGUF: %d layers (%d linear, %d full, "
        "%d MTP skipped), rope sections %s",
        num_layers,
        sum(recurrent),
        num_layers - sum(recurrent),
        nextn,
        sections,
    )

    vision = _vision_config_from_mmproj(gguf_path)
    if vision is None:
        return cfg

    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

    # NOTE: Qwen3_5Config.__init__ only accepts dict/None sub-configs (an
    # instance falls through both branches), so the built objects are
    # attached after construction.
    composite = Qwen3_5Config(
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
    )
    composite.text_config = cfg
    composite.vision_config = vision
    composite.architectures = ["Qwen3_5ForConditionalGeneration"]
    logger.info(
        "Found qwen3vl_merger mmproj: emitting the composite Qwen3_5Config "
        "(%d vision blocks, hidden %d, out %d)",
        vision.depth,
        vision.hidden_size,
        vision.out_hidden_size,
    )
    return composite


@cache
def build_qwen38_dflash2_config_from_gguf(gguf_path: str) -> Any:
    """The DFlash 2 drafter: 5-layer qwen3-shaped block-diffusion drafter.

    Standard dflash backbone (split q/k/v, per-head-dim q/k norms, SwiGLU,
    fc encoder over five target layers) plus the DFlash 2 additions: a
    two-tap dynamic convolution around every attention/MLP sublayer and a
    rank-256 path selector over the top-16 candidates per position. All
    five layers carry a sliding window but the file pins
    `dflash.attention.causal = false`, which overrides per-layer causality.
    `dflash.target_layers` are 1-based, `dflash.block_size` counts the
    anchor row (8 rows = 1 verified + 7 drafted). No embedding or output
    head; both are shared with the target at runtime.
    """
    from transformers import Qwen3Config

    reader = gguf_reader(str(gguf_path))
    g = lambda k, d=None: _field(reader, f"dflash.{k}", d)  # noqa: E731

    num_draft_layers = int(g("block_count"))
    one_based = [int(v) for v in g("target_layers")]
    pattern_raw = g("attention.sliding_window_pattern")
    pattern = (
        [bool(v) for v in pattern_raw]
        if pattern_raw is not None
        else [True] * num_draft_layers
    )

    cfg = Qwen3Config(
        vocab_size=_vocab_size(reader),
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
        sliding_window=int(g("attention.sliding_window")),
        use_sliding_window=True,
        tie_word_embeddings=False,
    )
    # Qwen3Config nulls sliding_window when use_sliding_window is falsy;
    # pin it regardless of that interplay.
    cfg.sliding_window = int(g("attention.sliding_window"))
    cfg.rope_parameters = {
        "rope_type": "default",
        "rope_theta": float(g("rope.freq_base")),
    }
    cfg.model_type = "qwen3_dflash2"
    # Must start with "DFlash": EAGLEConfig prepends the method name to any
    # architecture that does not already carry it.
    cfg.architectures = ["DFlash2QwenDraftModel"]
    # DFlash contract, in the conventions qwen3_dflash and the dflash
    # speculator consume.
    cfg.block_size = int(g("block_size"))
    cfg.n_predict = cfg.block_size - 1
    cfg.draft_vocab_size = cfg.vocab_size
    cfg.layer_types = [
        "sliding_attention" if sliding else "full_attention" for sliding in pattern
    ]
    cfg.target_layer_ids = [layer - 1 for layer in one_based]
    cfg.mask_token_id = int(_field(reader, "tokenizer.ggml.mask_token_id"))
    causal = g("attention.causal")
    cfg.dflash_config = {
        "target_layer_ids": cfg.target_layer_ids,
        "mask_token_id": cfg.mask_token_id,
        "swa_window_size": cfg.sliding_window,
        "use_aux_hidden_state": True,
        "causal": bool(causal) if causal is not None else None,
        # DFlash 2 extensions (see perf/qwen38_metal_design.md).
        "conv_kernel_size": int(g("conv_kernel_size")),
        "conv_group_size": int(g("conv_group_size")),
        "selector_rank": int(g("selector_rank")),
        "selector_top_k": int(g("selector_top_k")),
    }
    logger.info(
        "Built DFlash 2 drafter config: %d layers, block %d (%d drafted), "
        "target layers %s, selector rank %d top-k %d, causal=%s, mask %d",
        num_draft_layers,
        cfg.block_size,
        cfg.n_predict,
        cfg.target_layer_ids,
        cfg.dflash_config["selector_rank"],
        cfg.dflash_config["selector_top_k"],
        cfg.dflash_config["causal"],
        cfg.mask_token_id,
    )
    return cfg


def is_dflash2_gguf(gguf_path: str) -> bool:
    """DFlash 2 discriminator inside the shared `dflash` architecture.

    Three published drafter families share `general.architecture ==
    "dflash"`: the DeepSeek-V4 DSpark drafter carries `dflash.expert_count`,
    DFlash 2 carries the selector metadata, and the Muse-Glimmer drafter
    carries neither.
    """
    reader = gguf_reader(str(gguf_path))
    return "dflash.selector_rank" in reader.fields
