# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse-Glimmer-30B configuration.

Assembled from GGUF metadata (`muse-glimmer.*` keys plus the mmproj's
`clip.vision.*` keys) by `gguf_muse_glimmer.py`; there is no HF config.json
for this release. Reference: the model card of
meta-models/Muse-Glimmer-30B-GGUF and the tensor inventory of the published
k-quant GGUFs.

Text: 52-layer dense decoder, hidden 6656, SwiGLU FFN 19968, GQA 32/2 with
head_dim 128 and per-head QK-RMSNorm, sigmoid-gated attention output,
sandwich norms (pre + post for both halves), [local,local,local,global]
interleaving with a 2048 sliding window, RoPE theta 500k on local layers
only (global layers are NoPE), final logits scaled by `logit_scale` and
soft-capped at 20.

Vision: 50-block pre-norm ViT (Perception Encoder), width 1536, patch 14,
1024 learned positions, followed by a 3-layer MLP projector over 2x2
spatially merged patches (6144 -> 4096 -> 4096 -> 6656).
"""

from transformers import PretrainedConfig


class MuseGlimmerVisionConfig(PretrainedConfig):
    model_type = "muse_glimmer_vision"

    def __init__(
        self,
        hidden_size: int = 1536,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 16,
        intermediate_size: int = 8960,
        image_size: int = 896,
        patch_size: int = 14,
        num_positions: int = 1024,
        layer_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        projector_hidden_size: int = 4096,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_positions = num_positions
        self.layer_norm_eps = layer_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.projector_hidden_size = projector_hidden_size
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]


class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"

    def __init__(
        self,
        vocab_size: int = 202_048,
        hidden_size: int = 6656,
        intermediate_size: int = 19_968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131_072,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500_000.0,
        rope_parameters: dict | None = None,
        sliding_window: int = 2048,
        # True = sliding layer, False = global layer; length num_hidden_layers.
        sliding_window_pattern: list[bool] | None = None,
        final_logit_softcapping: float | None = 20.0,
        logit_scale: float = 1.0,
        attention_bias: bool = False,
        gated_attention: bool = True,
        sandwich_norms: bool = True,
        tie_word_embeddings: bool = False,
        vision_config: dict | MuseGlimmerVisionConfig | None = None,
        media_placeholder_token_id: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": rope_theta,
        }
        self.sliding_window = sliding_window
        if sliding_window_pattern is None:
            # [local, local, local, global] repeating.
            sliding_window_pattern = [(i % 4) != 3 for i in range(num_hidden_layers)]
        self.sliding_window_pattern = sliding_window_pattern
        self.final_logit_softcapping = final_logit_softcapping
        self.logit_scale = logit_scale
        self.attention_bias = attention_bias
        self.gated_attention = gated_attention
        self.sandwich_norms = sandwich_norms
        if isinstance(vision_config, dict):
            vision_config = MuseGlimmerVisionConfig(**vision_config)
        self.vision_config = vision_config
        self.media_placeholder_token_id = media_placeholder_token_id
