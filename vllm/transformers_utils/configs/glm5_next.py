# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fallback ``glm5_next`` config classes.

transformers >= 5.16.1 ships ``Glm5NextConfig`` / ``Glm5NextTextConfig`` /
``Glm5NextVisionConfig`` natively, and ``vllm/model_executor/models/glm5_next.py``
imports them unconditionally, so on a serving box the native classes are what
the model sees. These stand-ins exist for one purpose: building and inspecting
the config derived from a ``glm5-next`` GGUF (``gguf_glm5_next.py``) on a
machine whose transformers predates the native classes -- the offline unit
tests. They mirror the native dataclass field-for-field (defaults, derived
fields, ``model_type`` strings) so the derived config reads identically either
way, and are never registered globally: ``gguf_glm5_next`` imports them only
after the transformers import fails.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

# Native Glm5NextTextConfig defaults (transformers main, configuration_glm5_next.py).
_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 154880,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 45,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_shared_experts": 1,
    "n_routed_experts": 288,
    "routed_scaling_factor": 2.5,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "qk_nope_head_dim": 256,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "hidden_act": "silu",
    "max_position_embeddings": 1048576,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "mlp_layer_types": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "layer_types": None,
    "indexer_types": None,
    "swiglu_limit": 10.0,
    "linear_head_dim": 128,
    "linear_num_heads": 64,
    "linear_conv_kernel_dim": 4,
    "linear_lower_bound": -5.0,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "output_router_logits": False,
    "router_aux_loss_coef": 0.001,
    "index_kpool": 16,
    "index_kpool_always_select_tail": True,
}

_VISION_DEFAULTS: dict[str, Any] = {
    "depth": 24,
    "hidden_size": 1024,
    "hidden_act": "silu",
    "attention_bias": True,
    "attention_dropout": 0.0,
    "num_heads": 16,
    "in_channels": 3,
    "image_size": 336,
    "patch_size": 14,
    "rms_norm_eps": 1e-05,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 1536,
    "intermediate_size": 4096,
    "initializer_range": 0.02,
    "rope_parameters": None,
    "projection_intermediate_size": 10240,
    "swiglu_limit": 10.0,
}


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        pad_token_id: int | None = 154820,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        fields = dict(_TEXT_DEFAULTS)
        extra = {k: kwargs.pop(k) for k in list(kwargs) if k in fields}
        fields.update(extra)
        # Native __post_init__ derivations.
        if fields["num_key_value_heads"] is None:
            fields["num_key_value_heads"] = fields["num_attention_heads"]
        n = fields["num_hidden_layers"]
        if fields["mlp_layer_types"] is None:
            fields["mlp_layer_types"] = ["dense"] * min(3, n) + ["sparse"] * (n - 3)
        if fields["layer_types"] is None:
            fields["layer_types"] = [
                "linear_attention" if i % 4 != 3 else "deepseek_sparse_attention"
                for i in range(n)
            ]
        fields["layer_types"] = [
            "deepseek_sparse_attention" if t == "full_attention" else t
            for t in fields["layer_types"]
        ]
        if fields["indexer_types"] is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                fields["indexer_types"] = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern]
                    if isinstance(pattern, str)
                    else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                fields["indexer_types"] = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(n)
                ]
        linear_attn = kwargs.get("linear_attn_config")
        if linear_attn is not None:
            fields["linear_head_dim"] = linear_attn.get(
                "head_dim", fields["linear_head_dim"]
            )
            fields["linear_num_heads"] = linear_attn.get(
                "num_heads", fields["linear_num_heads"]
            )
            fields["linear_conv_kernel_dim"] = linear_attn.get(
                "short_conv_kernel_size", fields["linear_conv_kernel_dim"]
            )
            fields["linear_lower_bound"] = linear_attn.get(
                "gate_lower_bound", fields["linear_lower_bound"]
            )
            if (
                linear_attn.get("safe_gate", True)
                and fields["linear_lower_bound"] is None
            ):
                fields["linear_lower_bound"] = -5.0
        kwargs.pop("head_dim", None)
        fields["head_dim"] = fields["qk_rope_head_dim"]
        fields["qk_head_dim"] = fields["qk_rope_head_dim"] + fields["qk_nope_head_dim"]
        for key, value in fields.items():
            setattr(self, key, value)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"
    base_config_key = "vision_config"
    attribute_map = {"num_attention_heads": "num_heads"}

    def __init__(self, **kwargs: Any) -> None:
        fields = dict(_VISION_DEFAULTS)
        fields.update({k: kwargs.pop(k) for k in list(kwargs) if k in fields})
        for key, value in fields.items():
            setattr(self, key, value)
        super().__init__(**kwargs)


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {
        "vision_config": Glm5NextVisionConfig,
        "text_config": Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        elif text_config is None:
            text_config = Glm5NextTextConfig()
        if isinstance(vision_config, dict):
            vision_config = Glm5NextVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Glm5NextVisionConfig()
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Glm5NextConfig", "Glm5NextTextConfig", "Glm5NextVisionConfig"]
