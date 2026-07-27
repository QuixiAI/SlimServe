# SPDX-License-Identifier: Apache-2.0
"""Glm5vConfig — remote-code config carried inside the assembled GLM5V SGLang
checkpoint (referenced by config.json ``auto_map``; loaded with
``--trust-remote-code``, which the checkpoint already requires for the Kimi
image-processor remote code).

Self-contained: depends only on ``transformers``. Mirrors SGLang's in-tree
``KimiK25Config`` structure (``vision_config`` + ``text_config`` + media
placeholder fields) with GLM-5.2 as the text model:

* ``text_config``  -> ``GlmMoeDsaConfig`` (transformers-native ``glm_moe_dsa``).
* ``vision_config``-> MoonViT fields; ``text_hidden_size`` (projector output
  dim) retargeted to GLM hidden 6144.
* ``media_placeholder_token_id`` -> GLM ``<|image|>`` = 154854.
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Glm5vVisionConfig(PretrainedConfig):
    """MoonViT vision tower + PatchMerger projector config.

    Field names/defaults mirror SGLang's ``KimiK25VisionConfig`` (declared
    names like ``hidden_size``) while the official Kimi checkpoint's ``vt_*``
    names arrive via **kwargs and are stored as attributes — SGLang's model
    code reads both families (tower: ``hidden_size``; projector:
    ``vt_hidden_size``/``text_hidden_size``).
    """

    model_type = "glm5v_vision"

    def __init__(
        self,
        # Vision tower
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size=(2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        # MM projector
        mm_projector_type: str = "patchmerger",
        mm_hidden_size: int | None = None,
        vt_hidden_size: int | None = None,   # SGLang kimi_k25 projector reads this (== vision-tower hidden)
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        text_hidden_size: int = 6144,  # GLM-5.2 hidden (Kimi default is 7168)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else hidden_size
        self.vt_hidden_size = vt_hidden_size if vt_hidden_size is not None else hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size

    def __getattr__(self, name):
        # SGLang's kimi_k25 reads vt_-prefixed vision fields (vt_hidden_size, vt_intermediate_size, ...)
        # that our config declares without the prefix; alias any missing vt_* to the base attribute.
        # Reads __dict__ directly (no recursion) and raises normally if the base isn't set.
        if name.startswith("vt_"):
            d = object.__getattribute__(self, "__dict__")
            base = name[3:]
            if base in d:
                return d[base]
        raise AttributeError(name)


class Glm5vConfig(PretrainedConfig):
    """glm5v top-level config: MoonViT ``vision_config`` + GLM-5.2 ``text_config``."""

    model_type = "glm5v"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 154854,  # GLM <|image|>
        pad_token_id: int = 154820,
        use_unified_vision_chunk: bool = True,
        video_placeholder: str = "<|glm5v_video_placeholder|>",
        encoder_only: bool = False,
        language_only: bool = False,
        **kwargs,
    ):
        # Vision config (MoonViT).
        if vision_config is None:
            self.vision_config = Glm5vVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = Glm5vVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # Text config (GLM-5.2 / glm_moe_dsa), built via AutoConfig so the
        # transformers-native GlmMoeDsaConfig class is used.
        raw_text = dict(text_config) if isinstance(text_config, dict) else None
        if text_config is None:
            self.text_config = AutoConfig.for_model("glm_moe_dsa")
        elif isinstance(text_config, dict):
            tc = dict(text_config)
            tc.setdefault("model_type", "glm_moe_dsa")
            # Newer transformers (in the SGLang serving image) validates `layer_types`
            # via a StrictDataclass and rejects the legacy DSA value
            # "deepseek_sparse_attention". The DSA attention path is selected from
            # model_type + the DSA config fields (index_topk etc.), NOT from layer_types,
            # so drop it to pass validation without changing behavior.
            tc.pop("layer_types", None)
            self.text_config = AutoConfig.for_model(**tc)
        else:
            self.text_config = text_config

        # transformers 5.8.x GlmMoeDsaConfig drops/clobbers raw DSA fields the
        # sparse-attention path needs. SGLang applies this same restore for
        # bare GlmMoeDsaForCausalLM checkpoints (see its HfModelConfigParser;
        # fixed upstream by transformers PR #46338, gone once >= 5.10); our
        # top-level arch is Glm5v so we replicate it here.
        if raw_text is not None:
            for key in ("qk_rope_head_dim", "index_topk_freq"):
                if key in raw_text:
                    setattr(self.text_config, key, raw_text[key])
            if hasattr(self.text_config, "qk_nope_head_dim") and hasattr(
                self.text_config, "qk_rope_head_dim"
            ):
                self.text_config.qk_head_dim = (
                    self.text_config.qk_nope_head_dim
                    + self.text_config.qk_rope_head_dim
                )

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder
        self.encoder_only = encoder_only
        self.language_only = language_only

        # Propagate quantization config from the text model (Kimi pattern):
        # only the GLM text Linears are FP8; vision/projector stay bf16 by
        # construction in the model code.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
