# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2-Vision.

GLM-5.2-Vision pairs a MoonViT3d tower and patch-merging projector with the
GLM-5.2 (``glm_moe_dsa``) text backbone. The vision half is structurally the
same tower Kimi-K2.5 uses — llama.cpp reaches the same conclusion, routing
``PROJECTOR_TYPE_GLM5V`` through ``clip_graph_kimik25`` — so this reuses
:class:`KimiK25ForConditionalGeneration` wholesale and changes only what
actually differs:

* the text backbone is ``GlmMoeDsaForCausalLM`` rather than DeepSeek-V2/V3;
* the projector's output width. vLLM's ``KimiK25MultiModalProjector`` sizes
  ``linear_2`` from ``vision_config.mm_hidden_size``, but GLM-5.2-Vision's
  config carries ``mm_hidden_size = 1152`` (the tower width) and puts the
  projector's true output width in ``text_hidden_size = 6144``. The checkpoint
  weight is ``[6144, 4608]``, so the config is shimmed before the projector is
  constructed.
"""

from collections.abc import Iterable

import torch
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_k25 import (
    KimiK25DummyInputsBuilder,
    KimiK25ForConditionalGeneration,
    KimiK25MultiModalProcessor,
    KimiK25ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

logger = init_logger(__name__)


class Glm5vProcessingInfo(KimiK25ProcessingInfo):
    """Processing info for GLM-5.2-Vision.

    The Baseten checkpoint ships Kimi-K2.5's processor modules verbatim
    (``kimi_k25_processor.py``, ``kimi_k25_vision_processing.py``), so the
    preprocessing contract is inherited; only the config class and the
    placeholder token differ.

    ``Glm5vConfig`` lives in the checkpoint's remote code, so there is no
    importable class to assert against — ``PretrainedConfig`` is the tightest
    static bound available. GLM's placeholder is ``<|image|>`` (id 154854);
    Kimi's ``<|media_pad|>`` is absent from this tokenizer, so without the
    override the base class silently falls through to the config value.
    """

    config_cls = PretrainedConfig
    media_placeholder_token = "<|image|>"


class Glm5vDummyInputsBuilder(KimiK25DummyInputsBuilder):
    pass


class Glm5vMultiModalProcessor(KimiK25MultiModalProcessor):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    Glm5vMultiModalProcessor,
    info=Glm5vProcessingInfo,
    dummy_inputs=Glm5vDummyInputsBuilder,
)
class Glm5vForConditionalGeneration(KimiK25ForConditionalGeneration):
    """GLM-5.2-Vision: MoonViT3d tower + patch-merge projector + GLM-5.2 MoE."""

    # The text backbone. KimiK25 hardcodes DeepseekV2ForCausalLM; GLM-5.2's
    # entry maps to the same module but carries the DSA indexer config.
    language_model_architectures = ["GlmMoeDsaForCausalLM"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _shim_vision_config(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _maybe_ignore_quant_config(self, quant_config):
        """Keep the vision half unquantized under GGUF.

        A GLM-5.2-Vision GGUF contains only the text tensors -- the tower and
        projector are always sourced from the checkpoint's bf16 safetensors --
        so building them with the GGUF method would create packed parameters
        that nothing ever fills.
        """
        if quant_config is not None and quant_config.get_name() == "gguf":
            return None
        return super()._maybe_ignore_quant_config(quant_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(weights)


def _shim_vision_config(config) -> None:
    """Point the projector's output width at the text hidden size.

    ``KimiK25MultiModalProjector`` builds ``linear_2`` as
    ``Linear(hidden_size * merge_h * merge_w, config.mm_hidden_size)``. For
    Kimi that last term is the LLM width; for GLM-5.2-Vision it is the tower
    width (1152) and the LLM width lives in ``text_hidden_size`` (6144).
    Without this the projector is built 1152-wide and the checkpoint's
    ``[6144, 4608]`` ``linear_2.weight`` fails to load.
    """
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return
    text_hidden = getattr(vision_config, "text_hidden_size", None)
    if text_hidden is None:
        text_hidden = config.get_text_config().hidden_size
    current = getattr(vision_config, "mm_hidden_size", None)
    if current != text_hidden:
        logger.info(
            "GLM-5.2-Vision: projector output width %s -> %s (text_hidden_size)",
            current,
            text_hidden,
        )
        vision_config.mm_hidden_size = text_hidden
