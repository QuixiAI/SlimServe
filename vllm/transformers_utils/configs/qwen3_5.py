# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 model configuration.

Upstream vLLM vendors full copies of these config classes to support older
transformers releases. This fork pins transformers 5.15.0, which ships them,
so we re-export the canonical classes instead: the HF-checkpoint path
(AutoConfig) produces exactly these classes, keeping isinstance checks
consistent between the GGUF-built config and HF-loaded configs.
"""

from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
]
