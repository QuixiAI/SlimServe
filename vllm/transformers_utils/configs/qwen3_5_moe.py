# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5-MoE model configuration.

Re-exported from transformers (>= 5.15, pinned by this fork) instead of
vendoring upstream vLLM's copies; see qwen3_5.py for rationale. The MoE
variant itself is not served by this fork — these classes exist so the
shared Qwen3.5 model/config code can type-check and dispatch on them.
"""

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

__all__ = [
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
]
