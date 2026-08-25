# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Next model configuration.

Re-exported from transformers (>= 5.15, pinned by this fork) instead of
vendoring upstream vLLM's copy; see qwen3_5.py for rationale. Qwen3-Next
itself is not served by this fork — the Gated DeltaNet layer shared with
Qwen3.5 imports this class for typing.
"""

from transformers.models.qwen3_next.configuration_qwen3_next import (
    Qwen3NextConfig,
)

__all__ = ["Qwen3NextConfig"]
