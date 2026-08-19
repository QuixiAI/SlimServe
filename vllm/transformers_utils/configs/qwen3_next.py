# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3Next model configuration.

The installed transformers (>=5.x) ships ``Qwen3NextConfig`` natively, so this
module re-exports it instead of vendoring upstream vLLM's copy. It exists to
satisfy the lazy registration in ``configs/__init__.py`` and the import in
``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py``.
"""

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

__all__ = ["Qwen3NextConfig"]
