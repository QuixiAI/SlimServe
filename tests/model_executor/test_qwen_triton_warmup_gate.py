# SPDX-License-Identifier: Apache-2.0
"""Qwen Triton warmup model-type gate.

Regression (rtx3090 prod, 2026-08-29): the served qwen4_exp model type was
absent from the gate, so the entire Qwen Triton warmup silently skipped and
every covered kernel JIT-compiled on the first production request.
"""

from vllm.model_executor.warmup.qwen_triton_warmup import _QWEN_MODEL_TYPES


def test_qwen4_exp_passes_warmup_gate():
    assert "qwen4_exp" in _QWEN_MODEL_TYPES
    assert "qwen4_exp_text" in _QWEN_MODEL_TYPES


def test_qwen3_gdn_family_still_gated_in():
    for model_type in (
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    ):
        assert model_type in _QWEN_MODEL_TYPES
