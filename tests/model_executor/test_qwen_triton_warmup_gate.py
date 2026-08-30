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


def test_zero_kv_warmup_uses_only_the_zeroer_public_api():
    """Regression (rtx3090 crash-loop, 2026-08-30): zero-kv warmup must go
    through KVBlockZeroer.zero_block_ids, never a replica of its private
    _meta layout - the replica crash-looped the boot when _meta widened."""
    import inspect
    from types import SimpleNamespace

    from vllm.model_executor.warmup import qwen_triton_warmup as mod

    # No code path may read the zeroer's private _meta.
    src = inspect.getsource(mod)
    lines = [
        line
        for line in src.splitlines()
        if "_meta" in line and not line.lstrip().startswith(("#", "zeroer's"))
    ]
    assert lines == [], f"private _meta coupling reintroduced: {lines}"

    calls: list[list[int]] = []
    runner = SimpleNamespace(
        kv_block_zeroer=SimpleNamespace(zero_block_ids=calls.append)
    )
    assert mod._warm_zero_kv_blocks_with_runner_zeroer(runner)
    assert calls == [[0], [0, 1]]  # real API exercised per block count

    # A runner without a zeroer is a clean skip, not a crash.
    assert not mod._warm_zero_kv_blocks_with_runner_zeroer(SimpleNamespace())
