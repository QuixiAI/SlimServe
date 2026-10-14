# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from slimserve import registry
from slimserve.engine import apply_env, engine_kwargs, serve_argv


def _plan():
    return registry.resolve("qwen38-nvfp4-1", "metal", 1, None, 128 << 30)


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("dtype", ["turboquant_k8v4", "turboquant_3bit_nc", "nvfp4"])
@pytest.mark.parametrize(
    "builder", [engine_kwargs, lambda p: serve_argv(p, "localhost", 8000)]
)
def test_quantization_policy_rejects_target_and_draft_overrides(draft, dtype, builder):
    plan = _plan()
    if draft:
        plan = replace(plan, speculative_overrides={"kv_cache_dtype": dtype})
    else:
        plan = replace(plan, engine={**plan.engine, "kv_cache_dtype": dtype})
    with pytest.raises(registry.ProfileError, match="FP8"):
        builder(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("VLLM_QWEN4_EXP_TQ_MAIN_KV", "1"),
        ("VLLM_ATTENTION_BACKEND", "TURBOQUANT"),
    ],
)
@pytest.mark.parametrize("ambient", [False, True])
def test_turboquant_cannot_be_enabled_by_environment(monkeypatch, key, value, ambient):
    plan = _plan()
    if ambient:
        monkeypatch.setenv(key, value)
    else:
        monkeypatch.delenv(key, raising=False)
        plan = replace(plan, env={**plan.env, key: value})
    with pytest.raises(registry.ProfileError, match="TurboQuant"):
        apply_env(plan)


def test_every_quant_and_platform_keeps_vision_and_fp8_policy():
    data = registry._registry()
    for profile_id, profile in data["profiles"].items():
        for platform in profile["variants"]:
            for quant in registry.quants_for(profile_id, platform, 1 << 50, 1 << 50):
                plan = registry.resolve(
                    profile_id,
                    platform,
                    profile["gpus"],
                    quant.name,
                    1 << 50,
                    1 << 50,
                )
                kwargs = engine_kwargs(plan)
                registry.validate_cache_policy(kwargs)
                if "image" in plan.source.get("modalities", []):
                    assert not kwargs.get("language_model_only"), profile_id
                    limits = kwargs.get("limit_mm_per_prompt", {})
                    assert limits.get("image", 1) != 0, profile_id
                    assert limits.get("vision_chunk", 1) != 0, profile_id


@pytest.mark.parametrize(
    "override",
    [
        {"language_model_only": True},
        {"limit_mm_per_prompt": {"image": 0}},
        {"limit_mm_per_prompt": {"image": {"count": 0}}},
        {"limit_mm_per_prompt": {"vision_chunk": 0}},
    ],
)
def test_vision_cannot_be_disabled_by_a_profile_override(override):
    plan = _plan()
    plan = replace(plan, engine={**plan.engine, **override})
    with pytest.raises(registry.ProfileError, match="Vision"):
        serve_argv(plan, "localhost", 8000)
