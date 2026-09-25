# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Revalidating a config must preserve the allocated cache's geometry."""

from types import SimpleNamespace

import pytest

from vllm.config import CacheConfig
from vllm.platforms.metal import MetalPlatform


@pytest.mark.parametrize(
    ("requested", "resolved"), [(None, 16), (None, 800), (32, 32), (800, 800)]
)
def test_revalidation_preserves_resolved_cache_blocks(monkeypatch, requested, resolved):
    # Compatibility installation is unrelated to cache geometry and must not
    # patch the test process's torch operations on non-Metal CI machines.
    monkeypatch.setattr(
        "vllm.platforms.metal_compat.apply_compat_patches", lambda: None
    )
    cache = CacheConfig(block_size=requested)
    specified = cache.user_specified_block_size
    # Hybrid backend selection finalizes this after initial config validation.
    cache.block_size = resolved
    cache.mamba_block_size = resolved
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto", tensor_parallel_size=1, enable_dbo=False
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        compilation_config=SimpleNamespace(),
        model_config=None,
        cache_config=cache,
    )

    # DFlash's attention-config copy invokes the platform hook again on the
    # same cache object, after the physical KV cache has already been allocated.
    for _ in range(2):
        MetalPlatform.check_and_update_config(config)
        assert cache.block_size == resolved
        assert cache.mamba_block_size == resolved
        assert cache.user_specified_block_size == specified


@pytest.mark.parametrize(
    ("env", "requested", "expected"),
    [
        (None, None, False),  # opt-in: unset pins the synchronous scheduler
        (None, True, False),
        ("0", True, False),
        ("1", None, None),  # opted in: the config layer's resolution applies
        ("1", True, True),
    ],
)
def test_async_scheduling_is_opt_in(monkeypatch, env, requested, expected):
    # dsv4-xxs-1 (DSpark drafter) rolls its 2000-token exact-token anchor
    # under async scheduling (2026-09-12 bisect), so the platform default
    # stays synchronous; gated profiles opt in through VLLM_METAL_ASYNC_SCHED=1.
    monkeypatch.setattr(
        "vllm.platforms.metal_compat.apply_compat_patches", lambda: None
    )
    if env is None:
        monkeypatch.delenv("VLLM_METAL_ASYNC_SCHED", raising=False)
    else:
        monkeypatch.setenv("VLLM_METAL_ASYNC_SCHED", env)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto", tensor_parallel_size=1, enable_dbo=False
        ),
        scheduler_config=SimpleNamespace(async_scheduling=requested),
        compilation_config=SimpleNamespace(),
        model_config=None,
        cache_config=CacheConfig(block_size=None),
    )
    MetalPlatform.check_and_update_config(config)
    assert config.scheduler_config.async_scheduling is expected
