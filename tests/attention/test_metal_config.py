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
