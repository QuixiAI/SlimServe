# SPDX-License-Identifier: Apache-2.0
"""CPU guards for explicit sparse-TC decode opt-in and native fallbacks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backends.mla.quixicore_mla_sparse import (
    _sparse_tc_option,
    _sparse_tc_split,
)


@pytest.mark.parametrize("value", [1, "false", None])
def test_option_requires_boolean(value):
    config = SimpleNamespace(additional_config={"glm5_next_sparse_tc_decode": value})
    with pytest.raises(ValueError, match="boolean"):
        _sparse_tc_option(config, 8)


def test_option_is_off_by_default():
    assert not _sparse_tc_option(SimpleNamespace(additional_config={}), 8)


def test_option_rejects_unqualified_platform_heads_and_spec(monkeypatch):
    import vllm.platforms

    platform = Mock()
    platform.is_cuda.return_value = True
    platform.is_device_capability.return_value = True
    monkeypatch.setattr(vllm.platforms, "current_platform", platform)
    config = SimpleNamespace(
        additional_config={"glm5_next_sparse_tc_decode": True}, speculative_config=None
    )
    assert _sparse_tc_option(config, 8)
    with pytest.raises(ValueError, match="8 heads"):
        _sparse_tc_option(config, 16)
    config.speculative_config = object()
    with pytest.raises(ValueError, match="no speculation"):
        _sparse_tc_option(config, 8)
    platform.is_device_capability.return_value = False
    with pytest.raises(ValueError, match="SM80"):
        _sparse_tc_option(config, 8)


@pytest.mark.parametrize(
    "rows,split",
    [
        (0, 0),
        (1, 32),
        (2, 32),
        (4, 32),
        (8, 128),
        (16, 128),
        (32, 128),
        (33, 0),
        (64, 0),
        (2048, 0),
    ],
)
def test_decode_shapes_and_prefill_fallback(rows, split):
    q = torch.empty(rows, 8, 512, dtype=torch.bfloat16)
    cache = torch.empty(1, 576, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(num_prefills=0)
    assert _sparse_tc_split(True, q, cache, metadata) == split
    assert _sparse_tc_split(False, q, cache, metadata) == 0
    metadata.num_prefills = 1
    assert _sparse_tc_split(True, q, cache, metadata) == 0


@pytest.mark.parametrize(
    "heads,width,dtype",
    [(16, 512, torch.bfloat16), (8, 576, torch.bfloat16), (8, 512, torch.float16)],
)
def test_other_geometries_keep_native(heads, width, dtype):
    q = torch.empty(8, heads, width, dtype=dtype)
    cache = torch.empty(1, 576, width, dtype=dtype)
    assert _sparse_tc_split(True, q, cache, SimpleNamespace(num_prefills=0)) == 0
