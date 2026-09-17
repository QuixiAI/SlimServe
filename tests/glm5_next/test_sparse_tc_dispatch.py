# SPDX-License-Identifier: Apache-2.0
"""CPU guards for explicit sparse-TC decode opt-in and native fallbacks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backends.mla import quixicore_mla_sparse as mla_sparse
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
    # 16 heads per rank (TP4) fills the kernel's 16-row tile (2026-09-11).
    assert _sparse_tc_option(config, 16)
    with pytest.raises(ValueError, match="8 or 16 heads"):
        _sparse_tc_option(config, 4)
    # Speculative rows are ordinary query rows to this path (2026-09-10).
    config.speculative_config = object()
    assert _sparse_tc_option(config, 8)
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
def test_decode_shapes_and_prefill_fallback(rows, split, monkeypatch):
    """The SM80 table: SPLIT=128 above eight rows, native past 32."""
    monkeypatch.setattr(mla_sparse, "_sparse_tc_sm120", lambda: False)
    q = torch.empty(rows, 8, 512, dtype=torch.bfloat16)
    cache = torch.empty(1, 576, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(num_prefills=0)
    assert _sparse_tc_split(True, q, cache, metadata) == split
    assert _sparse_tc_split(False, q, cache, metadata) == 0
    metadata.num_prefills = 1
    assert _sparse_tc_split(True, q, cache, metadata) == 0


@pytest.mark.parametrize(
    "rows,split",
    [
        (0, 0),
        # Below the measured crossover the native SIMT path is still ahead.
        (1, 0),
        (4, 0),
        (8, 0),
        (15, 0),
        # From 16 rows the tensor-core kernel wins, at the one tile that
        # fits sm_120's 99 KB of shared memory (SPLIT=128 asks 144 KB).
        (16, 64),
        (32, 64),
        # Speculative rows are batch x (drafts + 1): 16 x 3 = 48.
        (48, 64),
        (64, 64),
        (65, 0),
        (2048, 0),
    ],
)
def test_decode_shapes_sm120(rows, split, monkeypatch):
    """sm_120 has its own table: a 64-wide tile, and only from 16 rows up."""
    monkeypatch.setattr(mla_sparse, "_sparse_tc_sm120", lambda: True)
    q = torch.empty(rows, 16, 512, dtype=torch.bfloat16)
    cache = torch.empty(1, 576, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(num_prefills=0)
    assert _sparse_tc_split(True, q, cache, metadata) == split
    assert _sparse_tc_split(False, q, cache, metadata) == 0
    metadata.num_prefills = 1
    assert _sparse_tc_split(True, q, cache, metadata) == 0


def test_sm120_tile_fits_its_shared_memory(monkeypatch):
    """The sm_120 tile must stay inside 99 KB: SPLIT x 512 bf16 latents plus
    the 16 x 512 bf16 query rows. This is the constraint that kept the switch
    off, so it is asserted rather than left to a comment."""
    monkeypatch.setattr(mla_sparse, "_sparse_tc_sm120", lambda: True)
    split = mla_sparse._SPARSE_TC_SM120_SPLIT
    latents = split * 512 * 2
    queries = 16 * 512 * 2
    assert (latents + queries) <= 99 * 1024
    assert (128 * 512 * 2 + queries) > 99 * 1024


@pytest.mark.parametrize(
    "heads,width,dtype",
    [(4, 512, torch.bfloat16), (8, 576, torch.bfloat16), (8, 512, torch.float16)],
)
def test_other_geometries_keep_native(heads, width, dtype):
    q = torch.empty(8, heads, width, dtype=dtype)
    cache = torch.empty(1, 576, width, dtype=dtype)
    assert _sparse_tc_split(True, q, cache, SimpleNamespace(num_prefills=0)) == 0
