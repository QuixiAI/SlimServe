# SPDX-License-Identifier: Apache-2.0
"""CPU contract test: the Metal router's dropped slots (id -1, weight 0) are
remapped onto unused experts before the tiled ggml_moe_mm_id routes, which
need every id >= 0 and unique within a token."""

import torch

from vllm.model_executor.layers.quantization.gguf.fused_moe import (
    _fill_dropped_router_ids,
)


def test_dropped_ids_fill_unused_unique_experts():
    E, K = 10, 4
    ids = torch.tensor(
        [
            [3, 0, 7, 1],  # all valid: unchanged
            [0, 2, -1, -1],  # partial: expert 0 selected and aliased by -1
            [-1, -1, -1, -1],  # no finite score at all
            [5, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    out = _fill_dropped_router_ids(ids, E)
    assert out.dtype == ids.dtype
    valid = ids >= 0
    assert torch.equal(out[valid], ids[valid])
    assert (out >= 0).all() and (out < E).all()
    for row in out.tolist():
        assert len(set(row)) == K, row


def test_dropped_ids_fill_when_e_equals_k():
    ids = torch.tensor([[2, 0, -1, -1]], dtype=torch.int64)
    out = _fill_dropped_router_ids(ids, 4)
    assert sorted(out[0].tolist()) == [0, 1, 2, 3]
    assert out[0, :2].tolist() == [2, 0]
