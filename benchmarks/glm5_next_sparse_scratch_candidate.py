# SPDX-License-Identifier: Apache-2.0
"""Rejected allocation experiment, retained only for diagnostic reproduction.

GPU graph tests passed, but explicit storage did not reduce peak allocated
bytes or materially improve latency. Never imported by serving.

Not KV state and not a process-global cache. One owner must belong to one
model/PP stage whose attention calls and graph replays are serialized.
Concurrent streams or overlapping microbatches require separate owners.
Outputs deliberately never alias this scratch. No allocation may resize
after a CUDA graph has captured a view.
"""

import torch


def sparse_tc_tile_capacity(max_rows: int, max_indices: int) -> int:
    if not 1 <= max_rows <= 32 or max_indices < 1:
        raise ValueError("expected 1..32 decode rows and positive index width")
    return max(
        rows * ((max_indices + split - 1) // split)
        for rows in range(1, max_rows + 1)
        for split in (32 if rows < 8 else 128,)
    )


class SparseTCWorkspace:
    def __init__(self, max_rows: int, max_indices: int, heads: int, device):
        if heads != 8:
            raise ValueError("shared sparse-TC scratch is scoped to TP8/H8")
        self.max_rows = max_rows
        self.max_indices = max_indices
        self.heads = heads
        self.tiles = sparse_tc_tile_capacity(max_rows, max_indices)
        self._storage = torch.empty(
            self.tiles * heads * 514, dtype=torch.float32, device=device
        )

    def get(self, rows: int, width: int, split: int, heads: int, device):
        if (
            not 1 <= rows <= self.max_rows
            or not 1 <= width <= self.max_indices
            or split != (32 if rows < 8 else 128)
            or heads != self.heads
            or torch.device(device) != self._storage.device
        ):
            raise ValueError("shared sparse-TC scratch shape/device/policy mismatch")
        parts = (width + split - 1) // split
        count = rows * parts * heads
        assert count <= self.tiles * heads
        partial_end = self.tiles * heads * 512
        maxima_end = partial_end + self.tiles * heads
        return (
            self._storage[: count * 512].view(rows, parts, heads, 512),
            self._storage[partial_end : partial_end + count].view(rows, parts, heads),
            self._storage[maxima_end : maxima_end + count].view(rows, parts, heads),
        )


def shared_sparse_tc_nope(
    q, cache, block_table, indices, topk_length, scale, *, split=32, workspace=None
):
    from vllm.quixicore.sparse_mla_tc import (
        _sparse_tc_part,
        _sparse_tc_reduce,
        sparse_tc_nope,
    )
    from vllm.triton_utils import triton

    if workspace is None:
        return sparse_tc_nope(q, cache, block_table, indices, topk_length, scale,
                              split=split)
    assert q.ndim == 3 and q.shape[1:] == (8, 512)
    assert cache.ndim == 3 and cache.shape[2] == 512
    assert q.dtype == cache.dtype == torch.bfloat16
    assert cache.stride()[1:] == (512, 1) and cache.shape[1] > 0
    assert indices.shape[0] == block_table.shape[0] == q.shape[0]
    assert topk_length.shape == (q.shape[0],)
    assert all(t.dtype == torch.int32 for t in (block_table, indices, topk_length))
    assert all(t.is_contiguous() for t in (q, block_table, indices, topk_length))
    assert q.is_cuda and all(
        t.device == q.device for t in (cache, block_table, indices, topk_length)
    )
    rows, heads, _ = q.shape
    out = torch.empty_like(q)
    partial, maxima, denominators = workspace.get(
        rows, indices.shape[1], split, heads, q.device
    )
    parts = partial.shape[1]
    _sparse_tc_part[(rows, parts)](
        q, cache, block_table, indices, topk_length,
        partial, maxima, denominators, block_table.stride(0), cache.stride(0),
        cache.shape[0], scale, heads, cache.shape[1], indices.shape[1], parts,
        split, num_warps=4, num_stages=1,
    )
    _sparse_tc_reduce[(rows, heads, 8)](
        partial, maxima, denominators, out, heads, parts,
        triton.next_power_of_2(parts), num_warps=4,
    )
    return out
