# SPDX-License-Identifier: Apache-2.0
"""Compact pool pages: chunking, ragged batches, rollback, stride and replay."""

import pytest
import torch

from vllm.model_executor.layers.glm5_next_indexer import _pooled_logits_kernel
from vllm.model_executor.layers.glm5_next_pool_cache import (
    cached_pool_logits,
    update_pool_cache,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def allocate(bs, lengths, padding):
    counts = [(n + bs - 1) // bs for n in lengths]
    total = sum(counts)
    physical = torch.randperm(total, device="cuda", dtype=torch.int64)
    table = torch.zeros(len(lengths), max(counts), device="cuda", dtype=torch.int32)
    offset = 0
    for r, count in enumerate(counts):
        table[r, :count] = physical[offset : offset + count]
        offset += count
    raw_backing = torch.full(
        (total, padding + 1, bs, 256), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    compact_backing = torch.full(
        (total, padding + 1, bs, 64), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    return table, raw_backing[:, 0], compact_backing[:, 0]


def slot_ids(table, req, start, end, bs):
    positions = torch.arange(start, end, device="cuda", dtype=torch.int64)
    return table[req, positions // bs].long() * bs + positions % bs


def update_both(rows, slots, ape, raw, compact):
    update_pool_cache(rows, slots, ape, compact)
    valid = slots >= 0
    raw[slots[valid] // raw.shape[1], slots[valid] % raw.shape[1]] = rows[valid]


def compare_logits(table, raw, compact, ape, requests, visible):
    r = len(visible)
    visible = torch.tensor(visible, device="cuda", dtype=torch.int32)
    requests = torch.tensor(requests, device="cuda", dtype=torch.int32)
    width = max(1, (int(visible.max()) + 3) // 4)
    q = torch.randn(r, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(r, 32, device="cuda") * (32**-0.5)
    reference = torch.full((r, width), float("nan"), device="cuda")
    result = torch.full_like(reference, float("nan"))
    _pooled_logits_kernel[(r, min(128, (width + 15) // 16))](
        q,
        weights,
        ape,
        raw,
        table,
        requests,
        visible,
        reference,
        width,
        table.stride(0),
        raw.stride(0),
        128**-0.5,
        BLOCK_SIZE=raw.shape[1],
        H=32,
        D=128,
        KP=4,
        ROW=256,
        BLOCK_P=16,
    )
    cached_pool_logits(q, weights, compact, table, requests, visible, result)
    for i, n in enumerate(visible.tolist()):
        # Invalid pool columns are not consumed by top-k and need not be
        # initialized on steps with no complete pools.
        torch.testing.assert_close(
            result[i, : n // 4], reference[i, : n // 4], atol=1e-6, rtol=1e-6
        )
        if n // 4 > 512:
            expected = reference[i, : n // 4].topk(512).indices.sort().values
            actual = result[i, : n // 4].topk(512).indices.sort().values
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("bs", [64, 1152, 4608])
@pytest.mark.parametrize("seed", range(8))
@torch.no_grad()
def test_long_pool_rounding_and_pruning(bs, seed):
    torch.manual_seed(3000 + seed)
    length = max(4096, bs + 17)
    table, raw, compact = allocate(bs, [length], padding=2)
    ape = torch.randn(4, 128, device="cuda")
    rows = torch.randn(length, 256, device="cuda", dtype=torch.bfloat16)
    update_both(rows, slot_ids(table, 0, 0, length, bs), ape, raw, compact)
    compare_logits(table, raw, compact, ape, [0] * 8, [length - i for i in range(8)])


@pytest.mark.parametrize("bs", [64, 128, 576])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("seed", [0, 21])
@torch.no_grad()
def test_ragged_chunked_pages(bs, padding, seed):
    torch.manual_seed(seed)
    lengths = [bs + 73, bs + 31]
    table, raw, compact = allocate(bs, lengths, padding)
    ape = torch.randn(4, 128, device="cuda")
    sequences = [
        torch.randn(n, 256, device="cuda", dtype=torch.bfloat16) for n in lengths
    ]
    cursors = [0, 0]
    for step, amount in enumerate([1, 2, 5, 17, 3, bs, bs]):
        rows, slots = [], []
        for req, seq in enumerate(sequences):
            end = min(len(seq), cursors[req] + amount + req * (step % 3))
            rows.append(seq[cursors[req] : end])
            slots.append(slot_ids(table, req, cursors[req], end, bs))
            cursors[req] = end
        update_both(torch.cat(rows), torch.cat(slots), ape, raw, compact)
        compare_logits(table, raw, compact, ape, [0, 1], cursors)
    assert cursors == lengths
    assert compact.stride(0) * compact.shape[0] * 4 == raw.stride(0) * raw.shape[0]
    if padding:
        assert torch.isnan(compact._base[:, 1:]).all()


@pytest.mark.parametrize("start", [40, 41, 42, 43, 60, 61, 62, 63])
@pytest.mark.parametrize("accepted_drafts", [0, 1, 3, 5])
@torch.no_grad()
def test_speculative_rollback_and_page_restore(start, accepted_drafts):
    torch.manual_seed(1000 + start + accepted_drafts)
    bs = 64
    table, raw, compact = allocate(bs, [128], padding=2)
    ape = torch.randn(4, 128, device="cuda")
    prefix = torch.randn(start, 256, device="cuda", dtype=torch.bfloat16)
    update_both(prefix, slot_ids(table, 0, 0, start, bs), ape, raw, compact)
    speculative = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)
    update_both(
        speculative, slot_ids(table, 0, start, start + 6, bs), ape, raw, compact
    )
    compare_logits(table, raw, compact, ape, [0] * 6, list(range(start + 1, start + 7)))
    committed = start + 1 + accepted_drafts
    # Simulate a byte-exact host/disk round trip, preserving the page stride
    # and ring together; unlike a process-side memo table, all state travels.
    snapshot = compact.cpu().clone()
    compact.fill_(float("nan"))
    compact.copy_(snapshot)
    replacement = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)
    slots = slot_ids(table, 0, committed, committed + 6, bs)
    # Add ignored graph-padding rows whose values must not enter any pool.
    rows = torch.cat([replacement, torch.full_like(replacement[:2], float("nan"))])
    slots = torch.cat([slots, torch.full((2,), -1, device="cuda", dtype=torch.int64)])
    update_both(rows, slots, ape, raw, compact)
    compare_logits(
        table, raw, compact, ape, [0] * 6, list(range(committed + 1, committed + 7))
    )


@torch.no_grad()
def test_graph_replay_with_new_rows_and_slots():
    torch.manual_seed(222)
    table, raw, compact = allocate(64, [192], padding=1)
    ape = torch.randn(4, 128, device="cuda")
    prefix = torch.randn(131, 256, device="cuda", dtype=torch.bfloat16)
    update_both(prefix, slot_ids(table, 0, 0, 131, 64), ape, raw, compact)
    saved_raw, saved_compact = raw.clone(), compact.clone()
    rows = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)
    slots = slot_ids(table, 0, 131, 137, 64)
    update_pool_cache(rows, slots, ape, compact)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update_pool_cache(rows, slots, ape, compact)
    for iteration in range(5):
        start = 128 + iteration % 4
        slots.copy_(slot_ids(table, 0, start, start + 6, 64))
        rows.normal_()
        compact.copy_(saved_compact)
        raw.copy_(saved_raw)
        graph.replay()
        raw[slots // 64, slots % 64] = rows
        compare_logits(
            table, raw, compact, ape, [0] * 6, list(range(start + 1, start + 7))
        )
