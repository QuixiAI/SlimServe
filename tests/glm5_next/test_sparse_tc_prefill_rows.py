# SPDX-License-Identifier: Apache-2.0
"""Parity gates for the prefill-shaped sparse tensor-core kernel
(`sparse_tc_nope_rows`): the pure-torch reference, the native SIMT kernel it
replaces on prefill chunks, and the partitioned tensor-core decode kernel."""

import math

import pytest
import torch

from tests.glm5_next.test_sparse_tc_candidate import reference


def make_case(rows, heads, bs, pages, width, seed, gaps):
    torch.manual_seed(seed)
    backing = torch.full((pages, 3, bs, 512), 7.0, device="cuda", dtype=torch.bfloat16)
    cache = backing[:, 1]
    cache.normal_(std=0.5)
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
    total = pages * bs
    indices = torch.full((rows, width), -1, device="cuda", dtype=torch.int32)
    lengths = torch.empty(rows, device="cuda", dtype=torch.int32)
    for r in range(rows):
        # Prefill rows see a growing prefix: row r selects up to width tokens
        # of [0, min(total, 64 + 37 * r)), sorted like the indexer's output.
        visible = min(total, 64 + 37 * r)
        n = min(width, visible)
        pick = torch.randperm(visible, device="cuda")[:n].sort().values.to(torch.int32)
        if gaps and r % 3 == 1:
            # Expanded short-context lists carry -1 gaps before the tail.
            pick[::5] = -1
        indices[r, :n] = pick
        lengths[r] = n if not (gaps and r % 3 == 2) else max(1, n // 2)
    return q, cache, table, indices, lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tile", [32, 64])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize(
    "rows,bs,gaps", [(1, 576, False), (7, 64, True), (64, 64, True), (257, 64, False)]
)
@torch.no_grad()
def test_rows_match_reference_and_native(rows, bs, gaps, heads, tile):
    from vllm.quixicore import quixicore_ops as qc
    from vllm.quixicore.sparse_mla_tc import sparse_tc_nope_rows

    pages = 64 if bs == 576 else 512
    q, cache, table, indices, lengths = make_case(
        rows, heads, bs, pages, 2080, 100 + rows + heads, gaps
    )
    scale = 1 / math.sqrt(256)
    actual = sparse_tc_nope_rows(q, cache, table, indices, lengths, scale, tile=tile)
    torch.cuda.synchronize()
    expected = reference(q, cache, table, indices, lengths.tolist(), scale)
    torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.002)
    native = qc.mla_decode_bf16_sparse_nope(
        q,
        cache,
        table,
        indices,
        lengths,
        bs,
        scale,
        partition_size=0,
        page_stride_bytes=0,
    )
    torch.testing.assert_close(actual.float(), native.float(), atol=0.004, rtol=0.004)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_rows_match_partitioned_tc_and_strided_pages():
    from vllm.quixicore.sparse_mla_tc import sparse_tc_nope, sparse_tc_nope_rows

    q, cache, table, indices, lengths = make_case(16, 16, 64, 512, 2080, 7, True)
    assert not cache.is_contiguous()  # the packed-slab stride: pages 3x apart
    scale = 1 / math.sqrt(256)
    rows_out = sparse_tc_nope_rows(q, cache, table, indices, lengths, scale)
    part_out = sparse_tc_nope(q, cache, table, indices, lengths, scale, split=64)
    torch.testing.assert_close(
        rows_out.float(), part_out.float(), atol=0.004, rtol=0.004
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_rows_fp8_matches_partitioned_fp8():
    from vllm.quixicore.sparse_mla_tc import sparse_tc_nope, sparse_tc_nope_rows

    q, cache, table, indices, lengths = make_case(8, 16, 64, 128, 2080, 11, True)
    fp8 = cache.contiguous().to(torch.float8_e4m3fn).view(torch.uint8)
    scale = 1 / math.sqrt(256)
    rows_out = sparse_tc_nope_rows(q, fp8, table, indices, lengths, scale, kv_scale=1.0)
    part_out = sparse_tc_nope(
        q, fp8, table, indices, lengths, scale, split=64, kv_scale=1.0
    )
    torch.testing.assert_close(
        rows_out.float(), part_out.float(), atol=0.004, rtol=0.004
    )
