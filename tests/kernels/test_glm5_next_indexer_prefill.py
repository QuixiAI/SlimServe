# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash pooled indexer's prefill path (pooled keys once per
request, tiled tensor-core matmul) against the per-row kernel, on a
synthetic paged cache with uneven requests and cached prefixes; and the
query-row -> block-table-row map of a prefill chunk."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import glm5_next_indexer as gi
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillMetadata,
    build_prefill_chunk_metadata,
)

H, D, KP, ROW = 32, 128, 4, gi._ROW_DIM
BLOCK_SIZE = 1088  # the served hybrid block size (a multiple of 64)
KSEL = 512  # index_topk 2048 // index_kpool 4
DEV = "cuda"

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only Triton kernels"
)


@pytest.fixture(params=[(8, 64), (2, 128)])
def prefill_tiles(request, monkeypatch):
    """Exercise the original and isolated SM120 candidate through full selection."""
    rt, pt = request.param
    monkeypatch.setattr(gi, "_ROW_TILE", rt)
    monkeypatch.setattr(gi, "_POOL_TILE", pt)


def _synth(query_lens, cached, seed=0):
    """A paged indexer cache for one prefill chunk: request i has
    cached[i] KV tokens before its query_lens[i] query rows (a cached
    prefix), so its visibility runs to cached[i] + row + 1."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_req = len(query_lens)
    seq_lens = [c + q for c, q in zip(cached, query_lens)]
    blocks_per = [-(-s // BLOCK_SIZE) for s in seq_lens]
    num_blocks = sum(blocks_per) + 3
    cache = torch.randn((num_blocks, BLOCK_SIZE, ROW), generator=g, device=DEV).to(
        torch.bfloat16
    )
    perm = torch.randperm(num_blocks - 1, generator=g, device=DEV) + 1
    bt = torch.zeros((n_req, max(blocks_per)), dtype=torch.int32, device=DEV)
    at = 0
    for i, nb in enumerate(blocks_per):
        bt[i, :nb] = perm[at : at + nb].to(torch.int32)
        at += nb
    R = sum(query_lens)
    q = torch.randn((R, H, D), generator=g, device=DEV).to(torch.bfloat16)
    w = torch.rand((R, H), generator=g, device=DEV) * H**-0.5
    ape = torch.randn((KP, D), generator=g, device=DEV)
    row_req = torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(query_lens)]
    ).to(DEV)
    visible = torch.cat(
        [
            c + torch.arange(1, n + 1, dtype=torch.int32)
            for c, n in zip(cached, query_lens)
        ]
    ).to(DEV)
    max_pools = max(1, max(seq_lens) // KP)
    return q, w, ape, cache, bt, row_req, visible, max_pools


def _select(by_request, q, w, ape, cache, bt, row_req, visible, max_pools):
    R = q.shape[0]
    logits = torch.full((R, max_pools), float("nan"), device=DEV)
    topk = torch.full((R, KSEL * KP + KP), -7, dtype=torch.int32, device=DEV)
    gi._pooled_select(
        q,
        w,
        ape,
        cache,
        bt,
        row_req,
        visible,
        logits,
        max_pools,
        BLOCK_SIZE,
        0.1,
        KSEL,
        topk,
        KP,
        by_request=by_request,
    )
    return logits, topk


@pytest.mark.parametrize(
    "query_lens,cached",
    [
        ([1000, 700, 1300, 5, 1000], [0, 0, 0, 0, 0]),
        ([500, 1000, 3, 900], [1500, 0, 2000, 64]),  # cached prefixes
        ([1], [0]),  # a single one-token chunk
        ([7, 9], [3, 2500]),  # tiny queries, one over a long context
    ],
)
def test_matches_per_row_kernel(query_lens, cached, prefill_tiles):
    q, w, ape, cache, bt, row_req, visible, max_pools = _synth(query_lens, cached)
    ref_logits, ref_topk = _select(
        False, q, w, ape, cache, bt, row_req, visible, max_pools
    )
    new_logits, new_topk = _select(
        True, q, w, ape, cache, bt, row_req, visible, max_pools
    )
    torch.cuda.synchronize()
    n_pools = visible // KP
    R = q.shape[0]
    p = torch.arange(max_pools, device=DEV)
    written = p[None, :] < n_pools[:, None]  # both leave the rest unwritten
    torch.testing.assert_close(
        new_logits[written], ref_logits[written], atol=1e-5, rtol=1e-5
    )
    assert not torch.isnan(new_logits[written]).any()
    # Same tokens per row. The two kernels accumulate in a different order,
    # so pools whose logits sit within float noise of the k-th largest may
    # legitimately swap in and out when a row has more pools than KSEL.
    ref_sets = ref_topk.sort(dim=1).values
    new_sets = new_topk.sort(dim=1).values
    for r in (ref_sets != new_sets).any(dim=1).nonzero().flatten().tolist():
        a = set(ref_topk[r][ref_topk[r] >= 0].tolist())
        b = set(new_topk[r][new_topk[r] >= 0].tolist())
        assert len(a) == len(b), r
        np_r = int(n_pools[r])
        assert np_r > KSEL, (r, sorted(a ^ b))
        kth = ref_logits[r, :np_r].topk(KSEL).values[-1]
        for tok in a ^ b:
            assert abs(float(ref_logits[r, tok // KP]) - float(kth)) < 1e-4, r
    for r in range(0, R, max(1, R // 16)):
        valid = ref_topk[r][ref_topk[r] >= 0]
        assert valid.numel() == min(int(n_pools[r]), KSEL) * KP + (
            int(visible[r]) - int(n_pools[r]) * KP
        )


@pytest.mark.parametrize("by_request", [False, True])
def test_tail_tokens_survive(by_request, prefill_tiles):
    """The incomplete tail pool's tokens (including the query's own token,
    three rows in four) sit right after the selected pools in every row,
    on every run: the expansion loop used to write -1 over those columns
    from other threads of the program and raced the tail store."""
    q, w, ape, cache, bt, row_req, visible, max_pools = _synth(
        [500, 1000, 3, 900], [1500, 0, 2000, 64]
    )
    n_pools = visible // KP
    n_sel = torch.clamp(n_pools, max=KSEL)
    tail_count = visible - n_pools * KP
    m = torch.arange(KP, device=DEV)
    cols = (n_sel * KP)[:, None] + m[None, :]
    want = torch.where(
        m[None, :] < tail_count[:, None], (n_pools * KP)[:, None] + m[None, :], -1
    )
    for _ in range(5):
        _, topk = _select(by_request, q, w, ape, cache, bt, row_req, visible, max_pools)
        torch.cuda.synchronize()
        got = torch.gather(topk, 1, cols.long())
        bad = (got != want).any(dim=1).nonzero().flatten()
        assert bad.numel() == 0, bad[:8].tolist()
        # and nothing valid after the tail
        after = topk[:, :].clone()
        after[torch.arange(after.shape[0], device=DEV)[:, None], cols.long()] = -1
        past = torch.arange(after.shape[1], device=DEV)[None, :] >= cols[:, :1]
        assert (after[past] == -1).all()


def test_prefill_row_req_uses_query_rows():
    """Request 0 has a cached prefix (more KV tokens than query rows), so
    the chunk's KV-indexed token_to_seq read by query row would put the
    later requests' rows on the wrong block-table row."""
    seq_lens = torch.tensor([1500, 1000, 1200], dtype=torch.int32)
    query_lens = torch.tensor([500, 1000, 1200], dtype=torch.int32)
    qsl_cpu = torch.zeros(4, dtype=torch.int32)
    qsl_cpu[1:] = torch.cumsum(query_lens, 0)
    chunk = build_prefill_chunk_metadata(
        0,
        3,
        qsl_cpu.to(DEV),
        qsl_cpu,
        seq_lens.to(DEV),
        seq_lens.to(DEV),
        seq_lens,
        torch.zeros((3, 4), dtype=torch.int32, device=DEV),
        1,
    )
    R = chunk.token_end - chunk.token_start
    assert int(query_lens.sum()) == R
    expect = torch.repeat_interleave(torch.arange(3, dtype=torch.int32), query_lens).to(
        DEV
    )
    assert torch.equal(gi._prefill_row_req(chunk, R), expect)
    assert not torch.equal(chunk.token_to_seq[:R].to(torch.int32), expect)
    visible = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
    assert int(visible[0]) == 1001 and int(visible[499]) == 1500
    assert int(visible[500]) == 1 and int(visible[R - 1]) == 1200


def test_tp_shard_dispatch_uses_fork_chunk_metadata(monkeypatch):
    monkeypatch.setattr(gi, "_TP_PREFILL_SHARD", True)
    monkeypatch.setattr(gi, "_PREFILL_MATMUL", True)
    chunks = [SimpleNamespace(token_start=3, token_end=3003, max_seq_len=32768)]
    metadata = DeepseekV32IndexerPrefillMetadata(chunks=chunks)
    assert not hasattr(metadata, "max_prefill_seq_len")
    assert gi._prefill_row_sharding_enabled(metadata.chunks)
    assert not gi._prefill_row_sharding_enabled([])
    chunks[0].token_end = 2050
    assert not gi._prefill_row_sharding_enabled(metadata.chunks)
    chunks[0].token_end = 3003
    chunks[0].max_seq_len = 32767
    assert not gi._prefill_row_sharding_enabled(metadata.chunks)
    chunks[0].max_seq_len = 32768
    monkeypatch.setattr(gi, "_TP_PREFILL_SHARD", False)
    assert not gi._prefill_row_sharding_enabled(metadata.chunks)


@pytest.mark.parametrize(
    "query_lens,cached", [([2], [0]), ([7, 9, 19], [0, 5000, 33000])]
)
def test_tp_pool_exchange_preserves_rows_tails_and_padding(query_lens, cached):
    """Every TP owner, including empty owners, behind decode and before padding."""
    q, w, ape, cache, bt, req, vis, max_pools = _synth(query_lens, cached, seed=4511)
    rows, leading, trailing = q.shape[0], 3, 5
    q_full = torch.zeros(rows + leading + trailing, H, D, dtype=q.dtype, device=DEV)
    w_full = torch.zeros(rows + leading + trailing, H, dtype=w.dtype, device=DEV)
    q_full[leading : leading + rows] = q
    w_full[leading : leading + rows] = w
    # KV starts identify requests, as in production chunk metadata.
    ks = torch.tensor(
        [
            0,
            *torch.tensor([a + b for a, b in zip(query_lens, cached)])
            .cumsum(0)
            .tolist(),
        ],
        device=DEV,
        dtype=torch.int32,
    )[req.long()]
    chunks = []
    for start in range(0, rows, 11):
        end = min(start + 11, rows)
        # Each subchunk retains the request table used by its row labels.
        # Include the prior request's start in the first row so row_req maps
        # to the sliced table's zero-based request IDs, as real metadata does.
        req_first = int(req[start])
        req_last = int(req[end - 1])
        chunks.append(
            SimpleNamespace(
                token_start=leading + start,
                token_end=leading + end,
                cu_seqlen_ks=ks[start:end],
                cu_seqlen_ke=ks[start:end] + vis[start:end],
                block_table=bt[req_first : req_last + 1],
                max_seq_len=max_pools * KP,
            )
        )
    expected = torch.full(
        (rows + leading + trailing, 2051), -7, device=DEV, dtype=torch.int32
    )
    ids = torch.empty((rows, KSEL), device=DEV, dtype=torch.int32)
    for c in chunks:
        lo, hi = c.token_start, c.token_end
        row_req = gi._prefill_row_req(c, hi - lo)
        logits = torch.empty(hi - lo, max_pools, device=DEV)
        sel = gi._pooled_topk(
            q_full[lo:hi],
            w_full[lo:hi],
            ape,
            cache,
            c.block_table,
            row_req,
            c.cu_seqlen_ke - c.cu_seqlen_ks,
            logits,
            max_pools,
            BLOCK_SIZE,
            0.1,
            KSEL,
            KP,
            True,
        )
        ids[lo - leading : hi - leading] = sel
        gi._expand_topk_kernel[(hi - lo,)](
            sel,
            c.cu_seqlen_ke - c.cu_seqlen_ks,
            expected[lo:hi],
            KSEL,
            KP=KP,
            KSEL=KSEL,
            OUT_W=2051,
            BLOCK_S=64,
        )
    owned = (rows + 3) // 4
    for rank in range(4):
        calls = []

        def exchange(local, dim, rank=rank, calls=calls):
            calls.append(1)
            assert dim == 0 and local.shape == (owned, KSEL) and local.is_contiguous()
            start, stop = rank * owned, min((rank + 1) * owned, rows)
            valid = max(0, stop - start)
            torch.testing.assert_close(
                local[:valid].sort(1).values,
                ids[start:stop].sort(1).values,
                rtol=0,
                atol=0,
            )
            assert (local[valid:] == -1).all()
            gathered = torch.full((4 * owned, KSEL), -1, device=DEV, dtype=torch.int32)
            gathered[:rows] = ids
            gathered[start : start + owned] = local
            return gathered

        group = SimpleNamespace(world_size=4, rank_in_group=rank, all_gather=exchange)
        out = torch.full_like(expected, -7)
        gi._pooled_prefill_tp_select(
            q_full, w_full, ape, cache, chunks, out, BLOCK_SIZE, 0.1, KSEL, KP, group
        )
        assert len(calls) == 1
        torch.testing.assert_close(
            out.sort(1).values, expected.sort(1).values, rtol=0, atol=0
        )
        assert (out[:leading] == -7).all() and (out[-trailing:] == -7).all()


def test_sm120_tile_changed_input_graphs_match_original_selection(monkeypatch):
    data = _synth([17, 5, 33], [0, 7000, 33000], seed=921)
    q, weights, ape, cache, _, _, visible, max_pools = data
    graphs, outputs = [], []
    for rt, pt in ((8, 64), (2, 128)):
        monkeypatch.setattr(gi, "_ROW_TILE", rt)
        monkeypatch.setattr(gi, "_POOL_TILE", pt)
        _select(True, *data)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = _select(True, *data)
        graphs.append(graph)
        outputs.append(output)
    valid = torch.arange(max_pools, device=DEV)[None, :] < (visible // KP)[:, None]
    for replay in range(3):
        generator = torch.Generator(device=DEV).manual_seed(1251 + replay)
        q.normal_(generator=generator)
        cache.normal_(generator=generator)
        ape.normal_(generator=generator)
        weights.uniform_(0, H**-0.5, generator=generator)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        reference, candidate = outputs
        torch.testing.assert_close(
            reference[0][valid], candidate[0][valid], atol=1e-5, rtol=1e-5
        )
        assert torch.equal(reference[1].sort(1).values, candidate[1].sort(1).values)
