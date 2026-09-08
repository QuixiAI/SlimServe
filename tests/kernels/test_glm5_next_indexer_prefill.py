# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash pooled indexer on a synthetic paged cache with uneven
requests and cached prefixes: the tail tokens of every row, and the
query-row -> block-table-row map of a prefill chunk."""

import pytest
import torch

from vllm.model_executor.layers import glm5_next_indexer as gi
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata

H, D, KP, ROW = 32, 128, 4, gi._ROW_DIM
BLOCK_SIZE = 1088  # the served hybrid block size (a multiple of 64)
KSEL = 512  # index_topk 2048 // index_kpool 4
DEV = "cuda"

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only Triton kernels"
)


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


def _select(q, w, ape, cache, bt, row_req, visible, max_pools):
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
    )
    return logits, topk


def test_tail_tokens_survive():
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
        _, topk = _select(q, w, ape, cache, bt, row_req, visible, max_pools)
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
