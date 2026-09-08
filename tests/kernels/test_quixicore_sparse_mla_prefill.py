# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head-batched sparse MLA prefill kernel (NoPE bf16 latents) against the
decode walk it replaces and against a float32 torch reference: -1 holes,
empty rows, out-of-table entries, strided pages, 16 and 32 heads."""

import math

import pytest
import torch

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore import quixicore_ops as qc  # noqa: E402
from vllm.v1.attention.backends.mla import (  # noqa: E402
    quixicore_mla_sparse_prefill as pf,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda"
BS, N_BLOCKS, MAX_TOPK = 64, 64, 2080
LENS = [0, 1, 37, 1000, 2048, 5, 640]
SCALE = 1.0 / math.sqrt(512)


def _inputs(heads, strided=False, seed=0):
    torch.manual_seed(seed)
    B = len(LENS)
    if strided:
        # blocks strided, slots packed at 512: the packed cross-layer slab
        big = (torch.randn(N_BLOCKS, BS * 512 + 1024, device=DEV) * 0.5).to(
            torch.bfloat16
        )
        kv = big[:, : BS * 512].view(N_BLOCKS, BS, 512)
    else:
        kv = (torch.randn(N_BLOCKS, BS, 512, device=DEV) * 0.5).to(torch.bfloat16)
    q = (torch.randn(B, heads, 512, device=DEV) * 0.2).to(torch.bfloat16)
    bt = torch.arange(N_BLOCKS, device=DEV, dtype=torch.int32).repeat(B, 1)
    idx = torch.full((B, MAX_TOPK), -1, device=DEV, dtype=torch.int32)
    for b, L in enumerate(LENS):
        idx[b, :L] = torch.randperm(N_BLOCKS * BS, device=DEV)[:L].to(torch.int32)
    idx[3, 5::7] = -1  # interleaved holes: skip, do not stop
    idx[6, 640:] = -1
    idx[6, 100] = N_BLOCKS * BS + 5  # past the block table: skipped
    tlen = qc.sparse_topk_tlen(idx)
    return q, kv, bt, idx, tlen


def _reference(q, kv, bt, idx):
    outs = []
    for b in range(q.shape[0]):
        toks = idx[b][(idx[b] >= 0) & (idx[b] < N_BLOCKS * BS)].long()
        if toks.numel() == 0:
            outs.append(torch.zeros(q.shape[1], 512, device=DEV))
            continue
        rows = kv[bt[b][toks // BS].long(), toks % BS].float()
        s = (q[b].float() @ rows.T) * SCALE
        outs.append(torch.softmax(s, dim=-1) @ rows)
    return torch.stack(outs)


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("strided", [False, True])
def test_matches_decode_walk_and_reference(heads, strided):
    q, kv, bt, idx, tlen = _inputs(heads, strided)
    psb = kv.stride(0) * kv.element_size() if strided else 0
    out = pf.sparse_mla_prefill_nope(q, kv, bt, idx, tlen, BS, SCALE, psb)
    walk = qc.mla_decode_bf16_sparse_nope(q, kv, bt, idx, tlen, BS, SCALE, 0, psb)
    torch.cuda.synchronize()
    assert out.shape == (len(LENS), heads, 512) and out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    # one bf16 rounding of the result apart from the walk (P in fp16 there
    # is fp32; the score sums differ in order)
    torch.testing.assert_close(out.float(), walk.float(), atol=1.6e-2, rtol=1e-2)
    torch.testing.assert_close(
        out.float(), _reference(q, kv, bt, idx), atol=2e-2, rtol=2e-2
    )
    assert torch.equal(out[0], torch.zeros_like(out[0]))  # tlen 0 -> 0


def test_many_tokens_smoke():
    """A prefill-sized batch: every token its own program, lists of every
    length; compared to the walk."""
    torch.manual_seed(1)
    B, heads = 3000, 32
    kv = (torch.randn(N_BLOCKS, BS, 512, device=DEV) * 0.5).to(torch.bfloat16)
    q = (torch.randn(B, heads, 512, device=DEV) * 0.2).to(torch.bfloat16)
    bt = torch.arange(N_BLOCKS, device=DEV, dtype=torch.int32).repeat(B, 1)
    lens = torch.randint(0, 2049, (B,), device=DEV)
    pos = torch.arange(MAX_TOPK, device=DEV)[None, :]
    idx = torch.randint(0, N_BLOCKS * BS, (B, MAX_TOPK), device=DEV, dtype=torch.int32)
    idx[pos >= lens[:, None]] = -1
    tlen = qc.sparse_topk_tlen(idx)
    out = pf.sparse_mla_prefill_nope(q, kv, bt, idx, tlen, BS, SCALE)
    walk = qc.mla_decode_bf16_sparse_nope(q, kv, bt, idx, tlen, BS, SCALE, 0, 0)
    torch.testing.assert_close(out.float(), walk.float(), atol=1.6e-2, rtol=1e-2)


def test_supports():
    assert pf.supports(torch.empty(2, 32, 512, dtype=torch.bfloat16))
    assert pf.supports(torch.empty(2, 16, 512, dtype=torch.bfloat16))
    assert not pf.supports(torch.empty(2, 8, 512, dtype=torch.bfloat16))
    assert not pf.supports(torch.empty(2, 32, 576, dtype=torch.bfloat16))
