# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 sparse MLA decode (quixicore mla_decode_fp8_v, NFP8 == 0):
partitioned vs unpartitioned launches and both bf16 geometries against a
float32 torch reference. Guards the lane-parallel vectorized row path
(VECBF16) added 2026-09-03."""

import math

import pytest
import torch

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore import quixicore_ops as qc  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda"
BS, N_BLOCKS, MAX_TOPK = 64, 64, 2080
LENS = [0, 1, 37, 1000, 2048]


def _inputs(width, heads):
    torch.manual_seed(0)
    B = len(LENS)
    kv = (torch.randn(N_BLOCKS, BS, width, device=DEV) * 0.5).to(torch.bfloat16)
    q = (torch.randn(B, heads, width, device=DEV) * 0.2).to(torch.bfloat16)
    bt = torch.arange(N_BLOCKS, device=DEV, dtype=torch.int32).repeat(B, 1)
    idx = torch.full((B, MAX_TOPK), -1, device=DEV, dtype=torch.int32)
    for b, L in enumerate(LENS):
        idx[b, :L] = torch.randperm(N_BLOCKS * BS, device=DEV)[:L].to(torch.int32)
    # interleaved -1 holes on one row: the kernel must skip, not stop
    idx[3, 5::7] = -1
    tlen = torch.tensor(LENS, device=DEV, dtype=torch.int32)
    return q, kv, bt, idx, tlen


def _reference(q, kv, idx, value_width):
    width = kv.shape[-1]
    scale = 1.0 / math.sqrt(width)
    outs = []
    for b in range(q.shape[0]):
        toks = idx[b][idx[b] >= 0].long()
        if toks.numel() == 0:
            outs.append(torch.zeros(q.shape[1], value_width, device=DEV))
            continue
        rows = kv.reshape(-1, width)[toks].float()
        s = (q[b].float() @ rows.T) * scale
        outs.append(torch.softmax(s, dim=-1) @ rows[:, :value_width])
    return torch.stack(outs)


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("partition_size", [0, 256, 128])
def test_nope_512_matches_reference(heads, partition_size):
    q, kv, bt, idx, tlen = _inputs(512, heads)
    out = qc.mla_decode_bf16_sparse_nope(
        q, kv.reshape(-1), bt, idx, tlen, BS, 1.0 / math.sqrt(512), partition_size
    )
    ref = _reference(q, kv, idx, 512)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 5e-3, err


def test_nope_partitioned_matches_unpartitioned():
    q, kv, bt, idx, tlen = _inputs(512, 8)
    scale = 1.0 / math.sqrt(512)
    a = qc.mla_decode_bf16_sparse_nope(q, kv.reshape(-1), bt, idx, tlen, BS, scale, 0)
    b = qc.mla_decode_bf16_sparse_nope(q, kv.reshape(-1), bt, idx, tlen, BS, scale, 128)
    assert (a.float() - b.float()).abs().max().item() < 1e-3


@pytest.mark.parametrize("heads", [8, 16])
def test_glm_576_matches_reference(heads):
    q, kv, bt, idx, tlen = _inputs(576, heads)
    out = qc.mla_decode_bf16_sparse_glm(
        q, kv.reshape(-1), bt, idx, tlen, BS, 1.0 / math.sqrt(576)
    )
    ref = _reference(q, kv, idx, 512)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 5e-3, err
