# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse NoPE-MLA prefill attention (groups of 4 queries over their pool
union, fp8 latent) against the pure-torch reference of the decode tests."""

import math

import pytest
import torch

qc = pytest.importorskip("vllm._quixicore_C")
_NEED = getattr(qc, "mla_sparse_prefill_fp8_smem_bytes", lambda: 0)()
_HAVE = (
    torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).shared_memory_per_block_optin
    if torch.cuda.is_available()
    else 0
)
if _NEED and _HAVE and _HAVE < _NEED:
    pytest.skip(
        f"the fp8 sparse prefill kernel needs {_NEED} B of shared memory per "
        f"block; this device opts in to {_HAVE} (the serving path skips it too)",
        allow_module_level=True,
    )
from tests.kernels.test_quixicore_sparse_mla_bf16 import (  # noqa: E402
    BS,
    DEV,
    MAX_TOPK,
    N_BLOCKS,
    _fp8_inputs,
    _reference,
)


@pytest.mark.parametrize("kv_scale", [1.0, 2.0])
def test_prefill_matches_reference_on_decode_fixture(kv_scale):
    q, data, dequant, bt, idx, tlen = _fp8_inputs(16, kv_scale)
    out = qc.mla_sparse_prefill_fp8(q, data.reshape(-1), bt, idx, tlen, BS, 1.0 / math.sqrt(512), kv_scale)
    ref = _reference(q, dequant, idx, 512)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 5e-3, err


def _pooled_inputs(T, ctx_tokens, kv_scale, seed=0):
    """T queries of one request over a ctx of 4-token pools: each query selects
    512 random pools (+ the 3-token tail), like the compact indexer."""
    torch.manual_seed(seed)
    nb = ctx_tokens // BS
    kv = (torch.randn(nb, BS, 512, device=DEV) * 0.5).to(torch.bfloat16)
    stored = (kv.float() / kv_scale).to(torch.float8_e4m3fn)
    dequant = (stored.float() * kv_scale).to(torch.bfloat16)
    q = (torch.randn(T, 16, 512, device=DEV) * 0.2).to(torch.bfloat16)
    bt = torch.arange(nb, device=DEV, dtype=torch.int32).repeat(T, 1)
    idx = torch.full((T, MAX_TOPK), -1, device=DEV, dtype=torch.int32)
    npools = ctx_tokens // 4
    for t in range(T):
        sel = torch.randperm(npools, device=DEV)[: min(512, npools)].sort().values
        toks = (sel[:, None] * 4 + torch.arange(4, device=DEV)[None, :]).reshape(-1)
        # The compact indexer always adds the 3-token tail; keep only tail
        # tokens not already inside a selected pool (the kernel attends over
        # the SET of selected keys, the torch reference weights duplicates
        # twice).
        tail = torch.arange(ctx_tokens - 3, ctx_tokens, device=DEV)
        tail = tail[~torch.isin(tail, toks)]
        toks = torch.cat([toks, tail]).to(torch.int32)
        idx[t, : toks.numel()] = toks
    tlen = (idx >= 0).sum(dim=1).to(torch.int32)
    return q, stored.view(torch.uint8), dequant, bt, idx, tlen


@pytest.mark.parametrize("T,ctx", [(1, 256), (6, 4096), (64, 8192)])
def test_prefill_pooled_selection(T, ctx):
    q, data, dequant, bt, idx, tlen = _pooled_inputs(T, ctx, 1.5)
    out = qc.mla_sparse_prefill_fp8(q, data.reshape(-1), bt, idx, tlen, BS, 1.0 / math.sqrt(512), 1.5)
    ref = _reference(q, dequant, idx, 512)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 5e-3, err
