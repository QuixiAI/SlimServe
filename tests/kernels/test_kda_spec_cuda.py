# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA speculative-row KDA kernels (kda_spec_fwd / kda_spec_commit) against
the vendored Triton forward and its store-free + replay-commit variants."""

import pytest
import torch

qc = pytest.importorskip("vllm._quixicore_C")
from vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent import (  # noqa: E402
    fused_recurrent_kda,
    fused_recurrent_kda_commit,
)

H, K, V = 16, 128, 128
LB = -5.0


def _inputs(R, T, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    N = R * T
    q = torch.randn(1, N, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, N, H, V, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, N, H, K, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(1, N, H, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(0, N + 1, T, device=dev, dtype=torch.int32)
    sidx = (1 + torch.arange(R, device=dev)[:, None] * T + torch.arange(T, device=dev)[None, :]).to(torch.int32).contiguous()
    prev = torch.randint(1, T + 1, (R,), device=dev, dtype=torch.int32)
    new = torch.randint(1, T + 1, (R,), device=dev, dtype=torch.int32)
    brow = torch.full((R,), -1, device=dev, dtype=torch.int32)
    brow[::3] = torch.randint(0, T, (len(brow[::3]),), device=dev, dtype=torch.int32)
    A_log = torch.randn(H, device=dev) * 0.1
    dt_bias = torch.randn(H * K, device=dev) * 0.1
    base = torch.randn(R * (T + 1) + 8, H, V, K, device=dev) * 0.1
    return q, k, v, g, beta, cu, sidx, prev, new, brow, A_log, dt_bias, base


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("R,T", [(1, 4), (8, 4), (32, 4), (5, 3), (4, 8)])
def test_cuda_spec_forward_matches_triton(R, T):
    q, k, v, g, beta, cu, sidx, prev, new, brow, A_log, dt_bias, base = _inputs(R, T, R * 10 + T)
    st_ref = base.clone()
    out_ref, _ = fused_recurrent_kda(q=q, k=k, v=v, raw_g=g, raw_beta=beta, A_log=A_log, dt_bias=dt_bias,
                                     lower_bound=LB, initial_state=st_ref, cu_seqlens=cu,
                                     ssm_state_indices=sidx, num_accepted_tokens=prev)
    st = base.clone()
    out = qc.kda_spec_fwd(q, k, v, g, beta, A_log, dt_bias, st, cu, sidx, prev, K**-0.5, LB, True, True, None)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(st, st_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("R,T", [(8, 4), (32, 4), (4, 8)])
def test_store_free_forward_plus_commit_matches_per_row_stores(R, T):
    """Deferred commit contract: the forward stores nothing; the commit
    replays the accepted rows and leaves the committed column (and the
    boundary column) equal to what the per-row-store forward wrote."""
    q, k, v, g, beta, cu, sidx, prev, new, brow, A_log, dt_bias, base = _inputs(R, T, 7 * R + T)
    st_ref = base.clone()
    out_ref, _ = fused_recurrent_kda(q=q, k=k, v=v, raw_g=g, raw_beta=beta, A_log=A_log, dt_bias=dt_bias,
                                     lower_bound=LB, initial_state=st_ref, cu_seqlens=cu,
                                     ssm_state_indices=sidx, num_accepted_tokens=prev)
    for name in ("cuda", "triton"):
        st = base.clone()
        if name == "cuda":
            out = qc.kda_spec_fwd(q, k, v, g, beta, A_log, dt_bias, st, cu, sidx, prev, K**-0.5, LB, True, False, None)
            torch.cuda.synchronize()
            assert torch.equal(st, base)
            qc.kda_spec_commit(k, v, g, beta, A_log, dt_bias, st, cu, sidx, prev, new, brow, LB, True)
        else:
            out, _ = fused_recurrent_kda(q=q, k=k, v=v, raw_g=g, raw_beta=beta, A_log=A_log, dt_bias=dt_bias,
                                         lower_bound=LB, initial_state=st, cu_seqlens=cu, ssm_state_indices=sidx,
                                         num_accepted_tokens=prev, store_states=False)
            torch.cuda.synchronize()
            assert torch.equal(st, base)
            fused_recurrent_kda_commit(k, v, g, beta, A_log, dt_bias, LB, st, cu, sidx, prev, new, brow)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-3, atol=2e-3)
        for r in range(R):
            cols = [int(new[r]) - 1]
            if 0 <= int(brow[r]) < int(new[r]):
                cols.append(int(brow[r]))
            for t in cols:
                torch.testing.assert_close(st[int(sidx[r, t])], st_ref[int(sidx[r, t])], rtol=1e-4, atol=1e-5)
