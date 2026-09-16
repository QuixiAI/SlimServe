# SPDX-License-Identifier: Apache-2.0
"""The hybrid-form MPS SDPA dense prefill (dense_causal_attend_sdpa) against
the torch fp32 absorbed-form reference (dense_causal_attend) on the same
latent rows: per-head decompressed keys + latent values must reproduce the
absorbed q . latent attention up to bf16 rounding."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)
DEV = "mps"


@pytest.mark.parametrize("Q", [5, 64, 300])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_sdpa_matches_absorbed_reference(Q, dtype):
    from vllm.v1.attention.backends.mla.metal_mla_sparse import (
        dense_causal_attend,
        dense_causal_attend_sdpa,
    )

    H, P, L, BS = 8, 256, 512, 64
    g = torch.Generator().manual_seed(Q)
    q_nope = (torch.randn(H, Q, P, generator=g) * 0.5).to(dtype).to(DEV)
    W_UK_T = (torch.randn(H, P, L, generator=g) * 0.05).to(dtype).to(DEV)
    nblk = (Q + BS - 1) // BS
    kv_cache = (torch.randn(nblk + 1, BS, L, generator=g) * 0.5).to(dtype).to(DEV)
    blocks = torch.arange(1, nblk + 1, dtype=torch.int32, device=DEV)
    kv = kv_cache[1:].reshape(-1, L)[:Q]
    scale = P**-0.5
    # Absorbed query for the reference: [Q, H, L]
    q_abs = torch.bmm(q_nope, W_UK_T).transpose(0, 1).contiguous()
    ref = dense_causal_attend(q_abs, kv_cache, blocks, Q, BS, scale, L)
    got = dense_causal_attend_sdpa(q_nope, kv, W_UK_T, scale)
    torch.mps.synchronize()
    assert got.shape == ref.shape == (Q, H, L)
    err = (got.float() - ref.float()).abs().max().item()
    sc = ref.float().abs().max().item() + 1e-6
    tol = 4e-2 if dtype == torch.bfloat16 else 2e-2
    assert err / sc <= tol, (err, sc, Q, dtype)
