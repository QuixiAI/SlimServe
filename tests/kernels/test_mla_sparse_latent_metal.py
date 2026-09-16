# SPDX-License-Identifier: Apache-2.0
"""Parity of the Metal sparse latent decode (``mla_sparse_latent_decode``)
against the torch path in ``metal_mla_sparse.sparse_attend_rows``."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.has("mla_sparse_latent_decode"):
        pytest.skip("mla_sparse_latent_decode not built")
    return quixicore_ops


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "R,H,W,seq", [(1, 64, 2080, 3000), (3, 8, 300, 700), (2, 4, 64, 40)]
)
def test_sparse_latent_decode_matches_torch(dtype, R, H, W, seq):
    from vllm.v1.attention.backends.mla import metal_mla_sparse as M

    qc = _qc()
    torch.manual_seed(0)
    dev = "mps"
    bs = 64
    nblk_req = (seq + bs - 1) // bs
    num_blocks = nblk_req * R + 3
    # serving layout: contiguous [num_blocks, block_size, 512]
    cache = (torch.randn(num_blocks, bs, 512, device=dev) * 0.5).to(dtype)
    q = (torch.randn(R, H, 512, device=dev) * 0.3).to(dtype)
    bt = torch.zeros(R, nblk_req + 2, dtype=torch.int32, device=dev)
    for r in range(R):
        bt[r, :nblk_req] = torch.arange(3 + r * nblk_req, 3 + (r + 1) * nblk_req)
    # indices: random positions < seq, with pads; one row all pad
    idx = torch.randint(0, seq, (R, W), device=dev, dtype=torch.int32)
    idx[:, W // 2 :: 7] = -1
    if R > 1:
        idx[-1] = -1
    scale = 512**-0.5
    # torch reference: run the body with the kernel gate pinned off
    old = M._SPARSE_KERNEL
    M._SPARSE_KERNEL = False
    try:
        ref = M.sparse_attend_rows(q, cache, bt, idx, bs, scale, 512)
    finally:
        M._SPARSE_KERNEL = old
    out = qc.mla_sparse_latent_decode(q, cache, bt, idx, scale)
    torch.mps.synchronize()
    assert torch.isfinite(out.float()).all()
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 2e-2, err
    if R > 1:
        assert out[-1].abs().sum() == 0
    # partition count independence
    out1 = qc.mla_sparse_latent_decode(q, cache, bt, idx, scale, 1)
    torch.mps.synchronize()
    assert (out1.float() - out.float()).abs().max().item() < 2e-2
