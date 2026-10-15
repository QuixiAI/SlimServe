# SPDX-License-Identifier: Apache-2.0
"""Metal GLM-5.3-Flash indexer kernels vs the torch producer
(`_pooled_select_native` with the Metal route pinned off): identical
selected token sets per row, pool logits within fp32 reduction error, and
the paged row insert bit-exact against the torch scatter."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.has("glm5_indexer_pool_logits"):
        pytest.skip("indexer kernels not built")
    return quixicore_ops


def _setup(R, seqs, bs=64, H=32, D=128, dtype=torch.bfloat16):
    torch.manual_seed(0)
    dev = "mps"
    nblk_req = (max(seqs) + bs - 1) // bs
    num_blocks = 1 + R * nblk_req
    cache = (torch.randn(num_blocks, bs, 2 * D, device=dev) * 0.5).to(dtype)
    bt = torch.zeros(R, nblk_req + 1, dtype=torch.int32, device=dev)
    for r in range(R):
        bt[r, :nblk_req] = torch.arange(1 + r * nblk_req, 1 + (r + 1) * nblk_req)
    q = (torch.randn(R, H, D, device=dev) * 0.3).to(dtype)
    w = torch.randn(R, H, device=dev) * (H**-0.5)
    ape = torch.randn(4, D, device=dev) * 0.2
    visible = torch.tensor(seqs, dtype=torch.int32, device=dev)
    row_req = torch.arange(R, dtype=torch.int32, device=dev)
    return cache, bt, q, w, ape, visible, row_req


@pytest.mark.parametrize(
    "seqs,ksel",
    [([1000], 512), ([3, 70, 2300], 512), ([9000, 40], 512), ([600], 8)],
)
def test_indexer_select_matches_torch(seqs, ksel):
    from vllm.model_executor.layers import glm5_next_indexer as GI

    qc = _qc()
    R = len(seqs)
    kp, bs = 4, 64
    cache, bt, q, w, ape, visible, row_req = _setup(R, seqs, bs=bs)
    out_w = ksel * kp + kp - 1
    pool_bound = max(1, max(seqs) // kp)
    ref = torch.full((R, out_w), -1, dtype=torch.int32, device="mps")
    old = GI._METAL_IDX
    GI._METAL_IDX = False
    try:
        GI._pooled_select_native(
            q, w, ape, cache, bt, row_req, visible, bs, 128**-0.5, ksel, ref,
            kp, pool_bound,
        )
    finally:
        GI._METAL_IDX = old
    got = torch.full((R, out_w), -1, dtype=torch.int32, device="mps")
    GI._METAL_IDX = True
    try:
        GI._pooled_select_native(
            q, w, ape, cache, bt, row_req, visible, bs, 128**-0.5, ksel, got,
            kp, pool_bound,
        )
    finally:
        GI._METAL_IDX = old
    torch.mps.synchronize()
    for r in range(R):
        a = sorted(int(x) for x in ref[r].tolist() if x >= 0)
        b = sorted(int(x) for x in got[r].tolist() if x >= 0)
        assert a == b, (r, len(a), len(b), a[:8], b[:8])
        # tail tokens present exactly once, right after the valid pools
        n_pools = seqs[r] // kp
        n_sel = min(n_pools, ksel)
        tail = list(range(n_pools * kp, seqs[r]))
        row = got[r].tolist()
        assert row[n_sel * kp : n_sel * kp + len(tail)] == tail
    # logits vs a torch replica of the pooled scoring (fp32 reduction order)
    logits = qc.glm5_indexer_pool_logits(
        q, w, ape, cache, bt, row_req, visible, pool_bound, bs, 128**-0.5
    )
    pk = GI._pool_keys_native(cache, bt, bs, ape, kp, 0, pool_bound)
    scores = torch.relu(
        torch.einsum("rpd,rhd->rph", pk, q.float()) * (128**-0.5)
    )
    ref_logits = (scores * w[:, None, :]).sum(dim=-1)
    torch.mps.synchronize()
    for r in range(R):
        n = seqs[r] // kp
        if n > 0:
            err = (logits[r, :n] - ref_logits[r, :n]).abs().max().item()
            scale = ref_logits[r, :n].abs().max().item()
            assert err <= 1e-3 * max(1.0, scale), (r, err, scale)
        if n < pool_bound:
            assert torch.all(logits[r, n:] == -math.inf)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("row", [256, 512])
def test_paged_row_insert(dtype, row):
    qc = _qc()
    torch.manual_seed(1)
    dev = "mps"
    bs, nb, T = 64, 7, 11
    cache_a = torch.randn(nb, bs, row, device=dev).to(dtype)
    cache_b = cache_a.clone()
    rows = torch.randn(T, row, device=dev).to(dtype)
    slots = torch.randperm(nb * bs, device=dev)[:T].to(torch.int32)
    slots[3] = -1  # PAD -> null block row 0
    # torch reference (same clamp convention)
    slot = slots.to(torch.int64).clamp_min(0)
    cache_a[torch.div(slot, bs, rounding_mode="floor"), slot % bs] = rows
    qc.paged_row_insert(rows, cache_b, slots)
    torch.mps.synchronize()
    assert torch.equal(cache_a, cache_b)
