# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuixiCore gfx942 paged lightning-indexer logits vs a torch reference.

The kernel exists because AITER's paged MQA-logits kernel addresses the
indexer cache with 32-bit buffer-load offsets and returns garbage past a
2 GiB byte offset on the packed cross-layer slab (GLM-5.2/MI300X, 2026-09).
Every case here uses the served slab's real block stride and places the
request's blocks on both sides of that boundary.
"""

import pytest
import torch

qc = pytest.importorskip("vllm._quixicore_C")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and hasattr(qc, "mqa_logits_paged_gfx942")),
    reason="requires the ROCm QuixiCore paged MQA logits op",
)

H, D, BS, TOPK = 32, 128, 64, 2048
STRIDE = 6081792  # bytes per packed block on the served GLM-5.2 slab


@pytest.mark.parametrize(
    "first_block,ctx,rows",
    [(3, 4562, 4), (352, 4562, 4), (360, 4562, 4), (700, 3000, 4), (700, 300, 2)],
)
def test_paged_logits_match_reference_across_2gib(first_block, ctx, rows):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_k_quant_cache_triton,
        fp8_mqa_logits_torch,
        indexer_k_quant_and_cache_triton,
    )

    torch.manual_seed(first_block)
    dev, fp8 = "cuda", torch.float8_e4m3fnuz
    nb = first_block + (ctx + BS - 1) // BS + 1
    need = nb * STRIDE + (1 << 30)
    free, _ = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(f"needs {need >> 30} GiB free on the device (a server may be up)")
    buf = torch.zeros(nb * STRIDE, dtype=torch.uint8, device=dev)
    cache = torch.as_strided(buf, (nb, BS, D + 4), (STRIDE, D + 4, 1))
    nblk = (ctx + BS - 1) // BS
    bt = torch.arange(first_block, first_block + nblk, dtype=torch.int32, device=dev)
    slots = torch.tensor(
        [int(bt[p // BS]) * BS + p % BS for p in range(ctx)],
        dtype=torch.int64,
        device=dev,
    )
    indexer_k_quant_and_cache_triton(
        torch.randn(ctx, D, device=dev, dtype=torch.bfloat16), cache, slots, 128, None
    )
    q = (torch.randn(rows, H, D, device=dev) * 0.3).to(fp8)
    w = torch.rand(rows, H, device=dev)
    seq = torch.tensor(
        [ctx - (rows - 1 - j) for j in range(rows)], dtype=torch.int32, device=dev
    )
    block_table = torch.zeros(rows, 128, dtype=torch.int32, device=dev)
    block_table[:, :nblk] = bt
    out = torch.full((rows, 8192), float("nan"), device=dev)
    flat = cache.view(nb, -1)
    qc.mqa_logits_paged_gfx942(
        q,
        flat[:, : BS * D],
        flat[:, BS * D :].view(torch.float32),
        w,
        seq,
        block_table,
        out,
        BS,
        16,
        16,
    )
    kf = torch.empty(ctx, D, dtype=fp8, device=dev)
    ks = torch.empty(ctx, 4, dtype=torch.uint8, device=dev)
    cp_gather_indexer_k_quant_cache_triton(
        cache,
        kf,
        ks,
        block_table[:1],
        torch.tensor([0, ctx], dtype=torch.int32, device=dev),
        torch.zeros(ctx, dtype=torch.int32, device=dev),
    )
    ref = fp8_mqa_logits_torch(
        q,
        (kf, ks.view(torch.float32)),
        w,
        torch.zeros(rows, dtype=torch.int32, device=dev),
        seq,
    )
    for r in range(rows):
        n = int(seq[r])
        a, b = out[r, :n], ref[r, :n]
        assert not torch.isnan(a).any()
        torch.testing.assert_close(a, b, atol=0.15, rtol=2e-2)
        k = min(TOPK, n)
        mine = set(torch.topk(a, k).indices.tolist())
        refs = set(torch.topk(b, k).indices.tolist())
        assert len(mine & refs) / k > 0.98
