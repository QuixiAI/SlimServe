"""Sparse MLA decode: per-head kernel vs head-grouped (MQA) configs.

Serving shape: H=64, LATENT=512, W=2048 top-k, block 64, bf16 latent pages
resident in a 4 GiB pool (DRAM-resident rotation). Parity vs the torch
reference at every R. Usage: mla_mqa_bench.py
"""
import time
import torch
from vllm.quixicore.ops import quixicore_ops as qc
from vllm.v1.attention.backends.mla import metal_mla_sparse as M

dev = "mps"
H, L, W, BS = 64, 512, 2048, 64
torch.manual_seed(0)
num_blocks = 4 * 2**30 // (BS * L * 2)          # 4 GiB of pages
cache = (torch.randn(num_blocks, BS, L, device=dev) * 0.5).to(torch.bfloat16)
scale = L ** -0.5


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3


for R in (2, 8, 16, 32):
    seq = 6000
    nblk = (seq + BS - 1) // BS
    q = (torch.randn(R, H, L, device=dev) * 0.3).to(torch.bfloat16)
    bt = torch.zeros(R, nblk + 1, dtype=torch.int32, device=dev)
    for r in range(R):
        base = (r * nblk * 7) % (num_blocks - nblk)
        bt[r, :nblk] = torch.arange(base, base + nblk)
    idx = torch.randint(0, seq, (R, W), device=dev, dtype=torch.int32)
    idx[:, W - 100 :] = -1
    tlen = torch.full((R,), W - 100, dtype=torch.int32, device=dev)
    old = M._SPARSE_KERNEL
    M._SPARSE_KERNEL = False
    ref = M.sparse_attend_rows(q, cache, bt, idx, BS, scale, L).float()
    M._SPARSE_KERNEL = old
    res = {}
    for cfg in (0, 3, 5, 6):
        out = qc.mla_sparse_latent_decode(q, cache, bt, idx, scale, 0, tlen, cfg)
        torch.mps.synchronize()
        err = (out.float() - ref).abs().max().item()
        ms = bench(lambda: qc.mla_sparse_latent_decode(q, cache, bt, idx, scale, 0, tlen, cfg))
        res[cfg] = (ms, err)
    line = f"R={R:2d} " + " | ".join(
        f"cfg{c} {ms:6.3f} ms err {e:.1e}" for c, (ms, e) in res.items()
    )
    print(line, flush=True)
