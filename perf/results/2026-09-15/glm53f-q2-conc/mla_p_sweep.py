"""Partition sweep for the sparse MLA decode kernels at R=8/32 (legacy cfg0,
MQA cfg3/cfg4). Same DRAM-resident setup as mla_mqa_bench.py."""
import time
import torch
from vllm.quixicore.ops import quixicore_ops as qc

dev = "mps"
H, L, W, BS = 64, 512, 2048, 64
torch.manual_seed(0)
num_blocks = 4 * 2**30 // (BS * L * 2)
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


for R in (8, 32):
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
    for cfg in (0, 3, 4):
        row = []
        for P in (1, 2, 4, 8, 16, 32):
            ms = bench(lambda: qc.mla_sparse_latent_decode(q, cache, bt, idx, scale, P, tlen, cfg))
            row.append(f"P{P:2d} {ms:6.3f}")
        print(f"R={R:2d} cfg{cfg}: " + " | ".join(row), flush=True)
