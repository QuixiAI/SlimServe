"""qgemm_mma variants (1 = r8k32 original, 2 = r16k32, 3 = r8k64, 4 = r16k64)
at the GLM-5.3-Flash Q2 dense shapes, M in 8/16/24/32, DRAM-resident
rotation; exactness of every variant vs variant 1 (fp32 accumulation order
differs only across K stages, so rel-err ~1e-3 at most). Server down.
Usage: mma_variants.py [copies_gib]"""
import sys
import time

import torch

from vllm.quixicore.ops import quixicore_ops as qc

dev = "mps"
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
FILTER = sys.argv[2] if len(sys.argv) > 2 else ""
ROWB = {8: (32, 34), 12: (256, 144)}
g = torch.Generator().manual_seed(0)


def wbytes(K, t):
    b, s = ROWB[t]
    return K // b * s


def make(N, K, t):
    w = torch.randint(0, 256, (N, wbytes(K, t)), generator=g, dtype=torch.uint8)
    if t == 8:
        blk = w.view(N, K // 32, 34)
        blk[..., 0] = 0x00
        blk[..., 1] = 0x2C
    else:
        blk = w.view(N, K // 256, 144)
        blk[..., 0] = 0x00
        blk[..., 1] = 0x2C
        blk[..., 2] = 0x00
        blk[..., 3] = 0x2C
    return w.to(dev)


def bench(fn, n, warm):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e6


SHAPES = [
    ("kda_v q8", 8192, 4096, 8), ("kda_out q8", 4096, 8192, 8),
    ("attn_out q8", 4096, 16384, 8), ("attn_q_b q8", 16384, 1536, 8),
    ("q_a|kv_a q8", 2112, 4096, 8), ("shexp g|u q8", 4096, 4096, 8),
    ("shexp down q8", 4096, 2048, 8), ("kda q|k q4K", 16384, 4096, 12),
    ("lm_head q8", 154880, 4096, 8),
]
for M in (8, 16, 24, 32):
    for name, N, K, t in SHAPES:
        if FILTER and FILTER not in name:
            continue
        nb = N * wbytes(K, t)
        copies = max(1, int(TARGET * 2**30 // nb))
        ws = [make(N, K, t) for _ in range(copies)]
        x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        ref = qc.ggml_mul_mat_mma(ws[0], x, t, N, None, 1).float()
        torch.mps.synchronize()
        cells = []
        for v in (1, 2, 3, 4, 1):
            y = qc.ggml_mul_mat_mma(ws[0], x, t, N, None, v).float()
            torch.mps.synchronize()
            err = ((y - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
            i = [0]

            def call():
                i[0] = (i[0] + 1) % copies
                qc.ggml_mul_mat_mma(ws[i[0]], x, t, N, None, v)

            us = bench(call, 24, 16)
            gbs = nb / us / 1e3
            cells.append(f"v{v} {us:7.1f}us {gbs:4.0f}GB/s e{err:.0e}")
        print(f"M={M:2d} {name:14s} N={N:6d} K={K:5d} | " + " | ".join(cells), flush=True)
        del ws
