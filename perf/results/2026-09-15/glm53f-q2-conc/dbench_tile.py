"""9..32-row dense candidates at the GLM-5.3-Flash Q2 shapes, DRAM-resident:
  sm    ggml_mul_mat_sm  (today's 9..32 route)
  tile  ggml_mul_mat_a8  (the prefill tile GEMM; M padded up to 32)
Usage: dbench_tile.py [copies_gib]"""

import sys
import time

import torch

from vllm.quixicore.ops import quixicore_ops as qc

dev = "mps"
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
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


def run(name, N, K, t, M):
    nb = N * wbytes(K, t)
    copies = max(1, int(TARGET * 2**30 // nb))
    ws = [make(N, K, t) for _ in range(copies)]
    x = torch.randn(M, K, generator=g).to(torch.bfloat16).to(dev)
    mp = ((M + 31) // 32) * 32
    xp = torch.zeros(mp, K, dtype=torch.bfloat16, device=dev)
    xp[:M] = x
    routes = {
        "sm": lambda w: qc.ggml_mul_mat_sm(w, x, t, N),
        "tile": lambda w: qc.ggml_mul_mat_a8(w, xp, t, N),
    }
    cells = []
    for rname, call in routes.items():
        i = [0]

        def dram():
            call(ws[i[0] % copies])
            i[0] += 1

        try:
            us = bench(dram, max(40, 2 * copies), copies + 4)
            cells.append(f"{rname} {us:7.1f} us {nb / us / 1e3:4.0f} GB/s")
        except Exception as exc:
            cells.append(f"{rname}   n/a ({type(exc).__name__}: {str(exc)[:60]})")
    print(f"{name:20s} M={M:2d} N={N:6d} K={K:5d} | " + " | ".join(cells), flush=True)
    del ws


SHAPES = [
    ("kda q|k Q4_K", 16384, 4096, 12),
    ("kda_v Q8_0", 8192, 4096, 8),
    ("kda_output Q8_0", 4096, 8192, 8),
    ("attn_output Q8_0", 4096, 16384, 8),
    ("attn_q_b Q8_0", 16384, 1536, 8),
    ("shexp gate|up Q8_0", 4096, 4096, 8),
    ("shexp down Q8_0", 4096, 2048, 8),
]
for M in (16, 32):
    for name, N, K, t in SHAPES:
        run(name, N, K, t, M)
