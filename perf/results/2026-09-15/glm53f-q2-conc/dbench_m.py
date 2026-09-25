"""Dense route bench across batch rows at the GLM-5.3-Flash Q2 shapes.

Same DRAM-resident rotation as perf/results/2026-09-14/glm53f-q2-w19-dram/
dbench.py (each shape cycles over >= 1 GiB of copies so the SLC cannot
serve it), extended along M and across the two production routes:
  vec  ggml_mul_mat_vec_a8   (M 2..4 q8_0 -> NR mb kernel; 5..8 -> generic
                              mb walk; q4_K 2/4/8 -> NR mm chunks)
  sm   ggml_mul_mat_sm       (simdgroup-MMA GEMM; the Python gate only
                              admits it at 9..32 rows today)
Run with the server down and VLLM_QC_Q8_NR=1. Usage: dbench_m.py [copies_gib]
"""

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
    routes = {
        "vec": lambda w: qc.ggml_mul_mat_vec_a8(w, x, t, N),
        "sm": lambda w: qc.ggml_mul_mat_sm(w, x, t, N),
    }
    cells = []
    for rname, call in routes.items():
        i = [0]

        def dram():
            call(ws[i[0] % copies])
            i[0] += 1

        try:
            n = max(40, 2 * copies)
            us = bench(dram, n, copies + 4)
            cells.append(f"{rname} {us:7.1f} us {nb / us / 1e3:4.0f} GB/s")
        except Exception as exc:  # route refuses this shape/M: say so
            cells.append(f"{rname}   n/a ({type(exc).__name__})")
    print(f"{name:20s} M={M:2d} N={N:6d} K={K:5d} {nb / 1e6:6.1f} MB | " + " | ".join(cells), flush=True)
    del ws


SHAPES = [
    ("kda q|k Q4_K", 16384, 4096, 12),
    ("kda_v Q8_0", 8192, 4096, 8),
    ("kda_output Q8_0", 4096, 8192, 8),
    ("attn_output Q8_0", 4096, 16384, 8),
    ("attn_q_b Q8_0", 16384, 1536, 8),
    ("attn_q_a|kv_a Q8_0", 2112, 4096, 8),
    ("shexp gate|up Q8_0", 4096, 4096, 8),
    ("shexp down Q8_0", 4096, 2048, 8),
]
for M in (2, 4, 8, 16, 32):
    for name, N, K, t in SHAPES:
        run(name, N, K, t, M)
