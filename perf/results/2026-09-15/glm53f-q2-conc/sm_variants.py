"""Sweep the in-tree qgemm_sm variants (raw extension module) at M=16 on the
two largest dense shapes, DRAM-resident, plus the f16probe (same pipeline,
raw half weights, no dequant) to split structure-bound from dequant-bound.
Usage: sm_variants.py"""

import time

import torch

from vllm.quixicore.ops import quixicore_ops as _wrapper

assert _wrapper.is_available()  # sets the metallib path the raw module needs
import vllm._quixicore_C as C  # noqa: E402  (raw module: all m.def entries)

dev = "mps"
g = torch.Generator().manual_seed(0)
ROWB = {8: (32, 34), 12: (256, 144)}


def make(N, K, t):
    b, sz = ROWB[t]
    w = torch.randint(0, 256, (N, K // b * sz), generator=g, dtype=torch.uint8)
    blk = w.view(N, K // b, sz)
    blk[..., 0] = 0
    blk[..., 1] = 0x2C
    if t == 12:
        blk[..., 2] = 0
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


def sweep(name, N, K, t, variants, M=16, target=1.0):
    b, sz = ROWB[t]
    nb = N * (K // b * sz)
    copies = max(1, int(target * 2**30 // nb))
    ws = [make(N, K, t) for _ in range(copies)]
    x = torch.randn(M, K, generator=g).to(torch.bfloat16).to(dev)
    print(f"== {name} N={N} K={K} M={M} ({nb / 1e6:.1f} MB x{copies})")
    for v in variants:
        i = [0]

        def dram():
            C.ggml_mul_mat_sm(ws[i[0] % copies], x, t, N, v)
            i[0] += 1

        try:
            us = bench(dram, max(40, 2 * copies), copies + 4)
            print(f"  variant {v:2d}: {us:7.1f} us  {nb / us / 1e3:4.0f} GB/s  {N * K / us / 1e3:5.0f} Gw/s", flush=True)
        except Exception as exc:
            print(f"  variant {v:2d}: n/a ({str(exc).splitlines()[0][:90]})", flush=True)
    del ws
    # structure-only probe: raw half weights, one copy (SLC), Gweights/s is the comparable unit
    try:
        wh = (torch.randn(N, K, generator=g) * 0.02).to(torch.float16).to(dev)
        xt = x.to(torch.float16).transpose(0, 1).contiguous()
        xp = torch.zeros(K, 32, dtype=torch.float16, device=dev)
        xp[:, :M] = xt
        us = bench(lambda: C.ggml_mul_mat_sm_f16probe(wh, xp, N), 60, 10)
        print(f"  f16probe   : {us:7.1f} us  {N * K / us / 1e3:5.0f} Gw/s  (no dequant; 2 B/weight, SLC-resident)")
    except Exception as exc:
        print(f"  f16probe   : n/a ({str(exc).splitlines()[0][:90]})")


sweep("kda_v Q8_0", 8192, 4096, 8, [2, 8, 9, 10])
sweep("attn_output Q8_0", 4096, 16384, 8, [2, 8, 9, 10])
sweep("kda q|k Q4_K", 16384, 4096, 12, [2, 8, 9, 10, 11, 12, 13])
