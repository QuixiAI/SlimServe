"""qgemm_mma_q8_0: correctness against a torch fp32 dequant reference and a
DRAM-resident timing against the SM route, at the GLM-5.3-Flash Q2 q8_0
shapes for M in 1..32. Run with the server down. Usage: mma_check.py"""

import time

import torch

from vllm.quixicore.ops import quixicore_ops as qc

dev = "mps"
g = torch.Generator().manual_seed(3)


def make(N, K):
    w = torch.randint(0, 256, (N, K // 32 * 34), generator=g, dtype=torch.uint8)
    blk = w.view(N, K // 32, 34)
    # finite scales: half in [~1e-3, ~4e-3]
    blk[..., 0] = torch.randint(0, 256, blk[..., 0].shape, generator=g, dtype=torch.uint8)
    blk[..., 1] = 0x14 + torch.randint(0, 4, blk[..., 1].shape, generator=g, dtype=torch.uint8)
    return w


def dequant(w, N, K):
    blk = w.view(N, K // 32, 34)
    d = blk[..., :2].contiguous().view(torch.float16).float()  # (N, nb, 1)
    q = blk[..., 2:].contiguous().view(torch.int8).float()      # (N, nb, 32)
    return (q * d).reshape(N, K)


def bench(fn, n, warm):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e6


SHAPES = [("kda_v", 8192, 4096), ("kda_output", 4096, 8192), ("attn_output", 4096, 16384),
          ("attn_q_b", 16384, 1536), ("q_a|kv_a", 2112, 4096), ("shexp gate|up", 4096, 4096),
          ("shexp down", 4096, 2048)]


def make_q4k(N, K):
    w = torch.randint(0, 256, (N, K // 256 * 144), generator=g, dtype=torch.uint8)
    blk = w.view(N, K // 256, 144)
    blk[..., 1] = 0x14 + torch.randint(0, 4, blk[..., 1].shape, generator=g, dtype=torch.uint8)
    blk[..., 3] = 0x10 + torch.randint(0, 4, blk[..., 3].shape, generator=g, dtype=torch.uint8)
    return w

# ---- correctness (one weight copy per shape) ----
worst = 0.0
for name, N, K in SHAPES[:3] + SHAPES[4:5]:
    if N % 32:
        print(f"{name}: N={N} not a multiple of 32, mma ineligible")
        continue
    w = make(N, K)
    wd = dequant(w, N, K)
    wm = w.to(dev)
    for M in (1, 5, 8, 9, 16, 17, 24, 32):
        x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16)
        ref = x.float() @ wd.T
        y = qc.ggml_mul_mat_mma(wm, x.to(dev), 8, N).cpu().float()
        # reference through the same half operands the kernel uses
        ref_h = x.to(torch.float16).float() @ wd.to(torch.float16).float().T
        err = (y - ref_h).abs().max().item() / (ref_h.abs().max().item() + 1e-9)
        err_f = (y - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        worst = max(worst, err)
        # strided out
        wide = torch.empty((M, N + 96), dtype=torch.bfloat16, device=dev)
        qc.ggml_mul_mat_mma(wm, x.to(dev), 8, N, out=wide[:, 96:])
        ok_s = torch.equal(wide[:, 96:].contiguous().cpu(), y.to(torch.bfloat16))
        print(f"{name:14s} M={M:2d}: rel-max-err vs half-ref {err:.2e}  vs fp32-ref {err_f:.2e}  strided {'OK' if ok_s else 'DIFF'}")
print(f"worst rel-max-err vs half-operand reference: {worst:.2e}")

# ---- timing, DRAM-resident (q8_0 shapes, then the q4_K KDA q|k) ----
for M in (8, 16, 32):
    for name, N, K, fmt in [(n, N_, K_, 8) for n, N_, K_ in SHAPES] + [("kda q|k Q4_K", 16384, 4096, 12)]:
        if N % 32:
            continue
        nb = N * (K // 32 * 34) if fmt == 8 else N * (K // 256 * 144)
        copies = max(1, int(2**30 // nb))
        ws = [(make(N, K) if fmt == 8 else make_q4k(N, K)).to(dev) for _ in range(copies)]
        x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        rows = {}
        for rname, call in (("mma", lambda w: qc.ggml_mul_mat_mma(w, x, fmt, N)),
                            ("sm", lambda w: qc.ggml_mul_mat_sm(w, x, fmt, N))):
            i = [0]

            def dram():
                call(ws[i[0] % copies])
                i[0] += 1

            us = bench(dram, max(40, 2 * copies), copies + 4)
            rows[rname] = us
        print(f"M={M:2d} {name:14s} N={N:6d} K={K:5d} | mma {rows['mma']:7.1f} us {nb / rows['mma'] / 1e3:4.0f} GB/s | sm {rows['sm']:7.1f} us {nb / rows['sm'] / 1e3:4.0f} GB/s | {rows['sm'] / rows['mma']:.2f}x", flush=True)
        del ws
