"""Is the per-slot routed-MoE GEMV at 32 rows ALU/issue-bound or DRAM-bound?
Same kernel calls as the serving path (w13 IQ2_XXS gate|up + swiglu, w2
Q2_K sum-folded), synthetic weights at the GLM-5.3-Flash Q2 shapes (288
experts, K 4096 / N 4096 merged gate|up; w2 K 2048 / N 4096), 32 rows x
top-8 = 256 slots, with the expert ids arranged so the number of DISTINCT
experts is 256 / ~173 (random) / 32 / 8. If time tracks the slot count
rather than the distinct-expert count, the kernel is bound by per-slot
work (dequant ALU), not by DRAM; grouping rows per expert would then pay.
Two weight copies alternate so the ~1.25 GB tensor is not SLC-resident."""
import os
import time

import torch

os.environ.setdefault("VLLM_METAL_MOE_IQ2_TEX", "1")

from vllm.quixicore.ops import quixicore_ops as qc

dev = "mps"
E, K1, N1 = 288, 4096, 4096      # merged gate|up rows (2048 gate + 2048 up)
K2, N2 = 2048, 4096
T, TOPK = 32, 8
g = torch.Generator().manual_seed(0)


def iq2xxs(E, N, K):
    kb = K // 256 * 66
    w = torch.randint(0, 256, (E, N, kb), generator=g, dtype=torch.uint8)
    blk = w.view(E, N, K // 256, 66)
    blk[..., 0] = 0x00
    blk[..., 1] = 0x2C   # d = half 0x2C00 ~ 0.0156
    return w.to(dev)


def q2k(E, N, K):
    kb = K // 256 * 84
    w = torch.randint(0, 256, (E, N, kb), generator=g, dtype=torch.uint8)
    blk = w.view(E, N, K // 256, 84)
    blk[..., 80] = 0x00; blk[..., 81] = 0x2C   # d
    blk[..., 82] = 0x00; blk[..., 83] = 0x2C   # dmin
    return w.to(dev)


w1 = [iq2xxs(E, N1, K1) for _ in range(2)]
w2 = [q2k(E, N2, K2) for _ in range(2)]
x1 = (torch.randn(T, K1, generator=g) * 0.5).to(torch.bfloat16).to(dev)
x2 = (torch.randn(T * TOPK, K2, generator=g) * 0.5).to(torch.bfloat16).to(dev)
tw = torch.rand(T, TOPK, generator=g).to(torch.float32).to(dev)
out2 = torch.zeros(T, N2, dtype=torch.bfloat16, device=dev)

cases = {}
cases["256 distinct"] = torch.arange(256, dtype=torch.int32).view(T, TOPK)
ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32)
cases[f"random ({len(ids.unique())} distinct)"] = ids
cases["32 distinct"] = (torch.arange(256, dtype=torch.int32) % 32).view(T, TOPK)
cases["8 distinct"] = (torch.arange(256, dtype=torch.int32) % 8).view(T, TOPK)


def bench(fn, n=30, warm=6):
    for _ in range(warm):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3


for name, ids in cases.items():
    ids = ids.to(dev).contiguous()
    i = [0]

    def f13():
        i[0] ^= 1
        qc.ggml_moe_a8_vec_swiglu(x1, w1[i[0]], ids, TOPK, 16, N1, T, None)

    def f13g():
        i[0] ^= 1
        qc.ggml_moe_a8_vec_swiglu(x1, w1[i[0]], ids, TOPK, 16, N1, T, None, group_nb=2)


    def f2():
        i[0] ^= 1
        qc.ggml_moe_a8_vec_sum(x2, w2[i[0]], ids, tw, TOPK, 10, N2, T, out2)

    a = qc.ggml_moe_a8_vec_swiglu(x1, w1[0], ids, TOPK, 16, N1, T, None).clone()
    b = qc.ggml_moe_a8_vec_swiglu(x1, w1[0], ids, TOPK, 16, N1, T, None, group_nb=2).clone()
    torch.mps.synchronize()
    same = torch.equal(a.cpu(), b.cpu())
    print(f"{name:24s} w13 {bench(f13):6.2f} ms  grp {bench(f13g):6.2f} ms (bit-identical={same})  w2sum {bench(f2):6.2f} ms", flush=True)
