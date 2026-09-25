"""kda_step (spec mode) at the c=16 serving shape: 16 requests x 2 tokens,
H=64, D=128, K=1 (num_spec 2), rows-per-simdgroup 1/2/4/8 via
VLLM_QC_KDA_SPEC_ROWS (read per call). Bit-identity is asserted; the
pool is 32 MiB per slot x 33 slots so the state traffic is DRAM-resident."""
import os
import time

import torch

from vllm.quixicore.ops import quixicore_ops as qc

DEV = "mps"
torch.manual_seed(0)
H, D, KS, R, num_spec = 64, 128, 4, 16, 2
lens = [2] * R
T = sum(lens)
C = 3 * H * D
L = KS - 1 + num_spec
slots = 1 + R * num_spec
conv_pool = (torch.randn(slots, C, L) * 0.5).to(torch.bfloat16).to(DEV)
ssm_pool = (torch.randn(slots, H, D, D) * 0.1).float().to(DEV)
table = torch.zeros(R, num_spec, dtype=torch.int32)
for r in range(R):
    table[r] = torch.arange(1 + r * num_spec, 1 + (r + 1) * num_spec)
table = table.to(DEV)
num_accepted = torch.tensor([1, 2] * (R // 2), dtype=torch.int32, device=DEV)
cu = torch.tensor([0] + [int(v) for v in torch.tensor(lens).cumsum(0)], dtype=torch.int32).to(DEV)
mixed_qkv = (torch.randn(T, C) * 0.7).to(torch.bfloat16).to(DEV)
g1 = (torch.randn(T, H * D) * 0.5).to(torch.bfloat16).to(DEV)
beta = torch.randn(T, H).to(torch.bfloat16).to(DEV)
g2 = torch.randn(T, H * D).to(torch.bfloat16).to(DEV)
conv_w = (torch.randn(C, KS) * 0.3).float().to(DEV)
A_log = torch.randn(H).float().to(DEV)
dt_bias = (torch.randn(H * D) * 0.2).float().to(DEV)
norm_w = (torch.rand(D) + 0.5).to(torch.bfloat16).to(DEV)
lb, eps, l2_eps, scale = -5.0, 1e-5, 1e-6, D**-0.5


def step():
    return qc.kda_step(
        mixed_qkv, g1, beta, conv_w, conv_pool, ssm_pool, cu,
        table[:, 0].contiguous(), A_log, dt_bias, lb, True, norm_w, g2, eps,
        scale, l2_eps, slot_table=table, num_accepted=num_accepted,
    )


ref = None
for rows in (1, 2, 4, 8, 1):
    os.environ["VLLM_QC_KDA_SPEC_ROWS"] = str(rows)
    for _ in range(8):
        step()
    torch.mps.synchronize()
    out = step().clone()
    torch.mps.synchronize()
    if ref is None:
        ref = out
    same = torch.equal(out.cpu(), ref.cpu())
    t = time.perf_counter()
    n = 40
    for _ in range(n):
        step()
    torch.mps.synchronize()
    us = (time.perf_counter() - t) / n * 1e6
    mb = R * H * D * D * 4 * 3 / 1e6
    print(f"rows={rows} {us:7.1f} us  {mb / us * 1e3:5.0f} GB/s (state read + 2 ckpt writes)  bit-identical={same}", flush=True)
