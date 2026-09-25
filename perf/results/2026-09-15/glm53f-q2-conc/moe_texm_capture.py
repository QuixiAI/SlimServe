"""Capture (save) / compare (check) the production iq2_xxs texm w13 outputs
across a kernel rebuild: bit-identity oracle for ALU-only rewrites of the
walk. Usage: moe_texm_capture.py save|check <file.pt>"""
import os
import sys

import torch

os.environ.setdefault("VLLM_METAL_MOE_IQ2_TEX", "1")
from vllm.quixicore.ops import quixicore_ops as qc  # noqa: E402

dev = "mps"
mode, path = sys.argv[1], sys.argv[2]
g = torch.Generator().manual_seed(5)
E, K, N, T, TOPK = 24, 1024, 128, 16, 8
w = torch.randint(0, 256, (E, N, K // 256 * 66), generator=g, dtype=torch.uint8)
blk = w.view(E, N, K // 256, 66)
blk[..., 0] = torch.randint(0, 256, blk[..., 0].shape, generator=g, dtype=torch.uint8)
blk[..., 1] = 0x28 + torch.randint(0, 8, blk[..., 1].shape, generator=g, dtype=torch.uint8)
w = w.to(dev)
outs = []
for dtype in (torch.bfloat16, torch.float16):
    x = (torch.randn(T, K, generator=g) * 0.5).to(dtype).to(dev)
    ids = torch.randint(0, E, (T, TOPK), generator=g, dtype=torch.int32).to(dev)
    for clamp in (None, 7.0):
        outs.append(qc.ggml_moe_a8_vec_swiglu(x, w, ids, TOPK, 16, N, T, clamp).clone().cpu())
torch.mps.synchronize()
if mode == "save":
    torch.save(outs, path)
    print("saved", [o.shape for o in outs])
else:
    ref = torch.load(path)
    ok = all(torch.equal(a, b) for a, b in zip(outs, ref))
    print("BIT-IDENTICAL" if ok else "DIFFERS", [(a - b).abs().max().item() for a, b in zip(outs, ref)])
