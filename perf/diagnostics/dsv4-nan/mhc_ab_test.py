"""Hostile A/B: quixicore dsv4_mhc_pre vs torch reference.

The intra-layer probes place the NaN birth between the embedding and
layer 0's attention input — i.e. inside hc_pre — on 7-row seam batches,
with per-boot expression. Model that: carve the residual from a larger
allocation whose surrounding bytes are hostile (NaN/inf/huge), sweep T,
and diff the kernel against mhc_pre_torch on identical values.
"""
import sys
import torch
from vllm.quixicore import quixicore_ops
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch

torch.manual_seed(7)
H, M = 4096, 4
M3 = M * (M + 2)  # 24: pre 4 + post 4 + comb 16
fn = (torch.randn(M3, M * H, device="cuda") * 0.02).float()
hc_scale = torch.tensor([0.1, 0.1, 0.1], device="cuda")
hc_base = (torch.randn(M3, device="cuda") * 0.01).float()
ARGS = dict(rms_eps=1e-6, pre_eps=1e-4, sinkhorn_eps=1e-4,
            post_multiplier=2.0, sinkhorn_repeat=3)

POISONS = {"nan": float("nan"), "inf": float("inf"), "huge": 3.0e38,
           "zero": 0.0, "neg": -1.0e30}
fails = 0
for poison_name, poison in POISONS.items():
    for T in (1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 16, 33, 66):
        pool = torch.full(((T + 8) * M * H,), poison, device="cuda",
                          dtype=torch.bfloat16)
        residual = pool[: T * M * H].view(T, M, H)
        residual.normal_(0, 1)
        post_k, comb_k, li_k = quixicore_ops.dsv4_mhc_pre(
            residual, fn, hc_scale, hc_base,
            ARGS["rms_eps"], ARGS["pre_eps"], ARGS["sinkhorn_eps"],
            ARGS["post_multiplier"], ARGS["sinkhorn_repeat"], None, 0.0)
        post_r, comb_r, li_r = mhc_pre_torch(
            residual, fn, hc_scale, hc_base,
            ARGS["rms_eps"], ARGS["pre_eps"], ARGS["sinkhorn_eps"],
            ARGS["post_multiplier"], ARGS["sinkhorn_repeat"])
        for name, k, r in (("post", post_k, post_r), ("comb", comb_k, comb_r),
                           ("layer_input", li_k, li_r)):
            kf = k.float().reshape(T, -1)
            rf = r.float().reshape(T, -1)
            k_nan = torch.isnan(kf).any(dim=-1)
            r_nan = torch.isnan(rf).any(dim=-1)
            # NaN mismatch is the bug signal; numeric drift is reported once.
            if bool(k_nan.any()) != bool(r_nan.any()):
                fails += 1
                rows = k_nan.nonzero().flatten().tolist()[:6]
                if fails <= 10:
                    print(f"NAN-FAIL poison={poison_name} T={T} out={name} "
                          f"kernel_nan_rows={rows} ref_nan={bool(r_nan.any())}")
print(f"{fails} failures across {len(POISONS) * 14} cases x 3 outputs")
sys.exit(1 if fails else 0)
