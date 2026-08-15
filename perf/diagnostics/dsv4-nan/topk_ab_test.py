"""Kernel-level A/B correctness test: persistent_topk vs FilteredTopK.

The dispatch threshold is env-controlled (VLLM_DSV4_FILTERED_TOPK_MIN_ROWS):
huge -> always persistent, 0 -> always FilteredTopK. Run this script once per
arm and compare failures. The logits tail beyond each row's length is filled
with hostile garbage (NaN / +inf / huge) to emulate recycled torch.empty
memory, which serving never initializes (clean_logits=False).
"""
import os
import sys
import torch

torch.manual_seed(0)
import vllm._C_stable_libtorch  # noqa: F401

TOPK = 512
STRIDE = 4096  # row stride (serving uses max_model_len; small stride is fine)
ARM = os.environ.get("VLLM_DSV4_FILTERED_TOPK_MIN_ROWS", "default")

failures = 0
cases = 0
for trial in range(20):
    for num_rows in (1, 2, 7, 8, 11, 16, 33, 48, 66, 96):
        g = torch.Generator(device="cuda").manual_seed(trial * 1000 + num_rows)
        lengths = torch.randint(
            TOPK + 1, 3100, (num_rows,), device="cuda", dtype=torch.int32,
            generator=g,
        )
        # Hostile background: what recycled allocator memory can look like.
        logits = torch.full((num_rows, STRIDE), float("nan"), device="cuda")
        logits[:, ::3] = float("inf")
        logits[:, 1::3] = 3.0e38
        for r in range(num_rows):
            L = int(lengths[r])
            logits[r, :L] = torch.randn(L, device="cuda", generator=g)
        out = torch.full((num_rows, TOPK), -7, device="cuda", dtype=torch.int32)
        workspace = torch.zeros(1024 * 1024, device="cuda", dtype=torch.uint8)
        max_seq_len = int(lengths.max())
        torch.ops._C.persistent_topk(
            logits, lengths, out, workspace, TOPK, max_seq_len
        )
        torch.cuda.synchronize()
        for r in range(num_rows):
            cases += 1
            L = int(lengths[r])
            idx = out[r]
            bad_oob = ((idx < 0) | (idx >= L)).sum().item()
            ref = torch.topk(logits[r, :L], TOPK).indices
            got = set(idx.tolist())
            want = set(ref.tolist())
            # Ties make exact match too strict; compare selected VALUES.
            got_vals = sorted(
                logits[r, [i for i in idx.tolist() if 0 <= i < L]].tolist(),
                reverse=True,
            )
            want_vals = sorted(logits[r, ref].tolist(), reverse=True)
            vals_match = (
                len(got_vals) == len(want_vals)
                and all(abs(a - b) < 1e-5 for a, b in zip(got_vals, want_vals))
            )
            if bad_oob or not vals_match:
                failures += 1
                if failures <= 5:
                    print(
                        f"[{ARM}] FAIL trial={trial} rows={num_rows} row={r} "
                        f"len={L} oob={bad_oob} "
                        f"missing={len(want - got)} extra={len(got - want)}"
                    )
print(f"[{ARM}] {failures} failing rows out of {cases}")
sys.exit(1 if failures else 0)
