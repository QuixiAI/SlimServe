"""Per-row bit-exactness of the widened q8_0 NR walk: for M in 2..32 every
output row must equal the batch-1 NR launch on that row alone (the kernel's
contract), and a strided [M, N] column-slice output must equal the
contiguous one. bf16, K > 512. Run with VLLM_QC_Q8_NR=1.

Every vec result is cloned: the entry returns a ring-buffer slot, and a list
of more results than the ring holds aliases the earliest ones (that bit the
first version of this test as a phantom row-0 mismatch)."""

import torch

from vllm.quixicore.ops import quixicore_ops as qc

dev = "mps"
g = torch.Generator().manual_seed(1)


def make(N, K):
    w = torch.randint(0, 256, (N, K // 32 * 34), generator=g, dtype=torch.uint8)
    blk = w.view(N, K // 32, 34)
    blk[..., 0] = 0
    blk[..., 1] = 0x2C
    return w.to(dev)


bad = 0
for N, K in ((8192, 4096), (4096, 16384), (2112, 4096)):
    w = make(N, K)
    for M in (2, 3, 4, 5, 6, 7, 8):
        x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        y = qc.ggml_mul_mat_vec_a8(w, x, 8, N).clone()
        ref = torch.cat(
            [qc.ggml_mul_mat_vec_a8(w, x[r : r + 1].contiguous(), 8, N).clone() for r in range(M)]
        )
        ok_rows = torch.equal(y.view(torch.int16), ref.view(torch.int16))
        wide = torch.empty((M, N + 64), dtype=x.dtype, device=dev)
        qc.ggml_mul_mat_vec_a8(w, x, 8, N, out=wide[:, 64:])
        ok_strided = torch.equal(wide[:, 64:].contiguous().view(torch.int16), ref.view(torch.int16))
        if not (ok_rows and ok_strided):
            bad += 1
        print(f"N={N:5d} K={K:5d} M={M:2d}: rows {'BIT-EXACT' if ok_rows else 'DIFF'}  strided {'BIT-EXACT' if ok_strided else 'DIFF'}")
print("ALL BIT-EXACT" if bad == 0 else f"{bad} FAILURES")
