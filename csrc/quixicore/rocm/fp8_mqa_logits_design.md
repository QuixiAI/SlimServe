# Bitwise-exact fp8 MQA logits on gfx942 — design notes

Target: replace `_fp8_mqa_logits_kernel`
(`vllm/v1/attention/ops/triton_fp8_mqa_logits.py`), the last in-tree Triton
kernel that JIT-compiles during inference on the ROCm GLM-5.2 serving path.

The bar is **bitwise equality with the Triton kernel**, the same bar the other
nine ROCm ports hold. Unlike those, this one is a GEMM, so the bar is not free —
it constrains the instruction and the reduction order. Everything below was
measured on this machine rather than assumed.

## Why bitwise is achievable

`tl.dot(..., input_precision="ieee")` on fp8 operands does **not** lower to
generic fp32 FMA. Dumping the compiled AMDGCN for the real GLM shape
(`M=512, H=64, D=128, N=512`) gives:

```text
MFMA instructions: {'v_mfma_f32_16x16x32_fp8_fp8': 32}
v_fmac_f32: 16
```

That is native fp8 MFMA with fp32 accumulate — a deterministic hardware
instruction reachable from HIP as
`__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8`. Same instruction + same operand
layout + same K order ⇒ identical bits. Triton is not doing anything
numerically privileged here.

For contrast, a *generic* fp32 dot is NOT bitwise-equal: measured over a
64×128×64 case, torch's fp32 matmul differs from `tl.dot` on 1 element in 2048
(max 3.8e-6), and a sequential FMA loop differs too. So the MFMA path is not
merely preferable, it is the only route to bitwise.

## Triton's layout, from the TTGIR

```mlir
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2],
                      instrShape = [16, 16, 32], isTransposed = true}>
tt.dot A:tensor<64x128xf8E4M3FNUZ, dot_op<opIdx=0, parent=#mma, kWidth=8>>
       B:tensor<128x64xf8E4M3FNUZ, dot_op<opIdx=1, ...>>
```

Launch config for the GLM/DSv4 sparse indexer (from `fp8_mqa_logits_gfx942`):
`NUM_HEADS=64, HEAD_SIZE=128, BLOCK_KV=64, num_warps=4, num_stages=1,
matrix_instr_nonkdim` 16 or 32 by `seq_len`.

So per KV tile the output is `[64 heads, 64 kv]` fp32:

- `warpsPerCTA=[2,2]` splits it into four 32×32 quadrants, one per wave.
- Each wave computes 2×2 = four `16x16` MFMA tiles.
- `HEAD_SIZE=128` with `instrShape` K=32 ⇒ **4 K-steps**, accumulating into the
  same registers.
- 4 tiles × 4 K-steps = 16 MFMA per wave; the kernel has two dot sites (main
  loop + masked tail), giving the 32 seen in the ISA dump.

## MFMA fragment layout for `16x16x32` fp8

Confirmed independently by three sources that agree:

- `QuixiCore-ROCm/kernels/quantization/qgemm/variants/rocm_cdna3/tm_qmm_mfma.cuh`
  documents the `16x16x16` f16 form: `A[m = l%16][k = 4*(l/16) + v]`,
  `D[m = 4*(l/16) + v][n = l%16]`.
- `llama.cpp/ggml/src/ggml-cuda/mma.cuh` (`AMD_MFMA_AVAILABLE`) for the
  `16x16x32` i8 form: `get_i = tid%16`, `get_j = 2*(tid/16) + l` over `int`
  elements of 4 bytes ⇒ `k = 8*(tid/16) + 0..7`.
- The CDNA3 ISA doubling of K from 16 to 32 with 8 elements per lane.

Giving, for `v_mfma_f32_16x16x32_fp8_fp8` on a 64-lane wave:

```text

A[M=16,K=32] : lane l, byte v in 0..7 -> A[m = l%16    ][k = 8*(l/16) + v]
B[K=32,N=16] : lane l, byte v in 0..7 -> B[k = 8*(l/16) + v][n = l%16    ]
D[M=16,N=16] : lane l, reg  v in 0..3 -> D[m = 4*(l/16) + v][n = l%16    ]

```

`kWidth=8` in the TTGIR matches the 8 bytes per lane per operand.

## Epilogue, and the part that still needs pinning

After the dot, per KV tile:

```text

scores = dot(q, kv)          # [64 h, 64 n] fp32
scores = scores *kv_scales[n]
scores = max(scores, 0)
scores = scores* w[h]
logits[n] = sum over h of scores    # tl.reduce axis=0, arith.addf

```

The multiplies are elementwise and order-free. **The head-sum is not.** It adds
64 fp32 values per column, and fp32 addition is not associative, so its
tree shape is load-bearing exactly like the sampler's `shfl.bfly` order.

With the layout above, the 64 heads (M) for one column live in:

- 4 accumulator registers per `16x16` tile,
- × 2 M-tiles per wave (the wave's 32 rows),
- × 2 waves (`warpsPerCTA[0] = 2`).

So the reduction is three staged: 8 values intra-lane, then across the four
`l/16` lane groups, then across the 2 M-waves through LDS. Matching bits
requires matching the pairing order at each stage. That order is recoverable
from the AMDGCN — look for the `v_add_f32` sequence with DPP row_shr/row_bcast
modifiers following the `v_mfma` block, and the LDS round-trip after it. The 16
`v_fmac_f32` in the dump belong to this epilogue, not the dot.

**This is the one remaining unknown.** Everything above it is settled.

## Build order

1. Kernel with the MFMA dot and a naive but *documented* epilogue order.
2. Differential test vs Triton (bitwise) — expect the dot to match and the
   epilogue to differ, which isolates the two failure modes. Keep the plain
   fp32 reference in the harness as the third leg: it separates "indexing or
   masking is wrong" from "accumulation order is wrong".
3. Pin the epilogue order against the ISA dump until bitwise.

Deliberately *not* built as a stopgap: the binding, dispatch guard, wrapper and
harness are identical for any epilogue, so nothing here is throwaway. The naive
fp32 path earns its place as a harness oracle, not as a shipped route.

## Non-goals

- The `NORMAL` (non-shuffle) layout and `ue8m0` scales are not on this model's
  path and are left to Triton.
- Matching Triton's autotuned tile choice is not required for correctness, but
  note it selects `BLOCK_KV`/`num_stages` against the 64 KiB LDS budget
  (`_gfx942_default_tile_fits_lds`), so a naive port can easily be *slower*
  than what it replaces. Perf must be measured against the Triton kernel, not
  just against a scalar baseline.
