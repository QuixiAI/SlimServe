# QuixiCore-CUDA vendored kernels

CUDA/Ampere serving kernels vendored from
[QuixiAI/QuixiCore-CUDA](https://github.com/QuixiAI/QuixiCore-CUDA)
(ThunderKittens-derived, SM80+), providing the NVIDIA counterpart of the
gfx942 kernel work this fork uses on MI300X.

- Vendored from commit: `08780aaa22cdc2d144b6beacb24df953134b34be` (2026-07-27)
- Source layout mirrors `QuixiCore-CUDA/kernels/` so files diff cleanly
  against upstream; resync by copying the same paths.
- Built as the `vllm._quixicore_C` pybind extension (CUDA builds only); the
  module name is injected via `TORCH_EXTENSION_NAME`, so sources are
  unmodified copies.
- Python-side access and the fallback/stub layer for ops this library does
  not implement yet live in `vllm/quixicore/`.

## Ampere-specific kernels developed here

`quant/q2k_ampere.cuh` (+ `q2k_ampere_test.cu`) is the A100/SM80 Q2_K
weight-only matmul written for this model's routed-expert shapes. It is not an
upstream vendored file — it originates in QuixiCore-CUDA at
`kernels/quant/` and is kept byte-identical in both trees. Rationale, budget
analysis and measured results are in `a100_glm52_design.md` alongside this file.

Status: repack is bit-exact vs the native q2_K formula; GEMV matches an fp64
reference to 0.76% (the int8-activation floor). Throughput on A100, config
`NR=4, QB=2`: 775 / 616 / 350 / 159 GB/s at M = 1 / 2 / 4 / 8, i.e. 44% of the
measured stream ceiling at M=1, falling off with M because the M dimension is
still a scalar `dp4a` loop. Use `NR=2` for a pure M=1 workload (1081 GB/s, 61%).
The IMMA restructure that unblocks M>=2 is design doc section 2.5b.

Self-test:

```bash
cd csrc/quixicore/quant
nvcc q2k_ampere_test.cu -std=c++17 -O3 -arch=sm_80 -o q2k_ampere_test.out
CUDA_VISIBLE_DEVICES=0 ./q2k_ampere_test.out
```

Keep local edits out of the vendored files where possible; put SlimServe-specific
glue in `vllm/quixicore/` or a separate binding file instead.
