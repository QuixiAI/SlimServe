# Getting maximum performance from TP8 on Kimi K3

## The governing constraint

The TP8 regression is an **expert-parallel launch-geometry problem**, not a
collective or generic kernel-launch-count problem. Matched rank-0 kernel traces
account for 17.90 ms of the 18.13 ms c8 TPOT gap:

| kernel | TP6 | TP8 | TP8 − TP6 per step |
| --- | ---: | ---: | ---: |
| IQ2_XXS `w13` vec | 181.08 µs | 411.68 µs | +21.21 ms (92 layers) |
| Q2_K `w2` vec | 106.40 µs | 70.41 µs | -3.31 ms (92 layers) |
| **net** | | | **+17.90 ms** |

The end-to-end baseline gap is 80.98 - 62.85 = 18.13 ms, leaving 0.23 ms of
noise/residual. Collectives were independently measured at 10.08 ms for EP8
and 10.63 ms for TP6, so they are not the differentiator.

## Why TP8 can still win

The excess work is removable without changing the model or sharding. For c8,
`moe_vec_q` launches one workgroup per output row per routed row:

- TP6: 1024 × (8 × 16) = 131,072 useful workgroups per `w13` call.
- TP8: 6144 × (8 × 16) = 786,432 workgroups per call.
- With 1/8 of experts local under EP8, about 688,128 TP8 workgroups (87.5%) see
  expert `-1`, write zero, and return.

An EP-aware token-major kernel can launch 6144 × 8 = 49,152 workgroups and loop
over the 16 routes inside each workgroup, computing only the roughly two local
routes per token. That preserves ample parallelism while removing the empty
route expansion.

Before the fix: `k3-6` 34.0 / 120.4 tok/s, `k3-8` 31.1 / 94.8 (c1 / c8).
TPOT: 28.74 / 62.85 ms vs 31.45 / 80.98 ms.

## Phase 0 — complete: attribute the regression

Attempted and abandoned. Synthetic MoE microbenchmarks produced 888 ms against
an 81 ms step, because these kernels' cost depends on the *routing
distribution*: random `topk_ids` over 112 local experts touch nearly every
expert's weights, while real EP routing touches far fewer. Attribution by
microbenchmark does not work for the MoE.

What did come out of reading the selection logic to build them: at c1 under
TP8+EP, `w2_rows = num_tokens * top_k` is 8 * 16 = exactly 128, landing on the
ROCm `VLLM_GGUF_MOE_VEC_W2` threshold. CUDA defaults that to 0 (always tile).
Tested end to end -- forcing the tile kernel is **worse**: 31.14 -> 28.88 tok/s
at c1, 94.80 -> 93.34 at c8. The ROCm default is correct; vec wins for w2 here.

The profiler now works by preventing PyTorch's bundled rocprofiler libraries
from mixing with the system ROCm 7.2.4 libraries, starting the target with
`ROCP_TOOL_ATTACH=1`, allowing ptrace, and attaching to rank 0 after HIP
initialization. See `HANDOFF.md` for the recipe and trace counts.

Exit gate met: the largest op class is named and the observed gap is accounted
for.

## Phase 1 — complete: EP-aware IQ2_XXS `w13` vector kernel

Keep the public I/O contract unchanged. Add a token-major variant in
`csrc/libtorch_stable/quantization/gguf/moe_vec.cuh` and select it from
`fused_moe.py` only when `expert_map is not None`, `top_k > 1`, and the `w13`
quant is IQ2_XXS.

The kernel should:

1. Launch over `(output row, token)`, not `(output row, token × route)`.
2. Loop over the token's top-k expert ids inside the workgroup.
3. Skip negative expert ids; compute each local route and write its existing
   flat `(token, route, row)` output slot.
4. Write zero to skipped route slots from the token-major workgroup, preserving
   the behavior on both ROCm and CUDA.

Do not apply this to `w2`: `w2` is called with `top_k=1` over the already-routed
rows, and TP8's existing kernel is 3.31 ms/step faster than TP6's.

Before writing the test, answer the project-required design questions:

- Module purpose: compute routed GGUF MoE vector products for quantized experts.
- I/O contract: preserve flat `(token, route, output row)` results, with zeros
  for `-1` expert-map entries and unchanged values for local experts.
- Failure guarded: a token-major implementation must neither overwrite another
  route nor leave non-local routes nonzero while skipping them.
- Cheapest level: extend the nearest GGUF MoE kernel pytest with mixed local and
  `-1` ids; compare the public op against the existing path/oracle. Do not add an
  end-to-end model test for this arithmetic contract.

Exit gate: kernel test passes, coherence 3/3, and the coherence-gated c8 TP8
TPOT recovers most of the measured 17.90 ms penalty without regressing TP6.

Exit gate met. The exact K3 TP8 shape improved from 406.89 to 112.23 µs per
`w13` op (3.63×) with bit-identical output. The model passed the Paris benchmark
gate and a separate 3/3 evaluation (`4`, `Paris`, `The Pacific Ocean`). At c8:

| profile | output tok/s | mean TPOT |
| --- | ---: | ---: |
| TP6 baseline | 120.4 | 62.85 ms |
| TP8 before | 94.8 | 80.98 ms |
| TP8 after | **124.62** | **60.78 ms** |

The change raises TP8 throughput 31.5% and puts it 3.5% ahead of TP6 at c8.

## Phase 2 — remaining wins after TP8 reaches parity

Only after Phase 1, re-profile and rank the remaining costs. Previous candidates
remain fusing dense `mul_mat_vec` quantization and reducing collective latency,
but neither explains the current TP8-vs-TP6 difference.

## What not to do

- **Do not write a HIP all-to-all to beat MORI.** Measured: EP8 collectives
  10.08 ms vs TP6 10.63 ms. Not the differentiator.
- **Do not enable `use_sequence_parallel_moe` for TP8/dp1.** Tried: TPOT 31.45
  → 82.15 ms at c1. The `dp > 1` gate is doing real work.
- **Do not repack `w2` transposed to get a finer MoE split.** Measured: 384 vs
  512 units per rank is a 3–7% difference. Not worth a requantization.
- **Do not force the MMQ tile kernel for w2 on ROCm.** Tried
  `VLLM_GGUF_MOE_VEC_W2=0`: 31.14 -> 28.88 tok/s at c1. The 128-row default is
  measured-correct, unlike the pre-fill it superficially resembles.
- **Do not trust `--ignore-eos` throughput alone.** It reported 154 tok/s on a
  model emitting `!!!!!!!!`. The harness now gates on a known answer; keep it.
