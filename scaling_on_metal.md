# How SlimServe scales with concurrency on Metal

Metal serving has historically shown a flat total-throughput curve: raise
concurrency and per-request speed drops by the same factor, so the box
serves ~the same tokens/s at c8 as at c1. llama.cpp on Apple Silicon shows
this; earlier SlimServe Metal campaigns did too. On the current stack the
curve finally bends the right way. Measured no-spec ladders (exact-token
harness, 1000/256, seeded shipped-defaults sampling):

| box | c1 | c2 | c4 | c8 | c4/c1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 Ultra (Qwen3.8 NVFP4, PR #12 campaign) | 11.06 | — | 20.68 | 23.7 | 1.87x |
| M5 Max (same stack, merged tree) | 19.4 | 34.7 | 51.7 | — | 2.66x |
| M5 Max canonical (adaptive spec) | 21.7 | 33.6 | 45.9 | 58.7 | 2.11x |

This document explains mechanically why the curve used to be flat and what
changed. Nothing here is speculative: every mechanism below shipped with a
measured, pinned gate in `perf/optimization_status.md` (UPDATE numbers
cited), and the flat-curve baseline is in the same notebook.

## Why the curve was flat

"No scaling" has a precise meaning: decode step time grows linearly with
batch size M, so tokens/step and step time cancel. Four separate costs
each grew ~linearly in M, and together they kept step time ∝ M:

1. **Quantized weight traffic ∝ M.** Decode is memory-bound on weight
   bytes. A matrix–vector kernel serves one row; at batch M the naive
   path runs M GEMV passes and reads the full quantized weight stream M
   times. The notebook's census caught this directly: the per-row path
   ran "4–5x off weight bandwidth (flat ~99 GB/s)" at small M — all
   bandwidth spent re-reading the same bytes. This is the core reason
   batching bought nothing: the GPU was already saturating DRAM at c1,
   and c4 asked it to read 4x the bytes.
2. **Per-request host work ∝ M.** Every attention layer did per-request
   Python: KV insert as `.to` + index math + two advanced-indexing
   scatters, per-request SDPA gather loops, per-request metadata
   assembly. Host time scaled with rows and frequently exceeded GPU
   time.
3. **Host–GPU serialization.** Eager `seq_lens.to("cpu")` pulls and
   similar D2H syncs drained the whole in-flight GPU queue every step
   (measured 60 ms/step at the draft build under async scheduling
   before the fix). While the host waits, the GPU idles; while the host
   encodes, the GPU waits. Fixed cost per step, paid at every M, and it
   grows with batch because there is more metadata to materialize.
4. **Attention under-occupancy at wide heads.** The monolithic paged
   kernel ran one simdgroup per (head, batch): at D=256 / 24 GQA heads /
   small batch that is ~96 simdgroups on a GPU that wants thousands —
   0.9 GB/s effective in the N4 census. Batch could not amortize a
   kernel that was idle-bound, not bandwidth-bound.

## What changed

Each wall got its own measured fix; concurrency scaling is the sum.

### 1. Weight-stationary batch GEMV (the decisive one)

`qgemv_*_mb` / `mv_ext` twins (Q4_K campaign 2026-08-11, then
`qgemv_fp8ch` v6 and `qgemv_nvfp4_planar` v7 for the compressed-tensors
path — UPDATEs 18/19): for M ≤ 8, **each quantized weight block is read
and decoded in-register once, then applied to all M activation rows**
(the block sits in two float4 registers; per-row work is a pair of vec4
FMAs). Weight traffic per step becomes ~1x regardless of M, so step time
is nearly flat in M and total throughput scales with M. Measured at
461–662 GB/s at serving shapes — the memory-bandwidth floor the modeled
step time predicted. Above the small-M band (M in [9, 32], e.g. c8+
verify shapes), routing crosses over to dense GEMM against the bf16
weights materialized at load, which batches well natively. The routing
is by measured M-crossover per kernel family, with per-family kill
switches.

This is the piece llama.cpp lacks in the band that matters: its Metal
quantized path serves M=1 with mat-vec kernels and engages simdgroup
mat-mat tiles only at larger M; in the continuous-batching band (M =
2..8) each additional sequence re-pays the weight stream, which is
exactly the flat curve. (Its host side also re-encodes the graph
serially per step, which is wall 2/3 in miniature.)

### 2. Per-request host work made O(1) per step

- **One-dispatch KV insert** (`kv_cache_scatter`, one kernel for all
  tokens/layers' rows in the step) replaces the per-layer `.to` + index
  math + two advanced-indexing scatters.
- **Steady-decode metadata reuse**: on steady uniform decode steps the
  GDN builder rebuild (8.2 ms/step) is skipped via a signature check and
  a 2-copy in-place refresh (UPDATE 43).
- **Fused glue**: add+RMSNorm fused (UPDATE 21), GDN prep/norm dispatch
  fusion (UPDATE 22) — fewer dispatches per layer, so the per-step
  dispatch count stops being the ceiling.
- **Muse single-CB** (UPDATE 54 design; eligible steps): the whole
  64-layer forward encoded into one command buffer, collapsing
  command-buffer boundary gaps that the census put at ~73% of channel
  time at c1.

### 3. Sync-free host/GPU overlap

Async scheduling plus the bound-metadata design: the attention metadata
carries a host-side upper bound (`seq_lens_cpu_max` /
`seq_lens_cpu_bound`) computed without touching the device, and the
exact host lens are materialized lazily only by the one path that needs
them (prefill SDPA). Kernel routing and split-K sizing read the bound.
The D2H queue-drain (60 ms/step at the draft build) is gone; the host
encodes step N+1 while the GPU executes step N at any batch size.

### 4. Attention that occupies the GPU

The D=256 decode rides a split-K partition/reduce pair: the context axis
is cut into 512-token slices, each (head, batch, slice) gets its own
threadgroup, and an exact online-softmax reduce merges. Occupancy scales
with context instead of collapsing at wide heads (UPDATE 20: c4 +9.1%,
c8 +16.5% on the NVFP4 campaign; +6.4% GSM8K on the Q2K path measured
independently on main). The hybrid KV pool's strided/page-local layouts
are addressed via an explicit 64-bit block stride so this works in place
against the shared pool — no repacking step to grow with M.

## What still limits scaling

- **KV/attention reads are inherently per-request.** Weight reads
  amortize; KV reads do not. As M grows, attention bytes grow linearly
  and eventually dominate — that is the c8 bend (58.7 = 2.7x, not 8x)
  and it steepens with context length.
- **The small-M band is hand-tuned.** The weight-stationary twins cover
  M ≤ 8; the M=8 collapse of an earlier generation (UPDATE 11) is the
  cautionary tale — geometry off by one tier and the whole effect
  vanishes. The M ∈ [9, 32] band leans on dense GEMM against
  materialized bf16, which costs resident memory.
- **Per-box crossovers.** The batch-1 D=256 SDPA/kernel crossover tuned
  on M1 Ultra was wrong for M5 Max (−19% on the no-spec path there,
  fixed via `VLLM_QC_PA256_MIN_CTX_B1`). Constants measured on one GPU
  generation are hypotheses on the next.

## TL;DR

Metal never scaled because batch-M decode paid M times for weights,
M times for host work, and a sync tax every step, on top of an
under-occupied attention kernel. This stack reads weights once per step
regardless of M (weight-stationary batch GEMV), does host work once per
step (one-dispatch scatter, steady metadata, fused glue, single-CB),
never blocks the pipeline for metadata (host-side bounds), and splits
attention across the context axis for occupancy. Concurrency then buys
what it always should have: more tokens per weight-stream pass.
