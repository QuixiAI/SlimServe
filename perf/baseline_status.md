# SlimServe Baseline Status

This file holds stable baseline snapshots for comparison. Raw outputs belong in
`perf/results/`; summarize only the numbers needed to compare future work.

## DeepSeek V4 Flash 0731

### Ampere A100 Exact Native Baseline - 2026-08-09

- Model: all-layer IQ2_XXS gate/up, Q2_K down, Q8 attention/shared/output GGUF.
- Workload: concurrency 1, 1,000 input tokens, exactly 2,000 output tokens,
  no speculation, full decode CUDA graphs.
- Retained change over the prior baseline: one fused native sparse-MLA decode
  launch consumes the main and extra value sources.

| Profile | GPUs | Aggregate tok/s | Median wall s | Exact | Scaling |
| --- | ---: | ---: | ---: | --- | ---: |
| `dsv4-2` | 2x A100 | 92.022 | 21.734 | yes | baseline |
| `dsv4-4` | 4x A100 | 112.573 | 17.766 | yes | 1.223x |

TP4 still fails the required 1.5x-over-TP2 sanity gate. Its minimum acceptable
value for this TP2 result is 138.03 tok/s. Raw results are in
`perf/results/2026-08-09/dsv4-a100-merged-mla-tp2/` and
`perf/results/2026-08-09/dsv4-a100-merged-mla-tp4/`.

### Superseded Ampere A100 Exact Native Baseline - 2026-08-08

- Model: all-layer IQ2_XXS gate/up, Q2_K down, Q8 attention/shared/output GGUF.
- Workload: concurrency 1, 1,000 input tokens, exactly 2,000 output tokens,
  no speculation, full decode CUDA graphs.

| Profile | GPUs | Aggregate tok/s | Median wall s | Exact | Scaling |
| --- | ---: | ---: | ---: | --- | ---: |
| `dsv4-2` | 2x A100 | 90.835 | 22.017 | yes | baseline |
| `dsv4-4` | 4x A100 | 110.218 | 18.146 | yes | 1.213x |

TP4 fails the required 1.5x-over-TP2 sanity gate; its minimum acceptable value
for this TP2 result is 136.25 tok/s. Treat 110.22 as an active defective-scaling
baseline, not a completed optimization target. Raw results are in
`perf/results/2026-08-08/dsv4-a100-indexer-group4-tp2/` and
`perf/results/2026-08-08/dsv4-a100-mhc-low-priority-tp4/`.

### ROCm

- Status: optimized platform reference.
- Hardware: TP2 on 2x MI300X.
- Main workload: exact 100,000 input tokens, exact 2,000 output tokens,
  prefix-cached 99,984-token shared prefix, fixed 46.17 GiB KV budget.
- Best recorded non-speculative long-context exact results. Concurrency 1-32
  are from the retained split-threshold sweep; the 64-way row is the later
  final Q2/Q8 kernel result:

| Concurrency | Wall s | Aggregate tok/s | Decode-window tok/s | Per-request tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 47.71 | 41.92 | 42.13 | 41.92 |
| 2 | 57.31 | 69.79 | 70.08 | 34.90 |
| 4 | 77.55 | 103.16 | 103.48 | 25.79 |
| 8 | 120.45 | 132.84 | 133.09 | 16.61 |
| 16 | 175.87 | 181.95 | 182.18 | 11.37 |
| 32 | 350.50 | 182.59 | 182.71 | 5.71 |
| 64 | 323.313 | 395.902 | about 393-420 steady | 6.186 |

- Kernel history: original 64-way baseline was 240.12 aggregate tok/s. Q2_K
  4x64 tile pass raised it to 325.01 tok/s (+35.3%), then Q2_K 4x32 plus
  adaptive Q8_0 raised it to 395.902 tok/s (+21.8% over previous, +64.9% over
  original).
- DSpark long-context guidance: on realistic chat workload, no spec 254.25
  tok/s, spec-3 285.32 tok/s (+12.2%), spec-4 252.73 tok/s, spec-7 194.47
  tok/s. Use spec-3 for batch-64 long-context; reserve spec-7 for low-batch
  latency.
- TurboQuant draft KV: `turboquant_k8v4` smoke passed with healthy acceptance
  at 500 and 4000 prompt tokens; batch-64 100k re-benchmark is still pending.
- 1M context: batch-1 supported with manual
  `kv_cache_memory_bytes=50_465_865_728`; 1,000,021 prompt tokens + 32 outputs
  ran in 1,847 s, 541.4 prefill tok/s.
- Source notes: `perf/optimization_status.md` and `perf_worklog.md`.

### Metal

- Status: optimized DS4 backend reference.
- Hardware: Apple M5 Max, 128 GiB unified memory.
- Workload: exact 1,000 input tokens, exact 2,000 output tokens, local
  DeepSeek V4 Flash 0731 hybrid GGUF.
- Current correctness-qualified results:

| Concurrency | Aggregate tok/s | Wall s | Mean latency s | Notes |
| ---: | ---: | ---: | ---: | --- |
| 1 | 33.684 | 59.375 | -- | server decode avg 35.63 tok/s |
| 8 original | 32.821 | 487.487 | 485.164 | before retained scheduler setting |
| 8 stable | 35.350 | 452.614 | 452.038 | retained, +7.70% vs original c8 |

- Retained setting: `--mixed-prefill-quantum 2048`.
- Source notes: `benchmarks/dsv4_metal_perf.md`.

### Ampere A100 TP2

- Status: exact-token baseline promoted 2026-08-09 after the NaN/OOV sampler
  fix (see `perf/optimization_status.md`, "A100 TP2 Lifecycle Crash Root
  Cause"). Full overlap enabled, no diagnostic env.
- Profile: `dsv4-2`
- Quant: `IQ2_XXS`
- Config: 19 GiB KV per worker (`kv_cache_memory_bytes=20401094656`),
  `max_model_len=1048576`, APC on, native `fp8_ds_mla` target KV, native
  TurboQuant `turboquant_k8v4` draft KV, DSpark k=5, PIECEWISE graphs
  (capture 32), async mHC. Planner: 3,646,636 logical KV tokens.
- Exact-token harness (`benchmarks/benchmark_dsv4_exact.py`, concurrency 1,
  8-token warmup, all runs `exact: true`, single lifecycle without restart):

| Stage | tok/s |
| --- | ---: |
| 1K in / 2K out, run 1 | 168.0 |
| 1K in / 2K out, run 2 | 168.7 |
| 12K cold | 89.0 |
| 12K hot (APC) | 93.3 |
| 128K cold | 37.1 |
| 128K hot (APC) | 38.2 |
| post-128K 1K/2K continuation | 111.8 |

- Mean spec-decode acceptance length ~3.5-3.7 (k=5). Earlier interval-log
  probes (~29-32 tok/s, acceptance 1.00) and the 82.4/106.7 tok/s
  stream-serialized runs predate the sampler fix and were quality-poisoned;
  treat them as obsolete.
- Note: the post-128K continuation at 111.8 tok/s vs the fresh-server 168 is
  unexplained (KV pool occupancy after the 1M-scale context is suspected);
  measure before treating it as a regression. Recurs on the hybrid quant
  (94.5 vs 168 on 2026-08-10).
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-tp2-kv-capacity/control/clean-v33-*.json`
  and `server-v33-clean.log`.

### Ampere A100 - final dsv4 profile family, 128K-lifecycle qualified 2026-08-10

All five A100 dsv4 profiles passed the full single-lifecycle qualification
(1K/2K x2, 12K cold/hot, 128K cold/hot, post-128K continuation; exact-token
harness, c1, spec decode on, every stage `exact: true`, zero preemptions):

| Profile (layout) | 1K/2K c1 | 12K cold/hot | 128K cold/hot | post-128K | hot c8 agg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dsv4-q4ktail-2` (TP2, 13 GiB KV) | 168.7 / 168.4 | 92.0 / 88.8 | 65.4 / 64.4 | 94.5 | 185.5 |
| `dsv4-q4ktail-4` (TP4) | 128.1 / 161.8 | 152.6 / 155.2 | 94.8 / 98.3 | 171.4 | ~416-606 |
| `dsv4-q4ktail-8` (TP4 x DP2, cap 64) | 39.9 / 40.0 | 26.9 / 37.3 | 3.0 / 26.4 | 40.0 | 921.8 |
| `dsv4-mxfp4-4` (TP4) | 110.8 / 110.8 | 76.6 / 75.5 | 55.7 / 57.2 | 110.5 | ~99-118 |
| `dsv4-mxfp4-8` (TP8, cap 64) | 164.9 / 164.5 | 117.1 / 116.5 | 80.4 / 81.5 | 165.0 | 275.1 |

- Hybrid TP2 KV budget 13958643712 (13 GiB) qualified through the full 128K
  lifecycle; 14 GiB crashed at 128K cold prefill. IQ2_XXS retains its
  19 GiB budget via `quant_overrides`.
- hybrid-8 c1 numbers are the DP2 characteristic (one active replica plus
  per-step DP coordination; 128K cold 3.0 = round-robin defeating warmup
  APC). Its service point is c8 aggregate: 921.8 hot, the box record.
  Single-stream champion remains hybrid TP8 (329.5 c1, unserved layout).
- post-128K dip is TP2-only (94.5 vs 168); TP4/TP8 continue at their
  fresh-server band, supporting the KV-occupancy theory.
- The 4-GPU profiles' lifecycle ran pairwise on a split box (hybrid-4 on
  GPUs 0-3, mxfp4-4 on 4-7 concurrently); minor cross-contention possible.
- c8 aggregates and the full {quant} x {layout} matrix, plus variance
  caveats (+/-30% boot-to-boot), are in `perf/optimization_status.md`.
  Hybrid TP2xDP4 fails engine init and is marked illegal.
- UPDATE (same day, post-qualification): the MXFP4 tensor-core grouped
  MoE tile (`VLLM_GGUF_MXFP4_MMQ_V2`, see optimization notebook) lifted
  dsv4-mxfp4-4 to 129/126 c1, 110.6/115.8 @12K, 208.4/200.3 hot c8, and
  dsv4-mxfp4-8 to 182.8/183.3 c1, 161.1/163.3 @12K, 302.6/328.3 c8 --
  the table above is the qualification record, not the current baseline.
- UPDATE 2: the segmented MoE pipeline with the fused SwiGLU+Q8_1 W1
  epilogue (`VLLM_GGUF_DSV4_MXFP4_SEG`, XPU-review port) further lifted
  the prefill-bearing stages: mxfp4-4 cold c8 208->277 (+33%), mxfp4-8
  cold c8 303->356 (+18%); decode-dominated stages par. (A 694.8 hot-c8
  sample on mxfp4-8 is recorded as variance, not a kernel claim -- see
  the optimization notebook caveat.)
- UPDATE 3 (final for 2026-08-10): capture-64 landed on every A100 tier;
  q4ktail-4 hot c8 moved into the 500-750 band (mechanism = graph-replayed
  48-token verify; magnitude partially acceptance-confounded), while
  mxfp4-4 c8 is capture-neutral (kernel-bound verify, see notebook) at
  ~187-212. mxfp4-4 128K re-measured at 79.0/80.7 (was 55.7/57.2, +42%,
  the segmented-tile prefill win). Hybrid seg tiles gated at 768 tokens
  by measured crossover; MXFP4 verify-width tile routing rejected by
  measurement (fused GEMV wins below ~72 tokens).
- Raw artifacts: `perf/results/2026-08-10/dsv4-lifecycle-qual/`,
  `perf/results/2026-08-10/dsv4-a100-matrix/`.
