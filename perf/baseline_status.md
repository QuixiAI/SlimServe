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

### Metal M1 Ultra (dsv4-xxs-1 campaign)

- Status: UPDATE 2026-08-11 (11) — drain-kill era: the one-step op
  census found per-step MPS pipeline drains in the v2 worker glue
  (blocking pageable H2D in sampler staged writes / async_copy_to_gpu /
  StagedWriteTensor, plus repeat_interleave syncs in
  get_compressed_slot_mapping). All removed BIT-EXACTLY (same
  shas/counters): step ~302 -> 291.6-292.8, matrix 15.09-15.14. GPU
  busy at steady decode is 49.5% (ioreg) — kernels are DONE as the
  bottleneck (GPU ~145 ms/step = 24 ms/token-slot at M=6, better than
  ds4's 33.6/token); the wall is Python encode pacing. Async
  scheduling: neutral at c1 (env VLLM_METAL_ASYNC_SCHED; KV pool
  raised 1.0 -> 1.5 GiB for its lookahead — kept). Next lever: the
  native C++ step tape (Batch 4).
  Earlier (UPDATE 10): multi-row MoE GEMV kernels (iq2_xxs gate|up
  -38%, q2_K down -36%) took step 385-389 -> 351; the c128_boundary
  metadata fix took 351 -> ~302 bit-exactly. At M=1 our per-op
  aggregate (~33 ms GPU/token) is at PARITY with ds4 (33.6 ms
  GPU/token, 25.53/25.88 tok/s re-measured on this machine).
  Earlier same day (UPDATE 8): the glue-elimination sweep, step
  498 -> 385 (fused mHC, compressor tail, router, finalize, SwiGLU
  clamp FIX + acts, producer top-k, o-inverse-RoPE, indexer-Q).
  Trajectory-lottery protocol governs number comparisons (see
  optimization notebook).
- Hardware: Apple M1 Ultra, 128 GiB unified memory
  (`iogpu.wired_limit_mb=122880`).
- Profile: `dsv4-xxs-1` (IQ2XXS-w2Q2K target + 0731 DSpark drafter,
  `turboquant_k8v4` draft KV, fp16, 1 GiB KV pool). Weights pinned at boot:
  "Pinned 114 Metal allocations (93.73 GiB)".
- Exact-token harness, concurrency 1, 1,000 input tokens, clean boot,
  no profiler:

| Output tokens | Aggregate tok/s | Wall s | sha |
| ---: | ---: | ---: | --- |
| 8 | 1.86-1.94 | 4.1-4.3 | 5d4697585c6e... |
| 2000 (matrix) | 15.09-15.14 | 132.1-132.6 | 3a325666be45... |

- Mechanism metrics (what kernel work should be judged on, per the
  trajectory-lottery protocol refinement): decode step time ~292 ms
  (291.6-292.8 post-drain-kill [bit-exact], 302 post-c128-fix
  [bit-exact waste kill], 351 post-moe-mr, 385-389 post-glue-sweep [387 finalize+clamp, 389
  router, 395 compressor-tail, 440 producer, 443, 446, 450, 457],
  488-498 eager-mHC, ~540 pre-wave-1). Acceptance/draft on the tracked
  off1 trajectory: 4.61 (off1 draws across retained trajectories:
  3.68 / 2.63 / 3.38 / 4.06 / 3.23 / 4.61 — the current draw is
  ABOVE-median, so part of the 12.66 headline is draw luck; per-offset
  swings +-1.5). The 3a325666 trajectory came from the multi-row MoE
  kernels (reduction-order roll, lottery-gated); judge future changes
  on step time + mean matrix tok/s across offsets, not the tracked row.

- Numerics note (2026-08-11): qgemv_mb is 1-ULP different from the looped
  q8_0 batch-1 kernel on scattered rows (fast-math codegen); fused mHC is
  reduction-order-ULP vs eager. 8-tok sha UNCHANGED through both; the
  matrix trajectory re-baselined twice (5a662b7a under eager mHC, then
  3f64cc30 under fused), each deterministic across repeat runs with
  coherent text. The 64-tok gate row was dropped: at that length tok/s is
  pure trajectory/acceptance noise. Per the protocol refinement, judge
  kernel changes on paired step time + mean matrix tok/s across prompt
  offsets, not on any single-offset number.
- The 8-tok sha is unchanged since the pre-indexer baselines. 2000-tok
  crosses the 2048 boundary onto the top-k sparse path (producer
  `metal_indexer.py`, K cache via `dsv4_indexer_kv_insert`); long-context
  prefill separately validated by exact needle retrieval at 2,366 tokens.
- GPU-side profile (2026-08-11): step ~0.54 s wall = 91% GPU-busy; wave 1
  removed the M=6 weight re-read; the serving-only census then showed
  ~17k of ~28k tiny MPS ops/step were eager-mHC glue, now replaced by the
  fused mHC kernels (step 488 -> 449 ms). Next targets: post-mHC re-census,
  MoE vec (~1.5 ms/layer), lm_head (2.8 ms), sparse-attn producer glue.
- Bar: antirez ds4 21.08 aggregate / 24.79 decode-only (re-measured
  2026-08-11 on this machine/GGUF at ctx 1024: 25.53 decode, 25.88
  steady, ~33.6 ms GPU/token at ~87% busy). Tracked off1 matrix
  15.09-15.14 on an ABOVE-median draw (4.61); mechanism step ~292 ms.
  At the median-ish draw 3.5-3.7 this step corresponds to ~12.6-13.4
  tok/s — judge on step time. To beat 21.08 at draw ~3.5 the step must
  reach ~165 ms. Kernels are at parity (GPU ~145 ms/step, 49.5% busy);
  the remaining ~147 ms/step is Python encode pacing — the native C++
  step tape (Batch 4) is the path.
- WATCH: compressor pages grew to ~1.3 GiB after long runs (weights are
  pinned and safe; transients/KV are not). If long-workload wall times
  degrade, extend pinning to the KV pool and post-boot allocations.
- Source notes: `perf/optimization_status.md` 2026-08-11 entries
  (esp. "Task #14 LANDED"), `perf/metal_m1ultra_campaign_v2.md`.
Live-smoke regression note (2026-08-23; not a replacement baseline): the exact
registered `dsv4-xxs-1` profile reached health after loading 93.63 GiB, but its
first tiny text request ran at about 0.1 generation tok/s and had accepted 0
of 5 drafted tokens after several minutes. The request was terminated after
about 12 minutes. Until that first-step/verify regression is fixed and the
exact workload is repeated, the 33.684 tok/s row above is a historical
correctness-qualified comparison point, not evidence that today's registered
profile is healthy. Raw log:
`perf/results/2026-08-23/qwen38-kv-gather/smoke-all/dsv4-xxs-1.log`.

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
- UPDATE 4: the MXFP4 SoA repack (enabled at load, bit-identical) moved
  the whole mxfp4-4 profile: c1 159.5/158.2, 12K 139.0/139.1, 128K
  89.9/92.5, c8 322.8/248.3, all at normal acceptance (3.0-3.54).
  Cumulative vs the morning qualification: c1 +43%, 12K +83%, 128K +61%,
  c8 ~2.5x. These are the current mxfp4-4 baselines.
- Raw artifacts: `perf/results/2026-08-10/dsv4-lifecycle-qual/`,
  `perf/results/2026-08-10/dsv4-a100-matrix/`.

### 2026-08-10 evening correction: deployed q4ktail-4 capacity + degeneration incident

The deployed-instance concurrency sweep recorded in
`perf/optimization_status.md` earlier today (c16 732.4 / c32 925.3 /
c64 1023.8) is retracted as a capacity claim: those rows ran with
acceptance 5.70-5.88 (k=5 ceiling 6.0), which matches the silent
BOS-loop degeneration discovered the same evening -- the server emitted
special-token loops the drafter predicts perfectly -- compounded by a
same-source prompt confound. The c1-c8 rows (acceptance 3.53-4.24)
remain plausible.

Valid replacement band, measured on fresh disjoint text with healthy
acceptance (3.9-4.6), both daemons under simultaneous load, summing
per-instance valid cells across equivalent runs: c16 ~1050-1150,
c32 ~1300-1370, c64 ~1400-1550 aggregate tok/s for the box (two TP4
instances). No single row yet has both instances simultaneously healthy
end to end; treat these as banded estimates until the degeneration race
is fixed. Trigger matrix, retractions, harness degeneration guard, and
the slimserve-canary auto-restart are documented in the incident entry
in `perf/optimization_status.md`. TP8-tier numbers share the same
spec+full-graph machinery and inherit the same risk caveat.

UPDATE 12 (2026-08-11, post step-tape): Baseline row UNCHANGED — matrix
off1-2000 15.06-15.14 tok/s | wall 132.1-132.8 | sha 3a325666be45 |
step ~292 ms; 8-tok sha 5d4697585c6e. The Batch 4 S1b native step tape
(22 non-indexer layers in C++, VLLM_QC_STEP_TAPE, default OFF) is
bit-exact (both sha gates identical with tape live) and serving-neutral
(15.14 / 132.09). CENTRAL REVISION: with the tape saturating encode,
decode-only GPU busy is 98.6% — the step is GPU-EXECUTION-BOUND at
~280 ms/step. The earlier "49.5% busy, kernels done, CPU encode is the
wall" reading was a command-buffer-fragmentation artifact of the ioreg
utilization counter. The bar (ds4 21.08 agg / 25.53 decode) now runs
through GPU kernel time: per-kernel GPU census next, then the top
kernel families and speculation economics.

## UPDATE 13 (2026-08-12): mhc_pre split + compress front512 — 15.83-15.86 tok/s

- dsv4-xxs-1 off1-2000 exact matrix: **15.83-15.86 agg tok/s, wall
  126.1-126.35 s, sha abaa1c24b187, step ~272 ms, draw 4.505** (was
  15.06-15.14 / 132.1-132.8 / 3a325666be45 / ~292 ms / 4.61).
- Changes: (1) dsv4_mhc_pre split kernel (dots pass: one simdgroup per
  (token,row|sqsum) job, 8-ahead load batching; finalize pass; bitwise-
  identical to the monolith, VLLM_QC_MHC_SPLIT=0 legacy fallback);
  (2) VLLM_QC_COMPRESS_FRONT default 1 (fused front512 compressor,
  ULP-class). Gates: 8-tok sha IDENTICAL; 2000-tok deterministic x2 +
  coherent + step-time (trajectory lottery for the front512 ULPs).
- 8-tok row: 1.95 tps, sha 5d4697585c6e unchanged.
- ds4 bar: 25.53 decode / 25.88 steady. Beat-the-bar arithmetic at draw
  4.505: step <= ~176 ms. Remaining plan: perf/optimization_status.md
  Batch 9 (fusion war) — swiglu vec bandwidth, mhc_post widening,
  attention-region glue, drafter block.

## UPDATE 14 (2026-08-12): Batch 9 wave 2 — mhc_post widening

- off1-2000 exact: **15.93 tok/s, wall 125.58 s, sha abaa1c24b187,
  step 270.4 ms, draw 4.505** (counters 1557/2220/444, bit-identical
  to UPDATE 13). Single gated run; sha-identity class.
- Landed on top of wave 1: dsv4_mhc_post grid (tokens,1) -> (tokens,8)
  (bit-exact slice partition; kill-switch is reverting the dispatch,
  no env knob).
- 8-tok row: sha 5d4697585c6e unchanged.
- ds4 bar unchanged: step <= ~176 ms at draw 4.505. Next: Batch 9
  wave 3 (attention glue folds: fp16-input qnorm/kv-insert kernels,
  sparse_attention -> o_padded direct write).

## UPDATE 15 (2026-08-12): Batch 9 wave 3 — attention glue folds

- off1-2000 exact: **15.98 tok/s, wall 125.19 s, sha abaa1c24b187,
  step 269.6 ms, draw 4.505** (counters 1557/2220/444). sha-identity
  class on top of UPDATE 14.
- Landed: fp16-input qnorm_rope_kv_insert (in-kernel RNE bf16 rounding,
  strided kv bind) and sparse_attention direct write into the fp16
  serving buffer (rounds through bf16 in-register). -129 glue
  kernels/step.
- 8-tok row: sha 5d4697585c6e unchanged.
- ds4 bar: step <= ~176 ms at draw 4.505; 93.6 ms/step to cut.

## UPDATE 16 (2026-08-12): Batch 9 wave 3b — compressor cast folds (neutral)

- off1-2000 exact: **15.91-15.99 tok/s, wall 125.10-125.74 s, sha
  abaa1c24b187, step ~269.4-270.8 ms, draw 4.505**. Bit-exact vs
  UPDATE 15; throughput unchanged within noise. Current serving state
  includes waves 1, 2, 3, 3b.
- 8-tok row: sha 5d4697585c6e unchanged.
- ds4 bar: step <= ~176 ms; ~94 ms/step still to cut. Glue-fold path
  exhausted; next work targets real GPU-time mass (op census, then
  fusion of heavy MPSGraph ops / drafter / occupancy work).

## UPDATE 17 (2026-08-12): Batch 9 wave 4A — NEW BASELINE, ds4 BAR BEATEN

- off1-2000 exact: **31.46-31.52 tok/s, wall 63.45-63.57 s, sha
  abaa1c24b187, step ~130.5-130.8 ms, draw 4.505** (counters
  1557/2220/444). Deterministic x2; sha-identity lineage intact all the
  way from the pre-Batch-9 baseline.
- **ds4+DSpark bar (25.53 decode / 25.88 steady): BEATEN, +23%.**
- Landed stack: waves 1 (mhc_pre split + front512), 2 (mhc_post
  widening), 3 (fp16 qnorm/kv-insert + sparse direct write), 3b
  (compressor cast folds), 4A (cos/sin + positions/slots + o_proj
  marshalling memos). Mechanism note in optimization_status (transient
  MPS allocation churn was falsely serializing the GPU).
- 8-tok row: sha 5d4697585c6e unchanged, 1.95-2.0 tps band.

## UPDATE 18 (2026-08-12): 4A bisect verdict, wave 4B, first long-context rows

- Bisect (env gates VLLM_QC_MEMO_{COSSIN,POS,WOA}, one boot per leg,
  sha abaa1c24b187 held on every leg): **the cos_sin_bf16_halves memo
  IS the 4A win** — COSSIN=0 regresses to 273.4 ms/step (pre-4A wall);
  POS=0 measures 131.0 (nil); WOA bounded ~0 by arithmetic. Mechanism
  sharpened: CB-level trace rows count intra-CB hazard stalls as
  "busy", so the pre-4A 97.5%-busy/270 ms picture was resident CBs
  stalled on false hazards from ~1,300 transient allocations/step in
  the insert chain. Rank future targets by transient allocations
  ADJACENT TO THE INSERT/ATTENTION CB CHAIN, not by raw counts.
- Wave 4B round 1 (strided qc_rms_norm bind + fused_qk fp32-weight
  memo; offline bitwise parity ALL-PASS): **130.1-130.3 ms/step,
  31.58-31.60 tok/s, shas unchanged — RETAINED (~-0.5 ms).**
- Wave 4B round 2 (census-guided index-dtype memos: front512 feeds,
  indexer short-topk fill once/step, indexer_compress_insert memo):
  shas unchanged, 130.9-131.6 ms/step — throughput-neutral within the
  noise band; RETAINED (strictly less work).
- CURRENT 1K/2000 BASELINE BAND: **130.0-131.6 ms/step, 31.3-31.6
  tok/s, sha abaa1c24b187** (8-tok 5d4697585c6e). ds4 bar remains
  beaten by ~+23%.
- --ctx 131072 costs NOTHING at 1K (130.0 ms/step measured; KV planner
  139,769 tokens at the profile's 1.6 GiB pool).
- FIRST Metal long-context rows (dsv4-xxs-1, --ctx 131072, exact
  harness, cold = no warmup on fresh boot, hot = immediate APC rerun):

| Stage | tok/s | wall s | sha | notes |
| --- | ---: | ---: | --- | --- |
| 12K cold (12000 in/2000 out, off 2001) | 8.76-8.81 | 227.1-228.2 | 82ae1aaa | prefill ~74.5 tok/s |
| 12K hot (APC) | 28.83-29.48 | 67.8-69.4 | 2267f541 | decode ~152-155 ms/step |
| 32K cold probe (32000 in/500 out, off 0) | 1.11 | 452.2 | 06aea929 | prefill ~73.7 tok/s |

  Both 12K rows reproduced bit-exactly across the round-2 build (same
  shas, two boots). Cold/hot shas differ by design (partial-block APC
  recompute changes reduction shapes).

- **FINDING — prefill is the new top surface: ~74 tok/s LINEAR
  (13.5 ms/token), only ~2x decode's per-token cost.** Chunked 2048-
  token prefill should amortize weight reads to hundreds of tok/s;
  something is per-token-serialized. Unprofiled as of this update.
- **OPEN BUG — 128K completions request wedges the API frontend**
  (3x reproducible incl. a fresh boot; 12K/32K fine on the same boot;
  tokenizer exonerated offline at 152K tokens, 0.25 s; wedge poisons
  all later requests; only reboot recovers). Investigation notes in
  optimization_status. 128K rows BLOCKED on this.
- Ops lesson: after killing a WEDGED server, wait for wired-memory
  release (memory_pressure -Q > 70% free) before rebooting — an
  immediate reboot raced the release and produced a one-off GPU
  command-buffer timeout that killed the first request pipeline.

## UPDATE 18a (2026-08-12): 128K addendum + machine event

- 128K functional proof: 128000-in/100-out COMPLETED under the
  phaseprof (sync-bracketed) boot — sha 27748c3cfae29d26, exact:true.
  The >64K wedge is an MPSGraph encode-queue pathology, mitigated by
  a scoped per-producer-call synchronize (metal_indexer.py, default
  on, VLLM_QC_LONGCTX_SYNC=0 to disable). Clean-boot 128K cold/hot
  ROWS STILL PENDING: from ~13:45 the machine (uptime 50 days) began
  failing FIRST requests with GPU command-buffer timeouts on builds/
  configs that had gated clean earlier — machine restart required
  before any further rows. All numeric rows in UPDATE 18 predate the
  degradation and were deterministic/sha-gated.
- Prefill roadmap (from the 53-chunk phase split): MoE FFN ~14%,
  attention ~19%, comp_full_compress 134 ms/call, attn_wqb_insert_c
  399 ms/call — grouped-GEMM MoE prefill + insert-chain work is the
  path from ~74 tok/s to competitive prefill.
- Post-restart checklist: boot dsv4-xxs-1 (prod ctx) -> 8-tok sha
  5d4697585c6e + off1-2000 sha abaa1c24b187 at ~130-131.6 ms/step ->
  then --ctx 131072 for the 128K cold/hot rows.

## UPDATE 18b (2026-08-12): machine restarted; re-gate blocked on iogpu sysctl

- The user restarted the machine (clears the AGX degradation from 18a).
  First post-restart boot was healthy, but the first request died with
  kIOGPUCommandBufferCallbackError**OutOfMemory**: the restart reset
  `iogpu.wired_limit_mb` to 0 (default ~96 GiB ceiling < 93.73 GiB
  pinned + KV + pools). Server killed (post-CB-error engines are
  poisoned). REQUIRED before any boot:
  `sudo sysctl iogpu.wired_limit_mb=122880`, verify with
  `sysctl iogpu.wired_limit_mb`. Then run the 18a checklist unchanged.
- No numeric baselines change in this update. All UPDATE 18 rows stand.
- NEW DOCUMENT: `perf/metal_m1ultra_retrospective.md` — consolidated
  campaign retrospective (what worked/didn't/surprises), measured
  roofline (step traffic ~22.5 GB -> effective 172 GB/s vs ds4's 283
  effective and 738 peak-measured; realistic target tier step ~80-90 ms
  = ~50-55 tok/s), cross-platform lessons (A100/ROCm), external-engine
  technique audit (ds4, llama.cpp), and the ranked avenue list
  (decode: re-census churn kill, SoA repack, split-K attention wiring,
  sum6 down; prefill: weight-stream-flat MoE tile + insert chain).

## UPDATE 19 (2026-08-12): post-restart RE-BASELINE — 127.2-127.6 ms/step

- The machine restart rolled the trajectory lottery (ULP-class input
  moved; kernels/routes trace-verified live; see optimization notebook
  "Post-restart forensics"). New standing gates:

| Gate | tok/s | wall s | sha | counters |
| --- | ---: | ---: | --- | --- |
| 8-tok (1000-in/8-out off1) | 2.4-2.5 | 3.2-3.3 | db2846cf721b | 2/10/7 |
| off1-2000 matrix | 30.92-31.00 | 64.52-64.69 | a936de0fa7c7 | 1537/2320/464 |

- **Step 127.2-127.6 ms (draw 4.31)** — mechanism slightly better than
  UPDATE 18's 130.0-131.6 (draw 4.505). ds4 bar (25.53 decode) remains
  beaten (+21% at this draw). Deterministic x3 across 2 boots.
- BOOT PROTOCOL (mandatory): verify `sysctl iogpu.wired_limit_mb` =
  122880; boot; send a tiny primer request BEFORE anything big; a
  first-request GPU timeout = poisoned engine, reboot.
- Focus per user direction: decode only. Long-context/prefill rows
  parked. Next decode work: retrospective avenue list
  (perf/metal_m1ultra_retrospective.md §9) — re-census transient churn
  at 127 ms, SoA repack, split-K attention wiring, sum6 down.

- UPDATE 19a (2026-08-13): wave 5 round 1 (transient-output ring,
  VLLM_QC_OUT_RING, default on) RETAINED bit-exact and
  throughput-neutral — baseline rows unchanged (ring gates: 128.3/128.5
  ms, shas/counters identical). Churn vein declared CLOSED; next levers
  are kernel-bandwidth work (SoA repack, split-K attention, sum6).

- UPDATE 20 (2026-08-13): wave 5 round 2 — Q2_K expert SoA repack
  (VLLM_QC_MOE_SOA, default on, load-time byte-neutral planes)
  RETAINED bit-exact: shas/counters IDENTICAL (db2846cf 2/10/7;
  a936de0f 1537/2320/464), **step 126.8-127.1 ms** (walls 64.34/64.47,
  30.98-31.05 tok/s) — floor-edge improvement consistent with the
  isolated -10% q2_K kernel win. IQ2_XXS stays AoS (three SoA layouts
  measured 2-4% slower on Apple LSU — see notebook; do-not-redo).
  Boot adds ~16 s synchronous repack in process_weights_after_loading.
  Baseline band now 126.8-127.6 ms pending more boots; gates otherwise
  unchanged.

## UPDATE 21 (2026-08-13): wave 6 split-K sparse-MLA — RE-BASELINE 122.6-124.2 ms/step

- Split-K decode attention (partition kernel + LSE reduce,
  `VLLM_QC_MLA_SPLITK` default on, target 768 SGs => P=2 at verify)
  RETAINED: step mean 127.4 -> **123.3 ms** across a paired 7-offset
  sweep (win at every offset, -2.8 to -4.9 ms); draw means 4.13 (off)
  vs 4.33 (on); tok/s means 29.7 vs 32.0. ULP class — the matrix sha
  ROLLED (expected; cross-partition LSE merge reassociates softmax).
  New standing gates:

| Gate | tok/s | wall s | sha | counters |
| --- | ---: | ---: | --- | --- |
| 8-tok (1000-in/8-out off1) | 2.4-2.5 | 3.2-3.3 | db2846cf721b (UNCHANGED) | 2/10/7 |
| off1-2000 matrix | 25.4 | 78.89-78.92 | 4d18b4fac460 | 1409/2955/591 |

- **Step band 122.6-124.2 ms** (n=7 offsets, mean 123.3). NOTE: off1
  itself is now a LOW-draw offset for this trajectory (3.38) — its
  25.4 tok/s is not a regression signal; use the 7-offset mean
  (~32 tok/s) for cross-baseline comparisons. Deterministic x2.
- 8-tok sha db2846cf identical because the 2-draft trajectory
  survived the ULP perturbation; identical short-gate shas after a
  ULP change must be liveness-checked (H=64 lesson, see notebook).
- Kill/tune: `VLLM_QC_MLA_SPLITK=0` reverts to the fused kernel
  (measured 127.4 ms mean this build); `VLLM_QC_MLA_SPLITK_TG`
  (default 768; 1536 untried).
- Next: sum6 down fold (§9 row 4), spec block re-tune (row 5).

- UPDATE 21a (2026-08-13): wave 7 sum6 down fold (VLLM_QC_MOE_SUM6,
  default on — q2_K down GEMV with the weighted slot-sum + output write
  folded into one kernel) RETAINED bit-exact: all shas/counters
  IDENTICAL (db2846cf 2/10/7; 4d18b4fa 1409/2955/591; off3 ce1bab26),
  liveness breadcrumb positive. Step off1 123.49/123.82 vs 124.2,
  off3 122.63 vs 122.63 — at worst neutral. All UPDATE 21 rows stand.

- UPDATE 21b (2026-08-13): waves 8-12 close the decode avenue list with
  NO baseline change — spec k=5 confirmed (k=6 -1.3 tok/s, k<5
  architecturally invalid), expert-grouped w13 RETIRED (2x regression,
  VLLM_QC_MOE_GROUP opt-in negative), kernel geometries confirmed
  optimal, split-K TG=1536 a wash. Wave 12 ceiling analysis: the step
  is GPU-SATURATED (cb_census busy 90.4 s > wall 78.8 s, 33 CBs/step)
  and 122.6-124.2 ms is the measured plateau; the ~80-90 ms tier is
  the pure-bandwidth floor, gap = serial time-multiplexing (see
  optimization notebook 2026-08-13 wave 12). All UPDATE 21/21a gates
  stand: 8-tok db2846cf 2/10/7; off1 matrix 4d18b4fa 1409/2955/591.

## UPDATE 22 (2026-08-13): prefill wave-1 v1 tiled w13 MoE GEMM — prefill RE-BASELINE ~99-105 tok/s

- qc_moe_mm_map0 + qc_moe_mm_id_iq2_xxs (llama.cpp mul_mm_id port) replace
  the per-slot w13 GEMV at prefill widths (>= 32 tokens;
  VLLM_QC_MOE_PREFILL_MM=0 reverts). Decode path untouched and gate-proven.
- Prefill walls (streaming TTFT, disjoint offsets, clean boot):
  512 -> 5.156 s | 1000 -> 9.550 | 2048 -> 20.108 | 3000 -> 29.520
  (99.3 / 104.7 / 101.9 / 101.6 tok/s; was 78.7 / 81.1 / 79.0 / 77.6).
  Still flat -> w2 GEMV, insert/compress, mhc remain the linear terms.
- Serving gates on this build: 8-tok off1 sha db2846cf721b UNCHANGED
  (2/10/7); off1-2000 sha rolled (prefill numerics) to 0adffb58c16a,
  determinism x2, counters 1597/2035/407, step 123.3-124.1 ms (plateau
  holds), wall 78.8 -> 56.0 s. MM=0 sentinel boot reproduces
  4d18b4fac460 / 1409/2955/591 exactly.
- FACT CORRECTION: routed experts E=256 (not 64). topk 6, hidden 4096,
  inter 2048, w13 [256, 4096, 1056] iq2_xxs AoS, w2 [256, 4096, 672->SoA]
  q2_K. All per-expert byte math in earlier entries scales accordingly.
- ds4 comparison target on this box: 277 tok/s prefill. Wave 1 v2 (w2
  tile, pair+SwiGLU fusion, map0 work queue) and wave 2 (insert/compress)
  are the remaining planned closers.

## UPDATE 23 (2026-08-13): PREFILL BEATS ds4 — 311 tok/s at 2048 (ds4: 277)

dsv4-xxs-1 prefill after the session-3 waves (v3a dual-half MoE tiles,
v3b mhc dots staging, v4a staged attention twin, v5 dense-causal MMA FA):

| width | TTFT s | tok/s | (session start) |
|-------|--------|-------|-----------------|
| 512   | 2.136  | 239.7 | 2.378 / 215.3   |
| 1000  | 3.289  | 304.0 | 3.788 / 264.0   |
| 2048  | 6.584  | 311.1 | 8.096 / 253.0   |
| 3000  | 9.855  | 304.4 | 11.721 / 256.0  |

Campaign total at 2048: 25.911 s -> 6.584 s (3.94x). ds4 target 277
tok/s: **111.9% — BEATEN**. Decode step unchanged (paired 120.2 vs
120.8 ms); 8-tok sha still db2846cf721b 2/10/7.

New serving baseline (all defaults): off1-2000 sha 3fc700d9818b,
counters 1496/2535/507 (v5 is ULP-class; VLLM_QC_MLA_PREFILL_FA_MMA=0
bit-exactly restores the previous ec0cc6c5908e 1520/2410/482 baseline).
Measurement: perf/results/2026-08-13/prefill_mm_v1/walls_v5.log, probe
perf/results/2026-08-13/prefill_baseline/prefill_probe.py (fresh boot
required — APC poisons repeat probes at the same offsets).

## UPDATE 24 (2026-08-14): session-3 final — 344.6 tok/s at 2048 (124% of ds4)

v6 (single-chunk scheduling, max_num_batched_tokens 2176) + v7 (native
prefill indexer top-k): 512 2.174/235.5 | 1000 3.150/317.4 | 2048
5.943/344.6 | 3000 9.901/303.0 (walls_v7.log). Serving baselines
unchanged from UPDATE 23 (off1-2000 3fc700d9818b 1496/2535/507, 8-tok
db2846cf721b); new long-ctx anchor dd5c1c87fe60 (2500-in/64-out
offset 0). This is the standing prefill baseline.

## UPDATE 25 (2026-08-14): session-4 — 369.5 tok/s at 2048 (133% of ds4); box wedged pending machine restart

v9 (q2_K dequant load-shaping) + v10 (qgemm wide-tile + transposed
store + host de-fluff for large-M q8_0): fully gated on boot_v8 —
off1-2000 3fc700d9818b 1496/2535/507, 8-tok db2846cf721b, long-ctx
anchor dd5c1c87fe60, wide-path breadcrumb liveness, decode step 118.7
ms. Walls (fresh disjoint id-window TTFT): 512 2.001/255.9 | 1000
3.189/313.6 | **2048 5.543/369.5** | 3000 9.692/309.5. This is the
standing prefill baseline. v12 (wide-gate tier rows>=512 && N>=8192)
passed 8-tok + off1-2000 shas (step 118.4) but its anchor + walls are
BLOCKED by post-xctrace box poisoning (see optimization_status v12) —
re-gate after machine restart. Artifacts:
perf/results/2026-08-14/prefill_ext_v1/.

## UPDATE 26 (2026-08-14): v12 fully re-gated post-restart — 2048 at 5.522 s / 370.9 tok/s (134% of ds4); STANDING BASELINE

Machine restarted (sysctl iogpu.wired_limit_mb=122880 re-set and
verified). v12 build (v9 + v10 + wide-gate tier rows>=1024 || rows>=512
&& N>=8192) fully gated on boot_v12b, all BIT-EXACT to the pre-restart
baselines (the restart did NOT roll the trajectory): 8-tok
db2846cf721b 7/10/2; off1-2000 3fc700d9818b 1496/2535/507, wall
65.79 s => step ~118.9 ms (unregressed); long-ctx anchor 2500-in
offset-0 dd5c1c87fe60, 3.541 s; wide-kernel breadcrumb fired.

Walls (fresh disjoint id-window streaming-TTFT, boot_v12b):
512 2.110-2.175/235-243 | 1000 3.094/323.2 | **2048 5.522/370.9** |
3000 9.639-9.736/308-311. vs v10 baseline: 1000 +3% (the v12 tier
firing at rows=1000), 2048 confirmed (new best), 3000 flat (chunk-2
tier win lost in noise), 512 -5..-8% (below-tier path is IDENTICAL
code to v10 — boot-level host-floor variance at the ~0.8 s-floor-
dominated size, not a regression). This is the standing prefill
baseline. ds4 reference 277 tok/s => 134% at 2048.

WEDGE ROOT CAUSE REVISED (supersedes UPDATE 25's poisoning story): the
two-chunk wedge is a deterministic FIRST-MULTI-CHUNK-REQUEST ordering
effect, not xctrace poisoning and not the build — see
optimization_status "v12 RE-GATE" entry and the boot protocol in
perf/prefill_handoff.md STATUS UPDATE 7. Artifacts:
perf/results/2026-08-14/prefill_ext_v1/ (**v12b**.json, boot_v12*.log).

## UPDATE 27 (2026-08-14): cleanup phase re-gated bit-exact — UPDATE 26 stands unchanged

The wave-1 cleanup of the campaign tree (dead experiments, env switches
37 -> 12, structure + comment hygiene; see optimization_status
"CLEANUP PHASE" entry) is a zero-numeric-change pass: on the ship tree
all three serving shas are BIT-EXACT (8-tok db2846cf721b 7/10/2,
off1-2000 3fc700d9818b 1496/2535/507, 2500 anchor dd5c1c87fe60) and
walls sit in the UPDATE 26 band. After a second-pass three-reviewer
audit (dead kernel deleted, mhc template collapse, tape re-enabled —
see the "CLEANUP REVIEW PASS" notebook entry) the final gate boot
measures 512 2.218 | 1000 3.102/322.4 | **2048 5.512/371.6** | 3000
9.649 (boot_final) — 2048 matches UPDATE 26 (5.522/370.9) within
noise. UPDATE 26 remains the standing baseline; no numbers move.

Two operational notes: (1) machine-boot variance is real — boots earlier
on today's restarted box measured ~3-5% slower walls for pre-cleanup and
cleaned builds alike (A/B-proven); judge walls against the band, not a
single boot. (2) The async-output completion-event path has a host-timing
race (same family as the first-multi-chunk boot wedge):
vllm/models/deepseek_v4/{compressor,metal}.py keep their phaseprof
brackets and marshalling-memo conditionals because removing them flips
the race (boot-level bisect in the cleanup entry). Root-cause fix is
queued follow-up work. Artifacts:
perf/results/2026-08-14/cleanup_gate/ (**ship.json, walls_ship.log,
boot**.log).

## UPDATE 28 (2026-08-14): origin/main merged — 256K profile becomes the serving config; anchors re-pinned

- Merged origin/main (A100 Q4_K fused MoE + split-K NaN fix + Muse-Glimmer
  Metal + the Metal DSV4 256K profile resize) into the M1 Ultra campaign
  branch; metallib and extension rebuilt from merged sources, all six kernel
  oracles pass.
- MERGE IS BIT-EXACT, proven by config A/B: with the profile pinned back to
  the campaign's 3072/1.5 GiB benchmark sizing, the merged code reproduces
  every pre-merge anchor exactly — 8-tok db2846cf721b 7/10/2, off1-2000
  3fc700d9818b 1496/2535/507 at 65.52 s, 2500x64 dd5c1c87fe60 51/70/14 at
  3.490 s.
- The shipping profile keeps main's intended 262144/16 GiB sizing (with the
  campaign's fp16 dtype and 2176 batched-token reserve). Under it the
  off1-2000 trajectory legitimately changes past the indexer engagement
  length and the anchor re-pins:
  - 8-tok: db2846cf721b 7/10/2 (unchanged).
  - off1-2000: **7ce993786ba1 1538/2320/464, wall 63.2-63.5 s = 31.6 tok/s
    end-to-end (2000/(wall-prefill) ~= 33 tok/s decode-only)** — faster than
    the 3072-config 65.5 s and back at the campaign-best whole-wall band.
    chars_per_token 4.76, dumped completion coherent.
  - 2500x64: dd5c1c87fe60 (text bit-exact with pre-merge even under 256K;
    counters shift to 52/65/13), wall 3.57 s, no wedge after full boot ramp.
- Prefill walls under the shipping config: 512 1.94-2.63 (jittery,
  fixed-overhead dominated) | 1000 3.09/323.3 | 2048 5.55-5.71 (best fresh
  cursor 5.552 = 368.9 tok/s, within noise of UPDATE 26's 5.512/371.6) |
  3000 9.56.
- Boot-ramp protocol NOTE hardened: a primer plus a 1000-token/8-tok request
  is NOT sufficient ramp before a multi-chunk prefill — one A/B boot wedged
  on 2500 after exactly that. The full ramp (primer, 8-tok, one long decode
  request) preceded every clean multi-chunk request this session, matching
  the cleanup-gate sequence.
- ds4 bars (same box): prefill 277 -> 369 tok/s (133%); decode 21.1 -> 31.6
  end-to-end (150%).
- Raw artifacts: perf/results/2026-08-14/merge_gate/ (boot_final.log,
  boot_ship.log, ship_*.json, gate_*.json, walls.log, completions_2000/).

## UPDATE 29 (2026-08-17): second origin/main merge — code exonerated by A/B; 2500 anchor re-pinned after an environmental trajectory re-roll

- Merged origin/main again (5 commits: steady uniform-decode metadata reuse
  VLLM_STEADY_DECODE_META, qwarp8 IQ2 W1 A100 dial, notebook/rocm updates).
  Two conflicts (build_attn_metadata signature union, notebook append order);
  every Metal-relevant hunk verified inert (steady path requires FULL
  CUDA-graph mode + opt-in env, never true on Metal; sparse_swa change is a
  character-identical extract-method refactor; sparse_mla/default.py additive
  or gated). Profile tests 48/48. Merge commit author auroter.
- Gate (shipping 256K config, full ramp, fresh boots):
  - 8-tok: sha db2846cf721b, 7/10/2 — BIT-EXACT, unchanged since pinning.
  - off1-2000: sha 7ce993786ba1, 1538/2320/464, walls 62.93-63.09 s
    (31.8 tok/s end-to-end) — BIT-EXACT vs UPDATE 28, slightly faster wall.
  - 2500x64: sha e973493bef44 51/60/12 @ 3.53-3.54 s vs pinned dd5c1c87fe60
    52/65/13 @ 3.57 s — DIVERGED, stable across three fresh runs on two
    boots (plus prefix-cached reruns, same sha).
- A/B EXONERATION of the merge: restored the only four Metal-relevant merged
  python files (attn_utils, sparse_swa, sparse_mla, model_states/default) to
  the pre-merge tree (eb5f8d08e) on a fresh ramped boot — 8-tok and
  off1-2000 reproduce bit-exact AND the 2500 gives the SAME new sha
  e973493bef44. The merge code cannot be the cause; the >2048 sparse-path
  trajectory re-rolled environmentally on this box some time after the
  08-15 pinning (same machine boot — uptime since Aug 14 13:12 — so not a
  restart re-roll; precedent: the 08-12 re-roll entry). Short-context
  serving remains bit-stable across days.
- RE-PINNED long-ctx anchor: 2500-in/64-out offset-0 sha e973493bef44,
  counters 51/60/12, ~3.53 s. dd5c1c87fe60 is retired with this
  explanation; treat any FUTURE flip as suspect until A/B'd the same way.
- Raw artifacts: perf/results/2026-08-17/remerge_gate/
  (boot_2500check.log, ab_premerge_2500.json with full response text for
  future divergence-point analysis).

## UPDATE 30 (2026-08-17): all three anchors re-pinned after the fp8 scale exactness fixes (insert 29de4e993 + decode da05ace06)

- Cause is KNOWN and code-attributable, so this flip is NOT suspect under
  the UPDATE 29 rule: the decode-side fix changes every dequantized fp8 KV
  value by ~2 ulps of scale on every step (fast-math exp2 was 2 ulps low at
  negative integer inputs; see the 2026-08-17 notebook entries). The
  strongest evidence is the 8-tok anchor itself flipping — short-context
  trajectories stayed bit-stable through every prior environmental re-roll.
- Gate (shipping 256K config dsv4-xxs-1, full ramp primer -> 8-tok ->
  off1-2000 -> 2500x64, fresh boot, server-side /tokenize gate driver):
  - 8-tok RE-PINNED: sha 573db39598e7ff4ca0818aef7fff6a1bd719c33439c2028f
    646a5011b5aca27e, counters 5/15/3, wall 1.69-1.72 s. Deterministic 2/2.
  - off1-2000 RE-PINNED: sha bb83cc3054a3698f1134f773831e4557050b6a02de0a
    1a9aac5460512882b944, counters 1581/2115/423, walls 57.26-57.35 s
    (34.9 tok/s agg). Deterministic 2/2.
  - 2500x64 RE-PINNED: sha f75e1d41ac3df6fa95e5e59cc2667c6cbaa47c617b2cf2
    2417fbe68c4d64479d, counters 43/105/21, walls 4.49-4.71 s.
    Deterministic 3/3.
- Decode health judged on step ms, not wall: off1-2000 step
  (57.3 - 5.5)/423 ~= 122.5 ms vs UPDATE 29's ~123.7 ms — unchanged
  (-1%, within boot variance). The off1-2000 wall gain (63 -> 57.3 s) and
  the 2500x64 wall increase (3.53 -> 4.49 s; 21 draft steps vs 12) are
  both acceptance-mix effects of the new trajectories, not kernel changes.
  Over the 2000-token sample acceptance IMPROVED (3.74 vs 3.31 accepted
  per draft).
- Retired shas: db2846cf721b (8-tok), 7ce993786ba1 (off1-2000),
  e973493bef44 (2500x64).
- Raw artifacts: perf/results/2026-08-17/decode_exact_gate/
  (boot.log, 8tok.json, off1-2000.json, 2500x64.json with full response
  text).

## Muse-Glimmer-30B (Metal, M5 Max 128 GB)

### Speculative Serving Baseline - 2026-08-14

- Profile: `muse-kdyn-1` (kquant-dynamic GGUF, DFlash k=16, spec always on).
- Workload: 256-token greedy essay bench (`muse_bench.py`), port 8078,
  warmup + 3 runs; first-run number is the matched-position comparison
  point (chassis declines thermally across a triplet).
- Verify GEMMs route to tensor-ops kernels (`qgemm_sm_t`, variant 14) on
  the M5 GPU neural accelerators; metallib at `-std=metal4.0`.

| Metric | Value |
| --- | ---: |
| Spec-on decode (run 1/2/3) | 16.26 / 16.05 / 14.59 tok/s |
| Plain fused decode (2026-08-13 ref) | 14.4 tok/s |
| Acceptance | 1.73/draft (unchanged) |
| llama.cpp same box (plain/spec best) | 26.75 / 30.8 tok/s |

Spec-on now beats plain decode; the stale `--no-spec` advice was removed
from `slimserve/profiles.json`. Raw artifacts:
`perf/results/2026-08-14/tensor-mma/` and
`perf/results/2026-08-14/qgemm-sm-profile/`.

### Superseding Speculative Serving Baseline - 2026-08-15 (rested protocol)

- Profile: `muse-kdyn-1`, fused verify default ON, tensor-ops kernel route.
- Protocol: 45-min idle chassis, serve, 120 s settle, 3x256-token greedy
  essay bench. This protocol replaces ad-hoc thermal states for all route
  comparisons.

| Metric | Value |
| --- | ---: |
| Spec-on decode (run 1/2/3) | 19.94 / 20.14 / 19.74 tok/s |
| Same-protocol eager route reference | 17.81 / 17.56 / 17.58 tok/s |
| Acceptance | 1.73/draft (unchanged) |

Raw: `perf/results/2026-08-15/plain-step-decomp/e2e_fused_cool.txt` and
`e2e_legacy_cool.txt`. Serving-side ceiling remains ~50-62 tok/s with the
current drafter (see `perf/drafter_requirements.md` for the 100 tok/s
drafter gate).

### Long-Context Arm Added - 2026-08-15 (rested protocol, final build)

| Metric | Value |
| --- | ---: |
| Short-ctx decode (run 1/2/3) | 19.97 / 19.97 / 19.86 tok/s |
| 10k-ctx cached decode (run 2/3) | 10.97 / 11.00 tok/s |
| Acceptance (mixed workload) | 1.66/draft |

Long-context numbers reflect CORRECT global-layer attention (the eager
window-clamp bug is fixed) plus the multi-query verify kernel inside the
fused encoder. Bench discipline: decode from cached-prompt repeat runs
only; long arms use this 10k form until a natural-document corpus arm
lands. Raw: perf/results/2026-08-15/plain-step-decomp/e2e_final_rested.txt.

## Qwen3.8-27B (qwen38-q2kxl-1, Metal M5 Max) - campaign started 2026-08-19

### llama.cpp Reference Bar - 2026-08-20 (build-qwen38, master ece963f41)

| Metric | Value |
| --- | ---: |
| Plain decode (run 1/2/3, matched positions) | 35.67 / 35.37 / 33.55 tok/s |
| Plain decode, code prompt | 34.35 tok/s |
| Greedy determinism (3 runs) | identical |

SlimServe serving path is under construction (profile gated
in-progress); no SlimServe baseline yet. Vendor DFlash 2 numbers for
this pairing: mean acceptance 4.80 at block 8, 2.7-3.4x over plain at
batch 1 => the spec target implied by the llama.cpp step time is
~90-120 tok/s. Raw: perf/results/2026-08-20/qwen38-llamacpp-ref/.

### First SlimServe Baseline - 2026-08-20 (plain decode, bring-up build)

| Metric | Value |
| --- | ---: |
| Plain decode (run 1/2/3) | 2.52 / 2.51 / 2.47 tok/s |
| Greedy parity vs llama.cpp | 504 chars identical, then forks (fp16 dequant numerics) |
| Layer parity vs llama.cpp | all 64 layers cos >= 0.9997 |

Known-attributed gap vs the 35.67 llama.cpp bar: sequential-python GDN
scan (dispatch-bound ~8.7x) + fp16 dequant bytes (2.3x). Raw:
perf/results/2026-08-20/qwen38-first-e2e/.

NOTE 2026-08-20: the "plain decode 2.52/2.51/2.47" row above was measured
at temperature 0, which is not a served configuration (greedy is banned
stack-wide -- user directive; see notebook (20)). Step-time attribution
stands; the protocol going forward is the model's shipped sampling
defaults (temp 1.0 / top_p 0.95 / top_k 20), seeded. Re-baseline lands
with the speculation measurements.

### Correctness-Complete Build - 2026-08-20 evening (V2 runner, native IQ, layout fix)

| Metric | Value |
| --- | ---: |
| Plain decode, essay (run 1/2/3) | 15.26 / 14.98 / 14.82 tok/s |
| Plain decode, GSM8K-style | 14.44 / 14.00 / 13.43 tok/s |
| Spec decode, essay | 4.06 / 3.80 / 3.50 tok/s (OPEN BUG: < plain) |
| Spec decode, GSM8K-style | 9.45 / 7.97 / 8.65 tok/s (OPEN BUG: < plain) |
| Spec acceptance, essay (Prometheus, notebook 25) | 2.71 mean / 0.244 draft rate (beats llama.cpp dflash2-pr 2.51/0.219) |
| Corruption gauntlet | 7/7 clean boots, 24/24 same-seed pairs identical |
| Layer parity vs llama.cpp | all 64 layers cos >= 0.9997 |

Sampling: shipped defaults (temp 1.0 / top_p 0.95 / top_k 20), seed 42,
in-process V2 runner, max_model_len 8192. llama.cpp plain bar: 35.67.
Raw: perf/results/2026-08-20/qwen38-consolidated/. The 2.5 tok/s row
above is superseded.

### Supported SlimServe Baseline - 2026-08-23 (fused Metal stack, DFlash k=3)

| Metric | Value |
| --- | ---: |
| Plain essay, 3x256 | 16.99 / 17.14 / 17.17 tok/s |
| Spec essay, 3x256 | 23.27 / 23.74 / 23.06 tok/s |
| Plain GSM8K-style, 3x256 | 16.77 / 16.86 / 16.81 tok/s |
| Spec GSM8K-style, 3x256 | 34.33 / 35.25 / 34.78 tok/s |
| Exact server, spec (128 in / 256 out, c1) | 18.646 tok/s |
| Exact server, plain (128 in / 256 out, c1) | 15.913 tok/s |
| Exact server spec advantage | 17.2% |

Sampling is the shipped configuration (temperature 1.0, top-p 0.95, top-k
20), seeded 42; greedy is not used. Offline rows use the V2 runner at
max_model_len 8192 for matched short-context comparison. The exact-server pair
uses the registered profile unchanged: 131072 max length, 12 GiB KV pool,
DFlash 2 k=3, OpenAI completion endpoint, 8-token warmup. Both server arms
honored exact token counts and produced healthy text. The real SlimServe server
also passed text and deterministic image requests.

Correctness: all 64 target layers remain cosine >=0.9997 against the llama.cpp
activation oracle; fused GDN exhaustive suite 147/147; the 64-bit hybrid KV
gather is exact before and beyond 2^31 source elements; 20 repeated speculative
requests stay finite and token-stable across the former corruption window;
final focused suite 17 passed and SlimServe suite 58 passed/1 skipped. Raw:
`perf/results/2026-08-23/qwen38-kv-gather/`.

## Qwen3.8-27B NVFP4

### MI300X Single-GPU Exact Baseline - 2026-08-19 (optimization pass 2)

- As below, plus: fused NVFP4-QDQ+Q8_1 activation kernel and bf16 GEMV
  epilogue (bit-identical, fewer kernels), nontemporal weight streaming, and
  aiter-tuned FP8 GEMM configs for this model's decode shapes
  (AiterPreshuffled kernels; lm_head stays RowWiseTorch).
- Workload: hot server, 1,000 input tokens, exactly 2,000 output tokens.

| Concurrency | Aggregate tok/s | Exact |
| ---: | ---: | --- |
| 1 | 170.92 / 170.99 | yes |
| 8 | 875.71 / 912.41 | yes |

Raw results: `perf/results/2026-08-18/qwen38-nvfp4-1-mi300x-perf/`
(opt2_*/opt3_* files).

### Superseded MI300X Single-GPU Exact Baseline - 2026-08-19 (native kernel)

- Model: unsloth/Qwen3.8-27B-NVFP4; profile `qwen38-nvfp4-1`, TP1, 262,144
  context, MTP spec k=2, FULL_DECODE_ONLY graphs (capture 64). Decode-width
  NVFP4 GEMMs run the vendored QuixiCore HIP packed-E2M1 q8 GEMV
  (csrc/quixicore/tm_rocm/qc_rocm_nvfp4.cu); wider batches use the
  load-time-dequantized bf16 copy through hipBLASLt.
- Workload: hot server, 1,000 input tokens, exactly 2,000 output tokens.

| Concurrency | Aggregate tok/s | Exact |
| ---: | ---: | --- |
| 1 | 157.09 / 157.10 | yes |
| 8 | 911.63 / 887.77 | yes |

Raw results: `perf/results/2026-08-18/qwen38-nvfp4-1-mi300x-perf/`
(native_* files). Headroom: MFMA packed MMQ for M >= 8 (retires the bf16
copy), aiter FP8 tuned configs for this model's decode shapes, drafter graph
coverage.

### Superseded MI300X Single-GPU Exact Baseline - 2026-08-18 (tuned)

- Model: unsloth/Qwen3.8-27B-NVFP4; profile `qwen38-nvfp4-1`, TP1, 262,144
  context, MTP spec k=2, FULL_DECODE_ONLY graphs (capture 64), NVFP4
  emulation with load-time dequant cache (bit-identical, ~22 GiB extra VRAM).
- Workload: hot server, 1,000 input tokens, exactly 2,000 output tokens.

| Concurrency | Aggregate tok/s | Exact |
| ---: | ---: | --- |
| 1 | 144.61 / 144.65 | yes |
| 8 | 822.89 / 952.79 | yes |

Raw results: `perf/results/2026-08-18/qwen38-nvfp4-1-mi300x-perf/`.

### Superseded MI300X Single-GPU Exact Baseline - 2026-08-18 (initial)

- Model: unsloth/Qwen3.8-27B-NVFP4 (compressed-tensors mixed FP8 + NVFP4
  safetensors; vision tower bf16; built-in one-layer MTP drafter).
- Profile: `qwen38-nvfp4-1`, TP1 on 1x MI300X, 262,144-token context,
  kv_cache_dtype auto, linear_backend auto, CUDA graphs off, MTP spec k=2.
- Workload: hot server, 1,000 input tokens, exactly 2,000 output tokens.
- NVFP4 MLP weights decode through the ROCm emulation kernel (gfx942 has no
  FP4 hardware); FP8 GEMMs run RowWiseTorch hipBLASLt (aiter shapes untuned).

| Concurrency | Aggregate tok/s | Median wall s | Exact | Spec acceptance |
| ---: | ---: | ---: | --- | ---: |
| 1 | 30.67 / 35.73 | 65.20 / 55.97 | yes | 81% / 99.6% |
| 8 | 230.57 / 227.20 | 60.35 / 58.52 | yes | 96.4% / 95.1% |

Raw results: `perf/results/2026-08-18/qwen38-nvfp4-1-mi300x-baseline/`.
Headroom (unmeasured): native gfx942 NVFP4 decode, aiter FP8 shape tuning,
graph capture for the hybrid GDN+MTP decode, Gemma-aware fused norm+quant.
