# SlimServe Baseline Status

This file holds stable baseline snapshots for comparison. Raw outputs belong in
`perf/results/`; summarize only the numbers needed to compare future work.

## Qwen3.8-Flash-Next (qwen4_exp)

### 4x RTX 3090 - qwen38fn-nvfp4-4 first validated record - 2026-09-06 (host-resident main KV @ native 262K)

- nvidia/Qwen3.8-Flash-Next-NVFP4 (ModelOpt mixed: NVFP4 experts via
  Marlin W4A16, FP8 drafter experts via Marlin W8A16, FP8 PLE tables
  host-pinned), TP4 + EP, fp8 main KV, max_model_len 262144, util 0.97,
  max_num_seqs 16, capture 48, MTP k=2, host tier 88 + NVMe 448 GiB/rank.
- Three-tier redesign (docs/host_resident_kv_design.md): the 13 QSA main-KV
  layers live in pinned host rows gathered over PCIe; 24 GPU hot rows;
  attention block 12,688 tokens; GPU pool 531,186 tokens = 2.03 max-length
  requests of resident state (util 0.975, max_num_seqs 8, capture 24).
- Gates passed: marker recall 54K/144K/243K, multi-turn tracking 3/3,
  image canary, 0 server errors.
- Workload: exact 1,000 in / 2,000 out, temp 1.0 / top_p 0.95 / top_k 20,
  seed 42, port 8001, idle box:

| Concurrency | Aggregate tok/s | Draft acceptance |
| ---: | ---: | ---: |
| 1 | 120.1 | - |
| 8 | 361.7 (5 running) | - |
| 16 | 335.1 at the earlier util 0.97 / seqs 16 config (4 running) | 57.9% |

- c16 == c8 because the packed slab admits ~4 running requests: each one
  charges ~18 rows x 10.56 MB (GDN align-mode state blocks + ring +
  attention) against the 0.78 GiB pool, so both numbers are effectively
  c4. A 64-row hot window measured 35% slower (2 running). Tuning of the
  pool (util 0.975, 16 rows, 6 seqs) recorded in the notebook.
- Raw: perf/results/2026-09-06/qwen38fn-nvfp4-4/.

### 8x RTX 3090 Current Baseline - 2026-09-02 (fp8 main KV @ native 262K, host tier on)

- Registered profile as deployed: TP8+EP, max_model_len 262144, main QSA
  KV fp8 e4m3 (`kv_cache_dtype: fp8`; operator 2026-09-02, replaces the
  bf16 mandate; NOT TurboQuant), gpu_memory_utilization 0.96, prefix
  caching, HostTierConnector 88 GiB/rank (15,773 slots, ~12.6M tokens)
  + NVMe tier 448 GiB/rank on the dedicated nvme1n1 (80,273 slots, ~64M
  tokens), max_num_seqs 32, capture 96,
  MTP k=2 + index share, thinking budget 2000.
- GPU KV pool: 3.64 GiB/rank = 496,174 tokens = 1.89x one max-length
  request (bf16 held 269,155 = 1.03x). Attention block 800 tokens.
- Workload: exact 1,000 in / 2,000 out, temp 1.0 / top_p 0.95 / top_k 20,
  temp server on port 8001, idle box.

| Concurrency | Aggregate tok/s (seed 42 / 43) | Draft acceptance |
| ---: | ---: | ---: |
| 1 | 135.7 | - |
| 8 | 594.2 / 590.4 (610.2 with the NVMe tier writing through, 2026-09-03) | 63.9-66.9% |
| 32 | 1,016.6 / 1,100.9 | 58.5-61.9% |

- vs the bf16 deployed reference (c1 129.8-157.8, c8 590.7-600.5, c32
  1,091.8-1,199.8): c1/c8 parity, c32 ~7-8% lower seed for seed.
- Correctness gates passed: tier eviction-restore marker recall, 6-turn
  multi-turn state tracking x3 seeds.
- Raw: perf/results/2026-09-02/qwen38fn-fp8kv/.

### 8x RTX 3090 Current Baseline - 2026-08-27 evening (KV-pool fix + correctness fixes)

- Same registered profile as the "Final Baseline" below plus:
  gpu_memory_utilization 0.95 (KV pool 2.23 -> 3.41 GiB/rank, 590,324
  tokens; the packed slab at 0.9 silently capped Running at 27 of 32),
  VLLM_ADMISSION_MAX_CONCURRENT=96 (429 + Retry-After past a bounded
  queue), the fresh-request classification and cudagraph dispatch-gate
  correctness fixes (1- and 3-token prompts produced garbage and could
  crash the engine), and the sliced PLE dilated-conv prefill transient
  (multi-turn prefill waves OOM'd at util 0.95).
- Workload: exact 1,000 in / 2,000 out, temp 1.0 / top_p 0.95 / top_k 20,
  seed 42, manual serve on an idle box (no QA traffic).

| Concurrency | Aggregate tok/s | vs prior baseline |
| ---: | ---: | ---: |
| 1 | 151.8 | +12-16% |
| 32 | 1,134.4 | +28.7% |

- Running peaks at the full 32 (Waiting 0, KV peak 86.9%); the prior
  "c32" rows below were effectively c27 plus a queue.
- Real-traffic reference (WildChat-1M multi-turn replay, 128 sessions,
  504 turns, thinking on): 32 concurrent sessions sustain ~780 out tok/s
  pre-fix; oversubscription to 64 buys zero throughput and 3.5x TTFT.
- Raw: perf/results/2026-08-27/qwen38fn-service-wildchat/.

### 8x RTX 3090 Final Baseline - 2026-08-27 (TQ main-KV @ native 262K)

- Profile `qwen38fn-fp8-8` as registered: TP8+EP, max_model_len 262144
  (native), TurboQuant k8v4 main QSA KV (VLLM_QWEN4_EXP_TQ_MAIN_KV=1,
  2.64x vs bf16), max_num_seqs 32, FULL_DECODE_ONLY graphs with
  max_cudagraph_capture_size 96, MTP k=2 + index share, triton GDN,
  NCCL_P2P_LEVEL=SYS on the QuixiAI P2P driver (32 GiB BAR1, iommu=pt).
- Includes the QSA quantized-KV tile-tier fix (narrow tiles + 4 warps for
  TQ/fp8 paths); every TQ/fp8 serving number recorded before 2026-08-27
  ~10:00 was afflicted by the wide-tile collapse and is superseded.
- Workload: exact 1,000 input / 2,000 output tokens, temp 1.0 / top_p
  0.95 / top_k 20, seed 42. slimserve-launched validation run:

| Concurrency | Aggregate tok/s | Draft acceptance |
| ---: | ---: | ---: |
| 1 | 130.9-136.2 | 58.9% |
| 8 | 547.0-585.0 | 58.1% |
| 16 | 838.2 | 62.4% |
| 32 | 879.6-881.6 | 64.5% |

- bf16 KV @131072 measures the same within noise at c8-c32 (587/870/861)
  and ~8% faster at c1 (148.8); it cannot fit the native 262144.
- Campaign net: bring-up (SHM collectives, bf16, 128K, c8-limited) peaked
  at 409.7 tok/s; the final config reaches 881.6 at c32 with double the
  context - 2.15x, from the P2P fabric (+36-58%), the capture-size fix
  (c32 +55%), and the quantized-KV tile fix (c16 +29%, TQ made ~free).
- Raw: perf/results/2026-08-27/qwen38fn-3090-p2p/ (bench_expP_*,
  bench_final_*, cliff_exp*.log chain documents the root-cause work).

### 8x RTX 3090 FP8 P2P Baseline - 2026-08-27 (historical)

- Same model/profile/workload as the 2026-08-26 bring-up baseline below,
  after the collective-fabric overhaul: QuixiAI P2P driver 610.57.04,
  32 GiB BAR1 on all GPUs (per-boot resize service), `iommu=pt`,
  NCCL_P2P_LEVEL=SYS (24.7 GB/s all-reduce busbw vs 2.7 SHM), plus the
  Phase-A winners pinned in the profile (triton GDN decode,
  index_share_for_mtp_iteration, MTP k=2).

| Concurrency | Aggregate tok/s | vs bring-up |
| ---: | ---: | ---: |
| 1 | 148.81 | +35.8% |
| 8 | 647.88 | +58.2% |

Raw results: perf/results/2026-08-27/qwen38fn-3090-p2p/. This is the
current reference for all qwen38fn-fp8-8 comparisons; the 2026-08-26 table
below is historical (no-P2P SHM collectives).

### 8x RTX 3090 FP8 Bring-up Baseline - 2026-08-26 (historical)

- Model: Qwen/Qwen3.8-Flash-Next-FP8 (block-128 FP8 experts, BF16 backbone,
  47.7 GiB FP8 n-gram table in pinned host memory via VLLM_QWEN4_EXP_PLE_HOST).
- Profile: `qwen38fn-fp8-8` -- TP8 + expert parallel, Marlin W8A16 MoE,
  max_model_len 131072, FULL prefill+decode graphs, MTP k=2.
- Workload: exact 1,000 input tokens, exact 2,000 output tokens, shipped
  thinking-mode sampling (temp 1.0 / top_p 0.95 / top_k 20), seed 42.

| Concurrency | Aggregate tok/s | Mean wall s | Exact | Draft acceptance |
| ---: | ---: | ---: | --- | ---: |
| 1 | 109.61 | 18.25 | yes | 54.2% |
| 2 | 206.77 | 18.08 | yes | 73.0% |
| 4 | 283.30 | 27.37 | yes | 67.8% |
| 8 | 409.66 | 37.44 | yes | 62.4% |

Raw results: perf/results/2026-08-26/qwen38fn-3090-bringup/. This is the
first live baseline on this platform; the no-P2P collective ceiling has not
yet been profiled, so treat it as a bring-up baseline, not an optimized one.

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

## Qwen3.8-27B — Metal M1 Ultra (qwen38-1), M2 baseline pinned 2026-08-17

- Status: first exact-token baseline for the bring-up path — GGUF Q4_K_M
  through the forced-V2 runner on torch-MPS GDN fallbacks (NO Metal GDN
  kernels yet; this is the floor M3 attacks, pinned so the kernel win is
  measurable). Worktree b1ee83671 + uncommitted M1 implementation.
- Profile: `qwen38-1` (V2 forced, kv pool 16 GiB, max_num_seqs 8,
  max_model_len 32768, no drafter, prefix caching off). Wired limit 122880.
- Harness: `benchmarks/benchmark_dsv4_exact.py --metrics-url none`,
  prompt overhead 0 (verified: no BOS/wrapper), source = perf-docs concat
  (25,368 tokens). All runs `exact: true`.

| workload | tok/s (aggregate) |
| --- | ---: |
| c1 1000 in / 256 out (x3, warmed boot) | 2.425–2.433 |
| c1 1000 in / 256 out (fresh boot 6) | 2.368 |
| c4 1000 in / 256 out (x2) | 5.387–5.389 |
| c8 1000 in / 256 out (x2) | 6.554–6.558 |
| c1 2500 in / 64 out (x3) | 1.110–1.115 |

- Derived (two-workload solve): pure c1 decode ~2.80 tok/s, prefill
  ~72 tok/s.
- Determinism: c1 completion sha 7c766419 identical across all four runs
  including the fresh boot. c4/c8 repeats fork at documented near-ties.
- Cross-engine bars, same box, sequential: llama.cpp d8df12e Metal on the
  IDENTICAL GGUF — pp1000 252.2 / pp2500 250.9 / tg256 21.03 tok/s;
  batched-bench 1000/256 B=1/4/8: 20.90 / 35.00 / 40.54 agg. MLX bf16:
  prompt 221, generation 10.85 tok/s (55.5 GB peak). llama.cpp = 7.5x our
  pure decode, 3.5x our prefill on the same weights.
- First-request-after-boot wobble (2/2 boots): the first forward after a
  fresh boot flips near-ties (<=0.5 nats observed); prime every boot with
  a throwaway request before determinism-sensitive runs. Open root-cause
  item, tracked for M3.
- Raw: perf/results/2026-08-17/qwen38_m2_baseline/. Notebook: UPDATE 5 in
  the Qwen3.8 bring-up entry of perf/optimization_status.md.

## Qwen3.8-27B — Metal M1 Ultra, M3 kernel baseline pinned 2026-08-17

- Change vs the M2 floor: the five gdn.metal kernels serve the GDN mixer
  (VLLM_METAL_GDN kill switch; fp32 chain; state_stride kernel param for
  the page-packed pools). Oracle 25/25 vs the torch fallbacks; c1
  completion sha 7c766419 identical to the fallback canonical.

| workload | tok/s (M2 floor) | tok/s (M3 kernels) |
| --- | ---: | ---: |
| c1 1000/256 | 2.43 | 7.205–7.211 |
| c4 1000/256 | 5.39 | 11.42–11.44 |
| c8 1000/256 | 6.56 | 11.86–11.90 |
| c1 2500/64 | 1.11 | 2.80–2.81 |

- Derived: pure c1 decode 8.73 tok/s (3.1x), prefill 161 tok/s (2.2x);
  42% / 64% of llama.cpp's same-GGUF 20.9 / 252.
- Parity re-gated (llama.cpp same GGUF): unchanged from the M1 gate
  within near-ties. Watch items and the one-shot first-6-chunk transient
  are in notebook UPDATE 7.
- Raw: perf/results/2026-08-17/qwen38_m3_kernels/.

## Qwen3.8-27B — Metal M1 Ultra, M4 baseline pinned 2026-08-18 (norm kernel + KV page-layout fix)

- Changes vs M3: gdn_gated_rmsnorm wired into serving, and the hybrid
  KV page-layout collision fixed (MetalAttentionBackend stride order
  (1,0,2,3,4) = page-local [num_blocks, 2, block, H, D]; dense-view
  addressing for the paged kernel and all cache reads/writes). The fix
  closed three correctness watch items (first-multi-chunk '!' transient,
  first-request wobble, 2500x64 intra-boot sha re-roll) — root cause and
  hunt in notebook UPDATE 8.

| workload | M3 kernels | M4 norm+layout |
| --- | ---: | ---: |
| c1 1000/256 | 7.21 | 8.748–8.749 |
| c4 1000/256 | 11.43 | 15.789 |
| c8 1000/256 | 11.88 | 16.521 |
| c1 2500/64 | 2.81 | 3.032–3.037 |

- Derived: pure c1 decode 11.06 tok/s (+27% vs M3), prefill ~164 tok/s;
  53% / 65% of llama.cpp same-GGUF 20.9 / 252. c8/c1 scaling 1.89x
  (llama.cpp: 1.9x). c1 step 90.4 ms (llama.cpp whole model 47.6 ms).
- Canonical greedy shas (temperature 0, exact harness prompts):
  c1 1000x256 = 2e4defe1 (norm-kernel canonical, preserved bit-exact
  across the layout fix); c1 2500x64 = f687018e (3/3 bit-stable).
  First request on a fresh boot is now bit-identical to warm repeats
  (max |dlogprob| 0.000000, 2/2 boots) — priming stays protocol until a
  longer soak says otherwise.
- DSV4 exposure: none (python-only change; MLA/TurboQuant backends
  untouched; no metallib rebuild).
- Raw: perf/results/2026-08-18/qwen38_m4_layoutfix/. Notebook: UPDATE 8.

## Qwen3.8-27B — Metal M1 Ultra, M4b baseline pinned 2026-08-18 (Q4_K mmvq kernel)

- Change vs M4: qgemv_q4k_nr — llama.cpp-layout q4_K decode GEMV
  (2 rows/simdgroup, 2 simdgroups/threadgroup, 4 blocks in flight,
  factored scales, no per-element half rounding), routed for q4_K
  batch-1 GEMVs; VLLM_QC_Q4K_NR=0 kill switch. metallib + .so rebuilt;
  DSV4 anchors re-gated ALL BIT-EXACT (anchor_regate_q4k_metallib/).
  Census that found it: UPDATE 9; kernel gate: UPDATE 10.

| workload | M4 norm+layout | M4b q4_K kernel |
| --- | ---: | ---: |
| c1 1000/256 | 8.749 | 11.197 (2/2) |
| c4 1000/256 | 15.789 | 15.674 (flat; mm path) |
| c8 1000/256 | 16.521 | 16.513 (flat; mm path) |
| c1 2500/64 | 3.037 | 3.238–3.261 (3/3) |

- Derived: pure c1 decode ~15.3 tok/s (+38% vs M4) = 73% of llama.cpp
  same-GGUF 20.9; c1 step ~65.5 ms (llama.cpp whole model 47.6) =
  35.1 ms idealized GEMV + ~31 ms non-GEMV (census closes to
  measurement). Prefill unchanged (~164 tok/s; M>17 GEMM path untouched).
- Canonical greedy shas: c1 1000x256 = 36ed113a (2/2; fork-gap verified
  vs 2e4defe1 — exact tie at pos 32 nudged 0.125 nats, pre-fork max
  0.168 nats); c1 2500x64 = 268721b3 (3/3). ramp 1000x32 = 0f9506fc
  (unchanged). Kill-switch boot reproduces 2e4defe1 bit-exact.
- Known flat spots: c4/c8 ride qgemv_mm (BPI=1 walk) — next lever;
  c8/c1 ratio now 1.47x pending that work.
- Raw: perf/results/2026-08-18/qwen38_q4k_gate/. Notebook: UPDATE 10.

## Qwen3.8-27B — Metal M1 Ultra, M4c baseline pinned 2026-08-18 (Q4_K batch mmvq kernel)

- Change vs M4b: qgemv_q4k_nr_mb — grid-split column-pair batch twin of
  the NR kernel (grid.y = M/2, batch-1 register budget per threadgroup,
  bit-identical per row to looped batch-1), routed for q4_K qgemv_mm
  chunks M in {2,4,8}; VLLM_QC_Q4K_NR_MM=0 kill switch (VLLM_QC_Q4K_NR=0
  still kills all NR routes). metallib + .so rebuilt; DSV4 anchors
  re-gated ALL BIT-EXACT (anchor_regate_q4k_mb/). Investigation +
  gate: UPDATE 11.

| workload | M4b | M4c batch kernel |
| --- | ---: | ---: |
| c1 1000/256 | 11.197 | 11.168 (sha 36ed113a bit-exact) |
| c4 1000/256 | 15.674 | 16.596-16.609 (+5.9%) |
| c8 1000/256 | 16.513 | 17.414-17.421 (+5.5%) |
| c1 2500/64 | 3.238-3.261 | 3.270 (sha 268721b3 bit-exact) |

- Batch q4_K microbench (serving route, GB/s gate_up/down): M=2
  247/188 -> 414/382; M=4 171/138 -> 217/211; M=8 91/81 -> 111/112.
  Idealized all-GEMV step at M=8: 204.5 -> 181.1 ms. Batch decode is
  now ALU-bound per column (UPDATE 11 cost model), and the q8_0/q6_K
  batch paths carry the same M=8 collapse (qkvz 107, attn.o 58 GB/s) —
  that plus non-GEMV are the open levers.
- Canonical shas unchanged: c1 36ed113a, 2500x64 268721b3, ramp
  0f9506fc. c4/c8 request-0 shas are tie-carriers (composition
  sensitive in both worlds) — NOT anchors; kill-switch boot reproduces
  old-baseline throughput (15.772/16.450) and the non-tie c4 request-1
  sha fbe69266 bit-exact.
- c8/c1 ratio 1.56x (was 1.47x). llama.cpp bars unchanged (c1 20.9).
- Raw: perf/results/2026-08-18/qwen38_q4k_mb_gate/. Notebook: UPDATE 11.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N2 baseline pinned 2026-08-19 (bring-up dequant path, pre-kernel)

- First exact-token baseline of `qwen38-nvfp4-1` (unsloth NVFP4
  compressed-tensors checkpoint, mixed FP8+NVFP4, served W4A16/W8A16).
  Path under measurement: N1 bring-up — weights dequantized once to
  bf16 at load (~40 GB resident), every linear a plain F.linear; NO
  QuixiCore GEMV kernels on the CT path yet. Boot ~35 s page-cached.
  Bring-up record: UPDATE 13; this snapshot: UPDATE 14.

| workload | Q4_K M4c (bar) | NVFP4 N2 bring-up |
| --- | ---: | ---: |
| c1 1000/256 | 11.168 | 7.02 / 6.99 / 7.02 (sha 8c58a4c6, 3/3 bit-stable) |
| c4 1000/256 | 16.596-16.609 | 14.370 / 14.356 |
| c8 1000/256 | 17.414-17.421 | 15.933 / 15.993 |
| c1 2500/64 | 3.270 | 2.552 / 2.550 / 2.556 (sha d0e07ddd, 3/3 bit-stable) |

- Canonical NVFP4 greedy shas: c1 1000x256 = 8c58a4c6 (3/3), c1
  2500x64 = d0e07ddd (3/3). c4/c8 request-0 shas are tie-carriers as
  in the Q4_K world (c4 run 2 flipped: 8c58a4c6 -> 8a969e98) — NOT
  anchors. c8 sha 6cad1354 was 2/2 but inherits the same caveat.
- Read: c1 at 63% of Q4_K (bf16-materialized reads ~40 GB/tok vs
  16.5), while c4/c8 already at 87%/92% (aten bf16 GEMM batches well).
  The kernel campaign's c1 target: 19.1 GB/tok real quantized reads
  put the GEMV floor at ~27 ms + ~31 ms non-GEMV ⇒ ~17 tok/s cap;
  every point between 7.02 and that cap is N3/N4 work.
- Prefill (2500x64): ~130 tok/s prefill-equivalent — N4 lever, noted
  not tuned.
- Raw: perf/results/2026-08-19/n2_nvfp4_baseline/. Notebook: UPDATE 14.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N3 baseline pinned 2026-08-19 (first CT GEMV kernels)

- Change vs N2: two new QuixiCore Metal GEMV kernels routed for M==1
  decode on the compressed-tensors path — `qgemv_fp8ch` (planar e4m3
  [N,K] × bf16 X, per-row scale epilogue; covers attn qkvo, GDN
  qkvz/out, mlp 56–63, lm_head) and `qgemv_nvfp4_planar` v6
  (packed-nibble [N,K/2] + e4m3 group scales [N,K/16]; select-free
  bit-pattern E2M1 decode, 2^14 rebias folded into the group-scale
  multiply; covers mlp 0–55). Kill switches: VLLM_QC_FP8CH=0,
  VLLM_QC_NVFP4=0. metallib + .so rebuilt. Variant study, serving
  gates, and the ops incident: UPDATE 15.

| workload | N2 bring-up | N3 GEMV kernels |
| --- | ---: | ---: |
| c1 1000/256 | 7.02 | 10.10 / 10.14 (sha 2e567ea7, 2/2) |
| c4 1000/256 | 14.37 | 14.38 (unchanged by design) |
| c8 1000/256 | 15.96 | 16.02 (unchanged by design) |
| c1 2500/64 | 2.55 | 2.82 / 2.83 (sha 95adbd97) |

- c1 +44% over N2; now 90% of the Q4_K M4c bar (11.168). Ramp-32
  indicative 14.50 tok/s.
- Correctness: float64 oracles ALL PASS for both kernels (maxrel
  3.8–5.7e-3, fp32-tree + bf16-rounding class; bit-stable across
  repeat calls; batch call == looped batch-1 bit-exact). Kill-switch
  boot (both routes off) reproduces the N2 canonicals BIT-EXACT
  through all new plumbing: c1 7.033 sha 8c58a4c6, 2500x64 d0e07ddd.
- M==1 gate is deliberate: routing the M≤8 batch loop through the
  GEMVs measured BELOW dense bf16 GEMM (c4 14.37→13.00, c8
  15.93→14.00 — the batch-1 loop re-reads weights M times); gating to
  M==1 recovered c4/c8. Weight-stationary `_mb` twins (q4_K UPDATE 11
  grid.y column-pair pattern) are the open c4/c8 lever.
- Clean-box microbench (GB/s): fp8ch qkv 471 / o 461 / gdn.qkvz 573 /
  gdn.out 479 / gate_up 615 / down 527 / lm_head 650 — idealized FP8
  side 19.4 ms vs 15.1 ms roofline; nvfp4_planar gate_up 426 / down
  447 — idealized NVFP4 side 19.5 ms vs 12.0 ms roofline. Combined
  idealized all-GEMV step 38.9 ms vs the ~27 ms roofline target;
  at 10.1 tok/s the step is ~99 ms, so non-GEMV (~31 ms class) plus
  GEMV shortfall are the next c1 levers (N3 geometry work + N4 #16).
- DSV4 exposure: metallib/.so rebuilt twice. Anchors re-gated ALL
  BIT-EXACT after the fp8ch build
  (perf/results/2026-08-19/anchor_regate_fp8ch/: 573db39598e7 5/15/3,
  bb83cc3054a3 1581/2115/423 57.3s, f75e1d41ac3d 43/105/21) and again
  ALL BIT-EXACT on the final nvfp4_planar build
  (perf/results/2026-08-19/anchor_regate_nvfp4_final/, 2/2 + 2/2 +
  3/3 deterministic).
- Raw: perf/results/2026-08-19/n3a_fp8ch_gate/ and
  n3b_nvfp4_gate/. Notebook: UPDATE 15.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N3b baseline pinned 2026-08-19 (mb batch twins)

- Change vs N3: `qgemv_fp8ch_mb` + `qgemv_nvfp4_planar_mb` — grid-split
  column-pair batch twins (grid.y = M/2, batch-1 register budget,
  weights decoded once per pair, rows bit-identical to looped batch-1
  by construction and by oracle). Routed for even M: NVFP4 M<=8, FP8
  M<=4 (at 8 bits the M/2 re-reads reach dense-bf16 parity by M=8 —
  measured crossover). Kill switches VLLM_QC_NVFP4_MB /
  VLLM_QC_FP8CH_MB. metallib + .so rebuilt; DSV4 anchors re-gated
  (anchor_regate_ct_mb/).

| workload | N3 GEMV kernels | N3b mb twins |
| --- | ---: | ---: |
| c1 1000/256 | 10.10 / 10.14 | 10.024 / 10.049 (sha 2e567ea7 bit-exact) |
| c4 1000/256 | 14.38 | 15.638 / 15.636 (+8.8%) |
| c8 1000/256 | 16.02 | 16.889 / 16.876 (+5.4%) |
| c1 2500/64 | 2.82 / 2.83 | 2.853 / 2.858 (sha 95adbd97 bit-exact) |

- Canonical shas unchanged (c1 2e567ea7, 2500x64 95adbd97). c4/c8
  request-0 shas remain tie-carriers; the c4 carrier rolled dense ->
  GEMV numerics (8c58a4c6 -> 467b35c3), stable 2/2. Null-check boot
  (both MB switches off) reproduces N3 exactly (10.023 / 14.354 /
  15.89, c1 sha 2e567ea7).
- vs Q4_K M4c bar: c1 90% (10.05/11.17), c4 94% (15.64/16.60), c8 97%
  (16.89/17.42). Remaining c1 levers: fp8ch decode-trick port +
  geometry (idealized all-GEMV 38.9 ms vs 27 ms roofline), then the
  ~31 ms non-GEMV class (N4 #16).
- Raw: perf/results/2026-08-19/n3c_mb_gate/. Notebook: UPDATE 16.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N3d baseline pinned 2026-08-19 (fp8ch v6-decode port; c8 passes the Q4_K bar)

- Change vs N3b: `qgemv_fp8ch`/`_mb` decode is now the select-free v6
  bit-pattern form (two half2 patterns per uint of 4 e4m3 bytes =
  value/2^8 exactly, subnormal-safe; 2^8 folded into the per-row scale
  epilogue; vec4 X staging) — microbench qkv 471->687, o 461->759,
  lm_head 650->732 GB/s (DRAM-honest), idealized FP8-GEMV step 19.4 ->
  13.0 ms. Routing: `_GEMV_MB_MAX_ROWS` (fp8ch) 4 -> 8 — the cheap
  decode moves the mb-vs-dense crossover (M=8 qkv 0.285 vs 0.455 ms).
  metallib rebuilt (.so unchanged); DSV4 anchors re-gated ALL
  BIT-EXACT (anchor_regate_v6port/).

| workload | N3b mb twins | N3d v6-decode port |
| --- | ---: | ---: |
| c1 1000/256 | 10.024 / 10.049 | 10.357 / 10.338 (+2.9%, sha 467b35c3 2/2 — ROLLED, see note) |
| c4 1000/256 | 15.638 / 15.636 | 16.309 / 16.259 (+4.1%, sha 467b35c3 HELD) |
| c8 1000/256 | 16.889 / 16.876 | 17.538 / 17.543 (+3.9%) |
| c1 2500/64 | 2.853 / 2.858 | 2.849 / 2.858 (sha 95adbd97 HELD bit-exact) |

- CANONICAL SHA CHANGE: c1 1000x256 = **467b35c3** (was 2e567ea7).
  Per-element decode is provably exact (float64 envelope unchanged);
  the roll is summation-order rounding — the straight-line decode
  lets fast-math reassociate the FMA chain the old select chain
  blocked. Deterministic 2/2; 2500x64 held bit-exact (64 steps, no
  tie hit); c4 held. Full attribution in UPDATE 18.
- vs Q4_K M4c bar: c1 92.7% (10.35/11.17), c4 98.1% (16.28/16.60),
  **c8 100.7% (17.54/17.42) — first bar crossed**. Remaining base-
  decode levers: nvfp4_planar geometry (529/452 GB/s vs the ~732
  DRAM-honest ceiling), then the ~31 ms non-GEMV class (N4 #16).
- Raw: perf/results/2026-08-19/n3d_v6port_gate/. Notebook: UPDATE 18.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N3e baseline pinned 2026-08-19 (nvfp4 v7 vectorized decode; c4+c8 past the Q4_K bar)

- Change vs N3d: `qgemv_nvfp4_planar`/`_mb` decode is half2-vectorized
  (four bit-pattern constructs per uint decode all 8 nibbles, no byte
  extraction, ~2.5 ops/value) and the group scale uses the select-free
  e4m3 pattern with one 2^22 fold. Microbench gate_up 529->631, down
  452->662 GB/s; idealized NVFP4 step 17.3->13.1 ms; **all-GEMV
  idealized ~26.1 ms = AT the ~27 ms campaign target**. No host
  change; nvfp4 mb bound stays 8. metallib rebuilt; DSV4 anchors
  re-gated ALL BIT-EXACT (anchor_regate_nvfp4v7/).

| workload | N3d | N3e v7 decode |
| --- | ---: | ---: |
| c1 1000/256 | 10.357 / 10.338 | 10.706 / 10.685 (+3.3%, sha fa58598b 2/2 — rolled as PREDICTED) |
| c4 1000/256 | 16.309 / 16.259 | 16.992 / 16.972 (+4.2%) |
| c8 1000/256 | 17.538 / 17.543 | 18.252 / 18.245 (+4.0%) |
| c1 2500/64 | 2.849 / 2.858 | 2.907 / 2.905 (sha 95adbd97 HELD bit-exact, +1.9%) |

- CANONICAL c1 sha: **fa58598b** (roll predicted in advance — same
  fast-math-reassociation class as N3d, attribution in UPDATE 19;
  2500x64 held bit-exact for the third consecutive build). c4/c8
  request-0 tie-carriers now vary per run — composition-sensitive,
  NOT anchors.
- vs Q4_K M4c bar: **c1 95.8% (10.70/11.17), c4 102.3% (16.98/16.60)
  CROSSED, c8 104.8% (18.25/17.42) CROSSED**. GEMV side is at target;
  the c1 gap is the ~31 ms non-GEMV class -> N4 census next
  (UPDATE 9 methodology on the current build).
- Raw: perf/results/2026-08-19/n3e_nvfp4v7_gate/. Notebook: UPDATE 19.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N4a baseline pinned 2026-08-19 (paged-attention D=256 split-K; c4 112% / c8 122% of the Q4_K bar)

- Change vs N3e: head-256 full-attn decode leaves the SDPA gather
  fallback for the split-K partition/reduce paged-attention pair
  (paged_attn_v2 @256; one serial-dispatch encoder; occupancy target
  1536, P<=64; new `max_context` op arg sizes partitions host-side).
  Crossover-routed (measured): batch >= 2 OR max_context >= 2048 ->
  kernel; batch-1 short context keeps SDPA (the kernel's fixed
  per-call cost loses there — c1 9.45/10.02 in rounds 1/2). Kill
  switch VLLM_QC_PA256=0; tuning VLLM_QC_PA256_SPLITK. metallib +
  .so rebuilt; DSV4 anchors ALL BIT-EXACT (anchor_regate_pa256/).

| workload | N3e | N4a split-K attn |
| --- | ---: | ---: |
| c1 1000/256 | 10.706 / 10.685 | 10.717 / 10.726 (sha fa58598b BIT-EXACT — SDPA route unchanged) |
| c4 1000/256 | 16.992 / 16.972 | 18.541 / 18.546 (+9.1%) |
| c8 1000/256 | 18.252 / 18.245 | 21.263 / 21.258 (+16.5%) |
| c1 2500/64 | 2.907 / 2.905 | 2.914 / 2.915 (kernel route, sha aedef4ec 2/2) |

- vs Q4_K M4c bar: c1 96.0% (10.72/11.17), **c4 111.7% (18.54/16.60)
  CROSSED, c8 122.1% (21.26/17.42) CROSSED**. ctx-32k decode step
  drops from unusable (SDPA gather) to 10.4 ms/call x16.
- Census (UPDATE 20): host exonerated (execute_model overlapped,
  0 sync kills, 19 cb/step, GPU p50 95); phase split mlp 33.5 /
  gdn 25.3 / full_attn 17.5 / glue 23.7. Remaining c1 levers by
  share: mlp glue (SwiGLU fusion), norms/residual glue, gdn
  non-GEMV multi-row recurrence — then N5 DFlash 2 (verify path =
  the expanded kernel route, already lit up by this change).
- Raw: perf/results/2026-08-19/n4{a,b,c}_pa256* + qwen38_n4_census/.
  Notebook: UPDATE 20.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N4b baseline pinned 2026-08-19 (fused add+RMSNorm)

- Change vs N4a: the 128-per-step residual seams run the new
  `add_rms_norm` fused kernel (one dispatch, bit-identical to the
  eager add + rms_norm chain — oracle-proven and sha-proven). Kill
  switch VLLM_QC_ADDNORM. SwiGLU hypothesis closed: already fused
  (qc_swiglu). metallib + .so rebuilt; DSV4 anchors ALL BIT-EXACT
  (anchor_regate_addnorm/).

| workload | N4a | N4b add+norm |
| --- | ---: | ---: |
| c1 1000/256 | 10.717 / 10.726 | 10.711 / 10.703 (sha fa58598b BIT-EXACT) |
| c4 1000/256 | 18.541 / 18.546 | 18.795 / 18.874 (+1.5%) |
| c8 1000/256 | 21.263 / 21.258 | 21.430 / 21.418 (+0.8%) |
| c1 2500/64 | 2.914 / 2.915 | 2.945 / 2.940 (sha aedef4ec BIT-EXACT, +1.1%) |

- vs Q4_K M4c bar: c1 95.9% (10.71/11.17), c4 113.4%, c8 123.0%.
- CANONICAL: c1 sha fa58598b, 2500x64 sha aedef4ec (both held).
- Remaining c1 gap is gdn non-GEMV + residual per-op overhead; the
  structural answer is the Muse one-command-buffer step loop
  (precedent: ~40 ms of an ~80 ms step was per-op host cost). Next
  campaign order: N5 DFlash 2, then re-rank.
- Raw: perf/results/2026-08-19/n4d_addnorm_gate/. Notebook: UPDATE 21.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N4c baseline pinned 2026-08-20 (GDN dispatch fusion; c1 98.4% of the Q4_K bar)

- Change vs N4b: the 48 GDN layers' decode chain drops from 12 to 3
  non-GEMV dispatches each (~430 fewer per step): `gdn_fused_prepare`
  (conv+silu with in-register history and state update, q/k l2norm,
  v, decay/beta — one dispatch reading the qkvz/ba projection rows
  in place; pure-decode routed) and `gdn_gated_rmsnorm_f32` (norm
  straight off the fp32 recurrence output, z read in place; decode
  and prefill), plus the forward_mps container restructure (normed
  rows go straight to out_proj). Kill switches VLLM_QC_GDN_FUSEPREP
  / VLLM_QC_GDN_FUSENORM. Oracle 20/20 BIT-EXACT incl. conv state
  pools. metallib + .so rebuilt; DSV4 anchors ALL BIT-EXACT
  (anchor_regate_gdnfuse/, 8th consecutive).

| workload | N4b | N4c gdn fusion |
| --- | ---: | ---: |
| c1 1000/256 | 10.711 / 10.703 | 11.019 / 10.968 (sha fa58598b BIT-EXACT, +2.7%) |
| c4 1000/256 | 18.795 / 18.874 | 19.215 / 19.281 (+2.0%) |
| c8 1000/256 | 21.430 / 21.418 | 21.683 / 21.681 (sha 467b35c3 2/2, +1.2%) |
| c1 2500/64 | 2.945 / 2.940 | 2.967 / 2.968 (sha aedef4ec BIT-EXACT, +0.9%) |

- vs Q4_K M4c bar: c1 98.4% (10.99/11.17), c4 115.7%, c8 124.5%.
- CANONICAL: c1 sha fa58598b, 2500x64 sha aedef4ec (both held).
- CALIBRATION: the win is ~2.4 ms/step vs the ~13 ms census-share
  estimate — marginal dispatch cost at batch 1 is ~5 us once the
  pipeline overlaps launches; sync-prof shares over-attribute launch
  cost. Remaining c1 gap (1.6%) needs a TRUE GPU-time ranking
  (xctrace Metal System Trace attach — no rebuild) before any
  further lever; the structural Muse step-loop and N5 DFlash 2
  re-rank after that.
- Raw: perf/results/2026-08-20/n4e_gdnfuse_gate/ +
  anchor_regate_gdnfuse/. Notebook: UPDATE 22.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, N5b baseline pinned 2026-08-20 (mv_ext batch GEMVs; spec c1 first passes no-spec)

- Change vs N4c: batches 3..8 (odd M included) of the fp8ch and nvfp4
  serving GEMVs route to the new mv_ext twins (`qgemv_fp8ch_mv4r`
  R1=4, `qgemv_nvfp4_mv4r` R1=2 — NR=4 rows per simdgroup, X via
  L1-served device vec4 loads, weights re-read ceil(M/R1) times; the
  llama.cpp kernel_mul_mv_ext precedent). Batch 2 keeps the mb pair
  twin, batch 1 unchanged, M > 8 dense. Kill switches
  VLLM_QC_FP8CH_MV4R / VLLM_QC_NVFP4_MV4R. Oracle: all batch rows
  BIT-IDENTICAL to looped batch-1. metallib + .so rebuilt; DSV4
  anchors ALL BIT-EXACT (anchor_regate_mv4r/, 10th consecutive).
- KEY RECALIBRATION (UPDATE 26): at M=8 the FMA work is the binding
  cost on M1 (~2.3-3.0e12 FMA/s effective issue rate; simdgroup_matrix
  is cooperative lane math, not a tensor core) — the "M=8 collapse"
  was mostly physics and batch-kernel headroom is ~1.3-1.7x, not the
  4x a weight-bound model predicted.

| workload | N4c | N5b mv_ext |
| --- | ---: | ---: |
| c1 1000/256 | 11.019 / 10.968 | 11.064 / 11.055 (sha fa58598b BIT-EXACT) |
| c4 1000/256 | 19.215 / 19.281 | 20.67 / 20.68 (+7.4%) |
| c8 1000/256 | 21.683 / 21.681 | 23.749 / 23.725 (sha 467b35c3 2/2, +9.5%) |
| c1 2500/64 | 2.967 / 2.968 | 2.977 / 2.974 (sha aedef4ec BIT-EXACT) |
| spec c1 (df2 k=7) | 9.59 / 9.60 | 11.11 / 11.082 (sha b46e676c HELD, +15.9%) |
| spec 2500/64 | 3.053 | 3.154 (sha ab7dde1b HELD) |

- vs Q4_K M4c bar: c1 99.0% (11.06/11.17), c4 124.3%, c8 136.3%.
- SPEC STATUS: c1 spec (11.11) now ABOVE no-spec (11.06) for the
  first time; spec 2500x64 +5.9% over no-spec. c4/c8 spec stay well
  below no-spec (verify M=32/64 rides dense) — spec is a c1 feature.
  Acceptance diagnostics 3.21-3.92 tok/step (interval-class).
- CANONICAL: c1 sha fa58598b, 2500x64 sha aedef4ec, spec c1 sha
  b46e676c (all held).
- Next: k-sweep 7 vs 3/5 (verify is compute-bound so k cuts verify
  FMAs linearly; k_sweep.sh staged), then Muse single-CB loop /
  TurboQuant KV re-rank.
- Raw: perf/results/2026-08-20/n5b_mv4r_gate/ + anchor_regate_mv4r/.
  Notebook: UPDATE 26.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, ADAPTIVE-SPEC baseline pinned 2026-08-21 (df2@k=3 promoted to canonical; c1 +23%)

- Change vs N5b: the CANONICAL profile qwen38-nvfp4-1 is now
  speculative: true with a batch-adaptive schedule
  (num_speculative_tokens_per_batch_size = [[1,1,3],[2,8,0]]): DFlash2
  drafts k=3 only when a single request is running; at batch >= 2 the
  scheduler sets zero draft slots and the V2 runner skips the drafter
  forward entirely (dynamic SD wired into the V2 runner + gdn_attn
  state-indices contiguity fix — UPDATE 28). Python + profiles.json
  only, no metallib/.so rebuild; DSV4 anchors bit-exact (11th
  consecutive, anchor_regate_adaptive/).

| workload | N5b no-spec | ADAPTIVE canonical (promoted) |
| --- | ---: | ---: |
| c1 1000/256 | 11.064 / 11.055 fa58598b | 13.651 / 13.62 (sha 8c58a4c6 2/2, +23.4%) |
| c4 1000/256 | 20.67 / 20.68 | 20.104 (97.2%) |
| c8 1000/256 | 23.749 / 23.725 | 23.275 / 23.26 (98.0%) |
| c1 2500/64 | 2.977 / 2.974 aedef4ec | 3.229 (sha f497f4a9, +8.6%) |

- vs Q4_K M4c bar: c1 122.2% (13.65/11.17), c4 120.9%, c8 133.6% —
  every workload now clears the bar, c1 for the first time.
- SHA SEMANTICS: c1 shas (8c58a4c6 spec / f497f4a9 spec-2500x64) are
  the promoted-canonical bit-exactness anchors. The NO-SPEC kernel
  anchors (c1 fa58598b, 2500x64 aedef4ec) remain the kernel-gate
  references — reproduce them by flipping the profile to
  speculative: false. Concurrency first-shas are composition-timeline
  dependent under the adaptive config (UPDATE 28) — treat c4/c8 as TPS
  gates, not sha gates.
- Residual: c4/c8 give back ~2-3% vs no-spec (spec-engine
  decode_query_len=4 classifies 1-token decodes as non-uniform ->
  extend path; transitional solo-spec steps). Recorded micro-lever,
  parked behind Muse/N6.
- Next: Muse single-CB step loop, then N6 TurboQuant KV, N7
  decommission gate.
- Raw: perf/results/2026-08-21/adaptive_spec_gate/ (ad_/ad2_/pr_) +
  anchor_regate_adaptive/. Notebook: UPDATE 28.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, GEMMA-NORM baseline pinned 2026-08-21 (fused target-layer norms; c8 passes the old no-spec)

- Change vs ADAPTIVE-SPEC: the 128 per-step GemmaRMSNorm target-layer
  norms route to new Metal kernels (gemma_rms_norm_dyn /
  gemma_rms_norm_add_dyn — exact ir-chain semantics, UPDATE 30).
  metallib + .so rebuilt; DSV4 anchors bit-exact (12th consecutive,
  anchor_regate_gemmanorm/).

| workload | ADAPTIVE (UPDATE 28) | GEMMA-NORM canonical |
| --- | ---: | ---: |
| c1 1000/256 | 13.651 / 13.62 8c58a4c6 | 13.900 / 13.912 (sha 8c58a4c6 HELD 2/2, +1.9%) |
| c4 1000/256 | 20.104 | 20.612 / 20.657 (+2.6%) |
| c8 1000/256 | 23.275 / 23.26 | 23.844 / 23.843 (+2.5%, ABOVE old no-spec 23.74) |
| c1 2500/64 | 3.229 f497f4a9 | 3.248 / 3.247 (sha ROLLED to d0e07ddd 2/2, +0.6%) |

- vs Q4_K M4c bar: c1 124.5%, c4 123.9%, c8 136.9% — the adaptive
  canonical now beats BOTH the bar and the old no-spec numbers at every
  measured workload.
- CANONICAL ANCHOR SHAS: c1 8c58a4c6 (held), 2500x64 d0e07ddd (rolled,
  deterministic 2/2, predicted ulp class — UPDATE 30). NO-SPEC kernel
  anchors via speculative:false are STALE for 2500x64 after this build
  (re-derive on next no-spec gate); c4/c8 remain TPS gates.
- Next: CPU time profile of EngineCore during c1-spec decode (the ~115
  ms/step host budget), then async-sched overlap vs Muse whole-step
  encode. Raw: perf/results/2026-08-21/gemmanorm_gate/. Notebook:
  UPDATE 30.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, SYNCFIX+ASYNC baseline pinned 2026-08-21 (queue-drain fix + async scheduling promoted)

- Changes vs GEMMA-NORM: (1) postprocess_state MPS scatter made
  sync-free (the boolean-mask index drained the whole GPU queue every
  spec step — UPDATE 31); (2) async scheduling enabled for dflash
  (whitelist + VLLM_METAL_ASYNC_SCHED=1 promoted into both qwen profile
  envs — UPDATE 32). Python-only; DSV4 path untouched; every sha
  BIT-EXACT through both gates.

| workload | GEMMA-NORM | SYNCFIX+ASYNC canonical |
| --- | ---: | ---: |
| c1 1000/256 | 13.900 / 13.912 | 14.252 / 14.269 (sha 8c58a4c6 2/2) |
| c4 1000/256 | 20.612 / 20.657 | 21.438 / 21.453 |
| c8 1000/256 | 23.844 / 23.843 | 24.520 / 24.520 |
| c1 2500/64 | 3.248 / 3.247 | 3.297 / 3.300 (sha d0e07ddd 2/2) |

- vs Q4_K M4c bar: c1 127.8%, c4 129.2%, c8 140.8%. Session cumulative
  vs the N5b no-spec start: c1 +29%, c4 +3.7%, c8 +3.3%, 2500x64 +10.7%.
- Note: async costs ~0.7% on 2500x64 vs the sync-fix build (3.313/3.328)
  — accepted for the c4/c8 wins; c1 flat.
- CANONICAL ANCHOR SHAS unchanged: c1 8c58a4c6, 2500x64 d0e07ddd.
- Next: async-mode cProfile census for the residual in-worker c1
  serializer; then N6 TurboQuant KV. Raw:
  perf/results/2026-08-21/{syncfix_gate, async_sched_gate}/. Notebook:
  UPDATEs 31-32.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, SYNC-HUNT baseline pinned 2026-08-21 (bound-mode metal_attn + GDN builder index_select; c1 14.80)

- Changes vs SYNCFIX+ASYNC: (1) metal_attn sync-free metadata — lazy
  exact host lens + bound-mode draft SDPA with a GPU visibility mask
  (UPDATE 33; spec c1 sha rolled 8c58a4c6 -> 467b35c3, predicted ulp
  class, deterministic); (2) gdn_attn spec-branch boolean-mask indexing
  replaced with CPU-nonzero + index_select (UPDATE 34, bit-exact);
  (3) gdn_attn repeat_interleave -> static segment-id (UPDATE 35,
  bit-exact, perf-NEUTRAL — confirmed 14.713/14.710 c1, 3.374/3.380
  2500x64, 21.56/21.52 c4, 24.594/24.599 c8; kept, wait relocated).
  All Python-only; DSV4 untouched.

| workload | SYNCFIX+ASYNC | SYNC-HUNT canonical |
| --- | ---: | ---: |
| c1 1000/256 | 14.252 / 14.269 | 14.788 / 14.801 (sha 467b35c3 2/2) |
| c4 1000/256 | 21.438 / 21.453 | 21.592 / 21.610 |
| c8 1000/256 | 24.520 / 24.520 | 24.628 / 24.649 |
| c1 2500/64 | 3.297 / 3.300 | 3.371 / 3.378 (sha d0e07ddd 2/2) |

- vs Q4_K M4c bar: c1 132.5%, c4 130.1%, c8 141.5%. Session cumulative
  vs the no-spec start: c1 +33.8%, c4 +4.5%, c8 +3.8%, 2500x64 +13.4%.
- CANONICAL ANCHOR SHAS: c1 467b35c3, 2500x64 d0e07ddd.
- KEY OPS FACT (recurring): any GPU-tensor op with a data-dependent
  output shape (boolean-mask indexing, nonzero, blocking D2H .to) drains
  the ENTIRE in-flight MPS queue. Find them with the cProfile census
  (VLLM_QC_PYPROF=1) — sync-bracketed profilers cannot see them.
- SYNC-HUNT PARKED (UPDATE 35): census round 4 found the relocated wait
  (repeat_interleave), the static-shape fix gated NEUTRAL — the last
  drain in the chain absorbs the queue tail, diminishing returns
  confirmed. Next: N6 TurboQuant KV, then N7 decommission gate. Raw:
  perf/results/2026-08-21/{boundmode_gate, gdnsync_gate, rilv_gate}/.
  Notebook: UPDATEs 33-35.
- N6 TURBOQUANT KV GATE 1 PASSED (UPDATE 36): profile qwen38-nvfp4-1-tq
  (TURBOQUANT backend + turboquant_k8v4 on target's 16 full-attn layers
  AND the DFlash2 drafter; GDN fp32 unchanged) is quality-gated: needle
  exact at 4.5k multi-chunk, deterministic 2/2 (c1 0c92d6e3, 2500x64
  4494d8e6 — profile-local pins). TPS tax vs canonical: c1 -11.3%
  (13.12), 2500x64 -13.4% (2.93), c4 -8.7%, c8 -6.0%. KV 4096 -> 1728
  B/token (~2.4x). CANONICAL DEFAULT UNCHANGED. Required fixing a
  latent GQA head-mapping bug in tq_attention_combined (head % Hkv ->
  head / (Hq/Hkv)) — the Metal TQ kernels had never been production-
  exercised; metallib rebuilt, DSV4 anchors re-gated BIT-EXACT (13th
  consecutive: 573db39598e7 / bb83cc3054a3 / f75e1d41ac3d). CAPACITY
  PAYOFF (same 16 GiB budget): TQ 292,882 KV tokens / 8.94x @32k vs
  canonical 158,038 / 4.82x — 1.85x. Raw:
  perf/results/2026-08-21/{tq_gate, anchor_regate_tqfix}/. Notebook:
  UPDATE 36.
- N7 DECOMMISSION DONE (UPDATE 37): profile qwen38-1 + source
  qwen38-27b (GGUF) RETIRED from the registry — every leg beats the
  Q4_K bar (132.5%/130.1%/141.5%/103.3%) and canonical needle recall
  is exact. The Q4_K bar rows above stay as the historical baseline.
  q4_K kernels remain in QuixiCore-Metal. NVFP4 is the sole Qwen3.8
  serving line (canonical + -df2 + -tq experimental).
- N8 262K LONG-CTX VALIDATED on -tq via the new Metal TQ dequant
  continuation route (UPDATEs 38-39): tq_decode_combined kernel +
  turboquant_dequant_kv_metal binding + dequant->SDPA/KV-tiled-softmax
  continuation (>128-token chunks; synthetic-decode retained below).
  Ladder at --ctx 262144, needle exact at EVERY depth: 16k=111 s (was
  406 s synthetic-decode), 65k=577 s, 131k=1492 s, 262k=4493 s at
  259,888 prompt tokens. Prefill cost model T(C) ~= 6.4e-3*C +
  3.9e-8*C^2 (quadratic constant 59x smaller). Short-ctx -tq re-gate:
  c1 13.08 sha 0c92d6e3 BIT-EXACT vs UPDATE 36 pin (no continuation on
  that leg); 2500x64 3.278/3.268 sha d0e07ddd = the CANONICAL sha, gap
  -13.4% -> -3.0%; c4 -7.2%; c8 -2.1%. DSV4 anchors BIT-EXACT 14th
  consecutive post metallib+.so rebuild. CANONICAL DEFAULT UNCHANGED at
  32k (c1 decode-side TQ tax -11.6% remains). 1M parked: out of native
  rope spec (max_position_embeddings 262144) + needs ~40 GiB KV budget.
  Raw: perf/results/2026-08-21/{anchor_regate_dequant, tq_dequant_gate,
  tq_longctx2}/. Notebook: UPDATEs 38-39.
- #16b SPLIT-K TQ DECODE ATTENTION RETAINED (UPDATE 40, 2026-08-21):
  tq_attention_splitk + tq_attention_reduce (PA256-style split-K, one
  simdgroup per token, zero token-loop barriers, block-table direct)
  replace the monolithic tq_attention_combined on the Metal hs256
  decode route (VLLM_QC_TQ_SPLITK=0 kill-switch; hs64/128 stay
  monolithic so DSV4 pins hold by construction). Kernel 12-33x;
  effective KV bandwidth ~7 -> ~220 GB/s; TQ decode now FASTER than
  canonical bf16 PA256 at every shape (2x at 32k ctx). -tq profile
  pins after the change: c1 1000x256 14.688/14.685 sha 228d0bf4 2/2
  (was 13.08/0c92d6e3; -11.6% tax now -0.8%); 2500x64 3.459/3.448
  sha d0e07ddd 2/2 (CANONICAL sha held; +2.1% ABOVE canonical);
  tail 2100x32 2.174/2.171 sha 1337d2f7 2/2 (new leg); c4 22.106
  (+2.2% above canonical); c8 25.148 (+2.0% above canonical); 4.5k
  needle exact; 16k needle exact at the 262k config (110 s wall).
  DSV4 anchors BIT-EXACT 15th consecutive (anchor_regate_tqsk/).
  -tq now beats canonical on 3 of 4 legs — canonical-default flip is
  a flagged decision (quality call, not TPS). Raw:
  perf/results/2026-08-21/{tq_splitk_bench, tq_splitk_gate,
  tq_splitk_longctx, anchor_regate_tqsk}/. Notebook: UPDATE 40.
- N9 ROPE QUEUE-DRAIN FIX RETAINED (UPDATE 41, 2026-08-24): the mrope
  torch replacement's repeat_interleave (data-dependent output shape)
  drained the whole in-flight MPS queue once per step — 65.8 ms/step,
  64% of the host profile. Replaced with the static-shape
  scatter+cumsum token->request map (metal_compat.py, python-only;
  VLLM_QC_ROPE_STATIC=0 null switch validated bit-exact). NEW
  CANONICAL BASELINE qwen38-nvfp4-1: c1 16.139 sha 467b35c3
  BIT-EXACT (144.5% of the Q4_K bar; was 14.80) / c4 23.16 (139.5%)
  / c8 25.62 (147.1%) / 2500x64 3.454 sha d0e07ddd BIT-EXACT. -tq:
  c1 16.08 sha 228d0bf4 BIT-EXACT (-0.3% vs canonical); c4/c8/2500x64
  not yet re-measured post-rope (UPDATE 40 values 22.11/25.15/3.45).
  DSV4 8tok anchor BIT-EXACT (python change, no binary rebuild; full
  anchor chain not required). Raw: perf/results/2026-08-24/
  {rope_gate, anchor_leg_rope}/. Notebook: UPDATE 41.
- OPS (UPDATE 42, 2026-08-24): tmp cleaner destroyed m2_source.txt +
  dsv4_gate.py + build recipes; all recovered byte-exact (m2_source
  reconstruction PROVEN by a null boot reproducing c1 467b35c3 and
  2500x64 d0e07ddd). Stable home: perf/results/harness_assets/ — all
  future chains must use that path. Notebook: UPDATE 42.
- N10 STEADY-CACHE METADATA RETAINED (UPDATE 43, 2026-08-24): the
  mamba-hybrid attn-metadata rebuild (GDN builder 8.2 ms/step, 10
  group builds x ~30 ops) is skipped on steady uniform all-spec decode
  steps: MambaHybridAttnMetadata.steady_signature() folds into the
  steady shape sig, GDN steady_decode_update refreshes just
  spec_state_indices + num_accepted (2 copies), metal/tq full-attn
  builders rebuild from the cached CommonAttentionMetadata with
  cm.seq_lens_cpu_upper_bound refreshed in place (the runner
  allocates that bound fresh each step — a frozen cached view
  silently truncated decode attention windows in gate round 2;
  root-caused via deterministic sha roll ac33cee5, fixed, re-gated).
  Python-only, VLLM_QC_STEADY_META=0 null switch (proven identical to
  UPDATE 41: 16.153 sha 467b35c3). NEW CANONICAL BASELINE
  qwen38-nvfp4-1: c1 16.334/16.346 sha 467b35c3 BIT-EXACT (146.3% of
  the Q4_K bar; was 16.139) / c4 23.02 (-0.6%, noise — hits rare
  under composition churn) / c8 25.57 (-0.2%, noise) / 2500x64
  3.444/3.452 sha d0e07ddd BIT-EXACT. -tq: c1 16.216/16.210 sha
  228d0bf4 BIT-EXACT (+0.8% vs 16.08). Canon + TQ 262k needles both
  exact. Round-1 lesson: upstream steady eligibility deref'd
  is_prefilling=None on this path (engine-dead crash; ramp helpers
  must verify generated text, not HTTP success). Raw:
  perf/results/2026-08-24/steady_gate/. Notebook: UPDATE 43.
- N11a FUSED DFLASH2 GROUPED CONV RETAINED (UPDATE 44, 2026-08-24):
  qc_dflash_conv kernel (serving/dflash_conv/) — one dispatch replaces
  the ~10-op eager chain per `_grouped_conv` call (~200 encodes/step
  across 4 calls x 5 drafter layers; the pad/arange/mask ops go with
  it). Kernel parity 37/37 BIT-EXACT vs the eager chain (per-op
  fp32-round mirror of MPS elementwise semantics), so drafts and
  acceptance are bit-identical — the serving delta is pure host-encode
  recovery. VLLM_QC_DFLASH_CONV=0 restores eager (null boot proven =
  UPDATE 43: 16.337 sha 467b35c3). Metallib + .so rebuilt: DSV4
  anchors ALL BIT-EXACT 2/2 each, 17th consecutive
  (anchor_regate_dflashconv/). NEW CANONICAL BASELINE qwen38-nvfp4-1:
  c1 16.428/16.438 sha 467b35c3 BIT-EXACT (+0.6%, 147.1% of the Q4_K
  bar) / c4 23.111 / c8 25.615 / 2500x64 3.454 sha d0e07ddd BIT-EXACT;
  needle exact. Calibration reconfirmed: the eager conv's 5-7 ms/step
  profiler share was mostly absorbed queue-tail wait — dispatch
  elimination recovered ~1 ms/step (~5 us marginal encode x ~200).
  Raw: perf/results/2026-08-24/{anchor_regate_dflashconv,
  dflash_conv_gate}/. Notebook: UPDATE 44.
- N11b PURE-PREFILL CAUSAL SDPA BOUND RETAINED (UPDATE 45,
  2026-08-24): all-prefill batches read the CPU bound (row-exact
  there — the async mirror only diverges on spec-decode rows) in the
  causal SDPA loop instead of the queue-draining seq_lens D2H (census:
  3.43 s over 12 calls, concentrated in multi-chunk prefills — a
  long-context TTFT tax, not leg TPS). Python-only,
  VLLM_QC_SDPA_PREFILL_BOUND=0 null proven = UPDATE 44. Gate: DSV4
  8tok prudence 573db39598e7 BIT-EXACT 2/2; canonical needle exact;
  c1 16.414/16.438 sha 467b35c3 BIT-EXACT; 2500x64 3.465/3.465 sha
  d0e07ddd BIT-EXACT (+0.3% — the predicted +2-7% missed: the 2500
  prompt is single-chunk; the drain lives in multi-chunk prefills);
  c4 23.106 / c8 25.60. CANONICAL BASELINE unchanged from UPDATE 44
  numbers (c1 16.43-class); the win books as multi-chunk prefill
  TTFT. Raw: perf/results/2026-08-24/sdpa_bound_gate/. Notebook:
  UPDATE 45.
- -TQ RE-MEASURE ON THE U43-U45 STACK (UPDATE 46, 2026-08-24,
  measurement only): c1 16.182/16.240 sha 228d0bf4 BIT-EXACT (-1.2%
  vs canonical 16.43); 2500x64 3.512/3.506 sha d0e07ddd (CANONICAL
  sha, +1.4%); c4 23.715/23.615 (+2.4%); c8 26.053/26.090 (+1.8%);
  needle exact. -tq now beats canonical on every leg above c1 with
  2.4x KV capacity; the default-flip decision (quality call, k8v4
  cos ~0.9955 floor) has fully current data. Raw:
  perf/results/2026-08-24/tq_remeasure/. Notebook: UPDATE 46.
- N11c DRAFT-BLOCK PAGED ATTENTION REJECTED (UPDATE 47, 2026-08-24):
  routing the drafter's non-causal SWA blocks through the expanded
  paged kernel lost TPS everywhere (c1 -7.7%, 2500x64 -4.1% with a
  ULP-trajectory fork to the old 95adbd97, c4/c8 noise-down). Two
  causes, both measured: kernel-vs-SDPA reduction order shifts the
  DFlash 16-way candidate/path selection (acceptance 1.59 -> 1.54
  accepted/draft), and the kernel is slower than one-request SDPA at
  draft shapes. Default flipped to opt-in (VLLM_QC_DRAFT_BLOCK_PA=1),
  branch quarantined as a documented diagnostic; revert cycle
  reproduced canonical exactly (c1 16.426 sha 467b35c3, box serving).
  KEY LESSON: drafter math is NOT reduction-order-tolerant — the
  candidate selector amplifies ULP noise into acceptance loss; only
  bit-exact or measured-neutral drafter changes are safe. CANONICAL
  BASELINE unchanged (UPDATE 44 numbers). Raw:
  perf/results/2026-08-24/draft_block_pa_gate/. Notebook: UPDATE 47.
- N12 FUSED QK-NORM-ROPE-GATE REJECTED/PARKED (UPDATE 48, 2026-08-24):
  qc_qk_norm_rope_gate (one dispatch replacing ~25/attn-layer) is
  bit-exact at decode shapes but torch-MPS eager numerics are
  SIZE-DEPENDENT — T=1000 prefill diverges by ~5 ppm single-ulp
  elements, forking the canonical trajectory (c1 d7c66851, -1.8%
  content-confounded; c4 +1.2% / c8 +0.7%). Not worth a full re-pin;
  default OFF, opt-in VLLM_QC_QKROPE=1. DSV4 anchors bit-exact 18th
  consecutive (rebuild cleared); null boot + revert cycle reproduce
  canonical exactly (16.412/16.437 sha 467b35c3, counters replay).
  KEY LESSON: validate kernel parity at SERVING shapes incl. prefill
  T — per-shape bit-exactness does not transfer across sizes on MPS.
  CANONICAL BASELINE unchanged (UPDATE 44 numbers). Raw:
  perf/results/2026-08-24/{qkrope_gate, anchor_regate_qkrope}/.
  Notebook: UPDATE 48.
- N13 ONE-DISPATCH KV INSERT RETAINED (UPDATE 49, 2026-08-24):
  kv_cache_scatter kernel bound and routed (block_mult=2 page-local
  variant; slot<0 skipped) — 5 torch ops -> 1 dispatch per attention
  layer (~80 encodes/step). Pure copies, bit-exact at any shape (the
  U48 size-dependence trap cannot apply). Gate: DSV4 anchors
  bit-exact 19th consecutive (kernel live in DSV4 drafter writes);
  c1 16.443/16.440 sha 467b35c3 BIT-EXACT / 2500x64 3.456/3.459 sha
  d0e07ddd BIT-EXACT / needle exact / c4 23.13 c8 25.62; TPS flat
  within noise (retained on proven equivalence + dispatch hygiene).
  VLLM_QC_KV_SCATTER=0 restores the torch path. CANONICAL BASELINE:
  c1 16.44-class sha 467b35c3 (147.2% of the Q4_K bar). Raw:
  perf/results/2026-08-24/{kvscatter_gate, anchor_regate_kvscatter}/.
  Notebook: UPDATE 49.

## Qwen3.8-27B NVFP4 — Metal M1 Ultra, MUSE SINGLE-CB baseline pinned 2026-08-24 (UPDATE 53; N14 default flip, c1 at the modeled ceiling)

- Profile `qwen38-nvfp4-1` with VLLM_QC_MUSE=1 in the profile env (the
  whole 64-layer target forward + final norm + drafter aux taps encoded
  into ONE command buffer on eligible uniform spec-decode batches,
  m <= 8; everything else eager bit-exactly). Kill switch VLLM_QC_MUSE=0
  restores the UPDATE 49 canonical.
- c1 1000x256: **17.068 / 17.142 tok/s, sha 467b35c3 BIT-EXACT with the
  pre-muse canonical** (+4.1% over 16.44; 154.4% of the Q4_K bar).
  c1 = the N3-era modeled bandwidth ceiling (~17 tok/s).
- 2500x64: 3.448 / 3.459, sha aa448847 deterministic 2/2 (the one
  re-pinned leg; TPS flat vs canonical 3.46; 30k needle EXACT).
- c4 23.12 / c8 25.59 (m > 8 -> eager fallback; shas match the
  canonical tie-carriers).
- DSV4 anchors bit-exact on the final build (24th consecutive):
  8tok 573db39598e7, off1-2000 bb83cc3054a3, 2500x64 f75e1d41ac3d.
- Null boot (VLLM_QC_MUSE unset pre-flip build): c1 16.419 sha 467b35c3.
- Raw: perf/results/2026-08-24/{muse_shadow_gate, anchor_regate_muse_p2}/.
- AUDIT REBUILD RE-GATE (UPDATE 54, 2026-08-25): pre-PR audit cleanup
  (dead muse_silu_mul_rows kernel deleted, NANPROBE harness removed,
  steady-cache conv_slots invalidation, host-side validation added);
  metallib + .so rebuilt. ALL PINS BIT-EXACT: DSV4 anchors 2/2 x3 (25th
  consecutive), c1 17.117/17.142 sha 467b35c3, 2500x64 3.453/3.434 sha
  aa448847, needle exact. Raw: perf/results/2026-08-25/
  {anchor_regate_audit, audit_regate}/.
- MERGE RE-GATE (UPDATE 55, 2026-08-25): origin/main's 55 parallel
  commits merged (b59af307c), metallib + .so rebuilt from the merged
  sources. DSV4 anchors: 8tok 573db39598e7 and off1-2000 bb83cc3054a3
  BIT-EXACT (26th consecutive); **2500x64 RE-PINNED to 73f41acf8ca0**
  (42/110/22, deterministic 2/2, wall 4.60-4.65 s, response coherent) —
  attributed to main's grammar-aware DSpark drafting (7d0b41f4a,
  09f714f37) shifting acceptance trajectories, not verified numerics.
  Old pin f75e1d41ac3d retired. Canonical Qwen3.8 pins: see UPDATE 55 in
  optimization_status (two boot defects fixed in the merged build first:
  vision-wrapper torchvision requirement -> language_model_only in the
  Metal variants; duplicate `_forward_core_mps` def shadowing the Metal
  GDN dispatcher -> renamed + unified dispatch). Raw:
  perf/results/2026-08-25/{anchor_regate_merge, merge_regate}/.
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

### MI300X Single-GPU Exact Baseline - 2026-08-26 (type-aware imatrix route)

- Model/profile: Unsloth Qwen3.8-27B UD-Q2_K_XL GGUF,
  `qwen38-q2kxl-1`, TP1, full 262,144-token context, fixed 96 GiB KV pool,
  `max_num_seqs=64`, compiled `FULL_DECODE_ONLY` graphs, DFlash2 Q4_K_M k=3,
  V2 runner with the drafter's rank-256 path selector.
- Workload: hot OpenAI completion server, 1,000 input tokens and exactly 2,000
  output tokens per request; temperature 1.0, top-p 0.95, top-k 20, seed 42,
  8-token shape warmup. One timed pass at c1; two retained timed passes at c8.

| Concurrency | Aggregate tok/s | Median request s | Spec accepted / drafted | Exact |
| ---: | ---: | ---: | ---: | --- |
| 1 | 77.23 | 25.90 | 1,285 / 2,142 | yes |
| 8 | 200.20 | 69.79 / 72.08 | 21,106 / 32,709 (two runs) | yes |

The c8 baseline combines two exact retained-route passes (197.15 and 203.33
tok/s); its median-request column reports the two run medians. The type-aware
ROCm route keeps compact importance-matrix kernels through
their measured M=16 or M=32 crossover instead of dequantizing every wider
forward. The c1 row remains the stable V2 measurement because the route is
unchanged at M=4; a 2026-08-26 c1 sample had anomalously low draft acceptance
and is recorded, but not promoted, in the optimization notebook.

The registered vision profile passed text and deterministic red-image smoke
at the same 64-sequence sizing. All exact c8 completions were healthy. Raw:
`perf/results/2026-08-26/gguf-mi300x25-imatrix-route/`; smoke:
`perf/results/2026-08-26/gguf-mi300x26-imatrix-route-smoke/`. The prior V2
baseline remains at `perf/results/2026-08-25/gguf-mi300x21-v2-exact/`.

The superseded legacy-runner measurements were 75.47 tok/s at c1 and 164.87
tok/s at c8; V2 improved those by 2.3% and 17.8%, respectively. Their raw
artifacts remain at `perf/results/2026-08-25/gguf-mi300x19-seq64-exact/`.

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

## GLM-5.2 A100 tier records - exact-token baseline 2026-09-06 (stack: copy-free sparse decode, tier active)
- Through `slimserve <id> --serve` with the 48 GiB/rank host tier, exact-token
  harness (1000 in / 300 out, temp 1.0 / top-p 0.95 / top-k 20, seed 42):
  | record                        | c1   | c8    | c16   |
  | glm52-q2k-4 (TP4, fp8 KV)     | 31.7 | 93.7  | 115.6 |
  | glm52-q2k-8 (TP8, bf16 KV)    | 72.5 | 166.5 | 223.1 |
  Raw: perf/results/2026-09-06/glm52-q2k-{4,8}-baseline/. Canaries pass on
  both. Supersedes the 2026-09-03 TP8 entry below (14.9 / 81.5 / 135.2).

## GLM-5.2 glm52-q2k-8 (A100 x8, bf16 KV @ 202752) - first exact-token baseline 2026-09-03 (superseded)
- Through `slimserve glm52-q2k-8 --serve`, exact-token harness (1000 in /
  300 out, temp 1.0 / top-p 0.95 / top-k 20, seed 42), aggregate output tok/s:
  | c1   | c8   | c16   |
  | 14.9 | 81.5 | 135.2 |
  Raw: perf/results/2026-09-03/glm52-q2k-8-baseline/. Before the
  partitioned bf16 sparse decode launch the same boot read 9.6 / 64.7 /
  110.4 (perf/results/2026-09-03/glm52-q2k-8-prepartition/). Text
  (+reasoning), image and tool canaries pass. The Q2_K GGUF MoE path is
  this record's remaining budget and was not touched today.

## GLM-5.3-Flash NVFP4 glm53f-nvfp4-8 - maximized record, 2026-09-06
- TP8, EP off, model-default 1,048,576 context on three KV tiers: VRAM
  pool 50.08 GiB/rank = 3,171,368 tokens (gpu_memory_utilization 0.95),
  pinned host tier 72 GiB/rank (576 GiB), disk tier 256 GiB/rank (2 TiB,
  per-rank O_TMPFILE in SLIMSERVE_KV_TIER_DIR). max_num_seqs 64 with a
  64-deep FULL_DECODE_ONLY capture. Boot through `slimserve glm53f-nvfp4-8
  --serve`; text (+reasoning), image, tool canaries pass.
- Exact-token (1000 in / 300 out, temp 1.0 / top-p 0.95 / top-k 20, seed
  42), aggregate output tok/s:
  | c1   | c8    | c16   | c32   | c64   |
  | 83.8 | 402.6 | 562.1 | 750.0 | 931.9 |
  Raw: perf/results/2026-09-06/glm53f-nvfp4-8-max/ (pre-rename path).
- Tier acceptance on this pool with VLLM_KV_TIER_VERIFY=1: 6/6 probes hit
  and resumed at block 68 (39,168 tokens), 106 restore ops each, 0/106
  mismatched on every restore, one promotion from disk, 6/6 recalled
  after a 3,980,143-token churn. WildChat deep-context leg on this record:
  see the 2026-09-06 notebook entry.

## GLM-5.3-Flash NVFP4 (glm53f-nvfp4-4 / glm53f-nvfp4-8, A100)

### A100 Exact Baseline - 2026-09-03 (compile on, partitioned + vectorized sparse decode, strided indexer)
- Stack: torch.compile active on the text model (kda_attention op, indexer
  as a splitting op), bf16 sparse MLA decode partitioned (P128) with the
  VECBF16 row path, pooled-indexer grid proportional to the actual
  context. Both records booted through `slimserve <id> --serve`, model-
  default 1,048,576 context, no speculative decoding, no host tier.
- Exact-token harness (1000 in / 300 out, temp 1.0 / top-p 0.95 / top-k 20,
  seed 42, warmed per concurrency), aggregate output tok/s:
  | record                 | c1   | c8    | c16   |
  | glm53f-nvfp4-4 (TP4)    | 73.8 | 332.1 | 464.6 |
  | glm53f-nvfp4-8 (TP8)    | 84.0 | 412.3 | 575.8 |
  Raw: perf/results/2026-09-03/glm53-nvfp4-{4,8}-baseline/.
- TP8 / TP4: +14% / +24% / +24% - below the 50% scaling gate. The gap is
  per-rank replicated work (91 custom allreduces per token at 8 ranks, the
  mHC stream kernels, the pooled indexer); the 8-GPU matrix's TP4xDP2 arm
  (two independent TP4 engines) is the upper bound that fusing it away
  would recover. Documented in optimization_status 2026-09-03.
- 8-GPU matrix (arm launches at 32K context, pre-strided-indexer): TP8
  83.0 / 343.7 / 447.7; TP8+EP 75.3 / 331.1 / 430.8; TP4xDP2 67.2 / 382.8 /
  527.6; TP4xDP2+EP 51.3 / 317.0 / 480.1. EP off on both records.
- Correctness gates at this baseline: text (+reasoning), image and tool
  canaries through both profiles; pooled-indexer parity vs transformers;
  bf16 sparse decode parity (tests/kernels/test_quixicore_sparse_mla_bf16.py).
  TP8 WildChat deep-context leg PASS (perf/results/2026-09-03/glm53-leg/
  glm53f-nvfp4-8/): 0 errors, 34/34 recall, max ctx 202,509, median
  190,704, 97,378 completion tokens in 1.25 h. The TP4 leg passed on
  2026-09-02 (19/19 recall to 114K) on the pre-kernel-fix stack.
- The 2026-09-01 TP4 numbers below predate all three fixes and are kept
  as history; their c8 331.4 is not reproducible and is superseded.
- Host + NVMe KV tiers ENABLED on both records (2026-09-04): 64 GiB
  pinned host RAM per rank plus a 128 GiB per-rank disk tier
  (SLIMSERVE_KV_TIER_DIR, operator environment). Layout: packed
  cross-layer slab over all 56 layers (4 KDA state groups + MLA +
  indexer), per-group block ratios in the connector (indexer 2176-token
  blocks over the 1088 hash block at TP4). Acceptance (TP4, 40K-token
  target, 1.9M-token churn of a 1.24M-token pool, VLLM_KV_TIER_VERIFY=1):
  host tier 48 GiB/rank - 6/6 probes hit and restored at block 36, 58
  restore ops each, 0/58 mismatched; disk tier with an 8 GiB host tier
  so restores promote from disk - 0/58 mismatched, 0 disk round-trip
  mismatches, 6/6 recalled. Raw: perf/results/2026-09-03/
  kvtier-acceptance/glm53-tp4-{host48,disk9}/.

### History: GLM-5.3-Flash NVFP4 (glm53f-nvfp4-4, A100 x4)

### A100 TP4 Exact Baseline - 2026-09-01 (bring-up, sparse DSA, no spec)
- Config: the registered glm53f-nvfp4-4/a100 record (TP4, EP off,
  QUIXICORE_MLA_SPARSE + sparse_mla_force_mqa, block 64, bf16 KV, Marlin
  NVFP4 MoE, FULL_DECODE_ONLY graphs; no speculative decoding yet).
- Exact-token harness (benchmarks/benchmark_dsv4_exact.py, 1000 in /
  300 out, temperature 1.0 / top-p 0.95 / top-k 20, seed 42):
  | concurrency | aggregate output tok/s |
  | c1          | 70.1  |
  | c8          | 331.4 |
  Raw: perf/results/2026-09-01/glm53-opt-sparse/.
- Same-harness A/B references: dense NoPE MLA (bring-up only) c1 74.1 /
  c8 343.6; expert-parallel arm c1 70.0 / c8 330.4 (dense) - EP off is the
  record's default. Eager -O0 (retired) was 5.4 / 41.6.
- Correctness gates at this baseline: text/image/tool canaries via the
  profile; 6,396-token planted-fact recall through sparse selection;
  pooled-indexer parity vs the transformers reference.
- Expected next moves: MTP speculative decoding (checkpoint ships the
  head), host tier once the packed planner handles KDA+MLA mixed block
  sizes, kernel-level tuning of the pooled-logits path.
