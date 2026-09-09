# GLM53 cached RMSNorm isolation

Frozen before the first GPU launch, 2026-09-09. Diagnostic only; no serving,
native library, quant, profile, tolerance or original-cache mutation.

## Question and fixed experiment

The source-bound cache audit identifies four reduction-width changes between
the legacy instrumented and native-only AOT namespaces. Do these exact generated
kernels produce different BF16 outputs with their recorded launch configs?

- ONE process, four rank-matched SM120 GPUs used sequentially, 16 GiB host
  memory cap, swap zero. No competing GPU job/build/server.
- Read-only audit input:
  `results/2026-09-09/runtime-control/native-order-autotune-cache-comparison.json`,
  SHA256 `249d26192e4a2ecc509f437e3560f982aa0b8eb82e2c53954c7212077bc636fb`.
- Four byte-identical source copies imported only from the new result directory;
  private Inductor/Triton caches. Use Inductor's `_precompile_config` and
  `make_launcher`, NOT the autotuning/run hook. Preserve the exact signatures,
  divisibility attributes, constants and compile options. Require each compiled
  kernel cache hash to equal its recorded serving-kernel hash before execution.
- Both recorded configurations: XBLOCK1; R0_BLOCK1024/8 warps or4096/16 warps;
  one stage. Three sources normalize in place; rank0 writes three outputs.
- Rows1/16/640/7616; CPU-generated normal inputs with seeds530901/530902.
  Three actual BF16 norm vectors: layer0 input norm at activation magnitude.125,
  layer22 post-attention norm at1, layer44 post-attention norm at8.
  These are synthetic activations, NOT archived pre-normalization model tensors.
- All96 cases run both configurations. Each checks two identical eager inputs,
  a CUDA-graph original-input replay, and a changed-input replay (seed+100).
  Reset in-place inputs outside the graph. Check bit-exact repetitions and graph
  equivalence, changed-input sensitivity, all three outputs where applicable,
  unchanged weights/read-only inputs and guard rows. These are functional safety
  checks, not a sanitizer qualification or a timing benchmark.
- Independently evaluate FP64 RMS/rsqrt/weight multiply, rounding once to BF16
  as the generated serving kernels do. Require finite output and at most one
  BF16 ULP from that oracle. This is the existing isolated norm accuracy
  contract; it does not replace any full-model exact-score gate.
- Record bit/numeric mismatch counts, affected rows, BF16 ULP and absolute/RMS
  differences for all192 original/changed cross-config pairs. Cross-config
  mismatch is an observation, not an abort condition. Preserve every failed
  attempt and partial record; never rename/repeat a failed run as a replacement.
- Recheck all original source/config hashes at teardown. No E2E TPS, full-model
  causality, production correctness or default/TC promotion claim follows.

## Command

From repo root after CPU tests and commit; verify empty GPUs first:

```bash
set -o pipefail
systemd-run --user --scope --unit=glm53-cached-rmsnorm \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CUDA_HOME=/usr/local/cuda-13.0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python benchmarks/kernels/check_glm53_cached_rmsnorm.py \
  --audit perf/results/2026-09-09/runtime-control/native-order-autotune-cache-comparison.json \
  --model /raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4 \
  --output perf/results/2026-09-09/cached-rmsnorm-isolation \
  2>&1 | tee perf/results/2026-09-09/runtime-control/cached-rmsnorm-isolation-launch.log
```

Next, if arithmetic differences are observed, design a narrow full-model
intervention preserving unrelated compiler choices and both original caches.
Do not enable global force-first-config or treat isolated differences as a
complete explanation of the older score mismatch.

## Preserved failed attempt and bounded continuation

The first process on22d609633 completed all24 rank1 cases, then failed during
the first rank3 graph capture. Its first two eager calls succeeded, but the
implicit `torch.cuda.graph.default_capture_stream` still belonged to GPU1.
The installed Torch source makes this stream process-global. Preserve the
invalid-argument failure, empty-graph warning and partial summary:
SHA256 `58f683152ed0f85e9257a7394275174cd708d8bd32af267b87c4d784ad0f78a7`.
All eight original source/config receipts checked so far remain unchanged;
no rank3 case completed, rank0/rank2 did not begin. GPUs released.

Fix ONLY the diagnostic stream binding to an explicit per-device stream.
The case list, source/compiled-hash gates, math and tolerances do not change.
After CPU tests/commit, run exactly the remaining ranks3/0/2 once, using the
same command with these three substitutions/addition:

- Scope: `glm53-cached-rmsnorm-remainder`.
- Output: `perf/results/2026-09-09/cached-rmsnorm-remainder`.
- Launch log: `runtime-control/cached-rmsnorm-remainder-launch.log`.
- Add `--ranks 3 0 2` (audit order is3/0/2 after filtering).

Combine only the original24 completed cases and the remaining72, verifying
exact coverage of96 unique rank/row/seed/site keys and all192 comparisons.
Do not rerun rank1 or overwrite/relabel the failed attempt as successful.

Continuation completes onada8f2915. Combined audit verifies all96 unique cases
and192 comparisons;100 pairs differ, maximum one BF16 ULP, all oracle/repeat/
graph/guard checks pass. All eight compiled-kernel bindings match recorded hashes.
Original caches unchanged; GPUs released. Raw combined analysis:
`runtime-control/cached-rmsnorm-analysis.json`,
SHA256 `b472082671205e9dc2c40c9cc744d9da94721b7cfbf8ab0a45b9c295132d660c`.
This isolated protocol is COMPLETE; it does not establish full-model causality.
