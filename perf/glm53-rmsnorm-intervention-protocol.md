# GLM53 normalization-only AOT intervention

Frozen before GPU hook qualification or serving, 2026-09-09.

## Hypothesis and isolation

The four RMSNorm reduction choices are a proven source of BF16 differences.
Test whether changing ONLY those choices explains the existing native-only /
instrumented-control score mismatch. Neither config has failed the FP64 oracle.
This is NOT a new normalization policy or performance/default promotion.

- New startup-only `SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC=control|legacy`.
  Same fixed native-order profile/recipe/TP4/noEP/no-spec/Marlin, BF16fn1/TC0.
  No tensor journals, launch blocking, verbose JIT, profiler or scheduling change.
- Resolve the actual `StaticAutotunerFuture` first; require the target's one
  selected launcher to have its recorded native-order binary hash. Recompile
  byte-identical source with exact Inductor metadata and recorded config,
  then require the replacement's recorded hash. Control substitutes the SAME
  native binary; legacy substitutes the audited legacy binary. Clear the cached
  callable so it cannot silently keep the previous launcher.
- Record every static-future launcher's before/after hashes and configurations;
  unrelated launchers must remain unchanged. Exactly one target binding per
  rank must seal before CUDA graph capture; missing/repeated/late targets fail.
- Prepare two byte-identical private copies of native AOT namespacec8b11c6e,
  including all model/artifact/cache files. Never modify original caches.
  Relocate serialized source filenames to private copies after verifying bytes;
  detach serialized original-cache save hooks. Compare every original file and
  file-set at audit. Changing cache paths is an isolation mechanism, not a
  fresh-compilation test. `VLLM_FORCE_AOT_LOAD=1` forbids silent fallback.
- Source-only changes in runner initialization/CLI install the diagnostic;
  model source, native libraries and graph-cache factors are unchanged. The
  no-op full-model control must prove that this is truly nonperturbing.
- Campaign receipts freeze22 serving/client/probe sources. Both modes are
  baseline-ineligible regardless of the number of quality passes requested.

## Before serving

CPU gates and lint, then ONE16GiB/no-swap GPU hook job. Four rank-matched sources,
control+legacy, rows16/640, seeds530901/531001, BF16 unit weights. Exercise the
real static-future result hook and startup seal; compare exact eager/repeat/
changed-graph outputs against directly compiled expected configs.16 cases,
no timing or full-model claim. All original96 reduction-isolation cases are
already complete and are not repeated here.

```bash
set -o pipefail
systemd-run --user --scope --unit=glm53-rmsnorm-intervention-gpu \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CUDA_HOME=/usr/local/cuda-13.0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_rmsnorm_intervention \
  --audit perf/results/2026-09-09/runtime-control/native-order-autotune-cache-comparison.json \
  --output perf/results/2026-09-09/rmsnorm-intervention-gpu \
  2>&1 | tee perf/results/2026-09-09/runtime-control/rmsnorm-intervention-gpu.log
```

After that job releases GPUs, prepare the private caches in an8GiB CPU scope:

```bash
systemd-run --user --scope --unit=glm53-rmsnorm-intervention-prepare \
  -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m benchmarks.kernels.prepare_glm53_rmsnorm_intervention \
  --audit perf/results/2026-09-09/runtime-control/native-order-autotune-cache-comparison.json \
  --output perf/results/2026-09-09/rmsnorm-intervention-caches
```

## Fixed serving sequence

Exactly TWO starts prescribed now, with a correctness stop between them:

1. `control`, output `rmsnorm-noop-control`: three full quality passes must match
   all nine existing native-only passes at EVERY text and needle-token score.
2. Only if control passes, `legacy`, output `rmsnorm-legacy-only`: three passes,
   compare all scores within this arm and against both native and older legacy
   references. A mismatch against legacy is retained evidence, not a widened
   gate. It means the four choices alone did not explain that full difference.

Each uses one boot,25 warmup/75 timed exact1000-in/300-out cold requests at
c1/c8/c16, three timing repetitions, text4/imageRed canaries, then three quality
passes (4096 text scores +168 needle-token scores / six contrasts each).
Require zero cached tokens on all workload requests. Preserve every result,
allocation/teardown warning, failure and slow sample. No replacement starts.
No source/native edits, builds, commits or competing GPU jobs during serving
or between these arms; audit control before deciding whether to launch legacy.

Use the completed native-order protocol command with the following substitutions:

- Scope `glm53-rmsnorm-noop-control` or `glm53-rmsnorm-legacy-only`,150GiB/no swap.
- Add `VLLM_FORCE_AOT_LOAD=1` and `SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC=control`
  or `legacy` as appropriate.
- `SLIMSERVE_GLM53_RMSNORM_MANIFEST` points to the prepared arm's absolute
  `manifest.json`; `VLLM_CACHE_ROOT` points to that arm's absolute `cache/`.
- Output `perf/results/2026-09-09/rmsnorm-noop-control` or `rmsnorm-legacy-only`;
  launch log under `runtime-control/` named for the output.
- Everything else remains the native-order protocol: unset all legacy ordering
  flags/journals/launch blocking/inherited NCCL_P2P_DISABLE, native order1,
  BF16fn1/TC0, CUDA13.0, source promptSHA0b665148, profileglm53-nvfp4-4,
  `--boots 1 --repeats 3 --concurrency 1 8 16 --input-tokens 1000
  --output-tokens 300 --cold-prefix --quality --quality-repeats 3`.

Audit actual22 source receipts, native/package/plan identities, four source-bound
replacement receipts, all unaffected launchers, direct AOT loads, original cache
snapshot, every request and every quality score. Require clean owned-process
teardown/GPU release before any further development. No TPS gain, default
promotion, TC exoneration or fresh-compilation invariance claim from these arms.
