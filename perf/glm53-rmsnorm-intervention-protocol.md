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

## Preserved pre-health failure and source-binding repair

The first control on ea2fff9ce directly loads all four AOT models, but fails
before health: ranks1/2/3 each own TWO distinct autotuner objects for their one
target source. Rank0 owns one. The hook replaced all seven objects with identical
native binaries, but the seal incorrectly required one object, not source
coverage. All72 recorded launchers have identical before/after hashes.
No benchmark/quality requests ran. The original legacy arm was NOT launched.

Verified controller PID2674507 was interrupted after the terminal worker errors;
its normal teardown completes, server exit1/controller exit130, GPUs release in
0.04625s, no zombies. Four shared-memory tracker warnings remain. The original
summary retains stale running/starting status because main caught Exception but
not KeyboardInterrupt; preserve it unchanged. Failure audit classifies the
terminal state and verifies all5172 original cache files and22 source receipts:
`runtime-control/rmsnorm-noop-startup-failure.json`,
SHA256 `baf6736a06fbd851328ee99af6a0be4acb8a896acfe57ba477ccef6be477ab93`.

Repair the hook to require source coverage and replace EVERY distinct binding
of the exact source. Missing sources, repeated replacement of the same object,
late targets and wrong source/native/replacement hashes still fail. Record binding
indices and actual object counts; do not confuse Python object count with the
four-kernel scope. Numerical gates are unchanged. Also fix interrupted startup
receipts to mark failure before teardown and rethrow the interrupt, not advance
to another boot. No changes to the old artifacts.

Fixed repair qualification: ONE new GPU hook run at
`rmsnorm-intervention-bindings-gpu`, same command/resource limits as above but
new scope/log/output. Exercise the observed1/2/2/2 separate objects, both modes,
rows16/640:28 cases. Preserve the earlier16-case result separately.

Then prepare fresh copies under `rmsnorm-intervention-bindings-caches` using
the same preparer (new scope/log/output). Preserve both prior copies. Exactly
TWO new, repaired diagnostic starts, still with a correctness stop:

1. `rmsnorm-noop-bindings-control`, modecontrol, three quality passes that must
   match all nine old native-only passes exactly.
2. ONLY if that passes, `rmsnorm-legacy-bindings-only`, modelegacy, same workload.

Use corresponding `glm53-...` scope and `runtime-control/...-launch.log` names,
and the new private manifests/cache roots. All other serving commands, fixed
request counts, source freezes, receipts, original-cache checks and exact-score
gates are unchanged. These are repaired diagnostic experiments, not replacement
timing samples; the first startup failure remains part of the campaign record.

## Second failure: lifecycle investigation before further serving

The repaired control on0d5ab7dc7 also fails before health. Ranks0/1/3 finish AOT
loading; rank2 fails inside loading with the repeated-object-ID guard. No rank
has sealed. All68 recorded launchers are unchanged; target occurrences1/2/2/2.
Bare IDs cannot distinguish same-object re-resolution from recycled IDs. The
fallback warning is misleading: forced-AOT rethrows instead of recompiling.
No benchmark/quality requests; neither legacy arm has run. Controller130,
server1, normal teardown, GPUs released. New failure-status handling works.
Audit verifies5172 original files/22 sources and preserves the raw summary:
`runtime-control/rmsnorm-noop-bindings-startup-failure.json`,
SHA256 `66f3ce5a072898e823c429ebd712fcdc02c6b3db3d8bd76c564e9b2777b1f4e5`.

Before prescribing any more full-model starts, extract the seven real serialized
AOT submodule artifacts without calling the outer model deserializer or loading
weights. First ONE rank2 loader observation, output `rmsnorm-loader-observe-rank2`,
in a16GiB/no-swap scope, all four devices visible. Use fresh private copies under
`rmsnorm-loader-caches`, observer records retaining strong references and the
actual concurrent `StandaloneCompiledArtifacts.load_all` path. No forward calls,
timing, model-quality claim or numerical intervention. Check original snapshot
afterwards. Preserve any loader-only failure. Then qualify an idempotent,
thread-safe repair against actual deserialization and repeated/aliased futures,
not just separate synthetic objects. Exact source/config/binary gates remain.

The first loader observation completes seven artifacts, no forwards, and records
one genuine same-future/same-object resolution on two threads. However its empty
Triton cache causes static-bundle misses; this is NOT full loader qualification.
Preserve it. Correct the probe to use the copied serving inductor/per-device
Triton directories and reject any missing static bundle. One corrected rank2
observation at `rmsnorm-loader-cached-observe-rank2`, same limits, then the repair.

Corrected observation completes7/7 artifacts,50/50 static kernels,13 resolutions
with no repeated objects in this scheduling sample. Original5172 files unchanged.
The earlier two-thread re-resolution was a non-target kernel; it establishes a
valid lifecycle, not the exact identity cause in the failed serving process.

Idempotent repair qualification, prescribed before running it:

- Retain strong target references; serialize upstream resolution and selection.
  Already-selected target objects skip upstream cache recheck, which could undo
  the intervention. Verify source and complete config/binary on every call.
  Before seal safely reapply the selected launcher and clear its cached callable;
  after seal allow only read-only repeats of an exact known selected binding.
  New late objects, changed source, wrong configs/hashes and missing coverage fail.
- ONE GPU hook run `rmsnorm-intervention-idempotent-gpu`: same28 numerical cases,
  but11 resolutions per object (initial, repeat,8 simultaneous same/alias calls,
  one read-only post-seal).14 objects total,154 target resolutions. Require exact
  direct-config eager/repeat/changed-graph agreement, both control and legacy.
- If that passes, ONE real loader process per rank per mode,8 total, sequential
  under16GiB/no-swap; outputs `rmsnorm-loader-{control,legacy}-rank{0,1,2,3}`.
  Use fresh copies `rmsnorm-loader-qualified-caches`. Require7/7 artifacts,
  every static bundle loaded, target coverage and hash/config receipts, original
  snapshot unchanged. No weights, full-model forwards or performance claims.
- No further full-model serving start is prescribed by this qualification alone.

## Qualified idempotent serving pair

All138 CPU tests,28 exact numerical GPU cases/154 resolutions, and8 real loader
processes pass. Independent audit verifies56 artifacts/400 static kernels and
all5172 original files. Qualification audit SHAb70122c0e1becadbdb01277bd279b779abaf1f2e5189a87aa18701485634efff.

Now prescribe exactly TWO full-model starts with the original correctness stop:

1. `rmsnorm-noop-idempotent-control`, control mode, exactly one boot.
2. ONLY after every no-op score matches all nine native-only passes exactly:
   `rmsnorm-legacy-idempotent-only`, legacy mode, exactly one boot.

Prepare fresh private copies `rmsnorm-intervention-idempotent-caches` with the
same preparer; never reuse/mutate earlier failed-run caches. Use the original
serving command substitutions with these names and their matching manifests,
same150GiB/no-swap cap, native-order1/TC0/BF16fn1, forced-AOT1, fixed recipe,
three timing repeats/c1,c8,c16/cold1000-in300-out, text/image, three quality passes.
Freeze committed source/native state through BOTH arms and their audits. Audit
using `runtime-control/audit_rmsnorm_intervention.py ARM --commit FROZEN_FULL_SHA`,
which now verifies binding identities, repeated resolutions, full configs and
unchanged unrelated launchers. No source edits/commits between these arms.
Both earlier no-op failures remain part of the record. No replacement timing
starts, numerical tolerance widening, default promotion or performance claim.

## Completed static-hook pair; graph coverage invalidates causal interpretation

On3298bacd1 both prescribed starts reach health and finish every request. The
no-op repeats all scores and matches all nine native passes exactly. The legacy
arm repeats its own scores but differs from both references. Source/cache/request
audits pass, but they prove only the static callbacks they observed.

A subsequent weight-free rank2 inspection finds FOUR actual graph-held bindings,
THREE still native and unintercepted (two distinct missed objects). All50 static
bundle entries loaded, so a successful forced-AOT load/static-future census does
NOT prove complete graph coverage. Preserved failure `rmsnorm-graph-bindings-rank2/`.
The live legacy arm's actual graph globals were not captured; do NOT claim it
tested a complete four-source intervention or disproved that causal hypothesis.
The repeatable partial-intervention score vector and all measurements remain
recorded, not promoted or retroactively overwritten.

Repair binds exact target objects in every finished PyCodeCache graph module,
after async compilation resolves and BEFORE its callable returns to the model.
Keep the static hook for safe relocation/early selection; add an independent
all-loaded-graph coverage check before capture. No source/config/hash relaxation.
Require symbol/object identity, selected binary/config and source hash for every
target global; missing, changed, renamed or removed bindings fail. The source-only
synthetic probe explicitly has no graphs; real serving/loader paths require them.

Predetermine ONE new28-case source-exact GPU run `rmsnorm-graph-coverage-gpu`,
then fresh copies `rmsnorm-graph-qualified-caches`, then EIGHT independent real
loader processes, control/legacy x ranks0..3, sequential16GiB/no-swap scopes.
Outputs `rmsnorm-complete-graph-{control,legacy}-rank{0,1,2,3}`. Use the external
`runtime-control/check_rmsnorm_graph_bindings.py` inspection as well as internal
coverage: every actual target global must have the expected selected hash and
be intercepted,7/7 artifacts and50/50 static kernels, original snapshot unchanged.
No full-model replacement starts are prescribed by this qualification alone.

Completed:146 CPU tests,28 numerical GPU cases/154 resolutions, and all8 real
graph inspections pass. Independent audit f480e98300607517aad2b1b59a9cda71146dc5e85793b1d913c843ea140e0297
verifies18 actual target globals,56 artifacts,400 static kernels and5172 unchanged
original files. GPUs released. No graph-complete full-model start yet.
Before the next serving pair, extend its offline auditor to verify graph_binding
and graph_coverage receipts, graph-held symbol/object coverage and resolved_by=graph;
the previous static-only auditor is insufficient. Then freeze fresh named arms,
still requiring exact no-op equality before legacy. No old result overwrites.

## Graph-complete serving pair (prescribed 2026-09-09)

The new tracked auditor `benchmarks/analyze_glm53_rmsnorm_intervention.py`
uses `analyze_glm53_rmsnorm_graphs.py` to check every graph receipt against the
independently qualified loader inventory (1/2/4/2 module-symbol bindings by rank).
It verifies source/module hashes, complete selected configs/binaries, resolution
ordering, graph coverage before seal, and repeated/aliased objects, including
`resolved_by=graph`. Static-only receipts cannot pass. Existing source/native,
original-cache, request/token/cache, quality and exact-score gates remain.
171 CPU tests pass; replay of all eight independent loader receipts passes.
These are audit checks, not new GPU qualification or full-model results.

Prescribe exactly TWO starts, with no retries or additional timing starts:

1. `rmsnorm-noop-complete-graph-control`, mode `control`, one boot.
2. ONLY after its audit confirms every score equals all nine native reference
   passes: `rmsnorm-legacy-complete-graph-only`, mode `legacy`, one boot.

Prepare fresh copies at `perf/results/2026-09-09/rmsnorm-complete-graph-serving-caches`
using the existing preparer and exact cache-comparison audit. The command is
the native-order protocol command, with these substitutions/additions per arm:

```text
--output perf/results/2026-09-09/<ARM_OUTPUT_ABOVE>
VLLM_FORCE_AOT_LOAD=1
SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC=control|legacy
SLIMSERVE_GLM53_RMSNORM_MANIFEST=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-09/rmsnorm-complete-graph-serving-caches/<MODE>/manifest.json
VLLM_CACHE_ROOT=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-09/rmsnorm-complete-graph-serving-caches/<MODE>/cache
```

Use one uniquely named150GiB/no-swap serving scope and preserved launch log per
arm. One boot, three repetitions at c1/c8/c16, cold exact1000-in/300-out,
25 warmup/75 timed requests, text/image canaries and three full quality passes.
Native-order1/BF16fn1/TC0, fixed quant/profile/native binaries, asynchronous
execution with all legacy ordering/journal flags absent. Recheck GPU ownership
before each start. Do not change sources, binaries or commits between arms or
before their audits. No other GPU work or native builds during either arm.

After each arm releases GPUs, audit in an8GiB/no-swap CPU scope:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m benchmarks.analyze_glm53_rmsnorm_intervention control \
  --commit <FROZEN_FULL_SHA>
```

Use `legacy` for the second arm. Audits create new exclusive output files
`runtime-control/<ARM_OUTPUT_ABOVE>-analysis.json`. Every score mismatch is
preserved. Exact no-op mismatch or any coverage/source/workload failure stops
the sequence. A qualified legacy/reference mismatch is causal evidence only
about this complete four-source intervention, not permission to widen quality
gates or promote native ordering/TC. All prior failed and partial runs remain.

### Completed result (2026-09-09 23:05 UTC)

Both starts and all audits complete on76afce776, with frozen sources through
the combined conclusion. No-op matches all9 native passes exactly (27 pairs);
complete legacy matches all3 older instrumented passes exactly (9 pairs).
Every text and needle-token score agrees, not merely their averages. Both modes
repeat exactly internally.9 actual graph bindings per arm,22 sources/native
binaries/5172 original files verified; private seed files unchanged. Both servers
and controllers exit0, GPUs released. All fixed requests and canaries complete.

The four source-bound RMSNorm configuration changes are sufficient to reproduce
this complete old/native score difference. This is causal evidence for the fixed
workload and AOT source set, not individual-configuration necessity or universal
determinism. No TC exoneration, new quality tolerance or performance promotion.
Preserve all earlier partial/failed results. This protocol is complete: do not
rerun its arms. Next qualify an explicit fresh-compilation reduction policy.
Combined conclusion SHA733c8b239cdb16f70c1914d5af878be0681ac78b7deb58266c3ec3aa8a9c811b,
`runtime-control/rmsnorm-complete-graph-pair-conclusion.json`.
