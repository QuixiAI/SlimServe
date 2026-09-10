# GLM53 bounded prompt-score qualification

Status: full serving series on414829029 and final closure PASS. Source freeze
ended. Retain the memory fix as a qualified opt-in; no production-policy/default
or speed promotion. Commands below are historical, not restart instructions.

## Hypothesis and scope

The completed indexer-serving series logged eight recovered 4,718,592,000-byte
allocation warnings in every arm. This is the 2 MiB-rounded size of a
[7616,154880] FP32 matrix. Source inspection suggests full-chunk prompt score
conversion/log_softmax; the actual warning has not been stack-attributed.

Keep the complete vocabulary projection and TP gathering unchanged, then bound
only row-independent scoring temporaries to 1024 rows using the existing Sampler
conversion/log_softmax, top-k and inclusive-rank operations. Requested outputs
are not scratch and can still be large for full-vocabulary requests. The existing
639-row score-journal path remains unchunked. No arithmetic substitute or fallback.

## Prescribed isolated CUDA process v1

Exactly one GPU0 process, one fresh private Inductor/Triton namespace. No model
server, native build, power/clock changes, simultaneous GPU work or retries. Stop
on any failure, preserve the partial record, and prescribe any follow-up separately.
GPU scope 16 GiB host RAM/swap0; CPU checks 8 GiB/swap0. Freeze listed probe
sources from launch until process exit. GPU release is checked after exit.

The script fixes 60 cases in order: BF16 rows1/639/1024/1025/2051/7616, vocabulary
154880, four score modes and k0/k5; then FP16/FP32 at1025 rows with raw modes/k5;
then all-equal BF16 ties at1025 rows, vocabulary154880/k5 and17/k17, four modes.
Offset rows and padded vocabulary test noncontiguous inputs and guards. Actual
Sampler methods, including compiled CUDA inclusive rank, are mandatory.

Each case runs control then chunked, one full untimed warmup per arm, then three
measurements per arm. Exact dtype/shape/value equality for scores, token IDs and
ranks is required for every call; input/guards unchanged. CUDA-event and wall
times plus peak incremental allocated bytes are recorded, not serving TPS.
After the7616-row raw-logprobs/k0 measurements, separately profile both warmed
arms with allocation history and operator shapes. These isolated allocation
stacks do not, by themselves, attribute the live-server warning.

```bash
systemd-run --user --scope --unit=glm53-prompt-scores-gpu-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-10/prompt-scores-cache-v1/inductor TRITON_CACHE_DIR=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-10/prompt-scores-cache-v1/triton .venv/bin/python -m benchmarks.kernels.check_glm53_prompt_scores --output perf/results/2026-09-10/prompt-scores-gpu-v1
```

## Required later gate

Only after local exactness and memory qualification, wire the helper into the
runner with small-request/journal semantics intact. Then prescribe a real fixed
`glm53-nvfp4-4`/`rtx6000`, recipe v1 control/candidate/return comparison with
unchanged quality gates, exact per-token score comparison, text/image canaries,
cold exact-token throughput and allocation-warning census. No serving promotion
based on the isolated probe. The concrete serving prescription is below.

## CPU result

184 tests pass in5.98s: helper tests and existing score-journal tests, CPU emulator
only. Raw `perf/results/2026-09-10/runtime-control/prompt-score-chunks-cpu-v1.xml`.

## Completed isolated result

One prescribed process, all60 cases and480 checked outputs pass exactly; three
timings per arm/case. No retry or replacement. Hardware UUID/driver/600W unchanged,
GPUs released after exit. Raw `perf/results/2026-09-10/prompt-scores-gpu-v1/`;
`result.json` SHA68b82a83a8a7bc0ba8cc9b4ae349e6de6a0882f280e51edfec41aa99993cc271.
Source hashes in that completed receipt precede this result documentation.

At7616 rows/BF16/raw-logprobs/k0, incremental peak allocation falls from
9,437,184,000 to1,270,996,480 bytes (86.5% less). CUDA median20.122 to19.597ms;
all three control20.093..20.140 and chunked19.584..19.606ms. This is a warmed,
fixed-order isolated scoring measurement, not a decode/prefill/serving speed win.
Small1025-row shapes have expected extra-launch overhead (~3..6%); this does not
affect the unchanged<=1024-row runner fast path planned for integration.

The separate allocation traces contain exactly two large control allocations,
4,718,264,320 requested bytes each: FP32 conversion (`aten::_to_copy`) and
`aten::_log_softmax`, with Python/C++ stacks through Sampler.compute_logprobs.
The chunked trace instead contains16 score allocations, maximum634,388,480 bytes
(1024 rows), with277,544,960-byte tail allocations. Rounded allocator accounting
explains the4,718,592,000-byte control blocks. This establishes isolated operation
attribution, not an actual live-server OOM stack. Serving integration and warning
elimination remain pending; profile/default/baseline are unchanged.

## Opt-in runner integration

`SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS=1` selects the qualified1024-row helper only
for GLM53 prompt-score chunks larger than1024. Absent/0 is the existing path.
The entire vocabulary projection, TP gather, small-request operations, bounded
score journal and asynchronous CPU transfers remain unchanged. No dtype change.
The knob is outside the compiled model/AOT cache key. An initially explored
generic vLLM environment setting was removed before GPU/serving use to preserve
the existing compiled-model environment and source identity.

The distinct `SLIMSERVE_GLM53_PROMPT_SCORE_DIAGNOSTIC=control` schema reuses the
qualified no-op indexer loader, private-cache relocation, scheduler/capture checks
and independent graph audit in EVERY arm. It does not activate indexer correction
or allocate selection arenas. Completed AOT evidence and the isolated scoring
receipt are pinned inputs; old frozen validators are not rerun. Chunked is held
to the same exact historical token-score requirement as both controls.

## Prescribed serving series v1

After the complete CPU gate and commit: exactly control -> chunked -> return-control,
one private copied original-AOT namespace/start each, chunk flags0/1/0. All model
kernels must remain at their qualified original bindings. No fresh-compilation,
arithmetic replacement, retry, replacement start, cache deletion or quality waiver.
Stop on any failure and close/audit the partial series before source edits.

Fixed profile `glm53-nvfp4-4`/`rtx6000`, recipe v1, TP4 Marlin, no EP/speculation,
BF16 activation/KV/lm_head, native-order1/BF16fn1/TC0 diagnostic reference. Registry
discovery must still identify the exact compatible profile. Three repeats each of
exact1000/300 cold-prefix c1/c8/c16; three4096-token text/168-token needle quality
passes, text/image canaries, and three cold32K/128K requests. All per-token quality
vectors must equal the historical original exactly. Census every allocation warning
per arm; report quality-scoring timing separately from decode/prefill timing.

CPU prepare/controller/audits8GiB, serving150GiB, swap0. No native builds or other
GPU jobs. Freeze all source/docs/commits from preparation through successful closure
or audited terminal failure. Private copies and all failed artifacts are retained.

```bash
systemd-run --user --scope --unit=glm53-prompt-score-serving-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.prepare_glm53_prompt_score_serving --output perf/results/2026-09-10/prompt-score-serving-v1

systemd-run --user --scope --unit=glm53-prompt-score-serving-v1-controller -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 bash -c 'for prompt_score_case in control chunked return-control; do .venv/bin/python -m benchmarks.kernels.run_glm53_geometry_serving launch "perf/results/2026-09-10/prompt-score-serving-v1/${prompt_score_case}/manifest.json" || exit "$?"; done'

systemd-run --user --scope --unit=glm53-prompt-score-serving-v1-close -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.run_glm53_geometry_serving close perf/results/2026-09-10/prompt-score-serving-v1
```

CPU history: runner-v1 252 pass/8.93s; runtime-only runner-v2 plus campaign323
pass/14.65s. First serving CPU gate387 pass/two fixture failures: missing required
native-order env in the conflict test and an accidentally matching empty inventory
in the predecessor test; corrected in tests. Raw XMLs remain under
`perf/results/2026-09-10/runtime-control/prompt-score-*.xml`.

Final521 CPU tests pass/24.24s,14 upstream Torch deprecation warnings (v3 XML).
Previous expanded v2 gate519 pass/24.28s. The extra final tests cover diagnostic
classification and single installation of the unchanged loader; per-quality-pass
wall timing is recorded for successful and failed passes. Ruff/diff pass.
Inspection joins1,042 source/evidence receipts and5,172 original cache files,
without preparing a cache or launching a server. Inspection SHA
834a1d92415f21eb491b3602918cc894778e4e57d02e2397cfb382ef3aaf44d9, raw
`runtime-control/prompt-score-serving-inspect-v1.json` under2026-09-10.
This inspection precedes the final timing/test edits; preparation pins committed
current bytes. Original helper/Sampler/rank implementation hashes must match the
completed isolated GPU receipt before any serving arm is admitted.

## Completed serving result

Exactly control/chunked/return-control, one private namespace/start each. All27
exact1000/300 timing rounds (225 measured requests), nine complete quality passes,
text/image canaries and cold32K/128K checks pass. Every4096 text and168 needle-token
score equals the historical original EXACTLY in every pass. Every arm is internally
repeatable; all unchanged quality windows pass. No quant/arithmetic/native changes.

All four workers in every arm pass before-forward/capture provenance checks:
seven AOT roots/46 entries,25 original launchers/two original target bindings,
original-policy8192-row scheduler envelope. No correction kernel/selection arena
is activated. All92 non-target bindings and the original target bindings are exact
to the qualified control, including return. No fresh-compilation claim follows.

Recovered4,718,592,000-byte allocation warnings: **8 -> 0 -> 8**. Both controls
have two warnings per rank, all during the first quality pass; candidate has zero
allocation-failure warnings over its complete workload. The bounded post-projection
scoring intervention therefore eliminates the observed recoverable allocation
failures in this serving series. Isolated stacks identify conversion/log_softmax;
no live OOM allocation stack was collected. Do not claim every serving allocation
or full-vocabulary requested-output allocation is bounded by the helper.

Diagnostic E2E tok/s, median [min,max], three repeats:

| Arm | c1 | c8 | c16 |
| --- | ---: | ---: | ---: |
| Control | 157.397 [157.032,157.517] | 578.121 [577.651,579.114] | 779.971 [777.735,780.952] |
| Chunked | 156.983 [156.879,157.144] | 579.758 [578.265,581.819] | 778.748 [778.438,779.963] |
| Return | 156.878 [156.844,157.158] | 579.686 [577.869,580.137] | 781.038 [780.347,782.019] |

Cold engine scheduled-to-first-token ms, median [min,max], all cached_tokens=0:

| Arm | 32K | 128K |
| --- | ---: | ---: |
| Control | 2580.002 [2577.780,2583.408] | 10891.981 [10846.741,10920.297] |
| Chunked | 2589.036 [2586.287,2590.958] | 10897.479 [10857.942,10930.248] |
| Return | 2586.528 [2584.310,2588.467] | 10924.595 [10880.917,10957.556] |

Complete quality-pass wall seconds in order: control89.689/89.928/89.758,
chunked89.581/89.902/88.667, return89.751/89.614/88.776. Medians89.758/89.581/
89.614s are effectively neutral; client serialization and full-model work remain.
Neither these timings nor the decode/prefill differences establish a speed win.
The retained value is lower score scratch and removal of recovered allocator failures.

Startup160.078/160.112/158.082s; GPU release0.407/0.141/0.138s. One transient
teardown zombie and one shared-memory resource-tracker warning recorded in EACH
arm. Do not describe allocation success as clean teardown. All process/audit exits0,
GPUs released, driver/UUID/600W unchanged. No retries, replacements or exclusions.

Closure verifies1,042 source/evidence receipts and5,172 original files, raw
`perf/results/2026-09-10/prompt-score-serving-v1/closure.json`, SHA
27c35be7e4a7bbd5de6b2b2a0d48b8af1a306c38e450590e0ff85e516eb8529b.
All private caches, logs, manifests and failed CPU reports retained. Freeze ended
before these result edits. Consume the completed receipts; do not rerun this closed
series or its expired frozen validators.

Decision: retain `SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS=1` as an opt-in memory fix.
The default profile remains unchanged because this series used the native-order1
diagnostic reference. Next qualify rollout under the production ordering policy,
then return to measured decode/prefill bottlenecks. No further GPU job is prescribed
yet; do not silently enable native ordering or change the selected recipe.
