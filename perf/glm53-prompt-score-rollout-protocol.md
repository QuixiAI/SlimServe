# Prompt-score memory fix: production-policy rollout v1

Status: CPU-qualified365 tests/35.88s, no GPU start yet. Previous native-order1 diagnostic series
is complete, exact-quality qualified, with allocation warnings8 ->0 ->8; closure
SHA27c35be7e4a7bbd5de6b2b2a0d48b8af1a306c38e450590e0ff85e516eb8529b.

## Fixed target and reason for this gate

`glm53-nvfp4-4`/`rtx6000`, recipe `glm53-redhatai-nvfp4-fp8-kda-tp4-v1`;
TP4 Marlin, no EP/speculation, BF16 activation/KV/lm_head, BF16fn1/TC0.
Native ordering is0 (production), all arithmetic/loader/profiling diagnostics off.
Only `SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS` changes0/1. No model projection,
quant, hardware power/clock, native kernel or compiler arithmetic policy changes.

Native ordering participates in the AOT key. This series therefore uses five
independently EMPTY private caches with normal compilation, not copied diagnostic
artifacts or forced AOT loading. Startup/compiler variability is observable and
must not be hidden by selecting an old cache or replacing a slow start.

## Prescribed series and gates

Exactly control0, candidate-1=1, candidate-2=1, candidate-3=1, return-control0.
One start each; no retry/replacement/exclusion. All use real SlimServe profile
discovery and launch. One full warmup per concurrency, three measured repeats of
exact1000/300 c1/c8/c16 with unique cold salts; text/image canaries, one4096-text/
168-needle-token quality pass per start, three cold32K/128K repetitions plus warmup.

Use the existing unchanged0.01 nat/token aggregate and EACH-window quality gate.
Before advancing, compare each actual start against the last two retained BF16fn1,
native-order0 production reference starts (`mhc-paired-serving-remainder/`, boot1/2).
These chronological references are pinned before observing new results, not picked
by score. At completion, compare the three actual candidates against the two fresh
controls using the existing1/3/1 gate. No synthetic duplicated observations, exact
score requirement borrowed from native-order1, or tolerance relaxation. All raw
token scores and deltas remain recorded. A historical-quality or execution failure
stops the series without consuming remaining starts.

Predeclared performance sanity gate: each candidate start's c1/c8/c16 median must
be at least97.5% of the lower fresh-control median; across all five start medians,
max/min must be<=1.025 at each concurrency. Each candidate's cold32K/128K engine
TTFT median must be<=102.5% of the slower fresh control. These are nonregression/
reproducibility gates, not criteria for claiming a speed win.
In addition, EVERY start must reach97.5% of the retained production BF16fn1
c1/c8/c16 medians156.791/578.993/779.940 tok/s. Five equally slow starts cannot
redefine the known throughput floor. Identity/driver/600W are checked before and
after each live start against the completed current-hardware receipt.

Memory rollout gate: both controls reproduce allocation-failure warnings and all
three candidates have ZERO such warnings. Preserve all other warnings and teardown
artifacts. Every started workload must reach health, finish real requests, pass
correctness and release all GPUs. No inference/build overlaps.

Parent controller/preparation/tests/audits8GiB; serving150GiB; swap0. Sources,
native libraries, profile and prompt are frozen from launch through completed or
audited failed report. No edits/commits during the series. Artifacts/private caches
are retained. All starts are recorded regardless of speed; never launch after a
terminal failed report. Changing the profile default requires all gates to pass.

After CPU qualification and commit:

```bash
systemd-run --user --scope --unit=glm53-prompt-score-rollout-v1-controller -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.run_glm53_prompt_score_rollout --output perf/results/2026-09-10/prompt-score-rollout-v1
```

The runner persists `rollout.json`, per-arm serving/quality/timing logs, independent
audit JSON and final file inventories. No default promotion or new baseline claim
is made by a successful process start, isolated helper test or partial series.

## CPU preparation evidence

Final365 tests pass/35.88s, including unchanged quality-envelope regression,
actual production profile resolution, inherited-policy isolation, one-shot start
order/interruption/failure preservation, source/runtime admission, memory scopes,
warning census and absolute/spread/relative performance gates. Ruff/diff pass.
Raw `runtime-control/prompt-score-rollout-cpu-v3.xml` under2026-09-10; earlier
v1/v2 reports363/364 pass are retained. CPU-only inspection verifies the pinned
historical quality means-2.731834/-2.735399 and current recipe/TP4/profile env.
The first inspection failed a harness check because `Plan.entry_file` is a lazy
environment-dependent property read after restoring the parent environment. The
check now runs inside the prescribed environment; inspection-v2 passed, with a
real-registry regression test. No model/GPU/cache/start was consumed by inspection.
