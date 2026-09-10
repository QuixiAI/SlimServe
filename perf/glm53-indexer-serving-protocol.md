# GLM53 selective indexer correction: model diagnostic

Status: integration CPU-qualified (554 tests); commit before the prescribed series.
Production defaults, recipe and original arithmetic are unchanged.

## Evidence and hypothesis

Completed AOT/leaf series `cf95040f3`, pair SHA
89fecf567bfe98ebcdb8ae6b948db7ad7387f4492877cba52c1f90ba65206061, verifies
480 actual bound-leaf cases and2,400 eager/replay observations, non-target exactness,
static image identity and arena guards. Consume those pinned completed receipts;
do not rerun old frozen readers. The qualified correction kernel, loader, static
ABI and selection writes remain byte-identical.

Hypothesis: fixing the measured indexer LayerNorm cancellation may preserve or
improve model-quality scores. Local one-ULP accuracy alone is insufficient; the
previous KV-only candidate passed local accuracy but regressed model quality.
This extra-launch diagnostic is not proposed as a speed optimization.

## Serving lifecycle and scope

Use the shared real-profile before-forward/capture lifecycle, actual graph roots,
independent binary and non-target inventory, private caches and one-attempt
controller. Distinct opt-in flag `SLIMSERVE_GLM53_INDEXER_CORRECTION` accepts only
control/correction with its prepared manifest. No geometry/KV/legacy diagnostic
mixing or alternative compiler policy. CLI/worker/campaign enforce the same flag.

The actual scheduler maximum must equal the runner maximum and fit8192 rows;
every configured capture size must fit. Reject DP/DCP>1, sequence parallelism or
microbatching. Keep per-binding fixed1,048,832-byte arenas; verify addresses and
end guards before forward, before capture and after capture. No leaf changes.

Both arms additionally check the real batch-padding dispatcher before forward:
positive integer input<=padded<=scheduler maximum, no DP token synchronization or
microbatch output. Require a single host thread, and one steady-state live GPU
stream. Startup/capture stream transitions explicitly synchronize once before
continuing; an in-capture transition is rejected. Record newly observed batch
shapes/streams and all transitions without per-token file writes. Existing vLLM
capture/wrapper/parallel-stream implementations are frozen evidence. These checks
are diagnostic instrumentation; there is no arbitrary cross-stream reentrancy
claim and no production TPS promotion.

## Required next gate

CPU tests must cover default inertness, exact manifest/evidence admission,
distinct policy/event dispatch, legacy KV/geometry regression, actual Torch
AOT/capture plumbing with simulated driver, scratch bounds/lifetime, and runtime
padding/thread/stream rejection. CPU8GiB/swap0, CUDA hidden, one OMP thread.
The complete CPU gate must pass before the series below starts.

## Prescribed model series v1

Fixed `glm53-nvfp4-4` / `rtx6000`, recipe
`glm53-redhatai-nvfp4-fp8-kda-tp4-v1`, TP4 Marlin, no EP/speculation, BF16
activation/KV/lm_head, qualified original attention combo and native binaries.
Native-order1/BF16fn1/TC0 match the completed diagnostic reference; native-order
remains off in production. No power/clock/quant/compiler or quality-floor changes.

Exactly one control, one correction, one return-control start, in that order;
one fresh private namespace each. All5,172 original files plus the single shared
qualified correction source are copied independently into each namespace. No
saved replacement cubin is injected. The unchanged loader compiles/checks/adapts
the qualified JIT as in the completed AOT series. All eight prior process/leaf
audits, graph bindings and input/output/flag receipts are joined without rerunning
expired validators. Only the closed loader notebook is released among prior
frozen files; the serving integration/client/stream sources are newly frozen.

Use the unchanged shared workload: exact1000 input/300 output, cold-prefix,
c1/c8/c16 with three repeats each; three quality passes of4,096 scored text
tokens plus all needle contrasts; text/image canaries; cold32K/128K prefill.
Both controls must reproduce the pinned historical per-token vectors exactly.
Every arm must repeat within-start exactly. The candidate's unchanged quality
gate is evaluated and recorded, never waived or called passing if it fails.
A repeatable candidate quality regression alone permits the prescribed return
control to establish reversibility. Any other load/validity/audit/teardown failure
stops the series; retain/audit partial evidence and do not retry/replace a start.
All92 non-target AOT bindings must match, and return restores the full inventory.

CPU preparation/controller/audit8GiB, serving150GiB, swap0 throughout. No builds,
other GPU jobs, source/doc edits or commits from preparation through successful
closure or audited terminal failure. The controller discovers the compatible
registry profile, checks fixed hardware and GPU release, and enforces predecessors.
The original historical oracle failure remains recorded even if correction passes
its independent gate. No production/default or stable TPS promotion in this series.

After CPU qualification and commit, from the repository root:

```bash
systemd-run --user --scope --unit=glm53-indexer-serving-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.prepare_glm53_indexer_correction_serving --output perf/results/2026-09-10/indexer-serving-v1

systemd-run --user --scope --unit=glm53-indexer-serving-v1-controller -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 bash -c 'for indexer_serving_case in control correction return-control; do .venv/bin/python -m benchmarks.kernels.run_glm53_geometry_serving launch "perf/results/2026-09-10/indexer-serving-v1/${indexer_serving_case}/manifest.json" || exit "$?"; done'

systemd-run --user --scope --unit=glm53-indexer-serving-v1-close -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.run_glm53_geometry_serving close perf/results/2026-09-10/indexer-serving-v1
```

The historical geometry filename is the shared controller, not the selected
intervention. Correction's manifest selects its flag, policy, graph/arena auditor,
case order and closure result keys. Run closure after success or first failure,
without launching unused cases. Preserve logs, failed attempts and private caches.

CPU preparation may inspect completed evidence without making a serving cache:

```bash
systemd-run --user --scope --unit=glm53-indexer-serving-inspection-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.prepare_glm53_indexer_correction_serving --inspect-only --output perf/results/2026-09-10/runtime-control/indexer-serving-inspection-v1.json
```

That inspection completed, joining1,032 source/evidence receipts and5,172 original
files. SHA06d82d9ba34a5e44a574d30bc6d7e6b128e4277770feac89442ced5c59792afb.
It predates final formatting/tests/protocol edits; preparation freezes the committed
bytes, not those earlier integration hashes. No GPU/cache/start was consumed.

## Completed CPU gate

Final554 tests pass/53.12s,14 upstream Torch deprecation warnings. Ruff and diff
checks pass. Includes actual Torch AOT/capture plumbing with simulated driver and
CPU-backed simulated CUDA arenas, independent runtime/graph audit, admission,
fresh shared-source copying, manifest roundtrip, quality-failure closure and
legacy loader/serving/campaign regressions. GPU/serving qualification is pending.

Raw under `perf/results/2026-09-10/runtime-control/`:
`indexer-serving-initial-cpu.xml`120 pass/14.63s;
`indexer-serving-new-cpu-v1.xml`82 pass/two fixture-copy failures (Torch's copy
restoration expected metadata absent from the minimal simulated tuner), corrected
by constructing a second explicit fixture owner;
`indexer-serving-new-cpu-v2.xml`108 pass/6.83s;
`indexer-serving-cpu-final-v2.xml`547 pass/two test-isolation failures (a new test
mutated a nested shared registry dictionary), corrected with a deep-copied plan;
`indexer-serving-cpu-final-v3.xml`554 pass/53.12s after additional correction
closure tests. The first final command named nonexistent CLI/ordering test files
and stopped before collection; no code/model/case attempt was consumed.
