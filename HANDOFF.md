<!--
Three active campaign handoffs live in this file. They cover different
platforms and different hardware, and each is current for its own campaign;
none supersedes another. The first two met on this file in the 2026-08-28
merge of origin/main and were joined rather than reconciled; the third was
added 2026-09-04.

  1. GLM-5.3-Flash on RTX PRO 6000 Blackwell (sm_120) -- section below
  2. NVFP4-on-Metal campaign (M1 Ultra / M5 Max)      -- second section
  3. MI300X GGUF profile record                       -- third section
-->

# HANDOFF - GLM-5.3-Flash on 4x RTX PRO 6000 Blackwell (sm_120), branch glm53-flash-sm120 (2026-09-08: reproducibility audit and campaign continuation)

## Mission

Serve **GLM-5.3-Flash from `RedHatAI/GLM-5.3-Flash-NVFP4`** on the sm_120 box
(4x RTX PRO 6000 Blackwell, PCIe, no NVLink, CUDA 13.0, sm_120) through
the `glm53-nvfp4-4` profile's new `rtx6000` record, then optimize the
QuixiCore/fork kernels for this exact model, quant and card toward the
per-token byte floor. Directive (operator, 2026-09-04): "an overfitted
scenario where we've got a specific quant of a specific model on specific
hardware running as fast as possible, approaching the limits of physics";
methodical, measured every step, no regressions, clean code. The plan,
physics, research digest and phase gates are in
`docs/glm53-flash-sm120-plan.md`; every measurement is a
`perf/optimization_status.md` entry and the record rows are in
`perf/baseline_status.md`.

## State (2026-09-09, early morning) - start here

Commit identity is Auroter <auroter@users.noreply.github.com>, not Eric Hartford.
The 61 misattributed local campaign commits have been corrected without changing
trees, messages or dates. Original hashes in this handoff and raw receipts map
through `perf/glm53-sm120-authorship-map.json`; a local backup branch preserves
the old history. Upstream history is unchanged and nothing was pushed.

### Latest checkpoint (2026-09-10): fresh production-policy rollout CPU-qualified

New `benchmarks/run_glm53_prompt_score_rollout.py` prescribes exactly one control,
three chunked candidates, one return with independent EMPTY compile caches and
native-order0 (production). No AOT transplant/forced loading or arithmetic diagnostics.
Three exact1000/300 c1/c8/c16 repeats, text/image, one quality pass/start and cold
32K/128K. Each start must satisfy pinned chronological production quality references;
final candidates also satisfy unchanged1/3/1 window/aggregate gates against fresh
controls. No duplicated observations or tolerance relaxation. Strict cold counts,
native/hardware/source freeze, all artifacts retained, no retry/replacement.

CPU365 pass/35.88s and real-registry/evidence inspection pass. No GPU start yet.
NEXT after commit: run the exact controller command in
`perf/glm53-prompt-score-rollout-protocol.md`. CPU8GiB/serve150GiB/swap0. Also
requires candidate performance nonregression,<=2.5% across-start TPS spread,
historical absolute throughput floor, and allocation warnings in controls but zero
in all three candidates. Failure stops rollout and leaves profile default unchanged.
The source freeze lasts through terminal report. Parent process owns no GPU context.

### Previous checkpoint (2026-09-10): prompt scoring eliminates recovered OOM warnings

Series on414829029 CLOSED/PASS: exactly control/chunked/return, all27 timing rounds,
nine quality passes, text/image and cold32K/128K pass. Every4096 text/168 needle
score equals the historical original exactly. All original AOT kernel bindings
remain exact; no correction/selection arenas. Recovered4.7GB allocator warnings
are8 ->0 ->8 (two/rank in each control, zero candidate). No restart/replacement.

E2E median c1/c8/c16: control157.397/578.121/779.971, chunked156.983/579.758/
778.748, return156.878/579.686/781.038 tok/s. Quality-pass medians89.758/89.581/
89.614s; cold32K/128K effectively neutral. This is a memory/reliability win, not a
speed claim. Isolated score scratch9.44GB ->1.27GB; live full-stack peak not measured.

Retain `SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS=1` as opt-in. No profile/default/quant/
native change: production ordering remains off, while the qualified series used
native-order1. NEXT: qualify this memory fix under the production ordering policy
before default rollout; then resume measured decode/prefill optimization. Do not
enable arithmetic diagnostics or change quant to seek a benchmark win.

Closure1,042 receipts/5,172 original files verifies; GPUs released, freeze ended.
Each arm still has one resource-tracker shared-memory warning/transient zombie.
Raw `prompt-score-serving-v1/closure.json` under2026-09-10, SHA
27c35be7e4a7bbd5de6b2b2a0d48b8af1a306c38e450590e0ff85e516eb8529b.
Full protocol/ranges/limits: `perf/glm53-prompt-score-protocol.md`. No next GPU job
prescribed. Use completed receipts after source edits, not expired validators.

### Previous checkpoint (2026-09-10): prompt-score serving integration CPU-qualified

Opt-in `SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS=1` wires the qualified helper into
GLM53 requests>1024 rows; complete projection/TP gather, small requests, journal
and async transfers unchanged. Defaults remain off; no quant/native/compiler
change. Distinct prompt-score diagnostic uses the exact qualified no-op indexer
loader and independent graph/lifecycle audits in every arm, not correction.
Final521 CPU tests pass/24.24s; inspection joins1,042 receipts/5,172 original files.
No serving process has launched yet. NEXT after commit: prepare, then exactly
control/chunked/return-control, one private original-AOT cache/start each. Every
arm must match historical full token-score vectors exactly. Three repeats of
exact1000/300 c1/c8/c16, text/image, quality and cold32K/128K; census allocation
warnings and record quality-pass wall timing. Full commands/resource limits and
failure history: `perf/glm53-prompt-score-protocol.md`. Freeze through closure.

### Previous checkpoint (2026-09-10): prompt-score scratch bounded, CUDA-qualified

Uninstalled helper `vllm/v1/sample/prompt_logprobs.py` chunks only post-projection
scoring into1024 rows with the existing sampler operations. CPU184 pass; exactly
one GPU process/60 cases/480 checked outputs all exact, including actual compiled
inclusive ranks, top-k ties, BF16/FP16/FP32, guards and noncontiguous head views.
At7616x154880 BF16/k0, incremental peak9.44GB ->1.27GB (86.5% less), isolated
CUDA median20.122 ->19.597ms. No model/quant/native/profile/default change.
Actual isolated allocation stacks identify both FP32 conversion and log_softmax;
the previous live-server warnings are not yet stack-attributed or eliminated.
GPU released, freeze ended. Raw `prompt-scores-gpu-v1/result.json` under2026-09-10,
SHA68b82a83a8a7bc0ba8cc9b4ae349e6de6a0882f280e51edfec41aa99993cc271.
NEXT: runner integration preserving the complete lm_head/TP gather and<=1024-row
score-journal path, then a prescribed fixed-profile quality/warning/timing series.
Full commands and limits: `perf/glm53-prompt-score-protocol.md`.

### Previous checkpoint (2026-09-10): selective indexer correction passes model quality

The full prescribed series on `7b8f93afb` is complete: exactly one control,
correction and return-control start, three repeats each, all audits/closure pass.
All nine complete text/needle score vectors match the historical original EXACTLY.
All unchanged quality windows, text/image canaries, exact1000/300 c1/c8/c16 and
cold32K/128K checks pass. All92 non-target AOT bindings stay exact; return restores
the full original inventory. The corrected path does not reproduce failed no-combo.

Actual runtime maximum8192, capture sizes1..64, one live thread/stream per rank;
two distinct1,048,832-byte arenas/rank retain addresses/end guards around capture.
This does not count live corrected elements or establish arbitrary cross-stream
reentrancy/fresh-compilation invariance. Historical original oracle stays failed.

E2E median c1/c8/c16: control157.160/579.913/781.671, correction
156.457/578.474/777.935, return156.954/579.502/781.081 tok/s. Small extra-launch
cost, not a speed win; prefill approximately neutral against return. Retain the
correction as a quality-qualified opt-in diagnostic, not a production default.
No quant/native/quality-floor or stable-baseline changes.

Closure verifies1,032 receipts and5,172 original files. GPUs released, driver/
UUID/600W unchanged, no retries/replacements. Freeze ended before these edits.
Raw `perf/results/2026-09-10/indexer-serving-v1/closure.json`, SHA
0d1302e05d6f2350d1bcf60be4d04691f489d6c3040f1bbca37fd890d139fa8b.
All protocols, manifests, caches and unsuccessful CPU development reports retained.

NEXT: resume measured performance work from this stable control, not another
unqualified arithmetic substitution. Before pursuing fused correction, measure
actual launch/selection coverage on model activations; unchanged scores alone do
not show how many values it repaired. Also inspect the repeated4,718,592,000-byte
allocator warnings: eight in EVERY arm, recovered with completed requests. The
size matches a2MiB-rounded [7616,154880] FP32 prompt-logit/score buffer; source has
full-chunk logits and FP32 log_softmax, but no allocation stack yet proves the
specific operation. Resource-tracker warnings/transient teardown zombies remain
recorded; no claim of clean allocation/teardown internals. No next GPU job prescribed.

### Previous checkpoint (2026-09-10): indexer serving integration CPU-qualified

Opt-in `SLIMSERVE_GLM53_INDEXER_CORRECTION=control|correction` now joins the
completed AOT/leaf receipts to the shared real-profile load/capture/workload
lifecycle. Qualified correction kernel/loader/ABI bytes remain exact. Separate
events, schema, CLI/worker admission and offline graph/arena checks; KV remains
the shared helpers' default policy and its regression tests pass.

The actual scheduler/capture envelope must fit8192 rows. Runtime padded batches
are checked before forward; DP/DCP/SP/microbatching and multiple worker threads
are rejected. Startup stream changes synchronize explicitly; live execution must
keep one stream. Both controls have identical diagnostic instrumentation. Arenas
retain guarded stable addresses through capture; arbitrary cross-stream calls
remain unqualified. No production baseline claims from these instrumented timings.

Final CPU554 pass/53.12s,14 upstream warnings; Ruff/diff checks pass. Earlier
fixture-copy and test-isolation failures are retained and corrected in tests only.
Inspection joins1,032 source/evidence receipts and5,172 original files; no model
start yet. NEXT after commit: prepare then exactly control/correction/return-control
under `perf/results/2026-09-10/indexer-serving-v1/`, per
`perf/glm53-indexer-serving-protocol.md`. CPU8GiB/serve150GiB, swap0. Freeze sources
through terminal closure; no retries/replacements or concurrent GPU/native work.
If a series is running, resume its controller; do not create a replacement.

### Previous checkpoint (2026-09-10): all-rank indexer AOT/leaf qualification passes

The prescribed series on `cf95040f3` completes: control ranks0..3 then correction
ranks0..3, one private cache/start each, no retries. All480 bound-leaf cases,
960 unique phases and2,400 eager/replay observations match the qualified isolated
outputs/flags. Every process loads seven actual roots/46 entries,25 launchers and
two target bindings. All92 non-target bindings compare exactly; correction adds
two independently observed static launchers per rank. Arena/input/output/replay
guards pass. Four small norm tensors per process; no model forward or timings.

Final pair audit passes; original5,172-file cache retained, GPUs released and
UUID/driver/600W configuration unchanged. Freeze ended before these notebook edits.
Raw `perf/results/2026-09-10/indexer-aot-v1/pair-analysis.json`, SHA
89fecf567bfe98ebcdb8ae6b948db7ad7387f4492877cba52c1f90ba65206061.

NEXT: opt-in full-model correction diagnostic using these completed receipts;
validate the actual scheduler/padding bound against the8192-row arena and the
serialized/capture lifecycle. CPU-qualify integration before prescribing the
control/correction/return quality series. Do not rerun expired frozen readers
after source edits. No production/default/quant or speed-baseline promotion.

### Previous checkpoint (2026-09-10): runnable indexer AOT/leaf series CPU-tested

The ordered preparation/controller/independent auditor now extend the shared
AOT lifecycle with actual graph-held leaf calls. Control ranks0..3 then correction
ranks0..3,60 leaf cases per process (two graph bindings x30 cases), both input
phases and five eager/replay observations. Output/flag hashes join the completed
isolated correction directly; guards include all unused selection-arena rows.
No model forward; explicitly four small norm tensors per process, not zero.

CPU336 tests pass/20.74s,14 upstream Torch warnings; Ruff/diff checks pass.
The actual preparation now includes all120 pinned prior case receipts. Qualified
kernel and loader policy remain unchanged; shared lifecycle/probe/auditor changes
are CPU-regression-tested, not yet GPU-qualified.

NEXT after commit: exactly the eight-process AOT/leaf series prescribed in
`perf/glm53-indexer-loader-protocol.md`, under `perf/results/2026-09-10/indexer-aot-v1/`.
One fresh cache each; freeze from preparation through terminal/pair audit.
CPU8GiB/GPU16GiB, swap0, no retries/replacements/concurrent GPU work. If successful,
continue to opt-in serving integration and scheduler/capture/model-quality gates.
No production default/quant or speed baseline changes.

### Previous checkpoint (2026-09-10): indexer loader foundation CPU-tested

The opt-in `glm53_indexer_correction_loader.py` policy reuses the existing
extra-launch lifecycle and observed static CUDA loader. A strict bridge compiles
the unchanged qualified JIT and checks its key/signature/options and in-memory/
disk cubin before adapting to Torch's static launcher. It does not inject a saved
binary. KV remains the default policy; source/mode/adapter/event naming is now
explicitly extensible. New correction events and hook marker remain separate.

`BoundIndexerCorrection` supplies fixed guarded uint8 storage per target binding,
8192-row diagnostic envelope (~1 MiB/binding), then invokes the already-tested
adapter with a correctly strided view. No CUDA allocation or compilation in run.
The independent graph inspector checks actual direct dispatch, appended static
runner/image and arena geometry/device. No serving hook or profile default yet.

CPU358 tests pass/20.50s,14 upstream Torch warnings; initial131 pass/7.72s.
Legacy KV/geometry loader and serving tests pass after the shared refactor.
Native driver/allocation are simulated in CPU tests; this does NOT qualify the
new static dispatch on GPU. The prior kernel source/cubin stays unchanged.
CPU inspection joins all eight archived target graph uses, seven AOT roots and
46 entries per rank,269 receipts and all5,172 original files. Raw
`runtime-control/indexer-loader-inspection-v1.json` under2026-09-10, SHA
`89f90eb95530b179f4150a5c2dbd37b87f5839f5e3ce62f4c5d169ee79fc240b`.

NEXT: add runnable AOT series preparation/controller/auditor using this base
builder and the shared AOT lifecycle. Validate the real static-launcher ABI with
synthetic leaf calls through each actual graph-held binding, not merely loading
modules; preserve original and qualified corrected output hashes/replay/guards.
Then run a separately prescribed all-rank control/correction series before any
model integration. Serving must check actual scheduler/padding limits against
the arena; no silent truncation, growth during capture or assumed reentrancy.
No GPU/model job or source freeze is active/prescribed. Details:
`perf/glm53-indexer-loader-protocol.md`.

### Previous checkpoint (2026-09-10): selective indexer correction passes GPU probe

On `f5ad4f884`, the one prescribed GPU process completes all120 rank-matched
cases and both input phases. Corrected indexer max1 BF16 ULP meets the unchanged
oracle. Actual GPU/CPU detector flags agree; zero original failures missed.
Q/KV, unselected indexer values, packed gate, guards, eager repeats and changed/
restored graph replay all pass. Original input/output hashes reproduce exactly.
All four rank numerical records agree. Across ranks:5,720 selected elements,
5,692 selected rows,484 changed elements; per unique matrix1,430/1,423/121.

Terminal audit verifies236 frozen sources/receipts and5,172 original files.
Four correction cubins are byte-identical, SHA
`bd00effc35c7c0619ce74d3819aa3322cc110e423ce157ce855994f30188453e`.
GPUs released; identities/driver/600W settings unchanged. Source freeze ended
after terminal audit; no retry/replacement, model launch or native build.

Diagnostic median bundle timings, original -> original+correction:
rows1:1.282->1.978us;16:1.472->2.218;640:2.300->5.553;7616:13.387->17.831.
Selection-write overhead is included. This is a correctness candidate with a
measured extra-launch cost, NOT a speed win, serving result or baseline change.
The historical original oracle failure remains recorded; model quality has not
been tested. Production attention/profile/quant/defaults are unchanged.

Raw `perf/results/2026-09-10/indexer-correction-v1/analysis.json`, SHA
`c6ac3d0399af92be467ef47831f512c4edee60ff08fe0772afb399c59a18cc65`.
CPU95 pass/5.13s. Exact commands/limits: `perf/glm53-indexer-correction-protocol.md`.
Use this completed receipt after future edits; do not rerun its expired freeze.

NEXT: qualify an opt-in graph-loader/serving diagnostic using the tested kernel
and exact original bundle. Verify actual AOT binding and forward/capture coverage
before a prescribed control/correction/return model-quality comparison. The KV
result demonstrates that local ULP parity is not enough. Keep the instrumented
kernel unchanged for that qualification; do not silently remove its flag writes.
No next GPU/model job or source freeze is active/prescribed yet.

### Previous checkpoint (2026-09-10): isolated indexer correction CPU-qualified

Opt-in probe only: `benchmarks/kernels/glm53_indexer_correction.py` and
`check_glm53_indexer_correction.py`. Original compiled combo remains first;
one-warp correction uses actual BF16 output/bias at fixed2^-12, recomputes only
flagged rows in FP64 through affine, and stores only flagged indexer elements.
Guarded selection flags expose actual detector coverage. No serving hook/default.

CPU95 tests pass/5.13s (initial5.15s), including actual source preparation, GPU-inert imports,
ABI/layout/stream order, unchanged oracle and independent detector/neighbor gates.
GPU numerical/replay/guard/timing validation is still pending, not implied by CPU.

NEXT after commit: one preparation, one sequential all-rank GPU process, one
terminal audit as prescribed in `perf/glm53-indexer-correction-protocol.md`.
120 cases/two input phases; only after all pass, rank0 fixed3-repeat graph
timings for rows1/16/640/7616. CPU8GiB/GPU16GiB, swap0. Freeze sources from
preparation through audit; no retry/replacement or model launch. Original GPU
oracle remains historically failed; full model quality remains a separate gate.

### Previous checkpoint (2026-09-10): cancellation detector CPU screen complete

The three thresholds prescribed before this CPU screen (2^-16, 2^-12, 2^-8)
all detect the 12 original and 14 split retained GPU failing scalars, plus all
21/19 failures of the separate/fused-affine FP32 CPU models. The detector uses
only BF16 output/bias in FP32, not an oracle or high-precision input moments.
At 2^-16 both CPU models flag102/99,312 rows (0.103%); at 2^-12 they flag
1,424/1,423 (1.434%/1.433%); at 2^-8 they flag21,734 (21.885%). These are CPU
proxy counts, not measurements of full GPU outputs or execution cost.

All60 previous precision records, including every output hash and scalar,
reproduce exactly against the pinned completed analysis from `cc9a508ea`.
CPU73 tests pass/3.54s; Ruff/diff pass. No GPU/model job, kernel implementation,
production/default/quant change or quality/oracle promotion.
Raw `runtime-control/indexer-cancellation-analysis-v1.json` under 2026-09-10,
SHA `3474f6692e98b1be257174827dd5387d8d08dedadd647f8c2935a0a18ec25a24`.

NEXT: implement an isolated opt-in Triton correction probe at fixed threshold
2^-12 (16x the tightest screened threshold, still ~1.43% CPU rows). Run the
original attention bundle first, detect cancellation from its actual BF16
indexer output, recompute flagged rows' moments/normalization/affine in FP64,
and overwrite ONLY flagged indexer elements. Preserve Q/KV/gate/stride guards
and every unflagged element exactly. Qualify CPU/import/layout checks before
prescribing GPU starts, actual branch coverage, oracle/replay/guards and timing.
The threshold is heuristic, not a proven bound; do not loosen the one-ULP gate.
Model quality remains a later independent gate. Original production attention
arithmetic stays unchanged. No GPU/model job or source freeze is active yet.

### Previous checkpoint (2026-09-10): indexer CPU precision boundaries isolated

`benchmarks/analyze_glm53_indexer_precision.py` consumes the pinned completed
rank-private probe, not its expired source freeze. All four rank records match;
all 60 packed-input hashes, four real layer11 weight hashes and 26 retained
failing scalars reproduce. Six CPU arithmetic models cover 12,711,936 unique
indexer outputs. These are NOT GPU-source reproductions or serving measurements.

FP32 separate affine has 21 outputs above one BF16 ULP (max22); fused-affine
emulation/FP64 affine alone each leave 19 (max20). FP32 moments with an FP64
tail leave nine (max8). Even FP64 normalization rounded to FP32 before FP64
affine leaves eight (max5). Full FP64 is the exact reference control, not a
qualified candidate. Affine-only precision is insufficient on this matrix;
both moment precision and the normalized-value rounding boundary matter.

CPU tests: 70 passed/2.94s; Ruff/diff checks pass. No GPU/model job, native build,
production/default/quant change, tolerance relaxation or new speed baseline.
Raw `runtime-control/indexer-precision-analysis-v1.json` under 2026-09-10, SHA
`358f4a155dbeea926e4e6e3f7b192c8c9e3ded3a0f4ee5b857d1951af7502764`.
Details and command: final section of `perf/glm53-attention-isolation.md`.

NEXT: screen a cancellation-triggered extended-precision recomputation on CPU,
including detector coverage and fraction of affected rows, before prescribing a
GPU prototype. It must not round the normalized value back to FP32 ahead of
affine. This is a hypothesis, not approval to install a replacement: the original
GPU oracle remains failed; GPU numerical/replay/guard and full model quality
checks remain required. Preserve original production attention arithmetic.
The rejected KV-only result and startup-variability investigation remain open
context; no next GPU/model job or source freeze is active.

### Previous checkpoint (2026-09-10): KV-only model comparison complete; candidate rejected

On `c42b72325`, exactly control/KV/return-control completed, one private cache and
one start each. All serving/worker/workload audits and final closure passed. Both
controls reproduce historical text/needle vectors exactly in all three repeats.
KV repeats exactly but fails 15/32 unchanged quality windows in every repeat;
it does NOT reproduce the failed no-combo vector. All 92 non-target AOT bindings
match across arms; return-control restores every original binding. No production
default, quant, native binary, quality-floor or failed indexer-oracle change.

Diagnostic E2E medians c1/c8/c16: control 157.332/580.279/778.425 tok/s;
KV 156.585/577.794/778.805; return 156.728/578.758/778.591. No speed win or new
baseline. All 27 exact-token rounds, nine quality passes, text/image canaries and
cold 32K/128K tests retained. No model retry, replacement or omitted case.

Closure verifies 386 frozen receipts and 5,172 original files; GPUs released,
UUID/driver/600 W settings unchanged. Source freeze ended before notebook edits.
Raw `perf/results/2026-09-10/kv-serving-v1/closure.json`, SHA
`a4e2ac3b120e3493cf11586a5b258123ad257d1de2cf67aacec8c1ace4478588`.
Commands are historical now; consume completed receipts after later edits rather
than rerunning their frozen readers. CPU gate remains 628 pass/52.07 s.

NEXT: preserve original attention arithmetic for production tuning. KV alone has
a reproducible model-quality effect but is not the complete no-combo explanation;
do not combine isolated results as if their effects were additive. The independent
indexer LayerNorm128 oracle still fails from affine cancellation (documented
max 5/28 BF16 ULP original/split); investigate its precision with the retained
worst-element evidence before proposing another replacement. No further GPU/model
job is prescribed yet. Campaign goal and production startup-variability work remain
open. Full results and limitations: `perf/glm53-kv-serving-protocol.md`.

### Previous checkpoint (2026-09-10): KV serving integration CPU-qualified

The opt-in KV policy now shares the geometry serving lifecycle/workload/controller.
Actual root modules feed KV seal/verify; original and appended launch receipts are
checked before forward and across capture. Binary observation remains open for
later non-target compilation; KV controller events have their own stream. Only
exact previously bound graph callbacks may follow target sealing. Production
defaults, quant, quality floors and the failed indexer oracle are unchanged.

CPU 628 pass/52.07 s; Ruff/diff checks pass. Pinned completed AOT evidence and all
5,172 original cache files verify. No model process, GPU probe or build ran in this
step. Qualified loader/compiler/adapter/native sources remain exact; the offline
event auditor and closed-stage notebook changes are explicitly accounted for.

NEXT after committing: prepare and run exactly control/KV/return-control once
each, under `perf/results/2026-09-10/kv-serving-v1/`. Full unchanged quality and
cold-prefix/c1/c8/c16/32K/128K workload. CPU 8 GiB/serve 150 GiB, swap0; audit and
stop at first invalid process, no replacements. Freeze preparation through terminal
closure. Exact commands/gates: `perf/glm53-kv-serving-protocol.md`. No new TPS
baseline or model-causality result yet. Previous completed checkpoint follows.

### Previous checkpoint (2026-09-10): all-rank KV AOT qualification PASSES

On `7b753a65a`, v2 completes exactly eight prescribed GPU processes: control
ranks 0..3, then KV ranks 0..3. All loads/audits and the final comparison exit 0.
Each case binds seven actual roots/46 entries, 25 original launchers and two
target globals. All eight candidate target bindings prove both the original
combo and appended KV launch against qualified source/config/whole-cubin
receipts. All 92 non-target bindings across ranks match control exactly.

Final check: all 246 frozen receipts and 5,172 original files verify; GPUs
released, UUID/driver/600 W settings unchanged. Source freeze ended. No model
weights/forward/capture, retries within v2, native build, TPS/default/quant change
or quality-gate promotion. v1's failed load stays terminal and preserved.
CPU: 475 passed in 42.84 s. Raw
`perf/results/2026-09-10/kv-aot-qualification-v2/pair-analysis.json`, SHA
e650a2a5f806085ab6f748169b1d54cf8950a2fd7656dfeda11d5e6f25a018f7.
Consume these completed receipts after later edits, not their frozen validators.

NEXT: integrate the qualified KV loader with the existing opt-in serving
lifecycle and prepare a bounded original/KV/return-original real-profile series.
Keep the qualified KV loader/compiler/native sources exact; explicitly account
for serving integration and offline-auditor changes. The shared geometry serving
helper currently passes all PyCodeCache modules to controller seal/verify; KV
requires actual root modules. Its stable-binding comparison also needs to ignore
the appended launch's process-local observed index, not its source/config/cubin.
Reuse existing profile validation, quality workload and lifecycle snapshots;
verify qualified bindings before forward and across capture without globally
sealing legitimate non-target compilation. No next model job prescribed yet.
Indexer oracle remains failed; model causality and optimization goal stay open.

### Previous checkpoint (2026-09-10): v1 terminal; exact helper-boundary fix ready

On `30ea612a6`, v1 stops on the first control-rank0 load; load/audit exit 1,
seven other cases unattempted. Seven static bundles loaded with no fallback.
The traceback proves the hook treated the standalone combo benchmark helper's
`call` export as a completed AOT graph before Torch's synchronous precompile.

Controller now binds only registered AOT roots and skips only the exact
source/hash/kernel-verified standalone helper. Wrong non-root targets still fail.
CPU reproducer: 147 passed in 6.77 s; full suite: 475 in 42.84 s, 14 warnings.
v1 receipts are retained; final check verified 246 frozen receipts/5,172 cache
files, GPU release and unchanged hardware. Source freeze ended before edits.

NEXT: commit then run the separately prescribed v2 preparation/controller/compare
commands at the tail of `perf/glm53-attention-isolation.md`. Same fixed eight
cases/gates, new cache namespace, no retry within either series. Freeze through
closure. No model/weights/forward/capture/TPS/default/quant/gate change.
Raw v1 `control-rank0/run/analysis.json` SHA
fe7113bacf2c8ba8d83aa932426b17a6e607b85d5c5362adcc24903c6c651217;
CPU `runtime-control/kv-aot-{root-boundary,v2-final}-cpu.xml` under 2026-09-10.

### Previous checkpoint (2026-09-10): actual-AOT KV qualification v1 ready

Dedicated preparer/runner/offline auditor implemented, reusing shared AOT
lifecycle and graph/driver/root checks. Actual-cache CPU inspection verifies
all 5,172 original files, one target/two graph uses per rank, seven actual roots
and 46 entries per rank. Final CPU: 471 passed in 42.52 s, including real concurrent
store loading without forwards/launches and one-attempt failure handling.
Raw `runtime-control/kv-aot-source-inspection.json` and
`kv-aot-{hooks,audit,runner,final}-cpu.xml`, under 2026-09-10.

NEXT: after commit, execute the exact v1 preparation/controller/compare commands
at the end of `perf/glm53-attention-isolation.md`. Fixed order: control ranks 0..3,
then KV ranks 0..3, eight processes total; CPU 8 GiB/GPU 16 GiB, swap0. Preparation
requires a clean committed worktree and starts the source freeze. No retries,
model weights/forward/capture, speed claim or native build. Stop on first failure,
retain/audit it and confirm integrity/release before ending the freeze. Candidate
must preserve all non-target bindings; the 25 original launchers/two targets
per rank plus appended KV receipts are audited. Source-level adapter remains
the only GPU-qualified KV stage so far. Production/quant/indexer gates unchanged.

### Previous checkpoint (2026-09-10): KV graph-loader implementation CPU-tested

`benchmarks/kernels/glm53_kv_loader.py` now binds the actual autotuner instance
`run` directly: original precompiled combo for control; original combo followed
by the qualified KV adapter for candidate. Original compile-results/launcher
metadata stays unchanged, and the appended launch is audited separately from
the live adapter by `audit_glm53_kv_graphs.py`. Atomic future resolution, finished
module binding, private-cache provenance and late-binding seals are implemented.

Shared lifecycle moved to `glm53_loader_hooks.py`; graph-root walking is reused
from the geometry auditor. Future geometry preparation freezes the new helper.
Final CPU suite: 449 passed in 41.23 s (14 upstream deprecation warnings), including
real Torch future/code-cache/static-launcher APIs with a fake CUDA driver.
Reports: `perf/results/2026-09-10/runtime-control/kv-loader-{cpu,dispatch-cpu,
regression-cpu,freeze-cpu,final-cpu}.xml`, plus `loader-hooks-cpu.xml`.
Earlier test failures remain recorded. The historical geometry-source guard
correctly rejects this changed loader; its test now expects that rejection in
both the policy and direct client, without relaxing qualification.

NEXT: implement preparation and bounded no-weights actual-AOT qualification for
this separate KV manifest. Join the completed adapter/attention-map receipts,
copy private caches and KV sources, discover actual serialized roots, freeze all
new helpers, and prescribe/commit the exact command order before GPU loading.
Do not pass a KV manifest to the thirteen-target geometry preparer. Then qualify
all ranks and unchanged non-target bindings before original/KV/return serving.
No new GPU/model run or source freeze is active/prescribed. CPU tests do NOT
qualify actual cached graphs, model quality or TPS; production/quant/defaults
and the separate failed indexer oracle gate are unchanged.

### Previous checkpoint (2026-09-10): KV-only adapter isolated qualification PASSES

On `f6ff10453`, exactly one rank-sequential GPU process completes all 120 cases;
probe/audit exit0. Adapter KV matches direct split KV over 203,390,976 BF16 values,
with exactly 556 changes from original. Q/indexer remain exact over
610,172,928 / 50,847,744 values. Input/original/direct-KV hashes match the completed
historical matrix. KV <=1 BF16 ULP, eager/changed-input graph replay/guards pass.
The separate indexer LayerNorm oracle failure remains FAILED; no model-quality
or performance qualification follows from this adapter test.

All eight actual rank-private cubins, 202 frozen receipts and 5,172 original
cache files verify. GPUs released, hardware unchanged, source freeze ended.
No retries, omissions, autotuning, model starts or native builds. CPU 88 passed
in 3.84 s. Raw `perf/results/2026-09-10/kv-overwrite-v1/analysis.json`, SHA
bfaa7a495e7b69f228661a4402f5a6d5229d53b6950b6e7ee8cfa6275975144d.
Consume the completed receipt after further edits, not its frozen validator.

NEXT: implement actual graph-loader binding/coverage for this qualified adapter
using the existing binary observer and actual AOT-root inventory. Source-level
graph replay is not actual serving-loader qualification. Inspect both original
and appended launchers and all non-target bindings before capture; do not let
Inductor's cached fast launcher bypass the intervention. No next GPU/model job
prescribed. Plan: `perf/glm53-attention-isolation.md`; production defaults unchanged.

### Previous checkpoint (2026-09-10): attention graph boundaries mapped

CPU analyzer `benchmarks/analyze_glm53_attention_contracts.py` verifies all eight
attention graph pairs (two per rank), with 75 pinned graph/source/evidence hashes.
Original bundle -> three split calls preserves input/weights/bias/output/rows/
stream, allocation/alias provenance, native call sequences and graph returns.
The indexer norm writes half of a shared K/gate buffer; this alias is checked.
This is static correspondence, NOT historical live coverage or model causality.

CPU final 67 passed in 7.14 s; earlier 14-pass report retained. No GPU/model run,
native build, quant/profile/default change or quality-gate change. Raw
`perf/results/2026-09-10/runtime-control/attention-contracts.json`, SHA
8337ae7e4ec383563ed2df0230f21e9424556feccbec3a93a61deea94caff3c8.

NEXT: implement/qualify a KV512-only diagnostic adapter: run original combo,
then overwrite only KV with the previously checked split kernel on the same
input/weight/address/stream. Preserve Q, indexer K/gate, H4096 and KDA exactly.
Qualify actual adapter/graph replay before prescribing original/KV/return model
starts. This is an extra-launch causal diagnostic, not a speed implementation.
Plan/evidence limits: `perf/glm53-attention-isolation.md`. No GPU job prescribed;
all previous failed series and the separate LayerNorm oracle failure stay failed.

### Previous checkpoint (2026-09-10): state/output isolation complete and bit-exact

On `4832f4ce7`, the prescribed state process/audit then output process/audit all
exit 0. Each completes 224 pairs (56 exact, 168 conditioned). State stages2/3
are bit-exact over 2,650,800,128 BF16 snapshots, 1,285,685,248 BF16 new values and
167,772,160 FP32 final-state values. Output warps8/4 is bit-exact over another
1,285,685,248 BF16 values. All exact-oracle/eager/replay/mutation/guard checks pass.
State verifies 83 receipts/two cubins; output verifies 312/two, including its
complete state predecessor. GPUs released; hardware unchanged; source freeze
ended. No retries, exclusions, autotuning, model starts or native builds.

Conditioned float64-reference errors remain observations, NOT qualification.
State max absolute snapshot/new/final: 0.03125/0.03125/0.005167722702026367;
output: 0.12482273578643799. No accuracy threshold was introduced or widened.
No TPS, production/default/quant promotion. CPU 53 passed in 7.06 s.

Raw `perf/results/2026-09-10/kda-{state,output}-v1/analysis.json` SHA respectively:
f28c002eacc2dbe1ab664f8c1fbded2c4745bc4604a1f4c4e805acfe644bc49c;
2b5c863eb453d252b05461e6d4ed02e8f171714b4c1713b49efd4e97daab943b.
Consume completed receipts after later edits, not historical frozen validators.

NEXT: KDA's tested choices show no numerical difference; this is not a proof for
all activations or unrecorded historical launches. Focus the next causal design
on remaining attention combo/split normalization differences (KV512/LayerNorm128)
and actual workload boundaries. Q1536 was exact in the earlier probe. Keep the
separate LayerNorm oracle failure explicit; neither RMSNorm geometry nor KDA tests
clear it. No next GPU/model job prescribed. All old failed series remain terminal.

### Previous checkpoint (2026-09-10): recompute isolation complete and bit-exact

On3bd347098, one prescribed GPU0 process completes196 pairs (28 exact identity,
168 conditioned). Four/eight-warp W/U/KG are bit-exact:1,124,974,592 BF16 values
per output. Eager/replay/changed-input/mutation/guard and exact identity checks
pass; probe/audit exit0.81 frozen receipts/two cubins verify; GPUs released,
hardware settings unchanged. Freeze ended; no retries, exclusions or model run.

Conditioned FP64-reference errors were observations, NOT accuracy qualification.
Max W/U/KG BF16-ULP27616/67/49; absolute0.0011707544/0.1209889725/0.0009765638.
Do not infer general numerical accuracy or explain aggregate ULP maxima without
element-level evidence. Both arms match exactly; no warp-choice cause shown here.
No TPS, default, quant or quality-gate promotion.

Raw perf/results/2026-09-10/kda-recompute-v1/analysis.json SHA
eb997c612a0347924b5a66b49c11a90ed969ba37e10ae5d6ac8ce4bf494fd6a9;
all196 records, summary and cubins/private cache retained. Source-freeze readers
are historical after further edits; consume the pinned completion receipts.

NEXT: state update at warps4/BV32/stages2 vs3, then output warps4 vs8/BK64/BV64.
State has mixed BF16/FP32 outputs and actual varlen/initial-state/GK/exp2 flags;
see perf/glm53-kda-choice-protocol.md tail. No next GPU/model job prescribed.
All earlier failed series/gates remain terminal/failed. Serving defaults unchanged.

### Previous checkpoint (2026-09-10): recompute isolation ready, v1 prescribed

Retained state/recompute/output candidates have different binaries; CPU screen
records36 candidates/seven exact TTIR groups. Recompute4->8-warps is the earliest
remaining numerical test. Probe CPU45pass4.45s, including independent ragged
reference, real JIT import/preparation, exact identity cases and audit failures.
Shared BF16 comparison default stays128 rows; new GPU probe uses8192-row tiles.

NEXT after commit: one preparation/GPU0 run/CPU audit for196 fixed pairs at
16GiB/8GiB/swap0. Commands, cases, evidence limits and failure policy are at
perf/glm53-kda-choice-protocol.md tail. Freeze from preparation through closure.
No v1 manifest/cache/GPU job created yet; no full-model job prescribed.
All completed/failed prior runs remain terminal; defaults/native/quant unchanged.

### Previous checkpoint (2026-09-10): gate isolation passes; no numerical difference

On c7b4bb4d1, one prescribed GPU0 process completes168 gate/cumsum pairs; audit
passes,74 source/evidence hashes/two actual cubins verify, GPUs released. Freeze
ended. No retries, discarded cases, model starts, autotuning or native builds.
Eight/two-warps are bit-exact on964,263,936 FP32 gate and7,533,312 beta values.
Float64-reference, eager/changed-input graph replay and guard/mutation checks pass.
This is local source/config arithmetic evidence, not historical live-choice proof
or model causality. No speed measurement/default/quant/quality-gate promotion.

CPU follow-up also finds identical whole TTIR/PTX/cubin bytes for the retained
intra sub-chunk warps2/stages2,3,4 candidates. Skip a redundant GPU test of those
exact binary candidates; their historical per-rank launch identities are unrecorded.

Raw perf/results/2026-09-10/kda-gate-v1/analysis.json SHA
dbb92de6bc14f00c86ba8d19a8a680ab8d5ad82ed696906d300a3432e5d9bf70;
runtime-control/kda-intra-stage-screen.json and its preserved reader. Commands,
matrix, source boundaries and full result: perf/glm53-kda-choice-protocol.md.
Do not re-run frozen v1 validators after later source/protocol edits; use completed
receipts and relevant live-source checks. All old failed series remain terminal.

NEXT: inspect remaining KDA recompute W/U, recurrent-state and output choices;
screen same-signature binaries first, especially stage-only differences, then
prescribe the smallest informative numerical comparison. No next GPU/model job
prescribed. Original H4096 settings retained; indexer/no-combo gates still failed.

### Previous checkpoint (2026-09-10): remaining KDA choice inventory complete

Geometry rejection recorded in e427bd85b (Auroter author/committer). CPU inventory
now verifies72 retained KDA tuning files against24 original rank-private files.
Failed fresh shared cache differs in19/24 rank/kernel choices across five kernels.
Earliest is gate/cumsum: BS32/stages3 fixed, eight -> two warps on every rank.
Shared disk winners are NOT historical per-rank launch receipts; do not infer
model causality or call these timings serving TPS. CPU16pass0.09s, no GPU run.

NEXT: execute the prescribed one-process gate/cumsum v1 diagnostic after commit.
Probe/tests ready:22 CPU pass3.54s, including real source imports and preparation.
168 pairs,16GiB/swap0 GPU0, fixed eight/two-warps, no tuning; freeze from preparation
through8GiB CPU audit. Exact commands/matrix/failure policy are at protocol tail.
No model execution is prescribed. No caches or GPU job for v1 launched yet.
Actual CUDA path re-exports kimi_k3/amd/ops/third_party/kda; generic FLA kda.py
is not the serving implementation. Protocol/evidence boundaries and raw receipt
are in perf/glm53-kda-choice-protocol.md.
Quant/default/native libraries unchanged; every failed quality gate remains failed.

### Previous checkpoint (2026-09-10): full-model v3 complete; geometry candidate rejected

On `ce6df61aa`, exactly the prescribed control -> geometry -> return-control
starts completed, each followed by successful independent worker/workload audits
and GPU release. Final closure passes:359 frozen receipts/5,172 original files
unchanged; all65 non-target graph/source/config/cubin bindings match across starts,
and return-control target bindings match control. Source freeze has ended.
All nine serving caches from v1-v3 and all prior failed attempts remain intact.
V1/V2 stay terminal with zero model starts; v3 has three, none replaced/excluded.

All three starts reach health, pass text/image canaries, complete the exact-token
matrix and cold32K/128K prefill, and pass actual AOT/capture checks on all four ranks.
Every start's three quality passes repeat all4,264 text/needle scores exactly.
Both controls match the pinned original vectors exactly and pass all quality floors.
Geometry fails12/32 unchanged window floors in every pass despite a better aggregate
mean. It differs from the failed no-combo vector at ALL4,096 text and168 needle
scores (text RMS delta0.412236). Thus these thirteen geometry changes are NOT
sufficient to reproduce that candidate's score vector on this workload. Their own
score change is reversible. This does not show RMSNorm is irrelevant or establish
which remaining compiler/attention/KDA difference explains the rest.

Diagnostic E2E c1/c8/c16 medians: control157.422/580.938/780.713,
geometry156.682/576.893/775.075, return156.959/579.190/777.006 tok/s.
No speed win; no production/default/quant/quality-gate promotion. Original stable
baselines and the separate failed indexer LayerNorm oracle gate remain unchanged.

Raw `perf/results/2026-09-10/rmsnorm-geometry-serving-v3/closure.json`, SHA
c25674eab322b7334c694e1354f0a2ab3c8b7f3b8bb9be4e8d1e6f92091cf597.
Each case has `workload-analysis.json`, `worker-analysis.json`, `launch.json` and
complete response/kernel/cache evidence. Consume these completed pinned receipts;
do not rerun the HEAD/source-frozen readers after subsequent commits.

NEXT: retain the original H4096 choices; do not expand or promote the rejected
geometry/no-combo policy. Use completed graph/attention evidence to inventory the
remaining old/fresh KDA and attention differences, then prescribe the smallest
useful isolation or correctness repair. Indexer oracle failure is still open;
neither this result nor matching averages clears it. No next GPU/model job is
currently prescribed. The optimization goal remains ongoing, with no new speedup
retained at this checkpoint.

### Previous checkpoint (2026-09-10): v2 client import terminal; direct entrypoint fixed; v3 prescribed

V2 on `1bea1f616` passes preflight but its benchmark client fails before creating a
campaign or starting a server: direct-file execution exposes `benchmarks/`, not
its parent, so geometry validation cannot import `benchmarks.kernels`. Serve-child
and audit exit1; zero model starts. Closure succeeds as terminal-failure, verifies
359 source receipts/5,172 original files, unused cases unlaunched, GPUs released.
All six serving caches across v1/v2 remain intact; both series are terminal.
Raw `rmsnorm-geometry-serving-v2/closure.json`, SHA
2b8bae9d33bcbec77982ce723c1bafcdbfe9a481ff20116473ad4fe5e96bf070.

The direct-file benchmark entrypoint now adds its owned repository parent to the
import path. CPU175 pass46.50s, including fresh subprocess tests without inherited
PYTHONPATH: helper imports and the actual client/profile/real-manifest validation
path pass, deliberately stopping before tokenizer/server work. Hardware discovery
alone is replaced in the latter CPU test. No model/GPU execution or TPS claim.
Raw `runtime-control/geometry-client-entry-cpu.xml`.

NEXT after committing: NEW v3, same bounded three-case workload/failure policy at
protocol tail and fresh caches under `rmsnorm-geometry-serving-v3/`. Freeze through
closure. V3 not prepared/launched yet; do not reuse or launch unused v1/v2 cases.

### Previous checkpoint (2026-09-10): v1 preflight terminal; source aliases fixed; v2 prescribed

On `8074c504f`, v1 prepared all three copies but the first control stopped in CPU
preflight. No server command, model start, forward, capture or GPU load occurred.
The preparer resolved symlinks; the validator compared unresolved qualified keys.
Nine paths differed only in spelling (seven virtualenv sources, two model metadata
aliases). The prescribed closure hit the same reader guard and is retained failed.

Independent closure verifies all 359 current receipts, all 206 qualified receipts
by resolved path, 5,172 original files and all three private copies unchanged;
unused cases unlaunched and GPUs free. Raw
`runtime-control/geometry-serving-v1-preflight-closure.json`, SHA
c27e14186c71626478a923c9daa225cb0d4f53073f447acf5d02e5d26a2ecdba.
V1 is terminal, all copies/markers/failed closure retained, source freeze ended.

Validator now matches the preparer's canonical paths while rejecting conflicting
alias digests. CPU174 pass35.65s, including a real v1-metadata round-trip for all
three case modes against actual completed AOT receipts (new CPU fixture paths,
no private cache copies/model/GPU load). Report `runtime-control/geometry-source-alias-cpu.xml`.
No loader/compiler/kernel math/native/quant/quality/default change.

NEXT after commit: NEW v2, same fixed three-case workload/failure policy at protocol
tail, new copies under `rmsnorm-geometry-serving-v2/`. Freeze from preparation through
closure; do not retry v1 or launch its unused cases. V2 not prepared/launched yet.

### Previous checkpoint (2026-09-10): full-model geometry controller ready; v1 prescribed

Integration committed as `71cb900aa` (Auroter author/committer). The new workload
auditor reuses existing exact-token, cold-prefill and unchanged 32-window quality
checks. Whole historical response documents are pinned, not only summary means.
Controls must exactly reproduce original text/needle scores; all arms must repeat
exactly within start. Geometry quality failure remains a recorded failure of the
unchanged floors, but may proceed to the prescribed return control to test
reversibility. That is a diagnostic continuation, never production qualification.
Any structural/workload/needle/repetition/control/release failure stops the series.

Controller checks every prior audit and complete retained file inventory, source
freeze, fresh private cache, exact environment and established model path. Serving
uses 150 GiB/swap0; controller/audits use 8 GiB/swap0. No retries or replacement
starts. Final closure checks actual non-target bindings, return-control kernels,
quality vectors and cold-prefill input identities. No production/default changes.

CPU: combined 586 passed (44.24 s); final serving/workload/launcher 108 passed
(30.98 s). Real-response CPU rehearsal first caught an incorrect seed check;
the benchmark uses seed42+request-index, not seed42 for every request. Fixed
rehearsal passes and retains the historical 12-window failure. All reports retained
under `perf/results/2026-09-10/runtime-control/geometry-*cpu.xml`.
Registered-profile dry-run explicitly selects `/raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4`
with `SLIMSERVE_CACHE=/raid/weights`. GPUs free, driver580.173.02/600W unchanged.

NEXT after committing: freeze sources and execute the NEW v1 protocol at the tail
of `perf/glm53-rmsnorm-geometry-protocol.md`: one preparation, then one control,
one geometry, one return-control, conditional on all predecessor diagnostic gates.
No caches or model starts have yet occurred for v1. Close/audit even a terminal
failure before releasing the freeze. All old failed series stay terminal; v6
no-weights qualification stays complete. No new TPS or model-quality claim yet.

### Previous checkpoint (2026-09-10): opt-in geometry serving integration CPU-tested

The qualified thirteen-source intervention now has a separate, default-off serving
hook (`SLIMSERVE_GLM53_RMSNORM_GEOMETRY=control|geometry`) and manifest schema.
Actual AOT roots/launchers are checked and targets sealed before the loader returns
to model execution, then checked before and after capture. Binary observation stays
active for legitimate later non-target compilation. The legacy diagnostic is intact.
Preparation supports three fresh per-start caches: control, geometry, return-control.
No serving caches have been prepared and no model/GPU job has run at this checkpoint.

CPU regression: 511 passed (10.03 s), 8 GiB/swap0, GPUs hidden. Real vLLM concurrent
store/Torch graph APIs are exercised with reduced outer fixtures and mocked CUDA.
Read-only evidence join verifies 340 source/evidence receipts and 5,172 original
files; all 32 campaign source files are now covered by a shared catalog. Qualified
live loader/compiler/kernel/native sources remain unchanged. Default profile dry-run
passes; neither these tests nor the worker auditor establish serving quality or TPS.
Raw `runtime-control/geometry-serving-source-freeze-cpu.xml`,
`geometry-serving-final-evidence.json` (SHA
8a09c4ed1a9dab521f7eac7528f5af7a4b622958ed6755cf398ba7b18a1902fd), and
`geometry-serving-default-dry-run.log`, all under `perf/results/2026-09-10/`.

NEXT: implement/test the full workload controller and causal auditor, preserving
all 32 quality-window floors, exact-token/text/image/quality-repeat/cold-prefill
workloads, predecessor audits, source freeze and GPU-release gates. Confirm the
established `/raid/weights` model location explicitly (`SLIMSERVE_CACHE`); the
default dry-run resolves under `/home/tiny/models`, not the campaign path. Commit
and prescribe the exact three-case series before preparation or serving. No next
GPU/model job is prescribed. Failed no-combo/indexer gates remain unresolved;
quant, production defaults and stable performance baselines are unchanged.

### Previous checkpoint (2026-09-10): actual all-rank RMSNorm geometry AOT qualification PASSES

v6 on62a7a3ad0 completes ALL8 prescribed attempts in order: control ranks0..3,
then geometry ranks0..3. All loads and independent audits exit0, followed by a
successful cross-rank comparison and final source/cache/release closure. No retries
or exclusions. Seven actual artifacts/46 entries per process; seven model graph
roots and25 launcher bindings per rank. All13 target sources cover35 bindings/arm
(9/8/8/10 by rank); ALL65 non-target bindings exactly match control source/config/
cubin. This includes original attention combo and KDA selections. Each process
verifies2 direct root calls and5 exact Torch writeback closures.206 frozen receipts/
5172 original files unchanged; all GPUs released. Source freeze has ended.

Raw `perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v6/`:
pair-analysis SHA46b9c695c2734759deb55d3c15ee085dc8e3bd8eea67153ab074cdd1cb4fe946;
closure SHAff9b29e175d6d8ac2f1c787d6ca35ee522786e20541a4ac5f3ffdf6fdfd98242.
Consume these pinned COMPLETED receipts with source/binary verification; do not
rerun the source/HEAD-frozen read_manifest after later commits. All40 private caches
from v2-v6 and every failed attempt are retained. v1-v5 remain terminal.

Scope: real-AOT/binary/live-graph qualification ONLY. The completed numerical
qualification remains624 passing pairs, but no model weights/forwards/capture or
TPS were run in v6. Production defaults/quant/native/math/quality gates unchanged.
The failed no-combo model quality windows and separate indexer gate are unresolved.

NEXT: implement and CPU-test opt-in full-model integration of the qualified loader.
Use the real profile and shared per-start private namespace, rank-local Triton
caches, actual artifact-root checks before forward/capture, and target sealing
separate from the binary observer's lifetime (serving can compile non-target work
later). Freeze the integration and fresh serving manifests explicitly; only then
prescribe a control/geometry/return causal series with identical quality floors,
exact-token workloads and retained failures. No next GPU/model job is currently
prescribed. Existing single-source legacy diagnostic is a precedent, not this new
geometry serving hook; do not overwrite its flag/schema. Goal remains ongoing.

### Previous checkpoint (2026-09-10): v5 terminal; exact writeback closure supported; v6 prescribed (now complete)

v5 on838689c68 stops after ONE control-rank0. Seven artifact/root bindings verify
during deserialization; all7 artifacts/46 entries load,26 CUDA images and17 imported
module sources match originals. Post-load exact-call identity gate fails.205 frozen
sources/5172 original files verify, GPUs released, other seven never launched.
Closure `rmsnorm-geometry-aot-qualification-v5/closure.json`, SHA
45c52321b9d81db6b151344f67a9a4d17c429be404daef8cb14c277c9541b09a.
v1-v5 terminal; all32 private caches from v2-v5 retained; freeze ended.

CPU replay of actual Torch post_compile on ALL28 serialized graphs, with inert
sentinel calls/no GPU, creates20 alignment/writeback wrappers and keeps8 direct
calls. Each wrapper closes over the original call and exact mutated input[4].
`runtime-control/geometry-post-compile-cpu.json`, SHA
5f358d74cf27baa047a585cfff440af2543f8142e2aee1680a29624f0f264fa6.
This explains a legitimate identity transition, but v5 did not retain the final
call's closure, so the next live gate must prove that exact transition.

Observer now records post-load state BEFORE rejection and accepts ONLY the
installed writeback code/globals/source plus exact original call, mutation object,
indices and serialized alignment plan. Direct-call roots still require identity;
unknown/nested/mutated wrappers reject. Related CPU324 pass7.23s,14 new negative
closure/receipt tests; `runtime-control/geometry-writeback-cpu.xml`. No kernel math,
serving default, quant or quality changes. NEXT after commit: NEW v6 same eight
conditional no-weights loads at protocol tail, fresh copies, frozen through closure,
16GiB GPU/8GiB CPU/swap0. Stop on failure; no model series or performance claim.

### Previous checkpoint (2026-09-10): artifact-root provenance CPU gate passed; v5 prescribed (now stopped)

All28 serialized model roots (seven/rank) have exact original source bytes/keys;
each rank's46 submodule entries map to its seven roots. CPU inspection only, no
post-compile/forward/GPU load. `runtime-control/geometry-artifact-roots-all-ranks.json`,
SHA33db464e59736405d7893e3c45ffb152403c2be16599b5db55ca56b4ae8ba10a.

New diagnostic `glm53_artifact_roots.py` observes each actual AOT deserialize,
its live CompiledFxGraph.after_deserialization call/runner/module binding, and
the identical artifact retained by vLLM. Complete module paths/hashes are saved
BEFORE validation, also on failure. Imported helpers must be original-exact and
cannot substitute for roots. Every target/non-target root launcher remains audited;
controller graph callbacks remain a separate cross-check. Preparation freezes the
new tool and actual Torch source helpers, plus serialized-root receipts.

Related CPU310 pass6.94s, including real concurrent vLLM load_all and actual
CompiledFxGraph/PyCodeCache imports (outer test AOT wrapper reduced, no GPU calls).
Initial104-pass/3-failure fixture cleanup and subsequent passes are retained.
Raw `runtime-control/geometry-artifact-root-{binding,binding-final,real-store}-cpu.xml`.

NEXT after commit: NEW v5 at protocol tail, control0..3 then geometry0..3,
one no-weights load per fresh private cache,16GiB GPU/8GiB CPU/swap0. Freeze
sources through loads/audits/closure; stop on any failure, no retries. v1-v4 and
their unused cases remain terminal. No model series, TPS or quality/default/quant
promotion; original numerical qualification and separate failed indexer gate stand.

### Previous checkpoint (2026-09-10): v4 terminal; artifact-root provenance repair

v4 TERMINAL on5213afeb5 after ONE control-rank0 load/audit. All7 artifacts/46
entries load and27 observed CUDA images match original whole bytes. Independent
inventory fails with `unexpected or changed graph source`. Every private generated
Python source matches the original snapshot: the inventory is including an extra
imported call-export module, not detecting source drift. Its path was not logged;
do not claim the unexpected module's identity is proven. All201 source receipts/
5172 original files verify, GPUs released. Other seven cases never launched.
Closure `rmsnorm-geometry-aot-qualification-v4/closure.json`, SHA
934fe0e8b53960e38c8270ffadf86990a77559bc642a53dc832a0ca8e559a428.
v1-v4 remain stopped; all24 private caches from v2-v4 are preserved. Freeze ended.

CPU source catalog finds76 call-export modules (19/rank), including generated
kernel benchmark helpers; only28 are mapped model roots. CPU-only trusted rank0
artifact inspection recovers the seven expected root keys from compiled_fw.result.
NEXT: verify serialized root source bytes on all four ranks, observe the actual
CompiledFxGraph.after_deserialization binding to current_callable/runner, and
retain a complete module inventory BEFORE validation. Audit all target/non-target
root globals; do not simply allow every cached callable. CPU tests and a new
committed protocol are required before any new GPU series. No model/forward/TPS,
quality/default/quant promotion or next GPU run currently prescribed.
Raw `runtime-control/geometry-{all-export-source-inventory,artifact-root-inspection}.json`.

### Previous checkpoint (2026-09-10): v3 loads seven artifacts, parser fixed; v4 prescribed (now stopped)

v3 TERMINAL: ONE control-rank0 on14ab95575 loads7 artifacts/46 entries and26
CUDA images match original whole bytes. Its independent AST inventory rejects
the bound `Runner.call` export (incorrectly expected top-level def). Three source
controller callback counts3/4/2 are NOT independent graph proof. Load/audit exit1,
other seven never launched; all201 source receipts/5172 original files unchanged,
GPUs free. All private copies and logs retained. Closure
`rmsnorm-geometry-aot-qualification-v3/closure.json`, SHA
352a2e15d3a2bae0bc67b1cfdfb0de4aaf6017c372ca4eaa15bbadee9f1ef6d6.

Auditor now resolves explicit exported instance/class/method AST bindings and
checks live function, instance, class, source line/filename and globals. All28
original graph sources match exported Runner.call and their recorded run symbols.
`runtime-control/geometry-bound-export-source-check.json`, SHA
c43d449aa00bc4ffffc0f628efd291a16d199f54c69df7770c971121b457da36.
Focused79 pass2.87s; final related278 pass3.47s; nine new actual bound-method tests.

NEXT after commit: NEW v4 protocol at tail. Same control0..3 then geometry0..3
no-weights loads, new copies under
`perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v4/`.
Freeze through preparation/loads/audits/closure,16GiB GPU/8GiB CPU/swap0, one GPU
workload, stop on any failure/no retries. v1/v2/v3 unused cases remain stopped.
No full-model series or quality/default/quant promotion; goal ongoing.

### Previous checkpoint (2026-09-10): v2 first AOT load stopped; exact cache lifecycle fixed; v3 prescribed

v2 TERMINAL: ONE control-rank0 load onad6656314 and its audit exit1; remaining
seven not launched. All7 static bundles/50 entries found without fallback and
15 actual CUDA loads match original whole bytes. Adapter rejects the legitimate
autotuner-created TRITON_CACHE_DIR before replacement compilation. No full graph
coverage, model weights/forward or TPS.201 source receipts/5172 original files
unchanged, GPUs free. All8 private caches/logs/receipts preserved. Closure
`rmsnorm-geometry-aot-qualification-v2/closure.json`, SHA
c84e1f76dcca924103b52e660b30130e120c2f22b68a89800219024a819ae044.

Fixed adapter accepts unset OR exact canonical private rank directory, before/
after template creation and compilation, without rewriting callback environment.
New CPU test executes installed CachingAutotuner constructor's actual unset->exact
path transition; no GPU compilation. Wrong-rank/shared/empty/alias still reject.
Error now records actual/expected paths (v2 did not log actual env values).
Focused70 pass2.83s; final related269 pass.

NEXT after commit: NEW v3 protocol at tail. Same eight no-weights AOT cases,
control0..3 then geometry0..3, in fresh private copies. Root
`perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v3/`.
Freeze through preparation/loads/audits/closure,16GiB GPU/8GiB CPU/swap0; stop on
any failure/no retries/other GPU work/builds. v1/v2 remain stopped; no model series
prescribed, quant/default/quality unchanged, optimization goal ongoing.

### Previous checkpoint (2026-09-10): AOT preparation API fixed; NEW v2 prescribed

v1 is TERMINAL: its ONE CPU preparation on7e2b8bd0e exits1 BEFORE creating any
private cache or GPU process. Torch exports standalone_compile as a function,
so dotted-import source freezing tried to access function.__file__. All5172
original files/183 source receipts verify; GPUs free. Closure
`runtime-control/geometry-aot-v1-preparation-failure.json`, SHA
1a337eb9639c4152ef2a6b02c875b97f5d512468eadd9d852dbeed36fc1f2e23.

Preparer now resolves real module objects with importlib. Added full CPU test of
all eight private copies plus frozen manifest readback, actual Torch modules,
unchanged originals and no overwrite. Focused65 pass2.75s; related264 pass.

NEXT after commit: NEW v2 commands at protocol tail. Same exactly eight no-weights
loads, control-rank0..3 then geometry-rank0..3, conditional on prior audits/release,
16GiB GPU/8GiB CPU/swap0. Root
`perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v2/`.
Freeze through preparation/loads/audits/closure; no retries/builds/other GPU work.
Do not launch v1. No model series, TPS/default/quality promotion; goal ongoing.

### Previous checkpoint (2026-09-10): no-weights AOT runner/auditor ready; bounded v1 prescribed

`check_glm53_geometry_loader.py` loads the seven actual artifacts/46 submodule
entries per rank without weights/forward/capture. Its `launch` action preserves
native logs/statuses and executes one16GiB/swap0 GPU child plus8GiB/swap0 audit.
`audit_glm53_geometry_loader.py` independently joins graph globals, observed binary
loads and per-source controller receipts; geometry requires all non-target bindings
exact against its control. Each next attempt requires ALL prior load/audit/release
checks and unchanged receipts/logs. No retries; source/original freeze verified.

CPU gate263 passed3.13s, including31 new audit/harness tests (first focused64 pass).
Preparation now freezes these tools, actual Torch AOT/async/static/bundle/cache
helpers and owned `vllm/compilation/caching.py`. Sources ready to commit/freeze.

NEXT after commit: execute the new v1 protocol at the TAIL of
`perf/glm53-rmsnorm-geometry-protocol.md`: prepare eight private copies, then
control-rank0..3 followed by geometry-rank0..3, sequentially, only after each prior
load/audit passes and GPU is released; final pair audit. STOP on any failure and
preserve all evidence. No edits/builds/commits/other GPU jobs through closure.
Root `perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v1/`.
No full-model series prescribed; previous series remain terminal. No new TPS,
quant/default/native/quality change; optimization goal still ongoing.

### Previous checkpoint (2026-09-10): actual loader adapter and independent graph inventory CPU-tested

New diagnostic-only modules, not installed in serving:

- `prepare_glm53_geometry_loader.py` joins the pinned completed A/B receipts to
  original sources/debug identities and both exact rank-local binary images.
  Implements eight independent private cache/manifest copies (two modes x four
  ranks); NO copies have been prepared yet. It records released helper changes
  explicitly and refuses to refresh changed serving/compiler/native artifacts.
- `glm53_geometry_loader.py` scopes the pre-load observer BEFORE the actual
  StaticAutotunerFuture/PyCodeCache hooks. Compiles with original debug provenance
  and unchanged per-device Triton cache semantics. Restores hooks safely.
- `audit_glm53_geometry_graphs.py` independently reads actual call globals,
  executable AST run symbols, source/config/binary/launcher-object bindings, and
  all non-target Triton globals for cross-arm comparison. It does not derive
  coverage from controller callback receipts or static references alone.

Related CPU suite232 passed2.80s (including33 new tests),8GiB/swap0. Exercises
installed PyCodeCache loading, StaticAutotunerFuture.result, static make_launcher/
load_kernel and four concurrent real imports; GPU driver calls are mocked and
replacement GPU compilation is separately mocked for its provenance/cache test.
Target sealing leaves the observer open for legitimate non-target compilation;
global observer sealing remains an explicit final qualification step.

Read-only actual-artifact join passes:13 targets,7 graphs/rank,183 source receipts,
5172 unchanged original files. The ONLY refreshed former helper is the already
committed MultiIntervention observer integration. Raw
`runtime-control/geometry-loader-source-check.json`, SHA
cbe7547de5f0218caa4e5e094a1a1f74b2b17e00a0087c83a8eb5f7b78cb4d84.
CPU reports `geometry-loader-{cpu,cpu-api,expanded-cpu,related-cpu}.xml` retain
initial fixture/API failure and erroneous test-path attempt as well as passes.

NEXT: implement the bounded no-weights AOT runner and OFFLINE receipt auditor
around this adapter/inventory; include these new sources and actual Torch loader
helpers in the new manifest freeze. Then prepare the eight private copies and
prescribe/control their sequential real GPU loads (seven artifacts/46 entries per
rank), comparing all non-target bindings and independently joining binary/controller
receipts. No GPU/model job is yet prescribed. Independent GPU query empty; no
native build, model load, throughput result, serving/default/quant/gate change.
The model correctness regression and optimization goal remain unresolved.

### Previous checkpoint (2026-09-10): static load observer integrated and CPU-tested

`benchmarks/kernels/glm53_binary_observer.py` observes the exact private cubin
path passed to the CUDA driver BEFORE load, compares retained raw bytes when
present, and retains strong object/metadata/module/function provenance after
Torch consumes them. Each static graph launcher's runner must reference that
same observed kernel. Wrong paths/keys/images/ranks, unobserved loaded objects,
changed handles, closed objects, and mismatched launcher bindings fail closed.

MultiIntervention now accepts this observer for loaded-static binary checks;
historical single-source controller unchanged. CPU tests use installed
StaticTritonCompileResult.make_launcher and StaticallyLaunchedCudaKernel.load_kernel,
with only driver calls mocked. Include generated-launcher execution, concurrent
aliases (one driver load), three-source control-to-geometry graph replacement,
seal/reuse, failed-load cleanup and preserving a changed foreign hook. Final
related suite254 passed5.75s, lint/diff pass. No actual AOT/GPU/model validation
of the new observer yet; nothing installed in serving.

Read-only check confirms ALL13 qualified candidate images already exist at their
exact original rank-local keys with identical whole bytes. No binary seeding or
debug normalization needed. `runtime-control/rmsnorm-geometry-candidate-cache-check.json`,
SHA017120c13bc5fc5b8261903cf8a4dd3ba782b67c00ccfe53a9f58cfcd40a7102.
CPU reports `rmsnorm-binary-observer-{cpu,final-cpu}.xml` under the same directory.

NEXT: prepare source-bound private manifests using the completed pair's actual
control/geometry keys and bytes; implement the observer/controller's actual AOT
loader qualification, reusing `check_glm53_aot_loader.py` (seven artifacts/46
submodule entries per rank, no weights/forward). Independently enumerate all
actual graph globals;35 static source/graph uses are NOT already live proof.
Record a bounded protocol before GPU work. No next GPU/model job prescribed.
Default/profile/quant/native/quality gates unchanged; goal ongoing.

### Previous checkpoint (2026-09-10): both source-exact geometry processes and audits PASS

Prescribed A and B on4b0fa3701 each complete312 pairs and exit0. All26 source/
config binaries verify before numerics in each process; every control reproduces
the original whole-cubin image. Both arms meet FP64<=1 BF16 ULP in ALL624 pairs;
eager repeat, changed-input replay, guards, read-only mutation and triple-output
checks pass. All312 corresponding inputs/outputs/metrics/binaries match EXACTLY
across the independent fresh-cache processes.183 receipts/5172 original files
unchanged; GPUs free. Pair/audits terminal, source freeze released.

Geometry vs control differs in14,820/5,286,248,448 paired elements per process;
these are bounded kernel-level arithmetic differences, NOT full-model causality
or a speed win. Real weights at the three prescribed sites, synthetic activations.
Default/quant/native/attention/KDA settings and indexer gate unchanged.
Raw `perf/results/2026-09-10/rmsnorm-geometry-preload-qualification/`;
pair-analysis SHA3b5bcff2c1d23b4b8a2ffa7c648267d5ee74003f67f6ce6495528d1c69516f69.

NEXT: implement/test a pre-load binary observer for already-loaded static CUDA
objects, integrate it into the multi-source controller, then qualify actual AOT
graph bindings before a model causal series. The source-only probe does NOT
qualify that controller. No next GPU/model job prescribed; goal ongoing.

### Previous checkpoint (2026-09-10): static binary API failure closed; pre-load observer corrected

First geometry pair is TERMINAL: ONE A on28842e5c3 stops before numerical work.
Torch's static CUDA adapter exposes cubin_raw, not asm, and clears the raw bytes
inside make_launcher/load_kernel. The observer was wrong; no kernel-quality or
TPS result. First control's disk key/config/whole bytes match original.182 frozen
receipts/5172 original files unchanged, GPU query empty; B never launched.
Closure `runtime-control/rmsnorm-geometry-binary-api-failure-analysis.json`, SHA
ae19129d827af87f55352f6c7b625513c8db81ab2a0bb84e2616486ce124d07a.

Probe now hashes real in-memory bytes BEFORE make_launcher, then compares disk
bytes. No post-load disk fallback or weakened gate. CPU test uses the installed
static adapter's actual load lifecycle with only the driver call mocked.
226 related tests pass5.67s. New discovery
`runtime-control/rmsnorm-geometry-preload-discovery.json`, SHA
d98fffd11b5f0f60880a3754ea6641201435aa9779bcfeb1bdb4ff95bd0d3a5f.

NEXT after commit: NEW preload manifest/A/audit/B/audit/pair-audit at protocol
tail, same312 pairs/process/gates. B only after A passes. Freeze sources through
pair/audits; old paths/unused B remain terminal.16GiB probes/8GiB CPU/swap0.
The multi-source CONTROLLER still needs pre-load observation for graph-held
static objects during its future real-AOT qualification; it is NOT serving-ready.
No model job/default/quality promotion; goal ongoing.

### Previous checkpoint (2026-09-10): source-exact geometry probe/auditor implemented

`benchmarks/kernels/check_glm53_rmsnorm_geometry.py` implements preparation,
source-exact run, offline per-process audit and independent-process comparison.
All26 original-source/config bindings compile and verify BEFORE numerical work;
control requires original whole-cubin bytes. Matrix312 pairs/process retains
all numerical failures, with the unchanged FP64<=1 BF16 ULP gate. Rank-private
caches and original debug identities; no timed autotuning. B is gated on A's
successful audit AND unchanged source/binary artifacts. Serving code unchanged.

Final related CPU suite224 passed5.72s; lint/diff pass. Report
`runtime-control/rmsnorm-geometry-probe-final-cpu.xml` under2026-09-10.

NEXT after commit: prepare ONE new manifest, then the prescribed
A/audit/B/audit/pair-audit in `perf/glm53-rmsnorm-geometry-protocol.md`. B only if
A passes; never retry either arm. Freeze through the pair/audits,16GiB probes/
8GiB audits/swap0, one GPU workload at a time. No model or real-AOT multi-target
loader job prescribed yet. Actual geometry candidate binaries remain unqualified
until this pair passes. All older stopped series stay terminal; goal ongoing.

### Previous checkpoint (2026-09-10): multi-source mechanics and exact discovery ready

CPU-only multi-source diagnostic implemented under `benchmarks/kernels/`, NOT
wired into serving. One atomic resolver covers concurrent aliases without repeated
upstream cache rechecks; every target needs exact source/config/cubin bytes and
actual graph coverage before sealing. No late targets/aliases or source-path
changes allowed. Historical single-source controller and serving code unchanged.
Final related CPU suite187 passed5.43s; lint/diff pass.

Final discovery verifies13 original sources/control binaries/debug identities,
six in-place/seven triple-output layouts,35 STATIC graph/source uses,178 receipts
and5172 unchanged original files. `runtime-control/rmsnorm-geometry-final-discovery.json`
SHA4294ff75ea2236c577b8104ea3a44055d65dcf68ca0449685eccf02a2c5f7ea4.
Initial discovery and both CPU reports retained. This is NOT a qualified
intervention manifest: candidate geometry binaries still need measurement.

NEXT: implement the source-exact numerical probe/auditor described in
`perf/glm53-rmsnorm-geometry-protocol.md`, then freeze commands before GPU work.
Follow with real-AOT multi-target loader qualification; only then prescribe a
full-model causal series. No next GPU job is prescribed at this checkpoint.
No model/GPU workload launched in this turn, no new TPS/default/quality claim.
The original-cache attention combo and KDA choices must stay fixed; indexer
gate remains failed, prior series terminal, goal ongoing.

### Previous checkpoint (2026-09-10): broader RMSNorm graph correspondence established

CPU-only analysis pairs28 old/new generated graphs (seven per rank), using the
actual AST rather than duplicated compile-time docstrings. Correspondence requires
identical kernel bodies, semantic consumers and ordered call arguments. It identifies
13 distinct4096-wide sources changing from XBLOCK1/R0_BLOCK4096/16 warps to
XBLOCK1/R0_BLOCK1024/eight warps (one stage): six input_layernorm and seven
post_attention_layernorm sources,3/3/3/4 across ranks. Final mean+norm retains
1024/eight-warps. Old graph references remain STATIC cache evidence, not proof
every old graph was loaded; new paths are restricted to completed live receipts.

Analyzer/tests: `benchmarks/analyze_glm53_norm_graph_roles.py`, four CPU tests pass,
lint/diff pass. Raw `perf/results/2026-09-10/runtime-control/norm-graph-role-pairs.json`,
SHA320a6f39c4f92cd78459f21e23653e1e900870620da29bed923f5cba5dbe315e.
Initial unpaired inventory and logs retained. No GPU workload, new TPS result,
production change or quality-gate change in this step.

NEXT: prepare and qualify a graph-complete multi-source diagnostic to change ONLY
these13 original-cache4096 launch choices, keeping attention combo and KDA choices
fixed. The existing single-target-per-rank controller is not sufficient as-is.
Require one atomic future resolver, per-source coverage and sealing, source/config/
binary qualification before any model run. No new GPU job or model series is
prescribed yet. All prior attempts remain terminal; indexer gate remains failed.

### Previous checkpoint (2026-09-10): full attention matrix audited; RMS shapes pass, LayerNorm does not

ONE rank-private probe on7c5a8c603 completes ALL120 predetermined pairs, after
all16 exact source/config/cubin images verify. All4 ranks' outputs/metrics agree
exactly, as do all30 prior rank0 records. All eager/replay/guard/mutation checks
pass; source/native/compiler135 receipts and5172 original files unchanged.
GPUs are free. This completes the prescribed probe/audits; NEVER rerun any of
the four historical attempts or the stopped model series. No next GPU job is
prescribed. Goal remains ongoing, production/default/TC/quant unchanged.

Synthetic qualification on real layer11 weights:

- KV512 and Q1536 both meet the <=1 BF16 ULP FP64 gate, both arms/all ranks.
- Q1536 combo/split bit-exact over610,172,928 paired elements. KV512 differs in
  556/203,390,976 elements, all within the oracle gate.
- Indexer LayerNorm128 differs in456/50,847,744 paired elements and FAILS its
  unchanged gate: max5 ULP historical combo/28 split.20/120 pairs fail, the same
  five large-row input cases per rank, in BOTH arms. Whole matrix NOT qualified.

Worst scalar has confirmed near-zero cancellation: weighted term
0.5742191482235658 plus bias-0.57421875 gives FP64 3.982235657895572e-7.
Split emits4.507601261138916e-7, absolute FP64 error5.25365603243344e-8.
This explains the large ULP count locally, NOT the full-model score-vector
change; do not widen a tolerance or silently clear the indexer gate.

Raw `perf/results/2026-09-10/attention-norm-rank-private-probe/`;
summary SHA64c60102d391baffa7a4391257a9b742ed6932e6aea63cdb8cae77a478880c78,
tracked auditor's analysis SHA0da742b0b8d7eb08651be0b32fff0b7c874aaa87781d01d47ce2ca3877e06bdb.
Supplement `runtime-control/attention-rank-private-supplement.json`, SHA
ca1f301a402d655b0e42db598437615864444b927ac723df4cb7db3f5d283f85,
verifies prior-rank0/cross-rank equality, scalar cancellation, and GPU release.

NEXT: use the recorded4096 body/config inventory to map actual old/new graph
roles, then isolate those broader reduction choices separately from attention
combo separation/indexer rounding and fresh KDA autotuning. Numerical sensitivity
does not establish causality. Q1536 equality here is synthetic, not a blanket
exoneration on model activations. Existing old/native four-RMSNorm sufficiency
still applies only to that historical pair. Before any new GPU/model job,
record a bounded protocol; preserve quality windows, TC0, and selected recipe.

### Previous checkpoint (2026-09-10): rank0 numerical result; rank-private probe prepared

ONE4eb7b664a process completes30 rank0 pairs before stopping at rank1's combo
byte gate. Rank0 RMS512/1536 both <=1 BF16 ULP vs FP64; Q1536 combo/split exactly
equal in all60 original/changed outputs. KV512 differs in139 elements total.
Indexer LayerNorm128 has114 pairwise differences; oracle max5 ULP(combo)/28(split),
failing5/30 pairs in BOTH arms. This is partial synthetic evidence, not model
causality or all-rank qualification. No numerical tolerance changed.

Rank1 failure: four historical combo cubins share ONE semantic key but have FOUR
debug-image hashes in their original rank-local caches. The probe's shared cache
returns rank0's already-compiled image. Closure verifies all41 ELF sections;
only.debug_line/.nv.merc.debug_line differ. All non-debug sections and metadata
match.135 frozen receipts/5172 original files unchanged; GPUs free. Partial audit:
`runtime-control/attention-first-writer-failure-analysis.json`, SHA
e7977af07d26066449b1b874a04a31806a8bd4252a4416b92ec8df823b1b55c3.

Fix is probe-only rank-private Triton caches via its scoped cache API, tested
with real CPU cache managers. Precompile/verify ALL16 binaries before numerical
work. Same120-pair matrix and gates; retain bounded worst-element values for
failed ULP checks to investigate LayerNorm cancellation without guessing.
NEXT after tests/commit: NEW attention-rank-private-manifest.json and ONE NEW
attention-norm-rank-private-probe, then audit including exact agreement with the
previous30 rank0 results (old fields). Protocol tail gives scope/path rules.
All previous attempts remain terminal. No model start or later GPU job prescribed.

### Previous checkpoint (2026-09-10): first-writer provenance verified for all16 kernels

Second probe on9de50bbe3 also stops BEFORE numerical launches. Historical combo
now reproduces whole cubin bytes. First split kernel shares its cache key with
rank1, whose filename remains in the original binary's debug data. Using rank0's
equivalent source as its debug identity is insufficient. Frozen closure compares
all38 ELF sections: only.debug_line and.nv.merc.debug_line differ; all non-debug
sections and compiler metadata match. PTX instruction prefix matches after
ignoring filename COMMENTS (initial literal-prefix assertion/log retained).
121 source/native/compiler receipts and5172 original files unchanged; GPUs free.
Closure `runtime-control/attention-provenance-failure-analysis.json`, SHA
797a63dd7e137bd3a63dc55be1262fa849be1c1e0d622d896119587c55b6b018.

CPU preflight now resolves the exact PTX .file1 first-writer identity for ALL16
sources, bounds it to the recorded cache, and checks identical function text AND
line position. Nine of12 split bindings use rank1's debug identity; original
rank-specific source/decorator metadata remains unchanged. Private cache paths
and exact whole-cubin gates remain mandatory. No source/binary substitutions.
Discovery manifest `runtime-control/attention-first-writer-discovery.json`, SHA
49ba4df5a3a83bcfcea30383ffcb9683b14a77bbb05f0cbf15dd4704badd9e07.

NEXT after commit: NEW attention-first-writer-manifest.json, ONE NEW
attention-norm-first-writer-probe, same120-pair matrix/gates; protocol tail.
Both earlier kernel attempts and all model series remain terminal. No model
start or subsequent GPU work prescribed. Freeze through audit; goal ongoing.

### Previous checkpoint (2026-09-10): preserve source provenance for exact binary replay

ONE attention-norm probe on14ec731c8 stopped before any numerical launch:
the first combo kernel has the correct cache key/config but different cubin bytes.
Frozen closure verifies all41 ELF sections: ONLY six debug/debug-relocation
sections differ. Every non-debug section (including executable code, constants,
resource metadata and relocations), compiler JSON, and PTX code prefix matches.
The cause is copying Python to a different filename, embedded in debug data.
The strict whole-binary gate correctly stopped the process; zero quality/TPS
results. GPU query empty. Preserve the failed probe/manifest/audit and closure:
`runtime-control/attention-norm-byte-failure-analysis.json`, SHA
07856ad2966f91ea6ed33a3380c388ce48128c1b8c2f906413e3c6aee362abb5.

Probe-only fix executes identical copied source with its ORIGINAL code filename
while keeping module.__file__/decorator/cache paths PRIVATE. No cubin seeding,
debug stripping, hash-gate weakening, or original-cache writes. CPU tests cover
both identities and reject changed/non-private copies. Serving code unchanged.

NEXT: after commit prepare NEW attention-provenance-manifest.json, then ONE
NEW attention-norm-provenance-probe, same120-pair matrix and unchanged gates.
Protocol tail gives paths/commands. Freeze through audit; no model starts or
subsequent GPU work prescribed. The initial kernel attempt is terminal.

### Previous checkpoint (2026-09-10): source-exact attention comparison prepared

The full-model no-combo series below remains STOPPED. New work is an isolated
kernel diagnostic, not another serving start. Logical Q/KV widths are1536/512;
the earlier2048 label confused a rounded compiler hint with the Q extent.
Both historical combo and new split kernels retain FP32 arithmetic until final
BF16 stores. The inspected4096 bodies do too; no new intermediate-rounding
regression is established. This is consistent with Inductor's default
codegen_upcast_to_fp32 and emulate_precision_casts=False.

Historical attention combo includes KV512, Q1536, AND indexer LayerNorm128,
all reading packed stride2336 at offsets1536/0/2048. The LayerNorm output has
stride256 with a128-element gap. Old combo uses XBLOCK2/RBLOCK1024/eight warps;
split KV uses XBLOCK2/one warp, Q2/1024/eight, LayerNorm8/two. All one stage.
Old coverage is static-future callback evidence, NOT a complete graph inventory
for these combo bindings. New split sources have actual graph-held receipts.

`benchmarks/kernels/check_glm53_attention_norms.py` preserves exact source bytes
and selected configs/cubin bytes, uses real layer11 weights, and tests one fixed
120-pair matrix across all four ranks. CPU discovery verifies16 target sources,
5172 original files, and the reference receipts. Also finds identical4096 bodies
with different old/new launch configs (many-to-many body matches, NOT one-to-one
model-site correspondence); broader4096 choices remain a separate confounder.

NEXT: after commit, final manifest preparation then ONE16GiB/no-swap probe and
8GiB audit, commands/gates at protocol tail. Freeze sources/native through audit.
No model job, performance claim, default change, or retry prescribed. The CPU
discovery manifest is not the final frozen manifest. Goal ongoing.

### Previous checkpoint (2026-09-10): real workload completes; no-combo policy fails quality gate

The no-combo series is now TERMINAL after its ONE fresh-a on59ae0c88f. Do NOT
launch fresh-b/cached-a. Unlike the prior compiler failure, this profile reaches
health and finishes the complete workload:25 warmup/75 timed cold1000/300
requests, text4/imageRed,168 quality requests/12288 text+504 needle scores,
32K/128K prefill (two warmups/six timed requests). Controller/server exit0;
GPUs released0.559451s, final independent compute query empty.

Every individual text/needle score repeats EXACTLY across all three passes.
Mean-2.7264740343091405, all needle rankings pass. BUT12/32 windows fail the
UNCHANGED0.01-nat floor versus fixed native controls, identically in all passes:
3/4/8/9/10/15/20/23/26/27/28/30. Worst window28 is0.098160 beyond the allowed
floor (0.108160 below the control). All36 comparisons to12 historical passes
differ. Aggregate improvement is NOT acceptance. No policy/TC/default promotion,
no independent-start reproducibility claim, no retry to seek a passing score.

Diagnostic E2E medians c1/c8/c16:155.740/574.245/777.768 tok/s; cold engine TTFT
32K2.584496s/128K10.875915s. Startup184.083560s; first text canary48.898762s
(cold JIT work), image0.567689s.6 recovered4,718,592,000-byte allocation warnings,
2 zombies at teardown, all logs retained. These are NOT new baselines or wins.

Supplemental audit on frozen sources verifies30 source receipts,24 benchmark
receipts,7 native libraries,5172 original files,36 graph modules/132 bindings/
68 reductions and actual metadata/cubin bytes. Reductions unchanged across
capture; one additional pointwise-only graph per rank appears during capture.
16 norm bindings have actual widths512/1536, outside the original4096 geometry probe.
Correction2026-09-10:2048 was the rounded size hint, NOT the Q norm width.
512: persistent XBLOCK2/one warp/one stage, cache keyATT5PLJ...;
1536 (hint2048): XBLOCK2/RBLOCK1024/eight warps/one stage, keyULYNKQF....
Their exact source paths/configs/binaries are in the supplemental audit.

Original prescribed audit failed first on an ENVIRONMENT CHECKER BUG: new
receipts include vLLM's TORCHINDUCTOR_COMPILE_THREADS=1 and
TRITON_CACHE_AUTOTUNING=1 import defaults; expected env omitted them. Also,
the sibling-cache inference introduced before launch was WRONG for this actual
AOT path: decorators redirect only Inductor, while Triton stays in the recorded
launch directory. Corrected offline checker now requires an explicit private
Triton root; no guessing/search fallback. Smaller-norm qualification remains
closed. Initial supplemental whole-snapshot equality and sibling-cache assumptions
also failed; both scripts/logs preserved. None invalidates the verified quality
failure. No model reruns were used to repair audit tooling.

Authoritative supplemental evidence:
`perf/results/2026-09-10/runtime-control/no-combo-first-workload-analysis.json`,
SHAeb333fb93c2ac28218d46ca7799d95ce2378f73ebc0ccfe5d49138b6e2530c87.
The original failed audit remains at `deterministic-no-combo-serving/fresh-a-analysis.json`.

NEXT: isolate the score change before new serving starts. Inspect the exact512/
1536 norm sources/configs and other emitted reductions against actual historical
graph bindings; qualify newly covered shapes with the existing oracle contract.
Separately assess disabled combo fusion and fresh-cache KDA autotune choices as
confounders. Their causal roles are UNPROVEN. The older four-RMSNorm sufficiency
result applies only to the completed old/native pair, not this new score vector.
Do not widen quality gates, clear TC, reuse stopped arms, or claim universal
determinism. No next GPU job is prescribed yet. All GPUs free, goal ongoing.

### Previous checkpoint (2026-09-10): no-combo frontend passes; NEW serving series prescribed

Corrected-policy frontend on532c62674 completes12/12, including independent
unequal-size pointwise branches. Original eight cases (outputs AND oracle metrics)
exactly match the first frontend.6 graphs/9 bindings/6 reductions, identical
before/after capture, qualified1/1024/eight-warps/one-stage choices, cubin bytes
verified, FP64 maximum1 BF16 ULP.10 sources/7 native libraries/5172 original
files unchanged; GPU-free. Not a TP4/model-forward result.
`perf/results/2026-09-10/runtime-control/no-combo-frontend-analysis.json`,
SHA6facb3efedd788e3dc527fd3687d09ddefd61096a528d8cc876c8b55e313a6f9.

Serving auditor now targets NEW
`perf/results/2026-09-10/deterministic-no-combo-serving/`, three explicit compiler
options (deterministic=True/combo_kernels=False/benchmark_combo_kernel=False).
Source-bound binary lookup handles vLLM's AOT indirection: compiler_interface
overrides the launch TRITON_CACHE_DIR with a sibling triton_cache of the actual
inductor_cache. No unrelated-cache search or file substitution. CPU127pass and
real-artifact replay pass. No serving numerical or native changes in this step.

NEXT: prepare once, then fresh-a/fresh-b/cached-a, exactly one start each, using
the NEW root and no-combo scope names. Per-arm8GiB preflight/audit,150GiB serving,
swap0. Full cold1000/300 c1/c8/c16x3, text/image, qualityx3,32K/128K prefill.
Freeze through ALL starts/audits, no retries/edits/builds/commits; stop on failure.
Commands and unchanged quality gates are at the protocol tail. None of this NEW
series launched at this checkpoint. OLD series remains terminal after compile
failure; never launch its unused arms. No default, TC or speed promotion.

### Previous checkpoint (2026-09-10 00:07 UTC): fresh compilation exposes timed-combo conflict

The first full-model deterministic series on7d43c93af is TERMINAL after fresh-a
fails BEFORE health/capture/requests. Do not launch its fresh-b/cached-a arms.
vLLM defaults enable combo_kernels=True/benchmark_combo_kernel=True. The actual
Inductor scheduler calls speedup_by_combo_kernel -> benchmark_fused_nodes ->
may_ban_benchmarking, which correctly rejects this in deterministic mode. This
is a compiler-option compatibility failure, not a model correctness/TPS result.

Frozen failure closure checks28 sources,7 native binaries,5172 original files,
24 benchmark source receipts; cache-b remains empty, no other arms started,
138 partial cache-a files preserved. Server/controller exit1; teardown complete,
GPUs released0.044349s, independent query empty. Raw series/audit preserved.
`runtime-control/deterministic-serving-failure-close.json`,
SHA67602718eeb3c8019bac83056b1b5c97f6a2e9b174b966f66d20064e4039456f.

Corrected opt-in candidate now records THREE explicit options: deterministic=True,
combo_kernels=False, benchmark_combo_kernel=False. Default profile untouched.
This disables optional horizontal combo fusion for qualification; it is NOT a
performance promotion. Disabling only its timing gate would accept unqualified
static combinations, so those remain future optimization work. Conflicts reject;
actual vLLM defaults/cache-key separation/installed scheduler guard tested.

NEXT: ONE fresh16GiB/no-swap extended frontend probe, output
`perf/results/2026-09-10/deterministic-reduction-no-combo-frontend`.12 cases:
in-place/three-copy/independent-pointwise-branch graphs x rows1/16/640/7616.
Same real norm weight/FP64<=1BF16-ULP/replay/guard/mutation checks; new branches
also require exact eager/replay and CPU operation equality. Compare the eight
original corresponding outputs to the completed first frontend. Freeze sources
through probe/audit. No corrected-policy full-model series prescribed yet.
Read protocol tail. Existing first-series auditor is historical, NOT the next
series command; adapt it with a fresh namespace only after this probe passes.

### Previous checkpoint (2026-09-09 23:55 UTC): frontend passes; full-model series prescribed

The ONE prescribed frontend GPU process on49f9bff98 completes8/8 cases. Two
actual vLLM RMSNorm IR graphs x rows1/16/640/7616, native lowering, real layer22
weight. Repeated eager/changed-input graphs/guards/mutation checks pass; FP64
maximum1 BF16 ULP. Four actual graph bindings are identical before/after capture,
all XBLOCK1/RBLOCK1024/eight warps/one stage. Offline audit checks emitted metadata,
selected cache keys AND cubin byte hashes,7 source receipts,7 native libraries,
all5172 original files and empty GPU compute query. Not a TP4/full-model forward.
Audit `runtime-control/deterministic-reduction-frontend-analysis.json`,
SHA48830fce4a1fec6868f23609dd6cd8510f164e32d263f5996d5de72f2b9b815b.
Initial offline audit's hex/base32 cache-key mismatch is corrected; failure log
preserved. No GPU retry. Do not rerun the completed frontend output.

Full-model auditor now implemented BEFORE launches: independent graph-source
symbol inventory versus actual globals, emitted policy/config/cubin checks,
source/native/original-cache/recipe/exact-token/quality/prefill gates. Cache paths
are now recorded in benchmark environment receipts. CPU122pass; real-artifact
replay passes4 frontend bindings,12 timing files,3 quality passes,24 cold prefill
requests. No serving/profile/quant/kernel/default changes.

NEXT: prepare `deterministic-reduction-serving/`, freeze sources/native binaries,
then exactly `fresh-a`, `fresh-b`, `cached-a`, one start each in150GiB/no-swap
scopes, with per-arm8GiB preflight/audit. Independent EMPTY VLLM/Inductor/Triton
caches A/B; cached return reuses A without deleting its first receipts. Full
cold1000/300 c1/c8/c16 x3, text/image, quality x3, cold32K/128K. Stop on failed
gate, preserve outputs, no retries/edits/commits through series and audits.
Commands/stop gates: `perf/glm53-deterministic-reductions-protocol.md` tail.
No full-model launch yet at this checkpoint; no policy or performance promotion.

### Previous checkpoint (2026-09-09 23:28 UTC): deterministic runtime policy passes

The TWO source-runtime qualification jobs on3e497c7c0 are COMPLETE:32 cases
pass,16 corresponding cases exactly identical across independent empty caches.
All four sources automatically select XBLOCK1/RBLOCK1024/eight warps/one stage,
emit known qualified binaries, never benchmark, and disable dynamic RBLOCK/
coordinate descent. FP64 maximum1 BF16 ULP; repeated eager/changed-input graphs/
guards/mutation checks pass.5 helper/source receipts,7 native binaries,5172
original files unchanged; GPU-free. No model forwards or frontend GPU codegen.
Audit `runtime-control/rmsnorm-deterministic-policy-analysis.json`,
SHA8d523f85d74d637e7fe4b01a74cf56a4f976863124288226f1c8a8fddba99332.
Do NOT rerun `rmsnorm-deterministic-policy-{a,b}`; all outputs preserved.

Opt-in candidate now wired: `slimserve ... --deterministic-reductions` and matching
campaign flag call `slimserve/deterministic_reductions.py`. Adds only the supported
Inductor deterministic=True option to a copied, RECORDED engine plan; requires
native-order1/fixed GLM53 RTX6000 recipe, rejects cached-source intervention and
global deterministic/batch-invariant controls. No new environment switch, runtime
source mutation, registry/default/quant/native changes.23 benchmark source receipts.
CPU220pass/14 existing warnings, including actual compilation cache-key separation,
real frontend metadata-generator propagation (backend ID mocked, no GPU codegen),
CLI/default-off/scope checks and benchmark receipt/command agreement. Real-machine
dry-run passes with expected config; raw `runtime-control/deterministic-plan-*`.

NEXT: no full-model launch with this option yet. Add read-only actual-loaded-graph
reduction receipts and their offline audit BEFORE prescribing a fresh-model series.
Must verify every graph-held reduction's metadata/config/binary, not infer coverage
from static callbacks or generated files alone. Then freeze independent empty-cache
starts plus a cached return, repeated full quality/exact cold timing and32K/128K
prefill. Protocol `perf/glm53-deterministic-reductions-protocol.md` tail. No serving
start names/launches prescribed yet. Native-order defaultOFF/TC0 remain; no policy
promotion, old quality-gate widening, TC exoneration or new speed claim.

### Previous checkpoint (2026-09-09): deterministic runtime-policy qualification

The RMSNorm causal investigation below is complete. New isolated probe
`benchmarks/kernels/check_glm53_deterministic_reductions.py` qualifies the
installed compiler's deterministic runtime selection, not a production cache
hook. Source inspection confirms config filtering, disabled dynamic RBLOCK
scaling and disabled coordinate descent.84 CPU tests pass; lint passes.

Prescribed next: exactly two sequential fresh16GiB/no-swap GPU processes,
`rmsnorm-deterministic-policy-a` and `rmsnorm-deterministic-policy-b`,16 cases
each (four exact sources x rows1/16/640/7616, real layer22 norm vector). The normal
automatic run path must never benchmark and must emit a previously qualified
binary. Check FP64 <=1 BF16 ULP, eager repeat/changed-input graphs/guards, then
every output hash across processes. Helpers/native sources and5172 original
files must remain unchanged. No source edits/commits between these jobs/audit.
Commands in `perf/glm53-deterministic-reductions-protocol.md`.

No GPU policy result yet. Serving/profile/native sources unchanged; TC0 and
native-order defaultOFF. This probe substitutes only decorator metadata during
isolated source import. Even if it passes, frontend propagation and fresh
full-model compilation/quality are still unqualified and remain the next steps.

### Previous checkpoint (2026-09-09 23:05 UTC): RMSNorm cause established

The graph-complete pair on76afce776 is COMPLETE and both audits pass. No-op
`rmsnorm-noop-complete-graph-control` matches EVERY text/needle-token score in
all9 native reference passes (27 cross-pairs). Complete legacy substitution
`rmsnorm-legacy-complete-graph-only` matches EVERY score in all3 older instrumented
reference passes (9 cross-pairs), repeats exactly internally, and differs from
native. The FOUR recorded RMSNorm config changes are SUFFICIENT to reproduce
the full old/native score difference for this workload/AOT source set. This
resolves that specific causal question; the earlier partial intervention's
negative interpretation remains withdrawn. Do not rerun completed arms.

Both arms verify9 actual graph bindings (1/2/4/2),22 source receipts,5172
unchanged original files, unchanged private seed files and native binaries.
One start each; no edits/builds/commits between arms or before all audits.
Each completes25 warmup/75 timed cold exact1000/300 requests, text4/imageRed,
168 quality requests/12288 text/504 needle-token scores. Control/legacy means
-2.727814820100083/-2.7282744364256297. Diagnostic E2E medians c1/c8/c16:
157.241/580.417/779.867 and156.888/577.303/780.346; startup154.087/154.083s.
No new performance win or production promotion. Both controllers/servers exit0,
GPUs free;8 recovered allocation warnings each. Teardown details in notebook.

Combined conclusion `runtime-control/rmsnorm-complete-graph-pair-conclusion.json`,
SHA733c8b239cdb16f70c1914d5af878be0681ac78b7deb58266c3ec3aa8a9c811b.
Arm audits b8a3f1223798174e1f1a4ca691b60e2b2b75357f4f7fa027cd0e073b343fe415
andac819cd548544053ee14cff2c6590ecc58dd5973ae2022c13bc2d28a4a5a3a13,
`runtime-control/rmsnorm-{noop-complete-graph-control,legacy-complete-graph-only}-analysis.json`.

NEXT: replace cache-dependent numerical choices with an explicit reproducible
compiler/normalization policy, qualify fresh compilation and model quality,
then re-test the TC candidate under a matched fixed policy and return to measured
decode/prefill optimization. Installed Inductor exposes `deterministic` config
and reduction filtering; inspect/qualify it before inventing a production cache
hook. The current hash-bound intervention remains a defaultOFF diagnostic, not
the production policy. Native-order defaultOFF/TC0 unchanged. This result does
NOT prove each config individually necessary, universal determinism, TC accuracy,
fresh-cache behavior, or resolution of startup throughput/allocator variability.

### Previous checkpoint (2026-09-09: graph-complete serving pair frozen)

Serving auditor is now tracked in `benchmarks/analyze_glm53_rmsnorm_intervention.py`
with tested graph-receipt gates in `analyze_glm53_rmsnorm_graphs.py`.171 CPU tests
pass; all8 preserved real-loader receipts pass replay,18 total graph bindings.
Requires the independently inspected module/symbol inventory, not static callback
counts; graph-source hashes, selected configs/binaries, live coverage and source/
cache/request/quality checks remain strict. Serving/kernel/profile sources unchanged.

Next prescribed pair: ONE `rmsnorm-noop-complete-graph-control`, then ONE
`rmsnorm-legacy-complete-graph-only` ONLY after every no-op score equals all9
native reference passes. Fresh copies `rmsnorm-complete-graph-serving-caches`.
Use the protocol tail and tracked auditor module with frozen full commit SHA.
No edits/builds/commits between arms or before audits.150GiB serving/8GiB CPU,
no swap; GPUs must be free before each start. No completed full-model result for
this graph-complete repair yet. All earlier failures/partial results preserved.

### Previous checkpoint (2026-09-09 22:31 UTC)

The idempotent static-hook serving pair on3298bacd1 is COMPLETE. No-op
`rmsnorm-noop-idempotent-control` passes every exact-score comparison against
all9 native reference passes (27 cross-pairs). Both arms complete all cold
exact-token requests, canaries and three quality passes;22 sources/5172 original
files/native hashes verify, no commits between arms/audits. GPUs released.

IMPORTANT: `rmsnorm-legacy-idempotent-only` is NOT a qualified complete
four-source intervention. It repeats its own scores exactly (mean-2.73100041148523)
but differs from old/native references. A follow-up weight-free rank2 inspection
PROVES the static hook can miss actual graph globals:3 of4 bindings still native,
two distinct missed objects. All50 static bundles loaded, so forced-AOT/static
callback coverage was insufficient. Live graph globals were not captured; do not
claim their exact missing count. Withdraw the provisional conclusion that all
four RMSNorm changes fail to explain the historical mismatch. That remains OPEN.
The no-op equality is valid; it does not validate legacy completeness. Both runs
and their original audits remain unchanged, including slow readings/warnings.

Current graph-coverage repair is now locally qualified, defaultOFF. It retains
the idempotent static hook, additionally binds exact target globals after each
PyCodeCache module finishes loading and BEFORE returning its callable, and checks
every loaded target symbol/object/source/config/binary at capture. No source/hash
or quality gate relaxation. Diagnostic sourceSHAc7675323370c64caf38eeb67280376391c98b051e272bf543b025023c129d3c4.
CPU146pass/14 existing warnings; GPU28 exact numerical cases/154 resolutions;
8 real loader processes with independent graph-global inspection all pass:
18/18 target graph bindings (1/2/4/2 per rank per mode),56/56 artifacts,400/400
static kernels, originals unchanged. No model forwards/weights in these checks.
Audit `runtime-control/rmsnorm-complete-graph-qualification-analysis.json`,
SHAf480e98300607517aad2b1b59a9cda71146dc5e85793b1d913c843ea140e0297.
No full-model start has used this graph-coverage repair yet.

NEXT: update the serving auditor for graph_binding/graph_coverage receipts and
resolved_by=graph (the existing audit_rmsnorm_intervention.py still audits the
previous idempotent pair). Then prescribe a fresh one-no-op/one-legacy pair with
new cache/output names, legacy only after exact no-op equality against all native
passes. Commit/freeze source before serving; no edits/commits between paired arms.
Do NOT reuse prior result directories/caches or infer full coverage from counts.
After actual complete intervention evidence, implement reproducible normalization
policy and return to measured performance/TC work. Native-order defaultOFF/TC0,
selected quant and all native binaries unchanged. Campaign remains active.

Additional raw: completed audits `runtime-control/rmsnorm-noop-idempotent-control-analysis.json`
(8f641ab9...) and `rmsnorm-legacy-idempotent-only-analysis.json` (20806d7c...);
coverage failure `rmsnorm-graph-bindings-rank2/loaded-graph-bindings.json`
(ab027656...). Static inventory `runtime-control/rmsnorm-serialized-static-inventory.json`
(51a2f1dc...) records400 entries/56 identical normalized graph sources and preserves
all6516 old/5172 native cache files. It inventories serialized configs, not historical
execution. Full details/results/teardown warnings are in the notebook tail.

### Previous checkpoint (2026-09-09 21:50 UTC)

The SECOND no-op `rmsnorm-noop-bindings-control` on0d5ab7dc7 also FAILED BEFORE
HEALTH. Ranks0/1/3 loaded AOT; rank2 failed inside loading on the bare-object-ID
guard, before seal. Same-object resolution versus recycled ID is unknowable from
those receipts. All68 observed launchers unchanged; no benchmark/quality requests,
neither legacy serving arm ran. Status repair works: root/run failed, controller130,
server1, teardown complete, GPU release0.048535s. Original5172 files/22 sources
verified. Audit `runtime-control/rmsnorm-noop-bindings-startup-failure.json`,
SHA66f3ce5a072898e823c429ebd712fcdc02c6b3db3d8bd76c564e9b2777b1f4e5.

Idempotent repair now qualified, defaultOFF: retain strong target references,
serialize upstream resolution + replacement, reuse known selected objects without
letting cache recheck undo the choice. Exact source/config/binary checks every call;
read-only known repeats after seal, new late objects fail. Source coverage is not
Python object uniqueness. CPU138pass/14 existing warnings. GPU28 exact cases now
exercise154 resolutions including concurrent aliases and post-seal callbacks.

NEW weight-free reproducer `benchmarks/kernels/check_glm53_aot_loader.py` exercises
the actual concurrent serialized submodule loading, not a toy Future. Eight
prescribed rank/mode processes all load7/7 artifacts,50/50 static kernels each,
with target coverage and unchanged unrelated launchers. No weights/forward calls.
Initial empty-cache observation retained: it showed two-thread re-resolution of
a non-target future but had bundle misses, so is not qualification. Corrected
cache-matched observation and all eight checks pass; no scheduling-cause claim.
Audit `runtime-control/rmsnorm-loader-qualification-analysis.json`,
SHAb70122c0e1becadbdb01277bd279b779abaf1f2e5189a87aa18701485634efff.
Original5172 files unchanged, all GPUs free. No native/profile/quant/TC changes.

NEXT: freeze the qualified repair and prescribe ONE full-model idempotent no-op,
`rmsnorm-noop-idempotent-control`, then ONE `rmsnorm-legacy-idempotent-only`
ONLY IF all three no-op quality passes match all nine native passes exactly.
Fresh private copies `rmsnorm-intervention-idempotent-caches/{control,legacy}`.
Use the protocol's fixed commands/workload and new matching cache/manifest paths;
no source/native changes or commits during pair/audits. Auditor
`runtime-control/audit_rmsnorm_intervention.py ARM --commit FROZEN_FULL_SHA`
now handles strong bindings and repeat receipts. No third serving start yet.
Old failures remain preserved. Native-order defaultOFF and cross-mode quality
gate still FAILED; reproducible norm policy/performance/TC work remains open.

### Previous checkpoint (2026-09-09 21:22 UTC)

Normalization-only intervention is implemented and locally qualified, defaultOFF.
CPU128pass; GPU28 exact cases cover real StaticAutotunerFuture hook, four source
bindings with1/2/2/2 separate objects, both modes, rows16/640, changed-input graphs.
Source/native/replacement binary hashes required; unrelated launchers recorded.
Use perf/glm53-rmsnorm-intervention-protocol.md, including its REPAIR follow-up.

First no-op on ea2fff9ce FAILED BEFORE HEALTH: cached AOT loaded on all ranks,
but the initial guard confused one source with one Python autotuner object.
Actual objects1/2/2/2; all72 observed launchers unchanged. No benchmark/quality
requests, legacy arm NOT launched. Verified controller2674507 was interrupted
after worker errors; server1/controller130, teardown complete, GPUs free.
The raw summary's running/starting status is STALE (KeyboardInterrupt bypassed
exceptException); authoritative failure audit preserves/classifies it:
runtime-control/rmsnorm-noop-startup-failure.json
SHAbaf6736a06fbd851328ee99af6a0be4acb8a896acfe57ba477ccef6be477ab93.
All5172 original cache files/22 source receipts verified before repair.

Repair afe72ca52 replaces EVERY distinct object of each exact source, records
binding indices, requires source presence and rejects wrong hashes/same-object/
late bindings. Interrupts now mark run/root failed before teardown and rethrow,
so no stale running receipt or next boot. Original raw failure is untouched.
28-case GPU receipt45c6b6be1af71b3c33e783d97f3d0d126fef9d8e6c8308ac6ad018cb04b41faa.

NEXT: exactly ONE repaired no-op `rmsnorm-noop-bindings-control`, then ONE
`rmsnorm-legacy-bindings-only` ONLY IF all three no-op quality passes match all
nine old native-only passes exactly. No source/native changes or commits during
the pair/audits. Fresh byte-identical private cache copies are already prepared:
perf/results/2026-09-09/rmsnorm-intervention-bindings-caches/{control,legacy}/,
each with manifest.json and cache/. All5172 files copied/verified, GPU-free.
Flags SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC=control|legacy, matching absolute
SLIMSERVE_GLM53_RMSNORM_MANIFEST / VLLM_CACHE_ROOT paths, VLLM_FORCE_AOT_LOAD=1.
All other settings use native-order protocol (TC0/BF16fn1/no observers/asynchronous).
Full cold1000/300/c1,c8,c16/three repeats, text/image, three quality passes.
Auditor runtime-control/audit_rmsnorm_intervention.py ARM --commit FROZEN_HEAD
is prepared but not yet run; it expects the repaired output/cache names and
checks all binding occurrences, original-cache snapshot, all22 sources and scores.
Do not rerun old audits into existing outputs. No full-model intervention result
yet; original cross-mode quality gate stays FAILED and native-order defaultOFF.

### Previous checkpoint (2026-09-09 20:49 UTC)

Source-exact cached RMSNorm probe COMPLETE: all96 prescribed cases/192
cross-config comparisons,100 pairs differ (every7616-row pair). Max one BF16
ULP; both configs satisfy FP64 oracle. All eight rank/config combinations
reproduce their exact recorded serving-kernel hashes. Same-input repeats,
changed-input graphs, guard/mutation checks and triple rank0 outputs pass.
This CONFIRMS an arithmetic confound from autotune choice, NOT the full-model
score-mismatch cause or invalid accuracy of either kernel. No serving/quant/
native/default/TC changes. Original caches unchanged, GPUs free.

Probe22d609633 first completed rank1, then failed at rank3 graph capture due
to Torch's process-global default capture stream still onGPU1. Preserve that
failed attempt. Probe-only per-device stream fixada8f2915 completed JUST the
72 outstanding rank3/0/2 cases; no rank1 replacement. CPU15pass, raw audit
verifies96 unique cases across both artifacts. Results/notebook/protocol:
perf/glm53-cached-rmsnorm-protocol.md; raw cached-rmsnorm-{isolation,remainder}/;
runtime-control/cached-rmsnorm-analysis.json
SHAb472082671205e9dc2c40c9cc744d9da94721b7cfbf8ab0a45b9c295132d660c.

NEXT: narrow full-model normalization-only intervention, with a no-op control
that first matches the nine existing native-only quality passes. Preserve all
unrelated cached launchers; require source/config/binary-bound receipts. Merely
editing best_config files is not proof of changed executed kernels: AOT bundles
static launchers, reload via StaticAutotunerFuture.result/recheck_autotune_cache.
No intervention code or launches yet. No global force-first-config, original
cache deletion, gate widening, default promotion or TC exoneration. The older
cross-mode gate stays FAILED. Current serving implementation remains95fe67870;
new commits only diagnostic/tests/docs. Continue toward an explicit reproducible
normalization policy, then measured performance and TC reevaluation.

### Previous checkpoint (2026-09-09 20:27 UTC)

Both prescribed native-only isolation arms are COMPLETE on6422d43d6. Together
with the initial normal run on95fe67870, every text/needle-token score is exact
across all NINE passes /36 pairings. Launch serialization alone does not change
these outputs; all three starts share the c8b11c6e AOT namespace. Old instrumented
control comparisons remain FAILED; do not relabel that gate or promote yet.
No serving/GPU job remains. Both new arms have eight recovered allocator OOM
warnings during quality; preserve them. Full notebook records request counts,
teardown warnings and hashes. No native/profile/quant/TC changes.

Strong next lead: compiler cache audit finds19 changed configurations of68,
including four source-identical RMSNorm kernels with changed R0_BLOCK/warps:
rank0 rms_norm_1 (jt), rank1 rms_norm_0 (4j):4096/16 ->1024/8;
rank2 rms_norm_1 (mv), rank3 rms_norm_0 (jo):1024/8 ->4096/16.
Source-to-config binding uses actual AutotuneCache key methods; each source
filename occurs in that rank's serialized model. This changes summation order,
but full-model causality is not proved. All four printed model graphs match.
Raw runtime-control/native-order-autotune-cache-comparison.json includes the
exact paths, four source texts, saved configs and all model/source hashes.
Next isolate these normalization choices at kernel level, then a narrowly
scoped full-model intervention if needed. Preserve BOTH caches; no global
first-config override or baseline/TC promotion. Current serving code is95fe67870;
6422d43d6 only records results/protocol. All commits use Auroter, no pushes.

### Initial normal-execution checkpoint

Normal native-order run on95fe67870 repeats EVERY text/needle-token score
across three passes with no journals/launch blocking. BUT all nine pairings
against instrumented3e1dac08f fail equality:4096 text scores differ, mean
absolute0.258485/max4.882419nat. Preserve this failed gate; default staysOFF.
All exact1000/300/cold requests and canaries pass; one-start medians
156.956/579.394/780.922 are qualification, not new competitive TPS. Native
hashes unchanged;19 source receipts verify. All four printed computation graphs
match; emitted-kernel/autotuner equivalence is not established. Scope exits0,
GPU release0.614s, GPUs free. Raw native-order-quality/ and
runtime-control/native-order-analysis.json; full details in the notebook.
Next fixed TWO-start isolation: one native-order + launch-blocking1, then one
native-order normal return; both no journals and three full quality passes.
See perf/glm53-native-order-protocol.md. No replacement starts or source changes
between those arms. Cross-mode mismatch cause, startup variability and TC remain
open. Finish these prescribed arms/audits before optimizing another kernel.

### Native policy implementation checkpoint

DefaultOFF SLIMSERVE_GLM53_NATIVE_ORDER now selects the qualified kernels
without enabling journals. Strict recipe/platform scope, no sorting fallback,
distinct graph-cache factor. Mixed diagnostic flags rejected.206 real GPU
wiring/graph cases pass under normal asynchronous execution; CPU608pass/one
existing skip after fixing a stale provenance-count assertion. Six original
CPU socket failures were sandbox denials; original logs retained. Native
QC39b302f0/corefe4/MoE1093 unchanged. Full native sanitizers remain qualified.
Next is ONE normal-execution serving start with three exact quality passes:
see perf/glm53-native-order-protocol.md for the frozen command and gates.
No journals or launch blocking, no promotion or new competitive TPS claim.

### Completed instrumented checkpoints

Small-M origin routing PASSES full-model qualification on c8ba8d56c:
one prescribed start, three quality passes, every text/needle score and all
1,024 short / 94 long tensors per rank exactly repeat and match the frozen
28d2249c1 control across all nine pass pairs. All 348 archives verify.
Both text/image canaries and all 25 warmup / 75 timed cold requests pass.
Raw stable-route-quality-diagnostic/ and runtime-control/stable-route-*
analyses; the notebook records hashes, warnings and limits.

The native stable path takes 17.09-21.28% less time than atomic routing plus
sorting across 32 synthetic shapes / 4,800 samples. This removes a separate
sorting launch; it is NOT a raw-router or serving TPS gain. Installed QC
33decd2f also repairs missing warp ordering in the existing router; core
fe4a7c2a and MoE 1093b8a4 are unchanged. All 746 non-router GPU copies match.
Both native policies pass functional/sanitizer checks; final wiring passes
713 GPU tests, 519 CPU tests / one skip and 52 baseline-exclusion tests.
Sorting is skipped only for the actual reused canonical alignment.

Keep the c16 repeat-2 slowdown: 387.96 versus 648.38 / 646.17 tok/s, including
a shared 4.371-second client-arrival gap before six requests' first tokens.
Cause unresolved; no replacement sample. These serialized/instrumented rates
are not production baselines. Stable routing remains opt-in; profile, quant
and TC defaults are unchanged.

M>16 direct-count/scatter512 prototype is qualified in isolation (2b689556c,
probe9b1ed132): 520 functional/memory/sync cases and eight bounded races pass.
One missing-env memcheck launch failed before kernels and is retained separately.
All12,000 samples verify; scatter512 strictly improves26/32 cases versus
scatter256, none strictly regress. M7616 random21.08us versus222.96us for
alignment-plus-sort; actualM6405.89-6.09us versus12.95-12.97us. Unstable
alignment alone still wins30/32 distributions. No serving integration/TPS claim.
Selected native policy now locally qualified: QC39b302f0, M17..32 original256;
M33..8192 direct1024/scatter512. All748 old GPU bodies identical, four added.
171 functional/memory/sync and four bounded races pass;4,800 native timing
samples verify, all32 cases17.47-90.45% faster than alignment-plus-sort.
ActualM6405.50-6.01us; M7616/random21.288us. Atomic alone still wins26 cases.
STABLE_ALIGN defaultsOFF; only actual native construction skips sorting.
Final CPU573pass/one skip. Source/native receipts and baseline exclusion updated.
Full-model equality PASSES on3e1dac08f: one prescribed start, all three quality
passes and all nine pairings against c8ba8d56c exactly match every score and
1024short/94long tensors per rank. All348 archives,25warmup/75timed cold requests
and text/image canaries pass. Raw stable-align-quality-diagnostic/ and
runtime-control/stable-align-* analyses. Source/native freeze verified.
Explicit verbose JIT logging captured specializations. The prior4.371s pause
did not recur (c16 651.98/649.48/645.40), but its cause remains unresolved.
These serialized/instrumented rates are NOT production baselines. Next qualify
one defaultOFF native-only ordering policy under normal asynchronous execution,
without journals/CUDA_LAUNCH_BLOCKING, preserving the recipe and quality gates.

### Chronological evidence (older next-step statements are historical)

Both current cold-prefix baselines are complete: three prescribed starts
per stack, all 27 timing rounds/225 exact 1000-in/300-out requests, text/image
canaries, 4096-token quality and six needle checks per start pass. Every
measured request reports zero cached tokens. Our unchanged selected recipe
delivers 155.903/575.555/779.064 E2E tok/s at c1/c8/c16, versus B12X
132.588/466.690/589.276: +17.6%/+23.3%/+32.2%. Cold 32K/128K engine TTFT
is 2.600/10.968s versus 2.714/10.824s: 4.2% lower at 32K, 1.3% higher at
128K. These are current request-level results, not new kernel speedups.
Raw: perf/results/2026-09-08/{cold-recipe-baseline,b12x-r281-cold-serving}/.

B12X is pinned R28.1 with the qualified host-driver/native-FA2 adaptations
below, no-spec/DCP1, and its unchanged 4096-token/prefill-compute-share0.4
scheduling. It uses DIFFERENT W4A4/FP8-KV precision, not our selected recipe.
These comparison measurements used mHC BF16 storage OFF and native QC
f52c2d3d27a30567. The subsequent paired-staging qualification below now enables
lossless BF16 fn storage on RTX6000; quant and spill-free indexer are unchanged.
Do not relabel the earlier comparison as flag1. No baseline starts were discarded.

Neither stack's client-decode metric is sustained full-concurrency decode:
both first-to-last windows include staggered cold prefill. During the common
interval when all requests are generating, median client arrivals are
673.36/956.43 tok/s for SlimServe c8/c16 versus B12X 663.52/958.55. These are
diagnostic windows, not replacement benchmark rates or pure GPU timings.
Batched decode is close; the large E2E advantage is not a kernel-only gain.
The cached reference's first c16 timing has a synchronized 2.67s client pause
of unresolved origin; keep its slow result. Raw stream analyses are under
runtime-control/{b12x,recipe}-cold-client-stream-windows.json.

Paired BF16 fn staging for mHC prefill passes all7020 actual-site cases
bit-for-bit and its five-round A/B/A: +15.3/+13.8/+13.3/+4.9% isolated
throughput at64/128/129/7616 rows against scalar BF16. This is NOT a serving
speedup. Memcheck and targeted race/sync checks pass. The native integration
is SM120/aligned-BF16 only, preserving scalar fallback for odd storage offsets
and other platforms/dtypes. Native QC136723a73c921cb333e5b1cb367610bb22b8cc624008c8d8d6684528d2717e9d
passed this qualification; the previous f52 binary is saved as runtime-control/
mhc-paired-native-before.so.54 operator tests, four alignment memchecks and
all7020 installed census cases pass. The fixed1/3/1 FP32/BF16/FP32 cold-profile
series is complete on native QC5d4d3790e9aeb6cb below. Its first process chain
stopped after candidate1 when Linux recorded a zombie still active in the GPU
driver. After a teardown-only fix, exactly the two outstanding candidates and
one return completed, with no replacement starts. All45 timing rounds/375
exact1000/300 requests, text/image,4096 scored tokens and six needle contrasts
per start, and30 cold32K/128K measurements pass. All native hashes and request
code are unchanged; only controller teardown/preflight/status changed.
Candidate E2E medians156.791/578.993/779.940 versus controls156.205/576.187/
779.276 and155.910/576.558/776.826. All nine candidate c1 readings exceed all
six control readings:0.38-0.57% median gain. Batched decode and prefill are
effectively neutral; candidate cold32K/128K engine TTFT2.58007/10.88374s.
Retain lossless BF16 storage in the RTX6000 profile only; other platforms and
model-global defaults remain unchanged.267 CPU tests pass/one GPU skip;
profile dry-run selects the same recipe and flag1. Raw mhc-paired-* directories,
runtime-control/mhc-paired-combined-analysis.json. New teardown observed real
driver-release delays0.990/0.735/0.461s and completed safely, exit0, GPUs free.
All69 existing mHC kernels preserve resource usage; two paired variants add no spills.
Current native after the additional mHC device-switch repair is QC
20588761728d7161ae774e84e176fdfe4550163e3bfab1a2797ebee86bff35fb.
Its complete GPU instruction dump is identical to QC5d4 below; only checked,
device-aware host launch setup changed.151 GPU tests and270 CPU tests pass
(one GPU skip). One prescribed unprofiled serving start then passes all nine
timing rounds/75 exact cold1000/300 requests, text/image,4096 scored tokens,
six needles and six cold32K/128K measurements. E2E157.127/578.478/780.015;
engine TTFT2.57624/10.85622s. This is a correctness/sanity check, not a new
multi-start performance baseline. Raw mhc-device-serving-check/.

The tensor-core mHC candidate has completed independent accuracy qualification
and opt-in native integration; the production profile remains OFF. Its ORIGINAL strict parity census
failed case2638/2700 at a BF16 rounding boundary; that run and gate remain
failed, with2637 completed cases and62 untested. The operator approved a
separate FP64 accuracy/model-quality evaluation, specified beforehand in
`perf/glm53-mhc-tc-accuracy-contract.md`. The NEW census completes all2700 eager
cases (all rows/columns) and8100 changed-input graph phases (all-output bit
checks, explicitly sampled FP64 rows). Both baseline and candidate pass every
pointwise accuracy gate; candidate RMS is noninferior in every checked case.
The exact original seed2240 still fails strict parity and passes independent
accuracy. Sourceaf3113740, nativeQC20588761, unchanged probe45d5e817; raw
`perf/results/2026-09-09/mhc-tc-accuracy-census/` and
`runtime-control/mhc-tc-accuracy-analysis.json`.313 CPU tests pass/one skip.
Full-size cold A/B/A timing completes: latency falls 33.5/33.2/29.6/29.5%
at 64/65/128/129 rows, but only 2.61% at 7616 rows (about 613.37->597.34us).
Do not extrapolate the small-batch gain to long-context serving. Probe memcheck
and synccheck each pass48 selected cases/144 graph phases with zero errors.
Three bounded racechecks pass at64/65/129: first24 matching launches per process,
not full-size race qualification. NativeQC4ce801557216c67a adds only two GPU
kernels; all734 existing instruction bodies are unchanged. All2700 native
cases/8100 changed-input graph phases match the frozen probe bit-for-bit;
195 GPU tests and319 CPU tests plus3 receipt tests pass (one CPU-suite skip).
The44 new native tests also pass memcheck and synccheck, candidate kernels only.
Opt in with VLLM_GLM5_MHC_PREFILL_TC=1; unsupported settings/dtypes/alignment
retain the old path, and requested features fail startup on stale native builds.
The fixed1/3/1 real-profile series is COMPLETE, but promotion is REJECTED for
now: all three candidates fail the predeclared per-window quality gate
(10/5/8 windows), although aggregate scores and all retrieval contrasts pass.
Both TC0 controls themselves differ by0.244 nat/token mean absolute score
delta, up to6.787 on one token. Cause is not established; do not blame or
exonerate the new arithmetic, or relax the gate. The TC0 same-process diagnostic
now completes three quality passes:168 requests all cached_tokens0,12,288 scored
tokens/18 positive retrieval contrasts. All three pairs differ on4095-4096
scores, mean absolute0.240-0.248nat/token, max3.452-3.954;25-28/32 windows differ
by>0.01. No restart or TC arithmetic is needed to reproduce the variation.
Raw mhc-quality-repeat-diagnostic/ and runtime-control/mhc-quality-repeat-analysis.json.
The serialized diagnostic also completes all three passes: all4096 scores differ
in each pair, mean absolute0.247-0.250nat/token, RMS0.427-0.437, max4.114-5.092.
All168 quality requests cached0;12,288 scored tokens/18 positive contrasts.
Raw mhc-quality-serialized-diagnostic/ and its runtime-control analysis. Launch
completion serialization did not remove the variation; it does not rule out
intra-kernel/cross-rank issues. The score-boundary trace now completes onc1764a5d0:
all4 workers x3 passes x639 GPU scores exactly match HTTP. Every stage agrees
across ranks within a pass, but the prompt-head INPUT already differs between
all three identical requests. Full-workload per-token RMS remains0.427-0.444,
so instrumentation did not remove the variation. Raw
mhc-quality-score-trace-diagnostic/ and runtime-control/mhc-quality-score-trace-analysis.json.
The compiled model-boundary trace now also completes ona29fedc15: all4x3
matches contain91 operations/995 tensor records. Across EVERY pair/rank the
first difference is site-008.input.x (layer3 FFN output entering layer4 mHC).
All86 preceding records, including embeddings/positions and site7 FFN input,
are identical. This brackets layer3 post-attention normalization, router,
routed/shared experts and final reduction; the exact operation is not yet known.
All state links between mHC sites remain unchanged and immediate model return
equals the later prompt-head input. All GPU scores equal HTTP. Full-workload
RMS variation0.410-0.440 persists, so the trace did not suppress it. Raw
mhc-quality-model-trace-diagnostic/ and its runtime-control analysis.
The first-MoE capture on7f011a235 and standalone replay now isolate the first
numeric difference to gate/up Marlin's dependence on within-expert assignment
ordering. All3 passes/all4 ranks have identical input, router choices/weights,
packed weights/scales and zero lock workspaces. All semantic routes are valid;
only sorted assignment ordering changes before the first numeric difference.
Gate/up changes5-16/5,242,880 values per pair/rank. Replaying each saved order
with otherwise fixed inputs reproduces its serving output bit-for-bit in all
96 eager repeats and36 changed-layout graph replays. This proves ordering
causes the first GEMM differences, NOT yet the entire downstream score spread.
Raw: mhc-quality-moe-trace-diagnostic/, runtime-control/mhc-quality-moe-trace-analysis.json,
and mhc-moe-up-replay-repo/ (standalone tool benchmarks/kernels/replay_glm53_moe_up.py).
The opt-in canonical-order diagnostic is locally qualified:407 CPU tests,
14 alignment GPU tests plus3 integrated tests pass; bounded memory/sync/race
checks are clean. This deliberately adds a sorting launch AFTER alignment to
test causality; it is not the final production implementation. The prescribed
single-start three-pass run mhc-canonical-moe-quality-diagnostic/ now completes
on63d7704b0: all4096 text-token scores are exactly identical across all3 passes,
as are all1024 traced model tensor hashes per rank and every1K needle score.
All168 requests are cold and all18 retrieval contrasts pass. Long8K/32K needle
scores still vary (max1.623/1.937nat per scored token); this is NOT complete
model repeatability. Native/recipe unchanged, TC0; sorting remains diagnostic.
Raw verification: runtime-control/mhc-canonical-moe-quality-analysis.json.
The installed top_k_per_row_prefill probe now confirms variable selection order
above512 pools and variable membership for cutoff ties: all360 calls still
select valid top-k scores. This remains a lead, not full-model causal proof.
The prescribed indexer-long-context-quality-diagnostic/ run now completes on
9aa123ebe: one start/three full quality passes, canonical-MoE1/TC0 unchanged.
All348 archives verify, all168 quality requests are cold, all4096 short scores
and1024 short model fingerprints/rank stay exact, all18 contrasts pass. The
exact8199-token request runs7616+583 chunks, all11 selector layers,94 tensors
per match/rank. The first differing traced stage EVERY pair/rank is layer3
selection order. Its logits/ranges are identical within each rank across all
three passes; all selected sets match and no first-layer cutoff ties exist.
About2.70million index positions change, across5564-5565 of7616 rows. Later
layer logits and final model outputs differ. Full8K/32K score max deltas are
2.401/1.124nat. This narrows the lead to order, not a complete causal fix yet.
Separately, fixed first-layer logits differ between rank groups0/1 and2/3,
including two pool-set differences; preserve that follow-up instead of assuming
cross-rank bit equality. Analysis: runtime-control/indexer-long-context-quality-analysis.json.
GPUs released, serving exit0. The saved-input native replay now passes all44
executions (warmup+5 eager+5 graph on each GPU): every set remains exact while
each GPU emits11 distinct orders. Raw indexer-saved-input-replay/ onbd08719af.
Next qualify an order-only model intervention; do not change tie membership or math.
The bounded observer remains default-off;436 CPU tests (one skip) and8 GPU
observer/integration tests pass. No performance/default promotion.
The order-only intervention is now locally qualified: opt-in
SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER=1, requiring model/index traces and
canonical MoE. One Triton sort orders the existing selected pool IDs ascending
and keeps -1 padding last; no score access, reselection or native arithmetic
change. Default-off returns the original native callable. The observer now
lives in the GLM-specific selector path and records AFTER ordering.
450 CPU tests/one skip,34 GPU tests, bounded memory/sync/race checks pass.
The prescribed canonical-index-order-quality-diagnostic/ run now completes on
145467c58, unchanged recipe/native/TC0/canonical-MoE1. All348 archives and168
cold quality requests verify. All short scores/model hashes remain exact,
including against the unsorted control; first-layer inputs and selected sets
are unchanged, and selected order is now exactly canonical on every rank/pass.
The8K/position0.75 candidate scores now repeat exactly, but other long contexts
still vary: order-only is a PARTIAL fix. The first remaining traced difference
is layer23 indices on GPU3 (identical logits/ranges), followed by layer27 logits
on every rank. Passes2/3 of the traced true-code request match all94 tensors;
other candidate requests still vary, so do not infer whole-workload stability.
Full analysis and cross-intervention verification: runtime-control/
canonical-index-order-{quality-analysis,intervention-comparison}.json.
The observer now accepts a bounded capture_layer (default3, only the11 DSA
layers), so the next prescribed run archives layer23 without changing math.
468 CPU tests/one skip and14 GPU tests pass. The first GPU matrix attempt had
13 passes and one fixture compile-cache-limit failure; clearing the synthetic
op fixture cache between independent cases fixes it, with no serving-limit
change. Both attempts are retained. Next: ONE start/three full quality passes
in index-layer23-quality-diagnostic/, same canonical MoE/index-order flags,
TC0/native/recipe and exact prompt IDs. Only the archive target changes3->23.
The prescribed layer23 run now completes on4cead10f6: all348 archives and168
zero-cache quality requests verify. All25 upstream tensors are identical
within ranks AND across allnine old/new pass pairs. Actual cutoff ambiguity
is proven: row6329 has1582 visible pools,511 scores above -8.358503341674805,
and exactly two at that score: pool994 and1398. GPUs0/2 swap994->1398 andGPU3
swaps1398->994 in pass1 versus2/3; GPU1 stays fixed. All other selected members
match. This is exact cutoff membership variability, not invalid top-k scores.
Short scores/all1024 short tensors/1K and8K-position0.75 stay exact; other long
scores vary, max0.811271nat at8K and0.687448nat at32K. Pass2/3 of the traced
true-code request match all94 tensors, but distractors still vary.
Raw index-layer23-quality-diagnostic/, runtime-control/index-layer23-
{quality-analysis,observer-comparison}.json. Next implement/qualify a bounded
opt-in native tie policy at the selector origin, preserving all score values
and strictly-better selections. TC stays OFF and both ordering kernels remain
diagnostic, not production defaults; full-workload repeatability is unsolved.
Locally qualified afterfc531bb83: opt-in native pool-ID cutoff tie policy under
SLIMSERVE_GLM53_CANONICAL_INDEX_TIES=1 (requires existing order/trace flags).
Separate guarded GLM entry, generic default untouched. Both the final insertion
comparison and oversized exact-bin selection use pool ID; signed zeros share
a tie group. No extra GPU launch/workspace for the native tie decision.
480 CPU tests/one skip and162 GPU tests pass. Native core build completes
CUDA13/-j2/80GiB/no swap; core is now8828383f2993a22058de53bf3dc83947edc43f082270fd945ab702f6916f35d5.
All4187 pre-existing cubin function copies have identical instruction encodings
and scheduling bits; exactly one new kernel.40 registers/no local spills,
17424 static shared bytes plus2048 dynamic. Backup index-ties-native-before.so
preservesd45b4ace. Memcheck/synccheck each pass4 selected cases; bounded
racecheck first8 matching launches passes, zero errors/hazards. Raw
index-ties-native-sass-final/ and runtime-control/index-ties-* logs/XML.
Both prescribed runs now PASS on8f79f3b5c. Saved-input replay retains all88
outputs: every score set is valid, inputs unchanged; each GPU's old selector
emits two memberships while the new selector always matches the independent
CPU oracle (pool994 at the real row6329 tie). Raw index-ties-saved-input-replay/.
ONE start/three complete quality passes in index-ties-quality-diagnostic/ now
produce identical ALL4096 text scores and ALL1K/8K/32K needle-token scores.
All1024 short tensor hashes and94 long trace tensors per rank match across
all passes; all348 archives verify. All25 upstream tensors before layer23
selection match the prior run across all nine pass pairs. Same recipe/TC0,
all earlier diagnostic flags/serialization, capture_layer23; new native tie
policy only. Long-context values can change:8K-position0.75 margin is now
35.229299 versus37.148055 previously; all18 contrasts remain positive. This is
repeatability for this workload, NOT unchanged general quality or a speed win.
Serialized E2E107.751/458.922/647.799 tok/s is not a baseline. Exit0, GPU
release0.108649s; no replacement starts or excluded samples. Analysis and
cross-run receipts: runtime-control/index-ties-{quality-analysis,
intervention-comparison}.json. Next remove extra diagnostic sorting launches
through qualified origin-level ordering, measure uninstrumented timing, then
revisit TC under the unchanged quality contract. Production defaults stay OFF.
Next fusion candidate locally qualified after4fc14c06f: separate native
glm53_top_k_per_row_ordered sorts512 IDs within the existing selector CTA;
no serving caller yet. CPU484 pass/one skip plus14 analyzer fixtures, GPU260
cases plus8 all-device guard cases pass. Memcheck/synccheck4cases each and
bounded racecheck8 launches have zero errors/hazards. Native core now92c8f136
(full hash/notebook); old8828383f preserved as index-fused-order-native-before.so.
40 registers/no spills; static shared19568+2048 dynamic versus17424+2048.
All4187 other kernel copies and the prior tie selector's valid-input path are
bit-identical. Four error-path assertion metadata/line-number instructions
change; the recorded checker permits ONLY these four, not selection changes.
Actual88-call replay then fixed FIVE-round A/B/A timing across18 actual/synthetic
shapes prescribed in the notebook. Neither has run yet; no speed claim/promotion.
First replay preflight stopped on a binary-verifier parser bug BEFORE GPU calls:
the older checker omitted separately printed scheduling words because of spaces
inside comments. Full saved disassembly was intact. New strict shared parser
and15 fixtures pass; rechecking BOTH builds restores the complete evidence,
including every scheduling word and the narrowly scoped assertion exception.
Use index-fused-order-native-sass/comparison-with-control-words.json, not the
earlier receipt. Failed preflight log retained; no replay/timing samples existed.
Completed on7d375d9ae: all88 saved-input calls now PASS exactly, all archived
outputs independently reverified. But the fixed18-shape/five-round A/B/A rejects
the radix fusion: actual full chunk66.417->116.890us warm (76% slower), last583
rows10.138->17.224us. Only the no-sort<=512-pool shortcut wins. Do not promote.
Raw index-fused-order-{saved-input-replay,timing}/ and runtime-control/
index-fused-order-analysis.json. Next replace, not accumulate, the rejected
radix candidate with a lighter in-block bitonic network and repeat qualification.
Serving defaults, diagnostic control and quality contract remain unchanged.
Bitonic replacement locally qualifies:268 GPU cases and native mem/sync/bounded
race checks pass;40 registers/no spills/static17424+2048 dynamic bytes. Core
nowfe4a7c2a (full hash/notebook), no serving caller. Binary comparison preserves
4185 other function copies and the GLM control's valid-input path; two generic
decode specializations also change compiler output. They are NOT on GLM's
pooled-select path, but matched32-case old/new regression suites both pass,
and the checker requires their exact binary/source-bound receipts. Do not
claim blanket unchanged code. Next same88-call replay and18-shape A/B/A into
index-bitonic-{saved-input-replay,timing}/, using corrected binary comparison
and both generic-decode qualification XMLs from the notebook. No timing yet.
Completed on934e20c90: ALL88 actual-input outputs match exactly and all18
fixed A/B/A shape medians improve warm. Full chunk66.191/57.661/66.212us
control/bitonic/return (12.9% less selector latency); conditioned85.984/71.648/
85.984us (16.7% less). Last583 rows~9% warm gain, conditioned essentially
neutral. Synthetic long decode gains~1.3-12.8% warm; short no-sort case~61-64%.
All270 warm/2700 conditioned samples and all archives verified; raw index-bitonic-
{saved-input-replay,timing}/, runtime-control/index-bitonic-analysis.json.
Next connect the qualified native entry via an OFF-by-default diagnostic flag,
then compiled-observer and full-model three-pass equality qualification. This
is a kernel win, NOT a serving TPS gain or production promotion yet.
Fused serving wiring is now an opt-in diagnostic under
SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED=1, requiring all prior canonical flags
and journals; the header records native-bitonic and the extra Triton sort is
omitted. CPU508 pass/one skip. First GPU suite exposed onlyGPU0 and therefore
failed three original-GPU replay cases after40 passes; full all-four rerun
is required/recorded separately. No native/default/profile changes. Next ONE
start/three full quality passes in index-fused-quality-diagnostic/ against
8f79's exact repeated tie-policy control; configs index-fused-{score,index}-config.json,
same prompt IDs/capture_layer23/TC0. See notebook for the complete frozen protocol.
Final all-four wrapper/GPU suite now PASSES43 cases46.89s; no kernel/source
changes were needed for the visibility-only rerun. Full-model qualification
COMPLETE on28d2249c1: one start/all three passes,75 priming/timing requests,
168 uncached quality requests,12,288 text scores/18 positive needle contrasts.
Every4096 text score and every needle token score repeats exactly; all1024
short and94 long tensors per rank match within the run AND against8f79's
stable-tie control across all nine pass pairs. All348 archives verify, as do
25 source hashes per run and the explicitly scoped native-binary comparison.
Full-model fusion preserves the selected tie policy's results on this workload.
Exit0, GPU release0.442477s; transient zombies/release delay retained. No retries
or excluded starts. Raw index-fused-quality-diagnostic/ and runtime-control/
index-fused-{quality-analysis,intervention-comparison}.json. Serialized E2E
107.845/457.533/645.385 is diagnostic, NOT a performance baseline. Retain the
qualified opt-in fusion; next construct stable MoE alignment at the source,
then qualify uninstrumented serving and revisit TC under the unchanged contract.
No profile/default/quant/TC change or cross-rank-equality claim.
Next small-M stable-routing probe passes324 synthetic eager/graph/device cases,
CPU508/one skip and324 mem/sync cases each, but bounded racecheck FAILS in the
existing scored top-k loop before scatter. Unchanged installed QC reproduces
the same shared-memory read/write warning. Timing is paused; add explicit warp
ordering before marking the selected winner and requalify BOTH atomic/stable
policies before measuring. Isolated probe80b05608 is preserved, all sources/
failures in the notebook/runtime-control/stable-route-* and native-route-*.
Native serving remains unchanged; no output mismatch or link to prior M640
score variation is proven. This is a new bounded sanitizer finding, not a
retraction of the full-model fused-indexer equality result above.
Explicit pre-overwrite warp sync now clears bounded M13/M16 racechecks for
BOTH probe policies (48 launches each, zero hazards), and the expanded324-case
suite plus full memcheck/synccheck passes with exact old IDs/weight bits.
Probe0fef8ce3 remains isolated; serving QC is still4ce80155. Next fixed32-shape
timing into stable-route-synced-timing/: separate installed->synced atomic,
synced atomic->stable, and synced atomic+sort->stable comparisons. See notebook
for frozen five-round A/B/A protocol. Native integration/real serving still owed.
Fixed timing now COMPLETE on5d1256853: all7200 samples verify. Stable scatter
costs at most0.1152us/2.60% versus synced atomic alone, but complete stable
ordering is17.16-21.01% faster than synced atomic plus diagnostic sort; all32
shape distributions strictly separated from both controls. M8 random6.1008/
4.8800/6.0992us A/B/A. This is synthetic warm-cache kernel throughput, NOT
serving TPS. Retain for opt-in native/serving qualification; M>16 unchanged.
Raw stable-route-synced-timing/ and runtime-control/stable-route-synced-analysis.json.
Native QC is now33decd2f (old4ce80155 preserved): both policies have the warp
sync fix, stable entry SM120-only, checked shape/capacity/device-aware launch.
All746 non-router GPU copies remain bit-identical;719 tests, full mem/sync and
bounded native M13/M16 racechecks pass. Native router code differs from probe
code; fresh native timing and real-profile qualification remain required.
Stable native entry has no serving caller yet. Next explicit diagnostic
provenance wiring, native timing, then full-model equality versus28d2249c1.
Upstream issue52525/PR52532 independently report Marlin ordering sensitivity;
the draft PR uses a post-alignment Torch sort, not a finished fast-path answer.
Keep the current recipe/native arithmetic;
do not relax quality gates or promote TC. This is diagnosis, not a throughput
baseline. TC stays OFF.
E2E control/candidate/return157.051/156.928/156.712 c1,
578.818/578.200/577.166 c8,781.849/779.553/777.069 c16. Candidate medians lie
between controls, but retain the slower c16 samples. Cold32K2582.669/
2576.940/2588.775ms;128K10868.380/10873.613/10924.900ms. Small32K gain,
no robust128K gain. All45 timing rounds/375 exact cold requests, all30 measured
prefills and30 retrieval contrasts complete; no retries, exclusions or code/
binary changes. Outputs: mhc-tc-serving-{control,candidate,return}/, native
QC4ce801 and retained BF16 storage in ALL arms; only TC changes0/1/0.
Raw combined performance and failed quality analysis are in runtime-control/
mhc-tc-serving-{analysis,quality-analysis}.json. All jobs ended and GPUs released.
The full timing/source receipts are in mhc-tc-qualified-timing/ and the native
integration proof is in mhc-tc-native-census/ under perf/results/2026-09-09/.
Do not equate this numerical qualification with a serving gain or promotion.

The FP8 sweep exposed a real decode-launch setup bug: an ELF GNU_UNIQUE flag
was shared across separately loaded CUDA modules, while kernel attributes
were not. The same flag also ignored device switches. Both call orders and
device0/1/0 reproduce invalid launches in the old path. The native wrapper
did not check launch errors and could return uninitialized output.
BF16/FP8 helpers now use module-local, thread-local, device-aware setup state,
check setup errors, and native wrappers check launch errors. Kernel math and
launch geometries are unchanged. The binary qualified at that checkpoint was
5d4d3790e9aeb6cbc92702a6d0754b625cecbb48a6ff93bde8a22b419ccb5760;
the 136 binary is preserved as runtime-control/fp8-setup-native-before.so.
Both call orders with both old/new probe modules pass the actual-weight
oracle; 145 GPU tests (including device switches and mHC) and 251 CPU tests
pass, one GPU-only CPU-suite skip. The failed sweep retains all24 completed
M1/M8 configurations and the failed M16 baseline. The corrected 12-config,
three-layer/three-batch, five-round cold A/B/A sweep passes all216 cases and
18 installed baseline checks. Current settings win every group except down
at M1: NT16/warps8/stages4 saves just0.05-0.09us (1.8-3.1%) per isolated call.
That candidate still needs routed-expert contention testing; no FP8 geometry
promoted. Raw fp8-launch-local-sweep/summary.json, source3c052d6af.
The model-free graph-creation-order return control also completes: all18
oracle phases pass; collected spans are4/4,4/0,4/4 for build-between,
prebuild-both, build-between. Each range has four runtime launches. Use one
range per serving process; whole-model observer requalification remains owed.

The spill-free pooled-indexer tile is retained in the RTX6000 profile:
`VLLM_GLM5_INDEXER_SM120_TILES=1` selects RT2/PT128/four warps. Three fixed
starts x three repeats plus a prescribed return-to-original control pass all
exact-token, text/image and expanded quality gates. Cold 32K/128K engine TTFT
is **2.606/10.933 s**, versus **2.634/11.433 s** in the return control: about
1%/4.4% lower latency. Decode stays **155.98/574.79/778.96** E2E tok/s at
c1/c8/c16; node-profiled gaps remain, with the observer caveat below.
All 14 full-selection tests,
changed-input graphs, memcheck, racecheck and synccheck pass. Other platforms
keep their original geometry. Raw: perf/results/2026-09-08/indexer-tiles/,
indexer-tile-serving/ and indexer-tile-return-control/.

Historical scalar-staging mHC BF16-storage A/B/A: one FP32 control, three BF16
starts, one FP32 return; all 45 exact timing rounds, text/image canaries,
4096-token quality and six needle checks per start pass. BF16 c1 median is
156.76 tok/s versus 155.98/156.00 in the controls, a repeatable ~0.5% gain.
c8/c16 differences are within the observed variation. Cold 32K/128K engine
TTFT is 2.614/10.994 s versus return 2.611/10.968 s: no prefill win; the trace
shows a small (~0.45 ms/full chunk) mHC cost. This older result kept storage
OFF pending the paired-staging follow-up, now completed and retained above.
All 7020 installed-kernel cases remain bit-exact, with unchanged FP32 arithmetic
and lossless checkpoint loading. Broader tests/sanitizers pass. Raw: the three
mhc-storage-{fp32-control,serving,return-control}/ directories in the notebook.

New graph-state evidence: the first FP32 control is mixed across ranks, not
one global state. Its c1 rank0 has 99 us/step of graph gaps and 788 us of
all-reduce duration; ranks1-3 have 424-428 us of gaps and 405-444 us of median
all-reduce duration. All have the same ~5.736 ms median graph span. In the
first BF16 start, all ranks have 423-430 us gaps and 409-447 us median all-reduce.
This supports rank0 waiting for slower graph execution elsewhere; do not treat
rank0's long collective duration alone as proof of a separately slow all-reduce
kernel. The fast/slow per-rank graph cause remains open. All eight replays/rank,
including first-replay skew, are retained in runtime-control/mhc-storage-all-rank-gaps.json.

Important profiling caveat: BF16 boot2's c8 trace changes from 108 us of gaps
in its first replay to 427-432 us in the next seven, on the SAME 1185-kernel
graph. Node tracing can perturb CUDA graph execution. Mixed-rank waiting is
visible in these traces, but its persistence without profiling is not yet
established. Test profiler on/off and whole-graph timing before changing graph
topology, stream attributes or the driver. Timed decode precedes profiling;
do not discard its measurements or equate a traced gap with recoverable TPS.
The one-GPU A/B/A probe confirms observer overhead: median paired graph latency
increases 17.93% for 1200 linear nodes and 55.78% for the two-branch graph.
All 18 phases pass exact output checks; post-profiler timings return near baseline.
This does not establish the full TP4 model's unprofiled graph span. Probe raw:
runtime-control/graph-profiler-bias/. The whole-model observer attempt below
is diagnostic and incomplete; it does not replace the unprofiled baseline.
The explicit --cuda-profile CLI and --cuda-traces campaign mode are prepared:
bounded CUDA API ranges, unchanged serving graph, and a separate full return
timing matrix. Run the campaign itself under Nsight process-tree graph tracing,
not its server child. The original lifecycle probe passed but created its
second graph after range one. Precreating both graphs reproduces missing
second-range graph activities despite four successful launches. The model's
first range records32 graphs per GPU (~5.660ms median); its c8 range has129
runtime launches but no spans. Keep that failed series as diagnostic evidence.
It also exposed Nsight-adopted zombies confusing teardown; a model-free
reproduction confirms the mechanism and fixed teardown records zombies without
accepting live workers. Use ONE range per model process via --cuda-traces
--cuda-trace-concurrency 1 or8; the full timing/return matrices stay unchanged.
Real-profile requalification is still owed. Raw: graph-observer-cold/ and
runtime-control/{nsys-prebuilt-graphs*,owned-group-*}.31 focused CPU tests pass.

The digest-pinned R28.1 reference has a reproduced container-startup failure
and a model-free A/B/A repair: its Bash startup hook selects compat libcuda
610.43.02, whose peer-memory imports fail with CUDA101 on this host. Skipping
that hook with BASH_ENV=/dev/null retains host libcuda580.173.02 and passes
all four ranks' every-peer writes/NCCL tests; restoring the hook fails again.
CUDA runtime13.3 and NCCL2.31.2 are unchanged. No host-driver changes or
transport disabling. The original fixed three serving starts all failed and
remain recorded in b12x-r281-serving/. A new fixed three-start campaign uses
--host-cuda-driver at b12x-r281-host-serving/ completed: all three starts
initialize B12X PCIe all-reduce and load weights, then fail vision FA2 warmup
with unsupported PTX toolchain (exit1, no OOM). Its FA2 binary contains only
SM80 cubins and CUDA13.3 PTX; host580 cannot JIT that PTX for SM120. A native
SM120 build of the image's exact FA2 source f3e1a4f74c99145c0717709860bf765de1703779
is complete, with only the architecture target changed. Independent FP64
oracle and original-binary comparison probe: benchmarks/kernels/probe_b12x_fa2.py.
The original image FA2 under its single-GPU compat driver passes all30
independent FP64-oracle/changed-input-graph cases (maxNRMS.002277). Rebuilt
binary qualification also passes all30 cases, bit-exact to the original in
every case and graph replay. Native library SHA256
31519f918c17425203dd7aaab19fe8b4c601e7f5ad5164e19aeb2e3f8cde5249;
76 native SM120 cubins, no PTX. Both containers exit0/no OOM. The next fixed
three-start serving campaign b12x-r281-native-serving/ is complete with the
explicit host-driver/qualified-FA2 adaptations. All reach health; starts1/2
fail the image canary with joined text RedRed; start3 passes and completes
all nine timing rounds/quality/cold prefill. Its75 timed requests all report
1000 cached prompt tokens versus0 in SlimServe's retained short-prompt data,
so its155.79/656.40/922.05 E2E medians are NOT a matched cold-prefix baseline.
Cold32K/128K engine TTFT2.729/11.000s; quality4096-token mean-2.913825,
all six needles pass. The one-start canary diagnostic now records distinct
reasoning Red solid image. and final content Red: the client concatenated
channels. Canaries now evaluate content only and retain all raw events;
interactive combined display and request bodies remain unchanged. Earlier
RedRed responses lack raw fields, so their exact split cannot be reconstructed.
The completed cold-prefix comparison above uses the same zero-cache gate
on all timing requests; no direct ratios from the cached series.
No vision disabling, canary relaxation, fastest-start selection or quant change.
`benchmarks/benchmark_glm53_b12x.py` now prepares a fixed three-start control
through that image's supported no-spec/DCP1/VRAM launcher; it reuses the exact
SlimServe workload functions via `benchmark_glm53_server.py`. CPU lifecycle,
failure-retention and tokenizer gates pass. Both tokenizers produce identical
427489 full-source token IDs despite serialized-default/template differences.
Do not run other GPU work alongside this control. It is a different
W4A4/FP8-KV configuration, never a replacement for the selected recipe.

The repaired Marlin library has completed the fixed three-start serving campaign:
all 27 exact timing runs, text/image canaries and expanded quality checks pass.
E2E medians are **155.91 / 574.42 / 779.05** at c1/c8/c16, performance-neutral.
Cold 32K/128K engine scheduled-to-first-token medians are **2.640 / 11.448 s**,
with explicit zero cached tokens, nine measurements per length. The original
production library had weight-tile reads overlapping reuse of the same shared
storage for block reduction. A single entry barrier removes the race in both
the isolated reproducer and installed auto-scheduled kernels. Memcheck and
synccheck pass; 24 changed-input cases across all five M tile sizes remain
bit-exact to the original. This is a native correctness repair, not a serving
speed claim or proof of numerical corruption in the old kernel. Scheduling
candidate remains unintegrated because one stricter oracle gate still fails.
Raw: perf/results/2026-09-08/marlin-repair-serving/. See baseline and notebook.

Goal: exceed the strongest reproducible B12X result on this hardware under
matched workloads and the model's recommended sampling. The historical R24
figures (169.9 / 737.8 no-spec, 247.8 / 903.2 MTP-3 at c1/c8) are sustained
context-zero decode, NOT this repository's 1000/300 complete-request TPS.
Do not divide one by the other or infer greedy sampling from acceptance
alone. B12X also uses a different NVFP4 checkpoint and FP8 KV. The local
R24 control (134.0 / 483.7 / 598.6 no-spec) did use our exact-token harness,
but disabled P2P/custom all-reduce after a collective initialization timeout.
The earlier claim that the CUDA version mismatch explained that failure is
not established: a model-free R24 container probe now passes every-pair IPC
writes and all three NCCL reductions with P2P enabled, loading host libcuda
580.173.02 with CUDA runtime 13.3. Its custom collective is not yet qualified.
R28.1 is now published with public source mirrors; see the 2026-09-08 audit
entries in perf/optimization_status.md. The matched cold-prefix control is
now complete above; other scheduler/speculation configurations remain separate.

The operator authorized autonomous continuation and incremental commits on
2026-09-08. The selected RTX6000 weight recipe is now explicit in the
registry: glm53-redhatai-nvfp4-fp8-kda-tp4-v1. It pins the original RedHatAI
and native ZAI revisions, the FP32 repairs, and the FP8 swap-set including
self-quantized KDA. It prepares an isolated directory and verifies sidecar
tensor digests before serving; it must not silently fall back to BF16.

The first fixed-count series is now complete (commit 143e18073): three
starts x three repetitions at c1/c8/c16, all 27 runs retained. Median
complete-request TPS is **156.11 / 576.31 / 775.44**; median client decode
TPS is 165.02 / 590.41 / 787.99. Every start passed text/image canaries.
All three node traces show the wider gaps; this does not prove the same
gaps persist without profiling. One cold c1 run includes a
157 ms first-JIT stall at the 1088-token state-copy boundary. Details and
limits: perf/baseline_status.md; raw: perf/results/2026-09-08/repro-baseline/.
The startup-copy fix has also passed a second fixed three-start series
(commit a2cd7a240): **156.40 / 576.57 / 776.29** median complete-request
TPS, with all 27 measurements retained. Its c1 range is 156.17-156.59;
the cold outlier is absent. Copy-kernel warmup happens before health and
benchmark warmups cover the full workload. Raw:
perf/results/2026-09-08/warmup-boundary/. Neither change claims to solve
unexplained startup throughput variation or improve steady-state throughput.

Record (rtx6000 profile, no-spec, fast/fast boot, 2026-09-08 12:29, run
mlapf-rec4): **c1 165.7 / c8 591.9 / c16 797.4**, gate -2.450. That is
the best of four starts, NOT a repeatable baseline, and +58% / +37% / +35%
over the Phase 0 baseline
(104.8 / 431.4 / 591.0). Every retained lever has a notebook entry with
its A/B (perf/optimization_status.md, 2026-09-04 through 2026-09-08).

Earlier target budgets mixed B12X sustained decode with our complete-request
timings. Those budgets are retired, not exit criteria. Establish a matched
control and measure actual routing/traffic before declaring a physical limit.

Historical attribution (profiled, rank 0, ms/step): c8 11.74, Marlin experts
5.34, fp8 dense GEMMs 4.0 busy but mostly
overlapped on graph-internal streams (net ~1.7), mHC 0.99,
custom AR 0.94, cuBLAS wmma 0.39, KDA 0.36, norms 0.32, sparse MLA 0.31,
indexer 0.23. c1 6.05: fp8 1.90, Marlin 1.11,
mHC 0.76, AR 0.45, cuBLAS gemv 0.44 (the bf16 lm_head GEMV is 0.2 of it),
norms 0.30, other 0.29, bf16 decode GEMM 0.25, indexer 0.20.

The sampler tie/FP64 repair is now validated through three fixed profile
starts (commit 4a7a5a73e): **156.08 / 575.67 / 777.04** median E2E TPS,
all 27 runs retained, all text/image and exact-token checks pass. It also
passes 60 kernel tests and clean memcheck/racecheck with an independent CPU
oracle. Every cutoff tie is retained; nucleus order is stable by token ID;
noise is vocabulary-indexed FP32/FP64. Seeded streams change from the old
32-draw implementation. Raw: perf/results/2026-09-08/sampler-exact/ and
sampler-serving/. Node-profiled graph gaps remain; unprofiled causality is
unresolved. A separate PyTorch reduction race report remains under
investigation; see the notebook.

The stronger quality baseline is complete (9de9ad871): three fixed starts,
all 27 timing measurements retained, E2E medians 155.93 / 575.96 / 778.33.
Each start scored 4096 explicit continuation tokens; mean log probability
ranged -2.735695 to -2.730236. All six equal-length retrieval contrasts ranked
the true code first at 1K/8K/32K context. This measures prefill quality, not
teacher-forced decode or broad capability. Raw: quality-baseline/ under the
same dated results directory. The serving binary/recipe were unchanged.

The isolated mHC/norm fusion assessment is complete and rejected: the existing
fused norm loses local A/B/A timing at batches 1/2/4/8 despite passing numerical
gates; it is not integrated. Cold-prefill attribution is now recorded, including
all-rank 32K/128K traces (the 128K trace samples only its first eight chunks).
The repaired library leaves the same 1185-node c1 graph and
about 0.429 ms of inter-kernel gaps. The
PyTorch reduction warning is isolated to the block-y/block-x shared-memory
boundary in global_reduce: a one-barrier isolated extension removes it, but
the installed Torch binary is unchanged and numerical corruption is unproven.

Routing capture is now validated (8c7f8fc23): 1890 actual decode steps,
no invalid records, all canaries/counts pass. Mean unique experts/layer are
8 / 52.85 / 89.38 at batch 1/8/16, not the old estimated 8/57/103. Raw:
perf/results/2026-09-08/routing-census/. Its synchronous scheduling and copies
make timing diagnostic-only. Unique weight footprint is not DRAM traffic;
MTP verifier reuse still needs its own capture.

Marlin counters are complete (00052fba9 probe): 27 actual-weight/routing cases,
54 launches, exact eager/graph output. Median cold DRAM bytes are within about
0.1-0.8% of unique packed expert weights plus group scales. Batch8/16 stream
much faster than batch1; prioritize a measured batch1 scheduling experiment.
These isolated profiled times omit serving overlap and are not a physical
ceiling. Raw: marlin-counters/cold-v2-summary.json in the dated results folder.

The output-parallel mHC experiment is rejected (25aa8ba35). Both versions pass
local parity but lose A/B/A timing across all 90 actual checkpoint parameter
sets at batches 1/2/4/8. The probe is quarantined under benchmarks/kernels;
no alternate serving path was added. Producer/consumer scheduling remains a
separate untested hypothesis.

Current candidate priorities (hypotheses, not physical ceilings):

1. mHC tensor-core prefill probe: separate independent FP64 qualification
   completes all2700 eager cases/8100 changed-input graph phases, with zero
   pointwise violations in both kernels. The original strict parity failure
   at T7616/site79/fused/seed2240 remains recorded, not reclassified. Follow
   `perf/glm53-mhc-tc-accuracy-contract.md`: sanitizers and gated native
   integration pass; the fixed control/candidate/return series completes but
   fails the per-window model-quality gate. Keep TC OFF and isolate the existing
   control/control score variability before considering promotion or more
   arithmetic changes. Same-process repeated TC0 scoring reproduces it; the
   first numeric difference is now reproduced by varying only the first MoE's
   within-expert assignment order. Canonical MoE alignment, pool output order
   and native pool-ID cutoff ties now make ALL prescribed short/long scores
   exact across three full-model passes on8f79f3b5c. The actual-input causal
   replays and full-model traces verify each intervention. Next replace the
   extra diagnostic sorts with qualified stable ordering at the source, then
   measure without tracing/serialization and re-evaluate TC fairly. This
   diagnostic result alone does not promote TC or establish a speed win.
   The unchanged probe is45d5e817, current nativeQC4ce80155. New artifacts are
   under perf/results/2026-09-09/mhc-tc-accuracy-* and mhc-tc-qualified-timing/.
   Five-round cold timing gives29-34% lower small-batch latency but only2.61%
   at the full chunk. No serving gain is claimed. Paired lossless storage is
   retained separately from this new
   arithmetic. Keep the current serving path while remaining gates run.
   Output-parallel arithmetic and actual 20-iteration
   norm fusion already lost; last-block synchronization was neutral. Do not
   repeat those experiments from the old three-iteration fixture.
2. FP8 launch geometry: full cold sweep completed; only downM1 NT16/8/4
   merits a bounded routed-expert contention check (0.05-0.09us isolated).
   All other current geometry retained. The old shared ROT24 fit L2
   and never varied warp count. Preserve the existing eight-stage contention
   evidence. The separate Marlin scheduling candidate failed its stricter
   numerical gate and remains unintegrated; the shared-memory race repair is
   retained. Measured Marlin traffic is already near the unique-weight floor.
3. Whole-model observer follow-up: fixed three starts, one bounded Nsight
   range per model process, explicitly choose --cuda-trace-concurrency 8.
   Preserve the full before/return matrices and every rank/boundary record.
   Do not change graph topology/driver or restart until a profiler looks fast.
4. Long-context prefill and weight staging: the matched 128K result is about
   1.3% slower than this B12X control. Keep the retained indexer geometry and
   use fresh attribution to select the next kernel/overlap experiment. B12X's
   L2-prefetch windows are a studied reference, not an implemented win here.
   The recipe's BF16 lm_head, activations and KV stay fixed. Sum neither
   overlapping kernel times nor mixed timing definitions into a speed claim.
5. MTP: the earlier k=1 probe served but regressed recommended-sampling throughput.
   Draft CUDA-graph capture remains the first cost-side experiment. Acceptance and
   actual verifier routing decide the benefit; estimated 57/103 expert counts do
   not prove a ceiling. Keep speculation off in the production record until a
   separately identified experiment passes full serving and correctness checks.

Deprioritized by current evidence: KDA f_b/g_b. All eight c8 replays in
sampler-serving/boot-1 contain 34 gate-projection kernels totaling about
86 us/step, not the older 0.39 ms attribution. Even a large local improvement
has little end-to-end value. Check fresh trace counts before reviving that item.

Retained since Phase 0, all default on the rtx6000 record (a switch named
here exists for one-factor A/Bs only):

- Phase 1 (2026-09-04): F32 sidecar for the RedHatAI downcast; KDA o_norm
  through the Triton kernel; g_a folded into the merged in-projection;
  f_b/g_b as one strided bmm; router gate through the QuixiCore GEMV;
  indexer wk/gate/weights_proj folded into fused_qkv_a_proj; fused
  routing + Marlin alignment (`glm_route_align`, made total over NaN
  logits on 2026-09-07 after five Xid 31 faults were traced to it; the
  cooperative mHC launch rejected on those same faults is now default for
  T <= 8, VLLM_DSV4_MHC_COOP_MAX_T).
- FP8 weight swap-set with KDA self-quantization (2026-09-07): the dense
  and DSA linears run block-FP8 from a sidecar next to the checkpoint
  (fp8-swapset.safetensors + fp8-swapset.json, built by
  `python -m slimserve.fp8_swapset --native <GLM-5.3-Flash> --model
  <GLM-5.3-Flash-NVFP4> --out <dir>/fp8-swapset.safetensors
  --self-quant-kda --tp-size 4`). The KDA part costs -0.014 nats mean NLL,
  inside the plan's 0.03 tolerance and stated in the notebook; the
  operator has now selected this explicit recipe. Do not silently swap
  back to BF16 KDA or change the weight quant during the campaign.
- Custom all-reduce over PCIe (VLLM_CUSTOM_AR_ALLOW_PCIE=1, cap
  VLLM_CUSTOM_AR_MAX_SIZE_MB) and NCCL P2P for the prefill reduce
  (NCCL_P2P_DISABLE=0, NCCL_P2P_LEVEL=SYS in the record env). The profile
  env is setdefault, so an operator export of NCCL_P2P_DISABLE=1 silently
  wins; the CLI plan print marks shadowed keys.
- Small-k sampler (csrc/quixicore/serving/topk_sample.cuh, top_k <= 32);
  sparse MLA decode partition + channel reducer (VLLM_MLA_SPARSE_REDUCE);
  mHC prefill partials (VLLM_DSV4_MHC_PREFILL_MIN_T); pooled indexer
  prefill (VLLM_GLM5_INDEXER_PREFILL_MATMUL); head-batched sparse MLA
  prefill (VLLM_MLA_SPARSE_PREFILL_TC).

Rejected, each with a notebook entry: forced 1-stage custom AR; the
swap-set sidecar without shared down; NCCL_PROTO=Simple; custom AR caps
of 64 / 128 MiB (superseded by P2P); the mHC last-block mode (neutral,
kept as a diagnostic); MTP k=1 and the Phase 4 prototype (above).

Current method: one factor per A/B, a predetermined number of starts and
repeats, every result retained, median and spread reported. Never restart
until a fast boot appears or exclude a run because it is slow. Startup
variability is an unresolved correctness-of-measurement issue to diagnose.
Profile attribution informs hypotheses; only matched serving measurements
establish a throughput win. The expert-bandwidth estimates below assume
near-uniform routing and are not measured DRAM traffic. The legacy 256-token
quality gate's -2.407..-2.478 band belongs only to that old workload, not the
new 4096-token explicit-ID reference. Compare matched quality cases and their
measured spread; do not apply an old absolute band to a new corpus. Prompt
scoring tests prefill, not teacher-forced decode. Keep a notebook entry for every
result including rejections. Warm the full measured workload before timed
runs, retain cold warmups separately, and do not discard measured repetitions.
Profiler captures use a 384-token round after timed runs. Attribute replays by
GPU launch correlation with benchmarks/analyze_cuda_graph_trace.py; CPU
launch timestamps can be several steps ahead. Never git stash or checkout
in the tree while a server boots or runs from it. Commit coherent checkpoints
under the maintainer's sole authorship. Repo docs
carry no machine paths; operator notes live outside the repo.


# HANDOFF — NVFP4-on-Metal campaign (updated 2026-08-25; CAMPAIGN COMPLETE through UPDATE 55 — PR #12 open, origin/main merged and re-gated bit-exact, QuixiCore-Metal port landed)

## Mission

Serve **Qwen3.8-27B from `unsloth/Qwen3.8-27B-NVFP4`** (Unsloth Dynamic
v3.0, safetensors, compressed-tensors) **on Metal (M1 Ultra)**, then
optimize it with the same rigor as the Q4_K campaign. Directive from the
boss via the user: SlimServe is **opinionated — fine-tuned for specific
quants per platform**. NVFP4 is the quant of record for this model
(~98% of BF16 quality). Kernels/plumbing written portably-minded;
Metal first. BF16 is NOT a serving target.

Additional directive (user, 2026-08-19): **use DFlash and TurboQuant
where possible.** ~~Resolved as MTP because "no DFlash drafter exists
for Qwen3.8"~~ — **that claim was FALSE when written** (never checked
online; caught by the user 2026-08-19). **DFlash 2 shipped ~2026-08-13
with a Qwen3.8-27B drafter**: `z-lab/Qwen3.8-27B-DFlash2` (mirror
`incoai/…`, Apache 2.0), 2B BF16, one 3.85 GB safetensors, block_size 8
(7 draft tokens/verify), LOSSLESS, claims 3.43x GSM8K c1. Config:
`DFlash2DraftModel` — 5 sliding-window(2048) qwen3 layers, hidden 5120 /
vocab 248320 / GQA 32-8 (all match target), non-causal block attention,
mask_token_id 248070, taps target hidden states at layers
[5,19,33,47,61], selector rank 256 top_k 16, two-tap grouped dynamic
conv (group 16). In-tree we have the DFlash **V1** stack (DFlashProposer
runner, qwen3_dflash.py, GGUF adapter/tests); V2 needs the selector,
dynamic convs, per-position candidates + path tracing, and a
`DFlash2DraftModel` registry entry. References: upstream vLLM PR #52816
and github.com/z-lab/dflash. **N5 = DFlash 2 integration (promoted,
now the active milestone); MTP (`qwen3_5_mtp`) demoted to fallback.**
**TurboQuant** applies directly — full Metal kernel port exists — and
is milestone N6.

## N14 Muse single-CB — COMPLETE, DEFAULT FLIPPED (2026-08-24 evening, UPDATEs 50-53)

- **STATUS: N14 DONE. The qwen38-nvfp4-1 profile now carries
  VLLM_QC_MUSE=1 (UPDATE 53); plain-boot validated c1 17.045 sha
  467b35c3; the box serves the new canonical. NEW PINS: c1 17.10 sha
  467b35c3 BIT-EXACT with the pre-muse pin (154.4% of the Q4_K bar,
  +4.1%) / c4 23.12 / c8 25.59 (eager fallback) / 2500x64 3.45 sha
  aa448847 (the one re-pinned leg; 30k needle EXACT). c1 now sits AT
  the N3-era modeled bandwidth ceiling (~17 tok/s). Kill switch
  VLLM_QC_MUSE=0 = UPDATE 49 behavior.**
- Machinery: muse_q38_init/layer_gdn/layer_attn/run (qc_metal_serving.mm)
  emit all 64 layers + final norm + the drafter's 5 aux taps into ONE
  command buffer; glue kernels in muse_step.metal; python wire-in
  vllm/model_executor/models/muse_q38_metal.py (lazy registration,
  eligibility = uniform pure-spec decode m<=8, serve/shadow modes with
  exact state snapshot/restore, VLLM_QC_MUSE_LAYERS cap-list bisection,
  VLLM_QC_MUSE_DEBUG=<layer> stage-dump isolation).
- Root causes found during bringup (15 gate rounds, all in
  optimization_status UPDATEs 51-52): GDN state pools are SHARED across
  layers with PER-LAYER slot windows; the 16 attention layers span 4 KV
  GROUPS with per-group block tables — muse takes per-layer/per-group
  tensor vectors for both. Bit-exactness seeds eliminated by emitting
  eager's own kernels (qc_swiglu, gemma norms) and per-op-rounding
  mirrors (sigmoid gate).
- CAUTION for future reading: an intermediate broken build measured
  "+26%" — fake speed from 12/16 attention layers reading wrong tiny
  page sets. The honest muse win at c1 is +4.0% so far; the muse-mode
  phase census (next) decides where the remaining time is.
- DSV4 anchors bit-exact through 23 consecutive re-gates today.

## Parked: RadixArk NVFP4-BF16-LMHead checkpoint (user decision 2026-08-24: stay on unsloth)

- `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` (SGLang org, pushed 2026-08-24)
  = NVFP4 MLP (group 16) + fp8 attn/GDN projections + **bf16 lm_head**
  (exists because SGLang can't load unsloth's fp8 lm_head). ModelOpt
  format, MIXED_PRECISION, per-tensor fp8 scales, W4A4 + fp8-static-KV
  recipe. Compatibility read (verified from its config.json): same arch
  (Qwen3_5ForConditionalGeneration 48+16), our NVFP4/fp8ch GEMVs apply
  as-is; gaps = ModelOpt->Metal kernel routing (fork's modelopt backend
  hard-wires Marlin), per-tensor->per-row fp8 scale broadcast, new
  profile + own pins (different weights). We'd serve W4A16/W8A16 + bf16
  KV (per the KV decision). bf16 head = +1.27 GB, ~2x lm_head pass
  bandwidth, better logits fidelity. Est. ~0.5-1.5 days bring-up.
- Sibling repo `RadixArk/Qwen3.8-27B-DSpark`: a DSpark drafter for this
  model — natural comparison vs DFlash2 if this checkpoint is ever
  brought up.
- DECISION: noted and parked; the campaign continues on
  `unsloth/Qwen3.8-27B-NVFP4`.

## Commit plan for UPDATEs 5-53 (user GO 2026-08-25; landed as this series
## after the pre-PR audit + cleanup pass, UPDATE 54)

63 dirty/untracked files on `qwen38-bringup` (last commit b1ee83671),
plus `tests/slimserve/test_profiles.py` in group 7 (the source-key rename)
and `config/vllm.py` moved to group 5 (DFlash2 scope). Series (each
buildable, library-first):

1. **qwen3.5/qwen3-next bring-up** — models/{qwen3_5,qwen3_next}.py,
   transformers_utils/configs/{qwen3_5,qwen3_5_moe,qwen3_next}.py,
   gguf_qwen35.py, gguf_adapters/{__init__,qwen3_5}.py, gguf_loader,
   gguf_config_parser, tokenizers/registry, models/{registry,config}.py,
   mamba_mixer2.py, gdn_mps_fallback.py, platforms/{metal,metal_compat},
   vllm/config/vllm.py, layernorm.py.
2. **Metal serving kernels + bindings** — all csrc/quixicore changes
   (qgemv, paged_attn_v2, gdn, rms_norm, kv_cache, turboquant,
   dflash_conv/, qk_norm_rope_gate/, muse_step.metal, tk_launch.h,
   qc_metal_serving.mm), vllm/quixicore_metal.metallib, quixicore/ops.py,
   muse_qwen38_design.md.
3. **CT Metal GEMV routing** — kernels/linear/{__init__,metal_dequant,
   nvfp4/metal,scaled_mm/metal}.py, compressed_tensors scheme edits,
   fp8_utils.py.
4. **GDN + attention + runner serving path** — qwen_gdn_linear_attn.py,
   v1 backends/{gdn_attn,metal_attn}.py, worker/gpu/{attn_utils,
   model_runner,model_states/*,sample/*}.py, metal_phaseprof.py.
5. **DFlash2 speculation** — qwen3_dflash2.py, spec_decode/dflash2/,
   spec_decode/{__init__,speculator}.py, qwen3_dflash.py.
6. **TurboQuant serving** — backends/turboquant_attn.py,
   ops/turboquant_native.py.
7. **Muse wire-in + profiles + registry** — models/muse_q38_metal.py,
   slimserve/{profiles.json,registry.py} (incl. the VLLM_QC_MUSE=1 flip).
8. **Perf notebook + docs** — perf/{optimization_status,baseline_status}.md,
   HANDOFF.md.

PR description must carry the "PR notes" section below (KV cache
intentionally unquantized). After commit: the QuixiCore-Metal twin-port
batch (kernel files byte-identical: turboquant incl. splitk/reduce,
dflash_conv, qk_norm_rope_gate, kv_cache block_mult, muse_step glue,
qgemv/gdn/paged_attn deltas).

## PR notes (must appear in the PR description at submission)

- **KV cache is intentionally NOT quantized in the canonical profile**
  (user decision, 2026-08-24). The TurboQuant k8v4 KV path
  (`qwen38-nvfp4-1-tq`) was fully built, debugged, and measured — it
  wins c4 +2.4% / c8 +1.8% / 2500x64 +1.4% with 2.4x KV capacity and
  a 262k needle-exact result — but its V-reconstruction floor
  (cos ~0.9955 vs bf16) was judged an unacceptable default quality
  trade. It ships as the opt-in long-context/capacity profile only.

## Repo state snapshot (2026-08-19; historical — see the N-milestone
## sections above for current status)

- **N2 + N3 COMPLETE (2026-08-19)**: exact-token N2 baseline pinned
  (UPDATE 14 + baseline_status snapshot: c1 7.02 sha 8c58a4c6 3/3,
  c4 14.37, c8 15.96, 2500x64 2.55 sha d0e07ddd 3/3), then the first
  CT GEMV kernels shipped (UPDATE 15): `qgemv_fp8ch` (461–650 GB/s at
  serving shapes) + `qgemv_nvfp4_planar` v6 (426/447 GB/s), routed
  M==1-only (batch loop measured BELOW dense GEMM at M=4/8 — `_mb`
  weight-stationary twins are the open c4/c8 lever). Serving: **c1
  10.10/10.14 sha 2e567ea7 (+44% over N2, 90% of Q4_K's 11.17)**,
  2500x64 2.82/2.83 sha 95adbd97, c4/c8 unchanged by design; ramp-32
  indicative 14.50. float64 oracles ALL PASS; kill-switch boot
  (VLLM_QC_FP8CH=0 + VLLM_QC_NVFP4=0) reproduces N2 BIT-EXACT.
  DSV4 anchors re-gated ALL BIT-EXACT after BOTH rebuilds
  (anchor_regate_fp8ch/ and anchor_regate_nvfp4_final/).
  **N3 mb twins SHIPPED same day (UPDATE 16)**: qgemv_fp8ch_mb +
  qgemv_nvfp4_planar_mb (column-pair grid-split; rows bit-identical
  to looped batch-1), routed even M (NVFP4 <=8, FP8 <=4 by measured
  crossover), kill switches VLLM_QC_NVFP4_MB/VLLM_QC_FP8CH_MB.
  **c4 15.64 (+8.8%), c8 16.89 (+5.4%)**; c1/2500x64 shas bit-exact;
  null-check boot reproduces UPDATE 15; kNvfp4Nib table removed;
  anchors re-gated again (anchor_regate_ct_mb/).
  **N3d fp8ch v6-decode port SHIPPED same day (UPDATE 18)**: select-
  free E4M3 half2 bit-pattern decode + 2^8 epilogue fold + vec4 X
  staging; fp8ch mb bound 4 -> 8 (crossover moved). **c1 10.35
  (+2.9%) / c4 16.28 (+4.1%) / c8 17.54 (+3.9%) — c8 PASSES the Q4_K
  bar (100.7%)**; 2500x64 + c4 shas HELD; **c1 canonical sha rolled
  2e567ea7 -> 467b35c3** (deterministic; summation-order rounding —
  fast-math reassociation the old select chain blocked; per-element
  exact, attribution in UPDATE 18); DSV4 anchors re-gated ALL
  BIT-EXACT (anchor_regate_v6port/).
  **N3e nvfp4 v7 vectorized decode SHIPPED same day (UPDATE 19)**:
  half2 constructs decode all 8 nibbles per uint + 2^22-folded
  select-free group scale; gate_up 631 / down 662 GB/s; **idealized
  all-GEMV ~26.1 ms = AT the ~27 ms campaign target — N3 GEMV work is
  CLOSED**. **c1 10.70 (95.8% of bar) / c4 16.98 (102.3% CROSSED) /
  c8 18.25 (104.8% CROSSED)**; 2500x64 2.91 sha 95adbd97 held
  bit-exact; **c1 canonical sha now fa58598b** (roll predicted in
  advance, reassociation class, UPDATE 19); DSV4 anchors re-gated
  ALL BIT-EXACT (anchor_regate_nvfp4v7/, 5th of the day).
  **N4 census + lever 1 SHIPPED same day (UPDATE 20)**: census =
  host exonerated, split mlp 33.5 / gdn 25.3 / full_attn 17.5 /
  glue 23.7; ROOT CAUSE found — head-256 full attn had NO paged
  kernel (SDPA gather every decode step). Split-K partition/reduce
  @256 shipped (one serial encoder, target 1536, P<=64,
  `max_context` op arg), crossover-routed (batch >= 2 OR ctx >=
  2048 -> kernel; batch-1 short ctx keeps SDPA — the kernel's fixed
  per-call cost loses there, measured across 3 gate rounds).
  **c1 10.72 sha fa58598b bit-exact (96.0% of bar) / c4 18.54
  (111.7% CROSSED) / c8 21.26 (122.1% CROSSED)**; 2500x64 2.914
  sha aedef4ec (kernel route); ctx-32k decode now viable (10.4
  ms/call x16). Kill switch VLLM_QC_PA256=0, tuning
  VLLM_QC_PA256_SPLITK. DSV4 anchors ALL BIT-EXACT
  (anchor_regate_pa256/, 6th of the day). The expanded
  DFlash-verify attention path (q_len>1) now routes to the kernel
  — N5 prerequisite done.
  **N4 lever 2 SHIPPED same day (UPDATE 21)**: fused add+RMSNorm
  (`add_rms_norm`, one dispatch per residual seam x128/step,
  BIT-EXACT by oracle AND by serving shas — c1 fa58598b / 2500x64
  aedef4ec both held): c4 18.87 (+1.5%, 113.4% of bar) / c8 21.43
  (+0.8%, 123.0%) / 2500x64 2.94 (+1.1%); c1 flat 10.71 (95.9%).
  SwiGLU hypothesis CLOSED — already fused (qc_swiglu). Kill switch
  VLLM_QC_ADDNORM. Anchors ALL BIT-EXACT (anchor_regate_addnorm/,
  7th of the day). OPS LESSON: gate scripts kill-first ALWAYS (a
  restored box + a clean-box-assuming gate = EADDRINUSE + a
  half-booted second EngineCore poisoning every batch run).
  **N4 lever 3 SHIPPED 2026-08-20 (UPDATE 22)**: GDN dispatch fusion —
  `gdn_fused_prepare` (conv+silu+q/k-norm+v+gate in ONE dispatch,
  reading qkvz/ba projection rows in place; pure-decode routed) +
  `gdn_gated_rmsnorm_f32` (norm off the fp32 recurrence output, z in
  place) + container restructure: 12 -> 3 non-GEMV dispatches per GDN
  layer (~430 fewer/step). Oracle 20/20 BIT-EXACT incl. conv state
  pools; serving shas HELD (c1 fa58598b / 2500x64 aedef4ec):
  **c1 11.02/10.97 (+2.7%, 98.4% of bar) / c4 19.25 (+2.0%, 115.7%)
  / c8 21.68 (+1.2%, 124.5%) / 2500x64 2.97 (+0.9%)**. Kill switches
  VLLM_QC_GDN_FUSEPREP / VLLM_QC_GDN_FUSENORM. Anchors ALL BIT-EXACT
  (anchor_regate_gdnfuse/, 8th consecutive). CALIBRATION LESSON:
  ~2.4 ms actual vs ~13 ms census-share estimate — marginal dispatch
  cost at batch 1 is ~5 us (pipeline overlap); sync-prof shares are
  upper bounds for dispatch-elimination levers.
  **N4 CLOSED / base-decode CEILING DECLARED (UPDATE 23, 2026-08-20)**:
  xctrace Metal System Trace attach on the live box (no rebuild)
  showed ~240 CBs/step (torch-MPS per-op commits, commitAndContinue
  already default; qc encode() exonerated), channel 81% busy, and 73%
  of channel time in 1-3 ms CB-granularity footprints whose encoder
  payloads are ~10x smaller — the step is CB-scheduling granularity +
  the 26 ms GEMV floor, not unfused kernels. No single-kernel lever
  remains; the only sizeable lever is the Muse whole-step single-CB
  loop (multi-day, re-ranks under speculation). Campaign totals vs N2:
  c1 +57% / c4 +34% / c8 +36%.
  **N5 DFlash2 INTEGRATION COMPLETE (UPDATEs 24-25, 2026-08-20)**: the
  full pipeline is live on Metal (drafter -> selector walk -> GDN spec
  verify with per-position state checkpointing -> rejection), oracle
  24/24 bit-exact, greedy-DETERMINISTIC serving (b46e676c 2/2), and the
  no-spec/spec divergence was proven to be an EXACT LOGPROB TIE flip
  (' Use' vs ' Report' both -1.538660) — greedy-correct, not a bug.
  Acceptance 2.93 tokens/step (essay prose, k=7). BUT spec is currently
  SLOWER (c1 9.59 vs 11.02) because the verify target forward at M=8
  costs 234 ms/step — the UPDATE 11 weight-stationary batch-GEMV
  collapse (~111 GB/s at M=8 vs 533-633 at M=1), now inherited by the
  NVFP4/fp8ch kernels, plus dequant+dense fallback at c4/c8 (M>8).
  Canonical profile stays no-spec; the twin `qwen38-nvfp4-1-df2` carries
  spec.
  **N5b COMPLETE (UPDATE 26, 2026-08-20): mv_ext batch GEMVs**
  (`qgemv_fp8ch_mv4r` R1=4 / `qgemv_nvfp4_mv4r` R1=2 — NR=4 rows/SG,
  X via L1-served device loads, the llama.cpp kernel_mul_mv_ext
  precedent) route batches 3..8 incl. odd M (kill switches
  VLLM_QC_FP8CH_MV4R / VLLM_QC_NVFP4_MV4R; batch 2 keeps mb, M>8
  dense). Oracle: batch rows BIT-IDENTICAL to looped batch-1. Gate:
  **no-spec c1 11.06 fa58598b BIT-EXACT / c4 20.68 (+7.4%, 124.3% of
  bar) / c8 23.74 (+9.5%, 136.3%) / 2500x64 2.98 aedef4ec BIT-EXACT;
  SPEC c1 11.11 b46e676c HELD (+15.9%) — spec BEATS no-spec at c1 for
  the first time**; spec 2500x64 3.154 (+5.9% vs no-spec); spec c4/c8
  still dense-bound (spec = c1 feature). Anchors ALL BIT-EXACT (10th).
  KEY RECALIBRATION: M=8 batch work is FMA-ISSUE-bound on M1
  (~2.3-3.0e12 FMA/s; simdgroup_matrix is lane math, not tensor
  cores) — the UPDATE 25 ~21 tok/s projection was wrong; kernel
  headroom at M=8 was ~1.3-1.7x and is now spent.
  **k-SWEEP DONE (UPDATE 27, 2026-08-20): k=3 WINS — spec c1 13.65
  (sha 8c58a4c6 2/2) = +23% over both k=7 and no-spec; 2500x64 3.239.**
  Mean acceptance ~3.0 REGARDLESS of k (drafter horizon ~3 tokens on
  prose) — larger k is pure wasted verify FMAs. Twin profile pinned to
  k=3.
  **ADAPTIVE SPEC PROMOTED (UPDATE 28, 2026-08-21): the canonical
  profile `qwen38-nvfp4-1` is now speculative:true with
  num_speculative_tokens_per_batch_size=[[1,1,3],[2,8,0]]** — the
  fork's dynamic-SD feature (UPDATE 27's "no disable-by-batch-size" was
  WRONG) wired into the V2 runner: sample_tokens skips the drafter
  forward when the scheduler's per-step K is 0 and reports zero-width
  drafts; plus a gdn_attn.py:253 fix (`block_table[:, 0].contiguous()`
  — the strided column view of the [batch, k+1] spec mamba block table
  crashed gdn_short_conv at batch >= 2 with zero drafts). Promoted
  gate: **c1 13.651/13.62 sha 8c58a4c6 2/2 BIT-EXACT vs the k=3 twin
  (+23.4%, 122.2% of the Q4_K bar — c1 clears the bar for the first
  time) / c4 20.104 (97.2% of no-spec) / c8 23.275/23.26 (98.0%) /
  2500x64 3.229 f497f4a9**; anchors bit-exact 11th consecutive. NO-SPEC
  kernel anchors (fa58598b / aedef4ec) reproduce via speculative:false.
  Concurrency first-shas are composition-timeline dependent under
  adaptive spec — c4/c8 are TPS gates, not sha gates. Twin -df2 now
  identical to canonical (kept as historical gate id).
  **MUSE SCOPING + GEMMA NORMS (UPDATEs 29-30, 2026-08-21): the c1-spec
  step is CPU-DISPATCH-BOUND** (~220 ms/step = ~100 GPU + ~120 host;
  xctrace 46% busy; op census 6831 torch dispatches/step; all 101
  .item()/step are CPU-cheap). Biggest glue block fixed: GemmaRMSNorm
  (128 target norms/step, torch-native since bring-up) now routes to
  gemma_rms_norm{,_add}_dyn Metal kernels (exact ir semantics; oracle:
  res_out bitwise exact, out 99.98-100%). Gate: **c1 13.90/13.91
  8c58a4c6 HELD / c4 20.61/20.66 / c8 23.844/23.843 (ABOVE old no-spec
  23.74) / 2500x64 3.248 sha ROLLED f497f4a9 -> d0e07ddd (predicted ulp
  class, 2/2)**; anchors bit-exact 12th. CALIBRATION: removing 19% of
  dispatches bought ~2% — host cost is NOT aten-count-proportional;
  fusion-by-fusion has poor marginal returns.
  **SYNC FIX + ASYNC SCHED (UPDATEs 31-32, 2026-08-21): cProfile census
  (VLLM_QC_PYPROF=1 in metal_phaseprof.py) found postprocess_state's
  boolean-mask MPS scatter draining the WHOLE GPU queue every spec step
  (67 ms/call self-time; invisible to sync-bracketed censuses; explains
  the 46%-vs-81% busy split — no-spec takes the int index_fill_
  branch). Fix = trailing dump slot + torch.where scatter (sync-free,
  identical values): +3.3% c1, ALL shas bit-exact. Reading: sync
  scheduling re-books the wait at end-of-step => async re-ranked; dflash
  added to the config async whitelist (vllm.py) + VLLM_METAL_ASYNC_
  SCHED=1 promoted into BOTH qwen profile envs: c4 +2.3% / c8 +2.1% /
  c1 flat / 2500x64 -0.7% accepted, ALL shas bit-exact. CURRENT
  CANONICAL: c1 14.27 8c58a4c6 (127.8% of bar) / c4 21.45 / c8 24.52
  (140.8%) / 2500x64 3.30 d0e07ddd. Session cumulative vs no-spec
  start: c1 +29%.**
  **SYNC-HUNT CLOSED (UPDATEs 33-35, 2026-08-21): census rounds 3-4
  named the relocating wait each time.** (33) metal_attn: eager
  seq_lens D2H at build -> lazy property + bound-mode draft SDPA (GPU
  visibility mask from seq_lens_gpu; one-block back-off): 2500x64
  +1.7%, c1 sha ROLLED 8c58a4c6 -> 467b35c3 (predicted ulp class,
  deterministic 2/2). (34) gdn_attn spec branch: 5x GPU[cpu-bool-mask]
  -> CPU nonzero + async_tensor_h2d + index_select: **c1 +3.2% to
  14.788/14.801 BIT-EXACT**. (35) gdn_attn ~335 repeat_interleave (GPU
  repeats tensor, data-dependent even with output_size) -> scatter_add
  segment-id + index_select (CPU oracle incl. zero-length requests;
  plain seg[starts]=1 is WRONG there): gated **NEUTRAL** (c1
  14.713/14.710 467b35c3 bit-exact, all legs flat) — KEPT per the
  pre-set rule, and the hunt is **PARKED**: third consecutive
  relocation; under async each removed drain banks only its overlap
  headroom, the LAST drain absorbs the queue tail. **FINAL CANONICAL:
  c1 14.80 467b35c3 (132.5% of bar) / c4 21.61 / c8 24.65 (141.5%) /
  2500x64 3.378 d0e07ddd. Session vs no-spec start: c1 +33.8%.**
  Remaining parked sync sites: gdn_attn has_initial_state
  (prefill-only), metal_attn target-prefill lazy materialization.
  **N6 gate 1 PASSED 2026-08-21 (UPDATE 36; see the N6 section below):
  `qwen38-nvfp4-1-tq` quality-gated after a six-run debug arc that
  fixed a latent GQA head-mapping bug in tq_attention_combined. TPS
  tax 6-13%, KV 2.4x smaller; canonical default unchanged.
  N8 DONE (UPDATEs 38-39; see the N8 section below): Metal TQ dequant
  continuation route (tq_decode_combined) unlocked long context —
  262k needle EXACT (75 min prefill), 2500x64 gap -3.0% at the
  canonical sha, c1 pin bit-exact, anchors 14th consecutive.
  #16b DONE (UPDATE 40; see the #16b section below): split-K TQ
  decode attention (tq_attention_splitk/reduce) — kernel 12-33x, -tq
  c1 tax -11.6% -> -0.8%, and the -tq profile now BEATS canonical at
  c4 (+2.2%) / c8 (+2.0%) / 2500x64 (+2.1%); anchors 15th
  consecutive; new pins c1 228d0bf4 / tail-2100x32 1337d2f7, 2500x64
  still the canonical d0e07ddd. FLAGGED DECISION: canonical-default
  flip to -tq (quality call — k8v4 cos ~0.9955 floor vs bf16).
  fp8-KV comparison DEPRIORITIZED: its motivation was the TQ decode
  tax, which is now -0.8% with 2.4x capacity — fp8 (2x capacity, new
  kernel surface) is dominated.
  N9 DONE (UPDATEs 41-42; see the N9 section below): the mrope
  repeat_interleave queue-drain (65.8 ms/step, 64% of the host
  profile) fixed with a static-shape token->request map — canonical
  c1 14.80 -> 16.139 sha 467b35c3 BIT-EXACT (144.5% of the Q4_K
  bar) / c4 23.16 / c8 25.62 / 2500x64 3.454 d0e07ddd; -tq c1 16.08
  sha 228d0bf4 bit-exact (-0.3%). OPS: tmp cleaner destroyed the
  harness assets mid-gate; recovered byte-exact into
  perf/results/harness_assets/ (USE THAT PATH).
  N10 DONE (UPDATE 43; see the N10 section below): steady-cache
  metadata for the mamba-hybrid path — GDN builder rebuild (8.2
  ms/step) skipped on steady uniform all-spec decode steps
  (signature + 2-copy steady_decode_update; VLLM_QC_STEADY_META=0
  null proven = UPDATE 41). Canonical c1 16.139 -> 16.334/16.346 sha
  467b35c3 BIT-EXACT (+1.2%, 146.3% of the Q4_K bar); 2500x64
  d0e07ddd + -tq c1 228d0bf4 bit-exact (+0.8%); c4/c8 noise-flat;
  both needles exact. Two engine-killing bugs found on the way (see
  section): upstream eligibility deref'd is_prefilling=None, and the
  runner's per-step-fresh seq_lens_cpu_upper_bound froze in the
  cached cm — deterministic sha roll from silently truncated decode
  attention.
  N11a DONE (UPDATE 44; see the N11a section below): fused DFlash2
  grouped-conv kernel (qc_dflash_conv, ~200 drafter encodes/step
  collapsed to 20; kernel parity 37/37 BIT-EXACT vs eager so drafts
  are bit-identical). Canonical c1 16.334 -> 16.428/16.438 sha
  467b35c3 BIT-EXACT (+0.6%, 147.1% of the Q4_K bar); 2500x64
  d0e07ddd bit-exact; c4 23.11 / c8 25.62 noise-flat; DSV4 anchors
  bit-exact 17th consecutive after the metallib+.so rebuild;
  VLLM_QC_DFLASH_CONV=0 null proven = UPDATE 43. (Note: drafter_propose
  nests inside the sample_tokens bracket — 32.7 = 24.4 drafter + 6.5
  reject + glue; no unattributed sampling cost.)
  N11b DONE (UPDATE 45): pure-prefill causal SDPA reads the CPU bound
  (row-exact for all-prefill batches) instead of the queue-draining
  seq_lens D2H (census: 3.43 s / 12 calls, multi-chunk-prefill
  concentrated). VLLM_QC_SDPA_PREFILL_BOUND=0 null proven = U44; DSV4
  8tok + c1 + 2500x64 pins all bit-exact; leg TPS unchanged (+0.3%
  2500x64 — the +2-7% prediction missed, single-chunk prompt); the
  win books as long-context TTFT (multi-chunk prefills lose a
  hundreds-of-ms host stall per chunk).
  UPDATE 46 (-tq re-measure, all legs, U43-U45 stack): c1 16.18-16.24
  sha 228d0bf4 BIT-EXACT (-1.2% vs canonical) / 2500x64 3.51
  d0e07ddd (+1.4%) / c4 23.6-23.7 (+2.4%) / c8 26.05-26.09 (+1.8%)
  / needle exact — -tq beats canonical everywhere above c1 with 2.4x
  KV; flip decision data fully current (quality call). NEXT candidates:
  N11c TRIED AND REJECTED (UPDATE 47): drafter block attention through
  the expanded paged kernel lost on every leg (c1 -7.7%, acceptance
  1.59 -> 1.54 accepted/draft + kernel slower than one-request SDPA
  at draft shapes; 2500x64 forked trajectory). Quarantined opt-in
  VLLM_QC_DRAFT_BLOCK_PA=1. KEY LESSON: drafter math is NOT
  reduction-order-tolerant — the 16-way candidate selector amplifies
  ULP noise into acceptance loss; only bit-exact or measured-neutral
  drafter changes are safe.
  N12 TRIED AND REJECTED/PARKED (UPDATE 48): fused qk-norm-rope-gate
  kernel (qc_qk_norm_rope_gate, 1 dispatch per attn layer, bf16
  parity 8/8 at small T) forked the canonical trajectory at prefill —
  torch-MPS eager numerics are SIZE-DEPENDENT (~5 ppm single-ulp at
  T=1000, exact at decode shapes); win was c4/c8 ~+1%, not worth the
  full re-pin. Opt-in VLLM_QC_QKROPE=1; anchors bit-exact 18th.
  KEY LESSON: validate parity at SERVING shapes incl. prefill T.
  N13 DONE (UPDATE 49): kv_cache_scatter bound + routed (block_mult=2
  page-local; 5 ops -> 1 per attn layer, ~80 encodes/step) — all pins
  + anchors (19th) bit-exact, TPS flat within noise, retained as
  dispatch hygiene. MUSE SCOPING (this session): the muse_step
  machinery (muse_glimmer.py + muse_step_init/layer/run — whole-model
  single-encoder emit loop) exists for the DENSE Muse-Glimmer arch;
  adapting to qwen38 = emit variants for qgemv fp8ch/nvfp4 (mb/mv4r),
  gdn_fused_prepare/recur_spec/norm, PA256/splitk, residual adds —
  target-forward-only scope (sampling/drafter stay eager), est. 2-4
  focused days, AND it re-pins every sha by construction — a
  trajectory-re-pin decision FLAGGED FOR THE USER (same class as the
  -tq default flip). Encode-crumb tier is now exhausted: remaining
  levers are Muse (user-gated), drafter linear batching (thin), and
  the parked-dangerous items.
  DECISION RESOLVED (2026-08-24, user): -tq default flip REJECTED —
  the KV cache stays unquantized (bf16) for canonical. Quality call:
  the k8v4 V-reconstruction floor (cos ~0.9955) is not acceptable as
  the default, and the c4/c8/longctx wins don't override it. The -tq
  profile REMAINS registered as the opt-in long-context/capacity
  profile. **PR NOTE (user directive): this decision + rationale MUST
  appear in the PR description when UPDATEs 5-49 are submitted** —
  see "PR notes" below.
  DECISION RESOLVED (2026-08-24, user): Muse single-CB = **GO**. The
  user accepted the trajectory re-pin ("we've done it before" — the
  muse_glimmer precedent). Plan: build opt-in-gated so canonical
  stays bit-exact during bringup; full quality revalidation (needle
  262k + long decodes + acceptance before/after) then NEW pins at
  flip time. Commit of UPDATEs 5-49 still awaits the user's word and
  can land at any point before the flip.
  NEXT: N14 Muse single-CB bringup (ACTIVE);
  commit of UPDATEs 5-49 (user-gated). Mechanical fallback:
  drafter linear batching (~58 unquantized MPSGraph linears/step);
  gdn NSG-multi-row (demoted — dispatch amortization, small);
  twin-port batch to QuixiCore-Metal (commit-gated, now incl.
  tq_attention_splitk/reduce + dflash_conv). Parked stacks: mrope torch-native
  (16 full-attn layers), metal_attn index_put_ KV updates, ~80 MPSGraph
  unquantized linears (drafter + candidates), Muse single-CB step loop
  (re-rank post-N6), uniform-decode classification for zero-draft
  batches.
  OPS LESSON (UPDATE 15): a TERM'd server can hang in shutdown holding
  ~101 GB wired while ps-based checks false-negative — verify kills BY
  PID and check `vm_stat` wired (~2–3 GB idle) before any measurement.
- **N1 COMPLETE (2026-08-19)**: `qwen38-nvfp4-1` serves on :8000 (user
  had the Q4_K server taken down first — one server at a time on this
  box). Full record: perf/optimization_status.md **UPDATE 13**; raw
  gates in `perf/results/2026-08-19/n1_nvfp4_bringup/`. Gates: helper
  dequant bit-exact (CPU+MPS, full E4M3 range); Gate A structural (11
  asserts incl. **lm_head = W8A16Fp8**, silent-bf16 trap did not fire)
  + 6 load-path weight comparisons ALL BIT-EXACT vs CPU oracle of raw
  checkpoint bytes; Gate B live-server ramp + 3340-tok needle exact +
  greedy probes. Indicative (NOT harness): c1 decode **8.83 tok/s** on
  the bring-up path (dequant-once to bf16 at load, ~40 GB resident,
  F.linear apply; `VLLM_METAL_CT_DEQUANT=call` = per-call low-mem
  fallback). eos resolved: generation_config = [248046, 248044], chat
  stops on 248046 like the GGUF campaign.
- Three NEW blockers found at N1 (beyond the N0 list, all fixed):
  fork-wide `KernelConfig.linear/moe_backend="aiter"` default leaks to
  every platform (Metal check_and_update_config resets to auto);
  `RopeState.prepare_positions` is Triton and this config has
  uses_mrope=True (torch replacement in metal_compat); checkpoint
  `kv_cache_scheme` fp8 + auto flips KV to fp8 (attention.py:299) with
  no Metal dense/GQA fp8-KV path (profile pins kv_cache_dtype=bfloat16
  explicitly; fp8/TurboQuant KV is N6's call).
- `qwen38-1` (Q4_K_M GGUF) profile intact, not serving. Fresh-boot
  sanity was c1 11.195 tok/s sha 36ed113a. Stays the fallback Metal
  path until the decommission gate (**N7**); restore via
  `restore_qwen_m4c.sh` (session-a99b scratchpad).
- Branch `qwen38-bringup`. 39 uncommitted entries in SlimServe (all
  N1–N3 edits) + 3 in ~/Code/QuixiCore-Metal (byte-identical kernel
  twins; the N3 fp8ch/nvfp4 kernels still need their QuixiCore-Metal
  twin port once gated).
- Q4_K campaign records: `perf/optimization_status.md` UPDATEs 1–12,
  `perf/baseline_status.md` tail (M4b + M4c). Canonical Q4_K shas:
  c1 36ed113a, 2500x64 268721b3, ramp-32 0f9506fc. DSV4 anchors
  re-gated bit-exact twice on 2026-08-18
  (latest: `perf/results/2026-08-18/anchor_regate_q4k_mb/`).
- Q4_K decode budget (M4b census): c1 step 65.5 ms = 35.1 ms idealized
  GEMV + ~31 ms non-GEMV. Box empirical bandwidth roofline ~700–740
  GB/s; best Qwen kernel 633 GB/s (q4_K gate_up M=1).

## N0 discovery results (verified 2026-08-18/19)

Raw artifacts (`n0_checkpoint_findings.md`, `nvfp4_config.json`,
`nvfp4_index.json`, `nvfp4_shapes.json`) lived in a session scratchpad
that is not preserved; the durable findings are all inlined below.

### Checkpoint anatomy — MIXED FP8 + NVFP4 (the headline)

- HF sha 7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108. 23.44 GB total:
  `model.safetensors` **22.57 GB single file** (loader must not assume
  shards) + `model_mtp.safetensors` 849 MB. Arch
  `Qwen3_5ForConditionalGeneration` (VL; vision tower bf16),
  compressed-tensors `format: mixed-precision`, two config groups.
- **group_1 NVFP4** (W4A4 g16 in checkpoint; we run W4A16):
  `mlp.{gate,up,down}_proj` **layers 0–55 only**. Tensors:
  `weight_packed` U8 [N, K/2]; `weight_scale` F8_E4M3 [N, K/16]
  (gate/up [17408,320], down [5120,1088]); `weight_global_scale` F32 [1]
  — **NOTE the name: not `weight_scale_2`**; `input_global_scale` F32
  (ignored on Metal).
- **group_0 FP8 E4M3 per-channel** (W8A8-dynamic in checkpoint; we run
  W8A16): all 16 `self_attn` q[12288,5120] (attn_output_gate doubles q)
  / k[1024,5120] / v[1024,5120] / o[5120,6144]; all 48 `linear_attn`
  in_proj_qkv[10240,5120] / in_proj_z[6144,5120] / out_proj[5120,6144];
  **lm_head [248320,5120]**; `mlp` layers 56–63. Format: `weight`
  F8_E4M3 [N,K] + `weight_scale` **BF16 [N,1]**.
- **bf16 (ignore list)**: embed_tokens, vision tower, GDN internals
  (conv1d, A_log, dt_bias, in_proj_a/b [48,5120], norms), all
  layernorms, and the **entire `mtp.*`** (separate file).
- `kv_cache_scheme`: fp8 static per-tensor; `k_scale`/`v_scale` BF16
  scalars per attn layer — feeds N6.
- Config quirks: eos **248044** per config.json (previous handoff said
  248046 — recheck at N1); mrope_interleaved, partial_rotary 0.25,
  full_attention_interval 4; 64 layers = 48 GDN + 16 attn confirmed.
- **Decode-token read budget**: 56 NVFP4 MLP layers 8.42 GB + 8 FP8 MLP
  2.14 + 16 attn 1.68 + 48 GDN 5.54 + lm_head 1.27 ≈ **19.1 GB/tok**
  (**FP8 side 10.6 GB > NVFP4 side 8.4 GB**). Q4_K reads ~16.5 GB/tok
  → NVFP4 c1 bandwidth ceiling is BELOW Q4_K's. At the ~700 GB/s box
  roofline: GEMV floor ~27 ms + today's ~31 ms non-GEMV ⇒ ~17 tok/s
  hard cap. **Beating Q4_K at c1 requires #16 (non-GEMV)**; NVFP4's
  format wins live in batch/prefill ALU (cheap decode) and MTP.
- lm_head risk RESOLVED: FP8, 1.27 GB/tok ≈ today's Q6_K 1.04 GB.

### In-tree assets (better than the prior handoff knew)

- metallib ALREADY ships: `nvfp4` format struct
  (`dequant.metal:431` — interleaved 9-byte {scale, qs[8]}; the
  checkpoint is PLANAR → layout mismatch, do not use as-is),
  `qgemv_nvfp4`(+bf16/moe) pipelines (`qgemv.metal:1622`),
  `qgemm_nvfp4` / `qgemm_frag_nvfp4` (`qgemm.metal:71,223`), and an
  nvfp4 lm_head sampler family that is **DEAD CODE** (zero host
  bindings anywhere — do not count on it).
- `fp8_raw` struct (`dequant.metal:517`) is **byte-identical** to the
  checkpoint's planar [N,K] e4m3 weight. `tk_e4m3_decode`
  (`dequant.metal:271`): 3-op arithmetic decode, subnormal-exact under
  offline-compile FTZ.
- Per-channel-scale epilogue precedents: `qgemv_w8a8`
  (`qgemv_int.metal:12` — w_scale[n] epilogue, 2 rows/simdgroup;
  launcher-only, unbound) and `qgemm_fp8_scaled` (`qgemm.metal:340` —
  rank-1 epilogue; drop a_scale for W8A16 prefill).
- **TurboQuant Metal port is complete**: `turboquant.metal`
  encode/decode + combined encode/attention serving kernels,
  instantiated for head sizes **64/128/256/512** × f32/f16/bf16;
  native launchers `turboquant_native.py`; `TurboQuantAttentionBackend`
  selectable on Metal (`metal.py:213`). Metal mileage today = DSV4
  drafter KV only; main-KV on the hybrid V2 runner is new integration.

### Bring-up blockers, in hit order (all confirmed with file:line)

1. `vllm/platforms/metal.py:78` `supported_quantization = ["gguf"]` —
   add `"compressed-tensors"` (enforced via `config/model.py:1172`).
2. `vllm/model_executor/kernels/linear/__init__.py`: no METAL key in
   `_POSSIBLE_NVFP4_KERNELS` (:446) nor `_POSSIBLE_WFP8A16_KERNELS`
   (:379); and :922 force-selects Marlin when `use_a16` — gate that on
   is_cuda.
3. fp8 allocations raise on MPS (`Undefined type Float8_e4m3fn`,
   verified torch 2.13.0): `compressed_tensors_w4a4_nvfp4.py:73`
   (scale [N,K/16]) and `fp8_utils.py:1255`
   `create_fp8_weight_parameter` (the FULL [N,K] weight). Allocate
   uint8, `.view(fp8)` lazily (view works on MPS; `.to()` does not).
   LUT pattern for python-side decode:
   `vllm/models/deepseek_v4/metal_indexer.py:46` `_e4m3_lut`.
4. Scheme selection: FP8 layers **auto-select**
   `CompressedTensorsW8A16Fp8` (Metal's fake sm80 fails the ≥89 W8A8
   gate, `compressed_tensors.py:818-837`) — free and correct. NVFP4
   lands on W4A4 (`use_a16=False`, checkpoint has input scales,
   `:734-743`) — needs a Metal override to W4A16 semantics (or the
   Metal kernel simply ignores input scales).
5. `compressed_tensors_w8a16_fp8.py:145` transposes weight to (K,N);
   our qgemv family wants (N,K) row-major and the host binding asserts
   contiguity — intercept before the `.t()` on Metal.
6. TRAP: lm_head scheme exceptions are swallowed
   (`compressed_tensors.py:183`) → silent fallback to an UNQUANTIZED
   2.5 GB bf16 lm_head. Assert FP8 lm_head at N1. Never set
   `head_dtype` on Metal (`logits_processor.py:110` raises; `:132`
   materializes an fp32 copy).
7. Pure-torch oracle pieces (`nvfp4_emulation_utils.py`):
   `break_fp4_bytes` is clean; `dequantize_to_dtype:391` and
   `ref_nvfp4_quant:435` hit the fp8 `.to()` break — patch with the
   u8 LUT. Triton decorators at module import — verify the stub shim
   on macOS.

### Landscape (who else runs NVFP4 + the technique haul)

- Competition on Metal: **only MLX** (`fp_qmv_fast`; the vllm-metal
  project serves NVFP4-mlx checkpoints on it). llama.cpp has
  GGML_TYPE_NVFP4 but its Metal backend declines the type (CPU
  fallback), and its GGUF conversion drops the fp32 global scale (ours
  is more faithful). "Faster than anyone" = **beat MLX** on this
  checkpoint; llama.cpp Q4_K c1 20.9 remains the platform ceiling ref.
- E2M1→fp16 decode: MLX 3-op branchless
  `as_type<half>((n&7)<<9) * 2^14` + sign select (subnormal 0.5 exact);
  Marlin gets 2 ops per 2 values after load-time nibble pre-positioning
  at halfword tops. E4M3→fp16: `(s&127)<<7` then ×2^8, **exact**;
  decode scales as UNSIGNED (bit7 never set in valid checkpoints).
- **E2M1×E4M3 product is exactly representable in fp16** → proven
  structure (ggml mxfp4 Metal kernel ⊕ MLX fp_qmv): 1 thread = one
  16-value group per iteration, unscaled FMA tree, ONE scale multiply
  per 16 MACs hoisted onto the block partial sum, fp32 cross-block
  accumulation, simd_sum, global scale once per output element.
- **Fold every power-of-2 rebias into one fp32 epilogue constant**
  (global × 2^22 if both decodes are left raw) — the Marlin pattern.
- Traps seen in the wild: e4m3 SUBNORMAL scales occur in real
  checkpoints (caused llama.cpp's PPL=5.8M bug — decoder must handle);
  global-scale inversion conventions differ per backend (the CT scheme
  stores `layer.weight_global_scale = 1/max` — get it right once at
  load); TP sharding of scales along K (vLLM #41511).
- Batch tiers (ggml precedent, matches our UPDATE 11 cost model):
  mul_mv (M=1) / ext r1 (M=2–5) / simdgroup-matrix GEMM above.

## Milestones (revised at N0 close + DFlash/TurboQuant directive)

**N1 — Bring-up (correctness first, slow OK): ✅ DONE 2026-08-19 (see UPDATE 13)**
- Plumbing unblocks 1–7 above (explain-before-edit each).
- `MetalNvFp4LinearKernel` + `MetalWFp8A16LinearKernel` with bring-up
  apply: dequant-once-to-bf16 at load (~45 GB unified RAM — bring-up
  only) or per-call dequant; W4A16/W8A16 semantics. Both CT and
  ModelOpt configs normalize to the same attr names post-load
  (`weight`, `weight_scale`, `weight_global_scale`) — write the Metal
  kernels against those and serve both for free.
- Model/loader: compressed-tensors safetensors on the qwen3_5 chain
  (single-file; `mtp.*` index entries resolve to
  `model_mtp.safetensors` — keep out of base load). Fused-shard scheme
  checks are safe here (q/k/v all group_0; gate/up both group_1) but
  `should_ignore_layer` RAISES on mixed schemes across fused shards —
  watch it.
- Register source `qwen38-27b-nvfp4` + profile `qwen38-nvfp4-1`
  (platforms ["metal"], engine cloned from `qwen38-1`); slimserve owns
  the 23.4 GB fetch.
- Gates: greedy parity vs the pure-torch emulation oracle + retrieval
  needle test (UPDATE 3/4 methodology). Assert lm_head came out FP8
  (blocker-list trap 6).

**N2 — Exact-token baseline: ✅ DONE 2026-08-19 (UPDATE 14)**
- `benchmark_dsv4_exact.py` + m2_source.txt, c1/c4/c8 1000x256 +
  2500x64, pin new canonical shas + wall clocks,
  `perf/baseline_status.md` snapshot. Bar: Q4_K M4c
  (c1 11.17 / c4 16.60 / c8 17.42); llama.cpp Q4_K c1 20.9 ceiling ref.
- Pinned: c1 7.02 (sha 8c58a4c6 3/3) / c4 14.37 / c8 15.96 / 2500x64
  2.55 (sha d0e07ddd 3/3). c4/c8 request-0 shas are tie-carriers, NOT
  anchors.

**N3 — Metal GEMV kernels (FP8 FIRST, then NVFP4): ✅ kernels SHIPPED 2026-08-19 (UPDATE 15); `_mb` twins + geometry OPEN**
- **3a FP8-channel GEMV + `_mb` twin (do first)**: carries 10.6 GB/tok
  — more traffic than the NVFP4 side — with zero unpacking; must hit
  near-roofline. Planar e4m3 decode (`tk_e4m3_decode` / fp8_raw idea)
  × bf16 activations, per-row `w_scale[n]` epilogue (`qgemv_w8a8`
  structure, drop a_scale, fp16 FMA not idot). Covers attn qkvo, GDN
  qkv/z/out, mlp 56–63, and lm_head (grid-split for 248,320 rows).
- **3b NVFP4 planar GEMV + `_mb` twin**: NEW kernel — do NOT repack to
  the interleaved struct. Buffers D, Wq, Wsc, X + N, K, global-scale
  (setBytes); fresh lane geometry (1 thread = one g16 block/iter);
  scale-on-partial-sum; fp32 accumulation; folded fp32 epilogue.
  Microbench decode variants: MLX 3-op vs Marlin-repack 2-op vs 8 KB
  product-LUT (expect ALU to win on M1; repack is load-time-free).
  Keep the UPDATE 10/11 lessons: 4+ blocks in flight, no weight
  materialization, grid.y column pairs for batch; M-wide unrolled
  bodies i-cache-thrash; TG-mem accumulators only reach parity.
- Gates (all mandatory): float64 oracle (adapt `q4k_oracle*.py`),
  serving-shape microbench (adapt `mmvq_bench*.py`), serving A/B with
  kill-switch envs, DSV4 anchor re-gate on ANY metallib/.so rebuild.

**N4 — Prefill + the transferred levers:**
- Prefill dequant-then-matmul first; then evaluate `qgemm_frag_nvfp4`
  (needs a planar variant) and `qgemm_fp8_scaled` (drop a_scale) for
  fused. Remember DSV4 PREFILL v10: host aten fluff around a GEMM can
  exceed the kernel time — check the wrapper before the kernel.
- **#16 non-GEMV ~31 ms/step is now the c1-decisive lever**: gdn
  NSG-multi-row (llama.cpp ggml-metal.metal:2704), ba-GEMV fusion,
  paged-attn head-256 gap (see N6 overlap), norm/glue. #17 batch ALU:
  simdgroup_matrix at M=8 (may partially dissolve — NVFP4 decode is
  cheaper per column).

**N5 — DFlash 2 speculation (INTEGRATION COMPLETE 2026-08-20, UPDATEs
24-25; serving default deferred behind N5b):**
- SHIPPED: full PR #52816 port (qwen3_dflash2.py, dflash2/ speculator
  with torch-native MPS walk, registry/config/V2 forcing,
  draft_logits_spec hook; the "reconcile upstream hooks first" step
  dissolved — the hooks are the PR's own additions); compressed-tensors
  lm_head accepted in compute_candidates (drafter ships no embed/lm_head
  and ties to the target's fp8ch head); MPS GDN spec path (UPDATE 24:
  gdn_fused_prepare spec-rewind mode + gdn_recur_spec per-position
  checkpointing + _forward_core_metal_spec, oracle 24/24 bit-exact,
  non-spec re-gates all bit-exact incl. 9th consecutive DSV4 anchors);
  three MPS fixes in the DFlash1 context-KV precompute (torch-native
  rms_norm x2, RoPE forward_native, metal_attn.do_kv_cache_update).
- Profile: twin `qwen38-nvfp4-1-df2` (speculator on source
  qwen38-27b-nvfp4, z-lab/Qwen3.8-27B-DFlash2 pinned 50307d4c,
  method=dflash k=7; drafter resolves from ~/models/Qwen3.8-27B-DFlash2
  = symlinked HF snapshot). Canonical qwen38-nvfp4-1 stays no-spec.
- GATE RESULTS (UPDATE 25): greedy spec output DETERMINISTIC (c1 sha
  b46e676c 2/2, same sha at c4 request-0); divergence from the no-spec
  fa58598b proven to be an EXACT logprob tie (' Use' / ' Report' both
  -1.538660) flipped by batched-verify kernel numerics — greedy-correct,
  gate redefined as determinism + tie-flip-only + acceptance +
  throughput. Acceptance 2.93 tok/step at k=7 (essay prose). Throughput
  BELOW no-spec (c1 9.59 vs 11.02; c4/c8 worse; 2500x64 +2.7%): phase
  census puts target_forward at 234 ms/step (M=8) — the UPDATE 11
  weight-stationary batch-GEMV collapse, inherited by NVFP4/fp8ch, plus
  the M>8 dequant+dense fallback at c4/c8.
- **N5b DONE (UPDATE 26, 2026-08-20): mv_ext batch GEMVs** for M in
  [3, 8] (fp8ch R1=4 / nvfp4 R1=2; odd M native — the compute_candidates
  M=R*7 lm_head fix came free; simdgroup_matrix GEMM investigated and
  parked: M1's MAC phase is issue-bound at ~2.3e12 FMA/s, the "M=8
  collapse" was mostly FMA physics and the ~21 tok/s projection was
  wrong). Gate: spec c1 9.59 -> 11.11 (b46e676c HELD) — ABOVE no-spec
  11.06 for the first time; no-spec c4 +7.4% / c8 +9.5%. Remaining from
  the old list: revisit max_num_batched_tokens (spec drops
  max_num_scheduled_tokens to 2000); simdgroup GEMM for M in [9, 32]
  only if c4/c8 spec ever matters.
- k-sweep DONE (UPDATE 27): k=3 13.65 / k=5 11.70 / k=7 11.11 spec c1;
  acceptance ~3.0 at every k (drafter horizon, not verify quality) —
  twin pinned to k=3; promotion to canonical blocked only on a
  batch-adaptive spec gate. Probabilistic drafting: implemented
  torch-native on MPS (no Philox parity — same stance as gumbel_sample);
  greedy is the serving mode.
- Fallback (demoted): the model's own MTP head (`qwen3_5_mtp`,
  `model_mtp.safetensors` 849 MB bf16, num_speculative_tokens 2 per
  the GGUF-source precedent) — only if DFlash 2 underdelivers post-N5b.

**N6 — TurboQuant KV: GATE 1 PASSED 2026-08-21 (UPDATE 36).**
- Profile `qwen38-nvfp4-1-tq` = canonical + TURBOQUANT backend +
  turboquant_k8v4 on the target's 16 full-attn layers AND the DFlash2
  drafter (`speculative_overrides` pins BOTH drafter fields — an unset
  drafter dtype inherits engine-global TQ with a metal_attn backend =
  boot reshape crash; a bf16 drafter cannot pad into the TQ max page).
- Six-run debugging arc (notebook UPDATE 36) found the Metal TQ kernels
  had NEVER been production-exercised ("DSV4 drafter TQ" is a
  CUDA/ROCm-profile fact; the Metal .so binds ONLY encode +
  attention_metal). Fixes: .contiguous() V guard in _store_kv (fused-
  QKV slice), Metal routes ALL continuation prefill through the
  synthetic-decode path (no dequant symbol on Metal), and THE BUG —
  `tq_attention_combined` mapped q-head->kv-head as head % Hkv where
  vLLM's GQA convention is head / (Hq/Hkv). One-line fix in
  turboquant.metal; parity cos 0.9955 at hs64/128/256 ctx 1-2500
  (scratchpad 04ba9b90 tq_parity.py); DSV4 anchors BIT-EXACT 13th
  consecutive post-rebuild (anchor_regate_tqfix/). PENDING: twin-port
  turboquant.metal to QuixiCore-Metal with the qgemv/rms_norm batch.
- Gate: needle "739214" exact at 4.5k multi-chunk prefill, det 2/2
  (c1 0c92d6e3, 2500x64 4494d8e6 profile-local pins), TPS c1 13.12
  (-11.3%) / 2500x64 2.93 (-13.4%) / c4 19.73 (-8.7%) / c8 23.17
  (-6.0%). KV 4096 -> 1728 B/token (~2.4x, target + drafter).
  CANONICAL DEFAULT UNCHANGED (TPS still rules at 32k).
- Remaining N6 options: long-ctx legs + max-admission sizing where the
  2.4x pays (262k/1M); fp8-KV comparison (checkpoint ships static k/v
  scales) if TQ tax matters; perf lever to close the 2500x64 gap =
  native TQ dequant kernel or a slots-matrix-free continuation
  (current Metal continuation builds an O(q_len x max_ctx) int32 slots
  matrix per layer call in turboquant_native.py). #16b
  (tq_attention_combined_hs256 replacing the PA256/SDPA head-256
  route) still unevaluated.

**N7 — Opinionated cleanup: DECOMMISSION DONE 2026-08-21 (UPDATE 37).**
- Criteria met: every leg beats the Q4_K bar (c1 132.5% / c4 130.1% /
  c8 141.5% / 2500x64 103.3%) and canonical needle recall is exact at
  4.5k multi-chunk. Profile `qwen38-1` + source `qwen38-27b` (GGUF)
  removed from slimserve/profiles.json (zero dangling refs; canonical
  dry-run clean). q4_K kernels STAY in the shared QuixiCore-Metal
  library per the boss directive; local GGUF files untouched.
- Remaining tail: 262k/1M context sizing — informed by UPDATE 36's
  capacity data (canonical bf16 KV holds 158,038 tokens under the
  16 GiB budget and cannot fit one 262k request; the TQ profile holds
  292,882 and can). Long-context serving likely rides the -tq profile
  with max_model_len raised.

**N8 — 262k long-ctx sizing + Metal TQ dequant route (2026-08-21, UPDATE 38;
UPDATE 39 pending the in-flight gate).**
- Phase 1 (UPDATE 38): `--ctx 262144` boots and ADMITS on -tq (native
  max_position_embeddings 262144, no rope scaling; `slimserve --ctx`
  replaces max_model_len). But prefill through the synthetic-decode
  continuation is O(ctx^2) with a ~24x-redundant constant: measured 16k
  prefill 406 s; model T(C) ~= 6.8ms*C + 2.3e-6*C^2 -> 262k ~= 22 h.
  BLOCKED -> ported the CUDA continuation design to Metal.
- Implementation (worktree, uncommitted): `tq_decode_combined` kernel in
  turboquant.metal (combined-slot dequant, decode math lifted from the
  split tq_decode + tq_attention_combined; 12 instantiations);
  `launch_tq_decode_combined` in tk_launch.h; `turboquant_dequant_kv_metal`
  binding in qc_metal_serving.mm + ops.py wrapper; turboquant_attn.py:
  restored the >128 continuation threshold on Metal, Metal branch in
  `_continuation_prefill` (1-D slot list from the block table — no
  (q_len, ctx) slots matrix), Metal guard on the CUDA Pi inverse-rotation
  (Metal K is never rotated), `_metal_tiled_continuation_attention`
  (KV-tiled fp32 online softmax, tile 4096, engaged when seq_len > 8192 —
  masked SDPA would materialize 0.5-26 GB per layer call at 262k), and
  the continuation workspace reserve re-enabled on Metal. metallib
  13,094,352 B + .so rebuilt; parity PASS (scratchpad 04ba9b90
  tq_dequant_parity.py): dequant K cos 0.99998, V cos 0.9955 (= the known
  4-bit error), tiled-vs-SDPA same-KV cos 0.99999, raw-KV oracle 0.996.
- GATED 2026-08-21 17:37 (UPDATE 39), ALL PREDICTIONS HELD:
  DSV4 anchors BIT-EXACT (14th consecutive). TQ short-ctx: needle
  739214 exact; c1 13.078/13.082 sha 0c92d6e3 BIT-EXACT vs the
  UPDATE 36 pin (no continuation on that leg — change provably
  isolated); 2500x64 3.278/3.268 sha d0e07ddd = the CANONICAL sha
  (greedy tokens now match canonical on this leg), gap -13.4% ->
  -3.0%; c4 -7.2%; c8 -2.1%. 262k ladder ALL EXACT: 16k=111 s (3.7x
  vs synthetic-decode 406 s), 65k=577 s, 131k=1492 s, 262k=4493 s at
  259,888 prompt tokens. Fitted prefill T(C) ~= 6.4e-3*C + 3.9e-8*C^2
  (131k predicted 1489 vs 1492 measured). 1M PARKED (out of native
  rope spec 262144 + needs ~40 GiB KV budget). Raw:
  perf/results/2026-08-21/{anchor_regate_dequant, tq_dequant_gate,
  tq_longctx2}/. Canonical default UNCHANGED at 32k.

**#16b — split-K TQ decode attention (2026-08-21, UPDATE 40): DONE, the
-tq profile now beats canonical on 3 of 4 legs.**
- The #16b evaluation (tq256_bench.py, clean box) REJECTED the item as
  posed — monolithic `tq_attention_combined` is 12-19x SLOWER than
  PA256 — but root-caused the entire -tq c1 -11.6% tax: one
  threadgroup per (q-head, batch), two TG barriers per token, ~7 GB/s;
  batch 4 costs the same as batch 1 (occupancy-bound proof). Slots
  build and table width exonerated.
- Implementation (worktree, uncommitted): `tq_attention_splitk` +
  `tq_attention_reduce` in turboquant.metal (PA256-style (H,B,P) grid;
  one SIMDGROUP per token — lane covers HS/32 elements, dot = one
  simd_sum, ZERO token-loop barriers; FWHT-domain V partials, one
  staged cross-simdgroup merge; reduce folds sink + single inverse
  FWHT; 24 instantiations); launchers in tk_launch.h;
  `turboquant_attention_splitk_metal` host op (PA256-style sizing,
  target 1536 via VLLM_QC_TQ_SPLITK_TARGET, ring_out tq_sk_*, one
  serial-dispatch encoder, max_context arg) + ops.py wrapper; routing
  in turboquant_native.py Metal branch (head_size==256 AND no sliding
  window only — hs64/128 DSV4 drafter stays monolithic BY CONSTRUCTION
  so the anchors hold; VLLM_QC_TQ_SPLITK=0 kill-switch); max_context
  plumbed host-side from seq_lens_cpu at _decode_attention +
  _uniform_query_attention (spec-verify hot path) + the synthetic-
  decode continuation site. metallib 13,327,040 B + .so 960,272 B.
- Parity 10/10 PASS (tq_splitk_parity.py): vs monolithic rel
  1e-9..3e-6 (summation order only); quant floor vs SDPA 0.9954-0.9958
  unchanged; ragged/ctx=1/ctx=5/48-blocks/sinks/hs128 covered.
- Bench: kernel 12-33x; TQ decode now FASTER than canonical bf16
  PA256 everywhere (ctx1000 b4: 0.229 vs 0.380 ms; 32k b4: 6.15 vs
  12.38 — 432 vs 1024 B/token at ~220 GB/s effective).
- GATED 2026-08-21 19:23, ALL PREDICTIONS HELD: DSV4 anchors BIT-EXACT
  (15th consecutive, anchor_regate_tqsk/). -tq: needle 739214 exact;
  c1 14.688/14.685 sha 228d0bf4 2/2 (rolled as predicted; -11.6% tax
  -> -0.8%); 2500x64 3.459/3.448 sha d0e07ddd 2/2 (CANONICAL sha
  HELD, +2.1% ABOVE canonical); NEW tail 2100x32 leg 2.174/2.171 sha
  1337d2f7 2/2 (synthetic-decode continuation through splitk); c4
  22.11 (+2.2% above canonical); c8 25.15 (+2.0%); 16k needle exact
  at the 262k config, 110 s wall. Raw: perf/results/2026-08-21/
  {tq_splitk_bench, tq_splitk_gate, tq_splitk_longctx,
  anchor_regate_tqsk}/. FLAGGED DECISION: -tq now wins c4/c8/2500x64
  and trails c1 by 0.8% with 2.4x smaller KV — flipping the canonical
  default is a quality call (k8v4 cos ~0.9955 floor), not a TPS one.

**N9 — mrope repeat_interleave queue-drain (2026-08-24, UPDATEs 41-42):
DONE — canonical c1 14.80 -> 16.14 (+9.6%), all pins bit-exact.**
- Live-box xctrace found one ~21 ms GPU-idle gap per c1 step (host
  still encoding when the queue drains); a fresh cProfile census
  (canonical env — the 08-21 census predated async sched and its
  67 ms postprocess_state lead was DEAD, killed by a statement-timer
  probe: 0.055 ms/step today) attributed 65.8 ms/step (64% of host
  profile) to torch.repeat_interleave in _metal_prepare_rope_positions
  — the N1 torch replacement for the Triton mrope kernel.
  Data-dependent output shape -> full MPS queue drain per step; the
  same class gdn_attn.py fixed earlier.
- Fix (metal_compat.py, python-only): static-shape scatter+cumsum
  token->request map (200/200 equal vs repeat_interleave on MPS incl.
  zero-length requests); patched DefaultModelState.prepare_inputs
  passes num_tokens=input_batch.num_tokens (host int, no sync).
  VLLM_QC_ROPE_STATIC=0 = dynamic null path.
- OPS incident (UPDATE 42): the tmp cleaner swept the a99b scratchpad
  mid-gate — m2_source.txt/dsv4_gate.py/build recipes GONE. All
  recovered byte-exact (m2_source = cat of three perf docs at commit
  2ae908e42, 69,105 B, proven by a null boot reproducing c1 467b35c3
  + 2500x64 d0e07ddd bit-exact; dsv4_gate.py from the 43e3 transcript
  Write call; build scripts from the a99b transcript). STABLE HOME:
  perf/results/harness_assets/ — every chain now uses that path.
- GATED 2026-08-24 (rope_gate/ + anchor_leg_rope/): Phase 0 null
  validation bit-exact; canonical c1 16.139/16.138 sha 467b35c3
  BIT-EXACT 2/2 (144.5% of the Q4_K bar) / 2500x64 3.454/3.447 sha
  d0e07ddd BIT-EXACT / c4 23.16 (+7.2%) / c8 25.62 (+3.9%); -tq
  needle exact + c1 16.08 sha 228d0bf4 BIT-EXACT 2/2 (-0.3% vs
  canonical; -tq c4/c8/2500x64 not re-measured post-rope); DSV4 8tok
  anchor BIT-EXACT (python-only change, no binary rebuild).
- Remaining step structure: ~80 ms GPU + residual encode burst
  (~620 Metal encodes/step: ~357 MPSGraph + ~265 custom). Next
  levers: steady-cache metadata for the mamba-hybrid path (gdn
  builder ~30 ops x 10 calls/step; extend attn_utils steady path +
  GDN steady_decode_update per the DSV4 sparse_swa precedent),
  drafter MPSGraph linears, then Muse single-CB.

**N10 — steady-cache metadata for the mamba-hybrid path (2026-08-24,
UPDATE 43): DONE — c1 16.139 -> 16.334 (+1.2%), all pins bit-exact,
default ON on Metal (VLLM_QC_STEADY_META=0 null).**
- Post-rope census: gdn_attn build 8.2 ms/step (10 group builds x
  ~30 tensor ops + h2d each, rebuilt from scratch every step);
  upstream steady machinery never fires here (FULL-graph + opt-in
  gated, disqualified whenever model_specific_attn_metadata exists).
- Change (python-only, 4 files): ModelSpecificAttnMetadata.
  steady_signature() protocol (interface.py, None default);
  MambaHybridAttnMetadata returns ("mamba-spec", k) only at uniform
  all-spec decode (CPU checks; adaptive-k changes break the sig);
  attn_utils folds the sig into the steady tuple, refreshes
  cm.max_seq_len + cm.seq_lens_cpu_upper_bound in place on hits,
  calls steady_decode_update on supporting builders with fresh extra
  kwargs, rebuilds the rest WITH those kwargs; GDN steady_decode_update
  = re-copy spec_state_indices from the live block table + refresh
  num_accepted (2 copies replace ~30 ops).
- Two engine-killing findings (both now recorded lessons):
  (1) upstream eligibility deref'd is_prefilling=None on this path —
  crash on first request; a non-None model-specific signature is now
  the all-decode guarantee (CPU-side, no MPS sync) and the tensor
  check governs only the no-metadata case. Ramp helpers must verify
  GENERATED TEXT — an instant HTTP 500 satisfies curl and read as
  "ramped" while the engine was dead (round 1's false-positive
  restore).
  (2) model_runner allocates seq_lens_cpu_upper_bound FRESH each step
  (np.zeros + from_numpy), so the cached cm's view froze at
  cold-build content and metal/tq builders (rebuilt on every hit)
  derived decode max_context from it — newest tokens silently
  excluded from full attention. Symptom: deterministic sha roll
  (ac33cee5 both runs) with PASSING needle and +2.7% "speedup"
  bought by wrong math. seq_lens / query_start_loc / gathered block
  tables / slot mappings are in-place persistent buffers (audited);
  query_start_loc_cpu freezes too but is content-equal under the
  uniform sig.
- GATED round 3 (steady_gate/): canon needle PASS; c1 16.334/16.346
  sha 467b35c3 BIT-EXACT 2/2; 2500x64 3.444/3.452 sha d0e07ddd
  BIT-EXACT 2/2; c4 23.02 / c8 25.57 (-0.6%/-0.2%, noise — hits
  rare under composition churn); TQ needle PASS; -tq c1
  16.216/16.210 sha 228d0bf4 BIT-EXACT 2/2 (+0.8%). Null boot
  reproduced UPDATE 41 exactly (16.153, 467b35c3). Honest-gain note:
  the predicted +4-8% did not materialize — adaptive-k sig breaks
  make hits rarer than the per-step build cost implied.

**N11a — fused DFlash2 grouped conv (2026-08-24, UPDATE 44): DONE —
c1 16.334 -> 16.428 (+0.6%), everything bit-exact, default ON
(VLLM_QC_DFLASH_CONV=0 null).**
- Post-N10 phase census: drafter_propose 24.4 ms/step serialized with
  only ~5 ms of GPU weight reads — host-encode-bound. Each drafter
  layer ran 4 eager _grouped_conv chains (~10 MPS ops each incl. a
  torch.arange and an F.pad) = ~200 encodes/step.
- qc_dflash_conv (serving/dflash_conv/): one thread per element,
  block-local taps + position mask folded in; per-op fp32-round
  mirrors MPS elementwise semantics — kernel parity 37/37 BIT-EXACT
  (bf16/f16/f32, blocks 4/8/9, both projection side views via storage
  offset + row stride, no contiguous copy; check_mps_strided).
  Launcher tk_launch.h, binding qc_metal_serving.mm, ops.py wrapper,
  routing in qwen3_dflash2._grouped_conv (stale-.so hasattr guard,
  layout guards fall back to eager).
- Gate: DSV4 anchors bit-exact 17th consecutive (metallib+.so
  rebuild); null boot = UPDATE 43 exactly; conv-ON c1 16.428/16.438
  467b35c3 / 2500x64 3.454 d0e07ddd / needle exact / c4 23.11 c8
  25.62. Calibration reconfirmed: eager-chain profiler shares are
  queue-tail-inflated; dispatch elimination recovers ~5 us x count.
- CORRECTION (post-gate): drafter_propose nests INSIDE the
  sample_tokens bracket (model_runner.py:1651 under the :1522 wrap) —
  32.7 = 24.4 drafter + 6.5 sample_and_reject + ~1.8 glue. There is
  NO unattributed sampling cost; the drafter remains the top host
  block. Next instrument: VLLM_QC_OP_CENSUS per-step op counts —
  count x ~5 us is the honest currency for encode levers.

## Tooling that transfers (recovered to `perf/results/harness_assets/`;
session scratchpads are swept by the tmp cleaner — never cite them as
durable locations)

- `build_metallib.sh` (17s full rebuild), `build_qc_metal.sh` (.so),
  single-file iteration loop: `xcrun metal -std=metal3.1 -O2 -I
  csrc/quixicore/metal/include/metal -I csrc/quixicore/metal/kernels/common
  <one .metal> -o mini.metallib` (~10 s) + `qgemv_mb_bench.m`
  (standalone Metal harness) + `pipe_info.m` (register-footprint
  introspection via maxTotalThreadsPerThreadgroup).
- NOTE (`cmake/metal.cmake:43-49`): editing `dequant.metal` or anything
  under `include/metal/` is a SUBSTRATE edit — the full metallib build
  tracks it via the DEPENDS glob, but a stale hand-built mini.metallib
  will silently keep old shaders.
- `mmvq_bench.py` / `mmvq_bench_mm.py` (serving-route GEMV bench),
  `q4k_oracle.py` / `q4k_oracle_mb.py` (float64 + bit-identity oracles),
  `q4k_mb_gate_a.sh` / `_b.sh` (serving gate scripts),
  `anchor_regate_q4k_mb.sh` + `dsv4_gate.py` (DSV4 anchor re-gate),
  `restore_qwen_m4c.sh` (clean-boot restore).
- N0 artifacts (session-04ba9b90 scratchpad): `n0_checkpoint_findings.md`,
  `nvfp4_config.json`, `nvfp4_index.json`, `nvfp4_shapes.json`.
- Census instruments (env-gated, in-tree): `VLLM_SYNCPROF=1`,
  `VLLM_QC_PHASE_PROF=1` (layer brackets in qwen3_next.py).
- Artifact backups: `artifacts_backup_m4/`, `artifacts_backup_m4b/`.

## Standing ops constraints (non-negotiable)

- Explain-before-edit for every project-code change (symptom, root
  cause w/ evidence, exact change, why safe — then proceed).
- Commit only when the user asks.
- Interval logs are diagnostics; exact-token harness output only for
  claims. Record everything in perf/optimization_status.md per
  perf/perf.md; raw artifacts under perf/results/YYYY-MM-DD/<run-id>/.
- DSV4 anchors re-gate on ANY metallib/.so rebuild (UPDATE 30 pins:
  8tok 573db39598e7, off1-2000 bb83cc3054a3, 2500x64 f75e1d41ac3d).
- Boot ramp protocol before multi-chunk prefill (primer + 1000-tok
  w/ decode + multi-chunk throwaway). Kill = TERM EngineCore AND
  api_server pids, verify dead, wait for memory before reboot.
- Milestone ritual: inline copy-pasteable compaction handoff in the
  reply at every milestone. Hold turn for <10-min waits.
- Do NOT use the AskUserQuestion tool — it breaks the user's TUI; ask
  in plain text.
- Qwen model facts: 64 layers = 48 GDN + 16 full attn, hidden 5120,
  ffn 17408, vocab 248,320; eos 248044 per NVFP4 config.json (old
  handoff said 248046 — recheck against the served tokenizer at N1).
  Metal victim canary hardcodes block 30 (delete after soak); NANPROBE
  kit env-gated VLLM_QC_NANPROBE=1.

---

# PARALLEL CAMPAIGN HANDOFF (merged from main 2026-08-25): Qwen3.8-27B + DFlash 2 GGUF/vision Metal serving (M5 Max box)

# Qwen3.8-27B + DFlash 2 Metal Serving Handoff

Updated: 2026-08-23 00:58 (M5 Max MacBook Pro, 128 GB, ~460 GB/s measured
stream). Written so a fresh agent can take over cold. Read this, then
`perf/optimization_status.md` entries (19) onward, then
`perf/qwen38_metal_design.md` (every verified tensor map + mechanism).

## Mission and hard rules

- Serve profile `qwen38-q2kxl-1` on Metal: Qwen3.8-27B (unsloth UD-Q2_K_XL
  GGUF, 64-layer hybrid: 48 gated-deltanet linear-attention layers + 16
  full-attention layers, head_dim 256, interleaved MRoPE; plus the
  mmproj-F16 `qwen3vl_merger` vision tower) speculated by the Inco AI
  DFlash 2 drafter (z-lab Q4_K_M GGUF, block 8, top-16 path selector,
  two-tap convs).
- Bars: llama.cpp plain decode on this box/artifact = **35.67 tok/s**.
  Vendor DFlash 2 acceptance 4.80 is a GSM8K number; the llama.cpp
  dflash2-pr branch on the SAME GGUFs/settings gets 2.51 tok/step on our
  essay prompt and 4.74 on a GSM8K-style prompt (task-domain dependence;
  compare acceptance only on matched arms).
- **Speculation is always on and must be net-positive** (memory
  `spec-always-fastest`); slower-than-plain spec is a BUG, never a
  documented config. This gate is now closed: registered DFlash k=3 beats
  plain on both retained prompt arms and the matched exact-server workload.
- **Greedy / temperature 0 is banned stack-wide** (memory
  `no-greedy-benchmarks`; the user removed the flag on purpose). Validation
  and benches use the model's shipped sampling defaults from the GGUF
  (`general.sampling`: temp 1.0 / top_p 0.95 / top_k 20), seeded (42).
  Layer-level parity (cosine on activations) needs no sampling and stays
  the correctness instrument.
- Commit authorship: Eric Hartford sole author, no assistance trailers
  (the repo's signoff hook adds his Signed-off-by). Commit with
  `env SKIP=markdownlint-cli2 git commit ...` (the notebook's pre-existing
  line lengths fail markdownlint; its auto-fix also corrupts `+ ~15`-style
  lines and `_foo` identifiers -- never let it run on perf/).

## State of the tree

Committed on `main` (pushed): `fe960935f` "vision, DFlash 2 spec e2e,
native IQ decode, hybrid-pool layout fix (15 tok/s plain)" on top of
`39efaa7d9` (correct plain decode, layer parity). Prior campaigns: Muse
`ad8e8e937` (20.1 tok/s spec, Metal), DSV4 A100 `bad7cfd46` (A100 box).

UNCOMMITTED in the worktree is one tested optimization stack (preserve all
of it; do not treat the native pieces as abandoned experiments):

1. **Fused target GDN, complete and routed.**
   `csrc/quixicore/metal/kernels/serving_glue/gdn_step.metal`, the binding
   in `qc_metal_serving.mm`, `vllm/quixicore/ops.py`, and
   `qwen_gdn_linear_attn.py` implement decode and multi-position verify
   (convolution + recurrent scan, fp32 state in place, exact store/resume /
   rollback slots) plus a fused gated RMS norm. The torch-native oracle
   remains intact. Kill switch: `VLLM_QWEN38_FUSED_GDN=0`.
   Correctness: 147/147 exhaustive cases, all uniform/ragged/null/mixed
   plan cases, all gated-norm cases. Durable real-geometry tests are in
   `tests/model_executor/test_qwen_gdn_metal.py`.
2. **Verify-band quant MM, complete and routed.**
   `dequant.metal`, `qgemv.metal`, and the binding admit the target's
   Q2_K/Q3_K/IQ1/IQ2/IQ3/IQ4_XS formats to M={2,4,8,16,17} MM instead of
   repeated GEMV. Real-GGUF M=8/17 error <=0.2674%; sampled M=8 kernels
   are 2.4-5.0x faster than eight GEMVs. The eight-wide IQ decoder's
   difference from the scalar decoder is rounding-only (<0.1%).
3. **Seeded/vectorized MPS rejection, complete and routed.**
   `rejection_sampler_utils.py`, `qwen3_dflash2.py`, and `speculator.py`
   key all selector/accept/residual/bonus draws by (seed, position), remove
   the draft-logit double temperature divide, and batch rejection fully on
   MPS. Full-vocabulary Gumbel emission is now one keyed uniform plus CDF
   inverse sampling. Monte Carlo passes; the clean spec bench is seed-stable.
4. **Fused DFlash 2 convolution, built and routed.**
   `csrc/quixicore/metal/kernels/serving_glue/dflash2_conv.metal` replaces
   each repeat_interleave/roll/clone/elementwise graph with one dispatch.
   Both sides pass the torch reference at real 5120-hidden / 320-group BF16
   geometry; a same-process real-geometry microbench is 5.65x faster.
   Kill switch: `VLLM_QWEN38_FUSED_DFLASH2_CONV=0`. Powered end-to-end A/B
   retained it: fused essay/GSM medians 16.42/38.12 versus 15.75/36.22.
5. **64-bit hybrid KV gather, complete and routed.** MPS `index_select`
   silently wrapped signed 32-bit element offsets on Qwen's interleaved,
   strided K/V source. Requests crossing physical block 1271 therefore fed
   bad K/V into layer 19 and produced all-NaN target logits. The native
   `kv_cache_gather_range` carries the physical block stride and address math
   in 64 bits, gathers only live rows, and is exact at blocks 1186/1271/1580.
   A 20x64-token repeated run remains finite and seed-identical through the
   old failure window and allocator wrap.
6. **Profile/tests/build.** `qwen38-q2kxl-1` is supported, registers DFlash
   k=3, and exports `VLLM_USE_V2_MODEL_RUNNER=1`. The real server passed text
   and image. The final focused suite (including the 5 GiB >2^31-offset case)
   is 17/17; `tests/slimserve` is 58 passed/1 skipped. Final metallib SHA-256:
   `539035eb15dea29152e11503fc1ee08676d5dfe08b9ef4cc241283092e887d4c`;
   deployed extension SHA-256:
   `ede784f0d4ecf7a5111fc55374987661ea3bcc4c48602189c0a009cb88c4efdb`.
7. **Shared live validation.** Registry discovery finds `dsv4-xxs-1`,
   `muse-kdyn-1`, and Qwen on this machine. Qwen passes text+image. Muse now
   passes text+image after fixing its parser's new-turn state and the real
   split `" to"` / `"=self<|message|>"` streamed header; raw SSE cleanly
   separates `reasoning_content` and final `content`. The combined parser and
   SlimServe suite is 62 passed/1 skipped. The complete matrix is not green:
   DSV4 reached health but its first request ran at about 0.1 tok/s with 0/5
   drafted tokens accepted and was terminated after about 12 minutes.

**CURRENT STATUS:** no correctness or profile gate is blocking Qwen serving.
The old 20 W power blocker is closed; the retained numbers below were captured
on AC power after the charger change. Qwen's remaining gap is performance
versus the 35.67 tok/s llama.cpp plain reference, not production-path
correctness. Separately, the current-machine profile matrix is blocked by the
DSV4 Metal regression described below; do not present that shared matrix as a
pass.

## Measured numbers (seeded shipped defaults, V2 runner, in-process)

| Build | Plain essay | Plain GSM8K | Spec essay | Spec GSM8K |
| --- | ---: | ---: | ---: | ---: |
| campaign start (V1) | 2.5 | -- | -- | -- |
| V2 runner, fp16 dequants | 6.4 | -- | 4.0 | -- |
| + layout fix (strided gather penalty) | 2.2 | 2.0 | 1.0 | 2.2 |
| **fe960935f** (+ gather fix, native IQ) | **15.0** | **14.0** | 3.5-4.1 | 8.0-9.5 |
| + fused GDN/MM/vector rejection (pre-conv) | 16.15-16.40 | 15.81-16.11 | **15.03-15.60** | **36.06-36.66** |
| + powered stack, k=7 | 16.99-17.17 | 16.77-16.86 | 16.05-16.35 | 36.82-38.19 |
| **supported profile, k=3** | **16.99-17.17** | **16.77-16.86** | **23.06-23.74** | **34.33-35.25** |

Acceptance (Prometheus counters, essay, notebook (25)): 2.71 tokens/step,
0.244 draft rate -- beats the llama.cpp dflash2-pr branch (2.51/0.219).
Correctness: all 64 layers cos >= 0.9997 vs llama.cpp eval-callback;
corruption gauntlet 7/7 clean boots, 24/24 same-seed pairs identical;
fused GDN exhaustive harness 147/147 plus all plan shapes; rejection Monte
Carlo PASS; current focused durable suite 17/17; vision tower ~1e-3 vs
llama-mtmd-cli; real SlimServe text and image requests pass; Muse-Glimmer was
unregressed by a fresh profile-exact text+image smoke after its reasoning
parser repair.

The old 8-position x 48-layer Python GDN inversion and the >2^31 hybrid-cache
corruption are closed. At k=3, M=4 verification is a better Metal operating
point than the trained/upstream k=7/M=8 width: essay median rises from 16.18 to
23.27 tok/s while GSM remains 34.78. The exact registered server produced 128
input + 256 output tokens at 18.646 tok/s spec versus 15.913 plain (+17.2%).

## Open bugs / items, ranked

1. Remaining perf versus llama.cpp: Q4_K GEMV measures only ~95 GB/s on the
   5120x6144
   ssm_out shape (4.7x off floor, pre-existing); head_dim-256 paged
   attention fast path (the 16 full layers run SDPA; paged path is
   64/128 only); selector walk and the other five-layer drafter graphs.
   The k=3 top ledger is target 70.19 ms and inclusive sample/propose tail
   16.42 ms per step, so target bandwidth is again the primary wall.
2. Fix the DSV4 Metal profile regression exposed by the attempted complete
   live-smoke matrix. `dsv4-xxs-1` loaded 93.63 GiB and reached health, then
   spent about 12 minutes on the first tiny request at ~0.1 tok/s; the first
   draft had 0/5 accepted. This is grossly inconsistent with its historical
   33.684 tok/s baseline and must be isolated at first-step/verify granularity.
   Qwen and Muse both pass their registered text+image arms, but the full
   three-profile matrix remains failed until DSV4 completes.
3. Cosmetics/hygiene: env-gated diagnostics remain in
   `models/qwen3_5.py` (`_Qwen38DumpState`, layer-parity instrument) and
   `qwen3_dflash2.py` (`QWEN38_DFLASH_DUMP` recall@k dump) -- zero-cost
   unset; remove at campaign close. Stale `autostash` entry in `git
   stash` is from Aug 7 (DSV4 era), safe to drop. A stray token quirk
   appeared in sampled answers at temp 1.0 (both text and vision) --
   unattributed, low priority.

## Scripts and raw artifacts

Durable copies are under `perf/results/2026-08-22/qwen38-fused-gdn/`:
`consolidated_bench.py {plain|spec}` (essay + GSM8K arms, 3 seeded repeats,
prints BENCH_JSON), `collapse_probe.py`, `spec_profile.py`, and
`spec_profile_top.py`. Patterns: in-process `vllm.LLM(**engine_kwargs)`
from `slimserve.registry.resolve("qwen38-q2kxl-1","metal",1,None,2**37)`,
`__main__` guard (EngineCore spawns), `SamplingParams(temperature=1.0,
top_p=0.95, top_k=20, seed=42)`, `max_model_len` 8192 +
`gpu_memory_utilization` 0.45-0.6 for qwen38 (Muse smokes must use
PROFILE-EXACT kwargs -- a max_model_len override breaks its image
profiling).

Final raw data is under `perf/results/2026-08-23/qwen38-kv-gather/`:
`run_summary.json`, `exact_spec.json`, `exact_plain.json`, `smoke.json`, and
the real-server log. Shared-profile evidence is in `smoke-muse-final.json`,
`smoke-muse-final/muse-kdyn-1.log`, and `smoke-all/dsv4-xxs-1.log`. The exact
server harness command uses
`benchmarks/benchmark_dsv4_exact.py` with explicit `--temperature 1.0
--top-p 0.95 --top-k 20 --seed 42`; never rely on that harness's legacy greedy
default for Qwen.

## Ops gotchas (each cost real time)

- The Mac SLEEPS and kills background runs/agents: hold it awake
  (`mcp adrafinil keep_awake`, lid-closed included) for any long run.
- A 20 W charger at 1% battery throttles this workload catastrophically.
  Require a high-wattage supply and battery reserve before any baseline or
  phase-profile run; check `pmset -g batt` and the charger wattage first.
- Refreshing `vllm/quixicore_metal.metallib` / `_quixicore_C...so`: rm-then-cp
  + `codesign -f -s - <so>`; cp over the mapped inode SIGKILLs on dlopen.
- llama.cpp builds: `env -u LDFLAGS -u CPPFLAGS` (a custom-LLVM env poisons
  links); `~/llama.cpp/build-qwen38` (master, plain oracle),
  `~/llama.cpp-dflash2/build` (PR #27342 spec oracle); check a binary is
  fresh before trusting it (`strings ... | grep` a known-new symbol).
- Registry bytes/sha come from `curl -I` / the HF paths-info API, never
  from summarized pages (a wrong byte count masqueraded as a broken
  download for an hour).
- Regression smokes use profile-exact engine kwargs.

## Reference facts (verified; full maps in perf/qwen38_metal_design.md)

- GGUF arch strings: target `qwen35`, drafter `dflash` (three-way probe:
  `dflash.expert_count` -> DSV4 DSpark, `dflash.selector_rank` -> DFlash
  2, neither -> Muse) across config parser, tokenizer registry, loader.
- llama.cpp converter conventions undone at load: +1 fold in every norm
  weight except `linear_attn.norm` (we use GemmaRMSNorm -> subtract 1),
  GDN per-V-head tensors in TILED order (pairing i_k = i_hv % 16, cfg
  `gdn_tiled_v_head_layout`; the FLA Triton kernels still assume grouped
  if this GGUF ever runs on CUDA), `ssm_a` stored as -exp(A_log)
  (A_log = log(-ssm_a)), conv1d (dim,kernel)->(dim,1,kernel), MTP block
  `blk.64.*` unmapped, fused `attn_qkv` row-split into q/k/v shards
  (GGUF quantizes per output row), full-attn gate fused inside `attn_q`
  (per-head [q|gate], matches the vendored split).
- Hybrid shared block pool: attention views are restrided blocks-first
  (attn_utils `_update_hybrid_attention_mamba_layout`); metal_attn's SDPA
  path uses the native 64-bit range gather. Do not restore MPS `index_select`
  on the strided pages view: beyond 2^31 elements it silently reads the wrong
  address, even though its small-cache microbench is fast.
- Quant formats: all native on Metal (qgemv + qgemm tiles incl. IQ1_S,
  IQ1_M, IQ2_XS, IQ2_S, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL);
  `_DEQUANT_TYPES` is empty; only the Q2_K embed table dequantizes.
- Drafter: 5 layers all NON-causal (`dflash.attention.causal=False`),
  block 8 counts the anchor (7 drafted), target layers [5,19,33,47,61]
  0-based, selector A/B tables {248320,256} Q4_K dequantized at load. The
  registered Metal serving depth is deliberately k=3 after the powered sweep.

---

# Handoff: MI300X GGUF profile record (2026-08-25)

> **Status update (2026-08-28, added during the origin/main merge).** The
> "actual open problem" below is **resolved**; read this section as the
> investigation record it was, not as current state. The illegal-memory-access
> fault at `max_num_seqs: 64` was root-caused to the DFlash2 two-tap
> convolution using the checkpoint's trained block width (8) instead of the
> active `1 + num_speculative_tokens` serving layout, and fixed in
> `_resolve_serving_block_size` (`vllm/model_executor/models/qwen3_dflash2.py`).
> The `qwen38-q2kxl-1` mi300x record ships at 64 sequences with the fixed
> 96 GiB KV pool -- the "fallback that passes" was not needed. Measured exact
> workload: c1 77.23 tok/s, c8 194.21 tok/s on the V2 runner, 200.20 tok/s
> after type-aware imatrix routing. Full write-ups: the three entries dated
> 2026-08-25/26 at the end of `perf/optimization_status.md`.

## One-paragraph state

The Qwen3.8-27B GGUF path on MI300X **works** — correct text and vision output,
DFlash2 speculation live — and that is committed and pushed. What is **not**
finished is the `qwen38-q2kxl-1` mi300x profile *record*: it is uncommitted in
the worktree, and the config values I picked fail in ways I had not finished
bisecting when this session ended. A working configuration is known (see
"Fallback that passes"). The open question is whether the failures are bugs
worth fixing or shapes to back away from.

## What is committed and pushed (do not redo)

`main` @ `2989f4b28b`, in order:

- `cd7e983d07` — **GDN value-head layout fix.** llama.cpp stores per-value-head
  gated-deltanet tensors in ggml *tiled* order (value head `hv` pairs with key
  head `hv % H`, expanded with `ggml_repeat_4d`, see
  `~/llama.cpp/src/models/qwen35.cpp:443`); the FLA kernels expand with
  `repeat_interleave`, i.e. HF *grouped* order. Selected by platform:
  `gdn_core_honors_tiled = current_platform.is_metal()`. Metal keeps its native
  tiled scan; elsewhere the v-head axis is normalized around the recurrence.
  The same normalization also had to go into `_forward_core_decode_non_spec`,
  whose early `return` sits between the two reorder points — that is why
  prefill was perfect while every decoded token was garbage.
- `b4cc16492c` — **DFlash drafter KV layout fix.** The drafter hardcoded the
  split cache layout `(2, num_blocks, block_size, H, head_size)` it was written
  against on Metal. TRITON_ATTN and ROCM_AITER_FA *pack*:
  `(num_blocks, H, block_size, 2*head_size)`. `_store_kv_at_slots` now detects
  the layout and **raises on anything unrecognized rather than guessing**.
- `a4ee5caea6` — NVFP4 regression gate (that profile still passes).
- `2989f4b28b` — **profiles are one config per platform.** Five profiles used
  to span platforms via `platform_overrides`; each id now stores one record per
  platform under `variants`, each tagged with its own `platform`. The CLI still
  takes no platform (it detects one). `registry.variant(id, platform)` gets one
  record. Two tests enforce it. `platform_overrides` is retired.

Earlier in the session (also pushed): IQ2_XXS dense-dispatch fix, `default:`
guards on the GGUF kernel switches, ROCm build repair, compiled-startup fix.
See `perf/optimization_status.md` for the full write-ups.

## Uncommitted work in progress

`git status` shows two modified files:

- `slimserve/profiles.json` — adds the `mi300x` record to
  `qwen38-q2kxl-1.variants`, retitles the profile ("Qwen3.8-27B GGUF on 1 GPU",
  was "on one Mac" which is wrong for a two-platform id), and adds
  `min_gpus.mi300x = 1` to the `q2kxl` quant.
- `tests/slimserve/test_profiles.py` — the smoke matrix asserted every mi300x
  profile drafts with DSpark; this one legitimately uses DFlash2, so it needed
  an exception next to the existing NVFP4 one.

62/62 tests pass with these changes.

## The actual open problem

I sized the mi300x record from the `qwen38-nvfp4-1` mi300x record (same model,
same card, already tuned) rather than from the Metal record. That changed four
things at once versus the config known to work, and then I guessed at which one
broke it instead of bisecting. That was the wrong method and it cost four runs.

| run | config delta from Metal baseline | result |
| --- | --- | --- |
| `gguf-mi300x` | `gpu_memory_utilization: 0.9`, 262144 ctx, 64 seqs, FULL_DECODE_ONLY | **PASSED**, 268.6 s load |
| `gguf-mi300x2` | fixed 96 GiB KV pool, `max_num_batched_tokens: 8192` | illegal memory access |
| `gguf-mi300x3` | fixed pool, batched tokens back to 2048 | illegal memory access, truncated answer |
| `gguf-mi300x4` | fixed pool, `cudagraph_mode: NONE` | **different** failure: `invalid configuration argument` in `_dummy_sampler_run` → `apply_temperature` |
| `gguf-mi300x5` | `max_num_seqs: 16`, graph capture restored | **was still running at handoff** — read `perf/results/2026-08-25/gguf-mi300x5/smoke.json` |

Raw logs for every run: `perf/results/2026-08-25/gguf-mi300x*/`.

### What is ruled out

- **Memory pressure.** GPU 1 was idle; every run allocated a valid KV cache
  (~1.2M tokens, 4.6x concurrency for 262144-token requests).
- **CUDA graph capture.** Disabling it did not fix the fault, it produced a
  *different, earlier* one. So the decode-path GDN changes from `cd7e983d07`
  are not implicated in the memory fault.
- **The profiling forward.** I claimed this was the discriminator and was
  wrong — the earlier passing `gguf-spec3` run also skipped it (it used the
  Metal record's fixed `kv_cache_memory_bytes`).

### Live hypothesis

Something about the larger shapes trips a kernel launch-configuration limit.
`invalid configuration argument` from `apply_temperature` during sampler warmup
is a bad grid, and the shape driving that is `max_num_seqs` (I raised it 16 →
64). `max_model_len: 262144` may interact. **If confirmed this is likely a real
bug worth fixing rather than a value to retreat from** — but bisect one variable
at a time before concluding anything.

### Fallback that passes

If the bisection stalls and you need a landable record, `gguf-mi300x` passed:
`gpu_memory_utilization: 0.9`, 262144 ctx, 64 seqs, FULL_DECODE_ONLY,
`max_num_batched_tokens` default. Its one flaw is a 268.6 s boot, of which
**142 s is an 8192-token profiling forward** JIT-compiling Triton GDN kernels
(`profile_run: LM dummy run`, boot +80.2 s → +222.5 s). A fixed
`kv_cache_memory_bytes` skips that phase entirely — which is why I reached for
it, and where the trouble started. Do not ship a record that faults, and do not
ship the slow one without saying why in the notebook.

## Verification recipes

Smoke the profile as registered (this is the gate that matters):

```bash
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "
import sys; sys.argv=['smoke','--profile','qwen38-q2kxl-1','--max-tokens','64',
 '--log-dir','perf/results/<date>/<run>','--output','perf/results/<date>/<run>/smoke.json']
from slimserve.smoke import main; sys.exit(main())"
```

Decode/prefill self-consistency (no external oracle; catches decode-path bugs
that prefill-only checks miss — this is what found the fast-path bug):
`/tmp/.../scratchpad/decode_equiv.py` in this session, or re-derive: generate N
tokens incrementally, then re-prefill the growing sequence for each token, and
compare. They must match exactly.

Per-layer parity against llama.cpp:

```bash
# reference (writes ~2700 tensors with sums and corner values)
~/llama.cpp/build/bin/llama-eval-callback -m <gguf> -p "<prompt>" -n 1 --temp 0 -ngl 99 < /dev/null
# ours: env-gated per-layer dump, arm with a prompt starting "The three most", >= 11 tokens
VLLM_QWEN38_DEBUG_DUMP=<dir> .venv/bin/python <script>
```

`llama-cli` needs `--no-conversation --single-turn` with stdin closed; `-no-cnv`
is ignored in this build and it will loop emitting prompts.

## Traps that cost real time today

1. **Sums are permutation-invariant.** They cannot distinguish right-values-
   wrong-order, and on a near-zero residual dominated by cancellation they are
   actively misleading (`linear_attn_out` read 204 vs 360 while the elementwise
   corners matched to quantization noise). I built and published a wrong root
   cause on this and had to retract it. **Compare corner values.**
2. **A missing probe is evidence.** The decode step printing no `conv_out` while
   prefill printed it is what exposed the bypassed fast path. When two paths
   disagree, diff *which code each executes* before diffing numbers.
3. **`pgrep -f <script>` matches the Bash wrapper's own command line.** I hung
   three shells with this today, despite a memory note warning about it. Wait on
   explicit PIDs with `kill -0`.
4. **Don't change four things and then guess.** See the table above.
5. `llama.cpp`'s `attn_output-N` is the GDN core output *before* `out_proj`; the
   counterpart to a module-level hook is `linear_attn_out-N`.

## House rules that bit me

- A profile is **model x quant x platform x config**. Never widen a profile to a
  platform it was not tuned on; add that platform's own record. I used a
  temporary-widening script all session before being corrected — don't.
- Ask the user about facts he already knows (was this artifact validated, what
  did that campaign run) instead of spending GPU runs deriving them.
