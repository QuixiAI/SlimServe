# GLM53 SM120 attention normalization isolation

## Current evidence, 2026-09-10

The completed KDA gate/recompute/state/output probes found no cross-config
differences on their fixed matrices. The intra-stage candidates had identical
binaries. These results narrow the search but do not prove equivalence for all
activations or establish historical live KDA choices.

The earlier attention probe found Q1536 exact, but KV512 and LayerNorm128 differ
between the original fused bundle and the fresh split kernels. KV512 passes the
existing one-BF16-ULP oracle gate on that synthetic matrix. LayerNorm128 fails
in BOTH versions; that independent failure remains open. Neither the rejected
RMSNorm geometry candidate nor any KDA result clears it.

`benchmarks/analyze_glm53_attention_contracts.py` now checks the retained graph
boundaries without importing generated code. It consumes pinned completed
receipts, not obsolete source-freeze validators. All eight attention graph pairs
(two per rank) match; the other 20 graph pairs have no attention bundle. The
75 graph/source/evidence hashes verify. In each pair it checks:

- One original bundle maps to exactly three split calls, matching exact input,
  weight/bias, output, row-count and stream expressions.
- Raw projection provenance, output allocations and alias chains agree.
- Native operation sequences and the entire graph return interface agree.
- Embedded compile-time docstrings do not count as runtime calls.

The shared BF16 projection is `[rows, 2336]`. Q occupies columns 0:1536;
KV 1536:2048; indexer K 2048:2176; its gate 2176:2304; weights 2304:2336.
Q and KV norm outputs are contiguous. Indexer K writes the first 128 columns
of a `[rows, 256]` buffer; the gate fills its second half. Q feeds both the
main and indexer projections; KV and the packed indexer buffer cross the graph
return boundary. Any intervention must preserve those aliases and neighbors.

CPU: 67 passed in 7.14 s (14 upstream Torch deprecation warnings); earlier
14-pass report retained. Tests reject swapped weights/biases/outputs, changed
streams/aliases/native consumers/returns and ambiguous buffer definitions.
Lint and diff checks pass. No GPU, model, native, profile or quant change.

Raw `perf/results/2026-09-10/runtime-control/attention-contracts.json`, SHA
8337ae7e4ec383563ed2df0230f21e9424556feccbec3a93a61deea94caff3c8;
`attention-contracts-{cpu,final-cpu}.xml`. Recorded command:

```bash
systemd-run --user --scope --unit=glm53-attention-contracts-analysis -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.analyze_glm53_attention_contracts --output perf/results/2026-09-10/runtime-control/attention-contracts.json
```

This is STATIC correspondence. It does not prove complete historical live
coverage, runtime input equality, pointwise-fusion equivalence, causality or
production quality. Generated sources are inspected, not installed or modified.

## Next implementation: KV512-only diagnostic

Use the original attention bundle unchanged, then overwrite ONLY its KV512
output with the recorded split KV kernel on the same packed input, weight,
output address and stream. Keep Q1536 and indexer LayerNorm128 on their original
path. This tests one arithmetic change and avoids introducing the independently
failed LayerNorm replacement. It is a diagnostic extra launch, NOT a production
implementation or proposed speedup.

Before model execution, qualify the actual adapter's eager and changed-input
graph replay: KV must match the split kernel exactly; every other output and
packed sibling must match the original bundle exactly. Reuse existing source,
rank-private binary, oracle and graph-loader machinery where applicable. Record
actual launch coverage rather than relying on this static map. Preserve the
original KDA, H4096 choices, native libraries, recipe and all quality floors.

After that qualification, prescribe a bounded original/KV-only/return-original
real-profile comparison. Compare all text/needle scores, not only aggregates,
and retain failed quality results. Do not assume KV alone reproduces the
no-combo vector: that candidate also changed H4096 and other attention settings.
Reversibility and contribution are distinct from production acceptance.

No next GPU/model run is prescribed by this design note. Finalize commands,
repetitions and failure handling before launch. Do not revive any terminal
series, weaken the indexer oracle gate or benchmark until a fast start appears.

## KV overwrite adapter v1: completed qualification (commands historical)

`benchmarks/kernels/glm53_attention_overwrite.py` retains the original eleven-arg
combo ABI, calls the precompiled original, then the precompiled split KV launcher
with arguments 0/1/5, rows/512, on the same stream. It does no allocation, copies,
tuning or serving installation. Host-call counters include capture, not GPU-only
graph replay. The new probe is `check_glm53_attention_overwrite.py`.

After final CPU tests and commit, exactly ONE new process uses GPUs 0..3
sequentially in a 16 GiB/swap0 scope. Preparation/audit use 8 GiB/swap0. No other
GPU work, serving, native build, source edit or commit from preparation through
closure. No retries, replacement processes or excluded cases.

- Compile and verify all eight source/config/actual whole-cubin bindings BEFORE
  numerics: one original combo and one split KV per rank, recorded launch configs,
  rank-private caches and preserved first-writer debug provenance. Never autotune.
- Fixed order: ranks 0..3, rows 1/3/16/640/7616, seeds 530901/530902, magnitudes
  0.125/1/8; 120 cases. Real layer11 BF16 weights and synthetic packed inputs,
  matching the completed attention matrix. Changed input uses seed+100.
- For original and adapter: two eager calls, capture, changed-input graph replay
  and eager comparison, then original-input replay. Inputs/weights, row guards
  and indexer output stride gap must remain unchanged; all outputs finite.
- Compare adapter KV exactly with a separate direct split-KV launch. Compare Q
  and indexer K exactly with original combo outputs. Require all input hashes,
  original outputs and direct KV outputs to match the completed historical
  matrix. The changed-KV counts must also match those retained paired records.
- KV retains the existing <=1 BF16 ULP float64-oracle gate. Indexer LayerNorm
  remains unchanged and its earlier failed accuracy gate remains FAILED. This
  qualifies the adapter's transformation, NOT the whole attention bundle or model.
- Record each case before its numerical audit, plus actual binary copies/caches,
  hashes, attempt marker and summary. Stop at first failure; retain and audit the
  partial attempt as terminal. Verify frozen sources, all 5,172 original cache
  files, unchanged GPU/driver/power identity and GPU release at closure.

Each command below once, from repository root after commit:

```bash
systemd-run --user --scope --unit=glm53-kv-overwrite-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_attention_overwrite prepare --manifest perf/results/2026-09-10/runtime-control/kv-overwrite-v1-manifest.json
systemd-run --user --scope --unit=glm53-kv-overwrite-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda-13.0 .venv/bin/python -m benchmarks.kernels.check_glm53_attention_overwrite run --manifest perf/results/2026-09-10/runtime-control/kv-overwrite-v1-manifest.json --output perf/results/2026-09-10/kv-overwrite-v1
systemd-run --user --scope --unit=glm53-kv-overwrite-v1-audit -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_attention_overwrite audit --manifest perf/results/2026-09-10/runtime-control/kv-overwrite-v1-manifest.json --output perf/results/2026-09-10/kv-overwrite-v1
```

No full-model run is yet prescribed. Even a passing adapter probe still needs
actual graph-loader binding/coverage verification before serving integration.

## Completed adapter v1 result

On `f6ff10453`, exactly one process completes all 120 prescribed cases, then its
audit passes. All eight source/config/actual whole-cubin bindings verify before
numerics. At closure, 202 source/evidence hashes and all 5,172 original cache
files remain unchanged; GPU release and device/driver/power identity verify.
No retries, excluded cases, autotuning, model starts or native builds. The source
freeze has ended. CPU final 88 passed in 3.84 s; earlier 39/87-pass reports retained.

| Output | Compared BF16 values, both phases | Adapter vs intended result | Adapter vs original |
| --- | ---: | ---: | ---: |
| KV512 | 203,390,976 | 0 differences from direct split KV | 556 differences |
| Q1536 | 610,172,928 | 0 differences from original | 0 differences |
| Indexer K128 | 50,847,744 | 0 differences from original | 0 differences |

All input hashes, original output hashes and direct-KV hashes exactly match the
retained historical matrix. The 556 KV changes reproduce its per-case counts.
KV's existing float64-oracle gate passes with maximum one BF16 ULP. Every eager,
original/changed-input graph replay, mutation, row-guard and stride-gap check
passes. The adapter only changes the prescribed output on these inputs.

The indexer LayerNorm accuracy gate remains FAILED. Its unchanged output is
not newly accuracy-qualified. This is adapter functional qualification, not
actual serving-loader coverage, full-model causality, universal accuracy or a
speedup. No profile/default/quant/quality-floor change.

Raw `perf/results/2026-09-10/kv-overwrite-v1/`: 120 case records, eight binary
receipts/private caches, source copies, attempt marker, summary and analysis.
Analysis SHA `bfaa7a495e7b69f228661a4402f5a6d5229d53b6950b6e7ee8cfa6275975144d`;
summary SHA `aa26ada81eda043f410dbc81d31c85877f9277eb8fc1019b2c836fd925aaa5c6`.
Consume these completion receipts after later edits; do not rerun the now-
historical frozen-manifest commands.

Next: actual graph-loader adapter and all-rank binding/coverage qualification.
Reuse the existing binary observer and serialized AOT-root inventory. Verify the
original combo and appended KV launchers, source/config/whole-cubin receipts,
complete graph coverage and unchanged non-target bindings. The installed
`CachingAutotuner.run` has a cached-launcher fast path; explicitly verify that
the live call path cannot bypass the overwrite. Only then prescribe the bounded
original/KV/return real-profile series. No next GPU/model job is prescribed yet.

## KV graph-loader implementation: CPU-tested, not GPU-qualified

The new benchmark-only `glm53_kv_loader.py` uses a separate
`glm53-kv-loader-v1` manifest and `control`/`kv` arms. Both arms assign the target
autotuner's instance `run` explicitly, bypassing both the ordinary dispatcher and
its possibly different cached fast launcher. Control calls the exact original
static launcher; candidate calls the already-qualified `KVOnlyOverwrite.run`.
The original `compile_results` and `launchers` entries are not replaced with a
misleading one-kernel description of a two-launch operation.

`audit_glm53_kv_graphs.py` independently walks supplied actual AOT-root modules,
using the geometry auditor's shared graph/source/call-export checks. It follows
the live adapter's appended launcher into its observed static CUDA object, then
checks its kernel name, full configuration, semantic key and whole-cubin hash.
The original combo is independently checked, and all non-target graph bindings
remain in the report for exact cross-arm comparison. Controller callback
coverage is an additional seal, not the source of the independent inventory.

Strong references plus one RLock protect shared target futures. An upstream
future is resolved once for each target object; aliases revalidate/reuse it.
Finished graph modules are bound too. Sealing rejects new targets/modules and
changed run/config/binary/source bindings, while allowing legitimate unrelated
compilation before capture completes. Source/debug provenance and rank-private
cache checks are reused. `glm53_loader_hooks.py` shares hook installation and
exact restoration with geometry, including foreign-hook preservation; future
geometry preparation includes this dependency in its frozen source list.

CPU final: 449 passed in 41.23 s, 14 upstream Torch deprecation warnings. Tests use
real CachingAutotuner, StaticAutotunerFuture, PyCodeCache and static launcher
objects, replacing only the driver's load/launch interface. They cover actual
eleven-argument combo and five-argument KV calls, stream/destination forwarding,
the persistent KV configuration without R0_BLOCK, concurrent imports, future
aliases, complete module binding, cached-path bypass, late targets, tampering,
non-target inventory/compilation and hook restoration. The preceding 75-pass and
111-pass reports remain; early fixture failures and the stale historical-source
test failure are retained too. That last failure was the runtime correctly
rejecting old qualification after implementation changes. The test now verifies
rejection at both policy and direct-client entrypoints; no guard was weakened
and no historical manifest/attempt was rewritten.

Final CPU command (no GPU, 8 GiB/swap0):

```bash
systemd-run --user --scope --unit=glm53-kv-loader-final-cpu -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest \
  tests/slimserve/test_kv_loader.py tests/slimserve/test_geometry_loader.py \
  tests/slimserve/test_binary_observer.py tests/slimserve/test_attention_overwrite.py \
  tests/slimserve/test_artifact_roots.py tests/slimserve/test_geometry_loader_audit.py \
  tests/slimserve/test_geometry_serving.py tests/slimserve/test_geometry_serving_runner.py \
  tests/slimserve/test_geometry_workload.py tests/slimserve/test_rmsnorm_geometry.py \
  tests/slimserve/test_rmsnorm_geometry_probe.py tests/slimserve/test_rmsnorm_intervention.py \
  tests/slimserve/test_rmsnorm_graph_audit.py tests/slimserve/test_attention_contracts.py \
  -q --junitxml=perf/results/2026-09-10/runtime-control/kv-loader-final-cpu.xml
```

Raw reports are `runtime-control/loader-hooks-cpu.xml` and
`runtime-control/kv-loader-{cpu,dispatch-cpu,regression-cpu,freeze-cpu,final-cpu}.xml`
under `perf/results/2026-09-10/`. Lint/diff checks pass. No native build, actual
GPU load, model start, serving-default/quant change or new TPS measurement.
The source-level adapter's prior GPU result remains valid historical evidence;
the new loader itself still needs actual all-rank AOT qualification.

Next implement a dedicated preparer/runner/auditor joining the completed adapter
and static attention-map receipts. Preserve the original AOT roots and all
non-target sources, copy the KV sources privately, reproduce their recorded
binaries from source, and freeze the new loader, shared hooks and audit dependencies. The geometry
preparer's thirteen-target assumptions do not apply. Prescribe command order,
failure handling and resource limits and commit before GPU loading. No new
GPU/model job is prescribed by this implementation checkpoint.

## Actual-AOT KV qualification v1: terminal; commands historical

The dedicated `prepare_glm53_kv_loader.py`, `check_glm53_kv_loader.py` and
`audit_glm53_kv_loader.py` now reuse the existing no-weights loading lifecycle,
artifact-root observer, binary checks and independent graph inventory. They
join the completed adapter and attention-map receipts without reviving their
old source freezes. The actual-cache CPU inspection verifies 5,172 original
files, one target source/two graph bindings per rank, seven AOT roots and 46
entries per rank. Inspection SHA
`5e3d5fe19d386a5ff3fc4c4077f2a603f1412f6cef9e45d9552d69feefd5b35d`,
`runtime-control/kv-aot-source-inspection.json`; this is not a prepared GPU run.

Final CPU gate: 471 passed in 42.52 s, 14 upstream deprecation warnings. This
includes real concurrent seven-artifact/46-entry store loading with fake CUDA
driver calls, no forwards/launches, and independent rejection of its deliberately
missing static-bundle coverage. Preparation copy/hash/no-overwrite checks,
driver/controller receipt joins, negative mutations and resource-limited
one-attempt launch/failure/audit behavior pass. Reports are
`runtime-control/kv-aot-{hooks,audit,runner,final}-cpu.xml`; prior reports retained.

After committing this protocol and implementation as Auroter:

- Prepare one NEW series `kv-aot-qualification-v1`, with eight independent private
  cache copies. Copy KV sources, not replacement cubins; reproduce the appended
  binary from source/config in the correct private rank cache. Freeze all new
  helpers, Torch code-cache/static-launcher code, native dependencies and receipts.
- Exactly eight GPU processes, sequentially: control ranks 0/1/2/3, then KV
  ranks 0/1/2/3. Each may run once, in 16 GiB/swap0; CPU preparation/controller/audit
  in 8 GiB/swap0. No model weights, forward, capture, timing or native build.
- Each process uses actual `store.load_all()`, with seven bundles, seven actual
  roots and 46 entries. Require no fallback and the original 25 bound launchers,
  including exactly two target graph bindings per rank. Candidate records the
  appended KV launch independently; non-target sources/configs/images must match
  that rank's control exactly. Seal bindings and verify again before exit.
- Hold recipe/native-order diagnostic settings fixed. Require the same four
  GPUs/UUIDs, driver and 600 W power limits as the completed adapter receipt,
  check before/after each process and require GPU release before the next.
- Audit every attempted process, including a failed load, preserving logs,
  markers and partial receipts. Stop the series at the first failure: no retry,
  replacement process or omitted failure. Confirm source/cache integrity and
  GPU release before ending a failed series' freeze. Do not edit during a live
  series. Keep every older failed series terminal.
- After all eight pass, compare their full receipts and frozen sources. Only
  then end this source freeze. This qualifies AOT loading/binding coverage,
  NOT numerical execution, model quality, production defaults or TPS. The
  separately failed indexer oracle remains failed. Model comparison is later.

Commands, once each from the repository root; the controller stops on failure:

```bash
systemd-run --user --scope --unit=glm53-kv-aot-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.prepare_glm53_kv_loader --output perf/results/2026-09-10/kv-aot-qualification-v1

systemd-run --user --scope --unit=glm53-kv-aot-v1-controller -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda-13.0 bash -c '
for kv_aot_mode in control kv; do
  for kv_aot_rank in 0 1 2 3; do
    .venv/bin/python -m benchmarks.kernels.check_glm53_kv_loader launch --manifest "perf/results/2026-09-10/kv-aot-qualification-v1/${kv_aot_mode}-rank${kv_aot_rank}/manifest.json" || exit "$?"
  done
done'

systemd-run --user --scope --unit=glm53-kv-aot-v1-compare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.audit_glm53_kv_loader compare perf/results/2026-09-10/kv-aot-qualification-v1
```

## v1 result and standalone-helper boundary fix

On `30ea612a6`, preparation completes. The first control-rank0 load fails at
`KVIntervention.replace()` before completing AOT loading; its offline audit
also exits 1, and the controller stops. Exactly one GPU process was attempted;
the other seven remain unattempted. All seven static bundles loaded exactly
(4/4/6/6/6/12/12 kernels), so this is not a bundle fallback or model-quality result.

The preserved traceback identifies `AsyncCompile.triton()`'s synchronous path:
it calls `load_kernel()` BEFORE `kernel.precompile()`. That imports the standalone
combo source via PyCodeCache. Its benchmark helper also exports `call`; the
controller mistook this uncompiled template for a finished model graph. One
actual graph binding had already succeeded through the static future path.

The fix binds only registered original AOT graph paths. It leaves the exact
standalone target source untouched after checking its path, source/debug hashes
and kernel symbol. Other non-root target modules still fail; actual graph/root,
config/binary/coverage and no-fallback gates remain intact. The CPU reproducer
covers the uncompiled helper, later finished roots, and wrong path/symbol/source
rejections. Initial 147 tests pass in 6.77 s; full 475 pass in 42.84 s (14 upstream
warnings). Final error messages include launcher/result counts and override state.

v1 is terminal, not retried. A final read-only check verified all 246 frozen
receipts and 5,172 original files, the sole failed attempt/seven unattempted cases,
GPU release and unchanged GPU/driver/power identity. Its source freeze ended
before edits. Raw `kv-aot-qualification-v1/control-rank0/`: load/audit logs,
partial root/module/binary/controller receipts, summary and failed analysis.
Analysis SHA `fe7113bacf2c8ba8d83aa932426b17a6e607b85d5c5362adcc24903c6c651217`;
launch SHA `edd66fa10d2b8565d6bbbd843c4514589d9867f06d970c01b68c548f55e412ae`.
CPU `runtime-control/kv-aot-{root-boundary,v2-final}-cpu.xml`. No weights,
forward, capture, timing, native build, default/quant change or gate relaxation.

## Actual-AOT KV qualification v2: completed; commands historical

After committing the helper-boundary fix, run the same eight-case order and
all v1 gates once in the NEW v2 namespace. This is a changed implementation,
not a replacement attempt in v1. Sources remain frozen from preparation to
terminal audit. The controller stops and audits at first failure; no retries,
extra starts, model forwards or performance claim. Exact commands:

```bash
systemd-run --user --scope --unit=glm53-kv-aot-v2-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.prepare_glm53_kv_loader --output perf/results/2026-09-10/kv-aot-qualification-v2

systemd-run --user --scope --unit=glm53-kv-aot-v2-controller -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda-13.0 bash -c '
for kv_aot_mode in control kv; do
  for kv_aot_rank in 0 1 2 3; do
    .venv/bin/python -m benchmarks.kernels.check_glm53_kv_loader launch --manifest "perf/results/2026-09-10/kv-aot-qualification-v2/${kv_aot_mode}-rank${kv_aot_rank}/manifest.json" || exit "$?"
  done
done'

systemd-run --user --scope --unit=glm53-kv-aot-v2-compare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.audit_glm53_kv_loader compare perf/results/2026-09-10/kv-aot-qualification-v2
```

## Completed all-rank AOT result

On `7b753a65a`, exactly the eight prescribed v2 GPU processes complete. Every
load and audit exits 0, then the cross-run comparison passes. Each rank/arm has
seven actual AOT roots and 46 entries, seven complete static bundles, 25 original
bound launchers and exactly two target graph bindings. The candidate's eight
target bindings across four ranks separately prove the original combo and
appended KV launch against qualified source/config/whole-cubin receipts. All
92 non-target graph bindings match their rank's control exactly. The exact
standalone-helper exclusion works without admitting it as a model root.

Final read-only closure verifies all eight successful launch records, 246 frozen
receipts and all 5,172 original cache files. GPUs are released; GPU UUIDs,
driver 580.173.02 and 600 W settings are unchanged. Freeze ended before notebook
edits. No retries within v2, omitted case, model weights/forward/capture, native
build or TPS measurement. v1's earlier failed attempt remains terminal and
preserved. CPU final remains 475 passed in 42.84 s, 14 upstream warnings.

Raw `perf/results/2026-09-10/kv-aot-qualification-v2/`: eight private prepared
manifests/caches, launch/load/audit records, actual root/module/graph/binary/
controller receipts and `pair-analysis.json`. Pair SHA
`e650a2a5f806085ab6f748169b1d54cf8950a2fd7656dfeda11d5e6f25a018f7`.
After further edits consume the pinned completion receipts, not these now-
historical source-frozen commands. No new production or model-quality claim.

Next integrate this qualified loader into an opt-in serving lifecycle, reusing
the existing profile validator and full quality/causal workload. Pass actual
root modules to KV seal/verify; remove only process-local observer indices when
comparing qualified bindings, including the appended launch's nested index.
Preserve all source/config/binary fields. Keep observation open for legitimate
non-target startup compilation, verify snapshots before forward and before/after
capture, and explicitly freeze/account for integration changes. Then prescribe
the bounded original/KV/return-original model series. No next model job has yet
been prescribed; the separate indexer oracle failure remains open.

## KV serving integration checkpoint

The opt-in serving integration is CPU-qualified: 628 pass/52.07 s (14 upstream
warnings), Ruff/diff checks pass. No GPU/model/native build in this stage.
Shared lifecycle now accepts KV's actual-root inventory and qualified loader;
separate event stream and independent appended-launch comparison work before
forward/across capture, while legitimate non-target compilation remains observed.
The qualified KV loader, adapter, compiler, live graph inventory and native
sources stay exact. The offline event checker gains a serving-only open-observer
option permitting exact duplicate graph callbacks after target seal; its original
no-weights mode remains globally sealed. This notebook is released only between
closed experiments. Both source changes are explicitly recorded by preparation.

Completed AOT receipts and 5,172 original files verify through pinned-data joins,
not historical frozen readers. Source inspection SHA
db8c8a056a7b44d7145593963c5f7d6a97574eb6b2197b9bb3663b891d8e0cd1.
CPU reports `runtime-control/kv-serving-{initial,lifecycle,integration,final}-cpu.xml`
retain all attempts, including the corrected CPU fixture path mismatch.

Next model series is now prescribed after the integration commit: exactly
control/KV/return-control once each, no replacements, same workload and quality
floors. Preparation begins the freeze; terminal closure ends it. Full commands,
resource limits, qualification boundaries and failure policy are in
`perf/glm53-kv-serving-protocol.md`. No TPS/default/quant or failed indexer-oracle
promotion follows from CPU tests.

## KV-only full-model result: rejected, reversible, not the no-combo vector

On `c42b72325`, exactly control/KV/return-control complete with all serve/audit exits
zero and complete final closure. All ranks pass before-forward and capture
binding checks; original/appended KV identities verify, all 92 non-target AOT
bindings match, and return-control restores the full original inventory.

Both controls reproduce historical per-token text and needle scores exactly in
all three repeats. KV repeats exactly but fails 15/32 unchanged quality windows
each time, and does not reproduce the failed no-combo vector. All six needle tests
still rank correct first. KV changes all 4,096 text scores: mean delta -0.00916196,
mean absolute delta 0.24100639, max absolute delta 5.25359750. Local <=1-ULP source
qualification was not sufficient for this full-model quality gate.

All 27 exact-token timing rounds, nine quality passes, text/image canaries and
cold 32K/128K tests retained. Diagnostic E2E c1/c8/c16 medians: control
157.332/580.279/778.425, KV 156.585/577.794/778.805, return
156.728/578.758/778.591 tok/s. No speed win or new baseline. No model retry,
replacement start, native build, quant/default change or relaxed quality floor.

Closure verifies 386 frozen receipts and 5,172 original files, GPU release and
unchanged hardware/driver/600 W settings. Freeze ended before notebook edits.
Raw `kv-serving-v1/closure.json`, SHA
a4e2ac3b120e3493cf11586a5b258123ad257d1de2cf67aacec8c1ace4478588,
under `perf/results/2026-09-10/`. Full commands, ranges and limitations:
`perf/glm53-kv-serving-protocol.md`. Do not rerun completed source-frozen readers.

Preserve original attention arithmetic for production tuning. KV alone is not
the full no-combo explanation; isolated model effects need not add linearly.
The separate indexer LayerNorm128 affine-cancellation oracle failure remains open
(max 5/28 BF16 ULP original/split). Inspect its retained worst-element evidence
before another replacement. No next GPU/model job prescribed yet.

## CPU indexer precision boundaries (2026-09-10)

Status: completed diagnostic; no kernel or serving promotion. Baseline remains
the source-exact probe's failed one-BF16-ULP oracle (original max5, split max28).
Hypothesis: the final affine's cancellation might be fixed by increasing only
affine precision. The experiment rejects that simple fix on the saved matrix.

`benchmarks/analyze_glm53_indexer_precision.py` joins four pinned completed
receipts, checks the entire 120-pair order and all-rank numerical equality, then
reconstructs the 30 unique cases and both input phases. Every packed-input hash,
all four real layer11 BF16 weight hashes and all 26 retained failing scalars
verify. The full FP64 control exactly matches the unchanged chunked oracle.
The original worst split point again has result 3.982235657895572e-7 and
cancellation ratio 2,883,902.4; its saved GPU result is not replaced by CPU data.

All models below round final outputs to BF16; 12,711,936 unique elements each.
CPU reductions, rsqrt and explicit dtype boundaries are **not** reproductions
of Triton trees, GPU instructions/contraction or saved kernel outputs.

| CPU arithmetic model | Outputs above 1 ULP | Failing phases /60 | Max BF16 ULP |
| --- | ---: | ---: | ---: |
| FP32 moments, normalization, separate affine | 21 | 11 | 22 |
| FP32 normalization, fused-affine emulation | 19 | 10 | 20 |
| FP32 normalization, FP64 affine | 19 | 10 | 20 |
| FP32 moments, FP64 recenter/rsqrt/normalization/affine | 9 | 7 | 8 |
| FP64 normalization rounded to FP32, FP64 affine | 8 | 6 | 5 |
| Full FP64 reference/control | 0 | 0 | 0 |

Fused-affine emulation evaluates the final product/add in FP64 and rounds to
FP32 before BF16. It and the FP64-affine model have identical BF16 outputs on
this matrix. That alone does not repair errors introduced earlier. Both the
FP32 moment boundary and FP32 normalized-value rounding can independently leave
failures even when subsequent arithmetic is FP64. This is not a proof that every
possible FP32 formulation fails or that wider arithmetic ensures model quality.

Next bounded hypothesis: use a cancellation detector to trigger extended-
precision recomputation for a small subset, preserving original outputs
elsewhere. First screen detector coverage and row/element fraction on CPU;
do not choose a tolerance after looking at a GPU result. Any eventual candidate
still needs GPU numerical, replay, mutation/stride guards, source/binary binding
and model-quality validation. The KV-only result shows why local ULP parity
cannot substitute for that model gate. No GPU/model job is prescribed here.

CPU tests: final 70 pass/2.94s (initial 47 pass/1.59s), including joins,
failed-gate preservation, rank drift,
input mutation/shape/dtype rejection, chunked FP64 control, scalar reconstruction
and exclusive output/failure retention. Ruff/diff checks pass. Resources: CPU
one Torch/OMP thread, 8 GiB, swap0, CUDA hidden and uninitialized. No native build,
production/default/quant change, speed measurement or quality-floor change.

Commands (CPU analysis performed once, on 95efefa6f plus hashed analyzer):

```bash
systemd-run --user --scope --unit=glm53-indexer-precision-cpu-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/slimserve/test_indexer_precision_analysis.py tests/slimserve/test_attention_norm_probe.py --junitxml=perf/results/2026-09-10/runtime-control/indexer-precision-cpu-v1.xml
systemd-run --user --scope --unit=glm53-indexer-precision-analysis-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.analyze_glm53_indexer_precision --output perf/results/2026-09-10/runtime-control/indexer-precision-analysis-v1.json
systemd-run --user --scope --unit=glm53-indexer-precision-cpu-final -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/slimserve/test_indexer_precision_analysis.py tests/slimserve/test_attention_norm_probe.py tests/slimserve/test_cached_rmsnorm_probe.py tests/slimserve/test_attention_contracts.py --junitxml=perf/results/2026-09-10/runtime-control/indexer-precision-cpu-final.xml
```

Raw `runtime-control/indexer-precision-analysis-v1.json` under 2026-09-10, SHA
358f4a155dbeea926e4e6e3f7b192c8c9e3ded3a0f4ee5b857d1951af7502764;
per-case metrics, output hashes, failing coordinates and scalar model values
retained. Historical GPU receipt hashes are pinned in the analyzer and report;
their completed audits are consumed without rerunning expired freeze checks.

## Cancellation detector CPU screen (2026-09-10)

Status: completed coverage/cost proxy, not a recomputation kernel qualification.
The analyzer's `--screen-cancellation` option prescribes thresholds 2^-16, 2^-12
and 2^-8 before execution. For BF16 output `y` and bias `b`, the FP32 detector is
`abs(y) <= threshold * (abs(y - b) + abs(b))`, excluding zero scale. No reference
or input moments enter this predicate. Exact zero output with nonzero bias is
flagged; zero output and zero bias is not affine cancellation.

All three thresholds detect all 12 original and 14 split retained failing GPU
scalars after rank deduplication, and all 21/19 above-one-ULP failures in the
separate/fused-affine FP32 CPU models. The full 60-record CPU precision baseline
(including numerical metrics, output hashes and scalar evidence) reproduces
exactly against its pinned completed receipt. No GPU source is imported/run.

| Threshold | CPU selected elements, separate/fused | CPU selected rows, separate/fused | Row fraction |
| --- | ---: | ---: | ---: |
| 2^-16 | 102 / 102 | 102 / 102 | 0.103% / 0.103% |
| 2^-12 | 1,431 / 1,430 | 1,424 / 1,423 | 1.434% / 1.433% |
| 2^-8 | 24,499 / 24,499 | 21,734 / 21,734 | 21.885% / 21.885% |

Denominators: 12,711,936 elements, 99,312 rows per CPU model. Actual GPU evidence
only supplies failing scalar locations, not all output values: the selected
element/row counts are CPU proxies, not actual GPU counts or execution cost.
All thresholds retain zero missed errors on this matrix; none is a proven bound
for unseen inputs. The original GPU oracle gate is still failed. Substituting
oracle values at flagged positions would be tautological, not a kernel test,
and is not reported as a corrected-kernel result.

Decision: use fixed 2^-12 for the next isolated prototype, a 16x wider trigger
than the tightest tested threshold at a ~1.43% row proxy cost. First preserve
the original bundle, then use actual original BF16 indexer output to flag rows.
Recompute flagged rows' moments/normalization/affine without the failing FP32
rounding boundary; overwrite only flagged elements. Q/KV, the packed gate,
input/weight guards and all unflagged indexer outputs must remain exact. CPU
layout/import tests precede a separately prescribed GPU numerical/replay/guard
and timing matrix. A serving experiment needs those gates and the independent
model-quality gate; no default/profile integration follows from this screen.

Source: `cc9a508ea` plus hashed screen changes. CPU73 tests pass/3.54s; Ruff/diff
pass. One CPU screen, 8 GiB/swap0, one thread, CUDA hidden/uninitialized; no
GPU/model starts, native builds, TPS, default/quant changes or tolerance changes.

```bash
systemd-run --user --scope --unit=glm53-indexer-cancellation-cpu-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/slimserve/test_indexer_precision_analysis.py tests/slimserve/test_attention_norm_probe.py tests/slimserve/test_cached_rmsnorm_probe.py tests/slimserve/test_attention_contracts.py --junitxml=perf/results/2026-09-10/runtime-control/indexer-cancellation-cpu-v1.xml
systemd-run --user --scope --unit=glm53-indexer-cancellation-analysis-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.analyze_glm53_indexer_precision --screen-cancellation --output perf/results/2026-09-10/runtime-control/indexer-cancellation-analysis-v1.json
```

Raw `runtime-control/indexer-cancellation-analysis-v1.json` under 2026-09-10,
SHA3474f6692e98b1be257174827dd5387d8d08dedadd647f8c2935a0a18ec25a24.
CPU XML and all per-phase detector counts/missed-coordinate lists retained.

## Selective correction GPU result (2026-09-10)

The follow-up on `f5ad4f884` completes exactly one all-rank process,120 cases,
both input phases. Corrected indexer max1 BF16 ULP passes the unchanged gate;
actual detector coverage, original hashes, all non-target/unselected outputs,
mutation/stride/selection guards and graph replay pass. All ranks agree exactly.
Unique-matrix GPU counts:1,430 selected elements,1,423 selected rows,121 changed
elements. No original failing location missed. The original historical oracle
failure is not relabeled; this is a new corrected-kernel qualification.

Diagnostic extra-launch median cost is0.696/0.746/3.253/4.444us at
rows1/16/640/7616, including selection writes. No speed win or serving result.
Model quality and actual graph-loader/forward/capture integration remain pending;
preserve the qualified instrumented binary and original production arithmetic.

Full protocol, timing ranges, sources and limits:
`perf/glm53-indexer-correction-protocol.md`. Terminal audit verifies236 receipts,
all5,172 original files and unchanged hardware; GPUs released and freeze ended.
Raw `indexer-correction-v1/analysis.json` under2026-09-10, SHA
c6ac3d0399af92be467ef47831f512c4edee60ff08fe0772afb399c59a18cc65.
