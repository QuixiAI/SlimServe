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

## Actual-AOT KV qualification v1: fixed protocol

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
