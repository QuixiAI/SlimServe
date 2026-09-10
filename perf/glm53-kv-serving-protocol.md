# KV-only real-model causal comparison

Completed on `c42b72325`: exactly control/KV/return-control, all audits and final
closure pass. KV fails 15/32 unchanged quality windows, repeating exactly; both
controls reproduce original per-token scores exactly. Candidate rejected. No
production/default/quant promotion or speed win. Source freeze ended. The commands
below are historical, not instructions to rerun a completed attempt.

## Hypothesis and fixed scope

The split attention path changes KV normalization by small BF16 amounts. The
qualified diagnostic launches the original eleven-argument attention combo,
then overwrites only KV with the exact split kernel on the same stream. Q and
indexer output remain original. This tests full-model causality, not speed:
the additional launch is not a proposed production optimization.

Baseline: original-policy `glm53-nvfp4-4` / `rtx6000`, fixed recipe
`glm53-redhatai-nvfp4-fp8-kda-tp4-v1`, TP4 Marlin, no EP or speculation,
BF16 activation/KV/lm_head, original attention combo/compiler settings.
Native-order1/BF16fn1/TC0 are diagnostic settings; production native-order stays
off. No quant, clock, power, native binary, quality floor or indexer-oracle change.

Completed evidence:

- Actual-AOT v2 pair SHA
  `e650a2a5f806085ab6f748169b1d54cf8950a2fd7656dfeda11d5e6f25a018f7`,
  `perf/results/2026-09-10/kv-aot-qualification-v2/pair-analysis.json`.
  Exactly eight prescribed loads/audits passed, seven roots/46 entries and
  25 original launchers each; two target bindings per rank and 92 unchanged
  non-target bindings across ranks. Each launch records GPU release and identical
  before/after UUID/driver/600 W configuration. There is no separate closure file.
- Adapter SHA
  `bfaa7a495e7b69f228661a4402f5a6d5229d53b6950b6e7ee8cfa6275975144d`:
  120 numerical/replay/guard cases, split KV exact, Q/indexer unchanged.
  These are not full-model numerical or performance qualification.
- Earlier terminal AOT v1 and rejected geometry model comparison stay retained.
  The separate indexer LayerNorm oracle remains failed.

## Serving implementation and evidence boundaries

`SLIMSERVE_GLM53_KV_DIAGNOSTIC=control|kv` and
`SLIMSERVE_GLM53_KV_MANIFEST` select a separate schema; unset is inert. Reject
legacy/geometry diagnostics, compiler-policy changes, wrong recipe or hardware.
Before forward and before/after capture, compare actual registered AOT-root
bindings to completed qualification. Preserve original and appended KV
source/config/whole-cubin identity. Ignore only process-local observation indices.
Target bindings seal before forward, but binary observation stays open for
legitimate non-target startup compilation. No token-loop checks are added.

Reuse geometry's lifecycle, workload, worker audit and one-attempt controller;
inject only the KV loader/inventory and separate controller-event audit. Controller
events after sealing may only repeat an exact previously observed graph binding.
Fresh cache contains the complete original namespace plus four qualified KV
source copies, no replacement cubins. Qualified loader/compiler/adapter/native
sources stay exact. The offline KV auditor's serving lifecycle option and the
closed attention notebook are the only explicitly released former sources.
All new integration/client/protocol sources and completed evidence are frozen.

## Executed workload and terminal conditions

Exactly control / KV / return-control, one start and private namespace each,
three measurements at each c1/c8/c16, exact1000 input/300 output, cold-prefix;
three quality repetitions, text/image canaries, cold32K/128K prefill. Use the real
registered profile through the existing campaign client. Hardware and workload
identities must match the pinned original reference. Keep every failed/slow result.

Both controls must reproduce historical per-token text and needle scores exactly.
Every arm must repeat within-start exactly. Evaluate all unchanged quality floors.
A repeated candidate quality failure may be retained as a diagnostic observation
and followed by the prescribed return-control; it is never a quality/performance
pass or production promotion. Stop on any other validity/load/teardown/audit
failure; audit partial receipts, never retry or replace a start. Cross-arm
non-target bindings and return-control's complete bindings must match exactly.

CPU preparation/controller/tests/audits:8 GiB, swap0. Serving:150 GiB, swap0.
One GPU workload at a time; no native build. Freeze sources from preparation until
terminal source/cache/receipt/hardware/release checks complete. Any interruption
stops only the uniquely named owned scope. Record complete or terminal-failure
closure, preserve all raw files and only then end the source freeze.

## CPU gate and executed v1 commands (historical)

CPU reports under `perf/results/2026-09-10/runtime-control/`:
`kv-serving-initial-cpu.xml` retains the eleven fixture cache-path failures;
the fixture used `gg/graph*.py`, whereas real Torch derives `ra/graph*.py`.
Correcting only the fixture gives 19 pass/5.97 s. Integration 145 pass/41.04 s,
final 628 pass/52.07 s. Ruff and diff whitespace checks pass. No qualified loader,
compiler, adapter, graph inventory or native code changed.

Read-only completed-evidence inspection passed and verified all 5,172 original
files; no caches copied or source freeze begun. Receipt
`kv-serving-source-inspection-v1.json`, SHA
`db8c8a056a7b44d7145593963c5f7d6a97574eb6b2197b9bb3663b891d8e0cd1`.
This predates final integration formatting/tests; preparation freezes the final
committed implementation, not that inspection's integration hashes.

After committing as Auroter, run from the repository root:

```bash
systemd-run --user --scope --unit=glm53-kv-serving-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.kernels.prepare_glm53_kv_serving --output perf/results/2026-09-10/kv-serving-v1

systemd-run --user --scope --unit=glm53-kv-serving-v1-controller -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash -c '
for kv_serving_case in control kv return-control; do
  .venv/bin/python -m benchmarks.kernels.run_glm53_geometry_serving launch "perf/results/2026-09-10/kv-serving-v1/${kv_serving_case}/manifest.json" || exit "$?"
done'

systemd-run --user --scope --unit=glm53-kv-serving-v1-close -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.kernels.run_glm53_geometry_serving close perf/results/2026-09-10/kv-serving-v1
```

The historical geometry filename is the shared controller/workload, not the
selected policy: the KV manifest schema selects KV flags, order, graph auditor
and closure result keys. Before the first model process, a CPU-only direct-client
check may stop immediately after real profile/manifest validation, before the
tokenizer or model. It must not consume or modify any case attempt/cache.

Run closure after either completion or first failure, without starting unused
cases. It checks frozen sources/original files, preserved attempted-case files,
GPU release and qualified UUID/driver/power; model success additionally requires
all worker and workload audits. Stop if another GPU workload is active. No edits,
builds or commits during this series. Commit measured results only after closure.

## Completed v1 result

Exactly three prescribed model processes on `c42b72325`, one per arm, through the
real `glm53-nvfp4-4` / `rtx6000` profile. All serve/audit exits are zero; final
closure is complete. Each rank verifies seven AOT roots/46 entries, 25 original
launchers and two target graph bindings before forward and before/after capture.
The candidate's original combo and appended KV binary identities both verify.
All 92 non-target AOT bindings match; return-control restores the complete original
binding inventory. This is AOT-root coverage, not an assertion that every later
non-root runtime kernel was independently compared across processes.

All 27 exact1000/300 timing rounds (225 requests), nine quality passes, text/image
canaries and cold 32K/128K tests complete. Both controls reproduce all historical
per-token text/needle scores exactly, including all repetitions. KV's three
complete score vectors are also identical to one another, but each fails 15 of
32 unchanged quality windows. All six needle tests still rank the correct answer
first. Mean text logprob: original -2.727814820100083, KV -2.7369767803805773.
All 4,096 text and 168 needle scores change; text mean absolute delta 0.2410064,
max absolute delta 5.2535975. Small aggregate mean drift is not proof of stable
individual predictions. KV does not reproduce the failed no-combo vector.

Diagnostic E2E tok/s, median [min, max] of three repetitions per cell:

| Arm | c1 | c8 | c16 |
| --- | ---: | ---: | ---: |
| Control | 157.332 [157.063,157.369] | 580.279 [577.965,580.348] | 778.425 [776.320,781.881] |
| KV | 156.585 [156.438,156.748] | 577.794 [576.835,578.856] | 778.805 [777.704,781.233] |
| Return-control | 156.728 [156.487,156.921] | 578.758 [578.743,579.569] | 778.591 [772.706,781.845] |

Cold prefill engine scheduled-to-first-token ms, median [min, max]:

| Arm | 32K | 128K |
| --- | ---: | ---: |
| Control | 2580.919 [2577.827,2582.692] | 10880.784 [10844.034,10916.345] |
| KV | 2587.057 [2584.781,2589.216] | 10908.757 [10869.901,10942.119] |
| Return-control | 2585.712 [2583.806,2588.888] | 10903.597 [10865.301,10939.129] |

Startup 164.083/162.111/160.092 s in prescribed order. The extra-launch candidate
has no speed win and fails quality; retain only as quarantined diagnostic code.
No new baseline, native build, production/default/quant change or quality-floor
relaxation. The independent indexer LayerNorm128 oracle remains failed.

Closure verifies all 386 frozen source/evidence receipts, 5,172 original files,
preserved per-case inventories, GPU release and identical UUID/driver/600 W
configuration. Source freeze ended before recording these results. Raw root
`perf/results/2026-09-10/kv-serving-v1/`, closure SHA
`a4e2ac3b120e3493cf11586a5b258123ad257d1de2cf67aacec8c1ace4478588`.
All three manifests, caches, launch logs, worker graph/binary/lifecycle receipts,
workload JSON and audits remain retained. No model retries or replacement starts.

Before model work, an optional CPU-only `runpy` wrapper omitted the direct script's
`benchmarks/` import directory and failed importing `benchmark_dsv4_exact`.
Correcting only that wrapper let all three real client/profile/manifest checks
pass before output creation/tokenizer/model. Neither invocation consumed a case
or changed frozen sources/caches. Both `client-preflight-v{1,2}.json` records are
retained. This was not a serving restart or a kernel/source change.

Decision: retain original attention arithmetic for production tuning. The KV-only
intervention is a reversible contributor to model score changes on this workload,
not the full no-combo explanation. Do not assume arithmetic effects are additive
across H4096/KV/indexer changes. Next inspect the retained indexer affine-cancellation
evidence and precision choices under the existing oracle gate, before qualifying
another replacement. No further GPU/model process is prescribed by this result.
