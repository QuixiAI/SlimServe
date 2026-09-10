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
