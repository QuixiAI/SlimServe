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
