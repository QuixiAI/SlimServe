# GLM53 SM120: remaining KDA choices

## CPU inventory, 2026-09-10

The completed RMSNorm geometry series rejected its candidate. Its original
settings remain; no failed series is reopened and no quality gate changes.
`benchmarks/analyze_glm53_kda_choices.py` reads completed receipts without
re-running obsolete HEAD/source-freeze validators or importing generated code.

All 72 KDA tuning files in control/geometry/return-control match the 24 original
rank-private files exactly. All six kernel tuning keys and ordered candidate
sets match those in the failed no-combo run's shared cache. Disk winners differ
in 19/24 rank/kernel pairs across five kernels:

| Kernel | Original ranks 0 / 1 / 2 / 3 (warps, stages) | Fresh shared disk winner |
| --- | --- | --- |
| Gate + chunk cumsum | 8,3 / 8,3 / 8,3 / 8,3 | 2,3 |
| Intra sub-chunk | 2,3 / 2,2 / 2,2 / 2,2 | 2,4 |
| Inter solve | 4,3 / 4,3 / 4,3 / 4,3 | 4,3 |
| Recompute W/U | 4,3 / 4,3 / 4,3 / 4,3 | 8,3 |
| Recurrent chunk state | 4,2 / 4,2 / 4,2 / 4,2 | 4,3 |
| Output | 4,2 / 8,2 / 8,4 / 8,2 | 4,2 |

Block dimensions agree: gate BS32, inter BK64, state BV32, output BK64/BV64.
Selection follows the installed Triton reader's lexicographic timing minimum,
including first-entry ties. These tuning times are not serving measurements.

Important evidence boundary: the original files are pinned by the completed
series. The failed fresh shared files are hashed now, not historically pinned
per rank. A shared file can conceal different in-memory choices made by concurrent
workers. Neither disk inventory proves which KDA config each worker actually
launched. No KDA causal claim follows from the geometry graph-binding audit.

The actual CUDA serving import is `kimi_k3/nvidia/ops/third_party/kda`, which
re-exports `kimi_k3/amd/ops/third_party/kda`. It is NOT the similarly named
`third_party/flash_linear_attention/ops/kda.py`. KDA prefill starts with the fused
gate/cumsum, followed by intra/solve, recompute, state and output. Decode uses a
separate packed recurrent kernel; the six settings above are prefill choices.

CPU regression: 16 passed in 0.09 s (8 GiB, swap0, GPUs hidden). Raw report:
`perf/results/2026-09-10/runtime-control/kda-choices-cpu.xml`.
Inventory `kda-disk-choice-analysis.json` in that directory, SHA
744500910b0930425294a1cf425cc05bf51a7a59a5f31d59639c2515f525984b.

## Next isolation (preparation only; no GPU command prescribed yet)

Test the earliest changed operation: the serving gate/cumsum source at BS32,
three stages, eight versus two warps. Synthetic BF16 gate inputs and packed beta,
real layer0 FP32 repaired A_log/dt_bias, TP4 shards (16 heads x128), lower bound-5,
chunk64, variable-length boundaries. Compare FP32 gate and beta outputs, eager
repetition, changed-input graph replay and a float64 mathematical reference.

This is an arithmetic diagnostic, not reconstruction of unrecorded historical
live winners, a model-quality gate, or a throughput qualification. Record all
differences; preserve the failed indexer gate and every existing serving floor.
Finalize the fixed matrix, source checks and one-process command after CPU tests.
No full-model run, source/default/native/quant change or autotuning is authorized
by this preparation note.

## Gate arithmetic v1: prescribed one-process matrix

CPU tests exercise the actual serving JIT import, full evidence preparation,
float64 oracle boundaries, exact launch arguments and audit rejection paths:
18 passed in3.86s, 8GiB/swap0, GPUs hidden. Raw `runtime-control/kda-gate-cpu.xml`.
Final combined regression:22 passed3.54s, `runtime-control/kda-gate-final-cpu.xml`.
The gate source equals both59ae0c88f (failed fresh) and ce6df61aa (geometry series).
This does not reconstruct their unrecorded live KDA choices or binary identities.

After committing this protocol and final CPU tests, prepare once and execute ONE
new GPU0 process, fixed order: rank0..3, then layouts, seeds and magnitudes below.
Both arms use the exact current serving JIT, direct explicit launch (no autotuner):
eight warps first, then two; BS32/BT64/stages3,16 heads x128, lower_bound-5.
The synthetic packed beta stride is6528, offset6144 (128 reserved beta rows,
only first16 live), not the historical pre-g_a-fusion6288 projection width.

- Layouts: single sequences1,63,64,65,1000; ragged17/63/65/855; single7616.
- Seeds530901/530902; BF16 random magnitudes0.125/1/8.
- Real layer0 repaired FP32 A_log/dt_bias, each TP4 shard;168 paired cases.
- Two eager repetitions, original-input graph replay, then in-place input change
  (+0.125 gate/-0.25 beta) and replay/eager comparison on the same addresses.
- Input/packed-sibling/output guards unchanged; repeat and replay bit-exact.
- Gate and beta mathematical oracle uses float64, chunk/sequence resets and
  exp/sigmoid/cumsum. Diagnostic FP32 integrity gate:
  abs(error) <=1e-6 +2e-6*abs(reference), every element.
  This new FP32 probe criterion does not change any BF16/indexer/model gate.
- Record pairwise FP32 differences, output hashes, actual compiled cubin bytes,
  source/compiler/repair hashes and device/driver/power identity. Cross-arm exact
  equality is an observation, not required. No performance timings are taken.
- Stop at first structural/replay/mutation/oracle failure; preserve partial
  results and always audit/close. No retries, replacements or omitted cases.
- Freeze sources from preparation through audit; no model or other GPU job and
  no native build. GPU process16GiB, CPU preparation/audit8GiB, all swap0.

From repository root, after commit (each command once):

```bash
systemd-run --user --scope --unit=glm53-kda-gate-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_gate prepare --manifest perf/results/2026-09-10/runtime-control/kda-gate-v1-manifest.json
systemd-run --user --scope --unit=glm53-kda-gate-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-13.0 .venv/bin/python -m benchmarks.kernels.check_glm53_kda_gate run --manifest perf/results/2026-09-10/runtime-control/kda-gate-v1-manifest.json --output perf/results/2026-09-10/kda-gate-v1
systemd-run --user --scope --unit=glm53-kda-gate-v1-audit -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_gate audit --manifest perf/results/2026-09-10/runtime-control/kda-gate-v1-manifest.json --output perf/results/2026-09-10/kda-gate-v1
```

If gate outputs differ, this identifies a source of local numerical sensitivity,
not its contribution to model scores. If exact, continue down the remaining KDA
stages with new evidence-based isolation. No full-model start/default promotion
follows automatically from either result.
