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

## Gate isolation (historical preparation)

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

## Gate arithmetic v1: completed one-process matrix (commands historical)

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

## Completed v1 result

On c7b4bb4d1, exactly one GPU0 process completed all168 pairs, with no retries,
replacements, tuning or omitted cases. Probe and CPU audit exit0. All74 frozen
source/evidence hashes and both actual compiled cubins verify; GPUs released and
GPU/driver/power identity unchanged. The source freeze ended after successful
audit. Consume the completed receipt after later documentation/source changes,
not the old frozen-manifest validator.

Eight versus two warps are BIT-EXACT on all964,263,936 paired FP32 gate values and
7,533,312 beta values (both input phases). Eager repeats, graph replay, changed-input
replay, packed-input mutation and output-guard checks pass. Every element passes
the prescribed float64-reference criterion. Maximum gate absolute error is
9.869150130725757e-5; maximum normalized error/allowed bound0.35640318234308466.
Beta maxima9.106284626358985e-8 and0.03896237065973383 respectively.

Decision: this gate geometry change produced no numerical difference on the fixed
matrix. This does not prove equality on all inputs or identify the cause of the
failed model scores. No speed measurement or production/default/quant promotion.
All existing indexer/no-combo/model gates remain unchanged.

Raw `perf/results/2026-09-10/kda-gate-v1/`:168 pair records, summary, two cubins,
private compiler cache, attempt marker and analysis. Analysis SHA
dbb92de6bc14f00c86ba8d19a8a680ab8d5ad82ed696906d300a3432e5d9bf70;
summary SHAd0ad46d69d1cc1fa1e91b85363496ab908c0e17e5741c258606a0763b651f063.

An additional CPU screen avoids a redundant next GPU test: the retained intra
sub-chunk candidates at warps2/stages2,3,4 have IDENTICAL whole TTIR/PTX/cubin bytes.
Cubin SHAdd1ff07093228222878f8cb132c9fa3d8f441c1d3c4167448be1d0736fdae928.
This applies to those exact compiled candidates, not unrecorded historical live
bindings. Raw reader/report: runtime-control/screen-kda-intra-stages.py and
runtime-control/kda-intra-stage-screen.json under2026-09-10.

NEXT: inspect remaining recompute W/U, recurrent-state and output choices. Screen
same-signature binaries first (especially stage-only changes), then prescribe the
smallest still-informative numerical comparison. Do not rerun the completed gate
matrix or a full model just to retest unchanged work. No next GPU job prescribed.

## Recompute v1 completed (preparation/protocol commands historical)

CPU screening finds genuinely different retained binaries for recompute4/8-warps
and state-update4-warps/stages2/3. Output4/8-warps also differs; its stage-only
choices are byte-identical within each warp count. Seven exact TTIR groups/36
candidate images are recorded in `runtime-control/kda-remaining-binary-screen.json`
by the preserved `screen-kda-remaining.py`. As before, cache presence does not
establish historical live choices or numerical causality.

Next numerical test isolates the earliest remaining operation: actual serving
`recompute_w_u_fwd_kernel`, four versus eight warps, stages3/BK64/BV64/BT64 fixed,
16 heads and K=V=128. No other KDA stage runs in this probe. The source matches
both model-series commits. No autotuner, generated-source substitution, native
build, model start or production change.

Prepare and run once AFTER commit and final CPU tests. One GPU0 process,196 pairs:
CPU final45passed4.45s (prior26passed4.50s), reports
`runtime-control/kda-recompute-{cpu,final-cpu}.xml`.
rank0..3, each of the seven gate-probe sequence layouts, then one identity case
(seed530901/magnitude1) and six conditioned cases (seeds530901/530902 x magnitudes
0.125/1/8). K is normalized random BF16; V is signed random BF16. A is BF16 identity
or identity plus0.015625-scaled signed strict-lower-triangular noise per64-token
chunk. This is a controlled stable triangular fixture, NOT captured model data.
Conditioned FP32 gate/beta fixtures come from the float64 formula and real layer0
FP32 repaired TP4 shard; identity fixtures use g=0/beta0.5.

Two eager repetitions, original-input graph replay, then k*=0.5/v*=-0.5 on the
same addresses and changed-input graph/eager comparison. Guard every BF16 output;
all inputs must remain unmodified except those prescribed host-side changes.
Record both actual compiled cubins, source/repair/compiler receipts, original and
changed-input output hashes, pairwise BF16 errors and float64-reference errors.

Identity cases MUST be exact to w=k*0.5/u=v*0.5/kg=k, before and after mutation.
The batched float64 reference has separately tested chunk/sequence boundaries and
BF16 operand round points. Its errors on conditioned cases are observations, NOT
a new accuracy tolerance or a replacement for the failed indexer/model gates.
Cross-arm equality is likewise an observation. No speed or production qualification
from this test. Stop on first structural/finite/replay/guard/identity failure, retain
partial records, and always audit/close. No retries, replacements or exclusions.

The existing BF16 comparison helper now accepts a positive chunk size; its default
128 is unchanged. The new GPU-only probe uses8192 rows per metrics tile to avoid
millions of diagnostic launches. CPU tests require identical comparison metrics.
Gate preparation receipt collection is shared; the completed gate v1 run is NOT
repeated, and its old source-frozen validator is no longer current.

Freeze from preparation through audit. GPU16GiB, CPU8GiB, all swap0. After commit,
each command once from repository root:

```bash
systemd-run --user --scope --unit=glm53-kda-recompute-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_recompute prepare --manifest perf/results/2026-09-10/runtime-control/kda-recompute-v1-manifest.json
systemd-run --user --scope --unit=glm53-kda-recompute-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-13.0 .venv/bin/python -m benchmarks.kernels.check_glm53_kda_recompute run --manifest perf/results/2026-09-10/runtime-control/kda-recompute-v1-manifest.json --output perf/results/2026-09-10/kda-recompute-v1
systemd-run --user --scope --unit=glm53-kda-recompute-v1-audit -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_recompute audit --manifest perf/results/2026-09-10/runtime-control/kda-recompute-v1-manifest.json --output perf/results/2026-09-10/kda-recompute-v1
```

Next action depends on the completed result; no full-model run prescribed.

## Completed recompute v1 result

On3bd347098, the one prescribed GPU0 process completes196/196 pairs (28 identity,
168 conditioned), then its CPU audit passes. No retry, exclusion or replacement.
Both actual cubins and81 frozen receipts verify; GPU release and unchanged
device/driver/power identity verify. Source freeze ended after closure.

The four/eight-warp settings produce BIT-EXACT W, U and KG outputs on both input
phases:1,124,974,592 compared BF16 values per output,3,374,923,776 total. Every
eager/replay/mutation/guard check and every exact identity oracle passes. This is
local arithmetic evidence, not proof for every input or unrecorded model launches.

Conditioned-case float64 reference errors remain observations, as prescribed:
maximum rounded-reference BF16 ULP distances W/U/KG27616/67/49, maximum absolute
errors0.0011707544/0.1209889725/0.0009765638. Large ULP distances around cancellation
or tiny values cannot be interpreted from those aggregate maxima alone. Do not
call these random fixtures oracle-qualified or waive any existing accuracy gate.
Both arms have identical outputs, so this probe does not implicate the warp choice
in those reference differences. No production change or TPS measurement.

Raw `perf/results/2026-09-10/kda-recompute-v1/`: all196 records, attempt marker,
summary, actual cubins and private compiler cache. Closure analysis SHA
eb997c612a0347924b5a66b49c11a90ed969ba37e10ae5d6ac8ce4bf494fd6a9;
summary SHAe49d798109bd60d10d0d3c9cf4595ba4a56e0f4025a4e48160c9902e1f939515.
Consume these completed receipts after further source/protocol edits, not v1's
now-historical source-frozen validators. All failed model/indexer gates unchanged.

NEXT: isolate state update (warps4/BV32/stages2 versus3), then output (BK64/BV64,
warps4 versus8; retained same-warp output stage-only variants are byte-identical).
State outputs include BF16 chunk snapshots/new values and FP32 final state; do not
reuse a BF16-only checker for all three. The actual serving state call supplies
an initial-state tensor even for zero-state sequences, varlen chunk offsets, GK
gating and USE_EXP2=True. Source is `third_party/flash_linear_attention/ops/chunk_delta_h.py`.
Use exact algebraic cases plus recorded numerical comparisons; no new model gate.
No next GPU/model job was prescribed at that checkpoint.

## State/output v1: completed sequential isolation (commands historical)

After committing this protocol and passing CPU tests, prepare/run/audit state
ONCE, then prepare/run/audit output ONCE only if the state audit is complete.
Freeze all sources from state preparation through output closure (or terminal
state failure). No intervening source edits, commits, native builds, autotuning,
model starts or other GPU work. CPU preparation/audits use 8 GiB, each GPU0 process
16 GiB, all swap0. Do not retry, replace, omit or restart a failed case/attempt.

Both stages use `benchmarks.kernels.check_glm53_kda_tail` with shared fixtures in
`glm53_kda_tail.py`. Each has 224 paired cases: ranks 0..3, the seven layouts
1 / 63 / 64 / 65 / 1000 / (17,63,65,855) / 7616, then identity, basis, and six
conditioned cases (seeds 530901/530902 x magnitudes 0.125/1/8). Identity/basis use
seed 530901 and magnitude 1. Thus 56 exact algebraic and 168 conditioned cases
per stage. Real repaired layer0 TP4 gate shards are used for conditioned fixtures;
these are mathematical/synthetic inputs, NOT captured model activations.

- State: actual `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, warps4/BV32,
  stages2 then stages3, H=Hg16/K=V128/BT64. Varlen offsets, initial state supplied,
  GK and exp2 enabled, scalar G disabled; save new values and final state. Compare
  BF16 chunk snapshots/new values and FP32 final state without narrowing it.
  Identity keeps nonzero state with K=W=0; basis accumulates one-hot K/V with W=0,
  checking chunk/sequence resets and up to 60 exactly representable visits.
  Conditioned K/W/V derive from the earlier recompute fixture's mathematical
  reference; seed 530902 also supplies nonzero initial state.
- Output: actual serving `chunk_gla_fwd_kernel_o`, warps8 then warps4, stages2,
  BK64/BV64/BT64, H16/K=V128/scale=128**-0.5. Identity uses Q0/Aidentity;
  basis uses one-hot Q/A0/identity chunk state, respecting BF16 scale rounding.
  Conditioned inputs use normalized Q, signed triangular A and BF16 chunk state.
- Each arm runs twice eagerly, captures/replays original inputs, then multiplies
  V and initial/chunk state by positive 0.5 on the SAME addresses and compares
  changed-input replay with eager. Inputs/guards must be intact, outputs finite,
  repetition/replay exact and output hashes changed by the prescribed mutation.
- Exact fixtures must match independent algebraic oracles in both phases.
  Conditioned float64-reference errors and cross-config differences are recorded
  observations, NOT new accuracy thresholds or full-model qualification. No TPS
  is measured. No existing indexer/no-combo/model gate or default is changed.
- Preserve every record, output hash/dtype/shape receipt, actual cubin, private
  cache and attempt marker. Stop on first structural/replay/guard/exact failure;
  audit the partial attempt as terminal. Output preparation independently checks
  all 224 state records, binaries, summary and frozen predecessor manifest, not
  just a status label. A terminal state failure forbids the output attempt.

From repository root, after commit, each command at most once:

```bash
systemd-run --user --scope --unit=glm53-kda-state-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_tail prepare --stage state --manifest perf/results/2026-09-10/runtime-control/kda-state-v1-manifest.json
systemd-run --user --scope --unit=glm53-kda-state-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-13.0 .venv/bin/python -m benchmarks.kernels.check_glm53_kda_tail run --manifest perf/results/2026-09-10/runtime-control/kda-state-v1-manifest.json --output perf/results/2026-09-10/kda-state-v1
systemd-run --user --scope --unit=glm53-kda-state-v1-audit -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_tail audit --manifest perf/results/2026-09-10/runtime-control/kda-state-v1-manifest.json --output perf/results/2026-09-10/kda-state-v1
```

Only after complete state closure and GPU release:

```bash
systemd-run --user --scope --unit=glm53-kda-output-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_tail prepare --stage output --predecessor perf/results/2026-09-10/kda-state-v1/analysis.json --manifest perf/results/2026-09-10/runtime-control/kda-output-v1-manifest.json
systemd-run --user --scope --unit=glm53-kda-output-v1 -p MemoryMax=16G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-13.0 .venv/bin/python -m benchmarks.kernels.check_glm53_kda_tail run --manifest perf/results/2026-09-10/runtime-control/kda-output-v1-manifest.json --output perf/results/2026-09-10/kda-output-v1
systemd-run --user --scope --unit=glm53-kda-output-v1-audit -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= .venv/bin/python -m benchmarks.kernels.check_glm53_kda_tail audit --manifest perf/results/2026-09-10/runtime-control/kda-output-v1-manifest.json --output perf/results/2026-09-10/kda-output-v1
```

Numerical differences would locate sensitivity in an isolated operation, not its
contribution to failed serving scores. Equality narrows this fixed matrix only.
Use these results to select the next investigation; no model start is prescribed.

## Completed state/output v1 result

On `4832f4ce7`, exactly one state process/audit then one output process/audit all
exit 0, with 224/224 paired cases each. No retries, omitted cases, autotuning,
model starts or native builds. All exact algebraic, finite-output, eager-repeat,
original/changed-input graph replay, mutation and guard checks pass. State checks
83 frozen source/evidence receipts and two actual cubins; output checks 312 and
two, including the complete state predecessor. GPU release and unchanged
device/driver/power identity verify. The source freeze has ended.

| Compared output | Type | Elements, both phases | Cross-config bit differences |
| --- | --- | ---: | ---: |
| State chunk snapshots | BF16 | 2,650,800,128 | 0 |
| State new values | BF16 | 1,285,685,248 | 0 |
| Final recurrent state | FP32 | 167,772,160 | 0 |
| Output | BF16 | 1,285,685,248 | 0 |

Conditioned-reference observations, not accuracy qualification: max absolute
snapshot/new-value/final-state errors are 0.03125 / 0.03125 /
0.005167722702026367; output 0.12482273578643799. Max BF16 ULP distances from the
rounded reference are 30192 / 30000 for snapshot/new values and 116 for output.
Final state uses FP32 comparison, not a BF16 ULP metric. These aggregate maxima
do not establish an element-level cause. Exact algebraic fixtures do qualify;
the conditioned random fixtures do not inherit an unprescribed accuracy gate.

Decision: neither state staging nor output warp choice changes these fixed
inputs. Combined with gate/recompute equality and the intra-stage identical
binaries, the examined KDA settings have not shown a numerical explanation for
the failed fresh-model scores. This is NOT proof across every activation or of
unrecorded historical live configurations. No speed measurement, serving/default
change or production promotion. All indexer/no-combo/model quality gates remain.
Next causal work should focus on remaining attention combo/split normalization
and actual workload boundaries. Do not automatically launch another model run.

Raw directories under `perf/results/2026-09-10/`: `kda-state-v1/` and
`kda-output-v1/`, each retaining 224 records, binaries/private cache, attempt,
summary and analysis. Analysis SHA respectively:

- State: `f28c002eacc2dbe1ab664f8c1fbded2c4745bc4604a1f4c4e805acfe644bc49c`.
- Output: `2b5c863eb453d252b05461e6d4ed02e8f171714b4c1713b49efd4e97daab943b`.

Summary SHA: state `7b9b5bd3da57e18d75ebd6bdd870201c8cdbe889effdced0317c3ac7c5446fef`,
output `7fe0e90ec181333f399f9fea1bf1801eeda9d4859c96a9ffd6bec923c058d1b7`.
CPU final 53 passed in 7.06 s; earlier 17/40-pass reports also retained under
runtime-control. After documentation/source edits, consume these completed
receipts rather than re-running the historical source-frozen validators.
