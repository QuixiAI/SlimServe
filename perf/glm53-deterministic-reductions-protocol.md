# GLM53 deterministic reduction policy qualification

Prescribed before GPU work, 2026-09-09. The complete RMSNorm intervention on
76afce776 proves that four runtime reduction choices suffice to reproduce the
older/native score difference. This next experiment qualifies a production-policy
candidate; it does not re-run the completed old/native isolation.

## Candidate and limits

Use the installed Inductor `deterministic` option, which filters reductions to
one configuration and disables both dynamic RBLOCK scaling and coordinate-descent
retuning. Do not enable the separate global PyTorch deterministic-algorithms or
batch-invariant modes. No profile/quant/native-binary/default changes yet.

First exercise only the runtime policy on the four exact generated sources.
The probe copies source bytes, temporarily substitutes only the decorator's
`inductor_meta.deterministic=False` with True during import, then restores the
decorator. It uses the normal `autotuner.run` path, not a manually selected
launcher; any attempted benchmarking is an immediate error. Require exactly
one config, retuning disabled, and the already-qualified historical binary hash.
This does NOT qualify frontend code generation or fresh full-model compilation.

## Fixed source-policy experiment

Exactly TWO sequential fresh processes/caches, output directories:

- `perf/results/2026-09-09/rmsnorm-deterministic-policy-a`
- `perf/results/2026-09-09/rmsnorm-deterministic-policy-b`

Each runs16 cases: four rank-matched sources x rows1/16/640/7616, one real
checkpoint norm vector (layer22 post-attention), seed530901 and changed seed531001,
unit activation magnitude. Test unchanged eager repetition, original/changed-input
graph replay, input/weight mutation and output guards. FP64 oracle gate remains
at most one BF16 ULP. Require all output hashes and selected configurations/binaries
to match across the two independent fresh processes. No timing or model claim.
Do not repeat completed cases into these outputs or replace failed starts.

Run only after CPU tests/lint and an immediate idle-GPU check. Each process gets
one16GiB/no-swap user-systemd scope; no other GPU work or native builds. Freeze
probe, imported helpers, compiler sources and native binaries through both jobs
and their audit. Originals remain read-only and all5172 files are rechecked.

```bash
systemd-run --user --scope --unit=glm53-deterministic-policy-a \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CUDA_HOME=/usr/local/cuda-13.0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_deterministic_reductions \
  --audit perf/results/2026-09-09/runtime-control/native-order-autotune-cache-comparison.json \
  --manifest perf/results/2026-09-09/rmsnorm-complete-graph-serving-caches/control/manifest.json \
  --native-audit perf/results/2026-09-09/runtime-control/rmsnorm-noop-complete-graph-control-analysis.json \
  --model /raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4 \
  --output perf/results/2026-09-09/rmsnorm-deterministic-policy-a
```

Preserve a separate launch log per job. Only if A passes and releases GPUs,
run B with the matching new output/scope/log. An8GiB/no-swap offline audit must
verify all32 cases, both complete matrices, helper/compiler/source/binary receipts,
all corresponding output hashes, original files, and final GPU release.

After this source-runtime qualification, test real frontend propagation through
the existing profile's `inductor_compile_config`, audit emitted reduction metadata
and independently compiled choices, and run fixed-start full-model quality and
performance gates. The cache-bound replacement hook remains diagnostic-only.
Do not assume the fixed policy will reproduce a historically mixed per-rank tree,
declare quality from aggregate proximity, or widen existing quality gates.

## Completed runtime-policy qualification and opt-in plan

Both prescribed processes complete on3e497c7c0:32/32 cases pass, with every
corresponding output exactly repeated across independent empty caches. All four
sources automatically choose XBLOCK1/RBLOCK1024/eight warps/one stage, and emit
the already-qualified binary for that rank/source. No benchmarking, dynamic
RBLOCK scaling or coordinate descent; oracle maximum1 BF16 ULP. All5172 original
files,5 source/helper receipts and7 native binaries unchanged; GPUs released.
Audit `runtime-control/rmsnorm-deterministic-policy-analysis.json`,
SHA8d523f85d74d637e7fe4b01a74cf56a4f976863124288226f1c8a8fddba99332.
Do not rerun these completed jobs into their outputs.

New `--deterministic-reductions` flag on SlimServe and the campaign harness
exposes an explicitly diagnostic candidate without changing registered defaults.
`slimserve/deterministic_reductions.py` copies the resolved engine plan and adds
only `compilation_config.inductor_compile_config.deterministic=True`. It requires
the fixed native-order GLM53 RTX6000 recipe, rejects the cached-source intervention
and conflicting global deterministic/batch-invariant controls. No runtime hook
or source replacement is used by this candidate. Benchmark receipts now hash23
implementation files and explicitly contain the compiler option and CLI flag.

220 CPU tests pass, including the actual vLLM compilation hash change, real
Inductor metadata generator under the config patch (backend identifier mocked,
no GPU codegen), default-off/no-mutation behavior, CLI scope and agreement between
the recorded benchmark plan and the launch arguments. Real-machine dry-run:

```bash
env -u NCCL_P2P_DISABLE SLIMSERVE_CACHE=/raid/weights \
  SLIMSERVE_GLM53_NATIVE_ORDER=1 \
  .venv/bin/python -m slimserve.cli glm53-nvfp4-4 \
  --deterministic-reductions --dry-run
```

No full-model start has used this option yet. Before prescribing that series,
add read-only actual-loaded-graph reduction receipts and an offline auditor so
frontend propagation and every selected reduction can be checked, not inferred
from static bundle counts or emitted source alone. Then freeze a fresh-compilation
series (independent empty caches plus a cached return), with full exact-token,
text/image, repeated quality and32K/128K prefill workloads. Match every score
across identical-policy starts, retain comparisons to older mixed-tree references,
and keep TC0/no promotion until the appropriate quality/performance gates pass.
No full-model start names or launches are prescribed by this checkpoint alone.

## Read-only graph receipts and bounded GPU frontend check

`slimserve/reduction_receipts.py` inspects actual PyCodeCache module globals
before and after CUDA graph capture. It records module/symbol/object bindings,
source hashes (raw and cache-root-normalized), complete launcher config dictionaries
and binary hashes. It never resolves a future, compiles, selects a launcher, reads
tensor values or changes an autotuner. Missing/uninspectable named RMSNorm globals,
unselected or nondeterministic reductions and active dynamic scaling fail closed.
The recorder runs only for native-order1 plus explicit deterministic=True, stores
exclusive rank/PID files beneath the private VLLM cache, and preserves failures.
Default paths remain unchanged.118 CPU tests pass; source/native defaults unchanged.

Before full-model serving, prescribe ONE16GiB/no-swap GPU process with a fresh
output `perf/results/2026-09-09/deterministic-reduction-frontend`:

```bash
systemd-run --user --scope --unit=glm53-reduction-frontend \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CUDA_HOME=/usr/local/cuda-13.0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_reduction_frontend \
  --model /raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4 \
  --output perf/results/2026-09-09/deterministic-reduction-frontend
```

Two real vLLM RMSNorm IR graphs (in-place and three returned copies), lowered
through VllmIRLoweringPass with the recorded compiler options, rows1/16/640/7616.
Eight cases, real layer22 BF16 norm vector, seeds530901/531001. Require finite
outputs, repeated eager/changed-input graph equality, input/weight/guard checks,
FP64 maximum1 BF16 ULP, and complete before/after live graph receipts. One GPU
does the arithmetic; the runner facade only tests the recorder's startup wiring,
NOT a TP4 forward or full-model equivalence. Never label this full-model validation.
Freeze sources through this job and audit; preserve failures and do not retry into
the same output. No full-model start is prescribed until this check is assessed.

## Frontend result and frozen full-model series (2026-09-09 23:55 UTC)

Frontend49f9bff98:8/8 pass, four actual graph bindings identical before/after,
selected1/1024/eight warps/one stage, FP64 maximum1 BF16 ULP. Eager repeats,
changed-input graphs, guards and mutation checks pass. Emitted metadata, compiler
cache keys and cubin byte digests verified;7 source receipts,7 native libraries,
5172 original files unchanged; exit0/GPU-free. Audit
`runtime-control/deterministic-reduction-frontend-analysis.json`,
SHA48830fce4a1fec6868f23609dd6cd8510f164e32d263f5996d5de72f2b9b815b.
First offline audit compared hex metadata keys directly with base32 launcher keys;
corrected using Triton's documented encoding. Both logs preserved, no GPU retry.

`benchmarks/analyze_glm53_deterministic_serving.py` prepares/audits this series.
`benchmarks/analyze_glm53_reduction_receipts.py` independently parses graph-source
Triton symbols and emitted policy without executing generated code. It checks
actual globals, complete reduction configs, compiled metadata and cubin bytes.
122 CPU tests pass; real-artifact replay verifies4 frontend bindings,12 exact
timing files,3 quality passes and24 prefill requests. Raw `runtime-control/`
`deterministic-serving-{cpu-final,audit-replay}.log` and replay script.

Prescribe exactly THREE starts, sequential: `fresh-a`, `fresh-b`, `cached-a`.
Prepare once after committing this checkpoint (clean tree required):

```bash
systemd-run --user --scope --unit=glm53-deterministic-prepare \
  -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m benchmarks.analyze_glm53_deterministic_serving prepare
```

This exclusively creates `perf/results/2026-09-09/deterministic-reduction-serving/`
and EMPTY `cache-a/`, `cache-b/`, records source/native/original-cache receipts.
No seeds, forced AOT namespace, cached RMSNorm substitution, remote cache, TC,
new quant, or changed driver/power/clocks. Original caches stay untouched.

Each arm requires its own8GiB/no-swap preflight immediately before serving:

```bash
systemd-run --user --scope --unit=glm53-deterministic-preflight-fresh-a \
  -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m benchmarks.analyze_glm53_deterministic_serving fresh-a --preflight
```

The serving command for fresh-a is below. Preserve a separate launch log via
`set -o pipefail`/`tee` under `runtime-control/`. Use identical arguments/env for
fresh-b/cached-a except scope/output name and fresh-b's cache-b paths.
Cached-a MUST reuse cache-a including its old receipt files, not a copy or seed.

```bash
systemd-run --user --scope --unit=glm53-deterministic-fresh-a \
  -p MemoryMax=150G -p MemorySwapMax=0 \
  env -u NCCL_P2P_DISABLE -u CUDA_LAUNCH_BLOCKING \
  CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda-13.0 \
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SLIMSERVE_CACHE=/raid/weights \
  VLLM_GLM5_MHC_BF16_FN=1 VLLM_GLM5_MHC_PREFILL_TC=0 \
  SLIMSERVE_GLM53_NATIVE_ORDER=1 \
  VLLM_CACHE_ROOT=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-09/deterministic-reduction-serving/cache-a \
  TORCHINDUCTOR_CACHE_DIR=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-09/deterministic-reduction-serving/cache-a/inductor \
  TRITON_CACHE_DIR=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-09/deterministic-reduction-serving/cache-a/triton \
  .venv/bin/python benchmarks/benchmark_glm53_campaign.py \
  --source /home/tiny/.local/scratch/slimserve-glm53/prompt-source.txt \
  --output perf/results/2026-09-09/deterministic-reduction-serving/fresh-a \
  --boots 1 --repeats 3 --concurrency 1 8 16 \
  --input-tokens 1000 --output-tokens 300 --cold-prefix \
  --quality --quality-repeats 3 --prefill --deterministic-reductions
```

After EACH exit and GPU release, audit before the next arm:

```bash
systemd-run --user --scope --unit=glm53-deterministic-audit-fresh-a \
  -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m benchmarks.analyze_glm53_deterministic_serving fresh-a
```

Freeze all sources/native binaries through ALL jobs and audits, no intervening
commits/builds/other GPU work. Every start keeps25 warmup/75 timed1000/300 requests,
text4/imageRed,168 quality requests/12288 text+504 needle scores, two long-prefill
warmups plus six timed requests (8 output tokens each, zero cached prompts).
Require exact score equality within/across starts, complete graph-source/global
coverage on all four ranks, emitted deterministic metadata/config/binary equality
(not process-local alias multiplicity), unchanged0.01-nat per-window quality gate
against fixed native controls and positive needle margins. Keep every comparison
to all12 older/native reference passes; do not require historical-tree equality
or mistake an aggregate proximity for a per-window pass.

Stop if a start/preflight/audit fails; keep failures, partial outputs and warnings.
Do not replace slow or failed starts, change gates, or continue B/return after a
failed preceding arm. A failed planned series can inform a newly prescribed
experiment, not a silent retry. No policy/TC/default/performance promotion from
this diagnostic series alone. After it, qualify any remaining numerical gate
before returning to TC or claiming a faster serving baseline.

## First series stopped; corrected no-combo policy (2026-09-10 00:07 UTC)

The first series is TERMINAL after its prescribed fresh-a on7d43c93af fails
before health or CUDA graph capture. No benchmark/canary/quality/prefill requests
were served. Do NOT launch fresh-b/cached-a or overwrite any of these outputs.
No numerical or throughput conclusion follows from this startup failure.

Exact stack: vLLM CompilationConfig defaults enable combo_kernels=True and
benchmark_combo_kernel=True; Inductor Scheduler.create_combo_kernel_nodes calls
speedup_by_combo_kernel -> benchmark_fused_nodes -> benchmark_gpu ->
may_ban_benchmarking. Deterministic mode intentionally rejects this unvetted
timed fusion decision. The small original frontend graphs did not cover it.

Failure closure through frozen sources confirms28 source receipts,7 native
libraries, all5172 original files and24 benchmark receipts; cache-b empty,
138 partial cache-a files retained, no other starts. Controller/server exit1,
teardown complete/GPU release0.044349s, final independent compute query empty.
The prescribed audit fails its incomplete-run gate, as required. Supplementary
hash-bound failure closure `runtime-control/deterministic-serving-failure-close.json`,
SHA67602718eeb3c8019bac83056b1b5c97f6a2e9b174b966f66d20064e4039456f.

Correct the opt-in plan to explicitly set deterministic=True, combo_kernels=False
and benchmark_combo_kernel=False. Registered defaults, quant/native binaries and
global numerical modes stay unchanged. This intentionally disables optional
horizontal combo fusion for qualification. Simply bypassing the benchmarking
ban is not a fix; merely disabling its timing gate accepts different static
combinations, an unqualified alternative. Restore/tune deterministic horizontal
fusion later if measured worthwhile. No claim that this is the fastest policy.

CPU regression reproduces the exact installed scheduler's benchmark guard with
no GPU, checks real vLLM defaults cannot re-enable combo fusion, rejects conflicts,
and proves cache-hash separation from BOTH the registered and first deterministic
plans. Existing CLI/campaign/quality/receipt tests remain in the matrix.

Before another model start, prescribe ONE fresh16GiB/no-swap extended frontend
probe with the corrected plan. THREE graphs: original in-place/three-copy plus
RMSNorm with independent unequal-size pointwise branches. Twelve cases at rows
1/16/640/7616, same real layer22 weight, seeds and oracle contract. Check extras
exactly against CPU operations and repeated eager/graph outputs. Compare the
eight original corresponding output hashes against the completed first frontend;
require inspected emitted reduction choices and byte-hashed cubins. Still NOT
TP4/full-model validation. Freeze sources/native through this probe and audit.

```bash
systemd-run --user --scope --unit=glm53-reduction-no-combo-frontend \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CUDA_HOME=/usr/local/cuda-13.0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_reduction_frontend \
  --model /raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4 \
  --output perf/results/2026-09-10/deterministic-reduction-no-combo-frontend
```

Preserve a separate launch log under2026-09-10/runtime-control. Do not rerun
completed/failed outputs. After this probe passes, adapt the serving auditor to
the three-option policy and prescribe a NEW fresh-cache series/namespace. The
older first-series commands above are archived, not authorization to continue
its unused arms. No corrected-policy full-model series is prescribed yet.

## No-combo frontend passes; corrected full-model series prescribed (2026-09-10)

One prescribed frontend process on532c62674:12/12 cases pass. The original eight
cases' output hashes and oracle metrics exactly match the first frontend.6 actual
graphs/9 bindings/6 reductions, unchanged before/after capture, all norm launches
1/1024/eight warps/one stage. FP64 maximum1 BF16 ULP; extra branches' exact CPU/
eager/graph checks pass.10 source receipts,7 native libraries,5172 original files
unchanged; exit0/GPU-free. Audit
`perf/results/2026-09-10/runtime-control/no-combo-frontend-analysis.json`,
SHA6facb3efedd788e3dc527fd3687d09ddefd61096a528d8cc876c8b55e313a6f9.
Do not rerun this completed probe. It is still not a full-model/TP4 qualification.

Auditor adapted to the explicit three-option no-combo plan and a NEW series root.
Code inspection found InductorAdaptor.initialize_cache redirects the worker's
Inductor/Triton directories into its AOT namespace. Binary audit now resolves
triton_cache from the actual emitted source's sibling inductor_cache (and keeps
the simple frontend inductor/triton layout), with private-root bounds and exact
key/metadata/cubin checks. Never scan an unrelated cache to make a receipt pass.
127 CPU tests pass16.04s; real frontend/timing/quality/prefill artifact replay passes.
No serving numerical or native changes in this auditor update.

Prescribe NEW series `perf/results/2026-09-10/deterministic-no-combo-serving/`:
exactly fresh-a, fresh-b, cached-a, one start each with the same complete workload
and unchanged per-window0.01-nat/positive-needle/exact-score/graph/source/native/
cache/teardown gates specified above. The older series remains failed/terminal.
The tracked auditor's default SERIES now points to this new root. Preparation
requires the hash-bound corrected frontend audit; cached return preserves A's
original receipt files. All failed/partial historical caches remain untouched.

After committing, prepare in one8GiB/no-swap scope:

```bash
systemd-run --user --scope --unit=glm53-no-combo-prepare \
  -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m benchmarks.analyze_glm53_deterministic_serving prepare
```

For EACH arm, preflight in a separate8GiB/no-swap scope with:
`.venv/bin/python -m benchmarks.analyze_glm53_deterministic_serving fresh-a --preflight`
(substitute the arm; use scope glm53-no-combo-preflight-<arm>). It verifies source
freeze, empty/private cached-return state, previous audit success and GPU idleness.
Then launch fresh-a below, preserving stdout/stderr and pipefail through tee:

```bash
GLM53_SERIES_ROOT=/home/tiny/Lazarus/SlimServe/perf/results/2026-09-10/deterministic-no-combo-serving
systemd-run --user --scope --unit=glm53-no-combo-fresh-a \
  -p MemoryMax=150G -p MemorySwapMax=0 \
  env -u NCCL_P2P_DISABLE -u CUDA_LAUNCH_BLOCKING \
  CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda-13.0 \
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SLIMSERVE_CACHE=/raid/weights \
  VLLM_GLM5_MHC_BF16_FN=1 VLLM_GLM5_MHC_PREFILL_TC=0 \
  SLIMSERVE_GLM53_NATIVE_ORDER=1 \
  VLLM_CACHE_ROOT="$GLM53_SERIES_ROOT/cache-a" \
  TORCHINDUCTOR_CACHE_DIR="$GLM53_SERIES_ROOT/cache-a/inductor" \
  TRITON_CACHE_DIR="$GLM53_SERIES_ROOT/cache-a/triton" \
  .venv/bin/python benchmarks/benchmark_glm53_campaign.py \
  --source /home/tiny/.local/scratch/slimserve-glm53/prompt-source.txt \
  --output "$GLM53_SERIES_ROOT/fresh-a" \
  --boots 1 --repeats 3 --concurrency 1 8 16 \
  --input-tokens 1000 --output-tokens 300 --cold-prefix \
  --quality --quality-repeats 3 --prefill --deterministic-reductions
```

After exit/GPU release run an8GiB/no-swap audit scope, command:
`.venv/bin/python -m benchmarks.analyze_glm53_deterministic_serving fresh-a`.
Only if complete, proceed to fresh-b with its own output/scope and cache-b paths;
then cached-a with its own output/scope and the EXISTING cache-a paths. Use the
same workload/flags. No forced AOT loading and no seeded caches. Native-order
remains diagnostic/defaultOFF; TC0. Keep sources/native frozen through all three
jobs and audits; no intervening edits/commits/builds or other GPU work. Stop on
any failed gate; retain every failed/slow start and all warnings, no replacements.
No performance or default promotion from these diagnostics alone.

## No-combo full workload completes but fails quality acceptance (series STOPPED)

One fresh-a on59ae0c88f completes the real profile and full prescribed workload,
with sources/native binaries frozen through all audits. All three quality passes'
individual text AND needle-token scores are exactly equal. Mean-2.7264740343091405,
all needle rankings positive. Nevertheless12/32 windows fail the unchanged
0.01-nat floor against fixed native controls:3/4/8/9/10/15/20/23/26/27/28/30.
Worst window28 is0.098160 beyond the permitted floor,0.108160 below the control.
All36 historical score comparisons differ. Improved aggregate score does not
override failed windows. Candidate NOT promoted; series TERMINAL. NEVER launch
its unused fresh-b/cached-a arms. Independent-start equality remains untested.

Measured diagnostics only: c1/c8/c16 E2E medians155.740252/574.244902/777.768328,
ranges[155.535387,155.742531]/[573.977641,575.604638]/[776.030609,777.828204].
Cold engine TTFT32K2.584496[2.582000,2.587168]s and128K10.875915
[10.840217,10.908407]s. Startup184.083560s, first text canary48.898762s due to
cold JIT work, image0.567689s.25 warmup/75 timed exact1000/300 requests,
text4/imageRed,168 quality requests/12288 text+504 needle scores, two prefill
warmups/six timed requests all complete.6 recovered4,718,592,000-byte allocation
warnings,2 zombies at teardown retained; controller/server exit0 and GPU release
0.559451s. Final independent compute query empty. No new speed baseline.

The prescribed audit initially fails on an expected-environment omission:
vllm/env_override.py sets TORCHINDUCTOR_COMPILE_THREADS=1 and
TRITON_CACHE_AUTOTUNING=1; new receipts expose these previously unrecorded keys.
Supplemental evidence verifies them explicitly. Further checker assumptions were
also corrected without changing the running model or its results:

- Before/after entire snapshots need not be identical: capture adds one
  pointwise-only graph per rank. Every pre-existing binding and every reduction
  remain unchanged. The initial supplemental script/log are preserved.
- The earlier sibling-cache claim is WITHDRAWN for this actual AOT path.
  decorators.py redirects only Inductor; Triton remains at the recorded launch
  root. InductorAdaptor's paired-directory behavior describes a different path.
  Supplemental audit checks exact keys/metadata/cubin bytes at the recorded
  private root. Its failed sibling-assumption script/log are preserved. Tracked
  checker now requires an explicit Triton root, with no search/fallback.
- Model graph inventory has512- and1536-wide norms as well as4096. Sixteen
  bindings need additional shape qualification. Do not force the4096 launch
  geometry on them or silently clear their oracle gate.

Supplemental audit verifies30 sources,24 benchmark receipts,7 native libraries,
5172 original files,36 post-capture graphs/132 bindings/68 reductions. Raw actual
source filenames, configurations and cubin digests are recorded, including512
persistent XBLOCK2/one warp and1536 XBLOCK2/RBLOCK1024/eight warps (one stage).
Correction2026-09-10:the2048 value in the original inventory was a rounded
compiler size hint. The Q norm's logical extent and output stride are1536.
The numerical acceptance failure is independently established and does not
depend on accepting these smaller norms as qualified.

Authoritative evidence:
`perf/results/2026-09-10/runtime-control/no-combo-first-workload-analysis.json`,
SHAeb333fb93c2ac28218d46ca7799d95ce2378f73ebc0ccfe5d49138b6e2530c87.
Its three supplemental audit logs and preserved failed script versions are under
the same runtime-control directory. Original failed audit stays in the series.

NEXT, no GPU job prescribed yet: inspect exact smaller-norm sources and actual
historical bindings, extend source/shape oracle coverage, and isolate the new
score-vector change. Disabled combo fusion, broader deterministic reduction
choices and independently fresh KDA autotuning are distinct candidate causes,
not established explanations. The completed four-RMSNorm old/native causal result
does not attribute THIS fresh-policy difference. Do not widen the quality gate,
restart into a good score, promote the policy, or exonerate TC. Subsequent GPU
work needs a newly recorded bounded protocol; existing series remain stopped.

## Source-exact attention combo/split comparison (2026-09-10)

No serving starts prescribed. Both stopped full-model series remain terminal.
CPU inspection establishes actual Q1536/KV512 and indexer LayerNorm128 shapes.
The original combo and new split sources both keep intermediate arithmetic FP32
until final BF16 stores, including the inspected4096 RMSNorm bodies. Native IR
includes a weight-dtype cast, but default Inductor compute-type upcasting elides
it; do not impose eager BF16 intermediate rounding on this compiled oracle.
Do not flip emulate_precision_casts or any serving numerical setting here.

ONE new kernel process, no restarts or tuning. Four rank-matched SM120 devices,
serialized work. Exactly120 combo/split pairs:4 ranks x rows1/3/16/640/7616 x
seeds530901/530902 x magnitudes0.125/1/8. Changed seed=seed+100. Four real BF16
layer11 weights:kv_a_layernorm, q_a_layernorm, indexer.k_norm weight/bias.
Packed input stride2336; offsets1536/0/2048 for512/1536/128, output strides
512/1536/256. Preserve the LayerNorm output gap and all surrounding guards.

Use all16 exact source files and the saved configurations. Compile those configs
only via the existing source-exact probe mechanism, no autotune/benchmark calls.
Require original cache keys AND actual cubin-byte hashes for BOTH arms. This is
not another test of automatic deterministic-policy selection or frontend codegen.
Historical combo evidence is static-future callbacks, not complete graph-held
coverage; synthetic inputs do not establish model-causality or cross-start safety.

Gates: exact eager repeat and original/changed-input graph replay, immutable
packed input and weight vectors, intact output row/gap guards. Each norm output
must be within1 BF16 ULP of the independently computed FP64 reference with one
final BF16 rounding. LayerNorm uses centered population variance+1e-6; RMSNorm
uses mean square+1e-5. Record all1440 oracle-output comparisons and720 pairwise
output comparisons. Pairwise exactness is an observation, not a prerequisite.
Finish the predetermined matrix on numerical mismatches and exit failed with
all output hashes/metrics preserved; stop immediately on launch/guard/receipt failures.
Never widen gates or rerun this output to turn a failed probe into a pass.

CPU discovery (working tree, no GPU) already verifies the inputs at
`runtime-control/attention-norm-discovery-manifest.json`, SHA
512ce4252c6c25ce3d762539a995e2846b012c118640966df42066e3ceec0bf7.
It also inventories identical4096 source bodies with differing configurations;
these matches are many-to-many, not runtime-site mapping or causal attribution.

After commit, prepare the final manifest in an8GiB/no-swap scope:

```bash
systemd-run --user --scope --unit=glm53-attention-norm-prepare \
  -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_attention_norms --prepare \
  --manifest perf/results/2026-09-10/runtime-control/attention-norm-manifest.json
```

Then exactly once:

```bash
systemd-run --user --scope --unit=glm53-attention-norm-probe \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CUDA_HOME=/usr/local/cuda-13.0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_attention_norms \
  --manifest perf/results/2026-09-10/runtime-control/attention-norm-manifest.json \
  --output perf/results/2026-09-10/attention-norm-source-probe
```

Preserve scope stdout/stderr with pipefail/tee under runtime-control. Freeze
sources/native binaries from final preparation through process exit and8GiB
offline audit (same module/manifest/output with `--audit`, scope
`glm53-attention-norm-audit`, MemoryMax=8G/MemorySwapMax=0).
Audit counts, all metrics/hashes, before/after original5172-file
inventory, compiler/helper/native source receipts, and final independent GPU
release. Do not promote the deterministic policy, TC, or a performance baseline
from this test. No subsequent GPU work is prescribed here.

## Initial attention probe stopped; preserve source provenance (2026-09-10)

The ONE probe on14ec731c8 stops before numerical launches: the first combo key
and config match but its cubin-byte hash differs. Its original manifest/output
are TERMINAL, never rerun. The prescribed audit rejects the incomplete matrix.
Frozen supplemental CPU closure verifies122 source/compiler/native receipts and
5172 original files. All41 ELF sections are compared: only six debug/debug-line
relocation sections differ; all non-debug sections, compiler metadata JSON and
PTX code prefix match. Original vs copied Python filenames explain the debug
content difference. Independent GPU query empty, process exit1. Zero numerical
cases, no model or performance result. Missing-pyelftools closure attempt is
preserved; the completed reader uses only the standard library.

Closure: `runtime-control/attention-norm-byte-failure-analysis.json`, SHA
07856ad2966f91ea6ed33a3380c388ce48128c1b8c2f906413e3c6aee362abb5.

Correct the probe loader, not the binary gate: verify copied/original bytes,
compile the copy with the original code filename for debug provenance, and keep
the module.__file__/Inductor decorator filename in the private probe directory.
No numerical-source edits, seeded/substituted cubins, stripped debug sections,
or original-cache mutation. Continue requiring exact whole cubin bytes.

After the tested fix is committed, prescribe NEW one-process comparison with the
same120 pairs, seeds/shapes/weights/gates,16GiB/swap0 and frozen sources. Prepare
in8GiB scope `glm53-attention-provenance-prepare` using the same module and
`--prepare --manifest perf/results/2026-09-10/runtime-control/attention-provenance-manifest.json`.
Then once:

```bash
systemd-run --user --scope --unit=glm53-attention-provenance-probe \
  -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES=0,1,2,3 \
  CUDA_HOME=/usr/local/cuda-13.0 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_attention_norms \
  --manifest perf/results/2026-09-10/runtime-control/attention-provenance-manifest.json \
  --output perf/results/2026-09-10/attention-norm-provenance-probe
```

After exit, same module/manifest/output plus `--audit` in8GiB/no-swap scope
`glm53-attention-provenance-audit`, then independent GPU query. Preserve all
stdout/stderr with pipefail/tee under runtime-control. No intervening edits,
builds, commits or GPU jobs through audit. No further job or serving start is
prescribed. Do not silently retry a failed numerical case or alter its gates.

## Shared first-writer provenance (2026-09-10)

The ONE9de50bbe3 provenance probe is TERMINAL, zero numerical launches. Original
combo now matches full cubin bytes, then first split kernel fails the byte gate:
its shared Triton cache entry was compiled first from rank1's equivalent source.
The bound rank0 source path is not the binary's debug filename. No model result.
Frozen CPU closure:121 receipts/5172 original files unchanged; all38 ELF sections
compared, only.debug_line and.nv.merc.debug_line differ. All non-debug sections
and metadata JSON equal. PTX instructions equal, with differing filename comments;
the initial literal-prefix assertion/script/log remain preserved. GPU query empty.
Audit: `runtime-control/attention-provenance-failure-analysis.json`, SHA
797a63dd7e137bd3a63dc55be1262fa849be1c1e0d622d896119587c55b6b018.

Resolve each binary's explicit PTX .file1 source in CPU preparation, not a cache
search. Bound it to that original cache and verify exact function text/line
position against the rank-bound source. Keep the latter's bytes and metadata;
only compile-time debug filename comes from the recorded first writer. Hash the
PTX, debug source, and original cubin as frozen inputs. ALL16 pass preparation;
nine split aliases refer to rank1's first-writer source. CPU discovery manifest
`runtime-control/attention-first-writer-discovery.json`, SHA
49ba4df5a3a83bcfcea30383ffcb9683b14a77bbb05f0cbf15dd4704badd9e07.
No weakened whole-cubin or numerical gates, source/body replacement, binary
seeding, source-cache writes, or serving changes.

After tests/commit, NEW fixed run (same120 pairs/gates/environment as above):

- Prepare8GiB/swap0 scope `glm53-attention-first-writer-prepare`; same module,
  `--prepare --manifest perf/results/2026-09-10/runtime-control/attention-first-writer-manifest.json`.
- Exactly ONE16GiB/swap0 scope `glm53-attention-first-writer-probe`; same module
  and new manifest, `--output perf/results/2026-09-10/attention-norm-first-writer-probe`.
- After exit,8GiB/swap0 scope `glm53-attention-first-writer-audit`; same module,
  manifest/output plus `--audit`; independently verify GPU release.

Keep sources/native frozen through audit. Preserve pipefail/tee logs under
runtime-control. Both earlier attempts stay terminal, not overwritten or resumed.
No subsequent GPU/model job prescribed, no numerical or performance promotion.

## Rank-private binary images and partial numerical evidence (2026-09-10)

ONE4eb7b664a first-writer probe is TERMINAL after30 rank0 pairs. Four rank0 binary
images verify exactly, repeat/replay/guard/mutation gates pass. RMS512/1536 stay
within1 BF16 ULP of FP64; Q1536 is bit-exact combo/split over60 outputs, KV512 has
139 changed elements. LayerNorm128 has114 changed elements and exceeds its gate:
max5 ULP(combo)/28(split),5/30 pairs failing in BOTH arms. Do not promote the
matrix or infer full-model causality from these partial/synthetic results.

Rank1's combo then fails the byte gate: historical rank-local caches hold FOUR
debug images under ONE semantic key; the probe's single cache returned rank0's
image. This is distinct from the already-resolved first-writer filename mapping.
All41 ELF sections compared; only.debug_line/.nv.merc.debug_line differ. Every
non-debug section and compiler metadata match.135 frozen receipts/5172 original
files verify, GPU query empty. Incomplete-matrix audit retained. Closure:
`runtime-control/attention-first-writer-failure-analysis.json`, SHA
e7977af07d26066449b1b874a04a31806a8bd4252a4416b92ec8df823b1b55c3.

Correct the probe cache layout to `triton/rank-<rank>/` using Triton's scoped
cache setting. CPU regression exercises actual cache-manager writes/reads with
the same semantic key in four disjoint temporary directories and verifies scope
restoration. No serving cache changes or seeded binaries. Compile/verify ALL16
images before any numerical execution so a receipt failure cannot interrupt a
partially completed numerical matrix again. Record bounded worst-ULP element
coordinates and actual/reference values for failed checks; accuracy gates remain
unchanged. This is not a retry to select a passing numerical sample.

After tests/commit, prescribe ONE NEW process, same120 pairs and existing gates:

- Prepare8GiB/swap0 scope `glm53-attention-rank-private-prepare`; same module,
  `--prepare --manifest perf/results/2026-09-10/runtime-control/attention-rank-private-manifest.json`.
- ONE16GiB/swap0 scope `glm53-attention-rank-private-probe`; same module/new
  manifest, `--output perf/results/2026-09-10/attention-norm-rank-private-probe`,
  same four-device/environment flags as previous commands.
- After exit,8GiB/swap0 scope `glm53-attention-rank-private-audit`; same module,
  manifest/output plus `--audit`; independent GPU query. Supplemental CPU audit
  must also verify exact agreement of the earlier30 rank0 records' outputs,
  metrics, input hashes and verdicts (new scalar-evidence field excluded).

Freeze through audit, preserve logs and every failed/partial run. Expected
LayerNorm failures stay failures; do not clear them to make the matrix pass.
No subsequent GPU/model job prescribed, no policy/default/performance promotion.
