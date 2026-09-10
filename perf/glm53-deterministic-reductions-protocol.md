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
