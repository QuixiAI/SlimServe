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
