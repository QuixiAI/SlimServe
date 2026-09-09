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
