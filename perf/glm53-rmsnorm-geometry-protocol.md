# GLM53 broader RMSNorm geometry isolation

Status: corrected pre-load A/B pair COMPLETE and audited on4b0fa3701. Both312-pair
matrices pass and match exactly across processes. All commands below are now
historical; do not rerun them. The first pair remains terminal after its observer
API failure. No model series, real-AOT loader job or other GPU job is prescribed.

## Question and fixed factors

Can changing only the thirteen identified original-cache H4096 RMSNorm launch
choices reproduce the failed fresh no-combo policy's score-vector change?

The completed graph correspondence identifies six input-layer norms and seven
post-attention norms, with 3/3/3/4 sources by rank. All change from
XBLOCK1/R0_BLOCK4096/16 warps to XBLOCK1/R0_BLOCK1024/eight warps, one stage.
Final mean+norm is not a target. The historical four-source sufficiency finding
concerns a different old/native pair and does not answer this question.

Keep the selected recipe, TP4, native libraries, original graph bodies/decorators,
attention combo kernels, KDA cache choices, native-order1, BF16fn1 and TC0 fixed.
Do not import fresh no-combo sources or enable its compiler policy. Do not widen
quality tolerances or clear the separate indexer LayerNorm gate.

## Completed CPU evidence

- Graph mapping: `runtime-control/norm-graph-role-pairs.json`, SHA
  `320a6f39c4f92cd78459f21e23653e1e900870620da29bed923f5cba5dbe315e`.
  Twenty-eight graph pairs; match body, semantic consumer and exact ordered
  call arguments, excluding duplicate compile-time docstrings.
- Final discovery: `runtime-control/rmsnorm-geometry-final-discovery.json`, SHA
  `4294ff75ea2236c577b8104ea3a44055d65dcf68ca0449685eccf02a2c5f7ea4`.
  Thirteen exact sources/control configs/cubin/PTX/metadata images verified;
  six in-place and seven triple-output layouts. All thirteen original debug
  identities point at their own source. Thirty-five static graph/source uses.
  All 178 source receipts and 5,172 original files verified. Discovery schema
  deliberately cannot be passed as a qualified intervention manifest.
- Controller: `benchmarks/kernels/glm53_rmsnorm_geometry.py`. Reuses the previous
  single-source controller under one atomic multi-target resolver. A target
  object reaches upstream cache resolution once, including concurrent aliases.
  Exact paths, source/config/cubin checks, strong references, complete per-source
  graph coverage, and no late bindings are mandatory. This is not installed in
  SlimServe or vLLM; the historical controller/serving flags remain unchanged.
- CPU regression: 187 passed, 5.43 seconds, 8 GiB/no-swap scope. Report
  `runtime-control/rmsnorm-geometry-final-cpu.xml`; initial reports preserved.

All raw paths above are under `perf/results/2026-09-10/`.
Static old graph uses are discovery evidence, not proof of historical live
coverage or of a working new hook. No throughput was measured here.

## Source-exact numerical qualification

The dedicated probe uses the discovered thirteen ORIGINAL sources,
the provenance-preserving loader and rank-private Triton cache scopes from
`check_glm53_attention_norms.py`, and the existing in-place/triple-output oracle,
guard, eager-repeat and changed-input replay helpers from
`check_glm53_cached_rmsnorm.py`. Keep generated metadata unchanged. Compile the
two explicit configurations directly; do not call timed autotuning.

Bounded matrix, frozen with the commands below before launch:

- Two fresh, sequential processes A/B, independent empty private caches.
  Run B only after A and its audit pass; no replacement starts or retries.
- Each process: thirteen sources x rows 1/16/640/7616 x seeds 530901/530902 x
  the three existing checkpoint-weight/magnitude sites in `SITES`: 312 pairs.
  Changed input uses seed+100. Both configurations in each pair.
- Compile all 26 source/config bindings before numerical work. Control must
  reproduce the exact recorded key and whole cubin bytes. Geometry must use the
  original source/decorator, not a binary borrowed from fresh no-combo graphs.
  Record its actual key and whole binary; require exact cross-process identity.
- Both arms: finite BF16 outputs and at most one BF16 ULP against the FP64
  oracle; repeated eager, original/changed-input replay, guards, no read-only
  mutation and triple-output agreement all pass. Retain the full prescribed
  numerical matrix, including failures; a failed A terminates this pair.
- Cross-process inputs, outputs, oracle metrics and binaries repeat exactly.
  Cross-configuration equality is observed, not assumed or required.
- Freeze probe/helper/serving/compiler/native sources through the pair and
  audits. One GPU workload at a time; 16 GiB/no-swap probes, 8 GiB audits.
  Preserve every artifact and verify all original cache files remain unchanged.

Implementation: `benchmarks/kernels/check_glm53_rmsnorm_geometry.py`, with
negative-gate tests in `tests/slimserve/test_rmsnorm_geometry_probe.py`. The
discovery artifact lacks qualified geometry binary receipts on purpose; this
probe establishes them. It also checks compiler metadata against each key/config,
in-memory cubins against disk bytes, exact output/metric phase coverage, and the
entire source freeze. B is programmatically gated on A's successful numerical
audit, unchanged summary, and rechecked source/binary artifacts.

Historical first-pair commands (STOPPED; do not rerun):

```bash
systemd-run --user --scope --unit=glm53-geometry-probe-prepare -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry prepare \
  --discovery perf/results/2026-09-10/runtime-control/rmsnorm-geometry-final-discovery.json \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-probe-manifest.json \
  --series-root perf/results/2026-09-10/rmsnorm-geometry-source-qualification
```

Exactly one process A, then its audit. Preserve stdout/stderr and exit statuses:

```bash
systemd-run --user --scope --unit=glm53-geometry-source-a -p MemoryMax=16G -p MemorySwapMax=0 \
  env CUDA_HOME=/usr/local/cuda-13.0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_rmsnorm_geometry run \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-probe-manifest.json --arm a
systemd-run --user --scope --unit=glm53-geometry-source-a-audit -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry audit \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-probe-manifest.json --arm a
```

Only after A passes and the independent GPU query is empty, exactly one B:

```bash
systemd-run --user --scope --unit=glm53-geometry-source-b -p MemoryMax=16G -p MemorySwapMax=0 \
  env CUDA_HOME=/usr/local/cuda-13.0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_rmsnorm_geometry run \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-probe-manifest.json --arm b
systemd-run --user --scope --unit=glm53-geometry-source-b-audit -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry audit \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-probe-manifest.json --arm b
systemd-run --user --scope --unit=glm53-geometry-source-pair-audit -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry compare \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-probe-manifest.json
```

Any structural, binary, freeze, or numerical failure terminates this pair;
preserve its partial/full evidence and audit before any new design. No code
edits, builds, commits or other GPU work between preparation and final audit.
Do not interpret these source probes as actual graph-loader or model validation.

## Subsequent gates, not yet prescribed

After numerical qualification, build private intervention manifests with both
verified source-specific binaries. Qualify the real static-future and PyCodeCache
loader hooks against all relevant original graphs, both arms/all ranks. Compare
the controller's coverage with an independent actual-module/global inventory;
neither static source references nor callback counts substitute for it. Reuse
the earlier real-AOT qualification approach, not obsolete raw helper APIs.

Only then wire an opt-in diagnostic into the actual profile/campaign, freeze a
bounded control/geometry/return full-model series and its auditor, and run it in
150 GiB/no-swap serving scopes. Preserve the existing per-window quality floors,
exact score-vector comparisons, cold exact-token workload and all failed/slow
starts. A diagnostic score change is not a speed win or production promotion.

## First pair closure and corrected pre-load pair

ONE A on28842e5c3 stops before numerical cases: the static CUDA adapter has no
`asm` attribute. Its `cubin_raw` is consumed and cleared by `load_kernel` inside
`make_launcher`. The original observer was wrong in both API and lifecycle.
The first control's emitted disk key/config/whole-cubin bytes match the original;
182 frozen receipts/5172 original files verify. A exits1, prescribed audit rejects
the incomplete binary set, GPUs are free, B never launched. Closure:
`runtime-control/rmsnorm-geometry-binary-api-failure-analysis.json`, SHA
`ae19129d827af87f55352f6c7b625513c8db81ab2a0bb84e2616486ce124d07a`.

Corrected probe hashes the actual in-memory image BEFORE creating the launcher,
then compares that hash with the emitted disk binary. The helper handles Triton's
`asm` and Torch's `cubin_raw`, rejects conflicts/missing bytes, and never falls
back to guessing a disk image after loading. A CPU test uses the installed
StaticallyLaunchedCudaKernel.load_kernel lifecycle with only its driver call
mocked, proving the consumption timing.226 related tests pass5.67s. The static
launcher implementation is added to the next frozen manifest.

This fixes the SOURCE probe. The multi-target controller will separately need
pre-load observation for graph-held static objects when implementing real-AOT
qualification; post-load bytes are unavailable there too. Do not call the
controller qualified or install it in serving on the strength of these CPU tests.

NEW discovery `runtime-control/rmsnorm-geometry-preload-discovery.json`, SHA
`d98fffd11b5f0f60880a3754ea6641201435aa9779bcfeb1bdb4ff95bd0d3a5f`.
Same13 sources/configs/312-pair matrix and all gates; only the observer changes.
NEW series, exactly A then (only after pass/audit/GPU release) B:

```bash
systemd-run --user --scope --unit=glm53-geometry-preload-prepare -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry prepare \
  --discovery perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-discovery.json \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json \
  --series-root perf/results/2026-09-10/rmsnorm-geometry-preload-qualification
systemd-run --user --scope --unit=glm53-geometry-preload-a -p MemoryMax=16G -p MemorySwapMax=0 \
  env CUDA_HOME=/usr/local/cuda-13.0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_rmsnorm_geometry run \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json --arm a
systemd-run --user --scope --unit=glm53-geometry-preload-a-audit -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry audit \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json --arm a
systemd-run --user --scope --unit=glm53-geometry-preload-b -p MemoryMax=16G -p MemorySwapMax=0 \
  env CUDA_HOME=/usr/local/cuda-13.0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_rmsnorm_geometry run \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json --arm b
systemd-run --user --scope --unit=glm53-geometry-preload-b-audit -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry audit \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json --arm b
systemd-run --user --scope --unit=glm53-geometry-preload-pair-audit -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_rmsnorm_geometry compare \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json
```

Freeze through the entire NEW pair and audits; no edits/builds/commits/other GPU
work, no replacement attempts, no new model job. Preserve stdout/stderr/statuses.
Stop on failure; all original first-pair paths remain immutable historical evidence.

## Completed pre-load pair result

Both prescribed processes exit0,312/312 pairs each; all26 source/config bindings
verified before numerical work in each. Original control whole-cubin bytes match.
Both geometries have maximum one BF16 ULP vs FP64; all eager/replay/guard/mutation/
triple-output checks pass. Final audit requires and observes exact equality of all
312 corresponding inputs, outputs, metrics and binaries across processes.
183 source/native/compiler receipts and5172 original files remain unchanged.
Independent GPU queries are empty after each process. Source freeze is released.

Per-process geometry/control comparison:14,820 differing elements among
5,286,248,448 paired elements. This is synthetic-input qualification on the three
real norm weight vectors, not actual model activations, score causality or TPS.
It does not clear the separate indexer LayerNorm gate or qualify the controller.

Final `rmsnorm-geometry-preload-qualification/pair-analysis.json` SHA
`3b5bcff2c1d23b4b8a2ffa7c648267d5ee74003f67f6ce6495528d1c69516f69`.
A/B analysis SHAs respectively
`9a7ea3bf24f4b811861ed373c7c10ba51a318871609ea6deecf026af3d926f6b` and
`9bc9117e8b3417864e80e05f560b25ac2232e5fde957dd2cf16d9a11b7d67592`.
Raw root: `perf/results/2026-09-10/`; all prior attempts and logs retained.

Next CPU work: observe the actual static CUDA driver-load input before bytes are
consumed, retain object/handle provenance, and tie each graph-held launch callable
to its observed kernel. Then qualify the multi-target hook against the real AOT
loader. No additional GPU process is prescribed until that implementation and its
bounded qualification protocol are ready.
