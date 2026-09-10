# GLM53 broader RMSNorm geometry isolation

Status: source-exact numerical probe/auditor implemented on 2026-09-10. The
two-process kernel pair below is prescribed after the final CPU checks and commit.
No model series or real-AOT loader job is prescribed yet. All earlier stopped
series remain terminal; never reuse their unlaunched arms.

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

After CPU tests and source commit, prepare once in an 8 GiB/no-swap scope:

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
