# GLM53 broader RMSNorm geometry isolation

Status: corrected pre-load A/B pair COMPLETE and audited on4b0fa3701. Both312-pair
matrices pass and match exactly across processes. Their commands below are
historical; do not rerun them. The first pair remains terminal after its observer
API failure. AOT v1 stopped during CPU preparation before any private cache/GPU
load. v2 stopped on its first GPU load's cache-lifecycle gate; v3 loaded all seven
rank0 artifacts but stopped at the auditor's export parser. v4 stopped at the
inventory's graph/helper classification. NEW no-weights AOT v5 is prescribed at the tail below;
no full-model series is prescribed.

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

## CPU observer integration checkpoint

The pre-load observer now lives in `benchmarks/kernels/glm53_binary_observer.py`.
It records the exact private file passed to the real static CUDA load method,
checks any available raw image against it, and retains strong identity plus the
resulting module/function handles. Post-load checks require this prior observation;
they do not guess a disk image. Generated launchers must bind their runner to the
observed kernel and agree on hash, warps and shared memory. Unobserved loaded
objects, closed/changed handles, bad paths/images/ranks and late loads after an
explicit observer seal fail. Hook cleanup preserves prior inherited/local APIs
and refuses to overwrite a foreign change made while active.

MultiIntervention accepts the observer for binary reads and launcher checks,
including cached replacements and graph verification. Its own target sealing
remains separate from the observer's global load seal: do not accidentally forbid
legitimate non-target graph compilation during future model capture. The actual
AOT/serving adapter must choose and qualify its hook lifetime explicitly.

CPU tests use the installed StaticTritonCompileResult.make_launcher, generated
launcher code and StaticallyLaunchedCudaKernel.load_kernel; only the driver is
mocked. They cover retained/consumed bytes, serialized objects without raw bytes,
one load under concurrent aliases, changed handles/images/launchers, cleanup,
and a three-target controller graph substitution followed by sealing/reuse.
Final related suite254 passed5.75s. This is NOT real GPU/AOT or model qualification.
No further GPU job is prescribed at this checkpoint.

The known-key cache join also verifies all13 qualified candidate images against
their exact original rank-local files and both completed probe outputs. Report
`runtime-control/rmsnorm-geometry-candidate-cache-check.json`, SHA
`017120c13bc5fc5b8261903cf8a4dd3ba782b67c00ccfe53a9f58cfcd40a7102`.
It uses completed qualification receipts after their source freeze ended; it
does not pretend the subsequently edited helper files still match old hashes.

Next implement private manifest preparation and actual AOT-loader adapters/auditor,
then prescribe the bounded all-rank/control+geometry qualification. Use the proven
seven-artifact loader path and independently verify actual graph globals. Do not
substitute static graph references, source probes or callback counts for coverage.

## Actual loader adapter CPU checkpoint

Preparation, scoped loader hooks and independent live-graph inventory are now
implemented in `prepare_glm53_geometry_loader.py`, `glm53_geometry_loader.py` and
`audit_glm53_geometry_graphs.py` under `benchmarks/kernels/`. NOT serving hooks.
Preparation joins the pinned completed pair to both original rank-local binaries,
copies the full unchanged namespace separately for every rank/mode, and records
released helper updates without silently changing native/compiler/serving receipts.
No private copies have yet been made with this preparer.

The adapter observes driver inputs before cached resolution, uses original debug
provenance for replacements, and leaves `TRITON_CACHE_DIR` UNSET. Torch resolves
the private Inductor cache's `triton/<rank>` directory exactly as in the original
AOT path; do not flatten this into a shared directory or change KDA cache behavior.
Both loader hooks remain active through inventory and target sealing. Their cleanup
restores unchanged hooks but preserves/rejects foreign edits. Target sealing does
not globally seal the binary observer; this is tested with a later non-target load.

The independent inventory walks actual module call globals and executable `.run`
symbols, checks source/graph hashes and selected binary/launcher-object provenance,
and requires the mapped graphs and every target binding. It also records non-target
Triton bindings for exact cross-arm comparison. Controller ownership/callback maps
do not supply its coverage. This CPU implementation is NOT proof that all actual
AOT graph globals have yet been loaded or that all are supported static launchers.

Related suite232 passes2.80s (33 new tests),8GiB/swap0, including installed
PyCodeCache/StaticAutotunerFuture/static launcher APIs and concurrent source imports.
Driver and replacement-compiler calls are mocked; actual artifact join verifies13
targets/7 graphs per rank/183 source receipts/5172 original files unchanged.
`runtime-control/geometry-loader-source-check.json` SHA
`cbe7547de5f0218caa4e5e094a1a1f74b2b17e00a0087c83a8eb5f7b78cb4d84`.

Remaining before GPU work: no-weights AOT runner, separate offline receipt auditor,
new source freeze including those tools and actual Torch loader helpers, private
manifest preparation, and a prescribed sequential eight-process protocol with
stop-on-failure/no retries. Then actual graph/global/binary coverage, unchanged
non-target comparison and cache/source audit must pass before any model series.
No next GPU/model job is prescribed at this checkpoint.

## Historical real-AOT qualification v1 (stopped in CPU preparation)

Runner `check_glm53_geometry_loader.py`, offline auditor
`audit_glm53_geometry_loader.py`; both under `benchmarks/kernels/`.
CPU gate263 passed3.13s (31 new audit/harness tests),8GiB/swap0; report
`runtime-control/geometry-loader-protocol-cpu.xml`. Initial focused64 pass2.55s
in `geometry-loader-audit-cpu.xml`. GPUs idle before source freeze.

Exactly EIGHT processes, in this order: control-rank0, control-rank1,
control-rank2, control-rank3, geometry-rank0, geometry-rank1, geometry-rank2,
geometry-rank3. One fresh identical full private namespace per process. Each
loads the original rank's seven cached artifacts/46 entries through the real
concurrent StandaloneCompiledArtifacts.load_all path, WITHOUT outer model
deserialization, weights, forwards, capture or timing. Never retry a process.

Before each next load: preceding load exit0, offline audit exit0, original/source
freeze unchanged, all predecessor receipts/logs unchanged, independent GPU query
empty. ANY failure terminates the entire sequence; preserve partial/full raw logs,
summary, binary/controller streams, audit and private caches. No edits, builds,
commits or other GPU workloads from preparation through final closure/audit.

Required per-process evidence: all seven actual graph sources and every mapped
target binding, no static bundle fallbacks, each graph launcher tied to its
observed exact CUDA load image, per-source selected config/cubin, independent
graph-global inventory joined to controller and binary receipts, complete target
and observer seals. Geometry additionally requires identical ALL non-target
graph/source/config/binary records to the corresponding control. The observer's
global seal occurs only after all loading/inventory/target sealing; there is no
later capture in this gate. This lifetime is NOT automatically a serving policy.

Prepare AFTER committing the implementation and this protocol:

```bash
systemd-run --user --scope --unit=glm53-geometry-aot-v1-prepare -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.prepare_glm53_geometry_loader \
  --pair perf/results/2026-09-10/rmsnorm-geometry-preload-qualification/pair-analysis.json \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json \
  --mapping perf/results/2026-09-10/runtime-control/norm-graph-role-pairs.json \
  --output perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v1
```

Invoke the following ONCE for each prescribed LABEL above, sequentially and only
if all earlier labels passed. LABEL is not a tuning choice. The launcher records
`launch.json`, native stdout/stderr in `load.log`/`audit.log`, exact commands,
exit codes/log digests and post-load independent GPU query. It starts its GPU
child in16GiB/swap0 and auditor in8GiB/swap0 scopes and rejects prior attempts.

```bash
systemd-run --user --scope --unit=glm53-geometry-aot-v1-launch-LABEL -p MemoryMax=8G -p MemorySwapMax=0 \
  env CUDA_HOME=/usr/local/cuda-13.0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.kernels.check_glm53_geometry_loader launch \
  --manifest perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v1/LABEL/manifest.json
```

Only after ALL eight pass and the GPUs are free:

```bash
systemd-run --user --scope --unit=glm53-geometry-aot-v1-final-audit -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.audit_glm53_geometry_loader compare \
  perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v1
```

This gate proves loader/graph/binary coverage, not model score causality or TPS.
Only after it passes may the opt-in serving adapter and bounded model causal
series be implemented/prescribed. Quant/defaults/native math/gates unchanged.

## v1 preparation closure; historical v2 (stopped on first load)

ONE CPU preparation on7e2b8bd0e exits1 BEFORE creating the private series or
launching any GPU load. Dotted import of `torch._inductor.standalone_compile`
resolves Torch's package-exported FUNCTION rather than its module, so source-file
freezing raises AttributeError. Original5172 files and183 source receipts verify,
GPU query empty. v1 is TERMINAL, none of its eight loads may be launched.
Closure `runtime-control/geometry-aot-v1-preparation-failure.json`, SHA
`1a337eb9639c4152ef2a6b02c875b97f5d512468eadd9d852dbeed36fc1f2e23`.

Fixed with explicit `importlib.import_module` resolution for all five loader
modules. New CPU test executes actual module resolution plus the WHOLE eight-copy
preparation, frozen manifest readback and source/original verification on small
fixtures, and rejects overwriting the output. Focused65 tests pass2.75s; related
264 pass. No GPU/math/default/gate change. Initial evidence and tests retained.

NEW v2: exactly the SAME eight rank/mode attempts, loader path, scopes, order,
no-retry policy and gates prescribed above, at a NEW series root. Commit the fix
then freeze through preparation/loads/audits/closure. Prepare once:

```bash
systemd-run --user --scope --unit=glm53-geometry-aot-v2-prepare -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.prepare_glm53_geometry_loader \
  --pair perf/results/2026-09-10/rmsnorm-geometry-preload-qualification/pair-analysis.json \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json \
  --mapping perf/results/2026-09-10/runtime-control/norm-graph-role-pairs.json \
  --output perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v2
```

Then invoke the previous `launch` command once for each prescribed LABEL in order,
substituting `v2` for `v1` in BOTH the outer unit name and manifest series path.
The launcher's child scope names are also v2. Only if all eight pass, invoke the
previous final `compare` command with v2 in BOTH unit and series path. No other
changes, replacement starts or model workload authorized by this protocol.

## v2 closure and exact cache-lifecycle fix; historical v3 (stopped)

ONE control-rank0 load onad6656314 exits1, followed by its prescribed failing
audit. The remaining seven loads were NEVER launched. All seven static bundles
load without fallback (50 entries total);15 actual CUDA loads match their original
whole cubin images. Replacement compilation stops at the diagnostic's overly
strict requirement that TRITON_CACHE_DIR remain absent. No graph-complete coverage,
replacement numerical result, model forward/weights or TPS claim. All201 frozen
source receipts/5172 original files verify, independent GPU query empty. v2 is
TERMINAL and its eight prepared caches/receipts/logs remain intact. Closure
`rmsnorm-geometry-aot-qualification-v2/closure.json`, SHA
`c84e1f76dcca924103b52e660b30130e120c2f22b68a89800219024a819ae044`.

Torch's actual CachingAutotuner constructor materializes an initially absent
TRITON_CACHE_DIR as the SAME rank-private directory. CPU test executes that real
constructor (no compilation/driver) and verifies the transition. The adapter now
accepts absent OR exactly `<private>/inductor_cache/triton/<rank>` and checks the
resolved canonical path before/after template creation AND compilation. It never
unsets/rewrites the variable in callbacks. Wrong-rank/shared/empty/aliased values
still reject. Error diagnostics now include the actual/expected cache paths; the
failed v2 run did not itself record those environment values. Focused70 pass2.83s;
related269 pass. The real loader must still qualify the corrected adapter.

NEW v3: repeat the SAME prescribed eight-case matrix at a NEW root, not a retry
of v2. Commit fixes/protocol, then freeze. Use the v2 preparation command above,
substituting `v3` in BOTH unit and output path. For each LABEL in the original
fixed order, use the launch command with `v3` in BOTH unit and manifest path;
child scopes are also v3. Final compare uses v3 unit/path only after all eight
loads/audits/release checks pass. Root:
`perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v3/`.
Same scopes/no-retry/no-edit/no-build/one-GPU-workload rules and every original
source/binary/graph/non-target gate. Do not launch any unused v1/v2 case.

## v3 closure and exported Runner.call support; historical v4 (stopped)

ONE control-rank0 process on14ab95575 loads ALL7 artifacts/46 entries; its26
observed CUDA images match original whole bytes. The independent AST inventory
then rejects the generated module's `call = runner.call` export because it only
recognized a top-level function. All3 sources have controller graph callbacks
(3/4/2), but those callbacks alone do NOT qualify actual graph coverage. Load and
prescribed audit exit1; remaining seven cases never launched. All201 sources/
5172 original files unchanged; GPUs free. v3 is TERMINAL, all copies/logs retained.
Closure `rmsnorm-geometry-aot-qualification-v3/closure.json`, SHA
`352a2e15d3a2bae0bc67b1cfdfb0de4aaf6017c372ca4eaa15bbadee9f1ef6d6`.

Auditor now follows the explicit exported instance/class/method AST bindings,
or a direct call function. It checks the live method's exact instance, class,
function, source filename/line and globals. It excludes compile-time strings and
unexported methods. Read-only source check matches ALL28 mapped original graph
exports and recorded kernel run symbols. Report
`runtime-control/geometry-bound-export-source-check.json`, SHA
`c43d449aa00bc4ffffc0f628efd291a16d199f54c69df7770c971121b457da36`.
Focused79 tests pass2.87s; related278 pass3.47s,8GiB/swap0. Nine new CPU tests
include actual PyCodeCache bound Runner.call imports and changed live exports.

NEW v4 after commit: SAME eight cases, order, independent private copies and
unchanged source/binary/graph/non-target/numerical gates. Use the explicit v2
prepare command with `v4` in BOTH unit/output path; use the launch/final compare
commands with `v4` in BOTH unit/series path. Child scopes are v4. Root:
`perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v4/`.
Freeze through terminal audits, one GPU workload at a time,16GiB GPU/8GiB CPU/
swap0, stop entire series on any failure. No retries or unused v1/v2/v3 cases.
This still does not prescribe a full-model run or change production defaults.

## v4 closure: root graphs must come from actual serialized artifacts

ONE control-rank0 on5213afeb5 loads7 artifacts/46 entries and27 original-exact
CUDA images. Load/audit exit1 at `unexpected or changed graph source`; the failing
module path was not retained. ALL private Python files equal the original snapshot.
The validator wrongly classifies every cached module exporting callable `call` as
a model root, although generated kernel benchmark helpers also export `call`.
All201 sources/5172 original files unchanged, GPUs released. v4 is TERMINAL;
remaining seven cases never launched. All copies/logs retained, freeze released.
Closure `rmsnorm-geometry-aot-qualification-v4/closure.json`, SHA
934fe0e8b53960e38c8270ffadf86990a77559bc642a53dc832a0ca8e559a428.

CPU catalog:76 call-export sources,19/rank (eight bound Runner,11 direct).
Only seven bound roots/rank belong to the norm mapping. Trusted rank0 pickle
inspection, with GPUs hidden and without post-compile, recovers those same seven
cache keys from the serialized compiled forward results. Full source-byte equality
and all-rank discovery still need verification. Do not infer root status from
callability, names or controller callbacks. Observe actual artifact-to-live-call
bindings; retain all loaded-module paths before validation, plus target/non-target
root launcher checks. No replacement/unused v1-v4 attempt or new GPU series is
prescribed at this checkpoint. Model quality and indexer gates remain unchanged.

## Artifact-root provenance qualification: NEW v5

CPU-only discovery matches ALL28 serialized root source bytes/cache keys against
the original graph files (seven/rank,46 submodule references/rank). Report
`runtime-control/geometry-artifact-roots-all-ranks.json`, SHA
33db464e59736405d7893e3c45ffb152403c2be16599b5db55ca56b4ae8ba10a.
No post-compile/forward/GPU calls in that check. The observer now scopes actual
AOTCompiledArtifact.deserialize to its payload receipt, observes the actual live
CompiledFxGraph.after_deserialization call/runner/module, and joins the identical
returned artifact to vLLM's loaded store. Do not use the deep-copied serializable
result as a substitute for the live graph. Save the complete loaded-module catalog
before coverage checks and on failure; original-exact imported helpers are recorded
separately. Root source/config/binary/target/non-target coverage remains mandatory.

CPU gate310 pass6.94s; real vLLM concurrent load_all and Torch graph loading,
with only the outer AOT fixture wrapper reduced. First fixture-cleanup failures
and subsequent passes retained in `runtime-control/geometry-artifact-root-*-cpu.xml`.
Actual GPU/AOT validation is still required. No serving code, quant, native binary,
compiler math, quality floor or indexer gate is changed.

Commit this implementation/protocol, then freeze all sources through closure.
Exactly eight attempts, one each in this order: control-rank0, control-rank1,
control-rank2, control-rank3, geometry-rank0, geometry-rank1, geometry-rank2,
geometry-rank3. Same no-weights/no-forward/no-capture/no-timing workload and all
previous source/binary/seal/coverage gates. Any failed preparation, load, audit or
release stops the entire series; never launch remaining cases or replace an attempt.
All predecessor audits, logs and source hashes must pass before the next case.
Keep one GPU workload at a time; no native builds or source edits during the series.

Prepare once in8GiB/swap0 with GPUs hidden (trusted local pickle discovery only):

```bash
systemd-run --user --scope --unit=glm53-geometry-aot-v5-prepare -p MemoryMax=8G -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.prepare_glm53_geometry_loader \
  --pair perf/results/2026-09-10/rmsnorm-geometry-preload-qualification/pair-analysis.json \
  --manifest perf/results/2026-09-10/runtime-control/rmsnorm-geometry-preload-manifest.json \
  --mapping perf/results/2026-09-10/runtime-control/norm-graph-role-pairs.json \
  --output perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v5
```

For each LABEL in the fixed order, ONCE, conditional on all predecessors passing:

```bash
systemd-run --user --scope --unit=glm53-geometry-aot-v5-LABEL-launch -p MemoryMax=8G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_HOME=/usr/local/cuda-13.0 .venv/bin/python \
  -m benchmarks.kernels.check_glm53_geometry_loader launch \
  --manifest perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v5/LABEL/manifest.json
```

Launcher retains load/audit logs and statuses, plus independent GPU-release query;
child scopes use v5,16GiB GPU/8GiB CPU/swap0. Only if ALL eight complete successfully:

```bash
systemd-run --user --scope --unit=glm53-geometry-aot-v5-compare -p MemoryMax=8G -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m benchmarks.kernels.audit_glm53_geometry_loader compare \
  perf/results/2026-09-10/rmsnorm-geometry-aot-qualification-v5
```

Audit final closure against original sources/cache and independent GPU query before
releasing the freeze. v1-v4 stay terminal. This protocol does not authorize a model
causal series or promote a production/default/performance change.
