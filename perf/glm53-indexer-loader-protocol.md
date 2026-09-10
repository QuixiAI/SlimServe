# GLM53 indexer correction loader qualification

Status: CPU-tested foundation; runnable GPU series and serving integration pending.
The fixed recipe/profile and original production attention remain unchanged.

## Qualified starting point

The isolated correction on `f5ad4f884` passed120 all-rank cases/two phases,
unchanged max1BF16ULP oracle, detector agreement, non-target preservation and
graph/guard checks. Its extra-launch timing is diagnostic, not a speed win.
Completed result: `indexer-correction-v1/analysis.json` under2026-09-10, SHA
c6ac3d0399af92be467ef47831f512c4edee60ff08fe0772afb399c59a18cc65.
The original historical oracle failure and pending model-quality gate remain
separate. See `glm53-indexer-correction-protocol.md` for the full GPU record.

The kernel source is unchanged, SHA
24ccefc2f8afcd74105a4715dc7bd930cf788fd5192d35c54098c98cd00173b7;
all four qualified cubins have SHA
bd00effc35c7c0619ce74d3819aa3322cc110e423ce157ce855994f30188453e.

## Loader foundation

- `glm53_indexer_correction_loader.py`: a separate schema, correction mode and
  hook/event names atop the shared extra-launch lifecycle. KV remains the
  default adapter policy, with its existing dispatch behavior preserved.
- Compile the unchanged JIT with its original signature/options; compare key,
  in-memory bytes and exact private-cache file against qualification. Convert
  that compiled object to Torch's actual static CUDA launcher. No new kernel
  generation, binary substitution or permissive launcher fallback.
- Static ABI is five tensor pointers followed by int32 N; grid(N,1,1). The
  existing binary observer records the actual driver input before loading and
  retains live object/handle provenance afterward.
- Each graph target binding gets fixed `[8194,128]` uint8 storage, including two
  guard rows (1,048,832 bytes). A host view supplies the first N working rows to
  the unchanged qualified adapter. No CUDA allocation/compile in dispatch.
  This is an explicit8192-row diagnostic envelope, not an asserted scheduler
  limit: future serving installation must check actual padded batch limits.
- The graph inspector follows live `tuner.run` to the correction adapter and
  its actual static runner; it does not infer appended identity from the
  original combo's compile results. Arena address/geometry/device drift is
  rejected. Original compile-result/launcher metadata remains truthful.

The arena assumes serialized execution of a binding, as in the intended vLLM
forward stream. Arbitrary concurrent cross-stream calls are not qualified.
Serving/capture validation must establish the real lifecycle; a CPU test is not
proof of CUDA graph or concurrent-workload safety.

## CPU evidence

358 tests pass/20.50s;14 upstream Torch deprecation warnings. Initial131 pass/
7.72s retained. Coverage includes the real Torch static constructor/generated
N-grid launcher with simulated native driver, strict source/config/image/ABI
admission, fixed-arena view offsets/bounds/rebinding, future/graph binding,
handle drift, wrong-device rejection and existing KV/geometry loader/serving
regressions. CUDA driver and allocation are simulated where needed; no GPU or
model launch occurred. Ruff/diff checks pass.

`prepare_glm53_indexer_correction_loader.py` currently writes an **inspection
base**, not runnable per-arm manifests. It joins the completed GPU receipt to
all eight archived target uses, seven actual serialized AOT roots/46 entries
per rank,269 source/evidence hashes and the5,172-file original inventory. It
rejects qualified-kernel or archived graph drift. Completed audits are consumed
without rerunning their expired source freezes.

CPU8GiB/swap0, one OMP thread, CUDA hidden:

```bash
systemd-run --user --scope --unit=glm53-indexer-loader-inspection-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.prepare_glm53_indexer_correction_loader --output perf/results/2026-09-10/runtime-control/indexer-loader-inspection-v1.json
systemd-run --user --scope --unit=glm53-indexer-loader-cpu-final -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/slimserve/test_indexer_correction_loader.py tests/slimserve/test_indexer_correction.py tests/slimserve/test_indexer_precision_analysis.py tests/slimserve/test_kv_loader.py tests/slimserve/test_kv_loader_audit.py tests/slimserve/test_kv_serving.py tests/slimserve/test_kv_serving_preparation.py tests/slimserve/test_binary_observer.py tests/slimserve/test_geometry_loader.py tests/slimserve/test_geometry_loader_audit.py tests/slimserve/test_geometry_serving.py tests/slimserve/test_geometry_serving_runner.py --junitxml=perf/results/2026-09-10/runtime-control/indexer-loader-cpu-final.xml
```

Raw under `perf/results/2026-09-10/runtime-control/`:

- `indexer-loader-inspection-v1.json`:89f90eb95530b179f4150a5c2dbd37b87f5839f5e3ce62f4c5d169ee79fc240b
- `indexer-loader-cpu-{v1,final}.xml`

## Required next work before model execution

Add runnable per-arm manifests, controller and independent offline auditor using
the shared AOT lifecycle. Preserve actual-root inventory and full non-target
binding comparisons. The new dispatch has not run on GPU: exercise each actual
graph-held target with the original synthetic matrix, both phases, original and
qualified corrected output hashes, arena/input/output guards and replay checks.
Loading modules alone cannot qualify the static-launcher ABI or workspace.

Prescribe the all-rank control/correction order, fixed matrix and failure policy
before GPU work; use fresh private caches, CPU8GiB/GPU16GiB and swap0, one GPU
workload at a time. Freeze sources from preparation through terminal audit.
Only after that gate should this policy join the before-forward/capture serving
lifecycle and a separately prescribed model control/correction/return series.
No GPU/model process is prescribed by this foundation checkpoint.
