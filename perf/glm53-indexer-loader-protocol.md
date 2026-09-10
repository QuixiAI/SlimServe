# GLM53 indexer correction loader qualification

Status: runnable AOT/leaf series implemented; CPU testing and GPU qualification
in progress. Serving integration remains pending.
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

## Runnable AOT/leaf qualification v1

This section supersedes the foundation's no-GPU next-step note above. Commit the
complete implementation and this protocol before preparation. One preparation,
eight sequential GPU processes, one audit per process, one final pair audit:
control ranks0,1,2,3 then correction ranks0,1,2,3. One fresh private namespace per
process. Stop on the first load/leaf/audit/release failure; no replacement starts.

Each rank loads all seven actual serialized AOT roots/46 entries. Independent
graph inventory expects25 bound launchers, two target bindings, and exact
non-target equality against that rank's control. Binary observation covers the
original and appended images; original metadata is not relabeled as correction.

After target seal, exercise **each of the two actual graph-held target runs**,
not an independently compiled substitute. Each binding runs30 cases: rows
(1,3,16,640,7616), seeds(530901,530902), magnitudes(.125,1,8). Both seed and
seed+100 inputs reproduce the completed isolated probe's input hashes. Five
observations (original, repeat, changed graph replay, changed eager, restored
graph replay) must reproduce its original or corrected output hashes by arm.
Correction flags must reproduce the prior GPU flag hashes at every observation.
Initialize each fixed arena to171 before a case; guard row0 and all unused rows
after the live slice must stay171. Input/weight/output/stride/gate guards and
eager/replay equality use the existing norm probe's checks.

This totals480 bound-leaf cases,960 unique input phases across the eight runs,
and2,400 repeated/replayed observations. Only four small real layer11 norm
tensors are read; the summary/auditor explicitly require four, not zero. There
is no outer-model deserialization, model forward, serving request, timer or TPS.
The original oracle failure stays historical; corrected hashes inherit the
previous one-ULP qualification only on that exact matrix.

Global binary sealing occurs after leaf validation. Re-inventory actual graphs
afterward; nothing may be rebound. Audit the complete leaf sequence and all raw
receipt hashes independently; controller attestations alone are insufficient.
Fresh source freeze spans preparation through the final successful pair audit
(or retained terminal failure audit). Keep the qualified correction kernel and
loader policy unchanged. No concurrent GPU work, serving, native build or tuning.

Resources: CPU preparation/controller/audit8GiB; GPU process16GiB, swap0 for all.
CUDA_VISIBLE_DEVICES=0,1,2,3 for rank-matched processes; each operates on its
prescribed rank. GPU identities/driver/power settings must match the isolated
qualification before and after each process. Confirm released contexts.

```bash
systemd-run --user --scope --unit=glm53-indexer-aot-prepare-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.prepare_glm53_indexer_correction_loader --prepare-series --output perf/results/2026-09-10/indexer-aot-v1
systemd-run --user --scope --unit=glm53-indexer-aot-controller-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 bash -c 'set -e; for mode in control correction; do for rank in 0 1 2 3; do .venv/bin/python -m benchmarks.kernels.check_glm53_indexer_correction_loader launch --manifest perf/results/2026-09-10/indexer-aot-v1/$mode-rank$rank/manifest.json; done; done'
systemd-run --user --scope --unit=glm53-indexer-aot-pair-v1 -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.kernels.audit_glm53_indexer_correction_loader compare perf/results/2026-09-10/indexer-aot-v1
```

An AOT/leaf pass still does not prove full-model quality, scheduler/padding
coverage, arbitrary multi-stream reentrancy or serving graph-capture behavior.
Those require the subsequent opt-in serving integration and separately prescribed
control/correction/return workload. No production or stable-baseline promotion.
