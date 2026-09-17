# Platform ops cards

How this box boots, builds, profiles and fails. Read the card for the
detected platform before the first boot; add to it in HANDOFF.md when the
box teaches you something new.

## Every platform

- Python: the repo venv (`<repo>/.venv/bin/python` or the venv named in the
  campaign brief) with `PYTHONPATH=<worktree>` so the worktree's sources
  win over the installed package. Lint with `uvx ruff check` /
  `uvx ruff format --check` on changed files.
- Boot only through the profile: `slimserve <id> --serve --host 127.0.0.1
  --port 8000 -y` (`--spec` / `--no-spec` to pin the drafter arm). A record
  with `status: in-progress` boots with `SLIMSERVE_SERVE_IN_PROGRESS=1`.
  `slimserve <id> --dry-run` shows the resolved command.
- Boot **detached** so the agent harness cannot kill the server with the
  session's process tree:

  ```python
  subprocess.Popen(cmd, cwd=worktree, env=env, stdin=subprocess.DEVNULL,
                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
  ```

  then poll `http://127.0.0.1:8000/health` until 200 (boots take 2-5 min),
  primer request, ramp request, then gates. Keep the boot script under
  `perf/results/<date>/` so the next session reuses it.
- Kill by PID lineage (`pgrep -f "vllm.entrypoints.openai.api_server"`,
  then the `EngineCore` children), wait for memory to release before the
  next boot. Never `pkill -f` a pattern the calling shell contains.
- One model resident at a time. Never rebuild native code while a server
  runs. Downloads are SlimServe's job (`slimserve <id> -y` fetches and
  resumes); do not block on a missing file, start the download and work on
  the adapter meanwhile.
- Raw artifacts: `perf/results/YYYY-MM-DD/<run-id>/` (git-ignored). Long-
  lived inputs: `perf/results/harness_assets/`. Durable scratch:
  `~/.local/scratch/<campaign>/`. Never `/tmp`.
- Profilers that already exist: `VLLM_QC_PHASE_PROF=1` (step-phase split;
  sync-inflated, trust the split not the totals), `VLLM_SYNCPROF=1`
  (host-sync census), the op census by `aten::empty`, the per-encoder /
  per-kernel timeline for the platform below. Prove the wall before fusing.

## Apple Silicon (metal)

- Working set: `recommendedMaxWorkingSetSize` is ~90% of physical. Set
  `sudo sysctl iogpu.wired_limit_mb=<physical MiB - 8192>` (e.g. 122880 on a
  128 GiB box) after every restart and verify before booting; the first
  request otherwise dies with `kIOGPUCommandBufferCallbackErrorOutOfMemory`
  and poisons the engine. Watch `vm_stat` on every boot; the previous
  engine's wired pages drain slowly (minutes) after it exits.
- Residency: weights are pinned (`Pinned N allocations` in the boot log);
  oversubscription shows as seconds per token and VM-compressor churn, not
  as an error.
- Boot ramp is mandatory: tiny primer (5-token prompt, 8 out), then one
  ~1000-token single-chunk request with decode, only then any multi-chunk
  prefill. A boot whose first request is multi-chunk can wedge the main
  thread in an event wait forever. Poison signatures:
  `kIOGPUCommandBufferCallbackError*` - kill both api_server and EngineCore
  and reboot. Repeated first-request timeouts on a build that gated clean
  earlier is driver degradation: restart the machine, do not bisect code.
- Build: metallib from `csrc/quixicore/metal/kernels/**/*.metal` with
  `xcrun metal -std=<the box's Metal standard> -O2 -I include/metal -I
  kernels/common`, then the `_quixicore_C` extension
  (`csrc/quixicore/tm_metal/qc_metal_serving.mm`) with clang++ against the
  venv's torch; `install_name_tool` the libc++ path and `codesign -f -s -`.
  Keep the recipe in `perf/results/harness_assets/build_*.sh`. ~40 s.
- **Never commit `vllm/quixicore_metal.metallib`** unless the box builds the
  same Metal standard as the tracked binary (an older macOS produces
  metal3.1 and would drop the newer boxes' tensor-ops kernels). State in
  the PR that the maintainer regenerates it with `cmake/metal.cmake`.
- Per-kernel attribution: `xctrace record --template 'Metal System Trace'`
  and join encoder labels to intervals; a least-squares fit over encoder
  intervals gives the per-op budget. Concurrent-dispatch regions collapse
  attribution into one row: profile with the region flags off, then
  re-enable. Metal GPU Counters are readable per kernel for GB/s.
- Numerics: MPS reductions can differ from a kernel's fp32 online order;
  `metal::exp2` is 2 ulp low at negative integers; fast-math reassociation
  rolls shas even when values are exact. bf16 recurrent state is a recorded
  quality hazard; keep fp32 state for linear-attention mixers.
- Power: AC power, `--power 100`-class settings in the bar engine, record
  wattage; a laptop on a small charger measures 20x slower.
- Only TP1 exists; scaling past one box is separate machines. Metal gets no
  host-RAM KV tier (unified memory); NVMe tier is the long-context story.

## NVIDIA (a100, rtx3090, rtx6000 / sm_120, and new cards)

- Build: `cmake --build build/temp.linux-x86_64-cpython-312 --target
  _C_stable_libtorch -j$(nproc)` (and `_quixicore_C`), copy the rebuilt
  `.so` into `vllm/` by rename, never overwrite a mapped file. Cap builds
  (`MAX_JOBS=8 NVCC_THREADS=2`, memory-limited scope) on boxes that OOM.
  Smoke: `python -c "import vllm._C_stable_libtorch, vllm._quixicore_C"`.
  Rebuild after any merge that touches `csrc/`.
- New architecture: add the platform to `slimserve/hardware.py:_classify`
  and `profiles.json` `platforms` (compute capability, VRAM, gate, notes),
  make sure `QUIXICORE_ARCHS` / `TORCH_CUDA_ARCH_LIST` intersect the card,
  guard optional subprojects (DeepGEMM) that lack a builder; list every
  SM80-gated feature with its status on the new card (shared-memory
  budgets differ: sm_120 has 99 KB, and a kernel compiled for 116 KB
  launches unchecked and fails silently unless the binding checks).
- Environment audit before the baseline: `nvidia-smi topo -p2p r`, locked
  clocks and power limits, CPU governor, `NCCL_*` exports in shell rc
  files (a box-wide `NCCL_P2P_DISABLE` forces host SHM), free host RAM,
  earlyoom thresholds, one GPU workload at a time.
- Tensor parallel is a sanity gate: TP2 >= 1.5x TP1, TP4 >= 1.5x TP2, TP8
  >= 1.5x TP4; similar tok/s across TP levels is a wrong design, not a
  local bug. PCIe-only boxes: custom all-reduce over P2P
  (`VLLM_CUSTOM_AR_ALLOW_PCIE=1`, buffer size raised for prefill chunks)
  beats NCCL; GeForce drivers disable P2P entirely - prefer EP and host
  tables over per-step all-reduces there.
- Graphs: `cudagraph_mode` FULL_DECODE_ONLY / PIECEWISE with a capture
  size that covers the largest speculative batch (registry test enforces).
- Profiling: Nsight Systems / the torch profiler for the per-kernel budget
  and launch counts; a "profiler pair" (before / after at c1 and c8) is the
  mechanism proof for every launch-count change.
- KV tiers: the pinned-host-RAM tier (`HostTierConnector`) and the NVMe
  tier are standing goals for every non-Metal profile; the eviction-
  restore acceptance with `VLLM_KV_TIER_VERIFY=1` is the only proof of
  restores.

## AMD (mi300x)

- AITER is a supported dependency; do not remove a working AITER path to
  drop the dependency, do fix AITER kernels when a workload exposes them.
- `VLLM_ROCM_USE_AITER=1` in the record env; `numa_bind` lands each rank's
  host arena on its socket. The host tier needs the connector generalized
  to the layout before it is enabled (issues #17 / #18).
- Kernel work in HIP under `csrc/quixicore/`; `~/QuixiCore/QuixiCore-ROCm`
  is the reference tree; serialized launches, device assertions and
  sanitizers are in scope.
- `block_size` and `sparse_mla_force_mqa` requirements differ per model
  backend; copy them from the closest existing record and state why.

## Intel (xpu) and anything else

Start from the closest record on the nearest platform, the parity-first
route (reference fork at parity, then optimize), and write the card as the
box teaches it.
