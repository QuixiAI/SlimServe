# SlimServe repository guidance

## Scope and ownership

- We own the complete serving stack, including Python orchestration, vLLM,
  ROCm/HIP code, communication paths, generated kernels, and handwritten GPU
  kernels.
- Do not stop at an upstream-library boundary or classify a kernel failure as
  external. Reproduce it, isolate the first failing operation, and fix the
  responsible layer in this repository or the owned dependency tree.
- Long-running and concurrent workloads are correctness requirements. A short
  smoke test does not replace the exact workload that exposed a failure.

## GPU implementation scope

- AITER is a supported dependency. Do not remove or replace a working AITER
  path solely to eliminate that dependency.
- AITER and its kernels are part of the stack we debug and fix when a supported
  workload exposes a failure there.
- Kernel work is in scope. Use serialized launches, device assertions,
  sanitizers, reduced reproducers, and targeted instrumentation as needed.

## Reference implementations

Use these local trees when checking algorithms, layouts, bounds, and kernel
behavior:

- `~/llama.cpp`
- `~/ds4`
- `~/QuixiCore/QuixiCore-ROCm`
- `~/QuixiCore/QuixiCore-Metal`

## SlimServe Orientation

- SlimServe is profile-driven. `slimserve/profiles.json` is the source of truth for supported models, quants, engine args, environment, and platform overrides.
- Missing model files are not a blocker. Run `slimserve <profile> ... -y`; `slimserve.fetch` downloads or resumes required files into `$SLIMSERVE_CACHE` or `~/models`.
- Do not hand-build unsupported serving commands when a profile exists. Use `slimserve <profile> --dry-run` to inspect and `slimserve <profile> --serve` to run.

## Agent Operating Discipline

- Never store anything that must survive a crash or reboot under `/private/tmp`
  or other system temp directories — machine deaths wipe them. Durable working
  state (worktrees, benchmark logs, handoff notes, fix branches) belongs in
  `~/.local/scratch` (create it if needed) or inside the repo's git-ignored
  areas such as `perf/results/`.

- Read the repo before deciding. Start from `AGENTS.md`, `perf/perf.md`,
  `perf/baseline_status.md`, `perf/optimization_status.md`, the relevant
  profile in `slimserve/profiles.json`, and the local implementation paths for
  the model/platform under discussion.
- Do not confuse "the process started" with "the system works." A serving path
  is not validated until the real SlimServe profile reaches health, serves the
  benchmark workload, passes correctness checks, and produces recorded TPS.
- Do not treat vLLM interval logger output as an authoritative benchmark.
  Interval logs are diagnostics. Use exact-token harness output, raw JSON, and
  reproducible commands for baseline or comparison claims.
- Do not make a placeholder fix and describe it as the real fix. If a change is
  an interim diagnostic, say so in the notebook and keep the production target
  explicit.
- Remove or clearly quarantine failed experiments. Do not leave disabled,
  memory-heavy, or misleading alternate paths in the serving code unless they
  are intentionally retained as documented diagnostics.
- When the user points at a precedent, inspect it before implementing. For this
  repo that often means reading the optimized ROCm/Metal path, GLM 5.2 Ampere
  path, DS4 kernels, and the QuixiCore CUDA/ROCm code that is actually relevant
  to the active serving path.
- Keep model downloads out of the reasoning loop. If SlimServe owns the
  download, run SlimServe and let it download or resume; do not block kernel or
  profile work because a model file is not already present.
- State uncertainty precisely. If only startup, smoke, or synthetic parity has
  been run, call it that. Do not imply end-to-end correctness, production
  readiness, or final performance without the corresponding evidence.
- Preserve user and prior-agent changes. The worktree may be dirty; understand
  nearby edits and build on them rather than reverting unrelated work.
- Finish the loop: implement, build, smoke, run the real profile or explain the
  concrete blocker, update the performance notebook, and leave the next command
  obvious.

## Kernel Work

- Develop serving kernels in this repo first. Vendored code used by SlimServe belongs under `csrc/quixicore/` or `csrc/libtorch_stable/`; modify those copies, then port the finished used pieces to `/home/ubuntu/QuixiCore/QuixiCore-CUDA`.
- Vendor only kernels and headers on the actual serving path. QuixiCore is a large kernel library; do not copy broad directories just because they exist.
- QuixiCore, `ds4`, `llama.cpp`, and vLLM Marlin are references and inspiration, not finished answers. Study them for layouts, scheduling, quant decode, and tensor-core strategy, then implement and tune the SlimServe path that this model and hardware actually need.
- SlimServe owns the inference stack all the way down to CUDA/HIP kernels. Do not assume an upstream or vendored kernel is "already done" when profiling shows room to improve.
- The primary purpose of this repo is to optimize serving performance for the target model on the target hardware. Keep pushing until the remaining bottlenecks are measured and defensible.
- When inference is already thoroughly optimized on one platform, study that implementation before implementing another platform. Preserve the algorithmic wins, data layout choices, fusion boundaries, and profiling lessons unless the new hardware gives a measured reason to diverge.
- For DeepSeek V4 Flash 0731, the ROCm implementation is the baseline reference for the current Ampere A100 work.
- For DSV4 0731 on A100, the target is native fused MoE: IQ2_XXS gate/up from vLLM's combined `[expert, gate|up, packed]` GGUF layout or split/aligned artifacts, SwiGLU, Q2_K down, and final weighted reduce.
- Do not use dequant or standalone Q2_K GEMV as the production DSV4 answer.
- Lazy persistent repack caches can duplicate tens of GiB and break TP2 memory. Prefer load-time replacement, aligned artifacts, or byte-neutral layouts.

## GLM 5.2/Ampere Reference

- Use the GLM 5.2 Ampere path as the local precedent: route/align, expert-contiguous gather, grouped GEMM, grouped SwiGLU, and finalize.
- For quantized A100 kernels, follow the Marlin lessons in `csrc/quixicore/a100_glm52_design.md`: layout-only repack, in-register narrow scale decode, `cp.async` staging, fp16 fragments, tensor cores for `M >= 2`, and aggressive decode-node fusion.
- The parked `q2k_moe_ampere.cuh` GEMV is diagnostic only; it was measured too small a win to be the serving solution.

## Validation

- Build native changes with `cmake --build build/temp.linux-x86_64-cpython-312 --target _C_stable_libtorch -j$(nproc)` and copy the rebuilt `.so` into `vllm/` for editable imports when needed.
- Smoke imports with `import vllm._C_stable_libtorch` and inspect relevant `torch.ops` schemas.
- End-to-end DSV4 performance must be measured through SlimServe profiles such as `dsv4-2` and `dsv4-4`, because those profiles own downloads, parser settings, DSpark/TurboQuant environment, KV dtype, and CUDA graph settings.

## Performance Notebook

- Rigorously record inference experiments and optimization data using the method in `perf/perf.md`, adapted from `/home/ubuntu/QuixiCore/QuixiCore-Metal/perf/perf.md`.
- Before changing or benchmarking performance-sensitive code, read `perf/baseline_status.md` and the relevant entries in `perf/optimization_status.md`. Know how the current profile compares with the other platform/profile baselines before interpreting a new number.
- Treat large throughput gaps as evidence, not background noise. If a profile that should be in the tens, hundreds, or higher tokens/s range produces 2 tok/s, assume something is wrong until the notebook and fresh measurements prove otherwise.
- Treat tensor-parallel scaling as a hard sanity gate. TP2 should be at least
  50% faster than TP1, TP4 should be at least 50% faster than TP2, and TP8
  should be at least 50% faster than TP4. These are minimum acceptable checks;
  the target should be better than that when the kernels, collectives, and
  scheduling are healthy.
- Similar TPS across TP levels is not fine. If TP2, TP4, or TP8 produce roughly
  the same throughput, treat it as evidence of a fundamentally wrong design, not
  just a local bug. Review how the better-performing SlimServe profiles and
  platform ports are implemented before inventing a new path.
- Stand on our own shoulders. Do not rediscover solutions we already built for
  another model, tensor-parallel size, or platform; start from the proven local
  design and adapt it with measurements.
- Keep the running notebook in `perf/optimization_status.md` and stable baseline snapshots in `perf/baseline_status.md`.
- Store raw benchmark logs and JSON under `perf/results/YYYY-MM-DD/<run-id>/`; `perf/results/` is git-ignored except for its `.gitignore`.
- Every retained or rejected optimization must have a baseline, hypothesis, correctness result, measured throughput, decision, and raw artifact location.

## Profile validation

- Live validation must discover every registry profile compatible with the
  current machine. Do not substitute one tensor-parallel size for another.
- Every supported profile must run with its registered DSpark drafter and
  TurboQuant draft KV configuration.
- For vision profiles, test both text and image requests. For text-only
  profiles, test text requests.

## Commit authorship

- Eric Hartford is the sole author. Do not add co-author or assistance
  trailers, and do not discuss automated assistance in commit messages.
