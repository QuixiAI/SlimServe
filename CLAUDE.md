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

- SlimServe is profile-driven. `slimserve/profiles.json` is the source of truth
  for supported models, quants, engine args, and environment.
- A profile is **model x quant x platform x config**. Platform is part of a
  profile's identity, not a list it ranges over: a config tuned for MI300X is
  not the same profile as one tuned for A100, even for the same model and
  quant. MI300X and Metal each need their own.
- The id a user types carries no platform (`slimserve dsv4-q4ktail-2`) because
  the CLI already detects the machine. Each id therefore stores one record per
  platform under `variants`, and every record states its own `platform`.
  `registry.describe(id)` gives the shared fields plus the platform list;
  `registry.variant(id, platform)` gives one platform's config.
- Never widen a profile to a platform it was not tuned and validated on, and
  never reach for a `platform_overrides`-style escape hatch to make one record
  serve two platforms -- that mechanism existed, it let an A100 config stand in
  for MI300X, and it has been retired. Add the platform's own record instead.
  Two tests enforce this: `test_a_profile_is_one_config_per_platform` and
  `test_no_profile_carries_another_platforms_environment`.
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
- Always merge. When a pull, rebase, or merge collides with local dirty state,
  semantically merge both sides — never resolve a conflict by picking one
  side wholesale.
- EXCEPTION, and it overrides "always merge": if the remote history was
  rewritten, do NOT merge. `git fetch && git reset --hard origin/<branch>`,
  then re-apply only your own unique work on top. Merging a stale clone into
  rewritten history restores everything the rewrite removed -- this happened
  on 2026-08-30, when a merge resurrected a purged credential and 19,427
  superseded commits. A merge that reintroduces deleted paths is never the
  correct resolution, however clean the semantic merge looks.
- Deployment configuration NEVER goes in the repo: no systemd units, env
  files, logrotate or tmpfiles entries, no host paths, usernames, ports or
  keys. `deploy/` is git-ignored and a CI guard
  (`.github/workflows/no-deploy-config.yml`) fails the build if it or a
  literal credential returns. Operators keep those files on their own
  machines. Parallel implementations of the same file must be unified
  so every platform's validated path keeps working, and the losing copy's
  functional changes must be grafted into the survivor, not discarded.
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

## Serving policy (standing)

- SlimServe serving ALWAYS has automatic prefix caching, automatic tool
  calling, and thinking enabled, and NEVER uses greedy sampling. These are
  enforced as registry-level `_SERVING_DEFAULTS` plus explicit per-profile
  values and a registry test. Prefix caching admits NO opt-out: every
  platform record states `enable_prefix_caching: true` and the registry
  test rejects anything else (operator directive 2026-08-30). For the
  others, a profile may opt out only for a model-level impossibility,
  stated in a note.
- Never rely on engine-layer defaults for these: vLLM silently defaults
  prefix caching OFF for hybrid (mamba/GDN) models, which shipped a
  profile with a 0.0% hit rate and full-history re-prefill on every turn.
  Production load is dominated by agentic traffic - long shared prefixes
  extended turn over turn - where prefix caching approaches a 100% hit
  rate and its absence is catastrophic, not marginal.
- Benchmarks and diagnostics sample at the model's recommended settings
  (temperature 1.0 / top_p 0.95 / top_k 20, seeded for reproducibility),
  never temperature 0, and never disable thinking to save time.
- Main KV is ALWAYS bf16 (`kv_cache_dtype: auto`) on rtx3090 and a100
  profiles - enforced by test: quantized KV was implicated in multi-turn
  tracking errors on Qwen3.8-Flash-Next and root-caused on rtx3090
  (operator 2026-08-29). The DSV4/A100 bf16 sparse-MLA page path is the
  NFP8=0 instantiation of the merged decode plus the fused bf16 insert,
  bf16 compressor stores, and bf16 prefill gather - design and gates in
  csrc/quixicore/dsv4_bf16_kv_design.md (kernel parity 2.5e-04, boot with
  TQ draft + full graphs, deep-context recall). Still owed there: the
  fp8-vs-bf16 exact-token throughput A/B and KV pool re-sizing (bf16 rows
  are 1024B vs fp8's 584B). bf16 main KV is ASPIRATIONAL on the remaining
  platforms: they keep their qualified configs (Metal fp8_ds_mla and the
  qwen38-nvfp4-1-tq TurboQuant variant) and flip only with an on-box
  requalification pass. One noted a100 carve-out (operator-approved
  2026-08-30): glm52-q2k-4 serves fp8 main KV at 131072 because 65.8 GiB
  of Q2K weights per 80 GB rank make bf16 KV at that length physically
  impossible; the record's note states the arithmetic and glm52-q2k-8
  remains the bf16 model-default-context record. Draft-model KV (DSpark
  TurboQuant k8v4) is exempt
  everywhere: rejection sampling verifies every draft token against the
  target, so draft KV precision can only affect acceptance rate and speed,
  never output content.
- Every profile serves its model's DEFAULT context unless genuinely
  impossible on the platform (operator 2026-08-29): GLM-5.2 202752,
  DSV4-Flash 1M, Qwen3.8-Flash-Next 262144 - as configured in the GGUF
  or config.json. Do not cap max_model_len to fit the VRAM KV pool;
  deep-context concurrency comes from the CPU-offloaded KV tier. A
  max_model_len ABOVE the default (the MI300X GLM records' 1048576) is
  that platform's explicit, validated extension - not a template.
- CPU-offloaded KV (the pinned-host-RAM tier, `HostTierConnector`) is the
  standing goal for ALL non-Metal profiles; Metal instead gets NVMe-backed
  KV offload later (unified memory makes a host-RAM tier meaningless
  there; issue #19). ENABLED and validated on qwen38fn-fp8-8/rtx3090
  (2026-08-28, mamba state-geometry fix landed: tail states are the
  engine's frozen align-mode boundary snapshots) and on all seven A100
  profiles (2026-08-29 WildChat deep-context sweep; DSV4's packed
  cross-layer slab registers directly, the GLM records force the packed
  layout via enable_cross_layers_blocks). The eviction-restore acceptance
  (marker recall after full GPU-pool eviction) is the standing check for
  tier changes. MI300X still needs the connector generalized to its
  layout (issues #17/#18); enable there only with on-box validation.
