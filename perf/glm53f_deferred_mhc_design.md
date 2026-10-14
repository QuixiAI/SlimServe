# GLM-5.3 TP8: extend coefficient overlap across the consumer

Status: design only, 2026-09-09. No serving implementation or speedup claim.
The current frozen native/TC comparison must finish before changing serving.

## Evidence and hypothesis

The retained mHC projection operator overlaps deferred post/Sinkhorn
coefficients with the urgent normalized input and one KDA projection, then
joins before returning. Other transitions finish coefficients before running
their consumer. Coefficients are needed by the *next* transition, not by the
current KDA, MLA or MLP consumer.

The previous short-context trace records 57 `_mhc_finalize` calls totaling
513.569 us and 66 `_finalize_phase` calls totaling449.827 us. Those are kernel
sums across streams, not additive critical-path time or a predicted gain.
The existing projection-only router experiment did not improve serving.
This proposal changes the lifetime of pending coefficient work; it is not
another attempt to retain that rejected router projection path.

Baseline code: `vllm/model_executor/layers/glm5_next_mhc_project.py`,
`vllm/model_executor/models/glm5_next.py`. Local reference:
`csrc/quixicore/serving/mhc_channel_owned_ampere.cuh` and the urgent/deferred
launches in `csrc/libtorch_stable/custom_all_reduce.cu`. Channel ownership
itself is a separate, deeper design; simple allreduce+transition fusion was
already rejected in the September3 notebook.

## Proposed dependency change

Current KDA overlap:

`partials -> urgent input + side coefficients -> projection -> join -> consumer remainder`

Candidate:

`join previous coefficients -> partials -> urgent input + side coefficients -> full consumer -> next transition joins`

The final model post-mix needs an explicit join because there is no following
transition to perform it. The first pre-only site and unsupported batch sizes
retain the existing synchronous behavior. Arithmetic, precision, consumer
weights, routing, cache layout and allreduce order remain unchanged.

## Non-negotiable safety contracts

- Use the fixed runtime stream-owner lookup, never a CUDA pointer embedded
  in an AOT/Inductor graph. No module-global stream registry.
- Join before *any* read of pending post/comb coefficients, including final
  post-mix, fallback paths and any diagnostic access. This is not permission
  for a general custom op to expose arbitrarily unsynchronized outputs.
- Keep side-read partials/parameters and side-written outputs alive. An
  early return removes the current function's lifetime protection. Establish
  appropriate stream recording or explicit ownership; test immediate tensor
  release, allocator churn and exception paths. Do not just delete the join.
- Preserve compiler-visible fresh-output/aliasing contracts. Verify the
  actual compiled/captured dependency graph; source order is insufficient.
- Initially guard to supported SM80 GLM shapes, no LoRA and no concurrent
  forwards/DBO on one model owner. Separate engines must remain independent.
- Measure CUDA graph pool growth. Do not hide a large persistent workspace
  or conservative cross-stream retention behind a small kernel-time win.

## Validation ladder

1. Isolated full-consumer chains against the current synchronous transition:
   M0/1/8/16/32/64/65 plus prefill, alternating attention/MLP consumers and the
   final post-mix. Keep current coefficient tolerances and exact BF16 output
   gates; do not relax them for asynchronous execution.
2. Eager, compiled and changed-input graph replays; many layers/iterations,
   aggressive allocator reuse, two model owners, exceptions, and fresh-process
   disk-cache restart. Verify no post-return writes reach reused storage.
3. Eleven/34/45-layer representative timing, both warm and disjoint weights,
   then the real registered profile's health, text/image/tool canaries,
   exact-token c1/c8/c16/c32 and long/tier gates. Profiling separate from timing.

Decision: queued hypothesis. No code has been enabled. First use the exclusive
GPU interval to evaluate already-prepared projection, row-sharding and top-k
diagnostics, then choose the next serving experiment from measured benefit.

Evidence: `perf/results/2026-09-09/glm53f-tp8-sparse-tc/attempt01/`
`decode-census.json`, `long-decode-census.json`; runtime stream regression and
full cached restart under `glm53f-tp8-runtime-stream-fix/` and
`glm53f-tp8-runtime-native/attempt{01,02}/`.
