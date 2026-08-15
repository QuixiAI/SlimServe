# Metal M1 Ultra decode campaign — plan of record

Goal: make `slimserve dsv4-xxs-1` (DeepSeek-V4-Flash 0731 IQ2XXS-w2Q2K) decode
faster than antirez's ds4 on the same Mac Studio M1 Ultra.

Baselines (2026-08-10, artifacts under `perf/results/2026-08-10/`):

- ds4: 21.08 / 20.51 agg tok/s c1 (24.79 decode-only per its server log).
- SlimServe: 0.0099 agg tok/s exact-harness (8-token run), ~0.1 tok/s
  steady-state interval estimate. Prefill healthy (~100 tok/s).
- Correctness reference: the 8-token greedy run's `response_sha256`
  `db2846cf721bf30ebbe83219fe64bac2c7fb68aa36ebb7294e34ed0fa6ad935b`
  (prompt offset 1, 1000 input tokens). Zero-numeric-change batches must
  reproduce it exactly.

## Why it is slow (evidence)

Per decode step, V2 runner path (confirmed live): ~10k-14k torch-MPS launches
plus ~20 host-device syncs, vs ds4's ~700 fused dispatches in 2-3 command
buffers with exactly 1 wait and a 4-byte readback. Cluster inventory and the
ds4 design reference are in the 2026-08-10 notebook entry in
`perf/optimization_status.md`.

## Batch 1 — overhead removal, zero numeric change

Greedy output must be bit-identical to the reference sha. One server restart,
one 8-token harness gate, then a longer run if the gate passes.

- [x] C128 compressor early-out wrongly gated on `is_cuda()`; extend to Metal
      (`vllm/models/deepseek_v4/compressor.py:145,384`). Was running a
      128-wide gather+softmax+RMSNorm every step on every C128A layer
      instead of 1-in-128.
- [x] Per-layer SWA/compressed slot-table rebuild hoisted to a per-forward
      cache on the forward context (`vllm/models/deepseek_v4/metal.py`
      `forward_mqa`). Removes ~1.9k launches/step (43x duplicated work).
- [x] TurboQuant constants (scaled centroids, ones, arange, -inf sinks)
      cached instead of rebuilt per call
      (`vllm/v1/attention/ops/turboquant_native.py`,
      `vllm/v1/attention/backends/turboquant_attn.py:747`).
- [x] Greedy rejection-verify on GPU: replace the CPU while-loop (full-logit
      `argmax().cpu().tolist()` right after the target forward) with dense
      scatter + cumprod tensor ops; all-greedy decided from
      `SamplingStates.temperature.np` (zero syncs)
      (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
      `_mps_greedy_verify`, `rejection_sampler.py:_verify`).
- [x] `gumbel_sample` MPS branch: `all_greedy` hint from CPU sampling state,
      unconditional-noise fallback otherwise; scalar-column sync removed via
      0-dim broadcast indexing (`vllm/v1/worker/gpu/sample/gumbel.py`).
- [x] Tensorize MPS branches in step prep: `mps_segment_ids`
      (scatter+cumsum row ids) + shape-from-numpy/content-on-GPU rewrites of
      `prepare_pos_seq_lens`, `combine_sampled_and_draft_tokens`,
      `prepare_prefill_inputs`, `expand_idx_mapping`, `post_update`
      (`vllm/v1/worker/gpu/input_batch.py`), `compute_slot_mappings`
      (`vllm/v1/worker/gpu/block_table.py`), `prepare_dflash_inputs`
      (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`). Unit-verified
      on MPS against loop references
      (`perf/results/2026-08-10/batch1-overhead-removal/test_batch1_mps.py`).
      Gotcha found: `req_states.last_sampled_tokens` is `[max_num_reqs, 1]`,
      not `[max_num_reqs]`.

### Batch 1 measured verdict (2026-08-10)

- Correctness: PASS — gate run sha `db2846cf7...` identical to the
  pre-change reference, exact=true, same draft/accept counts. All Batch-1
  changes are numeric-neutral end to end.
- Throughput: NO CHANGE — 832.5 s for the 8-token gate vs 809 s baseline
  (0.0096 vs 0.0099 agg tok/s). Hypothesis FALSIFIED: host syncs and
  step-prep launch count were not the bottleneck.
- Pivotal diagnostic: a 15 s `sample` of EngineCore during live decode
  (`perf/results/2026-08-10/batch1-overhead-removal/
  enginecore_sample_during_decode.txt`) shows 78% of wall in ONE stack:
  torch einsum -> MPS bmm -> `executeMPSGraph` ->
  `GPURegionRuntime::encodeOpWithCommitAndContinue` ->
  `[AGXG13XFamilyCommandQueue commandBuffer]` -> `semaphore_wait_trap`.
  The CPU is blocked waiting for COMMAND-BUFFER CAPACITY: the Metal queue
  is saturated with already-committed GPU work. The engine is
  GPU-throughput-bound; something on the step path costs minutes of real
  GPU time per step (8 tokens / ~2-3 spec steps took 832 s).
- Keep Batch 1 anyway: it is correctness-neutral, removes real overhead
  that will matter once the kernel-time problem is fixed, and the unit
  harness protects it.

### Revised next step (do this FIRST next session)

Microbenchmark the metallib kernels in isolation on this M1 Ultra with real
DSV4 shapes to find the minutes-per-step kernel class. Suspects, in order:

1. `ggml_moe_a8_vec` (MoE gate/up IQ2_XXS + down Q2_K, 256 experts, top-6,
   43 layers x 2 calls/step) — only ever throughput-validated on M5 Max;
   a slow path on M1-family GPUs (different simdgroup/bf16 capabilities)
   would produce exactly this signature.
2. `deepseek_v4_sparse_attention` and the TurboQuant draft kernels.
3. bf16 anywhere inside metallib kernels: M1 has no hardware bf16; emulation
   inside a hot kernel would be catastrophic rather than the ~2x torch-level
   tax.
4. The einsum/bmm itself (`metal.py:107` `_o_proj` else-branch — check
   whether wo_a is actually unquantized at runtime, and the
   `dspark_turboquant.py:197` twin).

Method: time individual `quixicore_ops.*` calls with realistic shapes on an
idle GPU (synthetic weight bytes are fine for timing), plus
`torch.mps.synchronize()` bracketing; compare against the M5 Max per-call
numbers in `benchmarks/dsv4_metal_perf.md`. Then fix or replace the guilty
kernel(s) for the M1 target. Batches 2-3 remain valid but re-rank after
this: fp16 and mHC fusion matter only once no single kernel eats minutes.

## Batch 2 — fp16 instead of bf16 (numeric change, own measurement)

M1 has no native bf16 (measured ~2x vs fp16 on MPS glue and matmul).

- [ ] Add `"dtype": "float16"` to `profiles.json`
      `dsv4-xxs-1.platform_overrides.metal.engine` (registry merges engine
      args; env is replaced, engine merged — verify after edit).
- [ ] Verify fp16 acceptance in `DeepseekV4FlashMLABackend.supported_dtypes`
      (`vllm/models/deepseek_v4/sparse_mla.py:35`) and TurboQuant
      (`turboquant_attn.py:110` lists fp16 already); metallib GEMV entry
      points are dtype-parametric.
- [ ] Correctness: greedy 8-token output compared against the bf16 reference
      (token-level; sha may differ legitimately — inspect text + longer run).

Expected: up to ~2x on everything still running through torch-MPS.

## Batch 3 — bind/port kernels (the road to 21+)

The metallib binds only 11 ops; sources for more exist unbound under
`csrc/quixicore/metal/kernels/` (sampling, norms, MoE routing). CUDA op
signatures in `vllm/quixicore/ops.py` are the contract; torch reference
implementations are the correctness oracle.

- [ ] Bind v2 sampler kernels for Metal and relax
      `_use_native_sample_kernels()`'s `is_cuda_alike()` gate
      (`csrc/quixicore/tm_metal/qc_metal_serving.mm:642`,
      `vllm/v1/worker/gpu/sample/gumbel.py:24`).
- [ ] Metal mHC port: `dsv4_mhc_pre` / `dsv4_mhc_fused_post_pre` /
      `dsv4_mhc_post` / `dsv4_hc_head`. Largest launch cluster
      (~3.3k ops/step, fp32 torch reference at
      `vllm/model_executor/kernels/mhc/torch.py`). ds4's equivalents
      (`~/ds4/metal/dsv4_hc.metal`) prove the fusion shape: hc_pre+norm one
      dispatch, hc_post+residual one dispatch.
- [ ] RMSNorm/fused_add_rms_norm Metal binding + a
      `MetalPlatform.get_default_ir_op_priority` returning
      `["quixicore", "native"]` (`vllm/platforms/metal.py`).
- [ ] kernel_config hygiene: `moe_backend`/`linear_backend` default "aiter"
      is inert on GGUF but a landmine; set "auto" on Metal in
      `MetalPlatform.check_and_update_config`.
- [ ] `_o_proj` 8-GEMV Python loop -> single grouped call
      (`vllm/models/deepseek_v4/metal.py:77`), same for the DSpark twin
      (`vllm/models/deepseek_v4/amd/dspark_turboquant.py:170`).
- [ ] Fuse the fp16 fallbacks that remain hot after measurement (q/kv
      RMSNorm ~16 ops/layer, MoE router ~12 ops/layer, SwiGLU clamp ~8
      ops/layer, MoE weighted reduce 4 ops/layer) into metallib kernels,
      following ds4's fusion boundaries (router+top6 one dispatch;
      gate+up+SwiGLU one; down+reduce+residual one).

Expected: collapses remaining per-layer torch glue toward ds4's ~15
dispatches/layer. This batch is where 21 tok/s becomes reachable.

## Batch 4 — structural (only if profiling still demands it)

- Command-buffer/commit amortization: eager torch-MPS commits far more often
  than ds4's 2-3 CBs/token. Options: larger fused kernels (preferred, Batch
  3), `torch.mps` commit hints, MPSGraph capture of the stable decode body.
- ds4-style CPU-encode/GPU-execute overlap is not reachable from eager
  PyTorch; revisit only with a captured graph.

## Measurement protocol (every batch)

1. Rebuild only if native code changed; restart server
   (`slimserve dsv4-xxs-1 --serve -y`), wait `/health`.
2. Gate: exact harness, c1, 1000-in/8-out, offset 1 (fast; sha-comparable
   for zero-numeric-change batches).
3. If gate passes and speed allows, longer run (100 out, then 2000 out once
   >5 tok/s) for the notebook.
4. Notebook entry per batch in `perf/optimization_status.md`: baseline,
   hypothesis, correctness result, measured throughput, decision, artifact
   path under `perf/results/2026-08-10/`.
5. End state: re-verify ds4 target with clean memory, then the full fixed
   workload comparison at c1 (and c8 if batching is healthy).

## Risks

- MPS op gaps: a tensorized replacement may hit an unimplemented MPS op —
  caught at smoke time; fall back per-site to numpy+single-H2D-copy.
- Non-greedy sampling on Metal still routes to Triton stubs (pre-existing);
  campaign only guarantees greedy/temp-0 serving until Batch 3 sampler
  binding.
- Draft-side (DSpark) op storm is measured smaller than target-side but not
  zero; if post-Batch-1 profiling shows it dominant, apply the same
  treatment to `dspark.py` / `dflash/speculator.py` internals.
