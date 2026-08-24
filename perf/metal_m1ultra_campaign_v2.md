# Metal M1 Ultra campaign v2 — independent audit + amended plan (2026-08-10)

Scope: independent audit of `perf/metal_m1ultra_campaign.md` (the active session's
plan of record) plus an amended plan. This file supplements v1; it does not
replace it. Evidence: full reads of `perf/optimization_status.md` (Metal
entries), `perf/baseline_status.md`, `benchmarks/dsv4_metal_perf.md`, a code
audit of `~/ds4` (DwarfStar), the vendored QuixiCore-Metal tree under
`csrc/quixicore/metal/`, and the current working tree.

## Verdict

The v1 plan is directionally correct and Batches 1-2 are exactly right. Three
things need amending:

1. **Batch 3 is mispriced.** The metallib already contains all 274 kernel entry
   points from the vendored QuixiCore-Metal tree (79 `.metal` files, 302
   `tk::launch_*` host launchers in `csrc/quixicore/metal/kernels/common/tk_launch.h`);
   only 11 are pybound in `csrc/quixicore/tm_metal/qc_metal_serving.mm:639-715`.
   Most Batch-3 items are ~10-line pybind wrappers, not kernel work. The one
   genuine port is mHC (no `dsv4_mhc_*` MSL exists anywhere).
2. **Batch 4 is the centerpiece, not a contingency.** ds4's own numbers prove
   the ceiling on this box is set by dispatch/encode structure, not bandwidth
   (see roofline below). Converging on ds4's structure gets us to ~25 tok/s;
   beating it "as much as we possibly can" requires the native step tape +
   GPU-resident sampling + cross-token pipelining that ds4 does not have.
3. **The bar may not be 24.79.** ds4 ships opt-in DSpark speculation
   (`--dspark --mtp`, greedy-only, ≤5 draft tokens; `~/ds4/ds4.c:32589`,
   `:35614`). It was OFF in our baseline runs. We must measure ds4's best
   config and beat that, not its default.

## Roofline (what physics allows on this box)

Tensor census of the exact serving GGUF (86,720,111,488 B, 43 layers,
256 experts top-6 + 1 shared, MLA latent 512, MQA):

- Routed experts: 72.56 GiB total → 6/256 active ≈ **1.70 GiB/token**
- Dense (attention Q8, shared expert, compressors, norms, lm head): **~7.2 GiB/token**
- Per-token weight traffic, autoregressive c1: **~9 GiB**

| Regime | GiB per emitted token | Ceiling @600-700 GB/s effective |
| --- | ---: | ---: |
| Autoregressive (ds4's regime) | ~9 | **65-80 tok/s** |
| Spec k=5, acceptance ~3.5: dense 7.2 read once + expert union ~9-10 + draft 5×0.72 | ~4.7-6.0 | **100-140 tok/s** |

ds4 measured: 24.79 decode M1 Ultra — ~28% of peak BW (its own speed-bench
shows M2 Ultra 23.22 < M4 Max 26.76 despite 1.46x bandwidth: it is
dispatch/latency-bound, with zero Ultra-specific tuning; device gate is a
two-bucket M1-M4 vs M5 string match, `~/ds4/ds4_metal.m:2375`).

The drafter is a 3-layer MoE clone (dflash, 256 experts top-6): **0.72 GiB
active per draft token** — a full k=5 round costs ~3.6 GiB, under half of one
target step. Dense weights are 81% of AR per-token traffic, and batched verify
reads them once per round: speculation raises the *ceiling*, not just the
constant.

## What ds4 actually is (the target we must beat)

- ~750-840 dispatches/token re-encoded on CPU every token into **2-3 command
  buffers, one encoder each, exactly 1 wait** (`~/ds4/ds4.c:30744-30757`,
  split schedule `:26766-26999`). ~48 µs/dispatch average at 24.79 tok/s.
- **No cross-token overlap**: commits, waits, then does a **517 KB full-vocab
  logits memcpy + CPU argmax** per token (`ds4.c:30761`); its GPU-argmax path
  exists but is only used by speculative replay.
- Fusion boundaries worth copying (each = 1 dispatch): hc_pre+norm+mix;
  qkv pair + compressor quad; router+shared gate/up+SwiGLU; top-6 finalize;
  IQ2_XXS gate+up+SwiGLU over all 6 experts; Q2_K down+sum6+residual;
  shared-down+HC expand. Net ~17-20 dispatches/layer.
- Kernel weaknesses we can exploit at c1: fp32-everything matvecs, scalar
  IQ2_XXS unpack with per-threadgroup LUT reload+barrier
  (`~/ds4/metal/moe.metal:3134-3183`), 64-thread threadgroups (nsg=2) on
  pre-M5, full fp32 activation re-read per threadgroup (no Q8_1 activation
  quant), and its best MoE kernels (nsg=1 fixed-route, pack2_overlap) are
  **M5-gated** — unclaimed headroom on M1 (`~/ds4/ds4_metal.m:38033-38044`).
- No continuous batching, no paged KV; c8 is flat (~35 agg on M5 Max).

## Amended plan

Batches 1-2 as in v1 (status: 5 of 6 Batch-1 items already in the working
tree; checkboxes in v1 lag — GPU greedy verify and gumbel sync removal are
done). **Next action is measurement, not code: run the 8-token sha gate.**

### Batch 2 amendments (fp16)
- Real blockers found: `vllm/models/deepseek_v4/sparse_mla.py:35`
  `supported_dtypes = [torch.bfloat16]` needs the edit (not just a check), and
  `mla.metal` / `add_norm.metal` are hard-typed bf16 (fp16 variants are edits,
  not instantiations). Acceptable interim: fp16 glue with bf16 kernel islands.
- Gate must include **acceptance length**, not just output text: a numeric
  change that silently degrades DSpark acceptance kills the endgame multiplier.

### Batch 3 repriced (bindings; pybind-surface unless marked)
In measured-impact order:
1. `lm_head_argcat_partials`/`_reduce` (+ a q8_0 instantiation for the OutQ8
   head; kernels at `csrc/quixicore/metal/kernels/quantization/lm_head/lm_head.metal:176,219`)
   — kills full-logit materialization and the worst sync.
2. Sampler suite (`argmax`, `top_k/p_sample`, penalties) + relax
   `_use_native_sample_kernels()` `is_cuda_alike()` gate (`gumbel.py:24`).
3. Spec-verify suite — `spec_verify_linear`, `rejection_greedy_sample`,
   `sample_recovered_tokens`, `spec_compact` (`sampling.metal:473,713,780,661`)
   are written to vLLM's contract. Bind these; keep the new Python
   `_mps_greedy_verify` as the correctness oracle.
4. `moe_route_grouped` (`moe.metal:100` — the DeepSeek noaux_tc router,
   already correct) — ~12 torch ops/layer → 1 dispatch.
5. `rms_norm`, `rms_norm_add`, `qk_norm_rope`; then SwiGLU (`glu.metal`),
   `moe_finalize`/`moe_gather`, `embedding_lookup` + `quantized_embedding_q2_K`.
6. `_o_proj`: hoist the 8 static per-group weight slices to load time
   (replace the parameter, don't cache a duplicate) and collapse the loop
   (`vllm/models/deepseek_v4/metal.py:77-101` + drafter twin
   `amd/dspark_turboquant.py:170-190`). Pure Python; ~1k launches/step.
7. **mHC Metal port — the one new shader.** ~3.3k ops/step today (20 sinkhorn
   iters/call, and `mhc_fused_post_pre` runs the loop twice). Fusion shape is
   proven in `~/ds4/metal/dsv4_hc.metal:1324` (hc-RMS+mix+split+weighted-sum+
   norm in ONE dispatch); oracle at `vllm/model_executor/kernels/mhc/torch.py`.
   Apple lesson recorded 3x in the vendored tree: threadgroup staging loses to
   cache reuse — do not port CUDA smem instincts.

### Batch 4 promoted: the native step tape (the road past parity)
Extend `qc_metal_serving.mm` from per-op `TorchEncoder` to a step encoder:
(a) first per-layer mega-encode (one C++ call encodes a whole layer's
dispatches into the current CB); (b) then a full-step fixed tape — prebuilt
argument table, whole decode body encoded C++-side into 2-3 CBs, one wait,
GPU sampling, 4-byte readback. This is ds4's architecture (its "fixed DS4
tape", `ds4.c:27505-27512`) done inside vLLM.
Then go where ds4 can't:
- **(c) Cross-token pipelining**: with sampling GPU-resident, encode step N+1
  while N executes. ds4 waits + CPU-argmaxes every token; this is worth ~1
  dispatch-latency-wall per token to us and is structurally unavailable to
  their default path.
- Note: today zero CB infrastructure exists (no ICBs/heaps/events anywhere in
  csrc/vllm) — this batch builds it. `commandBufferWithUnretainedReferences`
  and residency sets are cheap adds while we're in there.

### Batch 5: kernel superiority at c1 (beat their matvecs on their weights)
- Q8_1-quantize the decode activation once per step; fp16 math with fp32
  accumulation in qgemv/qgemv_moe (M1 fp16 ALU = 2x fp32).
- Vectorized IQ2_XXS unpack with an fp16 grid LUT (ds4 already generates one
  for MXFP4 — `~/ds4/metal/generate_mxfp4_half_lut.py`; do it for the E8
  lattice) instead of scalar sign-branch loops.
- Ultra-aware geometry: wider threadgroups / more rows per TG / split-K row
  partitioning to feed 64 cores; ds4 runs 64-thread TGs and never
  distinguishes Ultra. Port the *ideas* of its M5-only nsg=1 fixed-route and
  pack2 kernels down to M1.
- Byte-neutral load-time SoA repack of expert blocks if profiling shows
  uncoalesced block reads (A100 precedent: +43% c1 from layout alone; keep it
  load-time replacement per repo policy — no lazy duplicate caches).

### Batch 6: speculation economics (the ceiling raiser)
- Fix the verify-width disaster: `ggml_mul_mat_a8` pads M→32 with a double
  transpose + zero-fill (`qc_metal_serving.mm:613-631`) — 4-16x waste at
  M=2..8. Add small-M tiles (or route through `decode_linear`/`qgemm_frag`
  simdgroup-matrix paths that already exist for M≥2).
- Drafter-side: same dispatch treatment (3-layer model, 5 sequential Markov
  head iterations in Python today); fold into the tape.
- Sweep k∈{3,5,7} once fast (ROCm lesson: spec-3 won at batch, spec-7 at low
  batch); track accepted-length in every gate.

### Batch 7: measurement honesty / bar setting
- Measure **ds4 + DSpark** (`--dspark --mtp`, greedy) on the fixed workload —
  that is the real number to beat; record in `perf/baseline_status.md`.
- Re-verify ds4 baseline with clean memory (v1's stale-81GiB-server caveat).
- Track the triple every batch: tok/s, dispatches/step, syncs/step (Metal
  System Trace or signpost census), not tok/s alone.

### Batch 8 (flank, optional): c8 blowout
ds4 c8 is flat (~35 agg, no continuous batching, no paged KV). Once c1 is
healthy, publish the c8 curve: dense traffic amortizes 8x across requests;
aggregate roofline is >200 tok/s. This is a second, separate victory.

## Staged expectations (roofline-informed estimates, to be replaced by gates)

| After | Est. c1 tok/s | Bound by |
| --- | ---: | --- |
| B1+B2 (done + fp16) | 2-6 | remaining eager glue + syncs |
| B3 (bindings + mHC) | 12-22 | Python dispatch of ~1-2k ops/step |
| B4 (step tape + GPU sampling) | 30-45 | kernel quality, ds4-parity structure |
| B5 (kernel wins) | 45-60 | approaching AR roofline (65-80) |
| B6 (spec verify healthy) | 60-100+ | spec roofline (100-140) |

## Risks
- fp16 numerics: keep reductions/softmax/logits fp32; gate on acceptance
  length as well as text.
- Engine-loop overhead: at 60 tok/s the whole step budget is ~16 ms including
  vLLM Python scheduling; measure scheduler overhead once the GPU side thins.
- ds4 with DSpark may move the bar; measure early (Batch 7 first item is
  cheap and should run this week).
- MPS op gaps in remaining eager sites: fall back per-site to numpy +
  single-H2D copy, per v1.

## Addendum (2026-08-10 late): the minutes-per-step mystery is solved

The Batch-1 "GPU-throughput-bound" wall was neither kernels nor glue: it was
Metal weight buffers rotating through the macOS VM compressor. Loading reads
an 80.76 GiB GGUF mmap while writing 93.49 GiB of Metal buffers; the page
cache evicted the buffers into the compressor, and decode (which touches every
weight per token) then ran at the compressor's ~0.44 GB/s instead of unified
memory's ~600 GB/s. vm_stat proved perpetual rotation (compressions ==
decompressions while decoding, pool never drains). All kernels measured
innocent in isolation: MoE ~66 ms, sparse attn ~27 ms, dense GEMVs ~30 ms per
verify pass.

Fix (landed): windowed madvise during GGUF load + post-load GPU resident
sweep (`gguf_weight_utils.py`, `metal_worker.py`). Result: 110 s/token ->
1.41 s/token (0.71 tok/s) with residency stable across decode.

Consequences for this plan:
- The staged expectations stand, but from a real starting point now:
  1.41 s/step of torch-MPS glue/launch overhead is the current target.
- Batch 3 (bind + mHC) and Batch 4 (native step tape) are the path from
  0.7 -> 10+ tok/s; kernel superiority (Batch 5) then contests ds4's 21-25.
- Keep watching vm_stat on every boot: any config that grows the footprint
  past ~115 GiB total demand re-enters the thrash regime (fp8 KV growth at
  c8, bigger contexts). The resident sweep logs its GiB and seconds at boot.
- fp16 (Batch 2) is live serving config; its sha differs from the bf16
  reference legitimately. The dumped fp16 completions under
  `perf/results/2026-08-10/residency-fix/completions_*` are the comparison
  texts for all subsequent zero-numeric-change batches.

## Batch 4 Stage 1 design: native step tape (written 2026-08-11, post drain-kill)

Evidence base: steady decode is 49.5% GPU-busy; step 292 ms = ~85 ms Python/torch
encode of the target forward + ~145 ms GPU (with pacing bubbles) + ~35 ms Python
sampler/drafter + bookkeeping. Syncs and drains are dead (op census clean); async
scheduling is neutral at c1. The only lever of size left is collapsing per-op
Python/torch encode into one C++ encode pass per step.

Architecture (ds4 precedent: one C++ loop, ~2.75 command buffers/token):
- `qc_step_forward` host op in qc_metal_serving.mm. At load, Python registers a
  per-layer descriptor (dict of persistent weight/cache tensors + layer kind
  flags) via `qc_tape_register_layer(idx, ...)`; per step, one pybind call
  passes the dynamic tensors (hidden states, positions, slot mappings, block
  tables, attn metadata products) and scalars, and C++ encodes all 43 layers
  onto torch's current command buffer through the existing tk:: launchers.
  Same kernels as the Python path => bit-exact for the custom-op subset.
- Torch ops that must become kernels inside the tape (from opcensus1, per step):
  wq_b bmm [8,6,4096]x[8,4096,1024] (43x), o_proj einsum->mm (43x), router
  linear [6,4096]x[256,4096] (43x, + existing _metal_router_topk), indexer
  softmax [6,8,512] (21x) + where/gather glue, residual adds/muls, and the
  898 _to_copy dtype conversions (resolve fp16-vs-bf16 uniformly inside the
  tape; qc_probe_convert modes are built for verifying Metal rounding == torch).
  Bricks already vendored: decode_linear (needs fp16 instantiation; fp32/bf16
  exist), gemm_v3, activations/softmax. Compressor: dsv4_compress_front is
  built and byte-exact-parity-proven (24/24) — required here since the eager
  chain is torch; replicate the c128 skip decision CPU-side (it already is).
- Parity/gating: new-GEMM ops cannot be bitwise vs torch mm (reduction order)
  => layer-local compare harness (captured step inputs, tolerance), then the
  serving trajectory-lottery gates: deterministic x2, coherent text, judged on
  step time; 8-tok sha expected to change only if the lm_head/logits path is
  taped (keep logits on the torch path in S1 to preserve the 5d4697 gate).
- Staging:
  S1a: fp16 decode_linear instantiation + microbench/parity vs torch mm at the
       exact shapes (6x4096x{512,1024,2048}, 8x6x4096x1024 as batched rows).
  S1b: qc_step_forward covering the uniform layer body (norms, qkv gguf gemvs,
       qk-norm+rope+insert, wq_b, mqa, o_proj, router+moe, mhc, adds), with
       env VLLM_QC_STEP_TAPE=1 and per-layer fallback to Python for layer
       kinds not yet covered (indexer/compressor layers first run hybrid).
  S1c: full 43-layer coverage incl. indexer + compressor + SWA variants.
  S2:  tape the sampler/rejection/drafter chain (gumbel + rejection kernels
       exist; drafter is 3 layers x 5 iterations of the same body).
- Expected yield: S1 -70..-80 ms => step ~210-220 (matrix ~19.5-21 tok/s at
  draw 4.61); S2 => ~180; + conversion purge => <=165 = ds4 aggregate bar
  beaten at median draw; decode bar (25.53) in reach at 49.5%->90% busy.

## Batch 4 S1 verdict (2026-08-11) and the pivot back to GPU time

S1a+S1b landed and gated: the tape runs 22/43 layer bodies natively,
bit-exact (sha identity both gates), lazy registration, per-layer step
tensors (KV group unification splits identical cache specs into
multiple groups — a canonical-layer shortcut mis-slots KV inserts; found
by tape verify mode, fixed, 0 mismatches over a 300-token run).
Perf: NEUTRAL (15.14 / 132.09 / step 291.7). With encode collapsed the
GPU is 98.6% busy during decode -> the step was never encode-bound at
292 ms; it is GPU-execution-bound (~280 ms/step = target forward +
5 sequential draft iterations + verify sampling). S1c (indexer layers)
and S2 (sampler/drafter tape) are POSTPONED: they buy nothing until GPU
work drops below the encode floor (~85 ms).

Next work items, ranked:
1. Per-kernel GPU-time census at M=6 decode: xctrace Metal System Trace
   (encoder labels already name every quixicore dispatch) or per-phase
   cb-census; produce a ranked table like the CPU op census.
2. Attack the top families. Candidates by dispatch count/size: MoE vec
   kernels (43x swiglu-vec + 43x down-vec, 36 rows each), sparse
   attention (43x), mhc pre/post (172 dispatches, 20 sinkhorn iters
   each), gguf gemvs (wq_b 32768 rows Q8_0 is the largest single GEMV),
   wo_a bmm (fp16, 64 MB/layer read) -> qc_decode_linear_bh swap now has
   a GPU-time motivation, not just an encode one.
3. Batch 6 speculation economics: 5 sequential drafter iterations are
   GPU-serial; measure acceptance-per-GPU-ms and tune block size.
