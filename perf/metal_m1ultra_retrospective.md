# Metal M1 Ultra campaign retrospective and ceiling analysis

Written 2026-08-12, at the close of the ds4-bar campaign (`dsv4-xxs-1`,
DeepSeek-V4-Flash 0731 IQ2XXS-w2Q2K, Mac Studio M1 Ultra 128 GiB).
Companion documents: `perf/optimization_status.md` (the full experiment
ledger this distills), `perf/baseline_status.md` (standing numbers),
`perf/metal_m1ultra_campaign_v2.md` (the plan of record it started from).
Everything here is sourced from gated measurements recorded in the
notebook; nothing is projected without saying so.

## 1. Where we stand

| Milestone | decode step | off1-2000 tok/s | mechanism |
| --- | ---: | ---: | --- |
| Bring-up (2026-08-10) | ~104 s/token | 0.0099 | VM-compressor thrash |
| Residency sweep + madvise | 1.41 s/token | 0.72 | pages stay resident |
| MTLResidencySet pinning | ~600 ms | 4.38 | faults eliminated (36x) |
| Long-context indexer port | ~500 ms | 7.94 | >2048 ctx unblocked |
| qgemv_mb + fusion wave + c128 fix | ~302 ms | 14.66 | weight-stationary M=6, -50 ms waste |
| Drain kill + Batch 9 waves 1-3b | ~270 ms | 15.9 | glue folds exhausted |
| **Wave 4A cos_sin memo** | **130.0-131.6 ms** | **31.3-31.6** | transient-allocation churn removed |

- ds4 bar (same machine, same GGUF): 25.53 decode / 25.88 steady —
  **beaten by +23%**, with speculation as the structural advantage
  (3.70x multiplier; ds4+no-spec comparison is not the served config).
- First long-context rows on Metal: 12K cold 8.8 / hot 28.8-29.5 tok/s,
  128K functional (sha 27748c3c) behind the `VLLM_QC_LONGCTX_SYNC`
  mitigation. Prefill measured ~74 tok/s LINEAR — the top open surface.
- Uncommitted tree: 9,100 insertions over 48 tracked files + 14 new
  files (kernels, host ops, routes, instruments). Section 10 maps them.

## 2. The ceiling (roofline, measured bytes)

Tensor census of the served GGUF (86,714,775,900 bytes = 80.76 GiB):
routed experts 72.56 GiB (1,728 MiB/layer x 43; **6.75 MiB per
expert**: gate+up IQ2_XXS 4.125 + down Q2_K 2.62); dense read every
step ~7.15 GiB (attention Q8 5.44, shared expert 1.07, lm_head Q8
0.52, router 0.08, dense ffn 0.04); embed 0.99 GiB (lookup only).

Per decode step (M=6 verify + 5 DSpark draft iterations, draw 4.505
tokens/step on the tracked trajectory):

| Traffic component | bytes |
| --- | ---: |
| Dense weights (read once, weight-stationary mb kernels) | 7.68 GB |
| Routed experts: 36 (token,expert) slots x 6.75 MiB x 43 layers | 10.96 GB |
| Drafter (~0.72 GiB active x 5 iterations, campaign-v2 figure) | ~3.9 GB |
| KV/indexer cache at 1K ctx | ~0.05 GB |
| **Step total** | **~22.5 GB** |

Effective bandwidth ladder (per-step traffic / step time):

| Regime | eff. GB/s | step ms | tok/s @ draw 4.505 |
| --- | ---: | ---: | ---: |
| **Us today (130.8 ms)** | **172** | 130.8 | 31.5 |
| ds4's kernel efficiency (33.6 ms for its 9.5 GB M=1 token) | 283 | ~79 | ~57 |
| llama.cpp-class sustained decode | ~400 | ~56 | ~80 |
| Best single kernel measured on this box (lm_head q8_0 M=1) | 738 | ~30 | ~147 |

Reading: **we are at ~2.4x below ds4-efficiency physics and ~4.3x below
the box's demonstrated kernel bandwidth.** The bar is beaten but the
machine is not exhausted. Two caveats keep the upper tiers honest:
(1) the step is one serial 43-layer dependency chain of mostly small
kernels — per-kernel occupancy, not aggregate bandwidth, is the real
constraint (the Batch 8 xctrace lesson), so the 130-147 tok/s row is
physics, not a plan; (2) tokens/step is an acceptance draw — spec
economics (block size, drafter quality) move the tok/s rows at fixed
step time. **A realistic campaign target is the ds4-efficiency tier:
step ~80-90 ms, ~50-55 tok/s.** Levers in section 9.

Prefill: measured 74 tok/s LINEAR (13.5 ms/token). ds4 does 277 on
this box; a weight-stream-flat MoE prefill is bandwidth-cheap (~86 GB
per 2048-token chunk = physics of thousands of tok/s), so prefill is
compute/structure-bound: ~500+ tok/s class ceiling on M1 Ultra fp16.
The linearity is the same pathology A100 had pre-fix ("re-streams
expert weights per small row group"); it is a solved problem shape.

## 3. What worked (ranked by step-time recovered)

1. **Wave 4A cos_sin marshalling memo: -139 ms (272->131), 2.07x.**
   One python-side memo of a bf16 cast+split of the rope cache.
   Mechanism (bisect + xctrace + arithmetic): ~1,300 transient MPS
   allocations/step in the insert chain created per-heap false hazards
   that serialized resident command buffers; "busy" trace rows hid it.
   THE campaign lesson: on MPS, rank targets by transient allocations
   adjacent to the hot CB chain, not kernel counts.
2. **MTLResidencySet pinning: 36x** (and the sweep/madvise before it:
   78x). Torch-MPS weights are pageable anonymous memory; macOS
   compresses "idle" weight pages in waves without it. Metal-only
   failure class; permanent boot requirement (+ `iogpu.wired_limit_mb`).
3. **c128_boundary metadata fix: -50 ms, bit-exact.** 41 idempotent
   full compresses per step because one metadata field was never
   threaded. Cheapest win of the campaign; found by phase-bracket
   counting, not profiling ms.
4. **Fused single-dispatch kernels for eager chains** (compressor tail
   -45 ms, fused mHC -39 ms, mhc_pre split -20 ms, router -6.5 ms,
   producer top-k, o-inv-rope, finalize, SwiGLU, RMS family): together
   took 498 -> ~302. Decaying returns were measured and obeyed — the
   last folds bought ~1 ms for 480 dispatches removed.
5. **Weight-stationary small-M GEMV (qgemv_mb) +15.6%** and **ds4-shaped
   multi-row MoE kernels -35 ms**: M=6 verify was re-reading every
   weight 6x; register-resident activations + rows-per-simdgroup fixed
   the dense side, threadgroup-staged LUTs + NR0=4 the MoE side.
6. **Drain kill (+0.5 tok/s, bit-exact)**: every per-step blocking H2D
   found and removed; the drain is CONSERVED (fixing one moves the
   wait), so it only pays when the last one dies.
7. **SwiGLU clamp fix + exact-pow2 e4m3 scales**: correctness wins that
   re-rolled the trajectory; retained on semantics, judged on step ms.
8. **Split-K sparse-MLA decode (2026-08-13): -4.2 ms (127.4 -> 123.3
   mean over 7 paired offsets).** H=64 query heads leave only B*64
   simdgroups at verify; partitioning the candidate walk + paged-v2 LSE
   reduce recovered the serial-chain latency. ULP class — gated by
   determinism x2 + paired step ms + 7-offset means.
9. **Q2_K SoA planes (-10% kernel, ~-0.5 ms) and sum6 down fold
   (bit-exact, ~-0.5 ms, deletes the [T,6,4096] round-trip)**: small
   but free; both proved the enclosing-loop bit-exactness property that
   later waves reused.

## 4. What didn't work (measured; do not redo)

- **Launch-count reduction as a thesis, twice.** Batch 1 (sync removal)
  and the mHC kernels both measured ~zero while the real wall was
  elsewhere (thrash, then faults). A kernel-count fold pays ~7 us;
  an allocation fold can pay 100x that.
- **The native C++ step tape: bit-exact, serving-NEUTRAL.** The
  interpreter was never the wall once drains died; retained default-off
  as the diagnostic that proved GPU-boundedness (its 98.6% busy reading
  corrected the fragmented-CB 49.5% artifact).
- **wo_a via custom bh GEMM** (MPS bmm wins), **iq2_xxs ulong-shift
  decode** (9% slower; the LUT decode is at its ceiling — ds4 uses the
  identical one), **swiglu/save_partial geometry variants**
  (launch-bound), **fused mhc post+pre wiring** (slower), **q8_0 mb
  hoist** (broke bitwise for 8-13%; reverted).
- **Spec block resize either direction** (wash/loss at 292 ms era —
  re-tune queued now that steps are 2.2x cheaper), **removing
  speculation** (3.70x multiplier), **async scheduling at c1**
  (nothing to overlap when next input = last accepted tokens).
- **A100/ROCm negative results that transfer**: don't shrink batch-1
  producer math (fixed-overhead-bound), don't add barriers/fences/extra
  dispatches to overlap tiny work, don't threadgroup-stage activation
  blocks that are already cache-resident, don't trust isolated-stream
  microbenches to predict serving (repeatedly falsified there too).
- **(2026-08-13 wave additions):**
  - **IQ2_XXS SoA/pairing on Apple GPUs** (2-4% slower, three layouts
    incl. the A100 paired gate/up): the LSU eats unaligned narrow AoS
    loads free; the A100 layout win does not transfer.
  - **Divergent load shapes inside a bit-exact template**: fp
    contraction flips 1-ULP results (bf16 masks it — never trust
    bf16-only bitwise passes). Pin the load shapes, change only bases.
  - **Software cross-threadgroup weight dedup (expert-grouped w13)**:
    bit-exact but 123 -> 230 ms. Pair-owners serialize a latency-bound
    kernel and the L2/SLC already dedups co-resident duplicate reads.
    Also: any per-threadgroup O(slots) scan must be width-gated before
    it meets prefill (6000 slots wedged the engine).
  - **Host fire-and-forget second-queue prefetch**: the two-queue
    premise IS real (probe: 243 GB/s stream while a latency walk runs
    9.7% FASTER), but serving warms can't be aimed — the GPU executes
    ~a step behind the encode point, per-CB host commits cost ~50 us
    on the critical path (43/step = +2.5 ms), and a once-per-step
    background burst was exactly neutral (no DVFS tax in serving).
  - **Spec block k=6**: -1.3 tok/s mean over 7 offsets (+14.6 ms/step
    for +0.31 draw); k<5 is architecturally invalid (DSpark block 5).
  - **Method: identical shas after a ULP-class change mean
    "not engaged" until liveness is proven** (the H=64 target-128
    bug); **single-offset draw/tok-s is lottery — decide on paired
    step ms + means over >=5 offsets** (the off1 draw-collapse scare
    reversed with n=7: ON 4.33 vs OFF 4.13).

## 5. Surprises (the five wrong walls, in order)

The campaign's bottleneck diagnosis was wrong four times before it was
right, and each wrong theory was killed by a measurement worth keeping:

1. "Launch-bound" (bring-up) — actually VM-compressor thrash from
   working-set oversubscription. Killed by vm_stat + footprint.
2. "Sync-bound" (Batch 1) — syncs waited on real queued work. Killed by
   removing them: zero change.
3. "CB-submission-bound" — per-CB cost theory died when 10.8k removed
   launches changed nothing on healthy steps.
4. "CPU-encode-bound, kernels done, GPU 49.5% busy" — an ioreg artifact
   of fragmented CBs; the step tape saturated encode and proved
   GPU-execution-bound at 98.6%.
5. "GPU is 97% busy executing kernels" (pre-4A) — CB-level "busy"
   included intra-CB hazard stalls; removing allocation churn halved
   the step while busy% barely moved.

Other genuine surprises: MPS transcendentals match `metal::precise::*`
bitwise (probe method) except softplus — enabling bitwise fusion with
split boundaries; MPS pairwise-tree vs sequential reduction orders are
op-shape-dependent (both probed); torch MPS `exp2` is approximate at
integer inputs (real semantic bug found + fixed); `repeat_interleave`
with tensor repeats host-syncs; a >64K-token prompt sends MPSGraph's
encode queue into >100x degradation (mitigated with scoped syncs);
`iogpu.wired_limit_mb` resets on machine restart (OOM error class).

## 6. Instrument playbook (hard-won, reusable)

- `VLLM_QC_OP_CENSUS=<n>`: one-shot torch.profiler at step n; the
  chrome trace's python stacks are what found the 4A marshalling.
  Rank by `aten::empty` count (transient allocations), not kernel count.
- xctrace Metal System Trace: FIRST duration column only; CB-level rows
  cannot rank kernels and count hazard stalls as busy.
- Sync-bracketed phaseprof (`VLLM_QC_PHASE_PROF=1`): the split is
  structural, the absolutes are ~95% inflated for tiny-op regions.
  Bracket COUNTS are trustworthy (found c128 waste).
- ioreg GPU busy: only comparable within one CB regime.
- Offline kernel census: pipelines iterations, hides serial-occupancy
  cost; serving is one dependency chain. Demand critical-path evidence
  before fusing a bucket.
- Trajectory-lottery protocol: acceptance draw swings +-1.5 across
  offsets; judge kernel changes on paired step time + mean matrix tok/s
  across offsets. Bit-exact classes gate on sha identity; ULP classes
  gate on determinism x2 + coherent text + step time.
- ds4's discipline worth copying: every optimization ships with a
  bit-identical-logits A/B harness and a per-change `DISABLE_*` env.
  We converged on the same pattern (`VLLM_QC_*` kill switches).

## 7. Cross-platform lessons (A100/ROCm) — applied and pending

Applied on Metal already: multi-row/shared-activation MoE GEMV (=A100
route batching), fused SwiGLU epilogue at the W1 boundary, width/
boundary-threshold audits (c128 fix is the same bug class as the A100
capture-width cliff), spec economics measured not assumed.

Pending, ranked (details in the A100/ROCm entries, lines cited in the
notebook):

1. **Load-time SoA repack of IQ2_XXS/Q2_K** (scale plane | aligned code
   plane, byte-neutral, bit-identical). The MXFP4 analogue was 2.0x on
   the fused GEMV at every width and +26% c1 end-to-end. Our MoE
   kernels run 260-320 GB/s; this is the most credible single lever
   toward 400+. Must be load-time replacement (93.73 GiB pinned leaves
   no room for duplicates — the CLAUDE.md repack rule).
2. **Weight-stream-flat wide MoE tile for prefill** (the A100 mxfp4
   grouped tile went 8.8-57x by making cost flat in tokens; llama.cpp's
   mul_mm_id is the Metal-native reference shape).
3. **Two-regime MoE with a measured crossover** (A100: fused GEMV wins
   below ~72-768 tokens depending on quant; never let the prefill tile
   leak into decode).
4. **Same-shape dispatch merging** (A100 merged sparse-MLA sources,
   `max_abs_diff=0`, +1.3-2.1% e2e; audit our layer for remaining pairs).
5. **Quality-free quant upgrade probe** (Q4K-tail matched IQ2_XXS c1
   exactly on A100; if our decode stays occupancy-bound, a better tail
   may be near-free — quality per token is a campaign deliverable too).

## 8. External engines — what we already match, what we don't

Already matched or exceeded: IQ2_XXS threadgroup-staged LUT decode
(ours = ds4's = llama.cpp's shape), NR0=4/NSG=2 row geometry,
weight-stationary small-M dense GEMV (llama.cpp's mul_mv_ext skips
IQ2_XXS entirely — we cover it), Q2_K masked-nibble decode, residency
pinning (stronger than llama.cpp's heartbeat), fused router top-k,
single-wait step structure, bit-exact A/B discipline.

Not yet applied, ranked by expected decode value:

1. **Split-K decode attention + reduce kernel.** ds4 uses 12 KV splits
   (96 threadgroups vs 8 on a 64-core GPU); llama.cpp grows nwg to 32
   with a dedicated reduce kernel. Our `mla_decode_fp8_sparse` explicitly
   rejects `partition_size != 0` ("not wired yet",
   `qc_metal_serving.mm:199`) — the partition kernels are already in the
   metallib. Serialized attention families measured ~12-16 ms/step; in
   the serial-occupancy regime the recoverable share is real at 1K and
   grows with context (12K decode is 152-155 ms/step — attention scales
   with candidates; split-K matters MORE there).
2. **ds4's sum6-style down projection**: accumulate all 6 experts of a
   token in registers, fold the route-weighted combine AND the residual
   add into the epilogue — deletes the [T,6,4096] intermediate,
   `moe_weighted_sum`, and an add per layer. Bit-exactness needs care
   (our finalize matches MPS's strided sequential order).
3. **Route weight folded into the SwiGLU epilogue** (ds4 T3.1): our
   fused act kernel exists; the multiply is currently in the finalize.
4. **Device-atomic "last threadgroup runs the tail"** (ds4's router
   election, HC comb): kills remaining tiny serial dispatches without
   new CB boundaries. Candidates: softplus+router chain, sampler glue.
5. **Slab/size-class allocator discipline for transients** (ds4 ROCm
   found allocator traffic "a large fraction of decode time"
   independently). Our 4A memos killed the worst churn; a small
   MPS-side slab for the remaining per-step transients
   (~500-1,600 `aten::empty`/step) is the structural version.
6. **Layer-boundary CB splits tuned by position** (ds4 2/32): low value
   today (encode is overlapped; GPU-bound), revisit only if a future
   change makes encode visible again.

For prefill (with #2 of section 7): llama.cpp's two-phase
`mul_mm_id_map0` gather -> per-expert 64x32 simdgroup-matrix tiles with
early-exit on token count, dequant-to-threadgroup in the simdgroup_load
swizzle, bounds-check elision via function constants, and mask
block-skip flash attention. ds4's fused insert-chain kernels bound the
attention side. Do NOT port Metal-4 tensor ops (pre-M5 disabled,
measured slower on M2 Ultra).

## 9. Avenues to the ceiling, ranked

Decode (130.8 ms -> target ~80-90 ms):

> 2026-08-13 STATUS: rows 1-5 done, 6 deprioritized, 7 retired. Wave 12
> ceiling analysis (optimization notebook): the ~80-90 ms tier IS the
> pure-bandwidth floor of the current streams (MoE 10.96 GB + dense
> 8.3 GB + drafter ~3 GB per step); the plateau is **122.6-124.2 ms,
> GPU-saturated** (cb_census busy > wall), with the ~35-40 ms gap being
> serial time-multiplexing of latency-bound phases with weight streams.
> Every dispatch/dedup/fusion/geometry lever measured ~0 this session.
> The one identified ceiling-breaker: second-queue SLC weight prefetch
> into the latency windows (novel, unproven, medium project).

| # | Avenue | class | est. recovery |
| --- | --- | --- | ---: |
| 1 | ~~Re-census + churn kill~~ DONE 2026-08-13: output ring landed bit-exact, NEUTRAL — churn vein CLOSED (the 4A win was the specific marshalling cluster, not allocation count) | bit-exact | measured ~0 |
| 2 | ~~SoA repack IQ2_XXS/Q2_K at load~~ DONE 2026-08-13: Q2_K planes RETAINED (-10% kernel, step 126.8-127.1 ms, bit-exact); IQ2_XXS measured NEGATIVE in three layouts incl. the A100 pairing — Apple LSU eats AoS narrow loads free; do-not-redo | bit-identical | q2_K ~1 ms; iq2 closed |
| 3 | ~~Split-K sparse MLA decode (wire partition path)~~ DONE 2026-08-13: RETAINED default-on (VLLM_QC_MLA_SPLITK, target 768 => P=2 verify), step 127.4 -> 123.3 ms mean, win at all 7 paired offsets; TG=1536 untried | ULP/lottery | measured -4.2 ms @1K |
| 4 | ~~sum6 down + route-weight/residual epilogues~~ DONE 2026-08-13: RETAINED bit-exact (VLLM_QC_MOE_SUM6; oracle 8/8, shas identical, liveness proven) but ~NEUTRAL step — the decode step is not dispatch-latency-bound; ds4's 3-6 ms did not transfer | bit-exact | measured ~0-0.7 ms |
| 5 | ~~Spec block-size re-tune + drafter economics at the new step time~~ DONE 2026-08-13: k=5 CONFIRMED (k=6 -1.3 tok/s mean over 7 offsets; k<5 architecturally forbidden by DSpark). NEW FINDING: step(k) ~ 35.7 + 14.6 ms/verify-row — see the expert-grouped verify MoE candidate in the wave 8 notebook entry | trajectory | closed; k=5 stands |
| 6 | gate_linear out-cast, remaining ULP-class folds — DEPRIORITIZED 2026-08-13: ~86 tiny dispatches; sum6 measured ~90 deleted dispatches as ~neutral, so expected value ~0 | lottery-gated | est. ~0 after wave-7 lesson |
| 7 | ~~Expert-grouped w13 verify (dedup duplicate expert loads)~~ RETIRED 2026-08-13: bit-exact but 123 -> 230 ms — pair-owners serialize a latency-bound kernel and the L2/SLC already dedups co-resident duplicate reads; do-not-redo software weight dedup on Apple GPUs | bit-exact | measured -107 ms REGRESSION |

Prefill (74 -> 277 (ds4) -> 500+ tok/s class): grouped/flat MoE prefill
tile (7.2), insert-chain fusion (the 399 ms/call `attn_wqb_insert_c`
outlier first — suspected mini-MPSGraph pathology), mask block-skip FA,
then the real MPSGraph encode-queue fix to retire `VLLM_QC_LONGCTX_SYNC`.

## 10. Tree state (map for the future cleanup session)

- `csrc/quixicore/metal/kernels/serving/`: dsv4_mhc (split pre/post),
  rms_norm (+w32/strided), indexer (q_rope_quant, kv_insert,
  compress_insert, topk_decode, o_inv_rope, compress_front),
  dsv4_router, moe_finalize, qc_swiglu, probe (diagnostic);
  `matmul/decode_linear` (+bh, tape-era), `quantization/qgemv`
  (mb2-8, moe_mr, moe_mr_swiglu), warp dequant 8-span.
- Host/glue: `qc_metal_serving.mm` (ops, encoder labels, cb census,
  residency_pin, memos + `VLLM_QC_MEMO_*` gates), `metal_worker.py`
  (sweep, pinning, op census), model-side routes in
  `models/deepseek_v4/{metal,metal_indexer,metal_tape,compressor,
  attention}.py` and `common/ops/*`.
- Instruments (env-gated, default off): syncprof, pysampler, phaseprof,
  op census, cb census, qc_probe, verify modes.
- Quarantined-but-retained: step tape (`VLLM_QC_STEP_TAPE=0`), mHC
  forward_mps wiring, compress_front verify mode, q2k_moe_ampere.cuh
  (CUDA side). Each has an env kill switch and a notebook entry.
- Standing gates (post-restart re-baseline, UPDATE 19): 8-tok sha
  `db2846cf721b`, off1-2000 sha `a936de0fa7c7` @ 127.2-127.6 ms,
  counters 1537/2320/464, draw 4.31. (Pre-restart lineage: 5d4697585c6e
  / abaa1c24b187 @ 130.0-131.6 ms — a machine restart rolls the
  lottery.) Boot protocol: verify iogpu sysctl, then tiny primer
  request before anything big.
