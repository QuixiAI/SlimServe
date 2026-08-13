# SlimServe Optimization Status

This is the running performance notebook. Follow `perf/perf.md`: record the
baseline, hypothesis, controlled change, correctness result, throughput result,
decision, and raw artifact locations.

## 2026-08-09 - Retained Merged Sparse-MLA Decode And Exact Baseline

- Status: retained production improvement; tensor-parallel scaling gate still
  failed.
- Baseline: the previous exact native baseline was 90.835 tok/s at TP2 and
  110.218 tok/s at TP4 for concurrency 1, 1,000 input tokens, exactly 2,000
  generated tokens, no speculation, and full decode CUDA graphs.
- Hypothesis: DSV4's main and extra sparse-MLA sources have the same query and
  reduction shape, so launching the FP8-value decode twice per layer repeats
  fixed scheduling and reduction overhead.
- Change: extend the vendored native sparse-MLA kernel to consume the main and
  extra value sources in one launch and accumulate both into the same output.
  This removes one production CUDA-graph node per affected layer without
  changing cache ownership or arithmetic order within either source.
- Kernel result: isolated graph latency fell from about 27.56 us to 21.00 us,
  saving about 6.55 us per layer. Synthetic output matched exactly with
  `max_abs_diff=0`.
- End-to-end result: TP2 measured 92.003 and 92.040 tok/s (median 92.022), and
  TP4 measured 112.572 and 112.573 tok/s (median 112.573). All four runs set
  `exact=true` and returned exactly 2,000 completion tokens. This is +1.31% at
  TP2 and +2.14% at TP4 over the previous exact baseline.
- Scaling: TP4/TP2 improved from 1.213x to 1.223x, but still fails the required
  1.5x gate. The current TP2 result requires at least 138.03 tok/s at TP4;
  112.573 tok/s remains 25.46 tok/s below the minimum.
- Decision: retain the merged MLA launch. It removes measured replicated work,
  but is not sufficient to repair TP ownership by itself.
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-merged-mla-tp2/` and
  `perf/results/2026-08-09/dsv4-a100-merged-mla-tp4/`.

## 2026-08-09 - Rejected mHC Single-Leader Start Rendezvous

- Status: rejected and removed.
- Hypothesis: the urgent mHC kernel currently runs the cross-GPU start epoch
  exchange from all 16 CTAs. A sense-reversing local arrival counter could
  select one CTA per rank for the remote exchange without paying the extra
  cooperative grid barrier used by an earlier rejected leader design.
- Change: add an opt-in local CTA arrival counter, one cross-rank leader, and
  one local release epoch inside the existing urgent graph node. Rank-ordered
  BF16 reduction and all mHC arithmetic were unchanged.
- Correctness: TP4 eager and captured outputs were exact across 32 randomized
  seeds, including fused RMSNorm and Q8_1 output.
- Result: the TP4 fused norm graph regressed from 28.86 us in the same-binary
  control to 35.08 us, a 21.6% loss. Waiting for all 16 local CTAs before any
  peer data work costs more than the distributed remote flag traffic.
- Decision: remove the candidate. A viable synchronization redesign cannot
  introduce an all-CTA local rendezvous before reduction.
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-mhc-leader-start/`.

## 2026-08-09 - Rejected Q2 Producer-Progress Ownership

- Status: rejected and disabled by default; retained only as an explicitly
  gated diagnostic under `VLLM_DSV4_Q2_MHC_PROGRESS=1`.
- Baseline/hypothesis: the Q2_K producer and urgent mHC transition are adjacent
  on every layer. Publishing completed output tiles to a peer-reading consumer
  was intended to overlap the full-rank transition with the Q2_K producer tail.
- Implemented candidates covered 64-tile publication, whole-grid persistent
  barriers, barrier-free consumer phases, split streams, producer-first stream
  ordering, 16 logical 256-row publications, and a high-priority consumer
  stream. Corrected producer-first variants were exact.
- Root result: 64 tile handshakes added about 18 us and serial 64-row mHC
  chunks added another roughly 14 us. Reducing publication to 16 logical
  chunks removed most of that cost, but remained slower: TP2 measured 65.064
  us versus 61.805 us reference, and TP4 measured 52.950 us versus 49.151 us
  reference. The high-priority variant was also slower at 53.274 us versus
  49.148 us.
- Decision: A100 does not overlap this short Q2 producer with a polling mHC
  consumer cheaply enough. Do not enable the path in a serving profile. A
  future retry needs a materially different warp-specialized kernel with no
  inter-CTA polling protocol, not another publication-granularity sweep.
- Raw artifacts are under
  `perf/results/2026-08-09/dsv4-a100-q2-progress-v*/`.

## 2026-08-09 - Rejected Head-Sharded Indexer Ownership

- Status: rejected and disabled by default; diagnostic opt-in is
  `VLLM_DSV4_HEAD_SHARDED_INDEXER=1`.
- Hypothesis: shard 64 indexer heads across TP ranks, produce partial token
  scores, and have the persistent top-k node sum peer partials before selection.
- Correctness: peer partial-score reduction and global top-k were exact in the
  synthetic multi-rank harness and the real profiles returned exact token
  counts.
- Result: isolated peer top-k cost about 61 us at TP2 and 69 us at TP4, while
  H64, H32, and H16 paged-logit producer schedules all remained near 18 us at
  the C1 decode shape. The reduced head count therefore saved almost no
  producer latency and added a full peer reduction. Exact profile throughput
  fell to 89.865 tok/s at TP2 and 109.617 tok/s at TP4 before the merged-MLA
  change.
- Decision: partial-head ownership is fundamentally mismatched to this
  batch-one producer geometry. The next indexer ownership design is token
  sharding with direct peer merge of only rank-local top-k candidates; it must
  not restore the previous generic all-gather, pack, unpack, and second-top-k
  sequence.
- Raw artifacts:
  `perf/results/2026-08-08/dsv4-a100-head-sharded-indexer-tp2/` and
  `perf/results/2026-08-08/dsv4-a100-head-sharded-indexer-tp4/`.

## 2026-08-08 - DSV4 0731 A100 Exact Baseline And Scaling Gate

- Status: active baseline; scaling gate failed.
- Scope: the all-layer IQ2_XXS gate/up + Q2_K down model, concurrency 1,
  1,000 input tokens, exactly 2,000 generated tokens, no speculation, full
  decode CUDA graphs, and native Ampere kernels. TP2 and TP4 use the same model
  and workload.
- Current exact results are 90.808 and 90.862 tok/s at TP2, and 110.232 and
  110.205 tok/s at TP4. The medians are 90.835 and 110.218 tok/s.
- TP4/TP2 scaling is only 1.213x. The repository's minimum 1.5x gate requires
  at least 136.25 tok/s at TP4 for this TP2 result. The current TP4 path is
  26.03 tok/s, or 19.1%, below that minimum target.
- Correctness: every retained run set `exact=true`, returned exactly 2,000
  completion tokens, and completed the fixed prompt without a server error.
- Warm decode trace: TP2 has 1,513 graph kernels and settles near
  10.75-10.90 ms/replay; TP4 has 1,470 graph kernels and settles near
  9.01-10.65 ms/replay as the traced context changes. TP4 reduces routed
  weight work, but replicated projections, indexer work, mHC work, and two
  synchronization boundaries per layer dominate the remainder.
- The model is already using tensor-parallel routed experts, not expert
  parallelism: every selected expert is available on every rank and its
  intermediate dimension is sharded. Random expert-owner imbalance is
  therefore not the cause of the failed scaling.
- Corrected trace diagnosis: raw cross-process GPU timestamps contain stable
  per-device clock offsets. Relative to rank 0, barrier-end-derived offsets
  were 0, -10.268, -30.089, and -15.771 us; interpreting unnormalized starts
  overstated rank arrival skew as 30-35 us. After normalization, median
  producer/urgent arrival skew is about 16.4 us at the aligned-Q8 projection
  boundary and 10.2 us at the routed Q2_K boundary; median barrier-end skew is
  about 2 us. Each producer ends within 0.3-0.9 us of its local urgent launch.
- Urgent-kernel duration remains materially worse at TP4 even without
  cross-clock comparison: all-rank mean duration is 17.943 us at TP2 and
  26.499 us at TP4. Deferred mHC projection is still replicated and consumes
  roughly 1.5-2.0 ms of aggregate kernel time per token, though it overlaps
  foreground work on a low-priority stream.
- Decision: do not call 110 tok/s acceptable. Continue with design changes
  that remove replicated graph work or collective latency; sub-percent local
  fusions do not address the failed scaling gate.
- Raw artifacts:
  `perf/results/2026-08-08/dsv4-a100-indexer-group4-tp2/`,
  `perf/results/2026-08-08/dsv4-a100-mhc-low-priority-tp4/`,
  `perf/results/2026-08-08/dsv4-a100-active-indexer-trace-tp2/`, and
  `perf/results/2026-08-08/dsv4-a100-active-indexer-trace-current-tp4/`.

## 2026-08-08 - DSV4 TP4 Scaling Root-Cause Audit

- Status: root cause confirmed; no profile, placement, or model mismatch.
- Method: classify every graph-id-26 kernel from all two TP2 ranks and all four
  TP4 ranks across seven warm replays. The checked-in analyzer is
  `benchmarks/analyze_dsv4_tp_scaling.py`; its machine-readable result is
  `perf/results/2026-08-08/dsv4-a100-tp-scaling-root-cause/trace-classification.json`.
- Both SlimServe dry runs resolve the same all-layer IQ2_XXS/Q2_K GGUF, no
  speculation, FP8 KV, and full decode graphs. `nvidia-smi topo -m` reports
  NV12 between every GPU pair. Tensor-parallel placement is not the cause.
- Exact throughput is 11.009 ms/token at TP2 and 9.073 ms/token at TP4. The
  1.5x gate is 7.339 ms/token, so TP4 must recover 1.734 ms/token. Fitting
  `T(tp) = R + W/tp` to the two exact measurements gives `R = 7.137 ms` and
  `W = 7.744 ms`: 78.7% of current TP4 latency behaves as non-scaling work.
  With the current ownership boundaries, even infinite TP asymptotes to only
  1.543x TP2. Kernel retuning inside the existing design cannot provide a
  defensible 1.5x TP4 result.
- Mean aggregate kernel duration per replay, averaged across ranks:

| ownership/category | TP2 us | TP4 us | TP2/TP4 |
| --- | ---: | ---: | ---: |
| TP-sharded quantized linear/MoE | 5432.822 | 3706.243 | 1.466x |
| TP-sharded attention (intended) | 1640.020 | 1525.079 | 1.075x |
| mHC urgent collective/transition | 1525.121 | 2252.374 | 0.677x |
| mHC deferred replicated | 1518.867 | 1622.130 | 0.936x |
| replicated indexer/router/projection | 2721.468 | 2862.252 | 0.951x |
| replicated fixed Q8 projection | 465.462 | 485.445 | 0.959x |
| other graph work | 1660.968 | 1753.890 | 0.947x |

- Only 47.3% of TP2 aggregate kernel work is even intended to shard in this
  classification. Quantized linear/MoE work saves 1.727 ms but reaches only
  1.466x because batch-one kernels and graph launches do not halve with the
  weights. Attention halves local heads from 32 to 16 but improves only 1.075x.
  Everything else is flat or slower, and mHC gives back 0.831 ms of the local
  kernel savings before stream-overlap effects.
- The graph shape corroborates the ownership problem: TP2 has 1513 nodes per
  replay and TP4 still has 1470. The 43-node reduction is the TP4-only shared
  down-input quantization fusion, not broad TP scaling. There are still 85
  global mHC transitions per token.
- The model source explicitly replicates the fixed attention input projection
  (`fused_wqa_wkv`, `disable_tp=True`) and all 64 Lightning-indexer heads
  (`ReplicatedLinear` for `wq_b` and `weights_proj`). The router also scores all
  256 experts on every rank. By contrast, ordinary attention heads correctly
  shrink from 32 at TP2 to 16 at TP4. The indexer is therefore a concrete
  ownership defect, not merely a slow kernel.
- The native DSV4 MoE boundary is fused at the PyTorch-op level but remains two
  CUDA phases per layer: IQ2_XXS gate/up + SwiGLU + Q8_1, then repacked Q2_K
  down + weighted reduce. The measured one-node cooperative TP4 candidate was
  neutral end to end because it left the following global mHC boundary and all
  replicated work intact. It is not the missing 1.734 ms solution by itself.
- Corrected collective conclusion: each materialized producer ends only
  0.3-0.9 us before its local urgent mHC launch. Removing only that launch gap
  can save at most about 0.08 ms over 85 boundaries, far below the target.
  Producer/collective fusion remains useful only if it replaces the full-rank
  arrival barrier with deadlock-safe tile-progress publication so early peer
  tiles overlap the producer tail. Appending the existing urgent transition to
  a cooperative producer is not a sufficient design.
- Required structural work, in priority order:
  1. Implement head-sharded indexer query/weight projections and H16 local
     logits at TP4, with the existing persistent top-k node performing the
     peer partial-score reduction. Do not repeat the rejected token-shard +
     generic all-gather design.
  2. Implement tile-progress producer/collective transitions for both Q2_K
     down and aligned-Q8 output projection. Preserve rank-ordered BF16 math and
     pipeline peer readiness; node-count reduction alone is not the objective.
  3. Merge the two DSV4 sparse-MLA source launches into one source-selecting
     persistent kernel. The current wrapper launches main and extra sources
     separately in every layer even though they share query and reduction
     shape.
  4. Shard deferred mHC coefficient ownership only by publishing through the
     next existing transition handshake; all variants with an extra fence or
     barrier have already lost.
  5. Partition router rows and merge only rank-local top-k candidates inside
     an existing routing node after the larger indexer/collective defects are
     fixed.
- Decision: the 110.218 tok/s result is caused by fundamentally insufficient
  parallel ownership plus worsening per-layer synchronization. Do not spend
  the next iteration on another standalone GEMV geometry sweep or launch-only
  fusion.

## 2026-08-08 - A100 mHC Schedule And Deferred Sharding

- Status: async schedule retained; sharded candidates rejected and removed.
- Scope: DSV4 0731 IQ2_XXS, `dsv4-4`, concurrency 1, 1,000 input and 2,000
  output tokens, no speculation, locked 1,410 MHz A100 clocks.
- Baseline: the profile's low-priority asynchronous urgent/deferred split.
  Hypothesis: moving deferred projection onto the main stream, merging it into
  the urgent kernel, or distributing its 20 rows across ranks might reduce the
  TP4 arrival skew and replicated work.
- Schedule results: async measured 110.232 and 110.181 tok/s (median 110.206);
  sequential measured 101.655 and 101.664 (median 101.660); monolithic measured
  102.956 and 102.988 (median 102.972). All runs returned exact token counts.
  Async overlap is essential and remains the production schedule.
- Sharded candidate A computed local deferred rows and gathered the next urgent
  state through the existing handshake. It was exact in the four-rank graph
  harness for 16 seeds, but TP4 fell to 99.357 tok/s. Candidate B published
  encoded state to peers and added one barrier inside the deferred kernel; it
  was also graph-exact but reached only 109.656 tok/s versus the 110.218
  baseline. Both designs were removed.
- An owner-deferred candidate made rank 3 compute all 20 deferred projection
  rows and Sinkhorn once, then remote-write the final contiguous state to the
  peers for publication at the next existing urgent transition. It was exact
  across four ranks and 16 seeds, but the final publication fence raised the
  isolated graph from about 32.11 to 34.74 us and the real TP4 profile reached
  only 105.596 tok/s. It was rejected and removed.
- Decision: do not revisit deferred sharding with an extra synchronization
  point. A viable design must preserve async overlap and piggyback publication
  on an existing transition without extending the urgent critical path.
- Raw artifacts:
  `perf/results/2026-08-08/dsv4-a100-mhc-schedule-ab-tp4/` and
  `perf/results/2026-08-08/dsv4-a100-mhc-sharded-deferred/`, and
  `perf/results/2026-08-08/dsv4-a100-mhc-owner-deferred/`.

## 2026-08-08 - A100 mHC Arrival And Deferred Overlap Geometry

- Status: all candidates rejected and removed; retained production remains 16
  urgent CTAs, 16 logical partials, and a one-node 32-CTA/two-partition
  cooperative deferred kernel on the low-priority stream.
- Baseline/diagnosis: the retained graph-id-26 captures contain 10,591 kernels
  across seven TP2 replays and 10,290 across seven TP4 replays. Aggregate
  kernel duration is 14,907.402 us per TP2 replay and 14,051.698 us per TP4
  replay. TP4 saves about 1.7 ms in routed IQ2_XXS, Q2_K, grouped-Q8, and
  aligned-Q8 work, but urgent mHC grows from 1,567.100 to 2,361.672 us and
  deferred mHC grows from 1,508.681 to 1,607.889 us. Those mHC costs erase
  more than half of the measured sharding benefit.
- A rank-local preload candidate moved residual, coefficient, and local-input
  loads before the start barrier while preserving rank-ordered accumulation.
  It was exact on four ranks and 16 seeds, but its norm graph rose from 28.973
  to 29.081 us and exact TP4 fell to 109.831-109.902 tok/s versus a restored
  same-session control of 110.255-110.284. It was removed.
- Deferred output partition sweeps preserved every dot-product reduction tree.
  Four partitions/64 CTAs improved the isolated norm graph to 28.312 us but
  fell to 109.667-109.693 tok/s because the low-priority kernel interfered
  more with foreground work. One partition/16 CTAs slowed the graph to 29.999
  us and measured 110.203-110.242 tok/s, neutral/slightly below control. The
  retained two partitions/32 CTAs remain the measured balance.
- A DS4-style staged deferred design replaced the cooperative grid barrier
  with a normal 32-CTA partial kernel followed by a one-CTA Sinkhorn/finalize
  kernel on the same stream. It was exact and improved isolated norm timing
  slightly to 28.818 us, but the extra CUDA-graph node reduced exact TP4 to
  109.536-109.596 tok/s. It was removed.
- A corrected reduced-handshake urgent design separated eight physical CTAs
  from 16 logical reduction splits. Unlike the older invalid eight-split
  experiment, it computed each reduced dimension independently and reproduced
  all 16 partial slots, RMSNorm blocks, and Q8 blocks exactly across 32 seeds.
  Serializing two logical chunks per CTA raised the norm graph to 31.597 us
  and exact TP4 fell decisively to 105.638 tok/s. It was removed.
- Decision: local work motion, more/fewer deferred CTAs, splitting the deferred
  node, and reducing urgent handshake CTAs do not fix scaling. Isolated graph
  improvements are not sufficient evidence because low-priority occupancy and
  graph-node count dominate end-to-end behavior. The next mHC design should
  remove a foreground launch by fusing the Q2_K or aligned-Q8 producer with
  the existing collective/transition boundary; it must preserve the async
  deferred overlap and add no new cross-rank synchronization point.
- Raw artifacts:
  `perf/results/2026-08-08/dsv4-a100-mhc-local-preload/`,
  `perf/results/2026-08-08/dsv4-a100-mhc-deferred-geometry/`,
  `perf/results/2026-08-08/dsv4-a100-mhc-deferred-staged/`, and
  `perf/results/2026-08-08/dsv4-a100-mhc-physical8-logical16/`.

## 2026-08-08 - A100 Native MoE Decode Geometry And Fusion

- Status: all candidates rejected and removed; retained geometry remains eight
  warps per Q2_K CTA and two output rows per warp.
- Scope: native combined-layout IQ2_XXS gate/up, SwiGLU, Q8_1 handoff,
  byte-neutral repacked Q2_K down, and weighted final sum at TP4-local `I=512`
  and TP2-local `I=1024` decode shapes.
- Baseline/hypothesis: the Q2 down kernel duplicates Q8 activation staging per
  CTA. Wider CTAs or a cooperative W1/down fusion could reduce staging and one
  launch without changing quantization or materializing dequantized weights.
- Serialized row-geometry results: at `I=512`, rows-per-warp 2/4/8 measured
  34.816/37.888/43.008 us for the full W1+down operation. At `I=1024`, the
  controlled retained geometry was about 49 us while 4 and 8 rows measured
  51.200 and 56.320 us. Output hashes were identical across geometries.
- Staging the 4.5 KiB W1 Q8 input in shared memory was neutral at `I=512`
  (34.816 versus 34.816 us) and slower at `I=1024` (47.616 versus 47.104 us).
  It was removed.
- Wider Q2 CTAs were exact but slower under 20-launch CUDA-event samples:
  TP4-local 16-warps/2-rows measured 24.090 us versus 23.014 for the retained
  8-warps/2-rows path; TP2-local 32-warps/1-row measured 37.325 versus 35.661
  us. The roughly 4.7% losses show that occupancy is worth more than reducing
  activation staging copies in this form.
- A true cooperative TP4 decode candidate fused the 96-CTA IQ2_XXS/SwiGLU
  phase, a grid handoff, and the Q2_K weighted down sum into one native kernel.
  Its final 64-CTA/two-row down phase was bit-exact and improved repeated
  microtiming from 22.989-23.040 to 22.323-22.349 us (about 3%). Under the real
  CUDA-graph profile, however, it measured only 110.362-110.382 tok/s versus
  same-build control at 110.409-110.454 tok/s. It was rejected and removed;
  isolated stream timing did not predict graph scheduling cost.
- Correctness: every micro candidate matched the retained BF16 output SHA-256;
  every serving run returned exact prompt/completion counts. Completion hashes
  vary between repeated temperature-zero runs even on the same binary, so they
  are recorded but are not treated as deterministic numerical parity.
- Raw artifacts:
  `perf/results/2026-08-08/dsv4-a100-moe-decode-geometry/`,
  `perf/results/2026-08-08/dsv4-a100-moe-stage-input/`,
  `perf/results/2026-08-08/dsv4-a100-q2-cta-geometry/`, and
  `perf/results/2026-08-08/dsv4-a100-cooperative-moe/`.

## 2026-08-08 - A100 Aligned Q8 Decode Geometry

- Status: all serving candidates rejected and removed; benchmark methodology
  retained.
- Trace scope: the TP4 decode graph contains four aligned-Q8 projections per
  layer: `1536x4096/r1`, `4096x2048/r1`, `4096x8192/r2`, and
  `8192x1024/r4`. The aligned Q8 kernels account for about 1.83 ms of aggregate
  kernel time per token and the first urgent collective follows this work.
- Measurement correction: the old microbenchmark included activation
  quantization and repeatedly hit one resident weight. It now reports
  prequantized GEMV separately and rotates eight byte-identical aligned
  weights, exceeding A100 L2 even for the smallest active tensor.
- Existing-kernel selector candidate changed `4096x2048` from r1 to r2 and
  `4096x8192` from r2 to r1. Every geometry was bit-exact in the kernel harness,
  but the real TP4 profile fell to 109.516 and 109.605 tok/s versus same-session
  production control at 110.282 and 110.358. The dispatch changes were removed.
- A SlimServe-owned DS4/QuixiCore-inspired candidate used aligned 32-byte
  weight loads with one or two rows per warp and no shared reduction. Row-pair
  variants were not faster. The one-row variant improved only the deep-K
  microkernel, from about 29.92 to 27.13 us under rotated weights. End to end it
  measured 110.423 and 110.437 tok/s versus same-binary control at 110.324 and
  110.390: about +0.07%, below the retention threshold and irrelevant to the
  failed TP scaling gate. All candidate kernel and dispatch code was removed.
- Raw artifacts:
  `perf/results/2026-08-08/dsv4-a100-q8-decode/`,
  `perf/results/2026-08-08/dsv4-a100-q8-selector-tp4/`, and
  `perf/results/2026-08-08/dsv4-a100-q8-single-warp-tp4/`.

## 2026-08-08 - Locked A100 Clock Control

- Status: diagnostic; no production change.
- Change: locked all eight A100 graphics clocks at 1,410 MHz to remove clock
  drift from controlled comparisons.
- Results: exact TP2 measured 90.664 tok/s median and exact TP4 measured
  110.288 tok/s median. This did not explain the failed scaling ratio or
  materially improve the unlocked exact baseline.
- Decision: use locked clocks for kernel A/B work, but do not attribute the
  TP2/TP4 gap to clock variance. Reset clocks after the experiment session.
- Raw artifacts: `perf/results/2026-08-08/dsv4-a100-locked-clocks/`.

## 2026-08-08 - Retained A100 Native Path

- Status: retained production path.
- Native routed MoE reads the combined vLLM
  `[expert, gate | up, packed]` IQ2_XXS layout, performs paired gate/up and
  SwiGLU, emits Q8_1, runs repacked Q2_K down, and performs weighted reduction.
  It does not dequantize model weights or use standalone Q2_K GEMV as the
  production boundary.
- Load-time/first-use layout work retained after correctness and end-to-end
  validation: byte-neutral aligned Q8_0 SoA, paired aligned IQ2_XXS SoA, and
  repacked Q2_K down. The runtime does not retain an expanded dequantized expert
  stack.
- Other retained components: native Q8 shared experts and output projection,
  native BF16 projection GEMV, native router and hash router, active-channel
  MLA partitioning, graph-capturable persistent paged indexer logits, live
  prefix sizing, H64 grouped-token indexer scheduling, native mHC, fused custom
  all-reduce+mHC, FP16 mHC projection weights, and low-priority deferred mHC.
- Measured progression on the evolving same-day tree: the first corrected TP4
  exact run was 105.364 tok/s; native projection reached 108.522-108.549;
  native hash routing measured 108.832-108.854 versus 108.537-108.598 control;
  H64/live-prefix work reached about 110.09-110.14; low-priority deferred mHC
  reached 110.205-110.232 tok/s.
- DSpark correctness fix retained: the final deferred target tuple is consumed
  before returning from the target model. TurboQuant remains the configured
  draft KV backend in the SlimServe profile. Both still require a fresh
  acceptance-qualified TP2/TP4 benchmark after the no-spec scaling work.
- Raw artifacts are under `perf/results/2026-08-08/dsv4-a100-{projection,hash-router,indexer-group4,mhc-low-priority}-*/`.

## 2026-08-08 - Rejected A100 Experiments

- Indexer head sharding: exact but TP4 fell to 105.250-105.368 tok/s from
  108.848-108.864 control. The generic cross-rank merge cost exceeded the
  saved head work. Rejected; a future distributed indexer must fuse partial
  reduction and top-k rather than repeat this design.
- Short top-512 radix path: no-spec TP4 measured 110.141-110.213 tok/s versus
  110.451-110.524 control. Rejected and removed.
- Fused aligned-Q8 projection plus RoPE/FP8 preparation: bit-exact and faster
  when streamed, but CUDA-graph latency regressed from 14.69 us to 32.27 us for
  the head-owned topology and to 25.73 us for the four-segment topology.
  Rejected and removed.
- Raw-BF16 Q preparation inside paged indexer logits: exact top-512 across the
  context/TP sweep. TP4 reached 110.363-110.377 tok/s, only about +0.13% over
  the retained baseline, while same-build TP2 regressed from 90.711 to 90.367
  tok/s (-0.38%). Rejected and removed.
- Fused non-hash router plus top-k, projection bundling, mHC 8-split, paired
  CTA, alternate norm mapping, and leader-grid handshake variants either lost
  under CUDA graphs or failed the end-to-end threshold. None remain on the
  production path.
- Rank-sharded mHC projection with an additional scalar-exchange barrier was
  exact on all four ranks and 16 randomized seeds, but the production norm
  graph path rose from 35.36 to 45.15 us (+27.7%); the local-add norm path rose
  from 35.10 to 44.97 us (+28.1%). Rejected and removed; any retry must
  piggyback scalar visibility on an existing transition handshake rather than
  add a barrier.
- Raw artifacts:
  `perf/results/2026-08-08/dsv4-a100-indexer-head-shard/`,
  `perf/results/2026-08-08/dsv4-a100-short-topk-nospec-{tp4,control-tp4}/`,
  `perf/results/2026-08-08/dsv4-a100-fused-indexer-{logits,control}-tp*/`,
  `perf/results/2026-08-08/dsv4-a100-mhc-{grid-barrier,paired-cta,norm-map}-tp4/`,
  and `perf/results/2026-08-08/dsv4-a100-mhc-rank-shard-tp4/`.

## 2026-08-07 - DSV4 0731 A100 Native MoE Path

- Status: in progress
- Scope: DeepSeek V4 Flash 0731, `dsv4-2` and `dsv4-4`, A100, IQ2_XXS gate/up
  plus Q2_K down routed MoE.
- Baseline: ROCm DeepSeek V4 Flash 0731 optimized path is the cross-platform
  reference. Existing A100 numbers must be re-measured through SlimServe
  profiles after the model download resumes.
- Hypothesis: A100 inference is leaving performance on the table because the
  DSV4 routed MoE path is not using a model-specific fused native kernel path
  equivalent in spirit to the optimized ROCm/DS4 flow.
- Target change: support vLLM's combined `[expert, gate|up, packed]` GGUF
  layout or split/aligned artifacts, then run fused native
  `IQ2_XXS gate/up -> SwiGLU -> Q2_K down -> weighted reduce` without dequant
  and without treating standalone Q2_K GEMV as the final answer.
- Correctness: use exact-token serving validation and focused kernel parity
  tests for any new fused op.
- Implemented stage 1:
  - Added `csrc/quixicore/quant/dsv4_moe_ampere.cuh` on the real stable-libtorch
    serving path.
  - The Ampere kernel reads vLLM's combined W1 layout, stages each activation
    tile once, decodes paired gate/up IQ2_XXS weights at tile load, applies
    SwiGLU in registers, and emits Q8_1 directly for Q2_K down.
  - Removed the fp32 gate/up and fp32 mid tensors plus the standalone SwiGLU
    and mid-quantize launches. No model weights are dequantized, expanded, or
    persistently duplicated.
  - W1 uses 4 routed rows below 256 total routed rows and 8 at or above 256;
    W2 remains 4-wide and receives its own expanded expert-id view.
- Stage 1 correctness: valid synthetic IQ2_XXS and Q2_K blocks at TP2-local
  DSV4 dimensions (`H=4096`, local `I=1024`, top-8) produced finite output.
  At 128 routed rows, mean/max absolute error versus the generic path was
  `2.63e-6 / 5.35e-6`; at 512 routed rows it was
  `2.77e-6 / 1.63e-5`.
- Stage 1 kernel results on one A100, median CUDA-event timing:

| Tokens | Routed rows | W1 width | Fused us | Generic us | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 4 | 192.512 | 199.680 | 1.037x |
| 4 | 32 | 4 | 477.184 | 553.984 | 1.161x |
| 16 | 128 | 4 | 573.952 | 611.328 | 1.065x |
| 64 | 512 | 8 | 1495.040 | 1994.752 | 1.334x |

- Results: superseded by the exact TP2/TP4 baseline above. SlimServe downloaded
  and loaded the correct all-layer IQ2_XXS model; missing-model handling was not
  a blocker.
- Decision: retain stage 1. It is a correctness-qualified win at verification
  width and establishes the production operation boundary. It is not the final
  design: Q2_K down still uses the generic dp4a tile and materializes routed
  down outputs. The next controlled work is the short-K Q2_K design in
  `csrc/quixicore/dsv4_ampere_design.md`, followed by byte-neutral Ampere MMA
  layouts.
- Existing A100 diagnostics:
  - `/home/ubuntu/logs/dsv4-a100-tp2-native.log`: TP2 loaded the target at
    51.75 GiB/GPU in about 32.9 s, allocated 18.23 GiB KV cache, reported
    2,203,531 KV tokens and 2.10x maximum concurrency at 1,048,576-token
    requests, then failed warmup with `main_cache must be uint8`.
  - `/home/ubuntu/logs/dsv4-a100-tp2-dspark-tq.log`: TP2 with DSpark and
    TurboQuant loaded target + draft, reached health, and reported 11.89 GiB KV
    cache. No exact benchmark was run; only an idle/short health probe appears.
  - `/home/ubuntu/logs/dsv4-a100-tp2-tps.log`: TP2 with DSpark/TurboQuant
    reached health and served probes. vLLM interval logs showed generation
    throughput around 32.4, 31.5, and 28.8 tok/s during a 9-request probe, then
    around 30.4, 30.4, and 29.6 tok/s during an 8-request probe. DSpark accepted
    zero draft tokens in these probes (`Mean acceptance length: 1.00`,
    `Accepted throughput: 0.00 tokens/s`), so these are diagnostic only.
  - `/home/ubuntu/logs/dsv4-a100-tp4-dspark-tq.log`: TP4 with DSpark and
    TurboQuant loaded target + draft and reached health. No exact benchmark was
    run.
  - `/home/ubuntu/logs/dsv4-a100-tp4-tps.log`: TP4 with DSpark/TurboQuant
    reached health. It reported 28.88 GiB consumed per GPU, 0.80 GiB peak
    activation, 1.89 GiB CUDA graph memory, and 32.13 GiB KV cache. Interval
    logs showed single-request generation around 5.6 and 7.2 tok/s, then
    8-request generation around 34.4, 36.0, and 29.1 tok/s. DSpark again
    accepted zero draft tokens in the recorded probe.
- Current measured conclusion: superseded by the 2026-08-08 exact 90.835 TP2
  and 110.218 TP4 baseline. The older 30-36 tok/s interval probes mixed
  warmup/concurrency effects and are diagnostic history, not the headline.
- Raw artifacts: `/home/ubuntu/logs/dsv4-a100-*.log`, `/tmp/dsv4-*.log`,
  `benchmarks/kernels/benchmark_dsv4_moe_a100.py`, and exact TP2 logs under
  `perf/results/2026-08-07/dsv4-a100-fused-v1-tp2/` (kernel JSONL at
  `kernel/stage1.jsonl`).

## 2026-07-29 - ROCm Long-Context Concurrency Baseline

- Status: retained baseline
- Scope: DeepSeek V4 Flash 0731, TP2 on 2x MI300X, 100k input tokens, 2k output
  tokens, concurrency sweep 1-64, prefix caching enabled with a 99,984-token
  common prefix.
- Baseline/change: split ROCm GGUF MoE dispatch thresholds for W1 and W2 based
  on measured crossovers.
- Correctness: every request produced exactly 2,000 tokens from exactly 100,000
  input token IDs and reported exactly 99,984 cached prompt tokens; 32-request
  factual quality gate passed 32/32.
- Final exact sweep:

| Concurrency | Wall s | Aggregate tok/s | Decode-window tok/s | Per-request tok/s | TTFT p50/p95 s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.71 | 41.92 | 42.13 | 41.92 | 0.235 / 0.235 |
| 2 | 57.31 | 69.79 | 70.08 | 34.90 | 0.341 / 0.440 |
| 4 | 77.55 | 103.16 | 103.48 | 25.79 | 0.502 / 0.506 |
| 8 | 120.45 | 132.84 | 133.09 | 16.61 | 0.731 / 0.742 |
| 16 | 175.87 | 181.95 | 182.18 | 11.37 | 0.888 / 1.085 |
| 32 | 350.50 | 182.59 | 182.71 | 5.71 | 1.441 / 1.803 |
| 64 | 533.06 | 240.12 | 240.24 | 3.75 | 2.450 / 2.930 |

- Decision: retained as the first ROCm long-context baseline.
- Raw artifacts: `/tmp/longctx_baseline_128.{json,log}`,
  `/tmp/longctx_moesplit_128.{json,log}`,
  `/tmp/longctx_final_shared_2000.{json,log}`.

## 2026-07-29 - ROCm Q2_K Routed MoE Tile Pass

- Status: retained
- Scope: routed Q2_K expert kernels for DSV4/GLM-style TP2 serving shapes on
  MI300X.
- Hypothesis: routed Q2_K expert kernels dominated steady decode and the 4x128
  tile carried too much register/LDS pressure.
- Change: switch ROCm Q2_K MMQ tile from 4 routed rows x 128 output rows to
  4 x 64, then adjust dispatch thresholds.
- Kernel results:

| TP2 MoE shape | 4x128 us | 4x64 us | Speedup |
| --- | ---: | ---: | ---: |
| w13, 64 model tokens | 1201.7 | 851.7 | 1.41x |
| w2, 512 routed rows | 603.8 | 431.2 | 1.40x |
| w13, 512 verification tokens | 4995.4 | 3490.1 | 1.43x |
| w2, 4096 routed verification rows | 2517.9 | 1818.7 | 1.38x |

- End-to-end: 64-request exact 100k/2k improved from 240.12 to 325.01
  aggregate tok/s (+35.3%); wall time 533.06 s to 393.84 s.
- Correctness: matrix/vector parity passed; largest observed BF16 absolute
  difference was 0.25.
- Decision: retained.
- Raw artifacts: `/tmp/longctx_q2_y64_2000.json`,
  `/tmp/longctx_q2_y64_128.json`.

## 2026-07-29 - ROCm Q2/Q8 Final Kernel Pass

- Status: retained
- Scope: ROCm routed Q2_K MoE plus Q8_0 attention/dense projections.
- Change: Q2_K routed-expert tile moved from 4x64 to 4x32 with two-span Q8
  activation staging; Q8_0 matrix kernel added shape-adaptive 64/128 output-row
  tiles.
- Q2_K kernel results:

| TP2 MoE shape | Previous 4x64 us | Final us | Speedup |
| --- | ---: | ---: | ---: |
| w13, 64 model tokens | 884.506 | 797.335 | 1.109x |
| w2, 512 routed rows | 449.354 | 422.494 | 1.064x |
| w13, 512 verification tokens | 3037.452 | 2778.504 | 1.093x |
| w2, 4096 routed rows | 1592.988 | 1479.758 | 1.077x |

- Q8_0 projection results at batch 64:

| Projection | Previous 128-row us | 64-row us | Speedup |
| --- | ---: | ---: | ---: |
| o_proj, 6144 x 8192 | 410.936 | 314.051 | 1.308x |
| dense gate/up, 12288 x 6144 | 308.843 | 234.652 | 1.316x |
| q_b_proj, 6144 x 2048 | 113.630 | 91.397 | 1.243x |

- End-to-end: 64-request exact 100k/2k improved from 325.007 to 395.902
  aggregate tok/s (+21.8%); wall time 393.838 s to 323.313 s. Relative to the
  original 240.12 tok/s baseline, combined kernel work is +64.9%.
- Correctness: valid repeated Q2_K test blocks had `max_abs_diff=0`; Q8_0
  matrix/vector parity also had `max_abs_diff=0`.
- Decision: retained. The remaining 1,000 tok/s target gap was 2.53x.
- Raw artifact: `/tmp/longctx_q2_y32_q8_2000.json`.

## 2026-07-29 - ROCm DSpark And TurboQuant Evaluation

- Status: retained config guidance plus follow-up required
- Scope: DeepSeek V4 Flash 0731 TP2, DSpark draft, TurboQuant draft KV.
- DSpark acceptance sanity: three natural prompts x 128 output tokens produced
  19.49% draft-token acceptance, mean accept length 2.364, and 68.63 aggregate
  output tok/s. This was a functionality/acceptance probe, not a clean TPS
  headline.
- Synthetic repeated-token long-context DSpark failure was diagnosed as a bad
  workload artifact, not a serving-stack bug. Natural prompts kept healthy
  acceptance from 500 to 12k tokens:

| Case | Prompt tokens | Draft acceptance | Mean accept len |
| --- | ---: | ---: | ---: |
| natural_500 | 527 | 31.8% | 3.23 |
| natural_1500 | 1527 | 27.9% | 2.95 |
| natural_2500 | 2527 | 32.6% | 3.28 |
| natural_6000 | 6027 | 32.1% | 3.24 |
| natural_12000 | 12027 | 33.5% | 3.34 |
| shuffled_6000 | 6027 | 15.0% | 2.05 |

- Valid long-context batch-64 chat workload:

| Mode | Aggregate tok/s | vs baseline | Draft acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: |
| no spec | 254.25 | -- | -- | -- |
| dspark spec-3 | 285.32 | +12.2% | 56.7% | 2.70 |
| dspark spec-4 | 252.73 | -0.6% | 47.0% | 2.88 |
| dspark spec-7 | 194.47 | -23.5% | 29.6% | 3.07 |

- Valid long-context batch-64 raw completion workload: no spec 396.56 tok/s,
  spec-3 412.85 tok/s (+4.1%), spec-4 401.59 tok/s, spec-7 311.40 tok/s.
- TurboQuant draft KV smoke: `turboquant_k8v4` was coherent and healthy at
  500 prompt tokens (25.2%, mean 2.76) and 4000 prompt tokens (31.0%, mean
  3.17). FP8-KV reference was 31.8% / 32.1%.
- Decision: use DSpark spec-3 for batch-64 long-context serving; keep spec-7
  only for low-batch latency. TurboQuant draft KV works but still needs batch-64
  100k re-benchmarking.
- Raw artifacts: `/tmp/spec_accept_dspark7_new.json`,
  `/tmp/dspark_ctx_sweep.json`, `/tmp/dspark_cache_chunk.json`,
  `/tmp/dspark_longctx_ab_*.json`, `/tmp/dspark_longctx_chat_*.json`,
  `/tmp/dspark_tq_smoke.json`.

## 2026-07-29 - ROCm Prefix Cache And 1M Context

- Status: retained baseline/capacity finding
- Scope: DeepSeek V4 Flash 0731 TP2 on MI300X.
- Prefix cache verification: warm identical 12k prompt hit 11,904/12,017
  cached tokens and improved TTFT from 12.55 s to 1.10 s (11.4x). Forked
  prompt with same prefix also hit 11,904 cached tokens; partial overlap hit
  5,888/6,281 cached tokens.
- 1M context: `kv_cache_memory_bytes=50_465_865_728` works, giving 1,057,984
  KV tokens and 1.01x maximum concurrency for 1,048,576-token requests.
- End-to-end 1M smoke: 1,000,021 prompt tokens plus 32 greedy output tokens in
  1,847 s, or 541.4 prefill tok/s. Output was coherent and grounded in the
  corpus.
- Decision: 1M is supported at batch 1 with manual KV sizing; prefill dominates
  and DSpark is unaffordable at this length without draft KV capacity work.
- Raw artifacts: `/tmp/apc_verify.json`, `/tmp/ctx1m_smoke.json`.

## 2026-08-07 - Metal DSV4 Reference

- Status: retained cross-platform reference
- Scope: DS4 Metal backend on Apple M5 Max 128 GiB, DeepSeek V4 Flash 0731
  hybrid GGUF, exact 1000 input / 2000 output token harness.
- Correctness-qualified references:
  - Concurrency 1: 33.684 aggregate output tok/s, wall 59.375 s; server decode
    average 35.63 tok/s.
  - Concurrency 8 original baseline: 32.821 aggregate output tok/s, wall
    487.487 s, mean latency 485.164 s.
  - Concurrency 8 stable retained path: 35.350 aggregate output tok/s, wall
    452.614 s, mean latency 452.038 s; +7.70% throughput and -7.15% wall time
    versus original c8 baseline.
- Retained Metal change: `--mixed-prefill-quantum 2048` on top of the existing
  native row batches.
- Rejected Metal ideas recorded in `benchmarks/dsv4_metal_perf.md`: unsafe
  external routed-MoE batching despite speed, count-8 routed MoE fusions,
  alternate Q8 SIMD group counts, Q8 four-row decode due nondeterministic
  digests, command-buffer split changes below threshold, attention-output
  batching due digest drift, and several launch/fusion flags below noise.
- Decision: use as a cross-platform lesson source, especially for native
  row-batching, fused IQ2/Q2 geometry, correctness digests, and scheduler
  effects.
- Raw artifact: `benchmarks/dsv4_metal_perf.md`.

## 2026-08-09 - A100 Output-Owned DSV4 Trace And Ownership Rejection

- Status: rejected end-to-end candidate; trace retained
- Scope: DeepSeek V4 Flash 0731 exact decode, TP2 and TP4 A100, native
  IQ2_XXS routed gate/up, SwiGLU, Q2_K down, and shared Q8_0 expert.
- Baseline/change: compared the stable exact profile baseline (TP2 92.022
  tok/s, TP4 112.573 tok/s, 1.223x) with the output-owned hidden-channel
  implementation. The candidate keeps MoE and projection outputs local and
  gathers only at the final decoder boundary.
- Correctness: real `dsv4-2` and `dsv4-4` profiles reached health and completed
  the exact-token harness. Kernel microbenchmarks passed BF16 tolerances.
- Exact result: TP2 79.683 tok/s, TP4 97.904 tok/s, 1.229x. This misses the
  minimum TP4 gate of 119.525 tok/s for this candidate and regresses both
  absolute throughputs, so it is not retained as the default.
- Trace finding: arithmetic generally shrinks at TP4, but 85 replicated mHC
  transitions remain about 19 us urgent plus 13 us deferred at both TP sizes.
  The direct MoE boundary was about 72.1 us/layer at TP2 and 51.2 us/layer at
  TP4 (1.408x); its standalone shared-Q8 publication regressed from 10.36 us
  to 12.48 us. Final BF16 gather was only about 5 us and was not the cause.
  The first captured replay contained startup outliers and was excluded from
  steady-state collective conclusions.
- Decision: reject the end-to-end ownership wiring. Retain the trace as proof
  that replicated mHC and per-layer synchronization, rather than the final
  gather, are the scaling limit.
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-output-owned-e2e/` and
  `perf/results/2026-08-09/dsv4-a100-output-owned-trace/`.

## 2026-08-09 - A100 Channel-Residual mHC Schedules

- Status: rejected diagnostics; production candidates removed
- Hypothesis: channel-owning the four residual streams by hidden dimension
  would distribute mHC projection work if all 24 projection partials and the
  residual norm used one compact exchange, followed by one BF16 input-norm
  exchange.
- Correctness: the full-input reduce-owned path stayed within 0.0625 BF16 max
  absolute error over eight chained transitions at TP2 and TP4.
- Result:

| Schedule | TP2 us/transition | TP4 us/transition | TP2/TP4 |
| --- | ---: | ---: | ---: |
| Existing three-stage channel-owned | 32.59 | 32.60 | 1.000x |
| Two-exchange monolithic candidate | 26.43 | 25.22 | 1.048x |
| Replicated residual control | 24.19 | 24.58 | 0.984x |

- Decision: reject. Combining all projections saves launch/synchronization
  overhead, but the compact exchanges and fixed control work remain latency
  bound and the candidate is still slower than replicated residual state.
  Channel-sharding residual state is not a valid foundation for decode.
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-channel-owned-mhc/` and
  `perf/results/2026-08-09/dsv4-a100-channel-mhc-monolithic/`.

## 2026-08-09 - A100 Shared-Expert Publication Experiments

- Status: rejected diagnostics; public APIs removed and source diagnostics
  compile-disabled
- Hypothesis: remove the separate shared-Q8 publication kernel from the
  output-owned MoE boundary either by publishing from shared gate/up or by
  staging peer Q8 shards once inside output-owned down.
- Correctness: every retained measurement had zero max absolute error against
  the assembled-publication output for eager and captured graphs.
- Results below exclude the common standalone shared-W1 cost where the
  candidate still uses it:

| Output-owned boundary | TP2 us | TP4 us | TP2/TP4 |
| --- | ---: | ---: | ---: |
| Existing assembled publication, contaminated control | 48.88 | 34.58 | 1.413x |
| Shared-W1 cooperative publish fusion | 72.57 | 48.35 | 1.501x |
| Cooperative direct peer consumption | 59.64 | not retained | -- |
| Non-cooperative direct peer consumption | 50.54 | 40.29 | 1.254x |
| Restored assembled publication control | 47.64 | 33.47 | 1.423x |

- Decision: reject all three. Holding the shared GEMV cooperative grid through
  publication loses occupancy; making down cooperative pays residency and end
  barriers; direct peer reads make TP4 worse. The assembled local publication
  remains the fastest tested contract. The direct-read experiment had also
  increased every retained down-kernel CTA's shared state; restoring the
  original state recovered the control to 47.64 us at TP2 and 33.47 us at
  TP4, with zero error on every rank. The isolated boundary still scales only
  1.423x and therefore cannot satisfy the end-to-end 1.5x gate by itself.
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-fused-shared-publication/`, including
  `cleanup-tp2.log` and `cleanup-tp4.log`.

## 2026-08-09 - A100 TP2 Lifecycle Crash Root Cause (NaN -> OOV token 129280)

- Status: root cause found and fixed; requalification in progress
- Symptom: canonical 1K/2K request died ~7 output tokens into decode with an
  illegal CUDA memory access on both TP ranks. HANDOFF.md blamed a DSV4
  aux-stream ownership race; that theory is REJECTED.
- Evidence chain:
  - Per-path overlap switches (`VLLM_DSV4_OVERLAP_INDEXER`,
    `VLLM_DSV4_OVERLAP_MLA_COMPRESSOR`, `VLLM_DSV4_OVERLAP_INDEXER_INNER` in
    `vllm/models/deepseek_v4/attention.py`) showed EITHER inner-overlap side
    stream alone crashes (v23 compressor-only, v24 indexer-only), so the
    stream config only shifted timing/allocator layout.
  - Kernel Xid records: every crash was an MMU FAULT_PDE VIRT_READ at an
    identical VA on both ranks, low bits always `...a60000`.
  - `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` + cuda-gdb named the faulting
    kernel: `dsv4_router::bf16_hash_router`
    (`csrc/quixicore/serving/dsv4_router_ampere.cuh`), grid 8 (verify batch
    padded to capture size), block/token 0, faulting on the weight-row read
    seeded by `tid2eid[token_id * 6 + warp]` with a garbage token id.
  - A device-side guard + debug slot in the router recorded the value:
    `token_id == 129280 == vocab_size` (tid2eid/embedding/lm-head all have
    exactly 129280 rows), every step, both ranks, token_index 0 (bonus slot).
  - Step probes traced it to `last_sampled_tokens`: at onset (all 5 drafts
    rejected) the rejection sampler itself emitted 129280 as the recovery
    sample; `combine_sampled_and_draft_tokens` then wrote it into
    `input_ids[0]`, poisoning every subsequent step (self-sustaining).
  - Offline repro (`rejection_sample` with a NaN target-logits row, greedy
    temp=0, vocab=129280) reproduced `sampled=[129280]` exactly.
- Root cause: `argmax_combine` in
  `csrc/quixicore/serving/v2_sample_kernels.cuh` used a negated comparison
  (`!(v > ov || ...)`) that lets a NaN candidate replace the running best;
  masked -inf tail lanes of the last vocab block then win with the lowest
  masked index = 129280. A NaN target-logits row from the model forward
  triggers it. Whether the resulting wild `weight + expert*8192` read faulted
  depended on what the allocator placed after the 2 MiB gate weight, which is
  why serializing streams (v21/v22) "fixed" it: layout luck, not ordering.
  Those runs were silently quality-poisoned instead.
- Fixes (retained):
  - `v2_sample_kernels.cuh`: positive-form `argmax_combine` (NaN never wins),
    NaN-sanitized loads and in-vocab-only candidates in
    `v2_block_argmax_8192`, `v2_block_max_sumexp_8192`, `v2_gumbel_sample_k`.
  - Same NaN sanitize in the Triton fallbacks
    (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`).
  - `bf16_hash_router` bounds-guards token ids (OOB treated as padding) and
    records the first offender in a debug slot
    (`quixicore_ops.dsv4_hash_router_debug`), defense in depth.
  - Offline verification: NaN/-inf/garbage rows now sample token 0, no OOV in
    any scenario (`scratchpad/repro_rejection_oov.py` run pre/post fix).
- Open follow-ups:
  - WHY the target forward produces a NaN logits row at all (onset ~35 output
    tokens into the canonical run) is a separate open bug; with the sampler
    now NaN-robust it degrades to sampling token 0 instead of crashing.
  - Diagnostic switches (`VLLM_DSV4_OVERLAP_*`, `VLLM_DSV4_HASH_ROUTER_DEBUG`
    probes in `vllm/v1/worker/gpu/model_runner.py` and
    `rejection_sampler.py`) to be removed/quarantined after requalification.
  - Port the `v2_sample_kernels.cuh` fix to the ROCm copy
    (`csrc/quixicore/tm_rocm/qc_rocm_sample.cu`) and QuixiCore-CUDA.
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-tp2-kv-capacity/control/` (v23-v33 logs),
  `perf/results/2026-08-09/dsv4-a100-tp2-kv-capacity/coredumps/`.
- Requalification (clean server v33, full overlap, no debug env, one server
  lifecycle, all exact): 1K/2K 168.0 and 168.7 tok/s; 12K cold/hot 89.0/93.3;
  128K cold/hot 37.1/38.2; post-128K 1K/2K continuation 111.8. Spec acceptance
  length ~3.5-3.7. Baseline promoted in `perf/baseline_status.md`. Regression
  test added: `tests/kernels/test_rejection_sample_nan.py` (40 cases).

## 2026-08-09 - A100 Quant Strategy: Hybrid Serving + Profile Split

- Status: accepted; profiles updated, baselines measured
- Decision (user): datacenter GPUs serve the Q4K-tail hybrid
  (`Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown`, 90.9 GiB);
  IQ2_XXS is the MacBook-footprint quant. `dsv4-2` and `dsv4-4` now default
  to `Q4K-tail`; new `dsv4-4-mxfp4` keeps the MXFP4 build; new `dsv4-8`
  (MXFP4-only, TP8 first, TP4xDP2 to be benchmarked). `registry.py` gained
  per-quant `quant_overrides` (the 19 GiB TP2 KV budget stays with IQ2_XXS;
  hybrid starts at a provisional 14 GiB pending the same qualification).
- Hybrid runs the same fused native path for its 55 IQ2_XXS/Q2_K layers;
  layers 37-42 (Q4_K) take the generic MMVQ/MMQ route per layer.
- Exact-token results (concurrency 1 unless noted, all `exact: true`):

| Config | 1K/2K c1 | 12K cold | c8 1K/500 agg |
| --- | ---: | ---: | ---: |
| TP2 hybrid (14 GiB KV) | 168.5 / 168.0 | 89.2 | 185.5 |
| TP4 hybrid | 174.2 (first run 135.9, warm-up) | 150.7 | 416.4 |
| TP4 MXFP4 (pre-fused kernels) | 32.3 / 32.1 | 28.7 | 22.0 |

- Hybrid TP2 matches the IQ2_XXS baseline (168 tok/s) exactly: the +12%
  weight bytes cost nothing measurable at decode, so the quality upgrade is
  free at TP2. Same-artifact TP scaling (hybrid TP4/TP2): 1.03x at batch-1
  short-context (latency-bound: tiny all-reduce payloads and fixed per-step
  costs dominate), 1.69x at 12K, 2.24x at concurrency 8. The >=1.5x gate is
  the right check for the throughput regimes, not batch-1 short-context.
- TP4 MXFP4 at 22-32 tok/s with healthy acceptance (~3.5) exposed that A100
  had no fused MXFP4 expert path (generic MMVQ only; c8 slower than c1).
  Fixed the same day - next entry.
- Also fixed: drafter dummy-propose during profile_run crashed under
  FULL_DECODE_ONLY (TQ attention asserted on missing metadata;
  `attention.py` now zeroes the output for metadata-less profile runs).
- Raw artifacts: `perf/results/2026-08-09/dsv4-a100-hybrid-baseline/`,
  `perf/results/2026-08-09/dsv4-a100-tp4-baseline/`.

## 2026-08-09 - A100 Fused MXFP4 MoE Decode Path

- Status: implemented; correctness passed; serving measurement in progress
- Hypothesis: MXFP4 experts on A100 fell through to the generic MMVQ route
  (fp32 intermediates, separate gate/up/SwiGLU/down launches), producing the
  32 tok/s TP4 decode above. Porting the tuned IQ2_XXS/Q2_K pipeline shape
  (Q8_1-staged activation, fused gate/up + SwiGLU + route-weight + Q8_1
  emission, weighted down consuming the Q8_1 mid) to MXFP4 should recover
  most of the gap; MXFP4's table-based e2m1 decode is far cheaper than
  IQ2_XXS codebook decode.
- Implementation: `csrc/quixicore/quant/dsv4_mxfp4_moe_ampere.cuh`
  (`mxfp4_gate_up_swiglu_q8_1_decode`, `mxfp4_down_sum`,
  `repack_mxfp4_experts`), ops `ggml_dsv4_moe_a8_mxfp4` /
  `ggml_dsv4_repack_mxfp4`, dispatch in `gguf/fused_moe.py` for
  qweight 39/39 decode widths (tokens <= 8, top_k 6/8), env gates
  `VLLM_GGUF_DSV4_AMPERE_MXFP4` / `VLLM_GGUF_DSV4_REPACK_MXFP4`.
  The SoA repack is wired but NOT enabled at load: the fused path only
  covers decode, and prefill's generic MMQ kernels still read the raw
  17-byte AoS blocks. Raw-layout loads in the fused kernels until the
  repack-aware coverage is complete.
- Correctness: `tests/kernels/test_dsv4_mxfp4_moe.py` - fused vs fp32
  dequant reference over random MXFP4 experts, tokens {1,4,8} x
  intermediate {64,512}, max relative deviation < 3%; 6/6 passed. Also
  caught a real padded-input-stride bug (Q8_1 rows are 512-padded) before
  serving.
- Measured (`dsv4-4-mxfp4`, exact-token harness, all `exact: true`, coherent
  greedy output on smoke):

| Stage | generic (before) | fused (after) | gain |
| --- | ---: | ---: | ---: |
| 1K/2K c1 | 32.3 / 32.1 | 111.3 / 111.3 | 3.4x |
| 12K cold c1 | 28.7 | 75.0 | 2.6x |
| c8 1K/500 agg | 22.0 | 27.1 | 1.2x |

- Decision: retain. Decode widths (tokens <= 8) now run the fused path; the
  c8 regime still falls to the generic MMQ route (verify batches of
  8 reqs x 6 tokens exceed the gate), which is the remaining gap vs the
  hybrid's 416 tok/s c8.
- Next (not started): widen the fused path or add MXFP4 MMQ tiles for
  verify/prefill widths; enable the SoA repack once every consumer reads the
  split layout; revisit `dsv4-8` TP8 vs TP4xDP2 with these kernels.
- Follow-up survey of the tuned `~/QuixiCore/QuixiCore-XPU` NVFP4/MXFP4
  kernels (user pointer):
  - Ported the shared-activation pair-dot (`nvfp4_row_dot_pair` pattern:
    stage the Q8_1 activation block in registers once, dot gate and up
    against it). Correctness unchanged (6/6), throughput neutral at decode
    (111.0 vs 111.3 tok/s -- weight-bandwidth-bound, activation blocks were
    already L1-resident). Retained: strictly fewer L1 accesses, no cost.
  - The AMD/XPU byte-permute e2m1 table expand does NOT port directly:
    CUDA's `__byte_perm` takes 4-bit nibble selectors, not per-byte
    selectors, and the selector repack erodes the win. Scalar table loop
    retained (comment in `dsv4_mxfp4_moe_ampere.cuh`).
  - XPU stores FP4 scales as separate planes (ModelOpt layout), confirming
    the SoA repack direction for the wide-batch phase; its single-kernel
    whole-expert fusion (gate/up -> local memory SwiGLU -> down with fp32
    atomic accumulation, no mid quantization) is the candidate shape for the
    c8/verify-width path.
- Raw artifacts:
  `perf/results/2026-08-09/dsv4-a100-tp4-baseline/mxfp4-fused-*.json`,
  `server-mxfp4-fused-v3.log`.

## 2026-08-10 - Hybrid TP2 KV Qualification, MXFP4 Wide Gate, dsv4-8

- Status: accepted; profiles finalized, baselines promoted
- Hybrid TP2 KV budget: the provisional 14 GiB budget CRASHED at 128K cold
  (worker killed silently ~3 min into prefill; no Xid, no CUDA OOM string --
  long-prefill workspace exhaustion at the memory edge, the same failure mode
  that rejected the 20/21 GiB XXS candidates). 13 GiB
  (`kv_cache_memory_bytes=13958643712`) passed the full lifecycle, all exact:

| Stage | tok/s |
| --- | ---: |
| 1K/2K r1 / r2 | 168.7 / 168.4 |
| 12K cold / hot | 92.0 / 88.8 |
| 128K cold / hot | 65.4 / 64.4 |
| post-128K 1K/2K | 94.5 |

  (128K decode at 65 tok/s vs the 37 measured on XXS the prior evening; the
  XXS number likely included first-run JIT of the long-context kernels. The
  post-128K continuation dip recurs on hybrid: 94.5 vs 168.)

- MXFP4 fused-route width: raised the dispatch cap from tokens<=8 to an env
  limit (`VLLM_GGUF_DSV4_MXFP4_ROWS`, default 64; op cap 256). Correctness
  extended to width 48 (8/8 pass). DSV4 routing is near-uncorrelated, so the
  per-route warp-GEMV keeps beating the MMQ tiles across verify widths:

| dsv4-4-mxfp4 | gate 8 | gate 64 |
| --- | ---: | ---: |
| c8 1K/500 agg | 27.1 | 98.6 (3.6x) |
| 1K/2K c1 | 111.3 | 112.1 |
| 12K cold | 75.0 | 74.2 |

  Remaining c8 gap vs hybrid's 416 is prefill (chunked 1024-token batches
  still take generic MMQ) -- future MXFP4 prefill tiles.

- dsv4-8 (MXFP4, 8 GPUs): TP8 measured 167.6 c1 / 117.4 @12K / 148.2 c8
  agg, all exact = 1.50x / 1.58x / 1.50x over dsv4-4-mxfp4 -- meets the
  >=1.5x TP gate in every regime. TP4 x DP2 FAILS TO INITIALIZE: DSV4's
  router padding mask is not DP-padding aware
  (`topk_softplus_sqrt_kernels.cu:782` "is_padding size mismatch, expected
  4096" during the DP0/DP1 dummy run -- forward runs at the
  num_tokens_across_dp padded width, mask arrives unpadded). Profile
  finalized as TP8; DP enablement recorded as an open item, and DP2's
  theoretical c8 advantage stays unmeasured until that bug is fixed.
- Raw artifacts: `perf/results/2026-08-10/dsv4-a100-hybrid-qual/` (incl. the
  failed 14 GiB `qual-128k-cold.json` and both server logs),
  `perf/results/2026-08-10/dsv4-a100-mxfp4-wide/`,
  `perf/results/2026-08-10/dsv4-a100-tp8/`.

## 2026-08-10 - DP Enablement + Total-Throughput Matrix

- Status: accepted; DP-padding fixed, matrix measured, layout winners
  encoded in profiles via per-quant overrides
- DP fix: under data parallelism the naive dispatch/combine path all-gathers
  hidden states AND router logits across ranks
  (`moe_runner._maybe_dispatch`), so the router runs on a batch wider than
  the local padded batch. The forward-context `is_padding` mask describes
  only the local batch; slicing it to the gathered width mislabels other
  ranks' real tokens as padding (and a shorter mask tripped the
  `topk_softplus_sqrt` size check, which is what blocked DP engine init).
  Fix: `_get_padding_mask` in both router modules now returns the mask only
  when its width matches the requested token count exactly, else None
  (compute every row). Correct under DP at the cost of not skipping padding
  rows in gathered batches. TP4xDP2 now boots, serves exact, and produces
  coherent output.
- Throughput matrix (exact-token harness, 1K in; c8 = aggregate over 8
  streams x 500 out; c1 = 2K out. Methodology note: matrix c1 runs execute
  after c8 with the same source, so their prefill is APC-warm --
  within-matrix comparisons are consistent, cross-table ones are not):

| Cell | c8 agg | c1 |
| --- | ---: | ---: |
| hybrid TP2 (2 GPU, baseline table) | 185.5 | 168.5 |
| hybrid TP2xDP2 (4 GPU) | 275.6 | 188.3 |
| hybrid TP4 (4 GPU, baseline table) | 416.4 | 174.2 |
| hybrid TP8 | 354.5 | 329.5 |
| hybrid TP4xDP2 (8 GPU) | **567.9** | 247.1 |
| hybrid TP2xDP4 (8 GPU) | INIT FAILS | - |
| mxfp4 TP4 (baseline table) | 98.6 | 112.1 |
| mxfp4 TP8 | 148.2 | 167.6 |
| mxfp4 TP4xDP2 | 110.1 | 111.1 |
| mxfp4 TP2 shards | do not fit 80 GB | - |

- Findings:
  - **8-GPU total-throughput champion: hybrid TP4xDP2 at 567.9 tok/s c8**
    (1.36x hybrid TP4; each replica keeps the fused-path intermediate=512).
  - **Single-stream champion: hybrid TP8 at 329.5 tok/s c1** (~2x TP4;
    11.4 GiB weights/rank), despite losing its fused IQ2 path -- at TP8 the
    per-rank intermediate is 256 and the fused kernels require 512/1024,
    which also explains TP8 losing c8 to TP4 (354.5 vs 416.4). Extending
    the fused IQ2 path to intermediate=256 would lift both TP8 cells.
  - 4-GPU: TP4 beats TP2xDP2 in both metrics for hybrid.
  - MXFP4 keeps TP8 (148.2/167.6 vs 110/111 on TP4xDP2).
  - hybrid TP2xDP4 crashes in the DP4 dummy run (illegal access, distinct
    from the fixed mask bug; `dsv4-a100-matrix/server-hybrid-tp2dp4.log`).
    Not competitive with TP4xDP2 by construction, so left unfixed and
    marked illegal in the profile note.
  - Anomaly worth a follow-up: a TP2 replica under dsv4-4/8 profile
    settings measured 188.3 c1 vs dsv4-2's 168.5 -- suspect
    FULL_DECODE_ONLY vs PIECEWISE graph mode; try FULL_DECODE_ONLY on
    dsv4-2.
- Profile encoding: `dsv4-8` a100 now carries
  `quant_overrides: {Q4K-tail: {tensor_parallel_size: 4,
  data_parallel_size: 2}}` -- MXFP4 default stays TP8. `dsv4-4` stays TP4.
  Tests assert both layouts (36 profile tests).
- Raw artifacts: `perf/results/2026-08-10/dsv4-a100-matrix/`.

## 2026-08-10 - Fused IQ2 at TP8, Concurrency Curves, APC Methodology

- Status: accepted; fused path extended, methodology corrected, hot
  steady-state numbers recorded, layout winners confirmed
- Fused IQ2 at intermediate=256 (TP8 shard): the earlier "hybrid TP8 lost
  its fused path" finding was root-caused by coredump to
  `q2_k_down_sum_repacked` -- the launcher's else-branch silently ran the
  512 instantiation for any unlisted shard, reading 2x past every row
  (illegal access at first boot with the gate relaxed). Fixed: explicit
  256 instantiation with an idle-lane guard (16 subblocks over a 32-lane
  warp), the static_assert generalized, and the custom-allreduce-owned
  pending-down path pinned to its validated 512/1024 shapes. Boots, serves
  exact, coherent output.
- Harness: `--concurrency` now accepts {1,2,4,8} (was {1,8}).
- METHODOLOGY: aggregate tok/s conflates prefill amortization with decode
  throughput. Back-to-back identical runs measured 320 -> 680 tok/s at TP8
  c4 purely from APC state (second run's prefill is a cache hit), and a
  capture-64 experiment that shrank the KV pool looked like a kernel
  regression because it evicted APC between stages. Standard going forward:
  run each point twice, report the second (APC-hot steady state, decode
  dominated), and label cold numbers as such.
- Hot steady-state (hybrid, 1K in / 500 out per stream, aggregate tok/s):

| Config | GPUs | hot c4 | hot c8 |
| --- | ---: | ---: | ---: |
| TP4 (`dsv4-4`) | 4 | 518.7 | 416.7 |
| TP8 (fused-256, capture 32) | 8 | 679.8 | 205.4 |
| TP4xDP2 | 8 | 281.6 | **926.1** |

- The structure: the engine-level sweet spot is a verify batch that fits
  CUDA graph capture (per-engine concurrency 4 -> 24 tokens <= capture 32).
  At per-engine c8 the 48-token verify batches run eager and every config
  cliffs (TP8 679 -> 205, TP4 519 -> 417). TP4xDP2 wins total throughput at
  c8 because each replica sits at its sweet spot: **926.1 tok/s aggregate**,
  the highest measured on this hardware. TP8 remains the latency/low-
  concurrency choice (best c1-c4).
- capture-64 alone did NOT recover TP8's c8 (438 mixed-state; the capture
  sizes list tops out below the 48-token verify width) and its larger
  graphs shrink the KV pool. Follow-up: extend `cudagraph_capture_sizes`
  to include 48 (and re-derive the KV budget) -- if TP8's c8 then exceeds
  926, revisit the hybrid layout choice.
- Profile state (final): `dsv4-8` MXFP4 -> TP8; `dsv4-8 --quant Q4K-tail`
  -> TP4xDP2 (unchanged from the matrix decision, now confirmed with clean
  hot data); capture stays 32.
- Raw artifacts: `perf/results/2026-08-10/dsv4-a100-matrix/` (conc sweeps,
  hot pairs, `core256_*` coredump analysis logs).

## 2026-08-10 - TP8 c8 Cliff Root Cause: Graph Capture Width (FIXED)

- Status: accepted; `dsv4-8` a100 now ships `max_cudagraph_capture_size: 64`
- The user challenged the 205 tok/s TP8 hot c8 as a probable bug. Correct:
  the c8 verify batch is 8 reqs x 6 spec tokens = 48 tokens, and the
  capture-size list topped out at 32, so every decode step ran EAGER. With
  max 64 the derived list is [1,2,4,8,16,24,32,40,48,56,64]; 48 is
  captured and the cliff disappears. The earlier capture-64 experiment had
  been wrongly dismissed: its numbers were confounded by APC eviction from
  the larger graph memory (the hot-pair methodology now catches this).
- Hot steady-state deltas from capture 32 -> 64 (1K/500, aggregate):

| Config | hot c8 (cap 32) | hot c8 (cap 64) | hot c4 (cap 64) |
| --- | ---: | ---: | ---: |
| hybrid TP8 | 205.4 | **465.5** (2.3x) | 691.0 (unchanged) |
| hybrid TP4xDP2 | 926.1 | 925.7 / 858.4 (holds) | - |
| mxfp4 TP8 | 148.2 | **275.1** (1.9x) | - |

- Residual TP8-vs-DP2 gap at c8 (465 vs ~900): the 48-wide verify crosses
  the >=256 routed-row wide-layout threshold (288 rows) in the fused IQ2
  W1 route, and DP2 carries twice the aggregate KV/APC. Isolating the
  wide-layout cost (a c6 probe sits under the threshold but above capture
  32) is the next step if TP8 is to challenge DP2's crown.
- Layout decisions unchanged and now clean: `dsv4-8` MXFP4 -> TP8,
  Q4K-tail -> TP4xDP2 (925.7/858.4 hot c8, box record).
- Raw artifacts: `perf/results/2026-08-10/dsv4-a100-matrix/tp8-cap64-*`,
  `dp2-cap64-*`, `mxfp4-tp8-cap64-*`.

## 2026-08-10 - Gap Investigation, Final Profile Structure

- Status: accepted; profiles renamed and finalized per user direction
- TP8-vs-DP2 gap investigation (hybrid, hot c8 465-606 vs 858-926):
  - The wide-layout hypothesis is EXONERATED: a c6 probe (216 routed rows,
    4-wide layout, capture-40 graphs) measured 385.9 hot -- BELOW c8's
    606.3 (288 rows, 8-wide). No cliff at the 256-row threshold; the 8-wide
    layout is fine. The `VLLM_GGUF_DSV4_W1_WIDE_ROWS` runtime threshold
    (mirrored C++/Python) stays as an A/B tool.
  - Two honest residual factors: (a) per-engine width economics -- one
    48-token verify step on TP8's small shards yields less than 2x the
    throughput of two 24-token steps on TP4 shards; (b) run variance:
    +/-30% boot-to-boot on identical configs (APC block-overlap from
    preceding stages; spec-decode acceptance lengths swing 2.6-6.0 across
    measurement windows at temp=0 purely from text predictability).
    Every DP2 sample (858-926) still beat every TP8 sample (205-606), so
    the hybrid throughput ordering stands.
- mxfp4 8-GPU layout settled with hot pairs at capture 64: TP4xDP2 measured
  271.2 / 117.6 (unstable, best-case ties TP8) vs TP8's 217.7 / 275.1 and
  167.6 c1 (vs 111). TP8 wins for MXFP4: its decode is weight-bandwidth
  bound (TP8 halves per-rank bytes) and its c8 limiter is prefill, which DP
  does not relieve per engine.
- Final DSV4 A100 profile structure (user direction):
  `dsv4-hybrid-2` (TP2, 13 GiB KV), `dsv4-hybrid-4` (TP4),
  `dsv4-mxfp4-4` (TP4), `dsv4-mxfp4-8` (TP8, capture 64). The hybrid-on-8
  quant override was REMOVED per the 8-GPUs-get-the-quality-quant policy;
  the box throughput record (hybrid TP4xDP2, ~858-926 tok/s hot c8 agg)
  is therefore intentionally unserved and documented here for one-line
  resurrection (dsv4-mxfp4-8 --quant Q4K-tail with tp4/dp2).
- Housekeeping: the case-colliding root `handoff.md` moved to
  `perf/dsv4_a100_kernel_history.md` with an obsolescence header (its
  serving numbers predate the sampler fix); `HANDOFF.md` is the sole
  resume document.
- Raw artifacts: `perf/results/2026-08-10/dsv4-a100-matrix/tp8-probe-*`,
  `mxfp4-dp2-cap64-*`.

## 2026-08-10 - dsv4-hybrid-8 Profile Added and Accepted

- Status: accepted (user decision, reversing the earlier hybrid-on-8
  exclusion after the throughput data)
- Final A100 dsv4 set: dsv4-hybrid-2 (TP2), dsv4-hybrid-4 (TP4),
  dsv4-hybrid-8 (TP4 x DP2, capture 64 -- the throughput tier),
  dsv4-mxfp4-4 (TP4), dsv4-mxfp4-8 (TP8, capture 64 -- the quality/latency
  tier). All five: DSpark k=5 + TurboQuant draft KV + 1M max_model_len from
  the GGUF (test-enforced).
- dsv4-hybrid-8 acceptance run on the named profile: hot c8 aggregate
  921.8 tok/s (cold 752.3), exact -- reproduces the 858-926 record band.
- Long-context caveat: only dsv4-hybrid-2 has the explicit 128K
  cold/hot/post lifecycle qualification; the 4/8-GPU profiles are measured
  at 1K-12K and configured for 1M. Their 128K lifecycle pass is an open
  item.
- Raw artifacts:
  `perf/results/2026-08-10/dsv4-a100-matrix/dsv4-hybrid-8-accept-*`.

## 2026-08-10 - 128K Lifecycle Qualification: hybrid-4/8, mxfp4-4/8 (ALL PASS)

- Status: accepted; all five A100 dsv4 profiles are now 128K-lifecycle
  qualified (dsv4-hybrid-2 was qualified earlier the same day).
- Method: canonical single-lifecycle sequence per profile without restart
  (1K/2K x2, 12K cold/hot, 128K cold/hot, post-128K 1K/2K continuation),
  exact-token harness, concurrency 1, 8-token warmup. Phase 1 ran
  dsv4-hybrid-4 (GPUs 0-3) and dsv4-mxfp4-4 (GPUs 4-7) concurrently on a
  split box (minor cross-contention possible; hybrid-4 r1 128.1 vs r2 161.8
  suggests warm-in effects); phases 2-3 ran the 8-GPU profiles alone.
- Every stage on every profile: rc=0, `exact: true`. Zero preemptions in
  server logs. hybrid-8 KV planner: 8,125,613 logical tokens.

| Stage (c1 tok/s) | hybrid-4 (TP4) | hybrid-8 (TP4xDP2) | mxfp4-4 (TP4) | mxfp4-8 (TP8) |
| --- | ---: | ---: | ---: | ---: |
| 1K/2K r1 / r2 | 128.1 / 161.8 | 39.9 / 40.0 | 110.8 / 110.8 | 164.9 / 164.5 |
| 12K cold / hot | 152.6 / 155.2 | 26.9 / 37.3 | 76.6 / 75.5 | 117.1 / 116.5 |
| 128K cold / hot | 94.8 / 98.3 | 3.0 / 26.4 | 55.7 / 57.2 | 80.4 / 81.5 |
| post-128K 1K/2K | 171.4 | 40.0 | 110.5 | 165.0 |

- hybrid-8 c1 numbers are the known DP2 characteristic, not a defect: at
  concurrency 1 only one replica works while every step pays DP
  coordination (~40 tok/s ceiling). The 128K-cold 3.0 is DP round-robin
  defeating the warmup APC -- the timed request lands on the replica that
  did NOT serve the warmup, so the full 128K prefill falls inside the
  timed window. Prefix-affinity DP routing is the recorded fix if c1
  long-context ever matters on this tier; its service point is c8
  aggregate (921.8 hot, accepted 2026-08-10).
- post-128K dip is TP2-only: hybrid-4 continued at 171.4 (its fresh-server
  band) and mxfp4-8 at 165.0 (matches its 1K/2K 164.9) after the 1M-scale
  context, while dsv4-hybrid-2 dips 168->94.5. Supports the KV-pool
  occupancy theory scaling away with per-rank KV headroom.
- mxfp4-8 (TP8) has clean long-context economics: 128K hot holds 81.5
  (49% of its 1K/2K rate, vs hybrid-4 holding 61%).
- Raw artifacts: `perf/results/2026-08-10/dsv4-lifecycle-qual/`
  (per-stage JSON + server logs); driver script preserved as
  `lifecycle_qual_par.sh` in the session scratchpad (single-box phase
  ordering: split-4s, then each 8-GPU profile alone).

## 2026-08-10 - MXFP4 Tensor-Core Grouped MoE Tiles (wide batch / prefill)

- Status: accepted (dsv4-mxfp4-4 and dsv4-mxfp4-8 both validated e2e)
- Baseline: mxfp4-4 lifecycle numbers (same day): 110.8 c1, 76.6/75.5 @12K,
  ~99-118 c8 -- 50% of hybrid-4 at 12K and ~25% at c8 when activated-byte
  parity predicts ~59%. Kernel-level cause: the wide MXFP4 route ran the
  dp4a grouped tile `moe_mxfp4` at MOE_X=4 columns, re-streaming every
  expert's full weights per 4 routed rows (cost linear in tokens).
- Change: new grouped tensor-core tile `moe_mxfp4_mmq_v2`
  (csrc/quixicore/quant/dsv4_mxfp4_mmq_ampere.cuh): 128 expert rows x 64
  routed columns, K in 256-value iterations, int8 mma.sync. e2m1 nibbles
  decode through the fused path's 2x int8 table into exactly the dense
  Q8_0 MMQ v2 shared layout (0.5 folded into the e8m0 scale), so
  vec_dot_q8_0_q8_1_mma runs unmodified; only the loader, the
  sorted_token_ids gather, and the scatter write-back are new. Reads raw
  AoS or the SoA repack (repack still off). Dispatch: ggml_moe_a8 case 39,
  env VLLM_GGUF_MXFP4_MMQ_V2 (default on) -- the env also widens
  ggml_moe_get_block_size(39) to 64 so moe_align metadata matches; K not a
  multiple of 256 falls back to a 64-wide dp4a instantiation.
- Latent bug found and fixed in shared moe_q (moe.cuh): the activation
  scale load indexed token_offs[threadIdx.y] into a per-thread array of
  mmq_x/nwarps entries -- only correct when mmq_x == nwarps, which every
  existing instantiation happened to satisfy. Generalized to the real
  per-column loop (behavior-identical for square tiles, required for the
  64/8 fallback). Caught by the K=288 fallback test (rel error 2.4e6).
- Correctness: tests/kernels/test_dsv4_mxfp4_moe.py extended with
  ggml_moe_a8 vs fp32-dequant reference through real moe_align metadata:
  W1 shapes, W2 shapes (top_k=1), row-tail clamp, K%256!=0 fallback --
  16/16 pass, rel < 0.02.
- Kernel microbench (A100, E=256, top_k=6, TP4 shard: W1 1024xK4096 +
  W2 4096xK512, ms for the pair):

| routed tokens | dp4a (old) | mma tile (new) | speedup |
| ---: | ---: | ---: | ---: |
| 128 | 43.2 | 4.9 | 8.8x |
| 512 | 124.5 | 5.6 | 22x |
| 1024 | 236.4 | 6.4 | 37x |
| 2048 | 459.2 | 8.0 | 57x |

  (old is linear in tokens; new is weight-stream flat)

- E2e dsv4-mxfp4-4 (exact harness, spec decode, all `exact: true`):

| stage | before | after | delta |
| --- | ---: | ---: | ---: |
| 1K/2K c1 | 110.8 | 129.2 / 126.1 | +15% |
| 12K cold / hot | 76.6 / 75.5 | 110.6 / 115.8 | +44% / +53% |
| 1K/2K c8 agg | ~99-118 | 208.4 / 200.3 | ~2x |

  12K now holds 88% of the c1 rate (hybrid-4 holds ~95%); c8 at 208 is
  now within the activated-byte ratio of hybrid TP4's 417-606 band.

- E2e dsv4-mxfp4-8 (TP8, capture 64, all `exact: true`):

| stage | before | after | delta |
| --- | ---: | ---: | ---: |
| 1K/2K c1 | 164.9 / 164.5 | 182.8 / 183.3 | +11% |
| 12K cold / hot | 117.1 / 116.5 | 161.1 / 163.3 | +39% |
| 1K/2K c8 agg | 217.7 / 275.1 | 302.6 / 328.3 | +19-39% |

  TP8 12K now holds 88% of c1 (matches TP4's post-tile ratio); the c8
  gain confirms prefill was the TP8 c8 limiter after the capture-64 fix.

- Open levers recorded: fused SwiGLU+Q8_1 wide epilogue (QuixiCore-XPU
  glu_quant is the precedent; wide route still runs act + requant as
  separate elementwise steps), permuted expert-contiguous segments instead
  of 64-padded alignment for verify widths (XPU grouped_qgemm precedent),
  cp.async staging with the SoA repack, fused-vs-tile crossover sweep for
  the decode gate (VLLM_GGUF_DSV4_MXFP4_ROWS still 64).
- Raw artifacts: `perf/results/2026-08-10/dsv4-mxfp4-mmqv2/` (+ `-tp8/`),
  microbench script in session scratchpad `bench_mxfp4_mmq.py`.

## 2026-08-10 - Attention Reduce NaN-Guard Hardening (XPU review port)

- Status: accepted (behavior-neutral hardening)
- The QuixiCore-XPU merge_attn_states port names "the empty-partition guard
  whose absence caused NaNs in the CUDA lineage of this op". Audit of our
  merge sites: all four paged_attn_v2 reducers and the MLA online-softmax
  loops already guard empty partitions correctly (exact NEG_INF skip,
  all-empty -> 0, guarded sink), and NEG_INF constants agree between MLA
  writers and the shared reducers (-FLT_MAX both). The one gap: a NaN
  partial stat passes `mp == NEG_INF` and poisons the head's output (fmaxf
  drops NaN from the global max, but the NaN re-enters via expf).
- Change: all six weight-guard sites upgraded to `!(mp > NEG_INF)` --
  identical for every finite/empty input, degrades a NaN partial to
  "empty" instead of NaN output. Same philosophy as the sampler-side NaN
  sanitize. Relevant to the open NaN-origin item (task: rare all-NaN
  logits rows): if attention was the amplifier, this contains it.
- Rebuilt _quixicore_C, import smoke passed.

## 2026-08-10 - Segmented MoE + Fused SwiGLU Epilogue (XPU ideas 1+2)

- Status: accepted, default on (VLLM_GGUF_DSV4_MXFP4_SEG=0 reverts to the
  moe_align MMQ route)
- Design (from the XPU grouped_qgemm/glu_quant code review): device-side
  route grouping (histogram/prefix/scatter, no host sync, no sorted+padded
  metadata), STATIC worst-case grid (ceil(M/J)+E column tiles) with a
  per-block prefix-table walk -- CUDA-graph-capture-safe with varying
  routing; W1 tile epilogue fuses SwiGLU + route weight + Q8_1 emission
  (gate row r and up row I+r paired inside one 128-row tile; C spilled to
  the dead weight-staging smem), eliminating the [routes, 2I] half
  intermediate and the separate act + quantize passes; W2 reads the fused
  Q8_1 mid; a deterministic per-token reduce replaces atomic accumulation.
  J template {16, 64}: 16 below VLLM_GGUF_DSV4_SEG_J16_ROWS=1536 routed
  rows (quarter the masked-tail mma waste), 64 for prefill.
- Files: csrc/quixicore/quant/dsv4_mxfp4_seg_ampere.cuh, op
  ggml_dsv4_moe_a8_mxfp4_seg (gguf_kernel.cu), dispatch in
  gguf/fused_moe.py ahead of the moe_align machinery. Iteration recorded:
  per-block thread-0 serial table build cost ~10-15% at wide M; moved to
  the prefix kernel with cooperative smem loads.
- Correctness: 4 new tests (J16 x2, J64, invalid-route drop) vs fp32
  dequant reference -- 20/20 file total.
- Kernel microbench (op TOTAL incl. quantize/perm/reduce, vs the mmq
  GEMM-pair-only times which exclude ~0.5-2 ms of align/act/quant/reduce):

| routed tokens | mmq GEMMs only | seg op total |
| ---: | ---: | ---: |
| 48 | - | 2.55 ms |
| 128 | 4.87 ms | 3.59 ms |
| 512 | 5.56 ms | 6.02 ms |
| 1024 | 6.36 ms | 6.89 ms |
| 2048 | 7.97 ms | 8.50 ms |

  (adding the mmq route's external passes makes seg the winner at every
  width: ~-33% at 128, ~-12% at 2048)

- E2e dsv4-mxfp4-4 (exact, spec decode): c1 125.1/127.5 (par vs 129/126),
  12K 112.1/115.0 (vs 110.6/115.8), c8 cold 276.9 (vs 208.4, +33%), c8
  hot 204.3 (vs 200.3, par). Reading per the APC-hot methodology: the win
  lands exactly where prefill executes (cold c8, cold 12K); hot stages
  have cached prefixes so the wide path barely runs. Never worse; plus
  capture safety, determinism, and less metadata. Retained.
- E2e dsv4-mxfp4-8 (TP8, I=256 shapes, all exact): c1 184.2/186.2 (par vs
  182.8/183.3), 12K 162.4/159.5 (par vs 161.1/163.3), c8 cold 356.2 (vs
  302.6, +18%), c8 hot 694.8 (vs 328.3). CAVEAT on the hot number: +112%
  exceeds anything the seg path can cause at hot c8 (verify batches ride
  the unchanged fused path; hot prefixes are APC-cached) -- treat it as a
  favorable acceptance/APC window inside the documented +/-30%+ variance,
  not a kernel claim. The defensible wins are the cold-c8 pair: TP4 +33%,
  TP8 +18%, both where prefill actually executes.
- Raw: `perf/results/2026-08-10/dsv4-mxfp4-seg/` (+ `-tp8/`).

## 2026-08-10 - Merge Regression: Metal drafter class captured the A100 path

- Status: fixed (platform gate); found because the capture-64 A/B doubled
  as the first post-rebase serving boot.
- Two regressions from reconciling the force-pushed Metal/ROCm lineage:
  1. Boot failure: the shared decoder layer passes `prequant_input` (our
     Ampere aligned-Q8 plumbing) to its attention module; the remote-new
     `DeepseekV4TurboQuantDraftAttention` did not accept it. Fixed by
     accepting-and-ignoring in the draft class.
  2. Throughput collapse: after (1), dsv4-q4ktail-4 measured c1 95.3
     (baseline 128-162), 12K 89.9/90.2 (baseline 152.6/155.2), all exact.
     The harness's new spec-metrics deltas narrowed it fast: DSpark
     acceptance 1.9-2.0 vs the 3.5-3.7 band, and the per-position rates
     were the smoking gun -- 0.89, 0.05, 0.00, 0.00, 0.00 (position 0
     healthy, every parallel-drafted position dead) at decode widths,
     recovering to a normal cascade during wide c8 batches.
     Investigation path: (a) drafter attention_factory platform-gate --
     no effect (same acceptance with the factory off; gate kept anyway as
     the conservative validated config); (b) suspected NHD cache-layout
     change -- wrong, the old regime transposed a head-major allocation
     into the same NHD view the kernels always consumed; (c) file-swap
     probe with the pre-rebase turboquant_attn.py restored 129 tok/s and
     the healthy 0.94/0.82/0.46/0.14/0.04 cascade -> regression isolated
     to that file's rewrite. ROOT CAUSE: the rewrite honors the drafter
     config's declared non-causal layers (cam.causal) on every path; the
     native CUDA/HIP TQ decode kernels are causal-only, so non-causal-
     marked draft decode fed them an unimplementable flag and every
     same-step parallel-draft position lost intra-block visibility. The
     prefill path (flash/SDPA) implements non-causal correctly, which is
     why wide batches were unaffected. FIX: clamp causal to True at the
     metadata builder on non-Metal platforms (the drafter's per-step
     6-token forward routes through the CONTINUATION-decode path, not the
     decode portion, so a decode-site-only clamp measured no change --
     94.8/1.86; the builder-level clamp restores 129.5 tok/s and the
     0.92/0.75/0.46/0.15/0.02 cascade, matching the old-file probe
     exactly. Metal's kernels implement the flag and keep it.)
- Lesson recorded: post-merge unit suites are not serving validation --
  the first profile boot after any cross-lineage merge is part of the
  merge. The spec-metrics columns in benchmark_dsv4_exact.py (adopted
  from the Metal lineage in the same merge) are what made the diagnosis
  a one-liner; keep them.
- Also this session: sha256 pins landed for Q4K-tail
  (659e22fb...) and MXFP4 (0e3a161b...) hashed on-box with the IQ2_XXS
  hash matching the existing verified pin exactly (control); Q4_K pin
  pending a hash on the MI300X box (test-enforced allowlist).

## 2026-08-10 - dsv4-q4ktail-4 capture 32 -> 64 (c8 verify graphs)

- Status: accepted (profile shipped with capture 64)
- Hypothesis: c8 verify batches are 8x6 = 48 tokens > capture 32, so every
  c8 verify step ran eager -- the same cliff fixed on the 8-GPU tiers
  (TP8 hybrid 205 -> 465).
- A/B on the fixed serving stack (exact harness, spec metrics recorded):
  c1 122.4/153.0 and 12K 152.5/155.6 are par with baseline (128.1/161.8,
  152.6/155.2); c8 measured 290.4 (acc 3.02) / 752.0 (acc 5.98) vs the
  417 hot baseline. The 752 rides the high end of the documented
  acceptance window (2.6-6.0 by text position), so the honest claim is
  "hot c8 moves from the ~417 band into the 500-750 band with the
  46-token verify step captured"; the mechanism (eager -> FULL_DECODE_ONLY
  replay at verify width) is the same one measured 2.3x on TP8 with
  matched methodology. More paired runs would tighten the magnitude.
- Raw: `perf/results/2026-08-10/dsv4-q4ktail-cap64/`.

## 2026-08-10 - Hybrid (IQ2_XXS/Q2_K) Segmented Tiles: measured, gated at 768

- Status: retained with the measured crossover gate
  (VLLM_GGUF_DSV4_IQ2_SEG_TOKENS=768); kernels correct at both J widths
  vs the ggml_dequantize reference (rel < 0.03).
- Design: dsv4_hybrid_seg_ampere.cuh -- paired-IQ2-repack W1 loader into
  the Q8_0 mma tile with the fused SwiGLU+route-weight+Q8_1 epilogue;
  three-plane Q2_K W2 on m16n8k16 fragments with per-16 (d*sc, dmin*m)
  half2 planes and ones-matrix-mma min-term synthesis (upstream llama.cpp
  Turing recipe, ones-mma in place of their d2s6 y layout so plain
  block_q8_1 y tiles serve).
- Kernel A/B vs the tuned 8-wide fused pipeline (per-layer op pair, A100
  TP4 shapes E=256/hidden 4096/I=512, ms):

| tokens | fused | seg |
| ---: | ---: | ---: |
| 48 | 0.93 | 1.75 |
| 128 | 1.63 | 2.41 |
| 512 | 4.65 | 6.05 |
| 1024 | 8.77 | 6.85 |
| 2048 | 17.00 | 8.50 |

  The fused pipeline earns its keep below ~768 routed tokens (the
  notebook's standing warning that hybrid never had MXFP4's wide-path
  collapse, quantified); the tiles win 1.3-2.0x at prefill-chunk widths.
  Gate set to 768: prefill chunks ride the tiles, decode/verify widths
  keep the fused route.

- E2e note (gate was 32 during the run, i.e. seg over-applied): c1/12K/c8
  par with the capture-64 baseline; c1 cold 202.6 (acc 4.06). The c8
  cross-boot comparison is acceptance-confounded in BOTH directions
  because different kernel numerics change the greedy continuation and
  therefore the text's acceptance window (752@5.98 vs 294@3.37 on
  identical stages) -- `exact` checks token counts, not content. Op-level
  timing above is the retention evidence; a gate-768 e2e pair rides the
  next qualification pass.
- Raw: `perf/results/2026-08-10/dsv4-q4ktail-iq2seg/`, bench script in
  session scratchpad `bench_iq2_seg.py`.

## 2026-08-10 - MXFP4 verify-width tile routing: REJECTED with data

- Hypothesis: at c8 verify widths (48 tokens) the fused per-route GEMV
  re-reads expert weights once per route, and MXFP4's rows are ~1.7x
  IQ2's, so the segmented tiles should win there and the fused row gate
  (VLLM_GGUF_DSV4_MXFP4_ROWS=64) should drop.
- Measurement (per-layer op pair, A100 TP4 shapes, ms):
  8: 0.38/0.80, 16: 0.71/1.24, 32: 1.36/1.98, 48: 2.03/2.55,
  64: 2.69/2.94 (fused/seg). The fused GEMV wins at every verify width;
  extrapolated crossover ~72 tokens. The shared-activation per-route
  pattern plus zero padding beats the tile's fixed costs below ~70
  tokens even at MXFP4 byte weights.
- Decision: gate stays at 64 (already within a few tokens of optimal).
  Bench: session scratchpad `bench_mxfp4_verify.py`.
- Same session: dsv4-mxfp4-4 graph capture 32 -> 64 (the fix q4ktail-4
  and the 8-GPU tiers already carry; mxfp4-4 had been missed, so every
  earlier mxfp4-4 c8 number ran its 48-token verify eager).
- E2e verdict (all exact, acceptance 3.2-3.6 = normal band):
  c1 126.7/125.6 par; 12K 118.4/114.6 par-to-up; 128K 79.0/80.7 vs the
  stale qualification cells 55.7/57.2 (+42% -- the segmented-tile
  prefill win, now on the record); c8 211.9/186.8 = PAR, capture-64 is
  NEUTRAL for mxfp4-4. Why, and why that differs from its siblings: the
  mxfp4 TP4 verify step is kernel-bound -- the fused per-route GEMV
  costs ~2.0 ms/layer at 48 tokens (~120 ms/verify step), so eager
  launch overhead was already amortized. q4ktail (0.93 ms/layer) and
  mxfp4 TP8 (I=256 shard, ~4x cheaper per rank) were launch-bound at
  verify width, which is why graphs moved them 1.8-2.3x. Capture 64
  retained anyway (harmless, uniform with the family).
- The mxfp4-4 c8 limiter is therefore the verify-step MoE kernel time
  itself. Next levers, in order: a multi-wide fused GEMV (share weight
  reads across 4-8 routes, the tuned-IQ2 trick -- MXFP4's fused path is
  1-wide today) and the SoA repack + cp.async (helps this loader too).

## 2026-08-10 - MXFP4 SoA Repack Enabled: the whole profile moves

- Status: accepted, default on (VLLM_GGUF_DSV4_REPACK_MXFP4=0 reverts)
- The byte-neutral AoS(17)->SoA(scales | aligned codes) repack existed as
  a dormant template in every wide consumer; the original blocker
  (generic MMQ reading raw) no longer applies since the tile/seg/fused
  family owns every type-39 path. Load-time repack now runs for (39,39)
  pairs; flags thread through the seg op and ggml_moe_a8; raw-only MMVQ
  paths are fenced for repacked stacks.
- Kernel A/B (bit-identical outputs everywhere, maxdiff 0.00e+00):
  fused per-route GEMV 2.0x at all decode/verify widths (2.03 -> 1.01 ms
  at 48 tokens); segmented tiles 1.12-1.34x.
- E2e dsv4-mxfp4-4 (all exact, acceptance 3.0-3.54 = normal band, no
  confounds):

| stage | pre-repack (same day) | repacked | delta |
| --- | ---: | ---: | ---: |
| 1K/2K c1 | 126.7 / 125.6 | 159.5 / 158.2 | +26% |
| 12K cold/hot | 118.4 / 114.6 | 139.0 / 139.1 | +18% |
| 128K cold/hot | 79.0 / 80.7 | 89.9 / 92.5 | +14% |
| 1K/2K c8 | 211.9 / 186.8 | 322.8 / 248.3 | +33-52% |

  Cumulative today for mxfp4-4 vs the morning qualification: c1 111->159
  (+43%), 12K 76->139 (+83%), 128K 56->90 (+61%), c8 ~110->250-320
  (~2.5x). c1 now sits within ~2% of q4ktail-4's r2 (158 vs 162) despite
  1.7x the activated bytes.

- Multi-wide GEMV re-ranked: with aligned loads the fused path runs
  ~1.01 ms at 48 tokens; route dedup's remaining ceiling is ~1.67x at
  verify widths (~0.6 ms floor). Still the next fused-path lever, after
  the cheaper wins are exhausted.
- Raw: `perf/results/2026-08-10/dsv4-mxfp4-repack/`, benches
  `bench_mxfp4_repack.py` / `bench_seg_repack.py` in session scratchpad.

## 2026-08-10 - Deployed dsv4-q4ktail-4 concurrency sweep (production daemon)

- Setup: the slimserve-a systemd instance (GPUs 0-3, port 27830, ctx
  262144, served name DeepSeek-v4-Flash-0731, DSpark k=5 + TurboQuant),
  ~/.local/SlimServe-env runtime, instance B idle on GPUs 4-7. Exact
  1000-in/2000-out, 8-token warmup, single run per level, all exact.

| conc | agg tok/s | per-req tok/s | acc len | mean latency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 175.7 | 175.7 | 3.53 | 11.4s |
| 2 | 216.1 | 108.0 | 3.75 | 16.7s |
| 4 | 292.7 | 73.2 | 3.63 | 25.9s |
| 8 | 372.4 | 46.5 | 4.24 | 27.7s |
| 16 | 732.4 | 45.8 | 5.72 | 39.9s |
| 32 | 925.3 | 28.9 | 5.88 | 65.6s |
| 64 | 1023.8 | 16.0 | 5.70 | 115.8s |

- 1023.8 aggregate on one TP4 instance exceeds the 8-GPU TP4xDP2 hot-c8
  record (921.8) -- more concurrent sequences per engine beats a second
  engine at these widths. Acceptance climbs with concurrency (deeper
  same-source windows at higher c inflate the c16+ cells somewhat; the
  cross-level shape is still informative). c16 doubling c8 despite
  96-token verify batches exceeding capture 64 says eager verify at high
  occupancy is launch-amortized, consistent with the capture-neutral
  mxfp4-4 finding.
- The box now serves two such instances; aggregate ceiling with both
  loaded is a future measurement (host CPU contention unknown).
- Raw: `perf/results/2026-08-10/deployed-q4ktail-4-sweep/`.

## Historical Notes

- `perf_worklog.md` contains prior GLM-5.2 performance and correctness
  investigation history.
- `benchmarks/dsv4_metal_perf.md` contains DeepSeek V4 Flash 0731 Metal
  throughput history and should be studied when translating wins across
  platforms.

## 2026-08-10 - Naming note: dsv4-hybrid-*is now dsv4-q4ktail-*

Entries above reference the profile ids in force when each run was made
(dsv4-2 -> dsv4-hybrid-N -> dsv4-q4ktail-N for the Q4K-tail artifact).
"hybrid" was dropped because it does not discriminate (every serious GGUF
build is a mix) and broke the tag -> registry-quant mapping; q4ktail names
the distinguishing feature (layers 37-42 experts in Q4_K). Configs are
unchanged -- ids only.

## 2026-08-10 - INCIDENT: silent output degeneration under load (NaN-class), production dsv4-q4ktail-4

### Discovery

While measuring the dual-instance aggregate ceiling, instance A reported
acceptance 5.98 (k=5 ceiling is 6.0) on brand-new text while instance B
reported 3.7-4.3 on adjacent regions. Cross-swapping prompt regions showed
the anomaly follows the instance, not the content. Direct probing showed the
"fast" instance emits token 0 (`<|begin_of_sentence|>`) in an endless loop:
2000-token completions with exact usage counts that decode to almost no
visible text. The drafter predicts the loop perfectly, which is why
acceptance pegs at ~6.0 and throughput inflates. The instance stays poisoned
for every subsequent request (fresh prompts, short prompts after ~28 tokens)
until process restart. `exact: true` does not catch it: token counts are
honored; only the text is garbage.

### Trigger matrix (all cells: exact 1K/2K c16 loads on fresh disjoint text)

| GPUs | allreduce | graphs | spec | boots x runs | result |
| --- | --- | --- | --- | --- | --- |
| 0-3 (TP4) | custom | FULL cap64 | DSpark k=5 | 4 boots | degenerate by run 2, every boot |
| 0-3 (TP4) | NCCL | FULL cap64 | DSpark k=5 | 2 boots, 7 runs (incl. clean c1+c16+c32+c64 sweep) | clean |
| 0-3 (TP4) | custom | FULL cap64 | none | 1 boot, 2 runs | clean text |
| 0-3 (TP4) | custom, VLLM_DSV4_DEFER_TP_REDUCE=0 | FULL cap64 | DSpark k=5 | 1 boot, 3 runs | text clean, but acceptance collapses to 1.11 (~150 tok/s) - separate latent bug |
| 0-1 / 2-3 (TP2 pairs) | custom | PIECEWISE cap32 | DSpark k=5 | 1 boot each, 2 runs each | clean |
| 4-7 (TP4) | custom | FULL cap64 | DSpark k=5 | 3 boots, ~10 runs | clean |
| 4-7 (TP4) | NCCL | FULL cap64 | DSpark k=5 | 1 boot | degenerate at run 2 |

The last row disproves the custom-allreduce attribution the earlier rows
suggested (a profile mitigation was applied and then reverted the same
evening). The failure is a timing-sensitive race: flipping either the
allreduce implementation or the GPU quartet flips which configuration loses
the race. Every degenerate cell shares: TP4 + FULL_DECODE_ONLY capture-64 +
DSpark speculation + sustained c16+ verify load. TP2/PIECEWISE and no-spec
never degenerated (few runs; not proof). Hardware checked clean: zero ECC,
zero row-remap, no fresh Xids (GPU2/PCI 06:1B logged an Xid 13 warp
exception at 01:33 during earlier kernel dev; correlation with the first
failing quartet may be coincidence given the 4-7 NCCL failure).

Suspected mechanism: one NaN-class step emits token 0 via the sampler's NaN
guard; a BOS token in context self-sustains at temperature 0; the poisoned
sequence's KV blocks return to the pool unzeroed and the instance never
recovers, implying a pool/graph-replay reuse path that lets stale garbage
reach live sequences. The prime suspects are the spec-decode verify path
interacting with full-decode graph replay and the async aux streams
(indexer/MLA compressor overlap), not the reduce collectives themselves.

### Retractions and corrections

- Yesterday's deployed-sweep c16/c32/c64 rows (732.4 / 925.3 / 1023.8,
  acceptance 5.72-5.88) are retracted as capacity claims: acceptance within
  2% of the degenerate signature says those runs were substantially
  BOS-degenerate, on top of the sonnet.txt same-source confound. c1-c8 rows
  (acceptance 3.53-4.24) remain plausible.
- Same-source confound quantified separately: single-instance c32 on fresh
  diverse text = 666.6 tok/s at acceptance 4.52 vs 925.3 at 5.88 on
  overlapping sonnet windows.

### Valid capacity data (healthy acceptance 3.9-4.6, both instances under

### simultaneous load, per-instance cells)

| conc | per-instance tok/s (valid cells) | est. healthy box total |
| ---: | --- | ---: |
| 16 | 477-566 (B/cAR), 534-582 (A/NCCL) | ~1050-1150 |
| 32 | 635-686 | ~1300-1370 |
| 64 | 684-788 | ~1400-1550 |

No single sweep row yet has both instances simultaneously healthy end-to-end
(each sweep had one instance degenerate); the box totals are sums of valid
per-instance cells from different runs under equivalent contention, labeled
estimates. c1 per instance (NCCL boot, fresh text): 137.0 at acc 3.12 and
201.2 at acc 4.56 - content-dependent acceptance dominates c1 variance.

### Hardening landed

- `benchmarks/benchmark_dsv4_exact.py` now records per-request
  `chars_per_token` and hard-fails a run when any response decodes below
  0.5 chars/token, so a degenerate server can no longer post a record.
- Benchmark protocol: every run draws a never-served disjoint token region
  (window overlap and repeat-request caching both inflate acceptance).
- `slimserve-canary.timer` (5 min): long-prompt 60-token probe against both
  daemons; restarts an instance whose output decodes below 60 chars. The
  first (short-prompt) canary version missed partial degeneration - poisoned
  instances still emit ~28 good tokens on short prompts.

### Open (top priority, blocks perf work on this profile family)

1. Root-cause the race. Next discriminators: FULL->PIECEWISE graphs at TP4
   with spec (many runs); spec with turboquant draft KV -> auto draft KV;
   aux-stream overlaps disabled; zero freed KV blocks as a diagnostic.
2. The defer-off acceptance collapse (1.11) says the drafter consumes
   deferred-path state unconditionally; fix the fallback.
3. TP8 tiers use the same spec+full-graph machinery across all 8 GPUs;
   treat their qualification numbers as exposed to the same risk until the
   race is fixed.

Raw: `perf/results/2026-08-10/dsv4-degeneration-incident/` (51 files:
every benchmark JSON incl. degenerate runs, server logs for the manual
TP4/TP2/no-spec/defer-off boots).

## 2026-08-11 - Degeneration race: overnight bisection session 2

### Conclusions that DIED tonight (each looked solid on 1-2 boots)

1. "Custom allreduce is the culprit" - disproven yesterday (B stormed on
   NCCL), reconfirmed dead tonight.
2. "GPU 6 hardware fault" - {0,1,2,6} stormed while {0,1,2,7} was clean,
   twice, and GPU 6 sits on the PCI address that logged an Xid 13 warp
   exception during the 08-10 01:33 crash. But after a full driver+fabric
   reload the same {0,1,2,6} cell was clean AND instance A stormed on
   {0,1,2,3}. ECC/row-remap/NVLink counters clean throughout.
3. "Driver reload fixed it" - A stormed minutes after the reload.
4. "The eager fallback above capture-64 is the trigger" - NaN events
   occur at c10 (60 rows, graphed) and c11 (66 rows, eager) alike.
5. "persistent_topk (<=32-row dispatch) is the source" - forced-arm test
   INVERTED it: force-persistent boot clean (8 NaN lines), force-Filtered
   boot stormed (4,038 lines, 11/11 degenerate). But a later force-Filtered
   boot was clean and an aux-overlap-off boot stormed, so single-boot arm
   comparisons are worthless (see 7).
6. "Aux-stream overlap race" - VLLM_DSV4_INNER_ATTENTION_OVERLAP=0
   stormed.
7. THE ACTUAL INVARIANT SO FAR: whether a boot storms is decided at (or
   near) boot time - a boot lottery - and persists for that boot's
   lifetime. Configuration changes appeared causal only because each arm
   was sampled once. Storms never appeared at c1-c8 across ~50 runs; they
   appear on roughly half of boots under c11+ spec verify load. The
   original quartet-migration pattern was the same lottery sampled through
   daemon restarts.

### What survived scrutiny

- The failure chain: NaN logits row -> sampler guard emits token 0 -> BOS
  self-sustains at temp 0 -> request poisoned; poisoned requests recur as
  NaN rows in later batches; instance-wide collapse follows.
- DSpark speculation (next_n>1) is required; no-spec and TP2/PIECEWISE
  cells never stormed (limited n, but consistent).
- Kernel-level A/B: both persistent_topk and FilteredTopK pass a hostile
  unit test (5,760 rows, NaN/inf/3e38-poisoned tails beyond each row's
  length, lengths 513-3100, rows 1-96) - the top-k selection logic is
  sound against uninitialized-tail input in isolation.
- Serving-side rare NaN events (1-3 per boot) cluster at SMALL batches
  (7-11 rows, ramp/drain) even on clean boots.

### Instruments now landed (all default-off, env-gated)

- VLLM_NAN_WATCH=1: async per-step NaN-row tripwire on target logits in
  both model runners (V2 `vllm/v1/worker/gpu/model_runner.py` is the one
  serving actually uses). One small reduction per step, detection lags one
  step, no hot-path sync.
- VLLM_DSV4_TOPK_VALIDATE=1: same pattern at the corruption source -
  decode top-k indices checked against per-row seq_lens in
  sparse_attn_indexer; an out-of-range index means sparse attention will
  gather garbage KV.
- VLLM_DSV4_FILTERED_TOPK_MIN_ROWS: env-tunable dispatch threshold in
  csrc/libtorch_stable/topk.cu (default 32 = production heuristic; 0
  forces FilteredTopK, huge forces persistent) for forced-arm testing.
- Both production daemons run with VLLM_NAN_WATCH=1 permanently.

### In flight

Six-boot campaign (production config, dual tripwires, 2x c11 trigger runs
per boot) for powered per-boot statistics: storm rate, and whether
TOPK_VALIDATE violations precede NAN_WATCH events inside storm boots.
Raw: `perf/results/2026-08-10/dsv4-degeneration-incident/` plus
`/var/log/SlimServe/ss-camp*.log`.

### 2026-08-11 - Campaign results, FULL arm (6 boots, production config)

| boot | degenerate reqs | NaN lines | top-k violations |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 2 | 0 | 8 | 0 |
| 3 | 0 | 8 | 0 |
| 4 | 22/22 | 1348 | 0 |
| 5 | 22/22 | 1344 | 0 |
| 6 | 0 | 4 | 0 |

Storm rate 2/6. Two decisive facts:

1. ZERO top-k violations in 6 boots INCLUDING both storms: every decode
   top-k index stayed within [-1, seq_len). The corruption does not enter
   through out-of-range indexer indices. The NaN arises in the value
   path - attention gather/reduction over cache contents, MoE, or the
   mHC/Q8 pipeline - consistent with reading unwritten pool memory
   (split-K partials or unwritten KV slots), which is also the only
   mechanism found so far that naturally produces a per-boot lottery.
2. Storm onset is CONTAGION: both storms ignite from a 1-row event and
   spread to 54/56 rows within two steps. A NaN crosses requests almost
   instantly, implicating shared state: APC-shared prefix blocks (the
   trigger loads use overlapping windows), the KV block pool, or a
   step-global buffer. Clean boots show the identical warmup-phase burst
   (1/8 at step 6, 6/14 at step 7 - byte-identical on camp2 and camp4)
   and recover, so the warmup burst alone does not decide the storm.

Next instruments: per-row request-id logging on NaN events (to trace the
contagion path), and a NaN probe at attention-output vs MoE-output to
bisect the layer pipeline. PIECEWISE arm in flight.

### 2026-08-11 - Row-index fingerprints: the seam pattern

With NaN row indices logged, the rare events stop looking random:
`1/7 rows=[5]` at step 6 AND at step 545 (same boot, pw2), followed once
by `6/14 rows=[0-5]` (the first request's entire verify window one step
later). A 7-row decode batch is a request seam - one request with 6
spec-verify rows plus one with a single row - i.e. mixed decode_lens,
the `requires_padding` pack_seq_triton path in sparse_attn_indexer. The
FP8 branch packs Q with NO pad_value (the MXFP4 branch was given
pad_byte=0 precisely "so padded slots dequantize to 0 and can't produce
NaN/Inf in the logits kernel" - the FP8 branch kept the assumption that
"downstream context_lens masks stale slots"). Uninitialized fp8 pad
bytes can encode NaN, and whether the recycled allocator memory behind
the pad slots is hostile is decided per boot - the first mechanism
found that produces BOTH the positional determinism (last row at a
seam) and the boot lottery. PIECEWISE boots show the same rare seam
events without storms so far; the storm-contagion step remains to be
traced (APC-shared blocks / pool reuse suspected).

Next concrete reproducer: two staggered requests forming the 6+1 batch
shape, with seq_lens / packed-Q / unpacked-index instrumentation at the
seam; and give the FP8 pack a pad_value=0 like the FP4 branch, then
re-run the FULL-arm campaign as the fix candidate.

### 2026-08-11 - Seam theory corrected

Three corrections from direct testing:

1. torch's fp8e4m3fn cast SATURATES +-inf to +-448 (verified on device),
   so pack_seq_triton's default -inf pad is not a NaN generator via the
   torch cast.
2. More decisively: Triton on sm80 cannot compile _pack_seq_kernel for
   fp8e4m3 AT ALL ("type fp8e4nv not supported in this architecture") -
   the requires_padding decode path would crash the server if it ever
   executed, and it never has. That path is dead code on A100 serving:
   decode batches are always uniform next_n; short requests route
   through the prefill chunks instead.
3. Therefore the 7-row NaN batches (rows=[5] fingerprint) are
   decode+PREFILL mixes: one spec request's 6 verify rows plus a
   prefilling request's final chunk row. The suspect surface moves to
   the decode/prefill interaction inside sparse_attn_indexer - the
   shared topk_indices_buffer and gather workspace partitioning - or
   the mixed-batch handling in the sparse MLA forward itself.

The two seam fixes landed anyway as prophylactics (per-request
decode_len anchoring + zeroed pad lengths + pad_value=0), all no-ops
for the workloads this box actually runs. Next reproducer: one
long-prefill request submitted mid-decode of one spec request, dual
tripwires armed - the minimal decode+prefill mix.

### 2026-08-11 - Campaign verdict and production mitigation

PIECEWISE arm: 0 storms / 6 boots (rare seam events on 5 of 6 boots,
0-16 NaN lines, zero top-k violations, zero degeneration). FULL arm:
2 storms / 6 boots. Honest statistics: 2/6 vs 0/6 alone is Fisher
p~0.23 - not significant by itself. Pooled with the full incident
record (every storm across the 08-10/08-11 investigation was a
FULL_DECODE_ONLY boot, ~10+ storms across ~20 such boots; zero storms
in any PIECEWISE or TP2 boot ever), the direction is strong, and the
QuixiCore sparse-MLA builder's own TODO independently says FULL graphs
need buffer-persistence work this backend never received.

Clean-run throughput cost of the mitigation (solo c11): 463.1 -> 409.1
(-12%). Under production dual-instance c16 load the cost is smaller:
A 530-534 / B 488-490 tok/s, healthy acceptance 4.35-4.45, both
instances clean, zero NaN events on fresh boots.

MITIGATION APPLIED: all four A100 dsv4 tiers (q4ktail-4/8, mxfp4-4/8)
now serve PIECEWISE capture-32 (profile notes reference this entry).
The capture-64 c8 gains recorded on 2026-08-10 (UPDATE 3) are
deliberately given back until the race is fixed. Both daemons
restarted on the mitigated profile and verified.

OPEN root-cause work (P0, task #4): (a) the rare seam NaN events -
positionally deterministic (last verify row at decode+prefill mixed
batches, rows=[5] of 7) - occur under BOTH graph modes and remain
unexplained; reproduce with one long prefill submitted mid-decode of
one spec request, tripwires armed, then bisect attention-out vs
MoE-out. (b) The storm contagion (1 row -> 54/56 rows in two steps)
appears FULL-graph-gated; trace with per-request NaN row logging once
(a) is understood. (c) The ROCm-parity buffer-persistence work in
QuixiCoreMLASparseMetadataBuilder is the condition for re-enabling
FULL graphs, followed by a clean 6-boot tripwire campaign.
## 2026-08-10 - Metal dsv4-xxs-1 256K Resize, Path Repairs, Indexer Blocker

- Status: profile resized and partially validated; four retained fixes; one
  retained diagnostic gate; long context blocked on a missing Metal kernel
  family. Not a performance change; no throughput claims.
- Baseline: the Metal profile served max_model_len 3072 with a fixed 1 GiB KV
  pool, tuned for the exact 1k-in/2k-out matrix (33.684 tok/s c1, 35.350 c8
  in perf/baseline_status.md). Real (agentic) use needs 256K context, and the
  benchmark-shaped cap had hidden that most of the long-context path had never
  executed on Metal.
- Change (profile): metal override of dsv4-xxs-1 now sets max_model_len
  262144 and kv_cache_memory_bytes 17179869184 (16 GiB). Planner verifies
  2,225,562 logical KV tokens (~8.5x one full 256K request; ~7.7 KB per
  logical token all-in for the 43-layer fp8_ds_mla target cache plus indexer
  and 3-layer TurboQuant draft cache). Weights 80.8 + 6.5 GiB plus the pool
  is ~103 GiB against the ~115 GiB working set of the 128 GiB M5 Max.
- Found and fixed (all latent breakage from the A100 prequant/sampler work
  landing without the Metal path being re-exercised; current main crashed on
  the FIRST request at any context):
  1. DeepseekV4MetalAttention.forward lacked the new prequant_input
     parameter (vllm/models/deepseek_v4/metal.py) -- accepts and forwards.
  2. attn_gemm_parallel_execute called torch.cuda.is_current_stream_capturing
     unconditionally (attention.py multi-stream enable) -- raises on MPS
     builds; guarded with current_platform.is_metal() so ROCm (HIP torch.cuda,
     real graph capture) keeps exact behavior.
  3. DeepseekV4TurboQuantDraftAttention.forward (amd/dspark_turboquant.py)
     had the same stale signature -- accepts and ignores, per the module's
     existing del pattern.
  4. rejection_sample's MPS branch was greedy-only; any temperature > 0 fell
     through to a nonexistent Triton kernel. Added a torch implementation of
     full rejection sampling (ratio accept, residual-mass resample, bonus
     token; sample-and-match when draft logits are absent), following the
     gumbel_sample MPS precedent (torch randomness; no Philox seed parity).
     Consequence: Metal spec decode only ever worked for greedy harness
     traffic; a default-temperature client always crashed.
- Diagnostic gate (retained until measured): turboquant_attn.py's new
  _MAX_SLIDING_WINDOW_KV_SPLITS clamp (32 -> 16 on sliding-window layers) is
  gated off on Metal; the QuixiCore Metal decode kernel is only validated at
  32 splits, and the clamp landed with the A100 work between the last-good
  Metal commit (209265933) and now. Untested as the spec-hang cause; see open
  items.
- Correctness result (no-spec, fresh boot, port 8077): cold default-temp
  smoke returned exactly the requested string after 1651 s (one-time cold MPS
  pipeline compile; ~28 min); warm greedy '42' in 28 s; warm temp-0.7 in
  172 s. All with correct content and token counts.
- Long-context result: a ~25K-token needle prompt crashed the engine inside
  its FIRST 2048-token prefill chunk (prompt_tokens_total never advanced):
  fused_indexer_q_rope_quant launches a Triton kernel, and Metal has no
  Triton. The Triton call predates the A100 work -- the sparse-MLA indexer
  chain (fused_indexer_q_rope_quant, _fill_short_context_topk_indices, the
  compressor's indexer-cache insert ordering, and indexer_op fp8 scoring +
  topk) has NEVER had a Metal implementation. The old 3072 cap kept every
  request below the indexer engagement length, which is why the 1k/2k
  baseline never saw it. This is the blocker for 256K on Metal: port the
  indexer chain from the ROCm reference into csrc/quixicore metal serving.
- Open item (spec decode): with fixes 1-4, the first spec verify step stalled
  >= 35 min at 13 prompt + 1 generated tokens, main thread blocked in
  .cpu() -> MPSStream sync -> MTLCommandBuffer waitUntilCompleted (native
  stack in raw artifacts). The subsequent no-spec run showed cold compile
  alone is ~28 min, so the stall may have been cold compile of target plus
  drafter pipelines rather than a true deadlock. Re-test spec decode with the
  32-split gate in place and a >= 60 min first-request budget before
  concluding deadlock. Note: vLLM retitles processes (VLLM::APIServer /
  VLLM::EngineCore); kill by port owner, not by "slimserve" pattern, or a
  zombie API server holds the port with a dead engine.
- Decision: retain the profile sizing and fixes 1-4; retain the splits gate
  as a documented diagnostic; do not claim 256K serving until the indexer
  chain runs on Metal and the spec path is re-validated. Next commands:
  port fused_indexer_q_rope_quant + indexer_op to the Metal QuixiCore
  serving lib (reference: ROCm path + csrc/quixicore/tm_rocm), then
  `slimserve dsv4-xxs-1 --serve -y` and rerun the staged validation with the
  25K needle and a genuine 200K+ request.
- Raw artifacts: perf/results/2026-08-10/dsv4-metal-256k-real-use/
  (serve logs 1-6, no-spec log, native stack sample of the stalled step,
  staged validation transcript).

## 2026-08-10 - Muse-Glimmer-30B: New Model Family Bring-Up On Metal (muse-kdyn-1)

- Status: text serving VALIDATED end to end on Metal with DFlash speculation
  and the muse reasoning parser; vision executes end to end but produces
  spatially/chromatically scrambled descriptions (in progress); profile
  registered; two Metal gaps closed on the way.
- Scope: meta-models/Muse-Glimmer-30B-GGUF (kquant-dynamic default,
  kquant-17gb alternative, mmproj vision tower, DFlash drafter). Arch was in
  NO local stack (fork, upstream vLLM, mainline llama.cpp). References used:
  transformers commit fe95f5423d (merged 2026-08-10) and llama.cpp PR 26841
  (fetched as ~/llama.cpp branch muse-pr; llama-completion built in
  build-muse/ and used as the working reference).
- New support written (all in this repo):
  - Config/tokenizer: gguf_muse_glimmer.py builders (text+vision+dflash),
    parser dispatch by `dflash.expert_count` presence, llama4 (GPT-4o)
    pretokenizer split, MuseGlimmerConfig.
  - Models: muse_glimmer.py (dense 52L, GQA 32/2, QK-norm with GGUF-baked
    qk_scale 3.87, sigmoid-gated attention, sandwich norms with
    post_norm_eps 1e-8, [local x3, global] pattern, RoPE theta 500k
    INTERLEAVED (llama.cpp NORM type; the conversion un-permutes HF Q/K) on
    local layers only, NoPE globals, scale-then-softcap logits, and the
    WEIGHTLESS EMBEDDING RMSNORM from llama.cpp muse-glimmer.cpp:74 -- the
    final root cause of degenerate output; raw embedding RMS is ~0.06);
    muse_glimmer_dflash.py (stock DFlashQwen3 shape + torch-native context-KV
    precompute for MPS + shared target embed/lm_head); vision tower +
    3-linear projector + single-tile 896 processor in muse_glimmer.py.
  - Loader: MuseGlimmerGGUFAdapter (+dflash variant) with explicit name
    maps; embedding table dequantized to fp16 at load (~2.7 GiB) because
    Metal has no generic GGUF dequant kernel (binding the vendored
    dequant_gather shader is the follow-up).
  - Reasoning parser `muse_glimmer` for the Harmony-like
    `<|start|>assistant to=self<|message|>` format.
- Metal gaps closed:
  - Sliding-window attention wired: the vendored paged_attn_v2 shader
    already had `window` support; the binding hardcoded 0. Extended
    qc_metal_serving.mm + ops.py + metal_attn.py (kernel window on decode,
    banded mask + window-clamped gather on SDPA; gather also stops reading
    hybrid-manager-freed blocks)._quixicore_C rebuilt.
  - METAL_ATTN added to AttentionBackendEnum (first plain-Attention model on
    this backend).
  - apply_temperature torch fallback for MPS (sampler stage DSV4 never hit).
  - Multimodal pin_memory disabled under MPS at the reduce_data choke point
    (pin_memory() raises storage-device mismatch on the MPS backend).
  - Profile guards: kv_cache_dtype MUST be "auto" (fork default fp8_e4m3
    has no Metal path); gpu_memory_utilization null at base level breaks
    slimserve arg rendering (removed).
- Debug method that found the embedding-norm root cause: tokenize parity
  (identical ids), Q5_K/Q6_K/Q4_K qgemv parity vs numpy dequant (cos=1.0,
  kernels exonerated), layer-0 tensor-sum parity vs llama-eval-callback
  (matched), then graph diff exposed `embd_norm`. Offline greedy now
  produces 'The capital of France is Paris. ...'.
- Serving validation (port 8078, spec on): cold 'muse glimmer ok' (21 s incl
  first compile), warm greedy '42' 7 s, temp-0.7 runs (reply consumed by
  reasoning channel under small max_tokens -- parser behavior, not a bug).
  Image request: 1092 prompt tokens (68 text + 1024 image, single-tile
  geometry as designed), completes without crash.
- OPEN (vision correctness): red-bg/blue-circle test image is described with
  wrong hues/shape ('blue, green, yellow irregular shape'). Numerics are
  alive (pixel channels correct into the tower, outputs position-varying,
  projected std 1.02 vs text 0.06 -- rebalanced by the embedding norm).
  Suspect patch-embed weight axis order or the 2x2 merge/position-embed
  geometry vs llama.cpp's clip implementation. Next: diff tower layer-0
  against llama.cpp mtmd on the same image; test BGR/axis-flip hypotheses.
- OPEN (minor): spec_decode_num_drafts_created metric emits a unix
  timestamp; drafter acceptance-length not yet measured; temp-0.7 needs a
  larger-budget clean check; dynamic-aspect (smart_resize) preprocessing
  still single-tile square; TurboQuant draft KV not configured.
- Perf snapshot (not tuned): warm greedy short reply ~7 s wall including
  ~50-token reasoning channel. No exact-token benchmark run yet; README
  reference for this box: 26.6 tok/s no-spec / 50.2 with DFlash (llama.cpp/
  ExecuTorch measurements).
- Raw artifacts: scratchpad muse-serve.log, muse_probe.py, muse_variants.py,
  muse_layer0.py, muse_qparity.py, muse_vision_probe.py, muse_mm_gen.py;
  llama.cpp reference at ~/llama.cpp (branch muse-pr, build-muse/bin).

## 2026-08-10 - Muse-Glimmer Vision Root-Caused And VALIDATED (muse-kdyn-1)

- Status: text AND image serving validated end to end through the OpenAI API
  on Metal with DFlash speculation and the muse reasoning parser. The
  registered profile is live.
- Vision root cause (via llama.cpp PR 26841's clip implementation,
  tools/mtmd/models/muse-glimmer.cpp + clip.cpp set_input): the tower is NOT
  a plain ViT. It uses (a) 2D RoPE theta 10000 on every layer -- first half
  of head_dim rotated by 1-indexed width position, second half by height,
  GPT-J interleaved pairs per half; (b) sparse block-diagonal WINDOW
  attention over 32x32-patch windows with every 4th (1-based) and the last
  layer global; (c) window-order permutation around the block stack; and
  (d) a channel-outer pixel shuffle: the 6144 projector input is
  [c0s0..c0s3, c1s0..] (spatial fastest), the transpose of the naive
  per-patch concat. Before these, solid red and green were both described
  as "olive-brown"; after, red/green/blue solids are named correctly and
  the red-bg/blue-circle image is described as "A blue oval sits on a red
  background" (oval, correctly: the single-tile square resize stretches the
  4:3 test image; llama.cpp's aspect-preserving smart_resize is the noted
  follow-up).
- Server validation (staged, port 8078): cold 'muse glimmer ok' 22 s
  (first-compile inclusive), warm greedy '42' 7 s, image reply 21 s at 1092
  prompt tokens (68 text + 1024 image). No engine errors across the run.
- Remaining open items (unchanged from previous entry): exact-token
  performance benchmark (README reference for this box: 26.6/50.2 tok/s),
  spec acceptance-length measurement + the timestamp-poisoned
  spec_decode_num_drafts_created counter, smart_resize dynamic-aspect
  preprocessing, TurboQuant draft KV, ATEM tool-call parser.

### 2026-08-11 - Mixed-batch reproducer: negative on one boot

Two phases against one PIECEWISE boot with all three tripwires plus the
new per-layer birth probe (VLLM_NAN_WATCH_LAYERS): (1) one continuous
spec-decode stream + 160 sequential 24K-token chunked prefills (the
exact 6+1 batch shape of the historical rows=[5] events); (2) four
staggered spec-decode streams + 115 more prefills (7-35-row shapes).
~70 minutes, zero NaN events, zero top-k violations, no degeneration.

Interpretation is deliberately weak: this is ONE boot of a bug whose
expression is decided per boot - the same trap that killed six theories.
The mixed-batch shape hypothesis is unproven, not disproven. Rather than
burn more boots on the rare-event hunt in isolation, the 8-boot
FULL-graph validation campaign for the bt_per_token persistence fix
(commit 0a3163da6) doubles as the rare-event sampler: every boot runs
the birth-layer probe, so any event that fires during validation names
its layer for free. Success criterion: 0/8 storms (pre-fix FULL stormed
2/6) keeps FULL graphs on the table for restoration; any storm sends
the campaign logs - now with per-layer birth data - into the next
analysis round.

### 2026-08-11 - Birth attribution: layer 0 input; kernel exonerations

Intra-layer probes (vfix boots 1-3, two independent boots pre-refinement
plus one with phase slots) place the NaN at LAYER 0'S ATTENTION INPUT
(slot 100) with attention and MoE outputs clean - the attention guards
and Q8 quantization launder NaN to zeros while the residual stream
carries it to the head. Births are exclusively at <=8-row steps
(rows=[5] of 7, 1/8); the 11-35-row sightings are carry-forward.
The <=8-token gate exactly matches _norm_with_prequant's fused
ggml_dsv4_rms_norm_q8_1 path.

Hostile unit tests then EXONERATED each isolated component of the
layer-0 input chain on this GPU: dsv4_mhc_pre (70 cases x 3 outputs,
NaN/inf/huge pools, zero/tiny rows), ggml_dsv4_rms_norm_q8_1 (192
hostile-pool cases, T=1-8), plus persistent_topk and FilteredTopK
earlier (5,760 rows each). torch.index_select bounds-asserts, so a
stray -1 token id cannot silently junk-read the GGUF embedding. The
birth is therefore in INTEGRATION - graph replay, TP collectives, or
the aligned-Q8/prequant interplay - not in any kernel in isolation.

Method upgrade: the step-6 warmup event fires on most boots, so short
runs (300 output tokens) make single boots informative at ~4 min/boot.
Queued arms (3 boots each): baseline env, ALIGNED_Q8=0,
PREQUANT_ATTN=0, AUX_STREAMS=0. Note the prior "env-less boots show
zero events" observation is VOID - those boots predate the V2-runner
tripwire hook; no instrumented env-less boot exists yet.

### 2026-08-11 - vfix campaign verdict: 8 boots, 0 storms

FULL_DECODE_ONLY capture-64 WITH the bt_per_token persistence fix
(commit 0a3163da6): 8 boots, identical trigger protocol to the pre-fix
campaign, ZERO storms (pre-fix: 2/6). Fisher p~0.17 alone; combined
with the fix being mechanism-directed at the builder's documented
persistence gap, the storm path is plausibly closed. Rare seed events
persist unchanged (~2 per boot, 5 of 8 boots, always <=8-row births,
zero top-k violations) - consistent with the fix addressing storm
amplification, not the seed. The documented bar for restoring FULL
graphs (persistence work + clean 6-boot campaign) is met; restoration
decision deferred until the env arms attribute the seed, since the
same boots serve both purposes.

### 2026-08-11 - FULL graphs restored to production; seed campaign phase-blocked

Env attribution campaigns: the short-run pilot (base arm only, 1 event
in 3 boots) showed the 300-token protocol is underpowered; the focused
full-length campaign (4 boots baseline vs 4 boots VLLM_DSV4_ALIGNED_Q8=0,
interleaved) ran 8/8 boots with ZERO events in either arm - the box
entered a dormant phase mid-investigation (the same hours-scale drift
seen throughout). Uninformative by design rather than misleading; the
runbook in perf/diagnostics/dsv4-nan/README.md says to fire
campaign_env2.sh when production NAN_WATCH shows events again.

Production decision: the documented restoration bar (bt_per_token
persistence fix + clean multi-boot FULL campaign) is met - 0/8 storms
vs 2/6 pre-fix. All four A100 dsv4 tiers are back on FULL_DECODE_ONLY
capture-64 (+12% c11 over PIECEWISE); both daemons restarted and
NAN_WATCH stays on permanently, with the canary as the last line.
The rare sub-critical seed (single-token quality blips, both graph
modes, layer-0-input births at <=8-row batches, all kernels exonerated
in isolation) remains OPEN at reduced priority now that storms are
closed and detection is standing.

Cross-stream test repairs (Metal session collisions on main): muse-glimmer
added to the supported-sources set; dsv4 metal 256K resize reflected
(max_model_len 262144, 16 GiB KV); the dspark/TQ invariant now reads
each source's registered speculator method (muse uses DFlash k=16);
tool_call_parser required for all families except muse-glimmer, which
has no tool parser in this fork (auto tool choice no-ops without one).

### 2026-08-12 - Seed attribution: active-phase arms + id evidence

Three-arm discriminator during an active phase (3 boots each,
interleaved, full-length runs): async-mHC baseline 6 events, monolithic
mHC schedule 6, NCCL-only collectives 2. All arms fire - the mHC
async-split machinery and the custom allreduce are BOTH exonerated as
required components of the seed.

Input-id probe evidence (first captured birth with ids): the birth
row's input token is a NORMAL id (4042) in a batch of normal ids -
the OOV/zero-embedding theory is dead; the NaN is computed, not fed.
One step later the poisoned request's whole verify window carries
input ids [0,0,0,0,0,0]: the sampler guard emitted token 0, the
drafter drafted BOS five deep - the within-request contagion loop
observed directly. And rows=[5] holds across batch sizes 7, 8, AND 11:
a fixed memory offset, not a shape tail - consistent with a scribbled
allocation (cross-stream caching-allocator hazard) at a per-boot-fixed
address, which would also explain the boot lottery and why every
kernel is clean in isolation.

In flight: loop-until-capture with the full probe suite (input ids,
slot 98 = raw embedding output, slot 99 = post-hc-expand, slot 100+ =
intra-layer). Slot 98 lit => corruption precedes all layer math
(allocator/scribbler hunt); 98 dark + 100 lit => born in hc_pre/norm
under real weights (offline repro with dumped weights).

### 2026-08-12 - Full-probe capture: the values are innocent

Loop-until-capture landed a birth on its first run with the complete
suite. Verdict chain:

- Slot 98 (raw embedding output): CLEAN. Slot 99 (post hc-expand):
  CLEAN. Slot 100 (attention input, after hc_pre + norm): NaN, row 5,
  input id 989 (normal token).
- Offline exact replay of that batch - REAL layer-0 hc weights
  extracted from the GGUF (fp16 fn, scale [2.077, 0.0195, 0.238],
  base range -30.1..9.9), REAL embedding rows of the exact captured
  ids, REAL config (sinkhorn_iterations=20, eps=1e-6) - is
  BIT-IDENTICAL to the fp32 torch reference: maxdiff 0.0, zero NaN.
- Also exonerated offline: the fp16-fn template (630 outputs), the
  poisoned-allocator regime for the op's internal buffers (210 calls),
  20-iteration/1e-6 sinkhorn (216 outputs).

Conclusion: the deterministic math on the true inputs is clean, so the
serving buffer feeding hc_pre did not hold the true values at compute
time - a CONCURRENT WRITER modifies a main-stream tensor in the window
between adjacent kernels (slot-99 probe read clean; hc_pre/norm's
input or output was then scribbled before slot-100's read). Every arm
tested so far (async/mono mHC, custom-AR/NCCL) shared one async
machinery: the attention aux streams (VLLM_DSV4_AUX_STREAMS, default
on, three streams driving attn_gemm_parallel_execute overlap) - never
isolated with full-length instrumented runs. Aux on/off campaign in
flight (4 interleaved rounds).

### 2026-08-12 - record_stream fix is not the seed; graphs arm running

vfx2 (fixed helpers, aux on) fired with the identical fingerprint
(rows=[5], normal input id 343, slots 98/99 clean, 100 NaN). The
multi-stream record_stream repair stands as a genuine hazard-class fix
but is NOT the seed mechanism, and the aux on/off 2-2 split is demoted
to lottery-suspect (n=2 per arm).

Revised leading theory: under FULL graphs everything shares the
capture-time graph memory pool, and BOTH observed birth sites (the
hc_pre/norm window at layer 0, and inside layer-2 attention at slot
109 - present in two independent captures) would follow from ONE
missing cross-stream event edge during capture letting the pool alias
two live buffers. Replay then scribbles at a pool-layout-fixed offset:
the boot lottery IS the capture-time pool layout, and record_stream is
irrelevant inside graph pools. Every seed event observed with step
data sits on a replayed decode step (6, ~540). Discriminator running:
FULL vs NONE (all-eager) interleaved, 3 rounds, active phase.

### 2026-08-12 - Graphs eliminated; aux-off at 3/3 silent

gnone1 (cudagraph_mode NONE, fully eager target) fired with the
standard fingerprint: CUDA graphs are NOT required - the graph-pool
aliasing theory dies alongside its predecessors. Note the drafter kept
its own FULL graphs in that boot (dspark captures independently), and
spec decode itself remains never-seed-tested, as does ALIGNED_Q8 in an
active phase - the final arms campaign covers exactly those two plus
baseline, interleaved.

The straggling aux campaign delivered auxoff3 silent: aux-off is now
3/3 silent vs unfixed aux-on 2/2 firing. Since vfx2 showed the
record_stream (returned-tensors) repair does not stop the seed, if
aux-off genuinely suppresses it the channel is aux-stream-adjacent but
in another direction (inputs, or the shared workspace manager regions
consumed from both main and aux streams). Aux-off also stands as a
candidate production mitigation at the cost of the attention-overlap
perf if the final arms leave the mechanism unresolved.

### 2026-08-12 - Seed endgame: overlap disabled, faster AND silent

Final arms (active phase, baseline firing every round): no-spec is
CONFOUNDED (routes through the legacy model runner entirely - and
produced whole-batch NaN storms, a separate expression to investigate
on that runner) and ALIGNED_Q8=0 FIRES (6 events) - exonerated. The
elimination matrix is complete: the ONLY component whose removal
silences the seed is the multi-stream attention projection overlap
(VLLM_DSV4_AUX_STREAMS=0: 3/3 boots silent in active phases).

Structure finally explains the fingerprints: the overlap enables at
<=VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD (1024) tokens AND not during
capture - under FULL graphs, pure decode steps replay captured SERIAL
projections while mixed decode+prefill seam steps run EAGERLY with the
overlap live; those eager seam steps are exactly where every FULL-mode
birth sat. The race itself resists isolation: the GEMV op is canary-
clean in isolation, and a 20k-iteration standalone execute_in_parallel
repro with the real ops never fired - TP4 collectives/drafter/allocator
churn are needed, so root cause remains open at the concurrency level.

DEPLOYED: VLLM_DSV4_AUX_STREAMS=0 on all five A100 dsv4 tiers. Measured
FASTER with it off: clean-run c11 means 458.1 (aux-off, n=4) vs 426.7
(aux-on, n=12) - the overlap was a net loss at seam widths anyway.
Open root-cause thread: extend overlap_repro.py with TP + drafter
interleave; the no-spec legacy-runner whole-batch NaN storm.
## 2026-08-11 - Muse-Glimmer Optimization Pass 1: +75% no-spec decode

- Status: two retained fixes, one retained tuning decision, bottlenecks for
  the next pass measured and named.
- Method: 256-token greedy decode via the API (78-token prompt, 3 runs after
  a warmup, wall-clock over exact completion_tokens; harness and raw runs in
  perf/results/2026-08-11/muse-optimization-pass-1/). In-process layer-chain
  and per-op timings with torch.mps.synchronize fences.
- Baseline (pre-pass): spec-on 7.65 tok/s, no-spec 7.34 tok/s. Speculation
  was a wash. Reference for this box (llama.cpp/ExecuTorch, model card):
  26.6 no-spec / 50.2 with DFlash.
- Profiling: a decode step spent 104.8 ms of its 137 ms inside the 52-layer
  matvec chain. Per-op timing pinned the anomaly: qkv_proj took 447 us for
  ~17.5 MB of weights (39 GB/s) while every homogeneous projection ran at
  ~430-450 GB/s.
- Fix 1 (retained): GGUFLinearMethod.apply sliced the padded merged buffer
  with .contiguous() on EVERY call for mixed-quant-type merged layers --
  a per-forward device copy of the quantized bytes. Muse hits this on 24/52
  QKV layers (Q5_K+Q6_K) and 31/52 gate_up layers (~6.3 GB of copies per
  step). Now the contiguous per-shard views are materialized once and the
  padded buffer's storage is released (net-zero memory; the parameter object
  and shard maps are kept). qkv_proj 447 -> 88 us; mixed gate_up back to
  ~433 GB/s. DSV4/GLM are unaffected (homogeneous or dsv4-aligned paths).
- Fix 2 (retained): single-dispatch bf16 RMSNorm bound from the vendored
  QuixiCore Metal kernels (rms_norm / rms_norm_dyn) and wired via
  RMSNorm.forward_mps (~20 us/call, max_abs_err 0 vs native fp32 reference
  at D=6656 and D=128). Also retained: fused paged-attention batch expansion
  for uniform multi-query decode (spec verify 17 rows, draft 16), replacing
  the per-request SDPA gather loop (+3.5% alone).
- Result: no-spec 7.34 -> 12.78 tok/s (+74%, step 137 -> 78 ms); spec-on
  7.65 -> 9.76 tok/s (+28%).
- Tuning decision: k stays 16. DFlash acceptance measures 1.68 accepted per
  draft (73% pos-0, decaying to 6% by pos-4; mean 2.68 tokens per step).
  k=5 was tried and is WORSE (7.3 tok/s at identical acceptance), matching
  the measured small-M anomaly in the GGUF matmul path (a 16-row chain is
  slower than a 17-row chain: 293 vs 258 ms). Speculation is currently
  net-negative vs no-spec (9.76 vs 12.78); the profile notes recommend
  --no-spec for single-stream throughput until the small-M path is fixed.
- Next bottlenecks (measured, in order):
  1. Small-M (2..32 rows) GGUF matmul: a 17-row layer chain costs 2.46x a
     1-row chain when the incremental cost should be near-zero at weight
     bandwidth. This is what keeps DFlash net-negative; llama.cpp turns the
     same drafter into +89%.
  2. The remaining 1-row chain gap: ~78 ms step vs 37.6 ms reference =
     per-op dispatch spread across ~10 ops/layer plus ~20-30 ms engine
     overhead. Candidates: k-quant fused qkv/up-gate kernels (the vendored
     fused variants are q4_0-only today), rope+qk-norm fusion, sampler path.
- Raw artifacts: perf/results/2026-08-11/muse-optimization-pass-1/.

## 2026-08-11 - qgemv_mm: Weight-Stationary Small-M GGUF Matmul For Metal

- Status: kernel family retained (verified at kernel and chain level); the
  speculative path remains net-negative end to end -- the residual cost is
  now host-side, named below.
- Hypothesis: the small-M band (2..32 rows -- speculative verify k+1=17,
  draft blocks 16, small decode batches) was served by a per-row qgemv
  (linear in M: weights re-read per row) or a fragment GEMM measured
  4-5x off weight bandwidth (flat ~99 GB/s). A weight-stationary multi-row
  GEMV should hold the vec kernel's ~450 GB/s across the band.
- Change: `qgemv_mm<FMT, T, M>` in the vendored qgemv.metal -- same
  block-major walk and lane geometry as qgemv, each dequantized 8-wide span
  held in two float4 registers and applied to all M rows via two vec4 X
  loads per row (the first scalar-load version scaled poorly: load-issue
  bound, and M=32 spilled registers -- instantiations cap at M=17).
  Instantiated for q4_0/q8_0/q4_K/q5_K/q6_K x half/bf16 x M in
  {2,4,8,16,17}; the binding greedily decomposes larger batches and the
  routing (`_mmvq_batch_limit`) sends M<=17 to the vec route on Metal
  (32 was tried and regressed prefill: decomposed multi-pass loses to the
  flat GEMM above the single-dispatch band).
- Kernel result (real muse weights, M=17 vs the prior per-row loop):
  o_proj Q4_K 656 -> 169 us; gate Q5_K 839 -> 186 us; down_proj Q6_K
  3978 -> 1059 us. Beats the fragment GEMM at every M in the band. Parity
  vs the GEMM kernel < 2e-2 relative across M including the decomposition
  path (M=33).
- End-to-end (256-token greedy, back-to-back same-session): no-spec
  11.8-12.3 tok/s (yesterday's 12.78 was measured on a quieter machine;
  runs now drift within triplets, ambient load ~3.6), spec-on k=16
  8.6-9.2 tok/s with acceptance 1.87/draft (2.87 tokens/step).
- Analysis: with verify matmuls now ~2x cheaper, the spec step still costs
  ~320 ms against ~85 ms plain decode. The model-side delta accounts for
  under half of it; the remainder is host-side spec machinery -- the MPS
  torch rejection sampler (per-request loops + .cpu() syncs), the
  torch-native DFlash context-KV precompute (per-layer eager ops each
  step), and proposer bookkeeping. Speculation stays net-negative on Metal
  until that path is addressed; the profile note stands (--no-spec for
  single-stream throughput).
- Next: (1) batch the MPS rejection sampler to one device pass + one sync
  per step; (2) fuse the dflash context-KV precompute (the vendored
  qk_norm_rope kernel family is a candidate); (3) engine-side step
  overhead (~20-30 ms) shared with no-spec.
- Raw artifacts: perf/results/2026-08-11/muse-qgemv-mm/.

## 2026-08-11 - muse_step: The Whole Decode Forward In One Command Buffer

- Status: retained. Greedy generations are byte-identical between the fused
  and eager paths (the strongest available end-to-end parity check for a
  deterministic decode); no-spec decode 12.0 -> 14.4 tok/s in matched
  ambient conditions, 7.34 -> 14.4 (+96%) across the whole optimization arc.
- The deep fix identified by profiling: kernels were at bandwidth but the
  step spent ~40 ms in per-op Python/ATen/torch-MPS dispatch across ~1,800
  eager ops, plus engine overhead. This is the llama.cpp / CUDA-graph
  execution model brought to the Metal path: `muse_step_run` encodes all 52
  decoder layers (~900 dispatches: rms norms, GGUF matvecs, QK-norm,
  interleaved rope, KV scatter, windowed paged attention, sigmoid gate,
  SwiGLU, residual adds) into ONE command buffer from a C++ loop in
  qc_metal_serving.mm. Weights and scratch register once
  (muse_step_init/muse_step_layer); each step passes only hidden rows,
  positions, and the two KV-group metadata sets (SWA and full-attention
  block tables / seq lens / slot mappings).
- New glue kernels (csrc/quixicore/metal/kernels/serving_glue/muse_step.metal,
  a SlimServe addition beside the vendored tree): muse_rope_qk (interleaved,
  in place), muse_kv_store (paged scatter), muse_sigmoid_mul, muse_silu_mul,
  muse_add_inplace. Kernel names export namespaced (mittens::*) -- the
  encoder resolves them so.
- Python side: MuseGlimmerModel lazily registers on the first eligible step
  and takes the fused path only for pure single-token decode with no aux
  captures (spec-off); everything else -- prefill, mixed batches,
  speculative verify, vision -- keeps the eager path. Any registration
  failure logs once and permanently falls back.
- Where the remaining time is: step ~70 ms vs the ~43 ms weight-bandwidth
  floor. The forward is now ~48-55 ms; the rest is engine-side per-step
  overhead (scheduler/sampler Python), now the largest single line item.
- Next, in order: (1) extend muse_step to M=17 with aux-hidden capture and
  an on-device greedy rejection kernel -- that is what finally makes DFlash
  net-positive (the qgemv_mm groundwork already holds the verify matmuls at
  bandwidth); (2) engine step overhead; (3) last kernel margins vs
  llama.cpp (26.6 no-spec reference).
- Raw artifacts: perf/results/2026-08-11/muse-fused-step/ (parity harness
  and runs).

### 2026-08-12 - No-spec storms on V2 too: graph-replayed pure decode implicated

The V2-runner-default fix for DSV4 (DeepseekV4ForCausalLM added to
DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES - previously spec-off configs
silently fell through to the V1 runner, the confound in the earlier
no-spec arm) enabled the first clean single-variable no-spec test:
V2 runner, aux-off, FULL capture-64, no speculation. IT STORMED -
16,096 NaN lines, runs 0/11 -> 8/11 -> 11/11 degenerate, whole-batch
steady-state signature. So the no-spec storm is NOT the V1 runner and
NOT the aux streams.

Geometry explains why production (spec-on) is clean while no-spec
storms: with DSpark k=5 at c11, decode steps are 66 tokens > capture-64
-> EAGER; without spec they are 11 tokens -> FULL GRAPH REPLAY. The
no-spec flag silently moves pure decode onto replayed graphs - the same
regime the original storm incident implicated (and the bt_per_token
persistence fix addressed for the spec-shaped path). Discriminator in
flight: identical no-spec boot with cudagraph_mode NONE. Clean =>
FULL-graph replay of the 1-token decode path has its own capture
hazard; storming => no-spec batch dynamics themselves.

### 2026-08-12 - No-spec storm characterized: concurrency-dependent, config-independent

Discriminator ladder for the no-spec storm, all on the fixed V2-default
runner: FULL graphs stormed (16,096 lines), graphs NONE stormed HARDER
(24,116 lines, 11/11 from run 1), single-request c1 CLEAN (long and
short prompts, zero events, coherent output). Verdict: the no-spec
storm requires concurrent load - the chunked-prefill + 1-token-decode
mixed batches at c11 - and is otherwise independent of runner (V1/V2),
graph mode (FULL/NONE), and aux streams (fired with aux off). 3/3
no-spec boots stormed; contagion is 1 row -> whole batch within 2
steps, so the cross-request channel also exists without speculation.

Production exposure: NONE (profiles mandate DSpark; the spec-shaped
path is clean under the same loads). Operational consequence: --no-spec
is NOT usable for DSV4 A100 perf diagnosis until this is fixed - its
output degenerates under any concurrent benchmark. The next bisection
steps when this thread is picked up: minimal concurrency (c2/c3),
prefill-only vs decode-only mixes, and the indexer's no-spec decode
branch (next_n=1) vs the spec-shaped flattening branch - the largest
code fork between the two configurations.

### 2026-08-13 - BREAKTHROUGH: both bugs born in the first sparse-attention layer

The probe-liveness diagnostic boot (deterministic no-spec c11 storm)
delivered clean per-layer dumps and a surgical fingerprint: slots 0, 1
and 100-108 ABSENT (layers 0-1 fully clean; layer-2's attention INPUT
clean), slot 109 = LAYER-2 ATTENTION OUTPUT is the birth site (~100
deterministic events), cascading 110 -> layer outputs 2..9. Re-reading
the earlier spec-mode captures with this key: cap1's fresh birth was
also (109,1) - the (100..108, 7) chain there was carry-through of
already-poisoned rows re-entering the net, which I had misread as a
layer-0 birth (min-slot logic breaks once poisoned requests recirculate).

ONE BIRTH SITE FOR BOTH BUGS: layer 2 is the first compress_ratio=4
layer (ratios [0,0,4,128,4,...]) - the first C4A sparse-attention layer
with an indexer and sparse MLA over the compressed KV cache. The rare
spec seed and the deterministic no-spec storm are the same defect
class: sparse attention producing NaN from clean inputs, rare under
spec-shaped batches, deterministic under no-spec concurrent
chunked-prefill/decode mixes. Top-k indices are always IN RANGE
(validator zero across every storm); prime suspects are therefore
in-range-but-unwritten reads: fp8 KV bytes 0x7F/0xFF decode to NaN, so
a topk index (or compressed-slot mapping, ratio-4 bookkeeping) pointing
at an allocated-but-not-yet-written compressed KV slot yields exactly
this signature. The batch-shape dependence (decode_threshold 1 vs 6)
changes which rows sit in mixed batches while prefill chunks are still
writing compressed slots.

Next bisection (deterministic 10-min repro): instrument layer-2's
indexer decode path to compare each row's topk_indices against the
request's WRITTEN compressed length (not max seq len), and dump the
sparse MLA gather inputs for the first NaN row; then the ratio-4
insert/readback boundary (indexer_k_quant_and_cache vs
cp_gather_indexer_k_quant_cache) for off-by-one-chunk lag.

### 2026-08-13 - Retractions and re-aim: probes were on dead code

Three corrections from the o_proj arm and a code-path audit:

1. Fused o_proj EXONERATED: forcing the grouped MMQ path for all sizes
   (new diagnostic dial VLLM_DSV4_O_PROJ_FUSED_MAX_TOKENS=0) still
   storms (4,864 lines, 11/11 both runs). The <=32-gate geometry match
   was coincidence. (Also found: the VLLM_DSV4_NATIVE_Q8_O_PROJ=0
   fallback is bit-rotted on GGUF - crashes on .weight; unusable as a
   kill switch.)
2. RETRACTED: "sparse MLA core is clean". The KV byte-census probe sat
   in vllm/v1/attention/backends/mla/quixicore_mla_sparse.py, which is
   NOT the DSV4 A100 path. The real path is
   vllm/models/deepseek_v4/amd/rocm.py: forward_mqa ->
   rocm_sparse_attn_decode (merged SWA+sparse) and _forward_prefill.
3. RETRACTED (attribution): the bt_per_token persistence fix lives in
   that same unused builder - almost certainly a NO-OP for DSV4
   serving. The 0/8 vfix campaign result was phase luck, and the
   FULL-graph restoration decision now rests only on empirical
   cleanliness (0/8 + production NAN_WATCH clean since), not on a
   mechanism fix. The FULL-vs-PIECEWISE storm split (2/6 vs 0/6,
   p~0.23) may itself have been lottery.

New probe on the REAL path (VLLM_DSV4_ATTN_SPLIT_DEBUG=1): after
rocm.py forward_mqa, per layer, report whether NaN rows live in the
decode segment (rocm_sparse_attn_decode) or the prefill-chunk segment
(_forward_prefill), plus query health. Boot in flight on the
deterministic no-spec repro.

### 2026-08-13 - MECHANISM PROVEN: compressed KV cache holds NaN-coded fp8

Census on the real decode (mla_decode_fp8_sparse_dsv4 inputs, NaN rows
vs controls): request mapping correct (req ids sane vs num_decodes),
extents healthy (swa 128, topk 250), query clean at the first poisoned
layer - and the fp8 payloads of the gathered compressed-cache entries
contain NaN encodings (0x7F/0xFF) in ~30% of selected tokens
(80/250, 67/250, 73/250, 76/250 across dumps; layers 2, 4, 38). The
ratio-128 layer (extra_len=7) gathers clean; its q was already
poisoned (carry-through). The attention kernel is exonerated: it
faithfully attends over a corrupted compressed (ratio-4) cache.

Root cause is therefore ONE hop away, in the compressed-cache WRITE
path (the MLA compressor / its cache insert) under chunked prefill:
~30% of a 1000-token request's 250 compressed groups holding garbage
suggests whole chunks' compressed writes missing or mis-slotted (an
offset relative to chunk start vs sequence start, or partial-group
handling at chunk boundaries). The boot lottery falls out naturally:
unwritten slots hold recycled allocator bytes, and whether those
decode to fp8-NaN varies per boot. Next: extend the census to dump the
bad POSITION histogram (contiguous range => chunk-offset bug;
scattered => partial-group bug), then read the compressor's slot
computation in the cache insert.

Production exposure check still holds: spec-mode serving with aux-off
has zero NAN_WATCH events across days of load - the spec-shaped
schedule evidently avoids the triggering write pattern (to be
explained by the same fix). All diagnostics env-gated, default off.

### 2026-08-13 - RETRACTION: the fp8-NaN-content conviction was a census bug

The offline replay of the captured "bad" token runs CLEAN through the
real compress kernel, and inspection shows why: the fp8 payload of a
cached token is bytes [0:448]; bytes 448..575 hold the bf16 rope pair,
where 0x7F/0xFF single bytes are ordinary bf16 encoding bytes. Every
census in this hunt scanned [:512], counting 64 bf16 bytes per token -
expected false-positive rate ~40% of tokens, which is exactly the ~30%
"corruption" measured. The "compressed KV cache contains NaN" claim and
the compressor-write conviction are RETRACTED. What still stands,
measured with valid instruments: decode attention output NaN with clean
query, healthy extents, correct request mapping, and clean state-cache
gather. Corrected-census boot (448-byte window on both probes) in
flight to establish the real KV content status.
