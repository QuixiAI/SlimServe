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
suggests whole chunks' compressed writes missing or wrongly slotted (an
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

### 2026-08-13 - Hunt state: kernel convicted, mechanism still internal

Where the elimination stands after the full-entry audits (all with
TRUE cache addressing - two earlier census instruments were themselves
buggy and their convictions retracted: the 512-vs-448 window and the
584-vs-576 stride):

CLEAN, verified at the moment of NaN birth: query; top-k fp8 payloads;
SWA fp8 payloads; rope bf16 halves (isnan-proper); UE8M0 scales (max
122); extents; request mapping; state-cache gather; compressor writes
(every chunk, every size, immediate readback); the software e4m3
encoder (2.5M-value sweep); state slot mappings (no skips). Control
rows with identical input profiles compute finite outputs in the same
launch where neighbors go NaN.

DISPROVEN fix candidate: ml/es split-partial workspace was torch::empty
while the reducer reads min(partitions, token-length) slots per row =
all of them; initializing ml=-inf / es=0 (kept - correct hygiene, the
reducer guard treats unwritten slots as authentic empties) did NOT
stop the deterministic storm: 11/11 degenerate, 4,860 NaN lines.

CONVICTED but not yet dissected: quixicore_ops.mla_decode_fp8_sparse
_dsv4 (csrc/quixicore/tm_cuda/tm_cuda_serving.cu + mla_decode_fp8_v +
dsv4_attention_reduce_active_channels in paged_attn_v2_kernels.cuh)
produces NaN attention output from fully-audited-clean inputs with
initialized workspace, per-row nondeterministically, only under the
concurrent no-spec batch geometry (13/13 boots) and rarely under spec
(the production seed). Next step is mechanical: capture one NaN row's
complete op inputs (q row, both index lists + lens, the ~378
referenced 584B cache entries), replay py_mla_decode_fp8_sparse_dsv4
offline, dump tmp/ml/es to find which partial goes NaN, and bisect the
persistent writer kernel at that partition. The capture hook pattern
is already in ampere.py (_ampere_decode_debug); extend it to torch.save
on first detection like the compressor capture did.

Production posture unchanged and safe: spec + aux-off serving is clean
under load (the seed needs the geometry production avoids at width;
NAN_WATCH + canary standing). The no-spec flag stays documented as
broken until this kernel bug is fixed.

### 2026-08-13 - ROOT CAUSE FOUND AND FIXED: 0*NaN in the split-K reduce

The whole incident family - the rare production seed, the BOS-loop
storms, and the deterministic no-spec degeneration - reduces to one
IEEE trap in the DSV4 sparse decode's split-K reduction
(csrc/quixicore/serving/paged_attn_v2_kernels.cuh,
dsv4_attention_reduce_active_channels):

  value += tmp_out[partition] * weight;

The persistent writer (mla_decode_fp8_v) publishes finite ml/es stats
for balanced-away EMPTY partitions but never stores their 512-float
tmp vector; tmp was torch::empty. weight is exactly 0 for those
partitions - which looks like a correct no-op - but IEEE 0*NaN = NaN
and 0*inf = NaN, so whenever the recycled allocator bytes behind an
unwritten tmp vector decoded to NaN/inf, one empty partition poisoned
the row.

Proof chain: partials census (ml_bad=0, es_bad=0, tmp_bad~150 on
storm-shaped calls); sentinel fill of tmp mapped exactly one
unwritten partition per (batch,head) inside "written" stats AND
stopped the storm (deg 0/11, 0 NaN lines - first clean boot of that
config in 14); the one-line reducer guard (skip !(w > 0)) alone, no
sentinel, no debug: 3 full-length trigger runs, deg 0/11 each,
0 NaN lines.

Every mystery of the incident collapses into this mechanism:
- Boot/phase lottery = recycled allocator content behind unwritten
  partials.
- Batch-geometry gating = partition counts and tmp reuse patterns
  (spec c11 verify widths vs no-spec 1-token decode).
- Graph-mode correlation (FULL storms) = graph pools changing what the
  allocator recycles - correlation, not causation.
- Per-row nondeterminism and clean-input convictions = the poison
  entered through a buffer no input audit covered.
- The aux-off "3/3 silent" = phase luck and/or allocation-pattern
  shifts; the mHC/collective/runner theories were all epiphenomena.
- The earlier reducer NaN-guards (!(mp > NEG_INF)) helped only when
  the garbage happened to make ml NaN; finite ml with garbage tmp
  sailed through.

Fixes landed: the w>0 guard in dsv4_attention_reduce_active_channels;
same hardening in the three sibling reducers (one-warp, WARPS-wide,
and the plain template - one skipped only ml=-inf, one tested
weight==0.0f exactly which passes NaN weights, one was unguarded);
ml/es workspace init (-inf/0) kept as defense in depth. tmp remains
torch::empty by design - the guard makes unwritten vectors
unreachable. no-spec serving is UNBROKEN by this fix (the 3 clean
validation runs were no-spec).

## 2026-08-13 - c1 latency floor diagnosed (task #27): structural, not kernel-bound

Torch profile of steady c1 spec decode on the production q4ktail-4
config (fixed kernel, ~5K context, 140 tok/s). Per ~31ms spec cycle
(6-token verify, acceptance ~3.4 -> ~4.4 tok/cycle):

- Target verify step wall (execute_6 annotations): ~19.3 ms
- Pure kernel time inside the whole cycle: ~11.7 ms (19,887 launches
  over 16 steps; graph-replayed, so launch count is not the cost)
- => ~7-9 ms idle INSIDE the target step (breakable-graph eager
  sections around sparse attention + Python between segments), and
  ~11.7 ms of drafter + sampling + CPU orchestration between target
  steps.

Kernel breakdown per step (top): IQ2 gate_up+SwiGLU 1.28ms, MLA sparse
decode 1.06ms, aligned-Q8 GEMVs 1.33ms, drafter Q8/Q2K GEMVs ~0.9ms,
grouped-Q8 0.59ms, Q2K down 0.53ms, mHC 0.79ms, indexer 0.45ms,
custom-AR 0.35ms, long tail ~3.4ms (incl. 145x quantize_q8_1 of ~2.3us).

CONCLUSIONS:
1. TP2 ~ TP4 parity is now explained: the ~21ms fixed part of the
   cycle (drafter, orchestration, eager-break idle) does not shrink
   with TP; only the ~10ms kernel part does. Doubling ranks moves the
   cycle from ~34 to ~31ms - exactly the observed parity. This is not
   a defective-collective problem; it is Amdahl on fixed overhead.
2. Ranked c1 levers: (a) intra-step idle - fewer/cheaper eager breaks
   around sparse attention (largest, hardest: needs sparse-attn graph
   capture support); (b) drafter cycle cost (5 draft forwards -> the
   Q2K/Q8 GEMV drafter is ~1ms GPU but carries CPU orchestration);
   (c) kernel tail consolidation (marginal).
3. Task #26 re-scoped by data: the Q4_K tail (moe_vec_q block_q4_K)
   is only ~0.26 ms/step at c1 (~2.6% of kernel time) - fused Q4_K
   tiles are NOT a c1 lever. Their value is c8/prefill width and the
   quality-tier unlock; rank accordingly.
4. Task #28 (cp.async on seg tiles) is likewise a prefill/c8 lever;
   no c1 effect expected.

Raw trace: scratchpad prof-c1/ (4 ranks); server log prof-server.log.

## 2026-08-14 - Fused Q4_K (12,12) decode pair + cp.async seg pipelining: implemented, validated

Two kernel deliverables landed together (e2e arms running, results to
follow):

1. Fused Q4_K MoE decode pair (csrc/quixicore/quant/dsv4_q4k_moe_ampere.cuh,
   op ggml_dsv4_moe_a8_q4k): warp-per-row gate/up GEMV on raw block_q4_K
   with fused SwiGLU + route weight + Q8_1 emission, plus a Q8xQ4_K down
   weighted sum -- the tail layers' 7-launch generic route (vec w1, SwiGLU,
   requant, 4x moe_align, MMQ w2, weighted moe_sum) becomes 2 launches.
   Dispatch: (qweight_type, qweight_type2) == (12,12), decode widths
   (VLLM_GGUF_DSV4_Q4K_ROWS, default 64), kill switch
   VLLM_GGUF_DSV4_AMPERE_Q4K.
2. cp.async double-buffered y tiles in all four seg kernels (IQ2 W1, Q2K
   W2, MXFP4 W1/W2): span-parity double buffer, 4-byte cp.async.ca for the
   qs payload (36-byte block_q8_1 stride defeats 16B alignment), scale
   halves converted after the async issues, weight-decode global loads
   overlapping the in-flight y group. J=64 tiles moved to dynamic smem
   (57.6-61.7 KB, opt-in attribute); occupancy 3->2 CTAs/SM at J=64,
   accepted pending e2e numbers.

VALIDATION (the interesting part -- two instrument lessons):

- cp.async change: BIT-EXACT vs the pre-change build (same inputs, 6
  cases: iq2/mxfp4 seg x 8/96/1024 tokens). This is the right criterion:
  the change moves bytes differently but performs identical arithmetic in
  identical order. Pad columns keep stale qs under an exactly-zero scale
  (integer operands, cannot poison -- deliberately checked against the
  0*NaN incident class).
- Q4_K parity vs fp64 dequant reference: first two harness attempts
  FAILED for instrument reasons, not kernel reasons:
  (a) reference dequantized with the fp32 Q8_1 scale, but quantize_q8_1
      STORES the scale as fp16 -- every consumer sees the fp16-rounded
      scale; (b) even with (a) fixed, recomputing v in fp64 flips
      round(v/scale) by +-1 near .5 boundaries (fp32 kernel vs fp64 ref
      differ ~1e-6; each flip perturbs a whole 4096-wide output row by
      ~1e-3 of mean, and flips stack).
  Proof of kernel correctness: single-route lstsq decomposition of the
  fused-vs-reference diff = exactly 7 mid deltas, all +-1.0 quanta, all at
  v/scale frac ~= .5000-.5019. The proven generic MMVQ path shows the
  same-magnitude deviation vs the reference (bulk 0.29) -- the reference
  is the outlier through the quant cliff, not the kernels. Final metric:
  mean relative error (flip floor ~1.5e-3 vs O(1) for a wrong kernel);
  PARITY_PASS 21/21 across inter {256,512,1024}, tokens {1..64}, masked
  experts.

Lesson recorded: parity through a quantization cliff can never be
elementwise-tight for a reference computed in different precision; use
bit-exact A/B when arithmetic is unchanged, flip-decomposition + mean-rel
when it is not. (Same family as the fp8-census false convictions from the
degeneration incident: validate the instrument before believing it.)

Baseline for the arms (segon, pre-Q4K/pre-cpasync build, same boot
conditions, post reduce-fix): 12k-cold 144.5, 12k-hot 150.2, 12k-c4
234.7, 1k2k-c8 359.2 tok/s, all exact. Raw: perf/results/2026-08-14/
{gate768-pair,q4k-arms}/.

## 2026-08-14 - Q4_K fused + cp.async + gate-768: e2e verdicts (tasks #26/#28 closed)

Three-arm matrix on dsv4-q4ktail-4 (fresh boots, exact harness, same
prompts; baseline = segon arm from the same day, pre-change build). Raw:
perf/results/2026-08-14/{gate768-pair,q4k-arms}/.

tok/s (12k-cold / 12k-hot / 12k-c4 / 1k2k-c8 / 1k2k-c1):
- segon baseline: 144.5 / 150.2 / 234.7 / 359.2 /  --
- segoff (Q4K off):  148.1 / 150.3 / 234.2 / 324.0 / 119.7
- cpasync (Q4K off): 148.8 / 145.3 / 235.2 / 326.4 / 110.1
- q4kfull (default): 149.7 / 151.3 / 233.4 / 310.2 / 163.3

THE ACCEPTANCE TRAP, AGAIN: raw TPS says q4kfull c1 is +48%. Step-rate
normalization (tps / (1 + accepted/draft)) says otherwise:
- c1 steps/s: segoff 47.7, cpasync 47.6, q4kfull 47.5  -> IDENTICAL.
- c8 steps/s: baseline 105.9, segoff 100.1, cpasync 98.0, q4kfull 97.2
  -> ~6-8% cross-boot band, no arm effect beyond it.
All response sha256s differ across arms (slightly different numerics ->
divergent greedy text -> incomparable acceptance). Every apparent e2e TPS
delta in this matrix is acceptance/content variance. Rule reaffirmed: no
e2e speedup claim without step-rate normalization and matched outputs.

VERDICTS:
1. Fused Q4_K (#26): e2e step-rate NEUTRAL at c1 and c8. This is the
   c1-diagnosis prediction (2026-08-13) confirmed from the other side:
   decode is fixed-overhead bound, so removing 5 launches + moe_align
   from 6 tail layers vanishes into the idle window. Kept enabled:
   correctness proven, 2 launches < 7, zero cost anywhere, and the win
   materializes when eager-break idle shrinks (sparse-attn graph capture)
   or on tiers with more Q4_K layers.
2. cp.async double-buffering (#28a): kernel-level seg tile time
   1024 tokens 15.48 -> 12.73 ms (-18%), 2048: 19.38 -> 14.35 (-26%) --
   exactly the widths production seg serves (prefill chunks above the
   gate). Narrow widths regress (48: 3.84->4.52) but sit behind the
   768 gate and never run seg in production. E2e prefill neutral (MoE
   seg is a small slice of 12k prefill time). Bit-exact vs pre-change.
3. Gate-768 e2e pair (#28b): segon vs segoff identical at 12k
   (150.2/234.7 vs 150.3/234.2) -> the gate is e2e-neutral on this
   workload; post-cp.async kernel crossover still brackets 768 (fused
   wins <=512, seg wins >=1024). Gate retained at 768.

Production: both daemons restarted onto the committed build (A during
arms teardown, B explicitly); canary + NAN_WATCH active.

## 2026-08-15 - c1 host-side budget (py-spy on live production) + steady-decode metadata reuse

Correction to the 08-13 eager-break theory: decode runs FULL graphs (boot
logs: "Capturing CUDA graphs (FULL) 8/8" target AND drafter) - attention is
INSIDE the graph. Async scheduling is already enabled (AsyncScheduler
frames live in the engine). The fixed ~21 ms/cycle at c1 decomposes
(py-spy 100 Hz, 40 s on production A worker TP0 + engine):

worker: 23.9% target attn metadata build (~5.0 ms/cycle), 20.4% waiting on
output copy_event, 16.3% sampler/rejection (~3.4 ms, incl. 1.6% our
NAN_WATCH), 15.1% drafter python (~3.2 ms; incl. ~0.8 ms EAGER context-KV
insert through Python GEMV dispatch), 14.1% starving on next input, 2.4%
input prep. Engine: 85% idle (not the bottleneck). Worker CPU ~13.7
ms/cycle vs GPU ~11.7 ms, imperfectly overlapped -> 47.5 steps/s.

Metadata build detail: 55% torch-op dispatch (hundreds of tiny ops), 25%
python object churn, 18% Triton launch overhead. Builders per step: rocm
SWA subclass (ragged repack), FlashMLA/c128a subclass (ragged repack),
sparse_swa base (3 device ops + 30-field dataclass), indexer, plus
CommonAttentionMetadata construction per KV group. KEY INSIGHT: at FULL-
graph decode the replayed graph reads only the builders' persistent device
buffers - the Python metadata objects are only consumed at capture and by
eager/prefill steps, so a steady uniform-decode step only needs the
builders' device kernels re-run.

IMPLEMENTED (env VLLM_STEADY_DECODE_META, default off):
build_attn_metadata caches per-group (CommonAttentionMetadata, [(builder,
metadata, layers, supports)]) keyed on (num_reqs, num_tokens,
max_query_len) for all-decode FULL steps; on hit it refreshes drifting CPU
scalars (max_seq_len) and calls builder.steady_decode_update() -- device
work only -- implemented for DeepseekSparseSWAMetadataBuilder (+ rocm
ragged subclass) and DeepseekV4FlashMLAMetadataBuilder (+ rocm ragged
subclass); the indexer group full-rebuilds against the cached common
metadata (safe fallback). All cached tensor fields are views over
persistent buffers the runner refreshes each step; correctness gate =
exact harness + degeneration guard. A/B window running.

Remaining c1 roadmap by measured size: (1) this change (~5 ms class);
(2) sampler/rejection path 3.4 ms; (3) drafter python 3.2 ms (eager
context-KV insert -> capture; draft-loop consolidation); (4) full
CPU/GPU overlap (ceiling ~85 steps/s = +80%).

## 2026-08-15 - steady-meta A/B result + THE c1 finding

A/B (fresh boots, exact harness): metaoff c1 47.5 steps/s, metaon c1 47.5
steps/s (both runs each); c8 104.1 vs 103.8. Exact=true everywhere incl.
metaon c8 (the fast path engages at uniform c8 verify) -> the change is
SAFE but not yet the constraint. Left available, default off.

THE FINDING (this is the important one): c1 steps take 21.05 ms while c8
steps take 9.6 ms - on the same boot, same machinery, with 8x the
per-step work. And c1's 47.5 steps/s has been INVARIANT across every
change this session (Q4_K fused, cp.async, 5.6 ms of metadata Python
removed). Conclusion: the c1 critical path is a fixed ~21 ms serialized
round trip in the output -> engine -> schedule -> input chain that (a)
does not scale with batch work, (b) is fully hidden at c8, (c) shadows
all worker-side compute at c1 (removing 5+ ms of worker CPU changed
nothing because it sat inside the shadow). py-spy occupancy shares are
NOT critical-path attribution - the invariance test is.

Ruled out: HTTP/client (harness is stream=False), shm SpinCondition
quantization (busy-spins for 1 s before idling; 21 ms gaps stay in spin
mode), async scheduling absence (AsyncScheduler active, dspark
placeholders in use).

NEXT (highest-value tok/s lever, ~2x c1 if closed): timestamp one cycle
end-to-end across engine and worker (dequeue -> schedule -> shm
broadcast -> worker input wait -> launch -> copy_event -> get_output ->
engine receive) to locate the ~11 ms that is neither worker compute nor
GPU. Candidates: spec-decode output round trip that async scheduling
does not actually pipeline at bs=1 (worker starving 14% + engine idle
85% is consistent with lockstep), TP broadcast rendezvous, api-proc zmq
hop on the critical path.

CORRECTION (2026-08-15, same day): the "c8 steps (9.6 ms) faster than c1
steps (21 ms)" claim above was arithmetic error - steps/s = tps/(1+acc)
gives PER-REQUEST step rate; at c8 all 8 requests advance in one engine
step, so engine cycles are ~77 ms at c8 (slower, as expected for more
work). What stands: c1 cycle = 21.05 ms, INVARIANT across kernel + CPU
changes. Revised model: with async scheduling the worker's CPU prep for
step N+1 overlaps step N's GPU, so the c1 critical path = verify GPU +
rejection sync + 5 sequential drafter launches + D2H/H2D + engine
turnaround; the metadata build sat in the overlapped region (hence
neutral). Decomposition requires GPU-timeline gap analysis, not
occupancy shares -> torch trace of a steady c1 window.

## 2026-08-15 - RESOLUTION: c1 is GPU-BOUND (98% busy); host is fully overlapped

Fresh torch trace, steady c1 decode window (111.8 ms, rank1): GPU busy
109.4/111.8 ms = 98%, idle 2.4 ms, no gap class over 100 us worth
reporting. The serialized-round-trip theory is DEAD, and the 08-13
"~40% idle" decomposition was an artifact of dividing kernels by
annotation count. With async scheduling, ALL host work (metadata build,
sampler python, drafter orchestration) overlaps GPU execution - which is
why removing 5.6 ms of metadata Python and 5 launches/layer changed
nothing: c1 throughput is purely GPU work per cycle.

GPU budget per window (per-cycle scale by ~1/3.5): IQ2 gate_up+SwiGLU
13.0 ms (top), MLA sparse decode 11.0, aligned-Q8 GEMVs 13.5 (945x,
dense projections), DRAFTER Q8_0 GEMVs 9.0 (238x) + bf16 s16816gemm 3.5
(131x), grouped-Q8 6.0, Q2K down 5.4, indexer 4.6, mHC 8.0, custom-AR
3.6, quantize_q8_1 3.35 (1431 launches), reduce 2.7, topk 2.6, fused
Q4_K 2.4 (30x, active).

tok/s roadmap is therefore GPU-work reduction, ranked:
1. DRAFTER cost (~12 ms/window ~= a fifth of GPU time on the 0.5B
   drafter at TP4 with per-layer AR): candidates - drafter TP1
   (replicated, no AR), fewer drafter layers on the critical path,
   revisit k.
2. IQ2 gate_up decode (13 ms, biggest single): revisit qwarp8 /
   cooperative variants at this exact geometry, or W1 tensor-core at
   verify width.
3. MLA sparse decode 11 ms: partition/launch tuning at 6-token verify.
4. quantize_q8_1: 1431 tiny launches/window - batching/fusing into
   producers.
Steady-meta stays available (harmless, host-side); its value returns if
host ever re-enters the critical path (e.g. much faster kernels).

## 2026-08-15 - kernel-dial sweep: qwarp8 WINS (+2.7% c1 step rate), DEPLOYED

With c1 proven GPU-bound, swept the existing IQ2/Q2K kernel dials at c1
(fresh boots, exact harness, steps/s = the acceptance-free metric):
- VLLM_DSV4_W1_QWARP8=1: 48.8 / 48.8 steps/s  <- +2.7% vs 47.5 baseline,
  consistent across both runs, exact=true. First step-rate movement of
  the session: the 256-thread 8-lane-per-row IQ2 W1 decode variant beats
  the 1024-thread default at the current config (aux-off, capture-64).
- VLLM_DSV4_W1_COOPERATIVE=1: 47.7 / 47.6 (neutral)
- VLLM_DSV4_Q2_DOWN_ROWS=4: 47.4 / 47.8 (neutral)
Encoded VLLM_DSV4_W1_QWARP8=1 in dsv4-q4ktail-4 and dsv4-q4ktail-8 (same
TP4-shard kernel geometry); q4ktail-2 (1024-row shard) left default
pending its own measurement. Tests updated (58 pass). Both daemons
restarted onto it. The qwarp8 dial only affects the tokens<=8 decode
route, so c8+/prefill are untouched by construction.
Raw: perf/results/2026-08-15/kdial-sweep/.
## 2026-08-12 - DFlash Spec Gap: Reference Measurements Localize the Wall

- Question: model card advertises 250 tok/s spec on RTX 5090 (74.9 no-spec,
  3.1x multiplier); our Metal spec path measures ~9.8 tok/s against 14.4
  plain. Where is the loss: acceptance wiring, verify cost, or engine
  overhead?
- Method: llama.cpp muse-pr branch (build b10412) as reference on the same
  M5 Max, same GGUF pair, same 256-token greedy essay prompt, via
  llama-server (llama-cli on this branch is an interactive chat UI that
  ignores -no-cnv and cannot batch-bench; a 12h "hang" was it REPL-looping).
- Pitfall worth remembering: with a local -md file, llama.cpp defaults
  --spec-type to none (sidecar type inference only runs for HF downloads);
  the server silently serves plain decode with the drafter loaded. Explicit
  --spec-type draft-dflash is required. First "reference" numbers were
  plain decode in disguise (26.75 tok/s, no draft stats).
- Acceptance verdict: reference mean acceptance 3.08-3.14 tok/step at block
  16 on our prompt (0.148/drafted token) vs our 2.7-2.9. Our drafter wiring
  is CORRECT; ~10-15% quality gap only. The card's ~3.3 is their prompt mix.
- Throughput verdict: reference llama.cpp spec is NET NEGATIVE on M5 Max at
  block 16: 15.7 tok/s vs 26.75 plain. Sweep: nmax16 15.7, nmax8 14.0 (non-
  monotonic dip mirrors our 16-vs-17-row kernel anomaly), nmax4 29.5,
  default 30.8 (+15% best case, mean len 2.27).
- Root cause pinned by llama-batched-bench (no spec plumbing): batch-M
  decode step cost on their Metal stack is 36.3 ms at M=1 but 179 ms at
  M=17 (4.9x); M=8 117 ms, M=16 182 ms. The 196 ms spec step = 179 verify +
  ~15 drafter. The verify wall lives in the reference too; the 5090's
  3.1x comes from tensor cores making batch-17 verify ~= batch-1
  (14.1 ms/step vs 13.4).
- Physics on this machine: 19.6 GB weights / 460 GB/s = 43 ms BW floor;
  batch-17 verify FLOPs ~1 TFLOP / ~14 TFLOPS fp16 simd = ~70 ms compute
  bound. The card's own Apple row (26.6 -> 50.2, 1.8x) implies a ~66 ms
  spec step -- exactly the simd compute roofline. 50 tok/s on M5 Max is
  therefore demonstrated-achievable with a verify GEMM at the compute
  bound; neural-accelerator/simdgroup_matrix tiles could push toward the
  BW-floor ceiling (~3.1/50ms ~ 60 tok/s). "Half a 5090" is out of reach
  no-spec (BW ratio is 0.26) but ~quarter of 233 is the honest target.
- Decision: acceptance is not the work item. The work item is the M=17
  verify step cost in muse_step: simdgroup_matrix fp16-fragment GEMM over
  dequantized Q4K/Q5K/Q6K tiles (Marlin lessons per
  csrc/quixicore/a100_glm52_design.md), fused aux-hidden capture, and the
  on-device greedy rejection kernel to evict the MPS sampler host syncs.
  Target: verify step <= 90 ms first (spec ~32 tok/s at mean 2.9), then
  chase the 66 ms reference-implied roofline.
- Raw artifacts: perf/results/2026-08-12/llamacpp-dflash-reference/
  (config sweep, batched-bench M-curve, acceptance run, sweep script).

## 2026-08-13 - qgemm_sm: Weight-Streaming Split-K MMA GEMM For The Verify Band

- Status: retained; routed for M in [9, 32] on Metal. Spec-on end-to-end
  8.6-9.2 -> 11.05-11.79 tok/s (+28%), acceptance unchanged (1.70/draft,
  2.70 tok/step, from live server metrics), sample generations coherent.
  Spec step ~252 ms (was ~320) vs ~70 ms fused plain decode: still
  net-negative until the host-side spec machinery is fused.
- Hypothesis: with reference llama.cpp also 4.9x off bandwidth at batch 17,
  the M=17 verify matmul needs simdgroup_matrix (MMA) with a shape that
  wastes no weight streaming on padding, high threadgroup counts, and
  load/math overlap -- not just the existing qgemm tile (2 warps splitting
  32 columns: half the tg idles on padding at M=17, ~99 GB/s flat).
- Kernel: qgemm_sm<FMT, N_WARPS, RPW, SPLIT_K> in
  csrc/quixicore/metal/kernels/quantization/qgemm_sm/. Every warp computes
  all 32 (padded) columns for its own RPW weight rows; W dequants into
  warp-private shared slices; X K-tiles double-buffered; both staged one
  K-step ahead of the mma (single threadgroup_barrier per 32-wide K step).
  Split-K (grid.z) writes float partials folded by a deterministic reduce
  kernel -- no atomics, greedy verify stays bit-reproducible.
- Iteration ladder on real muse weights, M=17 (us; bw floor = one weight
  pass at 460 GB/s): naive 4x16 tile ~= old qgemm (o_proj 264 vs 243);
  +pipeline+8-row warps (r8) beat qgemv_mm 23-26%; +split-K 4 (r8sk4):
  o_proj Q6_K 148 (vec 281, floor 48.6), gate Q5_K 163 (vec 181, floor
  40.7), down Q4_K 671 (vec 990, floor 162). Parity <= 2.5e-3 vs host
  dequant fp32 matmul for all variants; still 3-4x off the bandwidth floor
  (per-step dequant+staging issue cost does not fully hide yet).
- Occupancy was the v1 killer: 16-row warps gave 104 threadgroups (~10% of
  thread capacity) with 624 serialized barrier steps. 8-row warps + split-K
  4 raise it to ~1664 threadgroups on the down projection.
- Routing: _sm_route_ok in gguf/linear.py sends M in [9, 32] (N % 16 == 0,
  K % 32 == 0, q4_0/q8_0/q4_K/q5_K/q6_K) to ggml_mul_mat_sm variant 9
  (r8sk4); qgemv_mm keeps M <= 8, the flat GEMM keeps prefill. The glue
  binding converts/transposes/pads (measured 15-40 us/call); the fused
  muse_step consumer will use the layout-native ggml_mul_mat_sm_pre.
- Next: (1) extend muse_step to M=17 fused verify with aux-hidden capture
  consuming qgemm_sm_pre directly; (2) on-device greedy rejection kernel;
  (3) fuse the DFlash proposer context-KV precompute -- together these
  target the ~150 ms/step host machinery, the last term keeping spec below
  plain decode.
- Raw artifacts: perf/results/2026-08-13/qgemm-sm/.

## 2026-08-13 - Fused M=17 Verify: Built, Correct, Not Yet Faster (Kept Gated)

- Status: implemented and correctness-validated; default remains the eager
  verify path (VLLM_MUSE_FUSED_VERIFY=1 enables the fused one) because the
  eager path is 20 ms/step faster at current kernel speeds. Recorded as a
  partial negative so the next agent does not re-derive it.
- What was built: muse_step_run_aux extends the single-command-buffer step
  to the k+1 verify shape -- qgemm_sm_rm (row-major bf16 X staged with a
  transposing cooperative load, bf16 transposed-store reduce), residual
  snapshots entering aux layers {2,14,26,38,50} via a muse_copy kernel,
  and attention as batch expansion: row i becomes a virtual decode request
  with seq_len base+i+1 over a stride-0 expanded block table (causality
  and sliding window from the existing kernel's clamps; zero new attention
  code). Acceptance identical to eager (643/6048 accepted/drafted on the
  bench workload, bit-stable across runs) and generations coherent.
- Iteration ladder (target forward at m=17, in-process, mps.synchronize):
  first cut 306.8 ms -> 199.7 ms via (a) vectorized ushort4 staging (the
  scalar 16-iteration transpose loop was 2.25x on K=19968), (b) four
  rotating split-K partial buffers (a single shared buffer serializes
  independent q/k/v/gate matmuls through Metal's hazard tracker), (c) a
  deep-K (>8192) route that pays one contiguous muse_xpose32 pass and uses
  the contiguous-staging kernel instead of scattered 64B tile reads.
- Why eager still wins (180.4 ms): at M=17 the eager path's per-op dispatch
  cost pipelines behind 300-1500 us kernel executions (CPU encodes ahead of
  the GPU), so encoder fusion buys little until the kernels approach the
  bandwidth floor. The muse_step M=1 win (dispatch-bound, 1800 ops over
  ~70 ms) does not transfer to the verify shape at current kernel speeds.
- CORRECTED step decomposition at k=16 (revises the 08-11 estimate of
  "~150 ms host machinery"): 252 ms/step = 180 ms target layers forward +
  ~70 ms everything else (lm_head ~8, DFlash drafter, MPS rejection
  sampler, engine). The verify GEMM wall -- 3-4x off the 43 ms/pass weight
  bandwidth floor -- is the dominant term and the priority lever. Spec goes
  net-positive vs the 14.4 tok/s fused plain decode at step < 200 ms; the
  card-implied 66 ms verify puts the same acceptance at ~40+ tok/s.
- BK=64 experiment (variant 10): bit-identical outputs; Q6_K o_proj 247 ->
  156 us (-37%, expensive dequant amortizes fewer barriers) but Q4_K down
  667 -> 810 and Q5_K gate_up 683 -> 803 regress. Routed per-format
  (Q6_K -> BK64) in quixicore/ops.py; e2e effect within run variance.
- End state: spec-on 11.2-11.5 tok/s, plain fused decode 14.4. Next, in
  order: (1) drive the verify GEMM toward <= 2x floor (M_PAD=24, per-row
  simdgroup-coalesced dequant at wider BK, mma_ABt col-layout staging, RPW
  sweeps -- measure each on gate_up/down); (2) re-test fused-vs-eager
  verify and flip the default when fused wins; (3) the ~70 ms non-forward
  tail (on-device greedy rejection, drafter fusion, lm_head route).
- Raw artifacts: perf/results/2026-08-13/fused-verify/.

## 2026-08-13 - Verify GEMM Roofline: Dequant-ALU-Bound, Paired-Plane Decode Is The Fix

- Goal reset (user): 100+ tok/s single-stream on M5 Max. Budget: one weight
  pass is ~43 ms at the ~460 GB/s measured stream rate -> ~23 steps/s, so
  100 tok/s needs BOTH a floor-speed verify step AND ~4.5-5 accepted
  tokens/step (tree/multi-candidate drafting on top of DFlash; M is nearly
  free once verify is flat in M). Neither alone suffices: floor-speed at
  today's 2.87 acceptance is ~66 tok/s.
- Discriminating probe: qgemm_sm structure with a bench-only f16_raw format
  (raw half weights, no dequant ALU, 4x the bytes) runs the down shape at
  770 us = 1.33x its own bandwidth floor (~345 GB/s sustained). The q4_K
  kernel on the same shape: 662-820 us = 4-5x ITS floor -- slower in
  absolute time than a kernel streaming 4x the bytes. Verdict: the tile
  flow (staging, barriers, split-K, occupancy) is healthy; the kernels are
  DEQUANT-ALU-BOUND. ~445 us of critical-path decode per down matmul =
  ~5 scalar ops/weight at ~1 op/cycle/lane, no vectorization.
- Rejected fix (measured): vectorizing the decoder's device byte loads
  (packed_uint2 / packed_ushort4 in the tk_dequant8 k-quant half
  specializations -- kept, they are strictly cleaner) moved r8sk4 only
  2-4%: the byte loads were already L1-hot. Also note the half-path span
  decoders DO exist (dequant.metal ~line 990) -- do not re-derive the
  missing-specialization theory; scales already unpack once per span.
- THE PLANNED FIX -- paired-plane decode at BK=64 (Marlin lesson, Metal
  form): in q4_K/q5_K a quant byte's two nibbles feed cols pos and pos+32,
  which land in the SAME BK=64 tile. Split whole byte-runs with two vector
  ops (qw & 0x0F0F0F0F, (qw >> 4) & 0x0F0F0F0F), convert uchar4->half4 in
  hardware, one half4 fma per 4 weights: ~14 ops per 16 weights vs ~96
  today. st tiles swizzle at 16-byte granularity, so each 8-half span is
  one swizzle unit: compute the swizzled base once, write two half4s.
  q6_K pairs at BK=128 (nibble planes 64 apart) -- keep it on the current
  BK=64 span path first (already its best: 145-156 us on o_proj).
  Projection: down 662 -> ~300-350 us (~2x floor), full-step verify
  matmuls ~150 -> ~80-90 ms, spec ~17-18 tok/s before fused-verify and
  tree-spec multiply further.
- Sequence to 100 tok/s: (1) paired-plane BK=64 kernels; (2) re-flip the
  fused verify default when it wins (encoder overhead matters again as
  kernels shrink) + on-device rejection to trim the ~70 ms tail; (3) tree
  speculation (multi-candidate DFlash blocks, tree-mask verify at M~32-48)
  to lift accepted tokens/step toward 4.5+; (4) chase the last kernel
  margin toward the 43 ms floor.
- Raw artifacts: perf/results/2026-08-13/qgemm-sm/ (f16 probe numbers in
  the session log; microbench_vectorized_loads.txt).

## 2026-08-14 - Paired-Plane Decode Landed; The Wall Is Now Multi-Factor

- Status: qgemm_sm_p kernels (paired-plane q4_K/q5_K at BK=64, 2/4/8-warp
  variants) retained, parity exact vs host dequant (rel == the span path's,
  bit-identical across warp counts). Shape-aware route: 2-warp for narrow
  shapes (gate 296 -> 217 us), 4-warp for big ones (gate_up 681 -> 643,
  down 710 -> 665). End-to-end effect ~5-8 ms/step -- real but small; bench
  triplets now show thermal decline (11.2 -> 9.0 tok/s run 1 -> 3), so
  sub-10 ms effects are unmeasurable e2e on this chassis without matched
  ambient discipline.
- Key correction to yesterday's roofline verdict: dequant ALU was A wall,
  not THE wall. With decode at ~2.4 ops/weight the big shapes sat flat --
  the f16_raw probe had hidden an X-restaging amplification term: every
  16-row threadgroup re-stages the full X K-slice, ~2*M_PAD*N*K/rows_per_tg
  bytes = ~532 MB of L2 traffic per big matmul vs 75 MB of weights. Wider
  tiles (4-warp) captured part of the predicted saving on gate_up, less on
  down; p8 regresses (threadgroup-memory residency). Best kernels now sit
  at 3.2-4.1x the weight-bandwidth floor and the remaining gap is a SUM of
  ~100 us-scale terms: residual X staging, MMA issue + fragment loads,
  barrier latency, weight streaming.
- Conclusion for the next session: incremental single-factor edits are
  exhausted. Breaking 2x floor wants a coherent redesign in one step --
  M_PAD=24 (verify needs 17; cuts X traffic and MMA work 25%), 4-warp
  paired tiles, possibly X persistent in registers across a multi-row-block
  N-walk (X is only 226 KB at K=6656 as half) -- plus an offline Xcode/Metal
  counter profile of qgemm_sm_p4 on the down shape to replace inference
  with measurement before writing it. The e2e chain (forward 180 ms of the
  252 ms step) still points every saved matmul-ms directly at throughput.
- The 100 tok/s ledger is unchanged: floor-speed verify (~43 ms) AND ~4.5-5
  accepted tokens/step (tree drafting) are both required; kernel work is
  the current critical path, tree speculation is unstarted and is the
  larger single multiplier (2.87 -> ~4.5 tokens/step).
- Raw artifacts: perf/results/2026-08-14/ (paired kernel microbenches in
  session log; e2e triplet in muse_bench output).

## 2026-08-14 (2) - Ablation Profile: The Wall Is MMA Execution; M5 Tensor Ops Are 9x Faster

- Method (replaces inference-from-deltas): stage-ablation variants of the
  production qgemm_sm_p4 kernel (host variants 21..27, BENCH-ONLY, keep-alive
  tails defeat DCE) -- each removes exactly one stage. Round-robin
  min-of-rounds timing cancels thermal drift. Script + raw JSON:
  perf/results/2026-08-14/qgemm-sm-profile/.
- Down shape (q4_K, N=6656, K=19968, 74.8 MB, floor 163 us), full p4 =
  634 us (3.90x floor). Exposed per-stage costs (delta vs full):
  MMA issue **290 us**; weight stream+decode 52; dequant ALU 25; X
  re-staging 21; tg-barrier 18; fragment loads 14. Staging-only (no MMA,
  a3) = 331 us = 2.04x floor; pure weight-stream (a5) = 223 us = 1.37x
  floor; f16_raw structural probe 748 us (4x bytes, consistent w/ 08-13).
- VERDICT: yesterday's "sum of ~100 us-scale memory terms" theory is wrong.
  The dominant wall is executing the MMA itself on the classic simdgroup
  path. All memory-system terms combined leave the kernel at 2.04x floor
  even with MMA free.
- Standalone matrix-pipe rate probe (probe.metal/probe.mm in same dir;
  Metal 4, -std=metal4.0, macOS 26.5 toolchain): chained
  simdgroup_multiply_accumulate at the exact production tile shape
  (A 32x64 half, B 64xM_PAD half, fp32 acc, 78-step K-walk, 832 tgs) vs
  mpp::tensor_ops::matmul2d (M5 GPU neural accelerators), whole-down-matmul
  equivalents:
    simdgroup  M_PAD 24/32/48: 952 / 1327 / 2019 us (6.3-6.7 TFLOPS)
    tensor_ops M_PAD 24/32/48: 149 /  153 /  269 us (43-55 TFLOPS)
  Notes: (a) tensor ops are ~9x the chained-simdgroup rate at M_PAD=32 and
  land BELOW the 331 us staging floor -> MMA moves off the critical path
  entirely; (b) M_PAD 24 vs 32 is FREE on the tensor units (no M_PAD=24
  redesign needed); (c) M_PAD=48 costs only +76% -- the tree-speculation
  verify shape stays near one weight pass, confirming the "M nearly free"
  premise of the 100 tok/s ledger on this hardware; (d) exposed MMA in the
  production kernel (290 us) is smaller than the standalone chained rate
  (1327 us) because 4 warps x 4 accumulator chains overlap with staging and
  cross-tg parallelism -- exposed vs standalone differ, both conclusions
  hold.
- Decision: redesign qgemm_sm around tensor_ops::matmul2d -- keep the
  measured-cheap paired-plane dequant staging and double-buffered pipeline,
  swap the per-warp simdgroup MMA for one cooperative 32x64 @ 64x32
  matmul2d per K-step (execution_simdgroups<4>, multiply_accumulate into a
  float cooperative_tensor, cT.store to the same split-K float partials ->
  deterministic reduce unchanged). Projection: down 634 -> ~350-400 us
  (staging-bound), then attack staging terms next (X traffic halves again
  at 64 rows/tg if occupancy allows). Requires bumping the metallib to
  -std=metal4.0 (from 3.1) -- validate all existing kernels + serving path
  after the bump.

## 2026-08-14 (3) - Tensor-Ops Verify GEMM Landed: Spec Beats Plain For The First Time

- Change: qgemm_sm_t (variant 14) -- the qgemm_sm_p staging pipeline
  (paired-plane dequant q4_K/q5_K, span dequant q6_K, BK=64, double-buffered,
  one barrier/step) with the per-warp simdgroup MMA replaced by one
  cooperative 32x64 @ 64x32 mpp::tensor_ops::matmul2d per K-step
  (execution_simdgroups<4>, multiply_accumulate into a float
  cooperative_tensor, cT.store to the same (4, N, 32) float partials;
  deterministic split-K reduce unchanged, so greedy verify stays
  bit-reproducible). Metallib bumped -std=metal3.1 -> metal4.0 (all 81
  kernel sources compile unchanged). Route: Q4_K/Q5_K/Q6_K with K%64==0
  and N%32==0 -> 14 in quixicore/ops.py; simdgroup variants remain as
  fallback for uncovered shapes.
- Microbench (M=17, matched hot-chassis conditions, parity rel identical
  to the simdgroup path per format):
    o_proj Q6_K 6656x4096:  465 -> 152 us (9.6x -> 3.1x floor)
    gate   Q5_K 4096x6656:  186 -> 132 us (4.6x -> 3.2x)
    down   Q4_K 6656x19968: 829 -> 422 us (5.1x -> 2.6x)
  The predicted staging-bound regime arrived: MMA (~153 us standalone on
  the down shape) is off the critical path; remaining gap to floor is the
  staging terms the ablation profile measured (a3 = 2.04x floor).
- E2E through the real profile (muse-kdyn-1, port 8078, 256-token greedy
  essay triplet): **16.26 / 16.05 / 14.59 tok/s** spec-on, vs 11.2-11.5
  before and 14.4 plain fused decode. Acceptance 647/373 = 1.73/draft --
  unchanged. Generations coherent. SPEC NOW BEATS PLAIN: removed the stale
  --no-spec note from slimserve/profiles.json (handoff step 5) and added
  the muse baseline snapshot to perf/baseline_status.md.
- Not done / next: (1) fused verify (muse_step_run_aux) still uses the
  2-warp simdgroup qgemm_sm_rm -- it did NOT get faster and eager
  widened its lead; port tensor-ops MMA into the rm kernel before
  re-testing the fused flip. (2) staging terms are now the wall (X
  re-staging + stream pattern); 64-row tiles / X-persistent designs are
  the next kernel iteration, informed by the ablation numbers. (3) tree
  speculation: rate probe shows M_PAD=48 costs only +76% on the tensor
  units (M nearly free) -- the largest remaining multiplier is unblocked.
- Raw artifacts: perf/results/2026-08-14/tensor-mma/ (e2e + microbench),
  perf/results/2026-08-14/qgemm-sm-profile/ (ablation + rate probe).

## 2026-08-15 - Plain-Step Decomposition: GEMVs Exonerated, The Gap Is Glue + Host

- Question (user): are we doing something fundamentally wrong, e.g. serving
  a dequantized model? AUDIT: no -- decode routes M=1 to native quantized
  GEMV, M 9..32 to the sm kernels; dequant-then-dense only serves
  prefill-size batches. Weights stream as GGUF blocks; per-step bytes match
  the file (17.96 GB blk + 0.92 GB lm_head = 18.9 GB -> 38.6 ms floor at
  460 GB/s). llama.cpp on this box does 26.75 tok/s plain = 37.4 ms/step =
  AT the floor: the reference proves the hardware can saturate.
- M=1 GEMV kernel-vs-floor microbench (all 8 per-layer shapes + lm_head,
  round-robin min-of-rounds; perf/results/2026-08-15/plain-step-decomp/):
  ffn_gate 0.98x, ffn_up 0.96x, ffn_down 1.03x, lm_head 0.88x (the big
  shapes BEAT the 460 GB/s measured-stream number, i.e. effective BW is
  ~500+); o_proj 1.23x, q 1.44x, attn_gate 1.56x; k/v 6-7x but only 3 us
  floor each (per-op dispatch; the fused step amortizes). Step GEMV total:
  42.7 ms vs 38.6 floor = 1.10x. THE GEMV KERNELS ARE NOT THE PROBLEM.
- In-process plain-decode split (no spec, sync-instrumented; uninstrumented
  ground truth 61.1 ms/step = 16.37 tok/s): layers_fwd 49.6 ms (min 46.4),
  compute_logits 2.1, sampler 2.0, execute_model residue 2.4, engine
  residue ~5-7. Non-GEMV forward (attention + norms + rope + gating +
  residual glue) ~= 49.6 - ~41 = ~8-9 ms. Serving through the API stack
  adds ~10 ms/step more (14.4 vs 16.4 tok/s same workload).
- Revised gap ledger to llama.cpp's 37.4 ms: ~4 ms GEMV stragglers
  (q/attn_gate/o_proj at 1.2-1.6x) + ~8-9 ms attention/glue + ~4 ms
  logits/sampler + ~7-9 ms host/engine (+~10 ms serving stack when through
  the API). The base step, not speculation, is the fundamental deficit;
  spec multipliers match llama.cpp's (1.12x vs 1.15x) on top of a 1.8x
  slower base.
- Attack order (plain step 61 -> ~45 ms target, serving stack included):
  (1) attention/glue inside the fused step (~8-9 ms; profile the fused
  command buffer per-op next), (2) host+engine residue (~7-9 ms; async
  scheduling / overlap encode with GPU), (3) serving-stack ~10 ms
  (API-path overhead), (4) GEMV stragglers (~4 ms; q/attn_gate/o_proj
  staging patterns). Then the same fixes carry to the verify step and the
  spec/tree roadmap multiplies from a healthy base.
- Raw artifacts: perf/results/2026-08-15/plain-step-decomp/.

## 2026-08-15 (2) - Verify/Drafter Split Measured; Device-X Tensor Kernel (v15) Landed

- Fresh verify-step decomposition with the tensor kernels (in-process,
  spec-on, 16.4 tok/s, ~153 ms/step at 2.51 tok/step; harness
  perf/results/2026-08-15/plain-step-decomp/verify_step_timing.py):
  target forward M=17 **119 ms** (was 180.4 pre-tensor-kernels), drafter
  propose 23.7 (= model fwd 12.6 + sample_draft 5.9 + prep glue ~5.2),
  compute_logits 4.8, rejection sampler 3.3, true engine residue ~10.
  NOTE the drafter runs OUTSIDE execute_model in this fork's loop --
  execute_model residue appears negative if you forget that.
  Drafter weights are 1.6 GB -> ~3.5 ms floor; its forward at 12.6 ms is
  3.6x off (dispatch-bound; it never got the muse_step fusion treatment).
- qgemm_sm_t2 (variant 15): device-X flavor of the tensor-op kernel -- X
  (<= ~426 KB, L2-resident) is handed to matmul2d as a device tensor slice
  per K-step instead of being staged to threadgroup memory. Halves
  threadgroup memory to 8 KB (occupancy up), drops the X-stage work and
  half the barrier pressure. Microbench (matched conditions, parity rel
  identical to v14): o_proj 181->174 us, attn_gate 150->89.5 (-40%),
  ffn_gate 415->353 (1.78x floor), ffn_down 423->370 (2.28x). Routed as
  the default for Q4_K/Q5_K/Q6_K (ops.py variant 15); pitfall paid: the
  ggml_mul_mat_sm TORCH_CHECK variant ceiling must be bumped with every
  new variant (first serve crashed 500 on "unknown variant 15").
- E2E (muse-kdyn-1, 256-token greedy triplet, 60 s cool-down): **17.84 /
  15.21 / 13.61 tok/s** -- run-1 matched-position vs yesterday's 16.26 =
  +9.7%. Acceptance 637/390 = 1.63/draft (healthy band; greedy token
  trajectories shift slightly because the device-B operand path changes
  fp accumulation order inside matmul2d). Generations coherent.
- 100 tok/s ledger position: 100 tok/s at ~4.75 tok/step (tree) needs a
  ~48 ms step. Measured step budget today: target fwd 119 (matmul floor
  36.6 + attn/glue ~20), drafter 23.7 (floor ~4), logits+rejection 8,
  engine ~10. Every leg has measured headroom; the two structural jobs
  are (a) verify matmuls from ~2.3x -> ~1.2x floor (uint4b-native
  matmul2d operands are the endgame -- the API takes half x uint4b
  directly; needs load-time byte-neutral repack + per-32-group scale
  correction), (b) drafter fusion (muse_step treatment for the DFlash
  block: 23.7 -> ~8), then (c) engine/serving residue (~10 in-process,
  +~10 through the API server), then (d) tree spec multiplies from the
  healthy base.
- Raw artifacts: perf/results/2026-08-15/plain-step-decomp/ (v15 bench,
  e2e triplet, both step decompositions).

## 2026-08-15 (3) - uint4-Native matmul2d Validated At 1.19x Floor; w8 Routed For Wide Shapes

- 8-warp device-X variant (qgemm_sm_t2w8, variant 16): wins only on the
  widest shapes (ffn_gate 420 -> 384 us, 1.93x floor); regresses on
  attn_gate/ffn_down. Routed for N >= 16384 (ffn_gate/up), v15 elsewhere.
  E2E triplet after route: 16.87 / 16.48 / 14.99 (flatter than the v15
  triplet, run-1 within thermal noise of the 17.84 record; acceptance
  identical 637/390).
- THE DECISION RESULT -- uint4-native matmul2d probe (probe2.metal/.mm,
  archived in perf/results/2026-08-15/plain-step-decomp/): down shape
  (N=6656, K=19968), B operand = packed uint4b weights tile-major
  ([n_tile=64][K][64] contiguous; packed formats forbid strided tensors,
  data_handle_type is device uchar*), A = row-major X^T (the serving
  layout -- the muse_xpose32 transpose disappears), per-32-group scale
  correction via multiply-mode into a tmp cooperative_tensor then
  per-element sc[n,g] fma into the main accumulator (n-coordinates
  precomputed once via get_multidimensional_index):
    u4_raw (no scaling)      196.9 us = 1.21x floor
    u4_scaled (q4_K scheme)  193.7 us = 1.19x floor   <-- TARGET HIT
    u4_k256 (overhead curve) 176.5 us = 1.09x
  vs v15 q4_K down = 370 us. Per-group scaling costs ~nothing; 156
  op.run calls per tg at tilek=32 cost ~20 us total. The min-term
  (dmin*m_g rank-1 correction from per-group X column sums) is not in the
  probe; it is a (N x K/32) @ (K/32 x M) side GEMM, ~1/32 the flops.
- Format coverage decision: q4_K -> uint4 native (4.56 bpw, -43% on down);
  q6_K -> int8 native (+26% bytes but current kernels are 3.6-4x off
  floor: o_proj projected ~71 vs 174 us); q5_K int8 is a WASH on
  ffn_gate/up (+50% bytes eats the efficiency win) -- the q5_K stream
  needs the staged path improved instead (BK=128 GEMV-style contiguous
  runs; the M=1 GEMV kernels prove the layout streams at 0.96x floor when
  access is row-sequential).
- Next session implementation order: (1) load-time repack for q4_K/q6_K
  (tile-major native arrays + scale/min planes; byte-neutral +1-26%, NO
  lazy caches -- load-time replacement per CLAUDE.md); (2) qgemm_sm_u4
  kernel per the probe (include the min-term side GEMM); (3) BK=128
  staged path for q5_K; (4) then drafter fusion (leg 2).
- Raw artifacts: perf/results/2026-08-15/plain-step-decomp/ (v16 bench,
  e2e triplet, probe2 sources + output).

## 2026-08-15 (4) - uint4-Native q4_K Path SHIPPED: 18.26 tok/s, Thermally Flat

- Implemented end-to-end from the probe spec: qgemm_sm_u4 kernel
  (tile-major packed uint4 B operand streamed by matmul2d itself, tilek=32
  multiply into a tmp cooperative tensor, cMain += sc[g,n]*tmp -
  mn[g,n]*xs[g,m]; scatter store into the m-contiguous (4,N,32) partials;
  qgemm_sm_u4_xsum precomputes the rank-1 min-term X column sums;
  deterministic reduce unchanged). Host op ggml_mul_mat_sm_u4 (X (32,K)
  half row-major -- NO transpose glue). Load-time repack
  (quixicore_ops.muse_u4_repack, exact integer nibble extraction, numpy,
  repack self-check rel 0.0): GGUFLinearMethod._create_muse_u4_weight
  registers wu/sc/mn buffers for single-shard Q4_K layers on Metal
  (muse: the 52 ffn_down layers, ~75 MB/layer bounded load-time
  duplicate; original qweight retained for GEMV/prefill routes; kill
  switch VLLM_MUSE_U4=0); apply() routes M 9..32 to the u4 op.
- Kernel parity on real repacked down weights: rel 1.0-1.4e-3 (same band
  as every staged kernel), FIRST TRY -- nibble order low-first, C coop
  coordinate idx[0]=n confirmed. Matched-condition round-robin: u4 315.4
  us (1.75x its 181 us floor incl. scale planes) vs v15 372.1 (2.29x) =
  -15.3%. (Bare-probe 194 us remains the kernel-only bound; the op adds
  xsum + reduce + encode + min-term streams.)
- E2E (muse-kdyn-1, 256-token greedy triplet, 120 s cool-down):
  **18.26 / 18.17 / 17.81 tok/s** -- new record AND thermally flat
  (-2.5% run1->run3 vs -20% for the ALU-heavy staged path: the tensor
  units + no dequant ALU draw less power per token). Acceptance 644/380 =
  1.69/draft; coherent greedy output. Progression: 8.6 -> 11.2 -> 16.26
  -> 17.84 -> 18.26.
- Pitfalls paid: at::zeros on the partials cost ~0 (scatter store covers
  every element; capacity*threads == tile exactly) -- use at::empty; the
  gguf linear.py `ops` symbol is the gguf shim, not quixicore_ops -- new
  entries need both layers; tk_launch.h gives each launcher its own
  template<class E> line -- inserting between them detaches one.
- Next: extend the native-operand route to q6_K (int8 repack: o_proj + q
  projections, projected ~-60%), then the q5_K staged BK=128 path, then
  drafter fusion (leg 2). The u4 op itself still has ~120 us of
  non-GEMM overhead (xsum/reduce/encode) worth folding away.
- Raw artifacts: perf/results/2026-08-15/plain-step-decomp/ (u4 parity
  script + matched bench + e2e triplet).

## 2026-08-15 (5) - q6_K int8-Native: Measured NEGATIVE, Not Routed

- Built qgemm_sm_u8 (int8-native q6_K, -32 folded into load-time repack,
  no min-term, per-16-group scale correction, tilek=16) + host op
  ggml_mul_mat_sm_u8 + exact repack (self-check rel 0.0, kernel parity
  2.5e-4). Tried both strided (K, N) row-major and u4-style tile-major B
  layouts.
- Matched round-robin on o_proj (Q6_K N=6656 K=4096): u8 259-262 us vs
  v15 224 us -> v15 WINS by ~15%. Both sit at ~4x the 67 us floor: the
  muse q6_K shapes are all small-K, where fixed costs (encode, reduce,
  dispatch latency, only 104 tgs/slice of parallelism) dominate and the
  tilek=16 run count adds overhead without a stream win. The u4 win came
  from big-K down (156 32-k groups amortize everything); q6_K has no such
  shape here.
- DECISION: u8 NOT routed (kernel + repack retained as bench-only
  artifacts; no serving-path change). The q6_K/small-K lever is fixed-cost
  reduction (fold reduce into the GEMM kernel, split_k tuning, or the
  fused-verify encoder), not operand format.
- Artifacts: perf/results/2026-08-15/plain-step-decomp/u8_*.

## 2026-08-15 (6) - u4_rm (bf16-in, glue-free) NEGATIVE; Kernel Micro-Arcs Exhausted

- Built qgemm_sm_u4b/_xsum + ggml_mul_mat_sm_u4_rm: original (M, K) bf16
  X straight in (dynamic m extent), (M, N) bf16 out via reduce_rm --
  intended to delete the per-call pad/cast glue. Parity 2.1e-3 (bf16 A
  products; acceptable band) but 431 vs 269 us for u4-with-glue: the
  dynamic-extent A (m=17 < descriptor m=32) forces matmul2d's
  edge-checked slow path on every run. NOT routed; kernels retained
  bench-only. Lesson: keep A extents static at the descriptor shape and
  pad on the host; the pad/cast glue is cheaper than the op's edge path.
- Also note wrapper-inclusive u4 measured 269 us in this (cooler) round
  vs 315-322 in hotter rounds -- the glue overhead estimate (~120 us) was
  inflated by thermal drift; treat ~270 us as the current down cost.
- Kernel-side quick wins are now exhausted. The remaining legs to 100
  tok/s are the structural builds, in this order: (1) drafter fusion
  (23.7 -> ~8 ms/step), (2) q5_K staged BK=128 (gate/up ~10 ms/step),
  (3) small-shape fixed-cost reduction / fused-verify with tensor rm
  kernels, (4) tree speculation (the ~1.7x tokens/step multiplier).
  Serving state: u4 route live, 18.26/18.17/17.81 tok/s validated.

## 2026-08-15 (7) - Native DFlash Input Prep on Metal (Token-Identical); Thermal Wall Reached

- Ported the Triton _prepare_dflash_inputs_kernel to Metal
  (csrc/quixicore/metal/kernels/serving_glue/dflash_prepare.metal + host op
  prepare_dflash_inputs in qc_metal_serving.mm, arg order mirrors the
  python native call; dtypes pinned int32/int64 per buffer with hard
  TORCH_CHECK guards). Gate `_use_native_dflash_inputs` now allows
  Metal, and the
  MPS Python fallback loop (two GPU->CPU .cpu() syncs + ~25 small
  dispatches per propose) is skipped when native is available.
  VLLM_QC_DISABLE_NATIVE=1 still forces the fallback for A/B.
- Correctness: greedy sky-blue probe TOKEN-IDENTICAL to the fallback path;
  acceptance 697/407 = 1.71/draft (healthy); serving clean.
- Performance: e2e 17.80 / 17.83 / 17.71 -- flat but on a heat-soaked
  chassis (in-process decomposition shows target_fwd inflated to 127.7 ms
  vs 119-123 earlier today: pure thermal drift). Instrumented propose glue
  4.2 vs 5.2 ms, but sync-instrumentation recreates the serialization the
  native path removes, so the honest effect is bounded by e2e noise
  (<= ~0.5 tok/s today). Keep it: it deletes real syncs/dispatches whose
  value compounds as the step shrinks, and it is the DSpark/DFlash-common
  path.
- THERMAL WALL: after ~6 h of sustained kernel/bench load every
  measurement is drifting (18.26 cool -> 17.8 plateau hot). Further
  microbench accept/reject decisions today would be noise-driven. Next
  session (cool chassis): re-baseline the triplet, then q5_K staged BK=128
  (gate/up), small-shape fixed-cost fold, drafter model-forward fusion,
  tree spec.
- Artifacts: perf/results/2026-08-15/plain-step-decomp/ (e2e_native_prep,
  verify_step_native_prep).

## 2026-08-15 (8) - BK=128 q5_K Kernel: Narrow Shapes -31%, Routed

- qgemm_sm_t4_q5_K (variant 17): BK=128 device-X tensor kernel with a
  dedicated q5_K paired-plane loader (16 contiguous qs bytes per lane per
  step vs 8 at BK=64; halved K-steps and barriers). host_name required --
  plain mittens-namespace kernels are not reachable by bare pipeline name
  (pitfall re-paid).
- Matched round-robin, parity identical per shape: attn_gate (N=4096)
  240 -> 165.8 us (5.9x -> 4.1x floor, -31%); ffn_gate (N=19968) v16
  378.5 vs v17 403.6 -- wide shapes keep the 8-warp BK=64 kernel.
  Interpretation: at low threadgroup counts the barrier/step count
  dominates; at high counts the wide tiles already amortize it.
- Route: Q5_K narrow (N < 16384, K % 128 == 0) -> 17; wide N -> 16; else
  15. Saves ~74 us x 52 layers ~= 3.9 ms/step (attn_gate).
- E2E on the heat-soaked chassis: 17.86 / 17.37 / 17.14 -- the same
  ~17.8-17.9 hot plateau as the two previous configs; a ~4 ms/step effect
  is below today's thermal noise. Kernel-level matched evidence is the
  decision basis; acceptance 644/380 = 1.69 and coherence unchanged.
- Artifacts: perf/results/2026-08-15/plain-step-decomp/ (v17_bench,
  e2e_v17).

## 2026-08-15 (9) - CRITICAL: Tree-Speculation Multiplier Measured -- 100 tok/s Ledger Revised

- Offline tree-acceptance study (tree_acceptance_study.py + results in
  perf/results/2026-08-15/plain-step-decomp/): logged the drafter's top-8
  candidates at every block position across 3 diverse greedy prompts
  (234 verify steps), replayed against the true continuations, computed
  tokens/step for full product-tree topologies. Alignment advances by the
  REAL linear emission (each logged block is conditioned on the real
  acceptance boundary); the numbers are therefore tight UPPER bounds on
  real tree acceptance.
- Results: linear k=1 sanity = 2.46 tok/step (matches production 2.5-2.7).
  Buildable <=48-node trees: < ~3.0. A 498-node tree: 3.09. A physically
  impossible 611,668-node tree: 3.47. The projected 4.5-4.75 is
  UNREACHABLE with this drafter: per-position acceptance decays 0.77 /
  0.50 / 0.19 / 0.11 -- beyond position ~3 the drafter has no signal, and
  no verification topology recovers information the proposals do not
  carry. (Consistent external evidence: llama.cpp's best spec config on
  this box is +15%, and its Apple-row card ceiling was ~50 tok/s.)
- REVISED CEILING: with tokens/step capped at ~2.7 (linear) to ~3.0
  (cheap tree), 100 tok/s would need a 27-30 ms step -- BELOW the 38.6 ms
  weight-pass floor. Physically impossible with this model+drafter
  artifact. Best reachable with everything else perfect: ~48-55 ms step
  x 2.7-3.0 tok/step = **~50-62 tok/s**.
- Path to an actual 100 requires a better DRAFTER ARTIFACT (training-side:
  deeper-position DFlash conditioning, larger drafter, or DSpark-style
  confidence truncation trading depth for width) -- outside the serving
  stack's control. The serving-side roadmap (drafter fusion, wide-q5_K,
  fixed costs, attn/glue, engine) remains valid toward the ~50-62 ceiling.
- Tree spec at <=48 nodes is still worth ~+10-20% AFTER the step
  approaches the floor (at a 50 ms step, 2.7 -> 3.0 = +5 tok/s for ~+15%
  step cost -- marginal; re-evaluate when the step is floor-bound).

## 2026-08-15 (10) - Wide-q5_K Wall Hunt: Partials Machinery Implicated; split1 Store Bug Open

- t2w8 ablations (variants 28-30: W-stage off / MMA+device-X off / dequant
  raw) on ffn_gate: ALL slower than the full kernel (383 -> 408-410) --
  no single stage dominates the wide shape's 1.93x floor. What every
  variant shares: the split-K partials machinery (~10 MB write + ~10 MB
  reduce re-read + reduce kernel ~= 60+ us on a 383 us matmul) and the
  launch structure. Prime suspect for the wide-shape fixed cost.
- Variant 31 (qgemm_sm_t2w8s1: SPLIT_K=1, direct half scatter store, no
  partials/reduce): timing suggestive (-43% in-process vs an inflated
  same-run baseline -- NOT trustworthy today) but PARITY FAILS (rel
  ~0.7-0.9) with an unresolved cooperative-tensor coordinate mapping in
  the manual scatter store (both index orders wrong; cT.store on the same
  cT is correct, so the accumulator itself is fine). NOT ROUTED --
  bench-only. Debug on a sane chassis: dump get_multidimensional_index
  pairs for the execution_simdgroups<8> destination layout, or use
  cT.store into a (N,32) float scratch + separate cast pass as the
  correctness-first fallback.
- Thermal state is now unusable for accept/reject decisions (v16
  measuring 716 us vs its true ~380; v15 down 737 vs ~370). Stopping
  measurement-driven work per the matched-conditions discipline.
- Artifacts: perf/results/2026-08-15/plain-step-decomp/
  (q5k_wide_ablation, v31_bench*).

## 2026-08-15 (11) - split-K=1 Kernel Fixed + Routed (Measured Shapes Only); Cool Re-Validation Pending

- v31 parity bug ROOT-CAUSED with a coordinate-dump probe (probe3 in the
  scratchpad, output in session log): for execution_simdgroups<8> the
  destination cooperative tensor's element slots are HALF invalid and some
  valid slots hold cross-simdgroup PARTIAL sums that only combine inside
  cT.store -- manual element readout is layout-unsafe (the 4-simdgroup u4
  config merely happened to have a complete-element layout). PITFALL for
  all future coop-tensor code: never scatter cT[i] manually; only
  cT.store() is layout-safe.
- Fix: v31 stores one float slice via cT.store (z=0) and the host folds it
  with the existing reduce at SK=1 (a float->half cast). Still deletes 3/4
  of partials traffic + 3/4 of reduce reads vs split-K=4.
- Parity: 3.4-8.9e-04 on ffn_gate / ffn_down / o_proj. Hot-chassis MATCHED
  round-robin (valid relatively): ffn_gate +43%, ffn_down +33% (also beats
  the u4 layer intercept head-to-head), o_proj +44%.
- ROUTE LESSON (regression caught): a first universal route (all N%64==0)
  sent v31 to UNMEASURED narrow shapes (merged QKV N=4608 -> 72
  threadgroups, attn_gate N=4096 -> 64) where split-K=1 loses too much
  parallelism; e2e dropped to ~13-14. Corrected to measured shapes only
  (N >= 6656: down/gate/up/o_proj); narrow Q5_K keeps v17, rest v15. The
  u4 layer intercept is now opt-in (VLLM_MUSE_U4=1) -- superseded by v31
  on its own shape.
- E2E today is UNRESOLVABLE: the chassis is in deep thermal recovery
  (ascending triplets 13.7 -> 15.3 across a bench; each 20 GB model load
  re-saturates it). The 18.26 record and this route MUST be re-validated
  on a cool chassis before further e2e claims; if cool e2e regresses,
  the previous route is the 15/16/17 selection in git history.
- Artifacts: perf/results/2026-08-15/plain-step-decomp/ (v31_parity_final,
  e2e_v31, e2e_v31b).

## 2026-08-15 (12) - Rested-Chassis Protocol: v31 Route Reads 16.05 Flat; Same-Protocol A/B Running

- New protocol: background job rests the chassis 45 min truly idle, then
  serves, settles 120 s, and runs the triplet (script archived in the
  session scratchpad; results e2e_v31_cool.txt). v31 route: **16.05 /
  16.08 / 16.09** -- perfectly flat (+-0.04!) but BELOW the 18.26/18.17/
  17.81 the u4-route config recorded midday yesterday under a different
  thermal prehistory.
- METHODOLOGY HOLE EXPOSED: hot-chassis matched round-robin wins do not
  automatically extrapolate to cool conditions -- under throttle, fixed
  costs (partials traffic, reduce dispatches) inflate disproportionately,
  so a fixed-cost-deleting kernel (v31) looks better hot than it may be
  cool. Absolute cross-day comparisons also depend on thermal prehistory
  (45-min overnight rest != midday 120-s cool-down).
- Discriminator IN FLIGHT: identical rest protocol with the legacy route
  (VLLM_SM_ROUTE=legacy env gate added to quixicore/ops.py + VLLM_MUSE_U4=1
  restores the u4 intercept) -- same 45-min rest, same settle, same bench.
  The pair decides the route on same-protocol evidence; whichever loses is
  kept behind the env gate for future A/Bs.
- Acceptance unchanged throughout (651/375 = 1.74/draft).

## 2026-08-15 (13) - Same-Protocol A/B DECIDES: Legacy Route Wins, v31 Reverted To Opt-In

- Identical 45-min-rest protocol, both routes: legacy (15/16/17 + u4
  intercept) **17.81 / 17.56 / 17.58** vs split-K=1 route (v31 on
  N >= 6656) 16.05 / 16.08 / 16.09. Legacy wins +9.6%. Acceptance
  identical (1.69-1.74/draft).
- v31 is therefore an E2E NEGATIVE despite sweeping hot-matched
  microbenches (+33-44%): (a) throttle inflates exactly the fixed costs
  (partials traffic, reduce dispatch) that v31 deletes, so hot-matched
  comparisons overweight them; (b) split-K=1 quarters per-matmul
  threadgroup parallelism, which plausibly degrades inter-op overlap in
  the real step -- isolated microbenches cannot see either effect.
- ROUTE DECISION: default reverted to the 15/16/17 selection with the u4
  intercept ON (the A/B winner); v31 kept behind VLLM_SM_ROUTE=split1.
  METHODOLOGY RULE going forward: hot-matched microbenches select
  CANDIDATES; route flips require the same-protocol rested A/B
  (scratchpad cool_bench scripts; 45-min idle + 120 s settle + triplet).
- NEW CANONICAL BASELINE (rested protocol, winning route): **17.81 /
  17.56 / 17.58 tok/s**. The midday 18.26 remains the observed best
  (different thermal prehistory); use the rested protocol for all future
  route comparisons.
- Raw: e2e_v31_cool.txt, e2e_legacy_cool.txt.

## 2026-08-15 (14) - Forward Attribution: MLP 51% / Attention 32% / Glue 17%

- Class-level sync attribution of the M=17 verify forward (35 steps,
  fwd_attribution.py; shares meaningful, absolutes sync-inflated):
  MLP 51%, attention 32%, norms/gating/embed/glue residue 17%.
- Against measured kernel costs (~80-85 ms of matmuls in the rested
  state), the forward carries roughly ~35-45 ms/step of NON-matmul work:
  paged attention at M=17, rope, q/k norms, 4 layernorms x 52 layers,
  residual/gating glue -- hundreds of small dispatches.
- Implication: the parked fused verify (muse_step_run_aux, one command
  buffer) is the right next arc -- it was abandoned at 199.7 vs eager
  180.4 ONLY because its matmuls used the old 2-warp simdgroup
  qgemm_sm_rm. Port plan: replace launch_qgemm_sm_rm inside the fused
  step with muse_xpose32 (already present for the deep-K route) +
  launch_qgemm_sm at the production variants (15/16/17) + the existing
  reduce_rm fold. Accept gate: rested 45-min A/B, VLLM_MUSE_FUSED_VERIFY
  on vs off.
- Raw: fwd_attribution_results.txt.

## 2026-08-15 (15) - FUSED VERIFY RE-FLIPPED: Tensor Kernels Inside The Fused Step Win

- The port was one function: emit_matvec in the fused muse_step already
  did xpose32 + launch_qgemm_sm + reduce_rm -- it was pinned to the old
  variant table (9-12). Updated to mirror the eager route (15/16/17, the
  rested-A/B winner) for k-quants with N % 32 == 0.
- Matched in-process fused-vs-eager (muse_verify_timing.py, back-to-back
  same process): fused target forward M=17 **95.6 ms** (min 93.8) vs
  eager **110.4** (min 107.0) -- fused wins by 14.8 ms/step (-13%),
  exactly the re-flip the 08-13 entry predicted once kernels got fast
  enough for encoder overhead to matter again. In-process rates during
  the run: 18.5-18.6 tok/s.
- Default flipped ON (VLLM_MUSE_FUSED_VERIFY=0 reverts). Rested-protocol
  e2e confirmation (45-min idle + settle + triplet + acceptance +
  coherence) launched in background; gate vs the canonical rested
  baseline 17.81/17.56/17.58.
- The forward attribution (entry 14) explains the win: ~35-45 ms of the
  eager forward is non-matmul dispatch/glue that the single command
  buffer absorbs.

## 2026-08-15 (16) - Fused Verify CONFIRMED E2E: 20.1 tok/s Rested, New Canonical Baseline

- Rested-protocol e2e with the fused verify default ON: **19.94 / 20.14 /
  19.74 tok/s** vs the same-protocol eager baseline 17.81/17.56/17.58 =
  **+13%**, matching the in-process forward delta (95.6 vs 110.4 ms)
  almost exactly. Acceptance 649/375 = 1.73/draft (unchanged); coherent
  greedy output. THIS IS THE NEW CANONICAL RESTED BASELINE.
- Campaign progression (spec-on): 8.6 -> 11.2 -> 16.26 -> 17.84 -> 18.26
  (observed best, midday) -> canonical rested 17.81 -> **20.14** with the
  fused verify. 2.34x over campaign start; +24% today alone.
- Step ledger position (rested): step ~= 127 ms = fused forward ~96 +
  drafter ~21 + logits/rejection ~8 + engine ~8-10. Remaining measured
  legs: drafter fusion (~15 ms; same emit_matvec/fused-step treatment now
  proven twice), sample_draft glue (~3), engine residue, wide-q5_K stream
  inside the fused forward, then the on-device rejection + drafter
  proposer fusion the 08-13 plan named. Ceiling unchanged: ~50-62 tok/s
  serving-side; 100 remains gated on a drafter artifact meeting
  perf/drafter_requirements.md.
- Raw: e2e_fused_cool.txt.

## 2026-08-15 (17) - lm_head v31 Candidate: Wash (+1.8%); Cheap Candidates Exhausted

- Matched bench of the shared lm_head (Q5_K, N=202048, streamed TWICE per
  step: target compute_logits M=17 + drafter sample_draft M=16, ~8.5 ms
  combined): v16 3430 us (1.71x floor) vs v31 3367 (1.68x) -- +1.8%,
  within noise. The wide head is stream-bound; the split-K partials cost
  amortizes over N=202048. No route change. Also confirmed: greedy
  drafting already takes the fast path (draft_logits is
  probabilistic-only), so sample_draft's remaining ~2 ms is glue.
- With this, the sub-session-sized candidates are exhausted. The next
  item of size is the drafter-forward fusion (bespoke emit-chain against
  DFlashQwen3Model incl. the plain-torch context-KV precompute;
  architecture notes in HANDOFF next-steps) -- a fresh-session arc.
- Session-final state: canonical rested baseline **19.94 / 20.14 / 19.74
  tok/s** (fused verify default, tensor kernel route). Campaign 8.6 ->
  20.14 (2.34x). Serving ceiling ~50-62 with the current drafter; the
  100 tok/s gate is the drafter artifact per perf/drafter_requirements.md.

## 2026-08-15 (18) - bf16 Drafter Measured: Quantization Is NOT The Collapse; Gate Confirmed

- Hypothesis tested: the drafter's positional acceptance collapse might be
  Q4_K quantization noise (deep positions have thin logit margins).
  Downloaded the published bf16 original
  (meta-models/Muse-Glimmer-30B-assistant, 5.1 GB -- SAME 5-layer
  architecture per its config) and ran the acceptance harness
  (DRAFTER=bf16 switch added).
- RESULT: statistically identical to the quantized drafter. Linear 2.52
  (vs 2.46), buildable trees ~3.0-3.1 (vs ~3.0), impossible 611k-node
  tree 3.46 (vs 3.47). **The collapse is architectural/training-depth,
  not precision.** No published artifact reopens the 100 tok/s path; the
  gate is a NEW drafter training effort per perf/drafter_requirements.md
  (memo stands, now with the precision variable eliminated).
- Infrastructure landed en route (keep -- it is the candidate-drafter test
  rig): (a) drafter load_format independence (a GGUF target no longer
  forces load_format=gguf onto a safetensors drafter; dflash/utils.py);
  (b) published-name remap in MuseGlimmerDFlashDraftModel.load_weights
  (encoder.fc -> model.fc, encoder.output_norm_enc -> model.hidden_norm,
  bare layers./norm. -> model.*); (c) model.mask_embedding marked as
  shared-loaded; (d) local config.json patches needed for the published
  checkpoint (documented in drafter_requirements.md): model_type -> qwen3,
  architectures -> DFlashMuseGlimmerDraftModel, vocab_size 202048,
  dflash_config.swa_window_size 2048.
- Search also confirmed: the muse-glimmer collection ships exactly ONE
  drafter architecture (GGUF quant + this bf16 original); no deeper
  variant exists publicly.

## 2026-08-15 (19) - Post-Fused-Verify Step Ledger (current shipped state)

- In-process split with the fused verify active (verify_step_timing.py +
  drafter_ctx_kv wrap; sync-instrumented): target fwd M=17 **97.0 ms**
  (was 119-127 eager -- fused win confirmed in-harness), drafter propose
  20.0 (= model fwd 11.3 [floor 3.5] + sample_draft 4.9 [shared lm_head
  ~3.4 + glue] + ctx-KV precompute 2.3 [plain-torch, ~35 small ops] +
  prep glue 1.5), compute_logits 4.1, rejection 4.1, engine residue ~31
  instrumented (~8-10 true). 18.6 tok/s instrumented in-process.
- Remaining serving items by measured size: drafter forward fusion
  (~8 ms, the bespoke DFlashQwen3 emit-chain -- the one remaining
  multi-hour arc), engine residue (~8-10), sample_draft/ctx-KV glue
  (~3-4 combined, high effort-to-win ratio), wide-q5_K stream inside the
  fused forward. Note: the two shared-lm_head passes (target 4.1 +
  drafter 3.4) cannot be batched -- they sit on opposite sides of the
  rejection dependency.
- The step is now ~127 ms rested (20.14 tok/s). All remaining serving
  work compounds toward the measured ~50-62 ceiling; the 100 gate remains
  the retrained drafter (requirements + test rig in
  perf/drafter_requirements.md, precision variable eliminated).

## 2026-08-15 (20) - Fused Drafter Step: Foundation Built (Inert), Splice Specified

- Built and verified (compiles, binds, importable; NOTHING calls it yet --
  serving unaffected): `dflash_step` namespace in qc_metal_serving.mm
  (init/layer/run mirroring muse_step, specialized: 5 layers, no
  attention gate, standard pre-norm residuals, all-SWA window 2048) plus
  a `muse_rope_qk_neox` kernel (half-split pairing; the drafter keeps
  NeoX while the target is interleaved -- muse_glimmer.py:155).
- Remaining splice (the risk-bearing part, for a fresh session):
  (a) registration in MuseGlimmerDFlashModel (walk self.layers; use the
  same shards() helper pattern as muse_glimmer.py ~line 320 for
  qkv_proj/gate_up_proj; kv_cache from attn.attn.kv_cache);
  (b) block-BIDIRECTIONAL attention metadata: per-row seq_len = base + m
  for EVERY row (the drafter denoises the whole block; contrast the
  verify path's causal base + i + 1), stride-0 expanded block table +
  slots exactly as `_maybe_fused_decode` builds them (read its metadata
  construction below line 395 of muse_glimmer.py first);
  (c) gated forward override (VLLM_MUSE_FUSED_DRAFTER=1, default OFF,
  try/except -> eager fallback like the target's pattern), splicing after
  the input assembly (embeds + fc encoder stay in torch initially);
  (d) validation gate: drafter outputs bit-comparable vs eager (token-
  identical greedy e2e), then propose timing (expect model fwd 11.3 ->
  ~5-6 ms), then the rested A/B.
- Expected win when landed: ~5-6 ms/step (~+1 tok/s at the current step).

## 2026-08-15 (21) - Fused Drafter Spliced End-To-End; Engages Cleanly; Proposals WRONG (Gated Off)

- The Python splice landed: _init_fused_step registration (incl. reusing
  _gguf_hetero_shards for the mixed-type QKV -- the merged buffer is
  RELEASED after warmup's eager forwards, a dim-1 IndexError otherwise),
  _maybe_fused_forward (bidirectional block: bt/sl expanded with
  seq_lens_gpu[:1] AS-IS for every row -- no per-row causal steps), gated
  VLLM_MUSE_FUSED_DRAFTER=1 with try/except eager fallback.
- With the gate ON: the fused path ENGAGES (zero fallbacks), serving is
  healthy, and greedy output is character-identical (the rejection
  invariant guarantees correctness regardless of drafter quality) -- but
  acceptance CRATERS to 1/58 vs ~2.5 expected on the probe: the fused
  drafter forward computes wrong proposals. DEFAULT REMAINS OFF; the tree
  is healthy.
- Debug plan (fresh session): activation-diff rig -- run eager and fused
  drafter forwards on identical inputs (the muse_verify_timing pattern),
  compare per-layer outputs. Suspects in order: (1) rope convention
  (kernel muse_rope_qk_neox implements half-split (j, j+half); check
  attn0.rotary_emb.is_neox_style at runtime -- qwen3_dflash.py:497 reads
  it, and an interleaved A/B is a one-line pipeline swap); (2) attention
  expansion semantics for the non-causal block (seq_len = base + m per
  row); (3) q/k norm eps or application shape; (4) kv_store slot layout
  vs the drafter cache group. NOTE the drafter GGUF has no
  attention_sink_bias tensor (ruled out).
- Infrastructure state: dflash_step (init/layer/run) + muse_rope_qk_neox
  compiled and bound; splice complete; ONLY the numerical bug remains
  between here and the ~5-6 ms/step win.

- A/B addendum: swapping the fused drafter to INTERLEAVED rope produces
  the identical 1/58 crater -- rope convention is ruled out as the sole
  bug. Remaining suspects for the activation-diff rig: attention
  expansion semantics for the non-causal block, q/k-norm application, kv
  slot layout, and the input-embed path (mask embedding substitution
  ordering). Kernel reverted to neox (the drafter's documented style).

## 2026-08-15 (22) - FUSED DRAFTER FIXED AND VALIDATED: The Bug Was Causal Semantics

- The activation-diff rig (fused_drafter_diff.py, iterated through 14
  output snapshots) localized and fixed the fused drafter in three moves:
  (1) STRIDE-0 SEQ_LENS: the kernel reads seq_lens as a contiguous int32
  array; a bare .expand(m) passed one element and rows 1+ read garbage
  (row-0-perfect/rest-garbage signature). (2) THE REAL SEMANTICS BUG:
  this fork's DFlash runs SWA layers CAUSAL (qwen3_dflash.py:59 -- no
  dflash_config.causal override in the published config), NOT
  block-bidirectional as llama.cpp's global non-causal suggested; the
  expansion is the target verify's exact per-row causal construction
  (base + i + 1). (3) A rig artifact (_fused_ready is an INSTANCE attr;
  class-level reset silently reused 1-layer registrations) manufactured a
  fake layer-1 divergence -- worth remembering for future monkeypatch
  rigs.
- End state: every layer matches eager at quantization precision
  (7-11e-3), full 5-layer forward rel 6.99e-03. Production under the
  gate: token-identical greedy, acceptance 695/392 = 1.77/draft (normal),
  zero fallbacks, warm-chassis bench 19.53/19.12/19.32 (suggestive +1
  vs the ~18.5 warm state).
- Rested A/B vs the canonical 19.94/20.14/19.74 running in background;
  flip the default on a win. Additional fixes landed en route: fused-
  verify m-gate (m in {3,5,6,7} would hit missing qgemv_mm pipelines --
  LATENT PRODUCTION CRASH exposed by the fused-verify default flip, now
  guarded), and the debug-buffer mlp_h split in dflash_step.
- Artifacts: fused_drafter_diff.py + out1..14 in
  perf/results/2026-08-15/plain-step-decomp/.

## 2026-08-15 (23) - Fused Drafter Rested A/B: WASH; Default Stays OFF

- Rested protocol, fused drafter ON: 19.87 / 19.73 / 19.68 vs canonical
  OFF 19.94 / 20.14 / 19.74 -- within protocol noise, acceptance
  identical (1.74/draft). No flip.
- Why no win: the drafter is only 5 layers -- its ~11.3 ms forward is
  mostly kernel time (~7 ms of matmuls at ~2x floor) + attention, not the
  dispatch storm that made the 52-layer target fusion worth 15 ms. The
  fused chain also pays one muse_xpose32 per matmul.
- The path RETAINS value as infrastructure: (a) parity-exact fused
  drafter forward, token-identical in production, behind
  VLLM_MUSE_FUSED_DRAFTER=1; (b) the platform for fusing MORE of propose
  (ctx-KV precompute + draft sampling into the same command buffer),
  where the real remaining drafter overhead lives (~8 ms of
  sample+ctx+glue); (c) re-test the flip when the step shrinks further
  and dispatch fraction grows.
- Session tally on the fused-drafter arc: 1 semantics discovery (fork
  DFlash SWA layers are CAUSAL), 1 latent production crash fixed (fused
  verify m-gate), 1 kernel-ABI rule confirmed (contiguous seq_lens), all
  documented.

## 2026-08-15 (24) - Fused Greedy Draft Sampling: Correct, Neutral, Config-Gated

- Built dflash_sample_greedy (one command buffer: shared-lm_head GEMM via
  the wide-N tensor kernel + new muse_argmax row-reduce kernel;
  softcap/logit-scale argmax-invariant and skipped) + get_top_tokens on
  MuseGlimmerDFlashDraftModel, integrating through the EXISTING
  use_local_argmax_reduction speculator hook -- zero propose
  restructuring. Engages cleanly (init log), token-stream and acceptance
  unchanged.
- Measured (sync-instrumented in-process): drafter_sample 5.17 vs 4.93 ms
  baseline -- NEUTRAL; the lm_head GEMM dominates both paths and the op
  saves only ~2 small dispatches (<= ~1 ms un-synced, below the rested
  protocol's noise floor at the current step size). Per methodology: NOT
  enabled in the profile; available via
  speculative_config.use_local_argmax_reduction for when the step
  shrinks.
- Drafter ledger after today: model-forward fusion (wash), sample fusion
  (neutral), prep glue (done), ctx-KV (2.3 ms, unfused -- fold into the
  fused drafter chain when that path gets re-tested). The drafter's
  remaining recoverable at current step size is small; the big open items
  are ENGINE RESIDUE (~8-10 ms in-process + the API-server gap) and the
  WIDE-q5_K STREAM (gate/up/lm_head at 1.7-1.9x floor, ~12+ ms/step).

## 2026-08-15 (25) - Engine Residue Decomposed: The Loop Is Free; The Ledger Fully Closes

- Engine-side wraps (pure-CPU timing): sched_schedule 0.04 ms,
  sched_update 0.03, runner_add_requests 2.75, worker_sample_tokens
  30.49. The step accounts EXACTLY: execute_model 105.0 (= fused target
  fwd 98.0 + logits 4.15 + glue ~3) + sample_tokens 30.5 (= drafter
  propose 20.0 + rejection 4.15 + sampling/bookkeeping ~6) + scheduler
  ~0.1 = 135.6 vs measured wall 136.3 ms/step. THE "ENGINE RESIDUE" WAS
  AN ACCOUNTING ARTIFACT -- the drafter runs inside sample_tokens,
  outside execute_model; the engine loop itself is free.
- Fully-closed step ledger (in-process instrumented; rested e2e ~127 ms):
  target forward 98 (floor 38.6 -- the gap is the wide-q5_K stream +
  attention inside the fused step), drafter 20 (worked to its practical
  floor today), logits 4.2, rejection 4.2, sampling glue ~6,
  add_requests 2.8.
- Strategic read: tail+drafter perfected saves ~25 ms -> ~110 ms step ->
  ~25 tok/s. The dominant remaining lever is the TARGET FORWARD's ~60 ms
  gap to floor, which is (a) the wide-q5_K stream (gate/up ~40 ms at
  1.9x floor; no native operand exists for 5-bit -- open kernel
  research), and (b) attention + residual glue (~20 ms). This is where
  all future serving sessions should aim first.

## 2026-08-15 (26) - CONTEXT SCALING MEASURED: The Bench Prompt Was Hiding The Real Bottleneck

- Decode throughput vs context length (192-token greedy, thermal-checked
  with a short-ctx rerun): FUSED verify 20.25 tok/s @ 78-token prompt ->
  **8.15 @ 1734** (short-again 20.05 confirms non-thermal). EAGER verify:
  14.36 -> 7.27 -- BOTH paths crater ~2.5x at only 1.7k context. This
  model's mission is long-context agentic work (131k max); the 78-token
  essay bench has been masking the dominant production bottleneck.
- Mechanism: step grows ~+200 ms at 1.7k ctx. The verify attention's
  per-virtual-row KV re-read (17 rows x ctx x 52 layers ~= 6.1 GB) only
  accounts for ~13 ms of that -- the paged attention kernel's per-row
  scan is ~15x off bandwidth at length (latency-bound serial K-walk).
  Proper multi-query attention at this shape should cost ~1-2 ms.
- PRIORITY REORDER (supersedes entry 25's ordering): the #1 serving item
  is now a MULTI-QUERY VERIFY ATTENTION kernel -- one shared KV pass for
  all 17 query rows with the per-row causal boundary applied in-kernel
  (and the drafter's block equivalently). Worth 2-3x at production
  context lengths, far more than the wide-q5_K stream's ~10%. Applies to
  BOTH the fused and eager paths (both measured equally afflicted; no
  config-level mitigation exists).
- Bench methodology fix required: add a long-context arm (>= 2k prompt)
  to the standard bench so this class of regression is visible; the
  short-prompt triplet alone is not representative of the product
  workload.

## 2026-08-15 (27) - CORRECTION to (26): The "Context Crater" Was Prefill Contamination

- Entry (26)'s 20.25 -> 8.15 finding conflated PREFILL with decode: the
  single-shot API bench divides completion tokens by a wall that includes
  processing the 1734-token prompt. With the prompt prefix-CACHED
  (run 2+), long-context decode measures **16.10 / 16.11 tok/s** on the
  expansion path vs 14.55 on the SDPA-gather path -- the expansion route
  WINS at 1.7k context, and the true decode penalty vs short context is
  ~20% (20.25 -> 16.1 at ~330 -> ~1900 ctx; +35 ms/step, consistent with
  the linear KV re-read + kernel factor).
- Consequences: (a) the multi-query verify attention kernel drops from
  measured-#1 back to a LARGE-context item (the linear re-read still
  crosses over somewhere past ~2k -- measure at 8k/32k before building);
  (b) the VLLM_EXPAND_CTX_MAX threshold knob is retained defaulting to
  no-op (backend + fused gate), ready for that measurement; (c) the m-gate
  crash fix from this arc remains valid and shipped.
- METHODOLOGY RULE (added to the bench discipline): single-request API
  benches measure decode ONLY from cached-prompt repeat runs (or subtract
  prefill via usage timings). Run-1 of any new prompt is a
  prefill+decode composite. Entry (26)'s bench-arm recommendation stands,
  with this correction applied.
- Prefill observation for the record: 1734 tokens prefilled in ~13-16 s
  (~110-130 tok/s) -- slow, a separate workstream if prefill latency ever
  matters for the product.

## 2026-08-15 (28) - 8k-Context A/B: Both Attention Routes Converge; Degradation Source Is SHARED And Unidentified

- Cached-prompt decode at 9871-token context: EXPANSION 9.08/8.88 tok/s
  vs SDPA-gather 9.14/9.19 -- a wash. Context scaling of decode
  (cached): ~330 ctx -> 20.25, ~1.9k -> 16.1, ~9.9k -> ~9.1.
- The multi-query/17x-re-read theory is DEAD as the primary mechanism:
  SDPA reads each key once yet degrades identically. The shared +167
  ms/step at 10k is NOT the KV bytes either (9871 x 4 KB x 52 layers
  ~= 2 GB/step ~= 4.5 ms at stream rate). Something both paths pay
  scales with context: candidates for the next in-process long-ctx
  decomposition -- (a) the paged/SDPA attention KERNEL time itself
  (per-key latency far off stream rate in both), (b) drafter-side
  context work (its 5-layer attention + any per-step context-length
  machinery), (c) per-step host prep proportional to blocks
  (block-table/expansion tensor builds at 617 blocks), (d) KV-cache
  locality (2 GB working set vs SLC).
- NEXT MEASUREMENT (bounded): the verify_step_timing harness with an
  ~8k cached prompt -- the existing per-stage wraps will attribute the
  +167 ms directly. Do this BEFORE building any attention kernel; the
  fix follows the attribution.
- The VLLM_EXPAND_CTX_MAX knob stays no-op by default (paths equivalent
  at every measured length).

## 2026-08-15 (29) - Long-Context Attribution: It IS The Attention Kernels (Precisely, Now)

- In-process decomposition at 9871-token cached context: target fwd M=17
  **207.8 ms** (vs 98.0 short: +110), drafter fwd 20.9 (vs 11.2: +10),
  ALL other stages flat (logits 4.2, rejection 3.1, ctx-KV 2.5, sample
  5.1, add_requests 2.8, scheduler ~0.07). The entire long-context loss
  is the two forwards' ATTENTION; matmuls do not scale with ctx.
- Per-layer arithmetic at 9.9k: ~2.1 ms of attention per target layer vs
  a ~87 us KV-stream floor (40 MB/layer; window-clamped local layers
  lower the average) -- the per-row scan is ~25x off stream rate. The
  SDPA gather pays an equivalent price differently (2+ GB of index_select
  gather per step), which is why (28) measured a wash.
- THE BUILD SPEC (next session-scale arc, now fully evidence-backed): a
  flash-decode-style VERIFY ATTENTION kernel -- per (kv_head, KV-chunk)
  threadgroup streams K/V once, scores all 17 query rows (x4 GQA heads)
  per chunk with online softmax and per-row causal boundary + window
  clamp, partials reduced across chunks. Floor ~4.5 ms total at 10k ctx
  vs ~110 today; also fixes the drafter (same kernel, 5 layers).
  Applies to both eager (backend route) and fused (launch swap) paths.
- Bench-arm note: the repetitive 8k filler prompt drops acceptance to
  2.06 tok/step (drafter sees degenerate text) -- long-ctx bench arms
  should use natural long documents, not repeated filler.

## 2026-08-15 (30) - Multi-Query Verify Attention: KERNEL PROVEN; E2E Composition Open

- BUILT AND VALIDATED at kernel level: paged_attention_verify (m rows
  cooperate per (head, partition) threadgroup; K/V tiles staged once per
  tg instead of once per row; per-row causal boundary + window in-kernel;
  partial layout shared with the v2 reduce). Parity EXACT vs the
  expansion kernel and SDPA at every tested config. Timing per layer
  (M=17, D=128, GQA 32/8): ctx 330: 666 vs 208 us (expansion wins short);
  1.9k: 695 vs 847 (+1.2x); **9.9k global: 1661 vs 6460 (+3.9x)**; 9.9k
  windowed: wash (expansion keeps SWA layers). Routed: eager backend
  sends global-layer verify to the MQ kernel at ctx > 1024; the fused
  verify hands off to eager past VLLM_FUSED_VERIFY_CTX_MAX (default
  3072) since its encoder does not yet call the new kernel.
- E2E at 9.9k cached: 8.95 vs 9.08 baseline -- NEUTRAL so far. Either the
  route is not engaging as intended (verify with a branch log) or the
  ~60 ms/step attention win on 13 global layers is offset by eager-mode
  overheads at a ~300 ms step (fused encoder loss ~15 ms + eager glue).
  NEXT: in-process attribution at 10k with fused OFF + a one-shot
  engagement log; if engaged-and-offset, plumb the MQ kernel INTO the
  fused encoder (pass ctx_len; swap the attention launch for global
  layers) to keep both wins.
- Also observed: short-ctx cached decode 22.24 tok/s (best single
  observation of the campaign; protocol-uncomparable to the 20.14
  canonical -- noted, not claimed).
- Artifacts: verify_attn_parity.py + outputs.

## 2026-08-15 (31) - EAGER LONG-CONTEXT CORRECTNESS BUG FOUND AND FIXED (via the MQ route debug)

- Chasing the MQ route's non-engagement exposed a REAL quality bug:
  vLLM's Attention() falls back to cache_config.sliding_window when
  per_layer_sliding_window is None, so ALL 52 muse layers -- including
  the 13 global NoPE layers -- carried window=2048 in the eager path
  (confirmed: per-impl logging showed window=2048 on every instance).
  Beyond 2048 context, eager mode silently clamped the model's
  long-range attention. The fused path always registered window=0 for
  globals and was correct -- which also means the entire pre-fused era of
  this campaign served DEGRADED outputs beyond 2k context in eager mode.
- Fix: MuseGlimmerAttention forces attn.impl.sliding_window = None for
  global layers post-construction (muse_glimmer.py, with the mechanism
  documented). Validation at 9.9k ctx (eager): acceptance recovered
  2.06 -> 2.78 tok/step on the same prompt (wrong attention had been
  suppressing target/draft agreement), throughput 7.92 -> 10.19 despite
  correct global layers costing MORE than wrongly-clamped ones -- the MQ
  kernel (now engaging, confirmed by log) is what makes correct
  affordable (13 x 1.66 ms vs 13 x 6.46 expansion).
- Fused remains the default at all contexts (it was always correct and
  is still fastest); the VLLM_FUSED_VERIFY_CTX_MAX handoff reverts to
  no-op. THE REMAINING ITEM: plumb paged_attention_verify into the fused
  encoder's global-layer attention site (ctx_len arg + launch swap +
  partials scratch) -- projected fused fwd at 10k: 207.8 -> ~145 ms.
- Product note for the record: any deployment that ran EAGER (pre-today
  default) beyond 2048 context served silently degraded long-range
  attention. The fix is in; short-context behavior is unchanged.

## 2026-08-15 (32) - MQ Kernel Plumbed Into The Fused Encoder: Long-Context Fused Forward -32 ms

- ctx_len threaded through muse_step_run/_aux (host-known from the fused
  gate); global layers at ctx > 1024 (and <= 32k partials cap) launch
  paged_attention_verify + reduce inside the same command buffer, using
  preallocated MQ partials scratch in the fused State (~18 MB).
- Validation at 9.9k ctx (in-process): fused target fwd **207.8 -> 176.0
  ms**; tok/step 2.78 (identical to the corrected eager -- acceptance
  parity confirms the fused MQ math end to end); in-process 12.47 tok/s
  vs 8.03 at the same point this morning (+55% net of the day's
  long-context work: window-fix quality + MQ kernel). E2E under a
  heat-soaked chassis: 10k cached 9.42/9.22 vs 9.08/8.88 (+4%; absolutes
  compressed by throttle -- short-ctx reads 16 vs its 22 best in the same
  state). Rested capstone (short triplet + 10k cached arm) launched in
  background.
- The long-context serving stack is now: correct global-layer attention
  everywhere, MQ-shared KV reads in both eager and fused paths, and the
  fused encoder default at every context. Remaining long-ctx headroom:
  the M=1 plain-decode scan at length (partition/reduce kernels exist --
  same plumbing pattern), and SWA-layer staging (window-capped, smaller).

## 2026-08-15 (33) - Decode/Verify Distinction In The Fused Attention Site + Partitioned Decode At Length

- Caught a latent bug in the just-landed MQ plumbing before it could
  bite: the fused attention site could not distinguish a VERIFY block
  (m rows = one request's queries -- MQ kernel valid) from a DECODE batch
  (m rows = independent requests -- the MQ kernel would read only block
  table row 0: WRONG for concurrent requests at long context). run_impl
  now takes rows_are_one_request (run_aux -> true, run -> false) and the
  MQ kernel is gated to verify blocks only.
- Bonus from the same distinction: decode batches (including m=1 plain
  decode) at ctx > 2048 now use the EXISTING partition/reduce kernels on
  global layers (chunk-parallel scan instead of one serial walk per
  (head,row)) -- the flash-decoding path the fused encoder never had.
  Single-request bench semantics unchanged (verify blocks dominate);
  concurrent long-context serving gets correct + faster attention.
- Validation deferred until the rested capstone releases the machine
  (its rest phase must not be contaminated); then: short smoke +
  acceptance + a 10k cached check.

## 2026-08-15 (34) - RESTED CAPSTONE: Final Configuration Validated On Both Arms

- Rested protocol, final build (fused verify + MQ global-layer attention +
  window-clamp fix + decode/verify distinction + partitioned decode):
  short **19.97 / 19.97 / 19.86** (== canonical 19.94/20.14/19.74 -- no
  regression from the entire long-context arc); 10k cached decode
  **10.97 / 11.00** vs 9.08/8.88 pre-fix (**+21%** at production
  context, AND the outputs are now correct -- the pre-fix eager numbers
  carried clamped global attention). Acceptance 1.66/draft mixed ✓.
- Day's final tally (16 hours of continuous campaign): short-ctx
  16.26 -> 20.14 canonical (+24%); 10k-ctx 8.9 -> 11.0 (+21%) with a
  silent quality bug fixed; six kernel generations; five correctness
  bugs found and fixed; 34 notebook entries; three methodology rules;
  two standing decision documents (drafter_requirements.md gating the
  100 tok/s target; the serving roadmap items with evidence).

## 2026-08-15 (35) - Drafter Training Feasibility Spike: Pipeline Proven, Timeline Measured

- Built and measured the two halves of the retraining pipeline (the gate
  to 100 tok/s), without starting any training run:
  capture 232 tok/s (aux payload 66.6 KB/token -> streaming pipeline
  mandatory, no disk shards); trainable reference module loads all 58
  published bf16 tensors cleanly and trains at 545 tok/s (5L) / 468
  tok/s (9L target depth) on MPS with loss decreasing.
- Combined local pipeline ~155 tok/s -> 100M tokens ~= 7.5 days
  machine-dedicated; 300M ~= 22 (optimizable ~2/3). Rented node remains
  the fast lane. Plan updated with measured numbers
  (perf/drafter_training_plan.md addendum); tooling in
  perf/drafter_training/.
- The go/no-go now has: measured requirement (E >= 4.0 on the harness),
  measured diagnosis (depth-wise conditioning, precision ruled out),
  measured pipeline rates, working tooling, and a one-command start.
  Decision remains with the user.

## 2026-08-15 (36) - Streaming Trainer Complete: The 100 tok/s Path Is One Approved Command Away

- stream_train.py: the full teacher->student streaming pipeline with the
  REAL DFlash objective (teacher top-K CE, per-position weights on the
  measured collapse zone), smoke-validated: 40 steps, loss 11.05 -> 9.70
  monotone; combined pipeline **124 tok/s measured** (100M tokens ~= 9.3
  local days). No training run started -- the smoke is minutes and
  validates only that the objective optimizes.
- Session-final state of the 100 tok/s question: requirement measured,
  cause diagnosed, plan costed, pipeline BUILT and smoke-validated,
  checkpoint gate ready. The remaining step is a human decision with a
  measured price tag: ~9-28 local machine-days or ~1 rented-node week.
  Serving stack: 19.97/19.97/19.86 short / 10.97-11.00 @10k, capstone-
  validated, correct at all context lengths.

## 2026-08-15 (37) - TIME-BOXED TRAINING PILOT RUNNING (3h wall limit; NOT the full run)

- Completed the last run-glue: streaming FineWeb-Edu corpus loader (2.15
  GB shard downloaded, 726k docs ~= 1B tokens), checkpoint/resume every
  100 steps, wall-clock limit, and export_checkpoint.py (trainer state ->
  servable safetensors dir with published naming + patched config,
  directly scoreable by the E-gate harness via the fork's safetensors
  drafter loading).
- PILOT LAUNCHED: 3-hour wall limit, 9-layer student, real corpus.
  Early health: loss descending on real data (8.80 -> 8.14 by step 80,
  continuing the smoke curve), ~146 tok/s pipeline, checkpoints landing.
  WHAT IT SHOWS: the full loop (train -> checkpoint -> export -> E-gate)
  works end to end, plus a real loss-slope data point. WHAT IT CANNOT
  SHOW: acceptance movement -- ~1.6M tokens is ~1.6% of the smallest
  planned run; E-gate on the pilot checkpoint is expected ~= baseline
  and validates PLUMBING only.
- The FULL run (9-28 local days or a rented week) remains gated on the
  user's go/no-go; the pilot is inside the session's normal multi-hour
  experiment envelope and is checkpoint-resumable into a full run if
  approved -- zero work wasted either way.

## 2026-08-15 (38) - Pilot Closed Out: Full Pipeline EXECUTED End To End; E-Gate At Baseline As Predicted

- The 3h pilot died with a session restart at ~step 730 but checkpointed
  at step 700 (~540k tokens). Loss on real FineWeb: 11.05 -> ~6.2, still
  descending. SUSTAINED pipeline rate ~47-58 tok/s (vs 124-146 early:
  thermal decay affects training too) -- local full-run timelines revise
  to ~2-4 machine-weeks for 100-300M tokens; the rented-node path is
  unchanged (~1 week).
- Checkpoint exported (102 tensors, 9L, 8.8 GB safetensors + patched
  config incl. layer_types x9) and E-GATED through the real serving
  stack: **linear E = 2.40 vs published baseline 2.46/2.52 -- at
  baseline, exactly as predicted for 0.5% of a real run's tokens.** The
  important validation: depth extension + 3h of training did NOT damage
  the published behavior; the fine-tune path is stable. Every step of
  the 100 tok/s pipeline has now been EXECUTED at small scale.
- Config recipe for candidates gains one item: layer_types must match
  the extended depth (harness recipe updated in practice; exporter
  should write it -- noted for the run glue).
- THE REMAINING STEP toward 100 tok/s is only scale: resume from
  perf/drafter_training/ckpt/latest.pt with the same command and
  ~2-4 local machine-weeks (or ~1 rented week), gating checkpoints on
  E >= 4.0. Go/no-go remains with the user.

## 2026-08-15 (39) - SCOPE RESET (user): Inference Engine And Kernels Only; Beat llama.cpp

- User direction, verbatim intent: we are writing an inference engine and
  custom kernels to serve Meta's published Muse-Glimmer + DFlash drafter
  AS-IS in SlimServe; the goal is to EXCEED llama.cpp's performance. No
  drafter training -- the vendor's own numbers (M5 Max: 26.6 plain /
  50.2 spec via ExecuTorch, SAME 5-layer drafter) prove ~50 tok/s is
  reachable with this artifact; our measured ~2.7 tokens/step matches
  the ecosystem, so the entire gap is STEP TIME (kernels + engine).
  The drafter-training workstream is closed; its 35 GB of artifacts
  removed (scripts + notebook history retained as record).
- Reference bar: llama.cpp on this box 26.75 plain / 30.8 spec (our
  measurement; predates llama.cpp PR 26842); ExecuTorch 26.6 / 50.2
  (vendor). Us: 20.14 spec / ~16 plain fused. Beat 30.8 first, then
  chase 50.2. Step math: 30.8 needs ~88 ms/step (we are at ~127);
  50.2 needs ~53 ms.
- Lessons pulled from llama.cpp PRs (user-provided): PR 26842 ships
  GPU argmax draft sampling for the greedy path (in-graph ggml_argmax,
  token-ids-only readback) -- the same design as our
  dflash_sample_greedy/get_top_tokens; now ENABLED in the muse profile
  (use_local_argmax_reduction: true). PR 26841 is the model-support PR.
- VENDOR BENCH CONFIG: the published numbers use the K-Quant-17GB
  UNIFORM Q4_K_M artifact, not the 19.7 GB dynamic quant our profile
  defaults to. Uniform Q4_K means the uint4-native tensor kernel covers
  most of the model (on dynamic, gate/up are Q5_K with no native
  operand) plus ~15% fewer bytes/step. Download + bench in progress
  (--quant kquant-17gb).

## 2026-08-19 (1) - NEW CAMPAIGN KICKOFF: Qwen3.8-27B + DFlash 2 on Metal

- User mandate (2026-08-19): support a new SlimServe profile on Metal --
  Qwen3.8-27B (unsloth UD-Q2_K_XL GGUF + mmproj-F16 vision tower)
  speculated by the DFlash 2 drafter released 2026-08-18 by Inco AI
  (z-lab/Qwen3.8-27B-DFlash2-GGUF, Q4_K_M). Prior Muse campaign state
  is committed and pushed at `ad8e8e937`.
- Artifacts (sizes from the HF tree API, hashes from the file pages):
  - unsloth/Qwen3.8-27B-GGUF `Qwen3.8-27B-UD-Q2_K_XL.gguf`
    10,676,423,744 B, sha256
    46151b52a5cad673d90a00222103254864326c251130b8fc4381d6f34386b3c8
  - unsloth/Qwen3.8-27B-GGUF `mmproj-F16.gguf` 927,607,488 B
  - z-lab/Qwen3.8-27B-DFlash2-GGUF `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
    1,143,006,752 B, sha256
    18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594
- TARGET MODEL IS NOT MUSE-SHAPED: unsloth config.json says
  Qwen3_5ForConditionalGeneration (model_type qwen3_5) -- 64 text
  layers in a 3:1 hybrid (48 linear_attention gated-deltanet layers,
  conv kernel 4, 16 K-heads/48 V-heads at 128 dim; 16 full_attention
  layers, GQA 24/4, head_dim 256, partial rotary 0.25, MROPE
  interleaved [11,11,10]), hidden 5120, vocab 248320, 262k ctx, native
  MTP head (1 layer, unused by us), and a qwen3_5_vision tower (27
  blocks, hidden 1152, patch 16, merge 2x2, out 5120). The linear
  attention layers are the big Metal support question, not the drafter.
- Drafter (incoai/Qwen3.8-27B-DFlash2 config.json): DFlash2DraftModel,
  qwen3-classed, 5 layers, SWA 2048, heads 32/8 at 128 dim, hidden
  5120, reads target_layer_ids [5, 19, 33, 47, 61]; dflash_config:
  block_size 8, conv_kernel_size 2 (two-tap dynamic, group 16),
  selector_rank 256, selector_top_k 16, mask_token_id 248070.
- DFlash 2 = DFlash + two additions (blog, 2026-08-18): (a) a path
  selector -- keep top-16 candidates per position, score adjacent pairs
  S_t(a,b) = U_t(b) + <A(a) o H(h_t), B(b)> (low-rank bilinear, 256-d,
  ~2M params, +0.6% cycle), then one greedy/sampled walk from the last
  verified token; (b) two-tap dynamic depthwise convolutions before and
  after every attention/MLP sublayer (k_t0*x_t + k_t1*x_{t-1}, base
  kernel + per-16-channel content correction, first position taps the
  last verified token) to kill suffix decay (+3% params, +0.7% cycle).
  Rejection sampling is unchanged -- output provably identical.
- Vendor numbers for THIS pairing (model card): mean acceptance 4.80 at
  block 8 (GSM8K 5.46 / MATH-500 5.28 / HumanEval 4.39), 2.7-3.4x over
  autoregressive at batch 1 in SGLang. Upstream vLLM launches it as
  method "dflash" + num_speculative_tokens 7 -- DFlash2-ness comes from
  the drafter's architecture, not a new method string. llama.cpp
  support is an unmerged PR (master has DFlash 1 only, incl. spec-type
  auto-detect #26814 and backend sampling #26958); oMLX also runs it
  (blog demo on an M5 Max).
- Next: map fork support surface (qwen3_5 target incl. gated-deltanet
  linear attention on Metal, vision tower loading, dflash worker
  changes for selector+conv), register the profile + sources, start the
  ~12.7 GB download through SlimServe.

## 2026-08-19 (2) - Support-Surface Map + Profile Registered (gated); Downloads Running

- Fork scout (very thorough, full detail in HANDOFF.md): the slimserve
  launcher needs ZERO code changes -- profiles.json is the only input,
  the speculator engine block passes through verbatim to
  SpeculativeConfig, and nothing in slimserve/ validates the method
  string (vLLM does: pydantic Literal + extra="forbid" at
  vllm/config/speculative.py:60-76, so keep method "dflash" like
  upstream does for DFlash 2). The DFlash runtime is reusable end to
  end: DFlashSpeculator's one-block layout (1 + k query rows),
  prepare_dflash_inputs (Triton + native Metal op), embed/lm_head
  sharing, rejection sampler. qwen3_dflash.py is the drafter base to
  subclass (Muse precedent: override only precompute_and_store_context_kv
  and get_top_tokens for Metal).
- NOT reusable / absent: the fork's models dir is trimmed to 25 files --
  no qwen3_5, no VL tower for it (registry entries for qwen3_vl etc. are
  dead names). qwen3.py::Qwen3ForCausalLM is dense text-only. The GGUF
  config parser and weight-adapter dispatch both hard-raise on unknown
  general.architecture strings; the drafter dispatch is a two-way probe
  on dflash.expert_count that needs a third branch for DFlash 2. The
  fused Metal dflash_sample_greedy is gated m in [9,17]; block 8 = 8
  rows falls outside it -- and DFlash 2 needs top-16 + path walk, not
  argmax, so draft sampling is new work regardless (eager first).
- THE BIG LIFT: 48 gated-deltanet linear-attention layers on Metal
  (recurrent fp32 state + short causal conv, conv kernel 4). No Metal
  kernel exists. References: upstream vLLM qwen3_next GDN Triton
  kernels, llama.cpp qwen3.5 (tensor naming + CPU/Metal reference), our
  hybrid KV manager from the Muse SWA/global split. Full-attention
  layers (16) reuse the Metal GQA path but need head_dim 256 + partial
  RoPE 0.25 + interleaved MRoPE checked against kernel support.
- Registered (uncommitted): sources["qwen38-27b"] + gated
  profiles["qwen38-q2kxl-1"] ("in-progress" -- the CLI refuses to serve
  until the path is real; live validation therefore skips it). Engine:
  reasoning_parser qwen3, tool_call_parser qwen3_xml (NEEDS LIVE
  VALIDATION -- picked as the Qwen3.5-era XML format parser),
  enable_prefix_caching false (recurrent state), num_speculative_tokens
  7, kv_cache_dtype auto. tests/slimserve/test_profiles.py updated: 48
  pass (source-set pin + modalities + the thinking/tool-calling default
  gate, which is what forced the tool_call_parser choice).
- Downloads running in the background via
  slimserve.fetch.ensure(plan, assume_yes=True) called from Python (the
  status gate blocks even --dry-run for in-progress profiles; using the
  fetch module keeps SlimServe the download owner). 12.7 GB total to
  ~/models/Qwen3.8-27B-GGUF and ~/models/Qwen3.8-27B-DFlash2-GGUF.
- Next: GGUF ground truth (dump general.architecture + full metadata of
  both files; the config-parser and adapter branches key off it), then
  target text model (gated deltanet on Metal is the critical path),
  vision tower, drafter class with convs+selector, speculator top-16
  path-walk extension. Staged plan with file:line anchors in HANDOFF.md.

## 2026-08-19 (3) - GGUF Ground Truth + DFlash 2 Reference Implementation Extracted

- GGUF arch strings confirmed WITHOUT waiting for downloads (HF API
  ?expand[]=gguf): target = "qwen35" (llama.cpp's exact arch name),
  drafter = "dflash" -- the SAME string as the Muse and DSV4 drafters.
  The fork's two-way drafter probe (dflash.expert_count) becomes
  three-way; the DFlash 2 discriminator is `dflash.selector_rank` (or
  the selector_hidden.weight tensor -- llama.cpp gates on the tensor).
- llama.cpp DFlash 2 reference FETCHED: PR #27342 ("spec : add DFlash2
  support (local convolution + candidate selector)"), single commit
  5ecbe1ac1 on recent master, now at branch `dflash2-pr` in ~/llama.cpp.
  Master also has arch LLM_ARCH_QWEN35 -- a working CPU/Metal reference
  for the hybrid target (recurrent-state model list, rs_rollback).
- Extracted drafter spec (llama-arch + src/models/dflash.cpp):
  - Metadata keys: dflash.block_size / conv_kernel_size / conv_group_size
    / selector_rank / selector_top_k.
  - Tensors: per layer `blk.N.attn_conv_base` {n_embd, kernel=2, 2},
    `blk.N.attn_conv_proj` {n_embd, 2*kernel*n_groups}, same pair for
    ffn_conv; top-level `selector_predecessor` {rank, n_vocab} (A, a
    token-embedding table), `selector_successor` {rank, n_vocab} (B),
    `selector_hidden` {n_embd, rank} (H). n_groups = n_embd /
    conv_group_size = 5120/16 = 320.
  - Conv placement per layer: dynamic = conv_proj @ normed_hidden,
    computed ONCE per sublayer pair; side 0 applied to the sublayer
    INPUT (pre-attn on noise_norm, pre-ffn on cur), side 1 to the
    sublayer OUTPUT, same dynamic. Conv math: for tap t in {0,1}:
    out += (base[:,t,side] + repeat_per_group(coeff[group,t,side])) *
    shift_block_local(x, t) -- shift zero-pads at block row 0, which IS
    the anchor/verified token in the 1+N layout, so position 1's tap-1
    naturally reads the last verified token (blog fig. 4).
  - Selector graph (post-sampling, per block): candidates =
    top_k(logits, 16); unary U = logits gathered at candidates; gate =
    selector_hidden @ hidden per position; for pos >= 1: A =
    get_rows(selector_predecessor, pos==1 ? anchor_id :
    prev candidates), scores[a,b] = <A(a) * gate_pos, B(b)> + U_pos[b]
    -> 16x16 matrix; graph packs [candidate_ids(f32), scores(256)] per
    position into a lattice row (padded to n_embd) returned instead of
    logits ("DFlash2 never consumes raw logits").
  - Host walk: predecessor := 0 (pos-1 matrix rows are identical --
    repeated across the anchor); per pos: greedy argmax over
    scores[predecessor, :] at T=0, else softmax((s - max)/T) sample,
    KEEPING the 16-way distribution per position for lossless rejection
    sampling downstream.
- vLLM mapping decided: keep method "dflash" (upstream does); the
  selector walk replaces get_top_tokens/argmax in DFlashSpeculator when
  the drafter config carries selector_rank; convs live inside the new
  drafter model class (DFlashQwen3 base + 2 conv-proj GEMMs and 4
  two-tap conv applications per layer, all block-local elementwise --
  Metal-friendly, no new kernel needed for correctness, fusion later).
- Also confirmed: the fork ALREADY vendors the full upstream GDN stack
  (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py --
  "Inference-only Qwen3-Next/Qwen3.5", plus fla ops under
  vllm/third_party/flash_linear_attention) but it is Triton/CUDA-bound;
  Metal needs torch-native MPS or Metal-kernel equivalents of
  chunk_gated_delta_rule / causal_conv1d / fused recurrent decode. The
  layer/topology code is reusable as the algorithm reference in-tree.

## 2026-08-19 (4) - Target Text-Model Chain Vendored + Fetch Resume Bug Fixed

- Vendored (dense-trimmed, agent-executed, verified): models/qwen3_next.py
  (837 -> 610 lines, MoE/EPLB/sequence-parallel dropped, MLP inlined,
  CUDA-only fused_qk_rmsnorm_rope_gate dropped -- eager split+QK-norm+RoPE
  kept) and models/qwen3_5.py (742 -> 400, dense text-only;
  ForConditionalGeneration deferred to the Muse vision pattern), plus
  configs/qwen3_5.py (upstream, MoE-free) and a thin configs/qwen3_next.py
  re-export (transformers 5.14.1 ships Qwen3NextConfig) -- the fork's
  configs/__init__.py had DANGLING registrations for all of these, now
  healed. Registry: +Qwen3_5ForCausalLM, dead Moe multimodal entry
  removed. Import + registry resolution verified on this machine (hybrid,
  inner-state); ruff clean.
- mamba_mixer2.py did NOT exist in the fork (GDN layer hard-imports
  mamba_v2_sharded_weight_loader from it). Vendored a minimal module with
  just that function; the FULL GDN module chain now imports on Metal
  (Triton placeholder engages at import; runtime kernels remain the
  campaign's critical path).
- Runtime landmines logged for the GDN stage: QwenGatedDeltaNetAttention
  picks forward_cuda on Metal (only xpu/cpu/rocm special-cased);
  torch.ops.vllm.qwen_gdn_attention_core + RMSNormGated device pick will
  need a Metal dispatch; v1/attention/backends/gdn_attn.py imports but
  its kernels are CUDA.
- FETCH RESUME BUG (slimserve/fetch.py): a short read (server closed the
  stream at 9.83 of 10.7 GB) hit the `done != bytes` branch which
  UNLINKED the .part before raising -- 9.8 GB discarded, restart from
  zero. Fixed: keep the .part on short read (transport failure, not
  corruption; the next attempt resumes via Range); the sha256-mismatch
  branch still deletes. Download relaunched with a 20-retry wrapper
  (scratchpad/fetch_qwen38.py).

## 2026-08-19 (5) - GGUF Parsers Landed: Drafter Config + Tokenizer Verified Against The File

- Drafter + mmproj side-fetched ahead of the 10.7 GB model (both dumped;
  full ground truth in perf/qwen38_metal_design.md). Deltas vs the HF
  configs worth remembering: mmproj image_size is 768 (HF vision config
  implied bigger), projector `qwen3vl_merger`; drafter target_layers are
  1-BASED [6,20,34,48,62] in GGUF, `dflash.attention.causal = False`
  EXPLICIT (all 5 layers non-causal despite an all-True SWA pattern +
  window 2048 -- the fork's per-layer causal table must obey the
  override), block_size 8 COUNTS the anchor row (7 drafted), selector
  A/B tables are Q4_K-quantized {256, 248320}, conv bases F32.
- Landed: vllm/transformers_utils/gguf_qwen35.py
  (build_qwen35_config_from_gguf for the target -- UNVERIFIED until the
  model file completes; build_qwen38_dflash2_config_from_gguf +
  build_qwen35_tokenizer_from_gguf + is_dflash2_gguf -- VERIFIED against
  the drafter file: config fields all correct, tokenizer roundtrip OK
  with the qwen35 pre split vendored from llama-vocab.cpp into
  gguf_native.QWEN35_PRETOKENIZER_REGEX). gguf_config_parser.py: qwen35
  branch + three-way dflash probe (expert_count -> DSV4, selector_rank
  -> DFlash 2, else Muse). tokenizers/registry.py: qwen35 branch + the
  dflash branch now discriminates DFlash 2 (which carries a full qwen35
  tokenizer) from Muse instead of hard-assuming Muse.
- Bug found while testing: transformers Qwen3Config NULLS sliding_window
  unless use_sliding_window=True -- pinned after construction. (Muse
  avoided this by using its own config class.)
- Next: drafter model class (DFlash2QwenDraftModel = DFlashQwen3 base +
  4 conv applications/2 dynamic projections per layer + selector head),
  drafter GGUF weight adapter, speculator selector walk; then the target
  side (weight adapter + GDN Metal runtime).

## 2026-08-19 (6) - DFlash 2 Drafter Model Class + GGUF Adapter Landed (unit-verified)

- vllm/model_executor/models/qwen3_dflash2.py: DFlash2QwenDecoderLayer
  (two-tap dynamic conv around every attn/MLP sublayer; one coefficient
  projection per sublayer pair reused for both sides; block-local shift
  zero-padded at the anchor row; non-divisible token counts treated as
  one block for profiling runs), DFlash2QwenModel (layers swapped via a
  new decoder_layer_cls hook on DFlashQwen3Model -- 2-line base change;
  Metal-native context-KV precompute BORROWED from the Muse drafter --
  the method is model-agnostic), DFlash2QwenDraftModel (shared
  embed/lm_head contract like Muse; selector A/B embeddings + H gate;
  select_draft_path implements the greedy walk WITHOUT materializing the
  16x16 lattice -- only the realized predecessor's score row is computed,
  7 batched steps of (n_blocks, 16) work). Registered as
  DFlash2QwenDraftModel -> qwen3_dflash2.
- UNIT-VERIFIED: conv math exact (0.0 max err) against a naive per-token
  per-tap reference on both sides; selector walk token-identical to a
  manual step-by-step reference; import + registry resolution clean.
- gguf_adapters/qwen3_dflash2.py: extends the Muse drafter adapter
  (backbone schema is identical); adds conv + selector name mappings and
  dequantizes the Q4_K conv projections and selector tables at load
  (plain fp16 modules; ~270 MB total). Loader dispatch extended with the
  selector_rank probe. NAME-MAP COMPLETENESS VERIFIED against the real
  file: 81/81 tensors mapped, zero unmapped, zero spurious entries.
- Temperature>0 selector sampling (keeping per-position 16-way
  distributions for lossless rejection) is deliberately deferred; the
  serving bench path is greedy.
- REMAINING drafter-side: speculator wiring (call select_draft_path
  instead of top-1 argmax when the drafter has a selector; plumb anchor
  token ids; respect causal=False non-causal attention on Metal), then
  an end-to-end drafter load test -- blocked on the target model
  existing, since the drafter shares its embed/lm_head.

## 2026-08-19 (7) - All Artifacts Local + Target Ground Truth; Registry Byte Bug; Quant-Matrix Finding

- CORRECTION to (1): the model file's true size is 9,828,981,664 B and
  its sha256 is fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0
  (LFS oid, confirmed via content-length AND the paths-info API). The
  10,676,423,744 B + 46151b52... recorded at kickoff came from
  small-model WebFetch summaries of the HF tree/blob pages and were
  WRONG; the drafter and mmproj rows were right. Lesson: registry
  bytes/sha come from `curl -I` content-length or the paths-info API
  verbatim, never from a summarized page. The wrong byte count caused
  every completed download to be rejected as a short read (the stream
  "ended early" at exactly the true size). profiles.json corrected
  (+ sha256 now recorded for the model file); fetch._download grew an
  already-complete-.part fast path (validate + rename, no network), and
  the finalize tail is factored into _finalize.
- FETCH COMPLETE, all three artifacts sha/size-validated in ~/models.
  Target config builder + qwen35 tokenizer VERIFIED against the model
  file: 64 layers (48 linear, 16 full at idxs 3,7,...,63), heads 24/4 at
  256, MRoPE sections [11,11,10] interleaved partial 0.25, GDN dims
  16K/48V at 128 conv 4, vocab 248320, MTP layer (block_count 65)
  excluded. Tokenizer roundtrip OK.
- Target tensor ground truth (866 tensors; design doc updated): full
  layers ship SPLIT q/k/v with the output gate FUSED INSIDE attn_q
  (width 12288 = 24*256*2); linear layers ship FUSED attn_qkv (10240)
  + attn_gate; pre-FFN norm is `post_attention_norm`; blk.64 = MTP,
  skip.
- THE QUANT-MATRIX FINDING: UD-Q2_K_XL spreads 15 tensor types incl.
  NINE i-quant formats (IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/
  IQ4_XS + Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K) across attn/ffn/ssm
  projections. Muse Metal kernels cover K-quants only; DSV4 Metal adds
  IQ2_XXS/Q2_K. Bring-up: load-time dequant for unsupported formats
  (correctness first), then native Metal decode per format ranked by
  bytes/step; llama.cpp ggml-metal (serves this exact artifact) is the
  porting reference. This materially widens the kernel workstream vs
  Muse.

## 2026-08-20 (8) - Target GGUF Adapter Landed; Metal Kernel Audit; A_log Trap Caught

- Metal quant-kernel audit (design doc updated): qgemv covers q8_0, q2_K
  q3_K, q4_K, q5_K, q6_K, iq1_s, iq2_xxs, iq2_xs, iq3_xxs, iq4_xs,
  iq4_nl; qgemm (M>1) covers the same MINUS iq2_xs and iq1_s. The
  artifact's formats missing from EITHER path: IQ1_S, IQ1_M, IQ2_XS,
  IQ2_S, IQ3_S = 179 tensors, 3.6 GiB quantized -> 22.3 GiB fp16 at
  load-time dequant (bring-up plan; model resident ~28 GiB vs 9.2).
  Native decode for those five formats is REQUIRED for the perf target
  (fp16 layers triple bytes/step); llama.cpp ggml-metal is the port
  reference.
- gguf_adapters/qwen35.py landed + loader dispatch: layer-type-aware
  name map (split q/k/v full layers with the gate fused inside attn_q --
  passes through, the vendored attention declares the doubled q shard;
  fused attn_qkv -> in_proj_qkv / attn_gate -> in_proj_z on GDN layers,
  a 1:1 match with the vendored stacking mapper), MTP block unmapped
  (iterator skips), conv1d reshaped (dim, kernel) -> (dim, 1, kernel),
  embed table dequantized (Q2_K). VERIFIED against the file: 851/851
  non-MTP tensors mapped bidirectionally, 15 MTP skipped.
- A_LOG TRAP CAUGHT BEFORE IT SHIPPED: GGUF `ssm_a` stores the no-scan
  form -exp(A_log) (llama.cpp multiplies it in directly:
  "gate = alpha_softplus * ssm_a"); the vLLM fla kernels compute
  -exp(A_log) themselves (fused_gdn_prefill_post_conv.py:136). The
  adapter converts A_log = log(-ssm_a). Silent-garbage class of bug;
  ground-truth discipline paid for itself.
- First engine load test RUNNING (no-spec, max_model_len 8192): expect
  weight loading to complete and the profiling forward to crash inside
  the GDN Triton ops -- the crash site defines the next work item (the
  torch-native MPS GDN runtime).

## 2026-08-20 (9) - First Engine Bring-Up Arc: Four Blockers Fixed In Sequence

Load test (no-spec, in-process LLM, max_model_len 8192) iterated through:

1. slimserve engine_kwargs leaked `tool_call_parser` (an api_server
   flag) into EngineArgs -- added to _SERVE_ONLY (serve_argv still
   passes it to the server).
2. Spawned EngineCore re-imports the launcher script: load scripts need
   a __main__ guard (scratchpad script fixed; not a repo bug).
3. MetalPlatform had no current_device() -- the GDN layer's RMSNormGated
   init calls current_platform.current_device() (only CUDA implemented
   it). Added classmethod returning "mps" to vllm/platforms/metal.py.
4. GGUF quant config demands all shards of a fused module agree on
   quantized-vs-skipped (is_layer_skipped_gguf): per-tensor dequant of
   attn_qkv (IQ2_S) left in_proj_z quantized inside in_proj_qkvz.
   Adapter now expands the dequant set to whole fused groups.
5. The vendored qwen3_5 stacked mapper sends in_proj_qkv in with a
   TUPLE shard id (0,1,2) -- the fused-module-from-checkpoint loader
   path cannot split quantized bytes (GGUFUninitializedWeightTypeParameter
   has no output_dim). Fix: GGUF quantizes each output row independently
   and the reader hands data as (rows, bytes_per_row), so the adapter
   row-slices attn_qkv into q/k/v (2048/2048/6144 rows) and emits
   per-shard names in_proj_{q,k,v}_shard mapped to scalar shard ids --
   suffix chosen to avoid the mapper's substring collisions
   (".in_proj_q" would corrupt ".in_proj_qkvz" names). Muse's mixed-type
   merged QKV (q4/q4/q6) is the precedent that scalar-shard GGUF merged
   loading works.

## 2026-08-20 (10) - WEIGHTS LOAD END TO END; GDN Torch-Native MPS Core In Flight

- After the (9) fixes the load test progressed through model
  construction AND full weight loading: 1250 tensors in ~70 s (row-split
  qkv shards + 179 dequantized i-quant tensors + embed table). The
  profiling forward then hit the predicted wall: GDN
  _warmup_prefill_kernels -> fused_post_conv_prep -> Triton placeholder
  (qwen_gdn_linear_attn.py:1033; CustomOp routes Metal to forward_mps
  which defaults to forward_native = the fla Triton path).
- Extracted the COMPLETE recurrence semantics from the in-tree Triton
  sources (fused_recurrent.py:122-175, fused_sigmoid_gating.py:125-135,
  call site :1610): per v-head fp32 state S[V=128,K=128], kv-head =
  hv // 3; per token: l2norm(q,k) (eps 1e-6 inside sqrt), q *=
  head_k_dim**-0.5, g = -exp(A_log)*softplus(a+dt_bias), S *= exp(g),
  delta = (v - S@k) * sigmoid(b), S += outer(delta, k), o = S@q. Conv:
  depthwise kernel-4 causal + SiLU over packed qkv (10240 dims), state
  (slots, dim, 3) via is_conv_state_dim_first.
- Agent task in flight: forward_mps + _forward_core_mps in
  qwen_gdn_linear_attn.py (torch-native, fp32 state, varlen prefill +
  decode, spec masks NotImplemented for now), verified against a
  line-by-line numpy transcription of the Triton kernel before touching
  the engine, then drive the load test to ENGINE UP or the next
  non-GDN blocker.

## 2026-08-20 (11) - llama.cpp Parity Oracle: Stale-Binary Detour, Clean Rebuild Running

- Started the llama.cpp reference lane (parity oracle + perf bar for
  this exact UD-Q2_K_XL artifact). First run failed: "missing tensor
  blk.64.ssm_conv1d.weight" -- the server treated the MTP block as a
  recurrent layer (printed n_layer 65, no nextn). Static analysis of
  master's qwen35.cpp showed correct logic (nextn read before the
  recurrent-layer fallback, guard i < n_layer()); the contradiction
  resolved as a STALE BINARY: build/bin/llama-server was from April
  (predating the July nextn/MTP handling; the arch itself is older, so
  it loaded far enough to fail late), and `cmake --build` in the stale
  dir exited 0 while rebuilding nothing. Clean out-of-source rebuild
  (build-qwen38, Metal on) running. Lesson: verify the binary carries a
  known-new string before trusting a reference measurement from a
  prebuilt tree (`strings | grep` the assert text).
- Also confirmed while scoping post-GDN risks: interleaved MRoPE is
  pure torch (mrope_interleaved.py uses forward_native -- MPS-safe),
  runner mrope plumbing exists in v1/worker/gpu (model_states/default,
  mm/rope), MambaSpec hybrid state allocation is handled in the
  worker's attn_utils, and metal_attn carries a per-metadata `causal`
  flag (Muse-era) so the DFlash 2 drafter's all-non-causal attention
  has a route.

## 2026-08-20 (12) - llama.cpp Reference Baseline CAPTURED (this artifact, this box)

- Clean build-qwen38 (master ece963f41, Metal; the env's custom-LLVM
  LDFLAGS/CPPFLAGS must be unset -- they poison the dylib link with
  missing C++ ABI symbols). Serves UD-Q2_K_XL correctly (MTP block
  handled; n_layer 64).
- PLAIN DECODE BAR: 35.67 / 35.37 / 33.55 tok/s (256-token greedy essay
  prompt, matched positions; the familiar thermal decline), code prompt
  34.35. Prefill 38.1 tok/s on the 11-token prompt (not meaningful).
- Greedy output IDENTICAL across 3 runs -- llama.cpp is a deterministic
  parity oracle for this artifact. The model thinks in <think> blocks on
  the code prompt (chat-template-free /completion).
- Raw JSON + server log: perf/results/2026-08-20/qwen38-llamacpp-ref/.
- Campaign math: DFlash 2 vendor multiplier 2.7-3.4x over plain =>
  llama.cpp-level step time + working DFlash 2 implies ~90-120 tok/s
  spec on this box. Our plain bar to beat first: 35.67.

## 2026-08-20 (13) - GDN MPS Core LANDED (verified 1.4e-6); IQ Tile Routing Gap Fixed

- Agent-implemented torch-native GDN path in qwen_gdn_linear_attn.py
  (+343 lines, no other files): Metal dispatch branch, forward_mps
  (both qkvz layouts), _forward_core_mps (batched decode across all
  sequences/heads incl. NULL-slot masking matching the Triton skip
  semantics; per-sequence varlen prefill scan with initial-state load +
  final-state scatter; spec masks NotImplementedError until the DFlash 2
  wiring; fp32 state), plus pure conv/scan helpers. VERIFIED against a
  line-by-line numpy port of the Triton kernels on CPU AND MPS: decode
  T=1, varlen [5,9,3] and T<width edge cases, with/without initial
  state, toy and real (16/48/128/128, dim-10240 conv) shapes -- max abs
  err 1.4e-6 (gate 1e-4), conv state writebacks bit-exact.
- Engine then advanced to the third GDN layer's out_proj and exposed a
  GENERAL GGUF-Metal routing gap: MMQ_QUANT_TYPES lists only IQ2_XXS
  (true for CUDA MMQ) so IQ3_XXS/IQ4_XS/IQ2_XS/IQ1_S at M > mmvq_safe
  fell through to ggml_dequantize = NotImplementedError on Metal --
  but qgemm.metal HAS tiles for all of them (iq1_s, iq2_xxs, iq2_xs,
  iq3_xxs, iq4_xs, iq4_nl). Fix: METAL_MMQ_QUANT_TYPES in gguf/utils.py
  + platform-aware pick in _fused_mul_mat_gguf. NUMERICALLY VERIFIED on
  MPS with real tensors from the artifact (M=64 vs gguf-py dequant
  ground truth): all five formats correct, max rel-to-mean err 0.3-1.5%
  = fp16 accumulation noise.
- Load-time dequant list SHRINKS to the formats with no Metal kernel at
  all: {IQ1_M, IQ2_S, IQ3_S} = 125 tensors, 2.65 -> 14.65 GiB fp16
  (+12 GiB; model resident ~21 GiB vs the previous plan's ~28).
- Load test rerunning.

## 2026-08-20 (14) - FIRST END-TO-END TOKENS (garbage); Bisect: Matmuls Exonerated

- ENGINE UP + GENERATED through the full SlimServe-owned path: load,
  profiling, KV/state allocation, prefill, 8 decode steps. Output is
  garbage ('g:...0...arks' + CJK/byte soup) -- a correctness hunt, with
  llama.cpp as the oracle (same artifact, deterministic).
- Facts so far: tokenizer EXONERATED (prompt ids byte-identical to
  llama.cpp /tokenize); step-0 top-5 is alien ('g','n','c','ch','3' vs
  oracle ':',' speed',' as'), so PREFILL compute is wrong, not decode.
  ssm_a EXONERATED (values all negative, magnitudes e^-5.5..e^-1 --
  confirmed no-scan form; A_log = log(-ssm_a) correct). Full-attn
  q|gate interleave MATCHES llama.cpp (per-head [q(256)|gate(256)],
  view offsets in qwen35.cpp:273-297 == vendored chunk(dim=-1)).
- Quantized matmuls EXONERATED at engine-realistic sizes: routed real
  artifact tensors through _fused_mul_mat_gguf (the real router incl.
  the muse sm-band 9..32) at M in {1,2,8,11,17}: ALL formats rel err
  0.2-1.3% (fp16 accumulation). Latent gap found: M>=33 with quantized
  N%32!=0 falls to the unimplemented Metal dequant (real modules there
  are fused past N%32 -- in_proj_ba is 96 rows -- so not the garbage
  source; logged as a routing TODO).
- Next discriminator running: 1-token prompt ("Paris") -- position 0
  makes RoPE identity and the GDN scan single-step, splitting
  weights/norm-layout bugs from position/mask/scan bugs.

## 2026-08-20 (15) - Discriminator: Pos-0 Plausible, Length Breaks It; Parity Hunt Launched

- 1-token prompt "Paris": our top-5 ['-', '-P', '', '-Pro', '－'] --
  PLAUSIBLE continuations (oracle: [',', ' ', ' (', ' -', "'t"]; note
  ' -' appears in the oracle's own top-5). 11-token prompt: letter soup
  (top-1 'g'). Decode from either degrades into byte soup within
  ~2 steps.
- Reading: weights, layouts, norms-as-weights, matmuls, tokenizer all
  approximately right (a hard layout bug would not produce
  "Paris-Provence"-style continuations). The failure grows with
  SEQUENCE structure: prefill scan/conv over length, state writeback at
  the prefill->decode boundary (ONETOK decode step 1 is already
  garbage), or pos>0 attention/rope on the 16 full layers.
- Checked: MRoPE with 1-D text positions bypasses the section logic
  entirely (mrope.py forward_native) = standard partial NEOX rope over
  64 dims, matching llama.cpp text semantics structurally.
- Agent launched: layer-by-layer parity vs llama.cpp eval-callback on
  the fixed 11-token prompt (env-gated per-layer dumps on our side;
  cosine-per-position; drill into the first diverging layer's stages).
  Top suspect flagged for it: the residual/norm WIRING order
  (post_attention_norm as sandwich vs pre-norm placement) -- exactly the
  class of bug that is invisible at pos-0-dominated single-token
  forwards and catastrophic with length.

## 2026-08-20 (16) - Housekeeping While Parity Hunt Runs: Regression Green, M>=33 Gap Closed

- Regression sweep over the shared-file edits (tokenizer registry
  reorder, dflash discriminators, METAL_MMQ routing, metal.py
  current_device, qwen3_dflash decoder_layer_cls hook): 48 profile
  tests pass; registry resolves all four model classes (Muse x2,
  DFlash2, Qwen3_5). Full Muse serving smoke deferred to the pre-commit
  gate (needs the GPU the parity hunt is using).
- The M>=33 latent gap from (14) is closed: on Metal, quantized shapes
  the tile cannot take (N % 32 != 0) past the vector-kernel batch limit
  now chunk rows through ggml_mul_mat_vec_a8 instead of falling into
  the unimplemented dequant. Verified on the raw 48-row Q8_0 tensor at
  M in {33, 64, 100}: rel err 0.16-0.23%.

## 2026-08-20 (17) - PARITY FOUND: Two Load-Convention Bugs; ESSAY Step-0 Now Matches Oracle Exactly

- Method: llama.cpp eval-callback (build-qwen38, -ngl 99) on the fixed
  11-token ESSAY prompt vs env-gated per-layer dumps from our engine
  (VLLM_QWEN38_DEBUG_DUMP hooks in vendored qwen3_5.py; scratchpad
  compare_forward.py samples llama.cpp's printed head/tail values +
  full-tensor sums per node).
- FIRST DIVERGENCE: layer 0 `attn_norm` (the very first RMSNorm).
  cos 0.9994 but magnitude ~1.95x. Cause: llama.cpp's converter folds
  +1 into every `*norm.weight` except `linear_attn.norm.weight`
  (~/llama.cpp/conversion/qwen.py:394), and our vendored model uses
  GemmaRMSNorm (x_hat * (1 + w)) -- we were computing x_hat * (w_hf + 2)
  on every input/post-attn/q/k/final norm. GGUF norm means measured
  ~0.97/1.23/1.94 (= HF + 1). FIX: `_degemma_gguf_norms` in
  vllm/model_executor/models/qwen3_5.py subtracts 1 at load when
  quantization == "gguf" (adapters untouched per constraints).
- SECOND BUG (code-proven, then confirmed by parity): the converter's
  `_LinearAttentionVReorderBase` stores every per-V-head GDN tensor
  (qkv V rows, z/b/a rows, A_log, dt_bias, conv1d V channels, out_proj
  COLUMNS) in ggml tiled-broadcast order -- q/k -> v pairing becomes
  i_k = i_hv % 16, not HF's i_hv // 3. Our MPS scan used
  repeat_interleave (HF grouped). Layout is self-consistent (out_proj
  cols are quantized Q3K/Q4K/IQ* so un-permuting at load is
  infeasible), so the FIX flips the expansion to tile semantics:
  cfg.gdn_tiled_v_head_layout in gguf_qwen35.py; tiled_gqa branch in
  _gdn_recurrent_scan_native (qwen_gdn_linear_attn.py), both MPS call
  sites (decode + prefill). Explains the pos-0-plausible /
  wrong-at-length signature: at t=0 the mispairing only swaps
  similar-magnitude q.k scalars; the recurrent state then compounds it.
- RESULT: layer-by-layer parity across ALL 64 layers, cos >= 0.9997 at
  every attn_in/attn_out/l_out reference point (was cos ~0.1-0.5 from
  layer 0). ESSAY step-0 top-5: [':' -1.04, ' speed' -2.04, ' as'
  -2.79, ' completeness' -3.11, ' (' -3.48] vs oracle [':' -1.01,
  ' speed' -2.07, ' as' -2.78, ' completeness' -3.14, ' (' -3.49].
  CUDA/ROCm FLA kernels still assume HF grouped order -- flagged in
  the module comment; must be wired before any non-Metal GGUF serving.
- Diagnostics to remove later: the _Qwen38DumpState block in
  qwen3_5.py (env-gated, zero-cost when VLLM_QWEN38_DEBUG_DUMP unset).
  Raw logs: scratchpad llamacpp_eval_11tok.log, dump/our_forward*.npz,
  probe_dump*.log.

## 2026-08-20 (18) - ROOT CAUSES FOUND AND FIXED: Converter Conventions, Not Our Math

- The parity hunt (llama.cpp eval-callback vs env-gated per-layer dumps,
  11-token prompt) put first divergence at LAYER 0's attn_norm: cos
  0.9994 but magnitude x1.95 -- and exposed TWO llama.cpp GGUF-CONVERTER
  conventions our path did not undo:
  1. NORM +1 FOLD (conversion/qwen.py:394): every *norm.weight except
     linear_attn.norm carries +1 so ggml's plain RMS_NORM*w reproduces
     HF's zero-centered Gemma-style norms. Our GemmaRMSNorm re-adds +1
     at runtime -> effectively x_hat*(w_hf+2) on all 64 layers. Fix:
     _degemma_gguf_norms subtracts 1.0 at load (gguf sources only).
  2. GDN TILED V-HEAD ORDER (conversion/qwen.py:446-609): when
     K-heads != V-heads the converter stores every per-V-head GDN tensor
     in tiled-broadcast order -- GGUF v-head j pairs with k-head j % 16,
     not HF's j // 3. Our scan used repeat_interleave (HF/Triton
     convention) -> every v-head read the WRONG q/k head. Self-consistent
     layout otherwise, so pos-0 stayed plausible while the recurrent
     state scrambled with length -- the exact discriminator signature.
     Fix: cfg.gdn_tiled_v_head_layout=True + tiled_gqa flag in the MPS
     scan (un-permuting at load is infeasible: out_proj columns are
     quantized). NOTE for any future CUDA run of this GGUF: the FLA
     Triton kernels still assume grouped order.
- VERIFIED: all 64 layers cos >= 0.9997 vs llama.cpp at every reference
  point (was 0.1-0.5 from layer 0); ESSAY step-0 top-5 matches the
  oracle within 0.04 nats across all five; greedy decode coherent and
  oracle-matching (":\n\n1.  **Completeness"). 48 profile tests pass.
- Diagnostics to remove before commit: _Qwen38DumpState in qwen3_5.py
  (env-gated, zero-cost unset). Raw artifacts: scratchpad
  llamacpp_eval_11tok.log, dump/our_forward*.npz, compare_forward.py.
- Parity + first-tok/s bench running (pre-staged script; 3 greedy
  256-token repeats + prefix-match vs the oracle).

## 2026-08-20 (19) - FIRST MEASURED BASELINE: 2.5 tok/s plain, 504-char Greedy Parity

- Greedy parity vs llama.cpp: the first 504 CHARACTERS (~120 tokens) of
  the 256-token essay continuation are byte-identical before the
  trajectories fork -- expected with 125 tensors dequantized to fp16 vs
  llama.cpp's native i-quant reads (Muse precedent: greedy forks from fp
  accumulation order alone). Output quality identical by inspection.
  Correctness milestone COMPLETE for plain text decode.
- FIRST TPS: 2.52 / 2.51 / 2.47 tok/s (3x greedy 256, in-process,
  max_model_len 8192). llama.cpp bar: 35.67. Gap analysis per the
  perf.md discipline: ~400 ms/step vs a ~46 ms bandwidth floor at the
  CURRENT bloated residency (21 GiB: quantized 5.6 + fp16 dequants) --
  i.e. ~8.7x dispatch/python-bound (the sequential GDN scan runs 48
  layers x per-token python steps), on top of a 2.3x bytes penalty from
  the IQ1_M/IQ2_S/IQ3_S fp16 dequants (native decode would put the
  floor near 9.2 GiB/step ~ 20 ms). Raw:
  perf/results/2026-08-20/qwen38-first-e2e/.
- Roadmap from here (order per the always-on-spec mandate): (1) GDN
  spec-state rollback in the MPS scan + drafter e2e + acceptance vs
  vendor 4.80; (2) GDN decode dispatch-cost attack (fold the per-layer
  step into fewer ops / a Metal kernel; muse_step-style fusion is the
  precedent); (3) native IQ1_M/IQ2_S/IQ3_S Metal decode (llama.cpp
  ggml-metal port) to reclaim the bytes; (4) head_dim-256 paged fast
  path. Spec at vendor acceptance 4.8 multiplies whatever step time we
  reach.

## 2026-08-20 (20) - POLICY: Greedy Is Banned Stack-Wide (user directive)

- User directive (emphatic): greedy/temperature-0 is not a supported
  configuration anywhere in this stack -- he had already removed the
  greedy flag deliberately, and validation/bench work MUST use the
  model's shipped sampling defaults (this GGUF: general.sampling temp
  1.0 / top_p 0.95 / top_k 20), seeded for repeatability. Saved to
  auto-memory (no-greedy-benchmarks); this repo's opinionated stances
  are the spec, the same way spec-always-on is.
- Purged the greedy defaults found in the stack: slimserve/chat.py
  Session.temperature 0.0 -> None (no override; /temp now rejects 0),
  smoke.py's temperature=0.0 pin -> seeded, no sampling override
  (stream.py chat_completion omits sampling params unless given). 57
  slimserve tests pass.
- Method corrections: layer-level parity (cosine on activations) needs
  no sampling and stays the correctness instrument. Cross-run and
  spec-vs-plain comparisons use SEEDED sampling -- this fork's rejection
  sampler keys gumbel by (seed, position) precisely so seeded spec ==
  seeded plain. The greedy-generated llama.cpp oracle files and the
  greedy 2.5 tok/s baseline in (19) are SUPERSEDED as protocol; the
  step-time attribution stands (sampling does not change bytes/step),
  and the re-baseline at shipped defaults folds into the spec agent's
  measurements.
- Spec agent redirected mid-flight: sampled selector walk (softmax over
  16 kept candidates per position, distributions retained for lossless
  rejection sampling, per llama.cpp PR #27342's host walk) is PROMOTED
  from deferred to the main deliverable, wired into the probabilistic
  draft_logits path; acceptance measured at shipped defaults (vendor
  4.80 was measured there).

## 2026-08-20 (21) - SPEC ACCEPTANCE ROOT CAUSE: Task-Domain Dependence; We BEAT The Reference

- Built the llama.cpp dflash2-pr oracle (worktree ~/llama.cpp-dflash2,
  clean-env build) and ran the SAME two GGUFs through their DFlash 2
  spec pipeline at OUR settings (temp 1.0 / top_p 0.95 / top_k 20,
  seed 42, --spec-draft-n-max 7):
  - Essay prompt (our bench prompt): draft acceptance 0.219 (119/544),
    mean 2.51 tokens/step -- WORSE than our 3.20. Deterministic across
    3 seeded runs. (Their tok/s not comparable: -v verbose logging.)
  - GSM8K-style prompt, same settings: acceptance 0.555, ~4.74
    tokens/step -- the vendor band (4.80 GSM8K, README 5.13-5.39
    GSM8K).
- VERDICT: the 3.20-vs-4.80 gap is dominated by TASK DOMAIN, not an
  implementation bug. Vendor/README acceptance numbers are GSM8K
  (low-entropy math); a temp-1.0 essay is a high-entropy task where the
  REFERENCE implementation itself gets 2.5 tokens/step. Our stack beats
  the reference on the same task (3.20 vs 2.51) -- consistent with our
  selector-assisted greedy walk vs llama.cpp's sampled walk, or
  numerics. Acceptance comparisons MUST be domain-matched from now on;
  the bench suite needs a GSM8K-style arm for acceptance and an essay
  arm for step-time.
- Raw: scratchpad llamacpp_spec_oracle.log + task outputs; oracle
  server on :8092 (left up for the agent's domain-matched run).
- Remaining spec work (unchanged): sampled selector walk with kept
  distributions (correct rejection-sampling math at T>0), recall@k
  replay as drafter-forward health check, our-engine GSM8K acceptance
  to confirm >= the oracle's 0.555/4.74.

## 2026-08-20 (22) - VISION LANDED: Tower Parity ~1e-3 vs llama.cpp, Real Image Through The Real Engine

- Agent-delivered vision side: qwen3_5_vision.py (742 lines: tower,
  processor stack, Qwen3_5ForConditionalGeneration with the mamba/hybrid
  interface trio + get_language_model for DFlash sharing), composite
  config flip in gguf_qwen35.py (text-only fallback intact), adapter
  _vision_weights (334/334 mmproj tensors, taps fused), registry entry.
- VERIFIED: tower output vs llama.cpp llama-mtmd-cli embeddings dump:
  token-0 values agree elementwise to ~1e-3, mean identical -- parity
  grade. Engine smoke with a real 768x512 image: shapes/colors question
  answered correctly ("red rectangle, yellow triangle, blue circle");
  text path intact on the multimodal engine; 48 profile tests pass.
- Notable implementation facts (full detail in the agent report):
  patch-embed = two GGUF conv taps stacked (llama.cpp conversion splits
  HF Conv3d), align-corners bilinear pos-embed interpolation in
  merge order, ggml VISION rope == HF rotate-half verified, v.post_ln
  is the MERGER norm, merger GELU exact-erf vs ggml tanh-approx
  (deliberate, within F16 noise), smart-resize bounds 4096..589824 px.
  Fork-specific processor fixes: _hf_processor_applies_updates=False +
  passthrough _call_hf_processor for the mm-only/profiling path.
- Open items from the smoke: a stray token ("Dinner") appears in BOTH
  text and vision answers at temp 1.0 -- text-side sampling/quant quirk,
  tracked separately; spec+image combined path unexercised.

## 2026-08-20 (23) - Muse "Regression" Root-Caused: My Smoke's Override, Not Any Commit

- The muse in-process failure (mm profiling: "0 prompt placeholders for
  1 image item") reproduced IDENTICALLY at the live tree, at the qwen38
  commit, AND at the fully-validated muse commit ad8e8e937 -- then
  VANISHED at ad8e8e937 with profile-exact engine kwargs. Culprit: my
  smoke script's max_model_len=8192 override; muse's profiling dummy
  builds an image prompt whose placeholder region does not survive the
  clamp. No commit broke anything. LESSON (gate method): regression
  smokes run with PROFILE-EXACT engine kwargs, no overrides.
- Also while bisecting: the stash's lone "autostash" entry dates to
  Aug 7 (parent fccfe105a, DSpark-TurboQuant era) -- a stale rebase
  leftover predating both campaigns, superseded by later commits. Left
  in place; safe to drop.
- Live-tree profile-exact muse smoke running to formally close the
  shared-file regression gate.

- GATE CLOSED: live-tree profile-exact muse smoke passes; seeded output
  byte-identical to the ad8e8e937 baseline run (same head text) -- the
  campaign's shared-file edits are numerically invisible to Muse.

## 2026-08-20 (24) - NATIVE IQ DECODE COMPLETE: All Three Formats, _DEQUANT_TYPES Empty

- Agent-delivered Metal decode for IQ2_S (82 B/block), IQ3_S (110 B),
  IQ1_M (56 B, fused fp16 scale in the scale-word top nibbles; reuses
  iq1s_grid with the -1+-delta form; IQ1M_DELTA=0.125 per current
  llama.cpp AND gguf-py -- not the historical 0.0625). Grids ported
  script-generated from ggml-common.h; math from ggml-quants.c
  dequantize rows, cross-checked vs gguf-py.
- VERIFIED on real artifact tensors through the real router at M in
  {1,2,8,11,17,33,64}: rel err 0.20-0.36% ALL cells (gate 2%); the
  10-format regression matrix unchanged (0.09-1.34%). Microbench
  (contended GPU, reported not tuned): GEMV M=1 at 182-290 GB/s --
  in-family with neighbors.
- _DEQUANT_TYPES IS NOW EMPTY: every quantized format on this artifact
  decodes natively on Metal (GEMV + tile GEMM). Only the Q2_K embed
  table still dequantizes (separate name-based clause; needs the
  dequant-gather kernel eventually). Model residency drops ~21 -> ~12
  GiB (9.2 quantized + fp16 embed); the plain-decode bandwidth term
  shrinks accordingly -- re-measure tok/s in the consolidation bench.
- GGML id gotcha recorded: IQ3_S=21 precedes IQ2_S=22. Operational
  gotcha: cp over a mapped .so inode -> macOS "Code Signature Invalid"
  SIGKILL on dlopen; use rm-then-cp + codesign -f -s -. (Applies to
  every metallib/.so refresh on this box.)

## 2026-08-20 (25) - SPEC E2E COMPLETE AND ORACLE-BEATING; Two Substrate Bugs Exposed

- GDN spec rollback DONE (verified 9.5e-7 vs numpy kernel ports incl.
  resume-after-acceptance for every A in {1..8}, NULL slots, both GQA
  layouts; conv slots bit-exact). Sampled selector walk DONE and ENGAGED
  (sparse 16-way distributions into draft_logits; unit err 5.1e-8).
  DFlash 2 fully up: drafter load, shared embed/lm_head, fc combine,
  non-causal metal attention (bool False confirmed at the metadata),
  native input prep, selector, rejection sampling, rollback per step.
- MEASURED (essay, seeded shipped defaults, OLD metallib, V2 runner):
  spec 4.02/4.12/3.62 tok/s, acceptance 2.71 mean / 0.244 draft rate
  (per-position 0.72/0.42/0.25/0.15/0.08/0.05/0.04) -- BEATS the
  llama.cpp oracle (2.51 / 0.219) on the same GGUFs+settings. Plain V2
  6.40/6.43 tok/s (V1 was 2.5 -- the V2 runner alone is 2.6x).
- SPEC < PLAIN on V2 (3.9 vs 6.4): a bug per spec-always-fastest;
  attributed to the 8-position python GDN scan per verify step + CPU
  rejection fallback -- the fusion workstream targets exactly this.
- SUBSTRATE BUG 1 -- V2-on-Metal decode intermittently corrupt
  (boot-dependent; prefill EXACT vs V1 at depth 0-32; some boots decode
  48/48 identical, others collapse to repetition by ~20; async/steady-
  meta/GDN-slots/block-tables/rope ruled out). Suspected MPS ordering
  race in V2 buffer machinery (V1's CpuGpuBuffer got metal_compat
  patches; V2's UVA pools assume CUDA depth-2 concurrency). NOTE: Muse
  serves on V2 clean -- the corruption is qwen38-specific (hybrid state
  machinery is the differential). Taints/depresses ALL V2 measurements
  incl. acceptance above.
- SUBSTRATE BUG 2 -- NEW-METALLIB REGRESSION: with the 08:43 metallib +
  empty _DEQUANT_TYPES, spec NaN-crashes (rejection multinomial, 3/3)
  and plain GSM8K floods "!"; plain essay TF clean. OLD build ran 256-tok
  spec fine. Suspect: the new IQ kernels' BF16 activation variants --
  the kernel verification ran fp16 x only. Testing standalone now.
- Seeded spec-vs-plain identity NOT testable on Metal yet: MPS
  gumbel/rejection fallbacks use unkeyed noise (native CUDA/HIP-only
  seed parity) -- porting seed-keyed sampling to Metal is a follow-up.
  Also: MPS rejection fallback lacks the Triton _sanitize_nan parity.
- V2 spec runs require VLLM_USE_V2_MODEL_RUNNER=1 (hybrid target
  defaults V1; V1 spec path crashes on torch.cuda.Stream).

## 2026-08-20 (26) - New Kernels Exonerated (bf16 too); Corruption Unified Under One Hunt

- Standalone router test of the three new IQ kernels PLUS IQ4_XS/Q4_K
  controls at BOTH fp16 and bf16 activations, M in {1,4,8,17,64}: zero
  NaN/inf anywhere; bf16 errors 1.5-5% across ALL formats including the
  old ones (bf16 precision band, not breakage). The new-metallib NaN
  crash is NOT the kernels' math.
- Unifying theory: substrate bugs 1 and 2 are ONE bug -- the V2-on-Metal
  boot-lottery decode corruption; the metallib swap merely shifts
  dispatch timing (more native kernels = different race window).
  Corrupt activations explain repetition collapse, "!" floods, AND NaN
  probabilities in the rejection multinomial.
- Sharpened hypothesis from the collapse signature: ~token 20 of decode
  after an 11-token prompt = the FIRST NEW KV BLOCK append (block 16).
  The V2 buffer layer already carries Metal-aware blocking in _h2d and
  UvaBuffer (Muse-era); the staged-write/block-append path and the UVA
  round-robin depth are the unpatched suspects. Agent launched with
  discriminating tests (collapse-point-vs-prompt-length tracking,
  max_concurrency=1, paranoid-sync bisect); fix bar: >= 6 consecutive
  clean boots + Muse-V2 smoke unregressed.

## 2026-08-20 (27) - Sleep Root Cause (ops) + DIVERGE@0 Evidence: Stale-GDN-State Hypothesis Promoted

- OPS ROOT CAUSE of today's agent stalls and stop-start verification:
  the MACHINE WAS SLEEPING (explicit in the last agent cut; the earlier
  "500" stalls match the pattern). Sleep now blocked machine-wide
  (adrafinil hold, lid-closed incl.) for long background runs -- add
  this to the campaign ops checklist alongside rm-then-cp+codesign.
- Verification speedups applied: gauntlet boots 3-in-parallel; probe
  boots folded into the sample; post-gauntlet validation moves to a
  persistent-server pattern (no more 2.5-min boot per measurement).
- CRITICAL EVIDENCE before my collapse probe was stopped: same-seed
  pair on the FIXED build, plen=2 -> DIVERGE@0 (second same-prompt
  request in one engine splits at the FIRST decode token). Muse is
  byte-identical seeded cross-boot on this machine => not generic MPS
  sampler nondeterminism; the differential is RECURRENT STATE SLOTS.
  Promoted hypothesis: stale GDN state -- torch.empty-allocated
  conv/ssm caches ("boot lottery" = zero pages by luck) and/or missing
  slot reinit on sequence start (has_initial_state=False must zero, not
  read). Handed to the hunt agent with a one-line discriminator
  (zero-fill at alloc + on sequence start) and instructions to
  re-evidence or revert its staged-write fix.

## 2026-08-20 (28) - ROOT CAUSE OF THE V2 CORRUPTION: Shared Block Pool vs K/V-First Cache Layout

- Machine slept ~10 h (09:40 -> 19:34; the 2 h hold expired). Session
  scratchpad AND agent transcripts were wiped, so the hunt agent's
  gauntlet evidence is lost and the agent cannot be resumed. Its FIX
  survived in the worktree (attn_utils.py). Sleep hold re-armed 4 h;
  all future run evidence goes under perf/results, not the scratchpad.
- THE BUG (attn_utils.py, agent-authored docstring is exact): hybrid
  models allocate attention blocks and mamba state pages from ONE block-id
  pool, so block i must map to the same bytes in every layer's view. The
  Metal attention backend's K/V-FIRST layout (2, num_blocks, ...) stores
  all K pages before all V pages => attention block i's K half sits at
  byte offset i*page/2 = INSIDE mamba page i//2. Every KV-cache write
  clobbers some GDN layer's conv/ssm state. Explains EVERYTHING: the
  boot lottery (which blocks the allocator hands out), collapse at the
  first new KV block (notebook (26)'s geometry), within-boot drift, the
  second-request DIVERGE@0 (clobbered slots, NOT a reinit bug), NaN in
  the rejection multinomial, and Muse's immunity (no mamba pages). The
  new metallib only shifted timing.
- THE FIX: _update_hybrid_attention_mamba_layout -- restride K/V-first
  attention views to blocks-first in place (as_strided_, no data move),
  mirroring upstream's V1-runner function that the V2 runner lacked.
  Verified by the agent in one instrumented boot; my correction in (27)
  (torch.empty / slot-reinit theory) is RETRACTED -- allocation is zeros
  and the "stale state" was the clobber.
- Gauntlet restarted by me: 3 parallel boots x 4 prompt lengths x
  same-seed pairs (catches first-gen collapse AND second-request
  contamination). Env-gated diagnostics left by the agent
  (QWEN38_STATE_PROBE in model_runner.py/metal_attn.py) to be removed
  before commit.

## 2026-08-20 (29) - Gauntlet: 3/3 Clean + Cross-Boot Identical; Muse Clean; Seed-Keyed MPS Sampling Confirmed

- Gauntlet boots 1-3 (fixed build, 3 parallel, 4 prompt lengths x
  same-seed pairs): ALL 12 pairs IDENTICAL, every BOOT_VERDICT CLEAN, and
  the outputs are identical ACROSS the three boots. Coherent on every
  arm (the GSM8K arm reasons in thinking style). With the hunt agent's
  instrumented boot: 4 clean. Boots 4-6 first attempt VOID (init OOM at
  util 0.2 alongside Muse -- not corruption); rerunning at 0.3.
- Muse profile-exact smoke on the final tree: ENGINE UP, coherent,
  exit 0. Its seed-42 text differs from this morning's because the MPS
  Gumbel path changed (below) -- expected, not a regression.
- gumbel.py (+72, agent-authored): stateless splitmix64 (seed, pos,
  column)-keyed Gumbel noise for MPS, replacing an unseeded global-RNG
  fallback that silently ignored per-request seeds. This retroactively
  explains the spec agent's "SEED_STABLE False plain-vs-plain" and
  makes seeded repeatability real on Metal (the gauntlet's IDENTICAL
  pairs are the proof). The spec-vs-plain seeded identity gate is now
  testable -- queued for the spec bench.
- Hunt diagnostics (QWEN38_STATE_PROBE hunks in model_runner.py and
  metal_attn.py) removed; imports verified.
- GAUNTLET COMPLETE: boots 4-6 CLEAN, 0 diverges => 7/7 clean boots
  (6 mine + the agent's instrumented one), 24/24 same-seed pairs
  identical, cross-boot identical, Muse unregressed. The shared-pool
  K/V-first layout fix is VERIFIED. Consolidated bench (plain then spec,
  sequential, essay + GSM8K arms, seeded shipped defaults) running;
  raw logs -> perf/results/2026-08-20/qwen38-consolidated/.

## 2026-08-20 (30) - Consolidated Bench Exposed A 3x Regression; Root-Caused To MPS Strided Gather

- Consolidated bench (new metallib + layout fix, seeded shipped defaults,
  in-process, util 0.45): PLAIN essay 2.57/2.20/2.08, gsm8k
  1.99/2.00/2.01 (seed-stable, coherent); SPEC essay 1.08/1.02/0.89,
  gsm8k 2.64/2.13/1.90 (coherent; NOT seed-stable -- the rejection
  path's draws are still unkeyed; get_metrics() returned no spec
  counters in-process, acceptance to be read from Prometheus in the
  server-based bench). Versus this morning's 6.4 plain / 4.0 spec: a
  ~3x REGRESSION. Treated as evidence per perf.md.
- Bisect: (1) metallib exonerated -- Q4_K GEMV identical on old vs new
  build (187.6 vs 182.6 us; both only ~95 GB/s on the 5120x6144
  ssm_out shape = a pre-existing 4.7x-off-floor kernel tuning item,
  logged); new build faster on IQ3_XXS (54 vs 82 us). (2) Thermal
  exonerated -- rested rerun after idle: 2.49/1.85/2.03. (3) ROOT CAUSE:
  the blocks-first restride makes kv_cache[0]/[1] non-contiguous
  interleaved views, and MPS index_select on a non-contiguous source
  takes a slow gather path: microbench 1.32-1.41 ms vs 0.07-0.08 ms per
  64-block gather (~20x), x16 layers x K,V per step, scaling with pool
  size. head_dim 256 keeps qwen38 on the SDPA route (paged fast path is
  64/128 only), so every decode step paid it. Writes unaffected
  (0.06 ms both).
- FIX (metal_attn.py SDPA read site): when the cache is blocks-first
  (stride(0) < stride(1)), gather whole (K,V) pages from the contiguous
  transpose(0,1) view and split -- 0.08 ms, bit-identical (verified
  torch.equal). Non-hybrid models (Muse) never restride and keep the
  original path. Re-bench running.

## 2026-08-20 (31) - PLAIN DECODE 15.0 tok/s (6x today's start); Gather Fix Verified

- First patch attempt referenced the parent kv_cache out of scope
  (NameError in EngineCore); fixed by reconstructing the blocks-first
  pages view from key_cache's own storage via as_strided (gate:
  key_cache.stride(0) == 2 * page_elems). Standalone equality vs the
  strided gathers: K and V bit-identical.
- PLAIN (seeded shipped defaults, in-process, util 0.45, same build):
  essay 15.26 / 14.98 / 14.82 tok/s, gsm8k 14.44 / 14.00 / 13.43.
  Seed-stable; text identical to the 2.2 tok/s run (correctness
  unchanged). That is 2.3x above this morning's 6.4 -- the native IQ
  kernels' byte savings (22 GiB of fp16 dequant traffic gone) were
  MASKED by the strided-gather penalty until now. Progression today:
  2.5 (V1) -> 6.4 (V2) -> 2.2 (layout fix, strided gather) -> 15.0.
  llama.cpp bar: 35.67. Raw: perf/results/2026-08-20/qwen38-consolidated/.
- Spec bench rerunning on the same build.
- SPEC (same build): essay 4.06 / 3.80 / 3.50 tok/s, gsm8k 9.45 / 7.97 /
  8.65. Coherent; not seed-stable (rejection path's draws unkeyed --
  open item). SPEC < PLAIN on both arms = OPEN BUG per spec-always-
  fastest (never a documented config). Attribution (unchanged, now the
  dominant item): the verify step pushes 8 positions through the
  per-position python GDN scan x 48 layers (~8x plain's scan cost) +
  python drafter forward + CPU-loop rejection fallback. The fused GDN
  step (design in perf/qwen38_metal_design.md) is the fix and starts
  next. Profile stays gated in-progress until spec is net-positive.
- Interval SpecDecoding metrics did not surface in this in-process run;
  acceptance for the record comes from the spec agent's Prometheus
  counters (essay 2.71/0.244, notebook (25)) and gets re-read in the
  server-based bench once fusion lands.

## 2026-08-22 - Spec Seed Determinism on Metal: All Draws Keyed

- Two days' gap (fused-GDN agent lost to a DNS outage before writing
  anything; resumed today with the machine held awake).
- Root cause of spec's seed instability: the MPS rejection fallback
  (rejection_sampler_utils.py) already built a seeded generator keyed by
  (seed, first verify position) but NEVER PASSED IT -- torch.rand(1) for
  the accept test and every torch.multinomial (target sample, residual
  recovery, bonus) drew from the global RNG. All five sites now use the
  generator. Also fixed a latent temperature bug there: draft_logits
  arrive temperature-applied (speculator output_processed_logits
  contract; the Triton path reads them as-is) and the fallback divided
  by temp again -- a no-op at the shipped temp 1.0, wrong elsewhere.
- The selector walk's categorical draw (qwen3_dflash2.py) used device
  exponential_ noise; it now takes (seeds, positions) from the
  speculator (self.seeds[idx_mapping], sample_pos view) and draws
  Gumbel noise from the stateless (seed, pos, column)-keyed uniforms,
  falling back to the device RNG only when unkeyed. Unit check: same
  seed => identical tokens AND identical sparse distributions;
  different seed => differs; 16 finite candidates per position.
- E2E seeded spec repeatability check deferred until the fused-GDN
  agent releases the GPU (its benches must not be contended).
- Durable tests added (the agents' scratchpad harnesses were lost in the
  Aug 20 wipe): tests/model_executor/test_qwen3_dflash2.py (conv vs
  naive reference both sides; greedy walk vs manual walk; sampled walk
  seed-keyed + sparse k-way contract) and test_qwen35_gguf.py
  (skip-guarded on the artifacts: target/drafter config fields, both
  adapters' bidirectional name-map completeness, qwen35 tokenizer pinned
  to the llama.cpp-verified prompt ids). 8 pass locally. The GDN
  scan-vs-Triton-port oracle gets re-derived into tests/ once the fused
  agent releases qwen_gdn_linear_attn.py.

## 2026-08-22 (32) - Fused GDN Decode/Verify and Gated Norm Verified

- BASELINE: fe960935f plain essay ~15 tok/s; spec essay 3.5-4.1 and
  GSM8K 8.0-9.5. The verify path ran the GDN recurrent scan as eight
  Python/MPS position steps in each of 48 target layers.
- HYPOTHESIS: fuse convolution-window update plus the gated delta-rule
  scan across all positions and heads, retaining fp32 recurrent state
  and the exact per-position store/resume rollback contract. Fuse the
  following RMSNormGated operation separately.
- IMPLEMENTED: `qwen_gdn_step` uses two Metal dispatches per layer
  (convolution then scan) for decode and verify; routing retains the
  torch-native oracle behind `VLLM_QWEN38_FUSED_GDN=0`.
  `qwen_gdn_gated_norm` removes the decomposed gated norm graph.
- CORRECTNESS: the exhaustive synthetic harness passed all 147 decode,
  verify, fp32/bf16/fp16, tiled/grouped, null-slot, and fp32-conv-state
  cases. Uniform, ragged, null-position, null-sequence, mixed, and
  mixed-ragged plan parity all passed. Gated norm passed all 12 dtype /
  shape cases. Durable real-geometry Metal tests now cover decode,
  speculative rollback, null slots, strided inputs, and gated norm.
- MEASURED (seeded shipped sampling; uncontended raw runs): fused plain
  essay 16.97/16.81/16.86 tok/s and GSM8K 16.57/16.58/16.54; with
  fused gated norm, essay 18.19/18.24/17.17 and GSM8K
  16.77/16.85/17.06. A later hot run was 16.15-16.40 essay, so retain
  the fusion but do not promote 18.24 as a stable baseline.
- DECISION: RETAIN. It closes the dominant recurrent-state correctness
  and launch-count problem, but speculation was still only ~7 tok/s
  essay / 13-15 GSM8K immediately after this change, so more of the
  M=8 verify path still needed work.
- RAW: `perf/results/2026-08-22/qwen38-fused-gdn/verify_gdn_step_fresh.log`,
  `verify_spec_plans_fresh.log`, `verify_gated_norm_fresh.log`,
  `bench_plain_fused.log`, `bench_plain_fused_norm.log`,
  `bench_plain_hot.log`, and `bench_spec_fused.log`.

## 2026-08-22 (33) - Quantized Verify-Band GEMM + Vectorized MPS Rejection

- BASELINE/HYPOTHESIS: after GDN fusion, the 8/17-row target verify band
  still dispatched repeated GEMVs for the target's IQ/Q2_K weights and
  rejection sampling still performed request/token CPU work. Enable the
  existing quantized MM band for every serving format whose new span
  decoder was verified, then keep rejection sampling batched on MPS.
- IMPLEMENTED: qgemv routing now admits q2_K, q3_K, IQ1_S/M, IQ2_XXS/XS/S,
  IQ3_XXS/S, and IQ4_XS to the M={2,4,8,16,17} MM kernels. The MPS
  sampler performs batched acceptance, prefix counting, residual/bonus
  emission, and deterministic (seed, position)-keyed draws without host
  round trips. Full-vocabulary Gumbel emission was replaced by one keyed
  uniform plus inverse-CDF sampling.
- CORRECTNESS: real GGUF tensor comparisons at M=8 and M=17 had maximum
  relative error 0.2674% across the sampled serving formats/shapes; the
  new eight-wide IQ decoder differs from the scalar decoder only by
  rounding (<0.1% relative). Rejection Monte Carlo passed its target
  marginal, theoretical acceptance-rate, deterministic batch-vs-single,
  and greedy-chain checks. A durable 8192-request test checks the emitted
  target marginal and `sum(min(p,q))` acceptance.
- MEASURED: MM routing moved spec to essay
  16.10/14.18/12.20 tok/s and GSM8K 30.40/31.19/33.91. Vectorized
  rejection plus keyed draws produced essay 15.60/15.14/15.03 and GSM8K
  36.66/36.49/36.06, with both arms seed-stable. The MM microbench shows
  2.4-5.0x lower M=8 time than eight GEMVs on the sampled model tensors.
- DECISION: RETAIN, but the hard gate remains OPEN: essay spec is still
  slightly below the contemporaneous 16.15-16.40 plain run even though
  GSM8K spec is >2x plain. Prompt-domain acceptance cannot excuse a
  slower supported arm under the spec-always-fastest rule.
- RAW: `perf/results/2026-08-22/qwen38-fused-gdn/mm_iq_bench.log`,
  `iq8_dequant_exact.log`, `rejection_mc_cdf.log`,
  `bench_spec_fused_mm.log`, and `bench_spec_fused_mm_vec.log`.

## 2026-08-22 (34) - Fused DFlash 2 Convolution; Final TPS Blocked by Power

- BASELINE/HYPOTHESIS: one proposal executes 20 two-tap convolution
  sides (two sides in each of two sublayers across five drafter layers).
  Each side was a decomposed repeat_interleave + roll + clone/zero +
  elementwise graph. Collapse it to one block-local Metal dispatch.
- IMPLEMENTED: `dflash2_two_tap_conv_{float32,float16,bfloat16}` computes
  both taps directly from compact per-group coefficients, zeros the
  predecessor at every eight-row block boundary, and preserves the
  activation-dtype rounding boundaries. The route has the kill switch
  `VLLM_QWEN38_FUSED_DFLASH2_CONV=0`.
- CORRECTNESS: both sides pass the torch reference at the real
  (3 blocks, 5120 hidden, 320 groups, BF16) geometry; differences are
  bounded to one BF16 ULP (relative <1%). The combined durable model /
  GGUF / GDN suite passes 14/14 after the deployed metallib rebuild.
- MEASURED: at the real 8x5120 proposal geometry, an alternating
  same-process microbench measured 0.0245 ms fused vs 0.1387 ms
  decomposed, 5.65x faster. This is a relative kernel result only.
- BLOCKER/DECISION: RETAIN pending powered end-to-end A/B. The Mac fell
  to 1% battery on a 20 W charger; model memory discovery regressed from
  ~20 s to 129.56 s and a phase probe reported only 1.28 tok/s. Those
  figures are power-throttling evidence, explicitly INVALID as a
  baseline. Re-run plain/spec, convolution on/off, text/image, and the
  real SlimServe server only after adequate high-wattage power is
  restored. Do not un-gate the profile yet.
- RAW: `perf/results/2026-08-22/qwen38-fused-gdn/dflash2_conv_microbench.log`,
  `power_blocker.log`, and `spec_profile_top_cdf.log` (invalid absolute
  profile retained only as power evidence).

## 2026-08-23 (35) - Hybrid KV Gather Fixed Beyond the 32-Bit Offset Boundary

- Status: retained.
- Scope: `qwen38-q2kxl-1`, Metal hybrid full-attention KV read path.
- Baseline: a 20-request repeated speculative run was correct for requests
  0-13, emitted only token 0 / `!` for requests 14-17, then recovered after
  the allocator wrapped. The first target logits of the bad requests were
  already all NaN; fused GDN and fused rejection kill switches did not alter
  the failure.
- Root cause: stage-finite hooks localized the first nonfinite value to layer
  19's full-attention output. Its physical block was 1271. Qwen's aligned
  attention page is 832x4x256 = 851,968 elements and K pages have a physical
  stride of 2x that value, so this block is beyond a signed 32-bit element
  offset. A standalone 5.05 GiB interleaved-cache reproducer proved scalar
  reads correct at blocks 1186/1271/1580 while MPS `index_select` on the
  strided pages view returned wrong/zero data at 1271/1580.
- Change: added `kv_cache_gather_range` in `kv_cache.metal`, its common
  launcher and pybind/wrapper, and routed Metal SDPA reads through it. The
  kernel receives the physical cache block stride explicitly and performs all
  address arithmetic in 64 bits. It gathers only the live logical token range;
  the existing Python gather remains the fallback.
- Correctness: direct large-cache gathers are exact at blocks 1186, 1271, and
  1580. The opt-in durable >2^31-element test passes. A 20x64-token reference-
  sampler speculative run stayed finite, coherent, and token-identical across
  every request, including the old failure window and allocator wrap.
- Decision: retain. This was a silent correctness failure in the owned cache
  layer, not a model, sampler, or GDN problem.
- Raw artifacts: `perf/results/2026-08-23/qwen38-kv-gather/run_summary.json`;
  reproducer and long-run scripts remain under
  `perf/results/2026-08-22/qwen38-fused-gdn/`.

## 2026-08-23 (36) - Powered Fusion A/B and Metal Draft Depth Sweep

- Status: retained.
- Scope: fused DFlash convolution, fused rejection, and DFlash 2 verify depth
  on the M5 Max; shipped sampling defaults, seed 42, AC power.
- Baseline: at the trained/upstream depth k=7, fresh powered medians were
  16.18 tok/s essay and 38.04 GSM8K-style versus plain 17.14 and 16.81. The
  essay hard gate remained open. Low-perturbation timing measured 101.33 ms
  target plus 23.61 ms `sample_tokens` per step; the latter already includes
  the nested 15.96 ms proposal.
- Fusion A/B: fused DFlash convolution medians were 16.42/38.12 versus the
  decomposed path's 15.75/36.22 (essay/GSM8K). Balanced same-process rejection
  A/B kept identical seeded text and measured fused medians 15.641/38.207
  versus reference 15.511/37.713. Both native paths are retained.
- Hypothesis: essay accepts only about two tokens per step, so verifying all
  seven trained suffix positions wastes more target bandwidth than it earns.
  Metal's quantized MM routes naturally cover total verify widths M=2,4,8;
  therefore sweep draft depths k=1,3,7 without changing sampling semantics.
- Results: k=1 reached 20.86 tok/s (1.56 tokens/step, target/tail 58.82/12.99
  ms); k=3 reached 23.22 (2.10 tokens/step, 70.19/16.42 ms); k=7 reached
  15.86 (2.03 tokens/step, 101.33/23.61 ms). A full 3x256-token k=3 run was
  essay 23.27/23.74/23.06 and GSM8K 34.33/35.25/34.78, all seed-stable. The
  matched plain runs were essay 16.99/17.14/17.17 and GSM8K
  16.77/16.86/16.81.
- Decision: set the registered Metal profile to k=3. It clears the always-on
  speculation gate by 35.8% on the essay medians and 106.9% on GSM8K while
  preserving lossless rejection sampling. The drafter remains trained at
  block 8; serving intentionally evaluates its first three suffix positions.
- Raw artifacts: `perf/results/2026-08-23/qwen38-kv-gather/run_summary.json`
  and the A/B/depth scripts under
  `perf/results/2026-08-22/qwen38-fused-gdn/`.

## 2026-08-23 (37) - Qwen3.8 Profile Promoted and Exact Server Path Validated

- Status: retained; profile supported.
- Scope: exact `slimserve qwen38-q2kxl-1 --serve` path, text and image, plus
  exact-token spec/plain comparison.
- Change: the profile now registers DFlash k=3, exports
  `VLLM_USE_V2_MODEL_RUNNER=1`, and no longer carries the stale "no Metal
  path" status gate. The live-smoke validator now compares each executable
  plan with its registered speculator; DSpark plans still additionally require
  TurboQuant attention and draft KV, while registered DFlash plans are no
  longer rejected before load. The exact-token harness gained explicit
  temperature/top-p/top-k/seed controls so this profile is never benchmarked
  greedily.
- Server correctness: the exact profile (131072 max length, 12 GiB KV pool)
  reached health in 15.09 s. The chat endpoint answered the text check with 4
  and the deterministic solid-red image check with Red. Reasoning parser,
  qwen3_xml tool parser, chat template, vision tower, and DFlash were all on
  the real API-server route.
- Exact-token result: 128 input + 256 output tokens, concurrency 1, 8-token
  warmup, temperature 1/top-p .95/top-k 20/seed 42. Spec k=3 produced exactly
  256 tokens at 18.646 tok/s with 111 accepted of 432 drafted tokens across
  144 draft steps (1.77 emitted tokens/step); plain produced exactly 256 at
  15.913 tok/s. Spec wins by 17.2%; both responses had healthy visible-text
  density (>2.6 chars/token).
- Validation: final native build succeeded; deployed symbols include the
  gather, GDN, convolution, and rejection ops. Focused suite including the
  large offset case: 17 passed. SlimServe suite: 58 passed, 1 skipped. Final
  metallib SHA-256 is
  `539035eb15dea29152e11503fc1ee08676d5dfe08b9ef4cc241283092e887d4c`.
- Raw artifacts: `perf/results/2026-08-23/qwen38-kv-gather/exact_spec.json`,
  `exact_plain.json`, `run_summary.json`, `smoke.json`, and
  `smoke/qwen38-q2kxl-1.log`.

## 2026-08-23 (38) - Current-Machine Live Matrix: Muse Fixed, DSV4 Regressed

- Status: Qwen and Muse pass their registered live arms; the complete matrix
  remains failed on an independent DSV4 Metal performance regression.
- Scope: registry discovery found all three compatible profiles on the M5 Max:
  `dsv4-xxs-1`, `muse-kdyn-1`, and `qwen38-q2kxl-1`. Every run used its
  registered target, drafter, KV configuration, and modalities.
- Muse root cause: the reasoning parser's initial prompt-state check treated a
  historical `<|message|>` as proof that the new assistant turn had already
  left reasoning. That bypassed the parser and exposed the generated
  ` to=self<|message|>` header as content. The header also arrives split as
  `" to"` then `"=self<|message|>..."`, which the old partial-header guard did
  not hold.
- Muse change/correctness: a rendered prompt ending in the newest bare
  `<|start|>assistant` now begins a fresh reasoning phase, recipient headers
  accept the real optional leading space, and split `to=` prefixes are held
  until parseable. Raw SSE now puts the self body only in
  `reasoning_content`, never exposes the control header, and emits the final
  answer in `content`. Registered text and solid-red image requests passed
  with DFlash k=16; load was 18.09 s and requests took 4.81/9.20 s. The parser
  plus SlimServe suite is 62 passed, 1 skipped.
- DSV4 evidence: `dsv4-xxs-1` loaded its 93.63 GiB target+DSpark stack and
  reached API health, but the first tiny text request was still running after
  about 12 minutes. Interval diagnostics reported only 0.1 generation tok/s;
  after the first draft there were 0 accepted of 5. This is orders of
  magnitude below the 33.684 tok/s correctness-qualified Metal baseline, so
  the run was terminated and is a hard profile failure, not a smoke pass.
- Decision: retain the Muse parser fix and Qwen profile promotion. Do not call
  the current-machine matrix green until DSV4's first-step/verify regression
  is isolated and the exact registered profile completes the real request.
- Raw artifacts:
  `perf/results/2026-08-23/qwen38-kv-gather/smoke-muse-final.json`,
  `smoke-muse-final/muse-kdyn-1.log`, and
  `smoke-all/dsv4-xxs-1.log`.
