> HISTORICAL RECORD (moved from the repo-root `handoff.md` on 2026-08-10 to
> resolve the case collision with `HANDOFF.md`, the current resume document).
> The throughput baselines in this file (e.g. 92/112.6 tok/s TP2/TP4) predate
> the 2026-08-09 rejection-sampler NaN fix and the token-129280 poisoning
> discovery -- serving-level numbers here are OBSOLETE. The kernel and
> ownership design history (mHC transitions, publication protocols, us-level
> microbenchmarks) remains the valid record of what was tried and rejected.

# DSV4 0731 A100 Optimization Handoff

Updated: 2026-08-09 UTC

## 2026-08-09 Continuation Status

- The retained exact baseline is now 92.022 tok/s at TP2 and 112.573 tok/s at
  TP4 after merging the two sparse-MLA value-source launches. Scaling is
  1.223x; TP4 still needs at least 138.03 tok/s.
- End-to-end output-owned hidden state was rejected at 79.683 tok/s TP2 and
  97.904 tok/s TP4. Its direct native MoE boundary scales, but fixed mHC and
  publication work erase the gain.
- Channel-residual mHC, fused shared-expert publication, cooperative/direct
  peer consumption, and a single-leader urgent-start rendezvous were all
  correctness-clean and slower. Their measurements and decisions are in
  `perf/optimization_status.md`; raw data is under
  `perf/results/2026-08-09/`.
- Rejected shared-publication APIs are removed from Python and libtorch
  bindings. Their CUDA diagnostics are compile-disabled, and the retained
  output-owned CTA state was restored. The control is exact at 47.64 us TP2
  and 33.47 us TP4 (1.423x).
- The single-leader urgent-start candidate regressed TP4 fused norm graph
  latency from 28.86 us to 35.08 us and was removed completely.
- The extension was rebuilt after cleanup. Native import, schema smoke,
  `git diff --check`, and 39 focused tests pass.
- The demonstrated root cause remains 85 global mHC transitions plus broadly
  replicated batch-one work. TP4 urgent mHC averages about 26.5 us versus
  17.9 us at TP2; local fusion, added publication protocols, and added
  rendezvous points are exhausted designs, not credible next steps.

## Objective And Non-Negotiable Requirements

Improve DeepSeek V4 Flash 0731 inference on Ampere A100 at TP2 and TP4 to
parity with the optimized DSV4 ROCm and GLM 5.2 Ampere profiles.

- SlimServe owns the full inference path through the CUDA kernels.
- Develop and benchmark kernels in this repository first. Port only the final,
  retained kernels to `/home/ubuntu/QuixiCore/QuixiCore-CUDA` afterward.
- The routed MoE production boundary must remain native and fused:
  `IQ2_XXS gate/up -> SwiGLU -> Q2_K down -> weighted reduce`.
- Support vLLM's combined `[expert, gate | up, packed]` GGUF layout or a
  byte-neutral split/aligned representation.
- Do not dequantize the model and do not present standalone Q2_K GEMV as the
  final solution.
- SlimServe automatically downloads missing artifacts. Use the real profiles;
  do not treat a missing local model as a blocker.
- TP4 must be at least 1.5x TP2. Similar TP2 and TP4 throughput is evidence of
  a fundamentally wrong design, not an acceptable local performance issue.
- TurboQuant and DSpark must ultimately pass correctness, acceptance, and
  exact TP2/TP4 throughput qualification.

Read `AGENTS.md`, `perf/perf.md`, `perf/baseline_status.md`, and
`perf/optimization_status.md` before changing performance-sensitive code.
`CLAUDE.md` duplicates the repository operating guidance for Claude-based
agents.

## Repository State

- Repository: `/home/ubuntu/SlimServe`
- Branch: `main`
- HEAD before the uncommitted work: `b3522ca4de5d9dc6a7a6c72a050e2b0878357d31`
- Hardware: 8x NVIDIA A100-SXM4-80GB, all pairs connected by NV12.
- The controlled experiments used 1,410 MHz locked SM clocks. The lock was
  reset with `sudo -n nvidia-smi -rgc` after the final restored-production
  smoke; lock clocks again before making direct performance comparisons.
- The worktree is intentionally very dirty and contains accumulated kernels,
  benchmarks, documentation, and performance data. Inspect `git status` rather
  than relying on an old file count.
- Do **not** reset, checkout, clean, or otherwise discard this tree. It contains
  the implementation under evaluation as well as prior/user work.
- There are no known live serving sessions. PID `1818689` is a week-old,
  sleeping orphan named `VLLM::EngineCore`; it has no reported GPU allocation.
  It was deliberately not killed in case another task owns it.

The native extensions were rebuilt after the most recent rejected experiments
were removed:

- `vllm/_C_stable_libtorch.abi3.so`, built 2026-08-08 22:39 UTC
- `vllm/_quixicore_C.cpython-312-x86_64-linux-gnu.so`, built 2026-08-08
  19:06 UTC

Both import successfully from `.venv`. `git diff --check` passes. Registered
production ops include:

```text
_C::ggml_dsv4_moe_a8
_C::ggml_dsv4_mul_mat_vec_aligned_q8_0
_C::ggml_dsv4_o_proj_q8_0
_C::ggml_dsv4_repack_iq2_xxs
_C::ggml_dsv4_repack_q2_k
_C::ggml_dsv4_repack_q8_0_aligned
_C::ggml_dsv4_rms_norm_q8_1
_C::ggml_dsv4_shared_gate_up_swiglu
_C::ggml_dsv4_shared_gate_up_swiglu_q8_1
_C_custom_ar::all_reduce_dsv4_mhc
_C_custom_ar::wait_dsv4_mhc
```

## Current Exact Baseline

Model on both TP levels:

```text
/home/ubuntu/models/antirez-deepseek-v4-gguf/
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

Workload: concurrency 1, 1,000 input tokens, exactly 2,000 generated tokens,
8-token warmup, no speculation, full decode CUDA graphs.

| Profile | Hardware | Exact runs (tok/s) | Median tok/s | Median wall |
| --- | --- | --- | ---: | ---: |
| `dsv4-2` | 2x A100 | 90.808, 90.862 | 90.835 | 22.017 s |
| `dsv4-4` | 4x A100 | 110.232, 110.205 | 110.218 | 18.146 s |

TP4/TP2 is only **1.213x**. The minimum project gate is 1.5x, so TP4 must be
at least **136.25 tok/s** at the current TP2 result. The present TP4 baseline
is 26.03 tok/s, or 19.1%, below that minimum. Treat this as an active failed
scaling baseline, not a completed optimization result.

The benchmark's `exact=true` verifies the requested prompt/completion token
counts. Native kernel correctness was also checked with focused synthetic
parity tests, but do not mistake exact token counts alone for full numerical or
quality equivalence.

Canonical raw results:

- `perf/results/2026-08-08/dsv4-a100-indexer-group4-tp2/`
- `perf/results/2026-08-08/dsv4-a100-mhc-low-priority-tp4/`

The top of `perf/baseline_status.md` and the 2026-08-08 sections of
`perf/optimization_status.md` are authoritative. Older lower sections in the
notebook may still describe pre-baseline diagnostics and should not override
the current numbers.

## Retained Native Implementation

The currently built source contains these retained improvements:

- Native routed IQ2_XXS gate/up with paired decode, SwiGLU, direct Q8_1
  intermediate emission, repacked Q2_K down, and weighted reduction.
- The real vLLM combined W1 expert layout is accepted; weights are not expanded
  into a persistent dequantized expert stack.
- Byte-neutral aligned Q8_0 SoA, paired/aligned IQ2_XXS SoA, and repacked Q2_K
  load-time/first-use layouts.
- Native Q8 shared experts and output projection, including shared gate/up,
  SwiGLU, and prequantized down input.
- Native BF16 projection GEMV with FP32 accumulation.
- Native router and native hash router.
- Active-channel MLA partitioning and Ampere sparse MLA kernels.
- Graph-capturable persistent paged indexer logits, live-prefix sizing, and
  grouped-token H64 indexer scheduling.
- Native mHC, fused custom all-reduce plus urgent mHC, FP16 mHC projection
  weights, and low-priority deferred mHC work.
- Correct TP routed-expert layout. Expert parallelism is disabled by default;
  each rank has selected experts and shards the intermediate dimension. Random
  expert-owner imbalance is not the cause of the scaling failure.
- DSpark's final deferred target tuple is consumed before target-model return.

Important implementation locations:

- `csrc/quixicore/quant/dsv4_moe_ampere.cuh`
- `csrc/quixicore/quant/dsv4_q8_ampere.cuh`
- `csrc/quixicore/quant/dsv4_shared_ampere.cuh`
- `csrc/quixicore/quant/dsv4_o_proj_ampere.cuh`
- `csrc/quixicore/serving/dsv4_projection_ampere.cuh`
- `csrc/quixicore/serving/dsv4_router_ampere.cuh`
- `csrc/quixicore/serving/mhc_ampere.cuh`
- `csrc/quixicore/serving/mhc_allreduce_ampere.cuh`
- `csrc/quixicore/serving/indexer_paged_logits.cuh`
- `csrc/quixicore/serving/mla_kernels.cuh`
- `csrc/quixicore/serving/paged_attn_v2_kernels.cuh`
- `csrc/libtorch_stable/quantization/gguf/gguf_kernel.cu`
- `csrc/libtorch_stable/quantization/gguf/mmvq.cuh`
- `csrc/custom_all_reduce.cuh`
- `vllm/model_executor/layers/quantization/gguf/fused_moe.py`
- `vllm/model_executor/layers/quantization/gguf/linear.py`
- `vllm/model_executor/layers/mhc.py`
- `vllm/model_executor/layers/sparse_attn_indexer.py`
- `vllm/models/deepseek_v4/ampere.py`
- `vllm/models/deepseek_v4/attention.py`
- `vllm/models/deepseek_v4/amd/dspark.py`
- `vllm/quixicore/ops.py`

The design record is `csrc/quixicore/dsv4_ampere_design.md`. The performance
notebook is newer than some stage wording in that design file; inspect the
actual source and notebook before assuming a listed future stage is absent.

## Measured Progression

On the evolving same-day tree, the first corrected TP4 exact run was 105.364
tok/s. Retained changes then measured approximately:

- Native projection: 108.522-108.549 tok/s.
- Native hash routing: 108.832-108.854 tok/s versus 108.537-108.598 control.
- H64/live-prefix indexer work: about 110.09-110.14 tok/s.
- Low-priority deferred mHC: 110.205-110.232 tok/s.

These are real incremental gains, but they do not solve TP scaling.

## Current Bottleneck Diagnosis

The 2026-08-08 all-rank root-cause audit supersedes any interpretation that
this is one slow GEMV or one bad GPU placement:

- Exact token time is 11.009 ms at TP2 and 9.073 ms at TP4. The 1.5x gate is
  7.339 ms, leaving 1.734 ms/token to recover.
- An Amdahl fit to those exact runs gives a 7.137 ms replicated/non-scaling
  floor, which is 78.7% of current TP4 latency. The current ownership design
  asymptotes to only 1.543x TP2 even at infinite TP.
- Across every rank and seven graph-id-26 replays, intended-sharded quantized
  work improves 1.466x and intended-sharded attention only 1.075x. Urgent mHC
  gets 0.677x slower, deferred mHC gets 0.936x slower, and the explicitly
  replicated indexer/router/projection bucket gets 0.951x slower.
- Both dry runs resolve the same IQ2_XXS/Q2_K model and decode settings. Every
  GPU pair is NV12. This is not a profile, model, or topology mismatch.

The reproducible classifier is `benchmarks/analyze_dsv4_tp_scaling.py`; raw
output is under
`perf/results/2026-08-08/dsv4-a100-tp-scaling-root-cause/`. The full table and
decision are at the top of `perf/optimization_status.md`.

Warm decode CUDA-graph traces show:

- TP2 graph id 26: 1,513 kernels/replay, typically 10.75-10.90 ms warm wall.
- TP4 graph id 26: 1,470 kernels/replay, roughly 9.01-10.65 ms as traced
  context changes.
- TP2 aggregate kernel duration was about 14.91 ms/step and TP4 about
  14.0-14.4 ms/step; streams overlap, so these sums exceed wall time.
- In the retained graph-id-26 captures, routed/aligned quantized work saves
  about 1.7 ms at TP4, while urgent mHC grows by about 0.795 ms and deferred
  mHC by about 0.099 ms. mHC therefore erases more than half of that sharding
  gain before accounting for stream overlap.
- Routed weight work shrinks at TP4, but replicated projections, indexer work,
  mHC work, and two synchronization boundaries per layer dominate what remains.
- Fused urgent mHC boundaries average roughly 37 us/layer at TP2 and 55
  us/layer at TP4, accounting for about 0.8 ms/token of the scaling gap.
- Deferred mHC projection is replicated and costs about 1.5-1.6 ms aggregate
  kernel time/token, although it runs on a low-priority stream.

The first cross-rank analysis incorrectly compared raw profiler timestamps
from different GPUs. Barrier-end events expose rank clock offsets relative to
rank 0 of approximately `[0, -10.268, -30.089, -15.771]` us. After applying
those offsets, TP4 producer/urgent timing is:

- Aligned-Q8/non-add boundary: median start skew 16.442 us, p95 35.926 us,
  median end skew 2.005 us.
- Routed-Q2/add boundary: median start skew 10.192 us, p95 19.252 us, median
  end skew 1.898 us.
- Each local producer ends within about 0.3-0.9 us of its urgent launch.

The collective itself also scales poorly: its all-rank mean duration is
17.943 us at TP2 and 26.499 us at TP4. Every non-add urgent event immediately
follows `aligned_q8_0_q8_1_gemv_kernel`; every add event immediately follows
`q2_k_down_sum_repacked`. Aligned Q8 is the largest measured local quantized
cost at roughly 1.83 ms aggregate per token, but controlled selector and
one-warp kernel experiments did not produce a meaningful serving gain.

Relevant traces:

- `perf/results/2026-08-08/dsv4-a100-active-indexer-trace-tp2/`
- `perf/results/2026-08-08/dsv4-a100-active-indexer-trace-current-tp4/`

Production `csrc/quixicore/serving/mhc_allreduce_ampere.cuh` currently uses 16
splits and 256 threads. Every block performs the start handshake; all ranks
compute four urgent mHC rows. After grid synchronization, split 0 performs the
release handshake and finalization. The remaining 20 rows run replicated on a
low-priority deferred stream. Both A100 profiles set
`VLLM_DSV4_MHC_SCHEDULE=async`.

## Most Recent Work And Rejected Experiments

The latest candidates were fully removed before the final production rebuild:

1. mHC schedule A/B retained low-priority async at 110.206 tok/s median;
   sequential reached 101.660 and monolithic 102.972.
2. Two deferred mHC sharding designs reached 99.357 and 109.656 tok/s. A third
   owner-compute design was four-rank exact but reached only 105.596 tok/s.
   Added barriers/fences erased the benefit of removing replicated work.
3. Wider Q2 CTAs were about 4.7% slower. A cooperative IQ2/SwiGLU/Q2 kernel
   improved isolated down timing about 3% but was neutral/slower in the real
   graph: 110.362-110.382 versus 110.409-110.454 control.
4. Existing Q8 row-selector changes fell to 109.516-109.605 versus
   110.282-110.358 control. DS4/QuixiCore-inspired one/two-row-per-warp kernels
   produced at most +0.07% end to end, so they were also removed. The Q8
   benchmark now times prequantized GEMV and rotates weights beyond L2.
5. Moving rank-local urgent work before the peer barrier was exact but fell to
   109.831-109.902 versus 110.255-110.284 restored control.
6. Deferred projection output partitions were swept at 16, 32, and 64 CTAs.
   The 64-CTA version improved isolated graph timing but fell to
   109.667-109.693; 16 CTAs measured 110.203-110.242. The retained 32-CTA
   geometry is the end-to-end winner.
7. Splitting deferred partials and Sinkhorn into two low-priority graph nodes
   was exact and slightly faster in isolation, but fell to 109.536-109.596.
8. A corrected eight-physical/16-logical urgent design preserved the original
   16-way numerical tree and was exact across 32 seeds. Its serialized CTA
   work raised norm graph latency to 31.597 us and exact TP4 fell to 105.638.
   It was removed; this is distinct from the older invalid eight-split test.

The source should contain none of these rejected experiment symbols:

```bash
rg 'AsyncOwner|deferred_owner|owner_deferred_fence|row_pair_kernel|COOPERATIVE_MOE|Q2_DOWN_WARPS|PHYSICAL_SPLITS|OWNED_SPLITS|urgent_splits' csrc vllm
```

The search was empty at handoff time. The detailed decisions and raw artifact
paths are in `perf/optimization_status.md`.

Other rejected and removed designs include:

- Indexer head sharding: TP4 105.250-105.368 versus 108.848-108.864 control.
- Short top-512 radix path: TP4 110.141-110.213 versus 110.451-110.524 control.
- Fused aligned-Q8 projection plus RoPE/FP8 prep: graph latency regressed from
  14.69 us to 25.73-32.27 us.
- Fused non-hash router plus top-k.
- Projection bundling.
- mHC leader-grid handshake, 8 splits, paired CTA, and alternate norm mapping.

Do not restore these without a materially different design and controlled
evidence. Full baselines, hypotheses, and artifact paths are recorded in
`perf/optimization_status.md`.

## Recommended Next Work

1. Replace the replicated H64 Lightning indexer with head ownership: TP4 ranks
   compute H16 query/weight projections and local partial logits, then the
   existing persistent top-k node peer-reduces scores before selection. Do not
   repeat the rejected token-shard plus generic all-gather implementation.
2. Design producer-progress collectives at both mHC boundaries. The local
   producer-to-urgent gap is only 0.3-0.9 us, so launch removal alone cannot
   matter enough. A useful kernel must publish tiles in a common persistent
   traversal and consume ready peer tiles while the producer tail runs,
   replacing the 10-16 us whole-rank arrival skew.
3. Merge the main and extra DSV4 sparse-MLA source launches into one
   source-selecting persistent kernel. They currently share query and reduction
   shape but create two source nodes per attention layer.
4. Remove replicated deferred mHC work only if publication piggybacks on the
   next existing transition. Sharded and owner-compute results prove that a new
   synchronization, fence, or graph node costs more than it saves.
5. Partition router rows and merge rank-local top-k candidates inside an
   existing routing node after the larger indexer and collective defects are
   fixed.
6. Revisit the ROCm and GLM low-batch designs before every structural change.
   At concurrency 1 this profile is launch/synchronization bound; epilogue
   fusion that does not change ownership or overlap synchronization is noise.
7. Once no-spec scaling passes, run fresh TP2/TP4 DSpark plus TurboQuant exact
   benchmarks and record draft acceptance. The DSpark final-deferred fix is in
   the tree and TurboQuant is configured by the profile, but neither has been
   acceptance-qualified on the final native tree. Older probes accepted zero
   draft tokens and are diagnostics only.
8. Update `perf/optimization_status.md`, `perf/baseline_status.md`, and raw
   result directories after every controlled experiment.
9. Port only the final retained SlimServe kernel set to
   `/home/ubuntu/QuixiCore/QuixiCore-CUDA`. This port has **not** been done.

## Running And Benchmarking

Use the same all-layer IQ2_XXS model at TP2 and TP4. SlimServe owns model
resolution/download, parser settings, KV dtype, graph settings, and speculative
configuration.

Inspect resolved commands first:

```bash
.venv/bin/python -m slimserve.cli dsv4-2 --quant IQ2_XXS --no-spec --serve --dry-run -y
.venv/bin/python -m slimserve.cli dsv4-4 --quant IQ2_XXS --no-spec --serve --dry-run -y
```

Run one server at a time on port 8000:

```bash
.venv/bin/python -m slimserve.cli dsv4-2 --quant IQ2_XXS --no-spec --serve -y
.venv/bin/python -m slimserve.cli dsv4-4 --quant IQ2_XXS --no-spec --serve -y
```

Canonical exact benchmark shape:

```bash
.venv/bin/python benchmarks/benchmark_dsv4_exact.py \
  --model /home/ubuntu/models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --served-model-name DeepSeek-V4-Flash \
  --source /home/ubuntu/ds4/tests/long_context_story_prompt.txt \
  --concurrency 1 \
  --input-tokens 1000 \
  --output-tokens 2000 \
  --warmup-output-tokens 8
```

Store stdout/stderr and benchmark JSON under a new
`perf/results/2026-08-08/<descriptive-run-id>/` directory. Some existing result
files contain a tokenizer info line before JSON, so direct `jq` may fail unless
that line is removed or skipped.

Native rebuilds:

```bash
cmake --build build/temp.linux-x86_64-cpython-312 \
  --target _C_stable_libtorch -j"$(nproc)"
cp build/temp.linux-x86_64-cpython-312/_C_stable_libtorch.abi3.so vllm/

cmake --build build/temp.linux-x86_64-cpython-312 \
  --target _quixicore_C -j"$(nproc)"
cp build/temp.linux-x86_64-cpython-312/_quixicore_C.cpython-312-x86_64-linux-gnu.so vllm/
```

Smoke and hygiene checks:

```bash
.venv/bin/python -c 'import vllm._C_stable_libtorch; import vllm._quixicore_C'
git diff --check
```

## Reference Implementations To Study

- DSV4 optimized ROCm notebook and design:
  `/home/ubuntu/QuixiCore/QuixiCore-ROCm/perf/`
- ROCm FP8 MQA logits design:
  `/home/ubuntu/QuixiCore/QuixiCore-ROCm/kernels/serving/variants/rocm_cdna3/fp8_mqa_logits_design.md`
- GLM 5.2 A100 notebook and design:
  `/home/ubuntu/QuixiCore/QuixiCore-CUDA/perf/`
- QuixiCore CUDA and ROCm sources:
  `/home/ubuntu/QuixiCore/QuixiCore-CUDA` and
  `/home/ubuntu/QuixiCore/QuixiCore-ROCm`
- DS4 implementation: `/home/ubuntu/ds4`
- vLLM Marlin/reference kernels:
  `/home/ubuntu/vllm/csrc/libtorch_stable/quantization`
- llama.cpp quant formats/kernels: inspect the local checkout if needed.

The key cross-platform lesson already confirmed is that low-batch TP decode is
launch and synchronization bound. ROCm's tile/layout wins and GLM 5.2's
collective/fusion boundaries should be starting points, but the final A100
implementation must be measured for this model and these exact shapes.
