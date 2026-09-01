# XPU kernel survey (2026-08-18): what the sibling Intel trees teach the SlimServe GGUF path

Sources: the reference vllm-xpu-kernels tree (HEAD 48ef52d, byte-identical
to the installed copy), the vllm fork there (046fdfdba), sonar/ (CUDA
GGUF reference + humming/int8 Xe2 kernels), STAGED_IMPROVEMENTS.md,
CRASH_ANALYSIS_decode_free.md, resume_optimization_work.md, improvements_list.md.
None of these trees has GGUF/IQ2/Q2_K SYCL code; the value is in patterns and
measured numbers.

## Anchors (measured there, Arc Pro B60 unless noted)
- Rooflines used: 456 GB/s, 90 TFLOP/s bf16, 182 TOPS int8. Best kernels reach
  84-91% (int8 act-quant 415 GB/s, moe_gather ~412, concat_and_cache 386).
- fp8 GEMV: N=2048,K=2048 latency-bound at 42% (191 GB/s); N=4096 BW-bound at 83%.
- Decode host-bound: ~1134 device kernels/token, device 82% idle; SYCL submit
  ~2.4 us/kernel vs torch dispatch ~23 us/op. Per-op fusions measured ZERO e2e.
- XPU graphs: 36.9 -> 67-68 tok/s (single B60, Qwen3.6-35B NVFP4) once the split
  MoE could be replayed. Prereqs: out=/static-buffer ops (out-of-place decode ops
  under FULL_DECODE capture -> free(): invalid pointer after ~128 tokens), no
  host sync in capture (repeat_interleave(..., output_size=)).
- 4x B70 (their commit msg): per-worker ZE_AFFINITY_MASK = host RSS 118 -> 34 GiB,
  +18% tok/s; custom P2P all-reduce hangs under pinning; oneCCL faster anyway.
- `-fsycl-id-queries-range=size_t` on the decode TU: 8x regression (61.5 -> 6.9).
- Lane-stride lesson: word-stride (4 B/lane) fixed a K=256 down-proj (+8% e2e) but
  regressed the K=2048 GEMV (less coalesced) -> choose stride granularity per K.
- Register M-tiling for M<=4: rejected twice (no reuse, register pressure).
- Grouped-GEMM (DPAS) vs GEMV crossover: M~48 tokens (VLLM_NVFP4_RELU2_GROUPED_M_MAX).

## Kernel skeletons to copy
- Decode GEMV (nvfp4_kernel.hpp): subgroup 32 per output row, 8 rows/WG (256),
  16 B weight loads (vec<uint32,4>), vec<T,8> activations, block scale applied to
  the block partial sum, reduce_over_group, lane0 store. No SLM.
- Routed MoE (nvfp4_moe_kernel.hpp): grid = tokens*topk pairs (no BLOCK_M padding,
  no gather); split gate/up -> g_buf -> SwiGLU+down with device atomic f32
  accumulate (+34% device) - the win only shows once launches are graph-replayed.
- Humming (sonar/.../humming.cpp): 4 lanes cooperatively load one 16 B word,
  select_from_group to fan out; scale via one load + group_broadcast (was 32
  redundant loads): +26.6% at M=1, +52.6% at M=16.
- Grouped GEMM (grouped_gemm/xe_2/gemm_xe2.hpp + gemm_xe2_policy.hpp):
  XE_DPAS_TT<8,float,bf16>, WGTile 128x128x16 / SG 4x2 for 4-bit block formats
  (tile_k == quant group; coarsening 16->32 cost ~10% accuracy), small-M ladder
  {8,16,32}x64 / SG 1x4, prefetch_dist 6 on A, B and the scale strip, scales
  decoded once per group into registers, applied to the B fragment before
  cute::gemm. Per-tensor scale stays outside the GEMM (applied per output segment).
  De-swizzle scale layouts at load. Recipe: NVFP4_GROUPED_GEMM_PLAN.md.
- P2P all-reduce (sycl/p2p_all_reduce.cpp): capturable one-shot over PCIe BAR,
  fixed 64 WG x 256, per-rendezvous flag arrays + generation counter, FIXED RANK
  ORDER accumulation (fp add non-associative -> silent TP divergence at world>2).
- llama.cpp CUDA GGUF reference (sonar/csrc/quantization/gguf/): moe_vec.cuh routes
  experts in grid.z; IQ2_XXS dot = int8 dot + 256-entry grid + sign byte, against
  q8_1 activations -> int8 DPAS reachable for prefill.

## Ranked plan for SlimServe's GGUF kernels (csrc/quixicore/xpu/.../gguf_routed.sycl.cpp, 88 GB/s today)
0. Roofline probe: same load pattern, no dequant math. Decides ALU- vs BW/latency-bound.
1. Lane occupancy per shape (already 32-wide units at K=4096; check the K=2048 down and
   the N-narrow dense Q8_0 layers), 8 rows/WG geometry, grid = pairs x row-tiles,
   K-split + atomic accumulate if WGs are still few at 1 token x 6 experts.
2. 16 B vector loads for weights and activations; hoist all invariants (format as
   template, no div/mod in the K loop, fold constants once).
3. IQ2_XXS: codebook (2 KB) + ksigns in SLM per WG; int32 sumi per 32-block, one
   scale multiply per block; branch-free sign apply ((v^m)-m).
4. q8_1-quantized activations to make the inner loop an int8 dot (and DPAS-ready).
5. Prefill: graft Q8_0/Q2_K/IQ2_XXS into the SYCL-TLA xe_gemm_4bits mainloop
   (tile_k 32; in-register grid expansion before reorder), reuse the small-M ladder;
   tune the GEMV/GEMM crossover as an env knob.
6. System: XPU graphs (out= variants first), native mqa_logits (QuixiCore-XPU has
   the XMX kernel), then all-reduce.
