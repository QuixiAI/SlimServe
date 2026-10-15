# LTX-2.5 on Apple Silicon: research survey (2026-09-30)

Pre-campaign survey for LTX-2.5 support + Metal kernel optimization on the
M1 Ultra 128 GiB Mac Studio. Three parallel web sweeps (Mac projects,
architecture/official assets, kernel prior art). Everything below is as
published by the linked source; "est." = our arithmetic, not measured.

## 1. Model facts (Lightricks/LTX-2, HF Lightricks/LTX-2.5, mlx-community config)

- DiT: dual-stream audio+video, 48 layers. Video 32 heads x 128 = 4096
  hidden, FFN 4x GELU; audio 32 x 64 = 2048. ~19B DiT (video ~14.5B,
  audio ~4.4B) + 3.2B text connectors = the marketed "22B". bf16 42 GB.
- Per block: full 3D self-attention over all F'xH'xW' latent tokens (not
  factorized), text cross-attn, bidirectional AV cross-attn, FFN. AdaLN-single,
  gated attention (2.5: out = attn * 2*sigmoid(gate)), qk RMSNorm, split RoPE
  with fractional 3D positions (fp64 freqs), patch 1x1x1 (tokens = voxels).
- Text encoder: Gemma-4 12B fine-tune + projection (26 GB bf16), 8-layer
  connector transformer.
- VAE: 128 latent ch, 32x spatial / 8x temporal, causal conv3d encoder.
  Decoders: conv (0.81 GB) or diffusion decoder (needs NATTEN; eager fallback
  on macOS ~10-12x slower, +24 GB). Audio VAE + HiFi-GAN vocoder 0.36 GB.
- Tokens = ceil(F/8) * H/32 * W/32: 768x512x121 = 6,144; 1280x704x121 =
  14,080; 1536x1024x121 = 24,576. Audio ~25 tok/s.
- Distilled pipeline (CFG=1): stage 1 = 8 sigmas at half res, stage 2 = 2x
  latent upscale + distilled LoRA (rank 450), 3 sigmas. Dev model: 30 steps,
  2-3 forwards/step (CFG + STG).
- FLOPs per forward (est.): 6,144 tok ~0.21 PFLOP (attn ~15%); 14,080
  ~0.57 PFLOP (attn ~27%); 24,576 ~1.2 PFLOP (attn ~40%). Consistent with
  H100 1.22 s/step at 720p x 121f (paper Table 1, ~460 TFLOP/s effective).
- Weights: official bf16 dev/distilled 42 GB, ComfyUI int8 "convrot" 21.5 GB,
  NVFP4 18.7 GB (Blackwell only). Community GGUF Q2_K 7.9 .. Q8_0 22.7 GB
  (joeygambino/LTX-2.5-Quantized, Abiray, realrebelai). MLX:
  mlx-community/ltx-2.5-mlx (bf16 ~110 GB set), -ditq8 (20.6 GB, cos 0.9982),
  -q8 (Gemma int8 13.6 GB). License: LTX-2.x Community License (gated).

## 2. Mac projects (ranked as baseline candidates)

| # | Project | Stack | Kernels | LTX-2.5 | Best numbers |
|---|---|---|---|---|---|
| 1 | dgrauet/ltx-2-mlx (122 stars, pushed 09-28) | Python MLX | stock mx.fast SDPA, MLX affine int8/int4 matmul, no custom Metal | yes, full pipelines | M5 Pro 64G 704x448x49 distilled int8 39.9 s / 20.9 GiB (vs PyTorch MPS bf16 142 s); M3 Ultra 1088x1920x97 stage-1 12.7-14.9 s/it, job 428 s (v0.15.5 regressed 3x, issue #138); M3 Max VAE 768x512x121 conv decode 6.0 s untiled / 52 GB peak; TeaCache ported (1.46-1.78x on 2.3) |
| 2 | Blaizzy/mlx-video PR #52 | Python MLX | stock | PR open 09-15 | M4 Max 128G 768x512x121 T2V 108.7 s / 37.8 GB; conv VAE 103 s(?) vs diff-VAE 149 s |
| 3 | Lightricks/LTX-Desktop (official, 2k stars) | PyTorch MPS bf16 + mps_sdpa | none | "LTX 2.5 Fast" | M5 Max 128G 1024x576x5s 142.75 s / 43 GiB; M1 Ultra 128G: solid green output (#165); M2 Ultra crashes in mps_sdpa (#171) |
| 4 | ComfyUI core MPS + GGUF | PyTorch MPS | none | bf16/GGUF | M1 Ultra (LTX-2.0) 29.79 s/it at 61f, 295 s/prompt; audio VAE >65536 ch fails; fp8 unsupported; black frames w/o split cross-attn |
| 5 | VincentGourbin/ltx-video-swift-mlx | Swift MLX | stock | yes | LTX-2.3 M3 Max 1024x576x241 bf16 1145 s; qint8/int4 SLOWER than bf16 |
| 6 | rinste/turbo-mlx (new 09-27) | Swift MLX | stock | yes | M1 Max: 2.5 5 s 768x512 with audio 197 s |
| 7 | james-see/ltx-video-mac (422 stars) | SwiftUI over ltx-2-mlx | stock | crashes (#88) | none |
| - | Draw Things | custom Metal (MFA, int8 attn, implicit-GEMM conv3d VAE) | yes | LTX-2 / 2.3 only, no 2.5 | conv3d VAE 2.4x on M1-M4; int8 attn + NAX are M5-only |

Only M1-generation LTX-2.5 datapoint: turbo-mlx 197 s (M1 Max, 5 s 768x512).
No M1 Ultra LTX-2.5 number exists anywhere. PyTorch MPS paths are 1.7-3.5x
slower than MLX and broken on M1/M2 Ultra.

## 3. Kernel prior art (Metal / MLX)

- No project ships custom Metal attention, conv3d, RoPE, or quantized-matmul
  kernels for LTX on M1-M4. ltx-2-mlx CLAUDE.md confirms stock SDPA.
- MLX SDPA: steel kernel, "optimized but not fully IO-aware" (mlx #2955);
  D128 supported. MLX conv3d: per-frame 2D Winograd decomposition (mlx #3785);
  implicit-GEMM only for batch>=2 (#4595).
- philipturner/metal-flash-attention: best-documented M1-class design, 62-86%
  ALU util on M1 family, head dim 64-256. ccv MFA v2 (Draw Things) builds on it.
- marcogva-hub/mlx-flashattention-steel: MLX extension with block-sparse,
  GNA, varlen, paged, quantized attention; M5 numbers only, M1-M4 get
  fallback kernels. Closest base for STA/Radial-style sparse attention.
- Draw Things conv3d VAE: implicit GEMM, 2.4x on M1-M4, 4x over MPSGraph;
  source only partially public (liuliu/example_matmul_metal4).
- FastVideo FastMetal-QAD (Wan only): int8 DiT weights, M4 Max 5B 720p 5 s
  151 s. Finding across projects: weight quant gives ~0x speedup on M-series
  when weights fit (compute-bound); attention FLOPs dominate.
- Caching: TeaCache ported (ltx-2-mlx, mlx-teacache); FBCache/MagCache/
  EasyCache unported but trivial (residual-diff gate). Sol Engine (NVlabs)
  published an LTX-2.5-specific recipe: FBCache + query-thresholded sparse
  attention + fusion, 1.45-1.90x e2e distilled (CUDA).
- Sparse video attention (STA, SVG2, Radial, SPADE, LiteAttention): 1.6-3x on
  Hunyuan/Wan, all CUDA. fp8 is emulated on Metal 4.1 (0.94x fp16): dead end.
  SageAttention: no Metal port planned; Draw Things int8 attn is M5-only.
- arXiv 2025-26 with Metal kernels: Rigel (2606.12765, fused GEMM+bias+GELU
  +6.5-12.9%), Open-TQ-Metal (2604.16957, int4 SDPA), BaseRT (2607.00501,
  simdgroup_matrix GEMM + online-softmax attn). Nothing video-DiT specific.

## 4. Physics on M1 Ultra (est.)

- Peak: ~20.8 TFLOP/s fp32/fp16 (64-core GPU), 800 GB/s. Distilled 5 s
  768x512 clip: stage 1 (1,536 tok x 8) ~0.37 PFLOP + stage 2 (6,144 tok x 3)
  ~0.63 PFLOP = ~1.0 PFLOP DiT -> 48 s at 100% ALU, ~70 s at 70%. Plus conv
  VAE decode (~6-10 s class), Gemma encode (small), audio.
- Weight read per forward 38 GB bf16 / 19 GB int8 = 48 ms / 24 ms at 800 GB/s:
  compute-bound at every practical token count. Quantization is for fitting,
  not speed, unless the matmul kernel itself is faster.
- Expected MLX baseline on M1 Ultra: 100-200 s for the same clip (extrapolated
  from M1 Max 197 s and M5 Pro 39.9 s at smaller res). Headroom to physics
  ~1.5-3x on kernels; caching/sparse attention multiplies on top.

## 5. Local state

- Machine: M1 Ultra 64-core GPU, 128 GiB, macOS 15.7.2, Xcode 26.3 / SDK 26.2.
- conda env vllm-mlx: mlx 0.31.2, torch 2.12.0 (MPS ok). No LTX weights cached.
- ~/Code/QuixiCore-Metal has attn_fwd/attn_fwd_sg/cross_attn/rotary kernels
  plus gemm_staged/gemm_v3 and MLX + torch-MPS bindings; no conv3d (listed as
  "no family claim" in capability-gaps.md).
- SlimServe perf/ has no video or diffusion notes; metal_m1ultra_retrospective.md
  holds the MPS lessons (residency pinning, transient-alloc churn, fusion).
