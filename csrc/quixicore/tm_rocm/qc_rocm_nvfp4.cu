// NVFP4 (compressed-tensors) native GEMV/GEMM for gfx942, in the QuixiCore
// CDNA3 quant-kernel style: the packed E2M1 weight never leaves its 4-bit
// form in memory, activations are Q8_1 (int8 blocks of 32 with an fp16
// scale), the inner loop is byte-permute table decode feeding sdot4, and the
// FP8-derived NVFP4 block scale plus the per-tensor global scale fold into
// one f32 multiplier per 16-value group.
//
// Weight layout ("planar", produced by a load-time layout-only repack of
// vLLM's nvfp4-pack-quantized rows): per 32-value chunk of a row, byte j
// (j in [0,16)) holds value j in the low nibble and value j+16 in the high
// nibble. Values 0..15 are scale group 0 of the chunk, 16..31 group 1, so a
// table_lookup_16 word yields four group-0 values in `lo` and four group-1
// values in `hi` -- the same two-span pairing the mxfp4 MMQ kernel uses, but
// on 4-byte-aligned rows (no GGUF 17-byte block hazard).
//
//   weight_planar [N, K/2] uint8   (repacked as above)
//   scales        [N, K/16] fp16   (exact upcast of the fp8e4m3fn scales)
//   activations   block_q8_1 rows from torch.ops._C.ggml_quantize_q8_1
//   effective weight = e2m1 * scale * global_scale
//
// Two kernels, as the mxfp4 pair is split: a GEMV that streams weight rows
// (wins below ~8 tokens) and an LDS-tiled MMQ (wins above). The e2m1 table
// holds 2x values so it stays integral; the 0.5 folds into the group scale.
//
// The nibble decode and dp4a follow QuixiCore-ROCm
// include/cdna3/common/quant_primitives.cuh and
// kernels/quantization/mxfp4_gguf/variants/rocm_cdna3; the scale math and the
// planar layout are NVFP4's.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

namespace qcnvfp4 {

constexpr int kWave = 64;
constexpr int kQ8BlockInts = 9;  // block_q8_1: {half2 ds; int8 qs[32]} = 9 ints

// 2x the E2M1 values so the table is integral; the 0.5 folds into the scale.
__device__ __constant__ static const int8_t kvalues_e2m1_2x[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ void table_lookup_16(int packed, int& lo, int& hi) {
  const uint32_t* table = reinterpret_cast<const uint32_t*>(kvalues_e2m1_2x);
  const uint32_t l = (uint32_t)packed;
  const uint32_t h = ((uint32_t)packed >> 4);
  const uint32_t l_lo =
      __builtin_amdgcn_perm(table[1], table[0], l & 0x07070707u);
  const uint32_t l_hi =
      __builtin_amdgcn_perm(table[3], table[2], l & 0x07070707u);
  const uint32_t h_lo =
      __builtin_amdgcn_perm(table[1], table[0], h & 0x07070707u);
  const uint32_t h_hi =
      __builtin_amdgcn_perm(table[3], table[2], h & 0x07070707u);
  const uint32_t l_sel = ((l >> 3) & 0x01010101u) * 0xFFu;
  const uint32_t h_sel = ((h >> 3) & 0x01010101u) * 0xFFu;
  lo = (int)((l_lo & ~l_sel) | (l_hi & l_sel));
  hi = (int)((h_lo & ~h_sel) | (h_hi & h_sel));
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
  return __builtin_amdgcn_sdot4(a, b, c, false);
}

// ------------------------------------------------------------------- GEMV
// One wavefront per output row n; each lane consumes one 32-value chunk
// (= one q8 block = two scale groups) per pass.
template <int M_TILE>
__global__ __launch_bounds__(256, 4) void nvfp4_gemv_q8_kernel(
    const uint8_t* __restrict__ w,          // [N, K/2] planar
    const __half* __restrict__ s,           // [N, K/16]
    const int* __restrict__ y,              // [M, y_blocks * 9] q8_1 blocks
    __hip_bfloat16* __restrict__ out_bf16,  // [M, N]
    int M, int N, long K, int y_blocks, float half_global) {
  const int rows_per_block = blockDim.x / kWave;
  const int n = blockIdx.x * rows_per_block + (threadIdx.x / kWave);
  if (n >= N) return;
  const int lane = threadIdx.x % kWave;
  const int m0 = blockIdx.y * M_TILE;
  const int m_count = min(M_TILE, M - m0);

  const uint8_t* wrow = w + (long)n * (K / 2);
  const __half* srow = s + (long)n * (K / 16);
  const long nchunks = K / 32;

  float acc[M_TILE];
#pragma unroll
  for (int m = 0; m < M_TILE; ++m) acc[m] = 0.0f;

  for (long c = lane; c < nchunks; c += kWave) {
    // The weight stream is touched exactly once per call: stream it
    // around the cache so it does not evict the (reused) activations.
    const uint32_t* wq = reinterpret_cast<const uint32_t*>(wrow + c * 16);
    int lo[4], hi[4];
    table_lookup_16((int)__builtin_nontemporal_load(wq + 0), lo[0], hi[0]);
    table_lookup_16((int)__builtin_nontemporal_load(wq + 1), lo[1], hi[1]);
    table_lookup_16((int)__builtin_nontemporal_load(wq + 2), lo[2], hi[2]);
    table_lookup_16((int)__builtin_nontemporal_load(wq + 3), lo[3], hi[3]);

    const uint32_t s2raw = __builtin_nontemporal_load(
        reinterpret_cast<const uint32_t*>(srow + c * 2));
    __half2 s2;
    __builtin_memcpy(&s2, &s2raw, 4);
    const float s0 = __half2float(s2.x) * half_global;
    const float s1 = __half2float(s2.y) * half_global;

#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
      if (m >= m_count) break;
      const int* blk = y + ((long)(m0 + m) * y_blocks + c) * kQ8BlockInts;
      const __half2 ds = *reinterpret_cast<const __half2*>(blk);
      int q8[8];
      __builtin_memcpy(q8, blk + 1, 32);
      int sumi0 = 0, sumi1 = 0;
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        sumi0 = dp4a(lo[wd], q8[wd], sumi0);
        sumi1 = dp4a(hi[wd], q8[wd + 4], sumi1);
      }
      const float d8 = __half2float(ds.x);
      acc[m] = fmaf(d8 * s0, (float)sumi0, acc[m]);
      acc[m] = fmaf(d8 * s1, (float)sumi1, acc[m]);
    }
  }

#pragma unroll
  for (int m = 0; m < M_TILE; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1)
      v += __shfl_xor(v, off, kWave);
    if (lane == 0 && m < m_count)
      out_bf16[(long)(m0 + m) * N + n] = __float2bfloat16(v);
  }
}

// -------------------------------------------------------------------- MMQ
// LDS-tiled variant for larger token counts, the mxfp4_mmq geometry on the
// planar layout: MMQ_Y weight rows x MMQ_X activation rows per block, NWARPS
// subgroups of 32 lanes (wave64 runs two subgroups per wave; every cross-lane
// step goes through LDS, so the subgroup width is only an indexing choice).
#define MMQ_WARP 32
constexpr int kChunkInts = 4;  // int32s of packed nibbles per 32-value chunk

template <int MMQ_X, int MMQ_Y, int NWARPS, bool NEED_CHECK>
__global__ void __launch_bounds__(MMQ_WARP* NWARPS, 2)
    nvfp4_mmq_q8_kernel(const uint8_t* __restrict__ w,  // [N, K/2] planar
                        const __half* __restrict__ s,   // [N, K/16]
                        const int* __restrict__ y,      // [M, y_blocks * 9]
                        float* __restrict__ out_f32,    // [M, N]
                        int M, int N, long K, int y_blocks, float half_global) {
  const int chunks_per_row = (int)(K / 32);
  const int chunks_per_warp = MMQ_WARP / kChunkInts;  // 8 chunks staged/step

  const int row0 = blockIdx.x * MMQ_Y;
  const int col0 = blockIdx.y * MMQ_X;

  __shared__ int tile_w[MMQ_Y * (MMQ_WARP + 1)];
  __shared__ float tile_s[MMQ_Y * 2 * (MMQ_WARP / kChunkInts)];
  __shared__ int tile_y[MMQ_X * MMQ_WARP * 2];
  __shared__ float tile_yd[MMQ_X * (MMQ_WARP / kChunkInts)];

  float sum[MMQ_Y / MMQ_WARP][MMQ_X / NWARPS] = {{0.0f}};

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  for (int c0 = 0; c0 < chunks_per_row; c0 += chunks_per_warp) {
    __syncthreads();

    // Stage MMQ_WARP ints of packed nibbles per weight row (8 chunks).
    const int cb = tx / kChunkInts;  // chunk within the staged span
    const int ci = tx % kChunkInts;  // int within the chunk
#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS) {
      int i = i0 + ty;
      if (NEED_CHECK) i = min(i, N - row0 - 1);
      const uint8_t* wrow = w + (long)(row0 + i) * (K / 2);
      int q4;
      __builtin_memcpy(&q4, wrow + (long)(c0 + cb) * 16 + ci * 4, 4);
      tile_w[i * (MMQ_WARP + 1) + tx] = q4;
    }
    // Stage the two group scales of each staged chunk.
    if (tx < 2 * chunks_per_warp) {
#pragma unroll
      for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS) {
        int i = i0 + ty;
        if (NEED_CHECK) i = min(i, N - row0 - 1);
        const __half* srow = s + (long)(row0 + i) * (K / 16);
        tile_s[i * 2 * chunks_per_warp + tx] =
            __half2float(srow[c0 * 2 + tx]) * half_global;
      }
    }
    // Stage the q8 blocks of the same chunks for MMQ_X activation rows:
    // 8 ints of quants (2 per lane group) and the d scale.
#pragma unroll
    for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
      const int j = j0 + ty;
      const int col = NEED_CHECK ? min(col0 + j, M - 1) : (col0 + j);
      const int* blk = y + ((long)col * y_blocks + c0 + cb) * kQ8BlockInts;
      // Each lane copies two quant ints of its chunk.
      tile_y[(j * chunks_per_warp + cb) * 8 + ci * 2 + 0] = blk[1 + ci * 2];
      tile_y[(j * chunks_per_warp + cb) * 8 + ci * 2 + 1] = blk[2 + ci * 2];
      if (ci == 0)
        tile_yd[j * chunks_per_warp + cb] =
            __half2float(*reinterpret_cast<const __half*>(blk));
    }
    __syncthreads();

    // Compute: lane tx owns weight row (tx + i*MMQ_WARP), subgroup ty
    // walks activation rows.
#pragma unroll
    for (int cc = 0; cc < chunks_per_warp; ++cc) {
#pragma unroll
      for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
        const int j = j0 + ty;
        const float d8 = tile_yd[j * chunks_per_warp + cc];
        const int* q8 = &tile_y[(j * chunks_per_warp + cc) * 8];
#pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_WARP) {
          const int i = tx + i0;
          int sumi0 = 0, sumi1 = 0;
#pragma unroll
          for (int wd = 0; wd < kChunkInts; ++wd) {
            int lo, hi;
            table_lookup_16(tile_w[i * (MMQ_WARP + 1) + cc * kChunkInts + wd],
                            lo, hi);
            sumi0 = dp4a(lo, q8[wd], sumi0);
            sumi1 = dp4a(hi, q8[wd + 4], sumi1);
          }
          const float s0 = tile_s[i * 2 * chunks_per_warp + cc * 2];
          const float s1 = tile_s[i * 2 * chunks_per_warp + cc * 2 + 1];
          sum[i0 / MMQ_WARP][j0 / NWARPS] +=
              d8 * (s0 * (float)sumi0 + s1 * (float)sumi1);
        }
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < MMQ_X; j0 += NWARPS) {
    const int col = col0 + ty + j0;
    if (col >= M) continue;
#pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_WARP) {
      const int row = row0 + tx + i0;
      if (row >= N) continue;
      out_f32[(long)col * N + row] = sum[i0 / MMQ_WARP][j0 / NWARPS];
    }
  }
}

// ------------------------------------------- fused NVFP4 QDQ + Q8_1 quant
//
// Composes the activation NVFP4 quantize-dequantize round trip (w4a4
// semantics, matching vLLM's triton _nvfp4_quant_dequant_kernel bit for bit:
// E4M3FN round-to-nearest-even on the group scale via rintf, the same fp4
// threshold table) with the Q8_1 block quantization, removing one kernel and
// the intermediate bf16 activation buffer per linear call.

// Round v >= 0 to the nearest E4M3FN-representable value (RNE), v <= 448.
__device__ __forceinline__ float e4m3fn_round_pos(float v) {
  if (v <= 0.0f) return 0.0f;
  int e;
  (void)frexpf(v, &e);                  // v = m * 2^e, m in [0.5, 1)
  const int step_exp = max(e - 4, -9);  // 3 mantissa bits; subnormal floor
  const float step = ldexpf(1.0f, step_exp);
  return fminf(rintf(v / step) * step, 448.0f);
}

// The fp4 rounding thresholds, matching _round_to_fp4 exactly.
__device__ __forceinline__ float round_to_fp4(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float a = fabsf(x);
  float r = a > 5.0f ? 6.0f : 0.0f;
  if (a >= 3.5f && a <= 5.0f) r = 4.0f;
  if (a > 2.5f && a < 3.5f) r = 3.0f;
  if (a >= 1.75f && a <= 2.5f) r = 2.0f;
  if (a > 1.25f && a < 1.75f) r = 1.5f;
  if (a >= 0.75f && a <= 1.25f) r = 1.0f;
  if (a > 0.25f && a < 0.75f) r = 0.5f;
  return r * sign;
}

__global__ void qdq_quantize_q8_1_kernel(const __hip_bfloat16* __restrict__ x,
                                         int* __restrict__ y, int M, long K,
                                         int y_blocks, float global_scale) {
  const long row = blockIdx.y;
  const long blk = (long)blockIdx.x * blockDim.y + threadIdx.y;
  if (blk >= y_blocks) return;
  const int lane = threadIdx.x;  // 32 lanes; group = lane / 16

  const long k = blk * 32 + lane;
  float v = 0.0f;
  if (k < K) v = __bfloat162float(x[row * K + k]);

  // --- NVFP4 QDQ per group of 16 ---
  float gmax = fabsf(v);
#pragma unroll
  for (int off = 8; off > 0; off >>= 1)
    gmax = fmaxf(gmax, __shfl_xor(gmax, off, 32));  // reduces within 16
  float scale = fminf(global_scale * (gmax * (1.0f / 6.0f)), 448.0f);
  scale = e4m3fn_round_pos(scale);
  const float out_scale = scale == 0.0f ? 0.0f : global_scale / scale;
  const float fp4 = round_to_fp4(fminf(fmaxf(v * out_scale, -6.0f), 6.0f));
  // The unfused path materializes the QDQ result in bf16 before the Q8
  // step; reproduce that rounding so both paths are bit-identical.
  const float dq =
      __bfloat162float(__float2bfloat16(fp4 * (scale / global_scale)));

  // --- Q8_1 over the 32-value block ---
  float amax = fabsf(dq);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, off, 32));
  const float d = amax / 127.0f;
  const float id = d > 0.0f ? 1.0f / d : 0.0f;
  const int q = (int)rintf(dq * id);

  int sum = q;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);

  int* out = y + (row * y_blocks + blk) * kQ8BlockInts;
  const int q0 = q & 0xFF;
  const int q1 = __shfl_down(q, 1, 32) & 0xFF;
  const int q2 = __shfl_down(q, 2, 32) & 0xFF;
  const int q3 = __shfl_down(q, 3, 32) & 0xFF;
  if (lane % 4 == 0)
    out[1 + lane / 4] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (lane == 0) {
    const __half2 ds =
        __halves2half2(__float2half(d), __float2half(d * (float)sum));
    __builtin_memcpy(out, &ds, 4);
  }
}

// ------------------------------------------------------- Q8_1 activation quant
// block_q8_1 = {half d; half s; int8 qs[32]}: d = amax/127, s = d * sum(qs).
// One wavefront per (row, block-span); trivially bandwidth-bound.
__global__ void quantize_q8_1_kernel(const __hip_bfloat16* __restrict__ x,
                                     int* __restrict__ y, int M, long K,
                                     int y_blocks) {
  const long row = blockIdx.y;
  const long blk = (long)blockIdx.x * blockDim.y + threadIdx.y;
  if (blk >= y_blocks) return;
  const int lane = threadIdx.x;  // 32 lanes, one value each

  const long k = blk * 32 + lane;
  float v = 0.0f;
  if (k < K) v = __bfloat162float(x[row * K + k]);

  float amax = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, off, 32));
  const float d = amax / 127.0f;
  const float id = d > 0.0f ? 1.0f / d : 0.0f;
  const int q = (int)rintf(v * id);

  int sum = q;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);

  int* out = y + (row * y_blocks + blk) * kQ8BlockInts;
  // Pack four int8 per lane group via ballot-free shuffles: lane 0 of each
  // 4-lane group assembles its int.
  const int q0 = q & 0xFF;
  const int q1 = __shfl_down(q, 1, 32) & 0xFF;
  const int q2 = __shfl_down(q, 2, 32) & 0xFF;
  const int q3 = __shfl_down(q, 3, 32) & 0xFF;
  if (lane % 4 == 0)
    out[1 + lane / 4] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (lane == 0) {
    const __half2 ds =
        __halves2half2(__float2half(d), __float2half(d * (float)sum));
    __builtin_memcpy(out, &ds, 4);
  }
}

}  // namespace qcnvfp4

static torch::Tensor py_nvfp4_quantize_q8_1(torch::Tensor x) {
  TORCH_CHECK(
      x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16,
      "x must be contiguous CUDA bf16");
  const int M = x.size(0);
  const long K = x.size(1);
  const int y_blocks = (int)((K + 31) / 32);
  auto y = torch::empty(
      {M, (long)y_blocks * qcnvfp4::kQ8BlockInts},
      torch::TensorOptions().dtype(torch::kInt).device(x.device()));
  if (M == 0) return y;
  constexpr int kBlocksPerCta = 8;
  const dim3 block(32, kBlocksPerCta);
  const dim3 grid((y_blocks + kBlocksPerCta - 1) / kBlocksPerCta, M);
  auto stream = at::cuda::getCurrentCUDAStream();
  qcnvfp4::quantize_q8_1_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
      reinterpret_cast<int*>(y.data_ptr()), M, K, y_blocks);
  return y;
}

static torch::Tensor py_nvfp4_qdq_quantize_q8_1(torch::Tensor x,
                                                double global_scale) {
  TORCH_CHECK(
      x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16,
      "x must be contiguous CUDA bf16");
  const int M = x.size(0);
  const long K = x.size(1);
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
  const int y_blocks = (int)(K / 32);
  auto y = torch::empty(
      {M, (long)y_blocks * qcnvfp4::kQ8BlockInts},
      torch::TensorOptions().dtype(torch::kInt).device(x.device()));
  if (M == 0) return y;
  constexpr int kBlocksPerCta = 8;
  const dim3 block(32, kBlocksPerCta);
  const dim3 grid((y_blocks + kBlocksPerCta - 1) / kBlocksPerCta, M);
  auto stream = at::cuda::getCurrentCUDAStream();
  qcnvfp4::qdq_quantize_q8_1_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
      reinterpret_cast<int*>(y.data_ptr()), M, K, y_blocks,
      (float)global_scale);
  return y;
}

static void py_nvfp4_gemv_q8(torch::Tensor y_q8, torch::Tensor w,
                             torch::Tensor s, double global_scale,
                             torch::Tensor out_bf16, int64_t m_tokens,
                             int64_t k_cols) {
  TORCH_CHECK(y_q8.is_cuda() && y_q8.is_contiguous() &&
                  y_q8.scalar_type() == torch::kInt,
              "y_q8 must be contiguous CUDA int32");
  TORCH_CHECK(
      w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kUInt8,
      "w must be contiguous CUDA uint8");
  TORCH_CHECK(
      s.is_cuda() && s.is_contiguous() && s.scalar_type() == torch::kHalf,
      "s must be contiguous CUDA fp16");
  TORCH_CHECK(out_bf16.is_cuda() && out_bf16.is_contiguous() &&
                  out_bf16.scalar_type() == torch::kBFloat16,
              "out must be contiguous CUDA bf16");
  const int M = (int)m_tokens;
  const long K = (long)k_cols;
  const int N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "GEMV path is for 1..8 tokens");
  TORCH_CHECK(w.size(1) * 2 == K, "w K mismatch");
  TORCH_CHECK(s.size(0) == N && s.size(1) * 16 == K, "scale shape mismatch");
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(y_q8.size(0) == M, "y_q8 rows");
  TORCH_CHECK(y_q8.size(1) % qcnvfp4::kQ8BlockInts == 0, "y_q8 layout");
  const int y_blocks = (int)(y_q8.size(1) / qcnvfp4::kQ8BlockInts);
  TORCH_CHECK((long)y_blocks * 32 >= K, "y_q8 too narrow for K");
  TORCH_CHECK(out_bf16.size(0) == M && out_bf16.size(1) == N, "out shape");

  auto stream = at::cuda::getCurrentCUDAStream();
  const float half_global = 0.5f * (float)global_scale;
  constexpr int kRowsPerBlock = 4;
  const dim3 block(qcnvfp4::kWave * kRowsPerBlock);
  const dim3 grid((N + kRowsPerBlock - 1) / kRowsPerBlock, 1);
  qcnvfp4::nvfp4_gemv_q8_kernel<8><<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      reinterpret_cast<const __half*>(s.data_ptr()),
      reinterpret_cast<const int*>(y_q8.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(out_bf16.data_ptr()), M, N, K, y_blocks,
      half_global);
}

static void py_nvfp4_gemm_q8(torch::Tensor y_q8, torch::Tensor w,
                             torch::Tensor s, double global_scale,
                             torch::Tensor out_f32, int64_t m_tokens,
                             int64_t k_cols) {
  TORCH_CHECK(y_q8.is_cuda() && y_q8.is_contiguous() &&
                  y_q8.scalar_type() == torch::kInt,
              "y_q8 must be contiguous CUDA int32 (ggml_quantize_q8_1)");
  TORCH_CHECK(
      w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kUInt8,
      "w must be contiguous CUDA uint8");
  TORCH_CHECK(
      s.is_cuda() && s.is_contiguous() && s.scalar_type() == torch::kHalf,
      "s must be contiguous CUDA fp16");
  TORCH_CHECK(out_f32.is_cuda() && out_f32.is_contiguous() &&
                  out_f32.scalar_type() == torch::kFloat,
              "out must be contiguous CUDA f32");
  const int M = (int)m_tokens;
  const long K = (long)k_cols;
  const int N = w.size(0);
  TORCH_CHECK(w.size(1) * 2 == K, "w K mismatch");
  TORCH_CHECK(s.size(0) == N && s.size(1) * 16 == K, "scale shape mismatch");
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(y_q8.size(0) == M, "y_q8 rows");
  TORCH_CHECK(y_q8.size(1) % qcnvfp4::kQ8BlockInts == 0, "y_q8 layout");
  const int y_blocks = (int)(y_q8.size(1) / qcnvfp4::kQ8BlockInts);
  TORCH_CHECK((long)y_blocks * 32 >= K, "y_q8 too narrow for K");
  TORCH_CHECK(out_f32.size(0) == M && out_f32.size(1) == N, "out shape");
  if (M == 0) return;
  TORCH_CHECK(M > 8, "use nvfp4_gemv_q8 for 1..8 tokens");

  auto stream = at::cuda::getCurrentCUDAStream();
  const float half_global = 0.5f * (float)global_scale;

  constexpr int MMQ_X = 32, MMQ_Y = 64, NWARPS = 8;
  const dim3 block(MMQ_WARP, NWARPS);
  const dim3 grid((N + MMQ_Y - 1) / MMQ_Y, (M + MMQ_X - 1) / MMQ_X);
  const bool need_check = (N % MMQ_Y != 0) || (M % MMQ_X != 0);
  if (need_check) {
    qcnvfp4::nvfp4_mmq_q8_kernel<MMQ_X, MMQ_Y, NWARPS, true>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w.data_ptr()),
            reinterpret_cast<const __half*>(s.data_ptr()),
            reinterpret_cast<const int*>(y_q8.data_ptr()),
            reinterpret_cast<float*>(out_f32.data_ptr()), M, N, K, y_blocks,
            half_global);
  } else {
    qcnvfp4::nvfp4_mmq_q8_kernel<MMQ_X, MMQ_Y, NWARPS, false>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w.data_ptr()),
            reinterpret_cast<const __half*>(s.data_ptr()),
            reinterpret_cast<const int*>(y_q8.data_ptr()),
            reinterpret_cast<float*>(out_f32.data_ptr()), M, N, K, y_blocks,
            half_global);
  }
}

void init_nvfp4(py::module_& m) {
  m.def("nvfp4_quantize_q8_1", &py_nvfp4_quantize_q8_1, py::arg("x"),
        "Quantize bf16 activation rows to GGUF block_q8_1 (int32 view).");
  m.def("nvfp4_qdq_quantize_q8_1", &py_nvfp4_qdq_quantize_q8_1, py::arg("x"),
        py::arg("global_scale"),
        "Fused NVFP4 activation QDQ (w4a4 semantics) + block_q8_1 quant.");
  m.def("nvfp4_gemv_q8", &py_nvfp4_gemv_q8, py::arg("y_q8"), py::arg("w"),
        py::arg("s"), py::arg("global_scale"), py::arg("out_bf16"),
        py::arg("m_tokens"), py::arg("k_cols"),
        "Packed-E2M1 NVFP4 GEMV (1..8 tokens), bf16 output.");
  m.def("nvfp4_gemm_q8", &py_nvfp4_gemm_q8, py::arg("y_q8"), py::arg("w"),
        py::arg("s"), py::arg("global_scale"), py::arg("out_f32"),
        py::arg("m_tokens"), py::arg("k_cols"),
        "Native NVFP4 GEMM: q8_1 activations x planar-packed E2M1 weights "
        "with fp16 block scales; f32 output. GEMV below 9 tokens, LDS-tiled "
        "MMQ above.");
}
