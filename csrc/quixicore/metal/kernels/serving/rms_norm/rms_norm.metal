#include <metal_stdlib>

// Weighted RMS norm for the serving decode path: one threadgroup per token
// row, fp32 square-sum reduction, then y = T(float(x) * rrms) * w with the
// final multiply in the weight dtype — mirroring vllm/ir/ops/layernorm.py
// rms_norm op for op (fp32 statistic, `.to(weight.dtype) * weight`, cast to
// the input dtype). Parity holds to reduction-order ulps only, same protocol
// as the fused mHC kernels.
//
// The eager MPS decomposition of this op is 6+ dispatches (cast, sqr,
// sum, rsqrt, two muls), and the serving path hits it hundreds of times
// per engine step.

using namespace metal;

namespace qc_rms {
constant constexpr int THREADS = 256;
constant constexpr int SIMDGROUPS = THREADS / 32;
}  // namespace qc_rms

// Contract: dispatch exactly THREADS (256) threads per threadgroup — the
// SlimServe qc_metal_serving.mm site does. The cross-simdgroup reduction sums all
// SIMDGROUPS shm slots unconditionally, and each slot is written only by a
// simdgroup that actually runs; a smaller dispatch would read uninitialized
// threadgroup memory. (With 256 threads, small D just contributes zeros.)

template <typename T>
inline void rms_norm_body(device const T* x, device const T* w, device T* y,
                          uint D, float eps, uint token, uint tid, uint sg,
                          uint lane, threadgroup float* shm,
                          ulong in_stride) {
  device const T* xt = x + (ulong)token * in_stride;
  device T* yt = y + (ulong)token * D;

  float sq = 0.0f;
  for (uint d = tid; d < D; d += qc_rms::THREADS) {
    const float v = float(xt[d]);
    sq += v * v;
  }
  sq = simd_sum(sq);
  if (lane == 0) shm[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < qc_rms::SIMDGROUPS; ++i) total += shm[i];
    shm[0] = rsqrt(total / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float rrms = shm[0];

  for (uint d = tid; d < D; d += qc_rms::THREADS) {
    const T scaled = T(float(xt[d]) * rrms);
    yt[d] = scaled * w[d];
  }
}

// fp32-weight variant (GGUF stores norm weights as F32): the reference then
// keeps the weight multiply in fp32 — y = T(float(x) * rrms * w).
template <typename T>
inline void rms_norm_w32_body(device const T* x, device const float* w,
                              device T* y, uint D, float eps, uint token,
                              uint tid, uint sg, uint lane,
                              threadgroup float* shm, ulong in_stride) {
  device const T* xt = x + (ulong)token * in_stride;
  device T* yt = y + (ulong)token * D;

  float sq = 0.0f;
  for (uint d = tid; d < D; d += qc_rms::THREADS) {
    const float v = float(xt[d]);
    sq += v * v;
  }
  sq = simd_sum(sq);
  if (lane == 0) shm[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < qc_rms::SIMDGROUPS; ++i) total += shm[i];
    shm[0] = rsqrt(total / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float rrms = shm[0];

  for (uint d = tid; d < D; d += qc_rms::THREADS) {
    yt[d] = T(float(xt[d]) * rrms * w[d]);
  }
}

#define instantiate_rms_norm(tname, T)                                       \
  [[host_name("qc_rms_norm_" #tname)]] kernel void qc_rms_norm_##tname(      \
      device const T* x [[buffer(0)]], device const T* w [[buffer(1)]],      \
      device T* y [[buffer(2)]], constant uint& D [[buffer(3)]],             \
      constant float& eps [[buffer(4)]],                                     \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float shm[qc_rms::SIMDGROUPS];                               \
    rms_norm_body<T>(x, w, y, D, eps, tgid.x, tid, sg, lane, shm, D);        \
  }                                                                          \
  [[host_name("qc_rms_norm_w32_" #tname)]] kernel void                       \
  qc_rms_norm_w32_##tname(                                                   \
      device const T* x [[buffer(0)]], device const float* w [[buffer(1)]],  \
      device T* y [[buffer(2)]], constant uint& D [[buffer(3)]],             \
      constant float& eps [[buffer(4)]],                                     \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float shm[qc_rms::SIMDGROUPS];                               \
    rms_norm_w32_body<T>(x, w, y, D, eps, tgid.x, tid, sg, lane, shm, D);    \
  }

// Row-strided input variants (in_stride in elements, unit inner stride;
// output is packed). Lets fused-GEMM split halves bind without the eager
// .contiguous() copy — identical element values, bit-exact.
#define instantiate_rms_norm_strided(tname, T)                               \
  [[host_name("qc_rms_norm_strided_" #tname)]] kernel void                   \
  qc_rms_norm_strided_##tname(                                               \
      device const T* x [[buffer(0)]], device const T* w [[buffer(1)]],      \
      device T* y [[buffer(2)]], constant uint& D [[buffer(3)]],             \
      constant float& eps [[buffer(4)]],                                     \
      constant ulong& in_stride [[buffer(5)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float shm[qc_rms::SIMDGROUPS];                               \
    rms_norm_body<T>(x, w, y, D, eps, tgid.x, tid, sg, lane, shm,            \
                     in_stride);                                             \
  }                                                                          \
  [[host_name("qc_rms_norm_w32_strided_" #tname)]] kernel void               \
  qc_rms_norm_w32_strided_##tname(                                           \
      device const T* x [[buffer(0)]], device const float* w [[buffer(1)]],  \
      device T* y [[buffer(2)]], constant uint& D [[buffer(3)]],             \
      constant float& eps [[buffer(4)]],                                     \
      constant ulong& in_stride [[buffer(5)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float shm[qc_rms::SIMDGROUPS];                               \
    rms_norm_w32_body<T>(x, w, y, D, eps, tgid.x, tid, sg, lane, shm,        \
                         in_stride);                                         \
  }

// Two packed segments of one row-strided input (the MLA fused q_a|kv_a
// projection: q_c = cols [0, d0), kv_c = cols [d0, d0 + d1)), each with its
// own fp32 weight and packed output, in ONE dispatch: grid (tokens, 2).
// Per segment it is exactly qc_rms_norm_w32_strided_* - bit-exact.
#define instantiate_rms_norm_dual(tname, T)                                  \
  [[host_name("qc_rms_norm_w32_dual_" #tname)]] kernel void                  \
  qc_rms_norm_w32_dual_##tname(                                              \
      device const T* x [[buffer(0)]], device const float* w0 [[buffer(1)]], \
      device const float* w1 [[buffer(2)]], device T* y0 [[buffer(3)]],     \
      device T* y1 [[buffer(4)]], constant uint& d0 [[buffer(5)]],           \
      constant uint& d1 [[buffer(6)]], constant float& eps [[buffer(7)]],    \
      constant ulong& in_stride [[buffer(8)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float shm[qc_rms::SIMDGROUPS];                               \
    if (tgid.y == 0)                                                         \
      rms_norm_w32_body<T>(x, w0, y0, d0, eps, tgid.x, tid, sg, lane, shm,   \
                           in_stride);                                       \
    else                                                                     \
      rms_norm_w32_body<T>(x + d0, w1, y1, d1, eps, tgid.x, tid, sg, lane,   \
                           shm, in_stride);                                  \
  }

instantiate_rms_norm(float16, half);
instantiate_rms_norm(bfloat16, bfloat);
instantiate_rms_norm_strided(float16, half);
instantiate_rms_norm_strided(bfloat16, bfloat);
instantiate_rms_norm_dual(float16, half);
instantiate_rms_norm_dual(bfloat16, bfloat);
