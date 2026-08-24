#include <metal_stdlib>

// DeepSeek-V4 mHC (hyper-connections) for Apple Metal.
//
// Port of csrc/quixicore/serving/mhc_ampere.cuh to the decode/verify regime
// this box actually serves (tokens <= ~32). The A100 kernel splits each token
// across 32 blocks and grid-syncs; here one threadgroup owns a token
// end-to-end, so the whole pre block -- 24-way projection, RMS statistic,
// gates, softmax and the full Sinkhorn iteration -- is a single dispatch
// instead of the ~230 aten launches the torch decomposition costs per call.
//
// Numerics follow the torch reference (kernels/mhc/torch.py) operation for
// operation, including the round-to-activation-dtype of the mixed residual
// between post and pre: parity against mhc_post_torch + mhc_pre_torch holds
// to reduction-order ulps only.

using namespace metal;

namespace dsv4_mhc {

constant constexpr int HC = 4;
constant constexpr int MIXES = 24;          // 2*HC + HC*HC
constant constexpr int THREADS = 256;       // 8 simdgroups
constant constexpr int SIMDGROUPS = THREADS / 32;

// threadgroup scratch layout (floats)
constant constexpr int SHM_PARTIAL = 0;     // [MIXES + 1] dot totals + sqsum
constant constexpr int SHM_PRE = MIXES + 1; // [HC] pre-mix gates
constant constexpr int SHM_COEFF = SHM_PRE + HC; // [HC + HC*HC] prev post+comb
constant constexpr int SHM_SIZE = SHM_COEFF + HC + HC * HC;

inline float sigmoid_f(float v) { return 1.0f / (1.0f + exp(-v)); }

// Row/column normalizations of the 4x4 matrix held one element per lane on
// lanes 0..15. Rows are lane-aligned quads, so XOR 1/2 stays inside a row and
// XOR 4/8 walks a column; lanes 16..31 shuffle among themselves and never
// contaminate the live lanes.
inline float row_reduce_sum(float v) {
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  return v;
}
inline float col_reduce_sum(float v) {
  v += simd_shuffle_xor(v, 4);
  v += simd_shuffle_xor(v, 8);
  return v;
}
inline float row_reduce_max(float v) {
  v = max(v, simd_shuffle_xor(v, 1));
  v = max(v, simd_shuffle_xor(v, 2));
  return v;
}

template <typename T>
inline void mhc_pre_body(
    device const T* x,            // [tokens, H]
    device const T* residual,     // [tokens, HC, H]
    device const float* post_mix, // [tokens, HC]
    device const float* comb_mix, // [tokens, HC*HC]
    device const float* fn,       // [MIXES, HC*H]
    device const float* scale,    // [3]
    device const float* base,     // [MIXES]
    device T* residual_out,       // [tokens, HC, H]
    device float* next_post,      // [tokens, HC]
    device float* next_comb,      // [tokens, HC*HC]
    device T* layer_input,        // [tokens, H]
    uint H, float rms_eps, float pre_eps, float sink_eps, float post_mult,
    int sinkhorn_repeat, uint token, uint tid, uint sg, uint lane,
    threadgroup float* shm) {
  const uint total = HC * H;
  device const T* res_t = residual + token * total;
  device T* res_out_t = residual_out + token * total;
  device const T* x_t = x + token * H;

  if (tid < HC) shm[SHM_COEFF + tid] = post_mix[token * HC + tid];
  if (tid < HC * HC)
    shm[SHM_COEFF + HC + tid] = comb_mix[token * HC * HC + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each simdgroup owns projection rows {sg, sg+8, sg+16}; simdgroup 0 also
  // owns the square-sum and the residual_out store (every simdgroup computes
  // the identical rounded value, so a single writer suffices).
  const uint o0 = sg, o1 = sg + SIMDGROUPS, o2 = sg + 2 * SIMDGROUPS;
  device const float* fn0 = fn + o0 * total;
  device const float* fn1 = fn + o1 * total;
  device const float* fn2 = fn + o2 * total;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, sq = 0.0f;
  for (uint stream = 0; stream < HC; ++stream) {
    const uint off = stream * H;
    for (uint d = lane; d < H; d += 32) {
      const uint flat = off + d;
      float v = shm[SHM_COEFF + stream] * float(x_t[d]);
      for (uint i = 0; i < HC; ++i) {
        v += shm[SHM_COEFF + HC + i * HC + stream] *
             float(res_t[i * H + d]);
      }
      const T rounded = T(v);
      if (sg == 0) res_out_t[flat] = rounded;
      const float value = float(rounded);
      if (sg == 0) sq += value * value;
      acc0 += value * fn0[flat];
      acc1 += value * fn1[flat];
      acc2 += value * fn2[flat];
    }
  }

  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  acc2 = simd_sum(acc2);
  if (sg == 0) sq = simd_sum(sq);
  if (lane == 0) {
    shm[SHM_PARTIAL + o0] = acc0;
    shm[SHM_PARTIAL + o1] = acc1;
    shm[SHM_PARTIAL + o2] = acc2;
    if (sg == 0) shm[SHM_PARTIAL + MIXES] = sq;
  }
  // Device fence: phase 3 (and other simdgroups) re-read residual_out.
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  if (sg == 0) {
    const float inv_rms =
        rsqrt(shm[SHM_PARTIAL + MIXES] / float(total) + rms_eps);
    const float mix =
        (lane < MIXES) ? shm[SHM_PARTIAL + lane] * inv_rms : 0.0f;
    if (lane < HC) {
      shm[SHM_PRE + lane] =
          sigmoid_f(mix * scale[0] + base[lane]) + pre_eps;
    } else if (lane < 2 * HC) {
      next_post[token * HC + lane - HC] =
          sigmoid_f(mix * scale[1] + base[lane]) * post_mult;
    }

    // Sinkhorn on lanes 0..15; matches torch: softmax(dim=-1) + eps, one
    // column normalize, then (repeat-1) x (row, column) normalizes.
    float logit = (lane < HC * HC)
                      ? shm[SHM_PARTIAL + 2 * HC + lane] * inv_rms * scale[2] +
                            base[2 * HC + lane]
                      : 0.0f;
    const float row_max = row_reduce_max(logit);
    float m = exp(logit - row_max);
    m = m / row_reduce_sum(m) + sink_eps;
    for (int it = 0; it < sinkhorn_repeat; ++it) {
      if (it > 0) m = m / (row_reduce_sum(m) + sink_eps);
      m = m / (col_reduce_sum(m) + sink_eps);
    }
    if (lane < HC * HC) next_comb[token * HC * HC + lane] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device const T* src = res_out_t;
  const float p0 = shm[SHM_PRE + 0], p1 = shm[SHM_PRE + 1];
  const float p2 = shm[SHM_PRE + 2], p3 = shm[SHM_PRE + 3];
  for (uint d = tid; d < H; d += THREADS) {
    const float v = p0 * float(src[d]) + p1 * float(src[H + d]) +
                    p2 * float(src[2 * H + d]) + p3 * float(src[3 * H + d]);
    layer_input[token * H + d] = T(v);
  }
}

#define ARGS_FUSED(T)                                                        \
  device const T* x [[buffer(0)]],                                           \
      device const T* residual [[buffer(1)]],                                \
      device const float* post_mix [[buffer(2)]],                            \
      device const float* comb_mix [[buffer(3)]],                            \
      device const float* fn [[buffer(4)]],                                  \
      device const float* scale [[buffer(5)]],                               \
      device const float* base [[buffer(6)]],                                \
      device T* residual_out [[buffer(7)]],                                  \
      device float* next_post [[buffer(8)]],                                 \
      device float* next_comb [[buffer(9)]],                                 \
      device T* layer_input [[buffer(10)]],                                  \
      constant uint& H [[buffer(11)]],                                       \
      constant float& rms_eps [[buffer(12)]],                                \
      constant float& pre_eps [[buffer(13)]],                                \
      constant float& sink_eps [[buffer(14)]],                               \
      constant float& post_mult [[buffer(15)]],                              \
      constant int& sinkhorn_repeat [[buffer(16)]],                          \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]

#define instantiate_mhc_fused(tname, T)                                      \
  [[host_name("dsv4_mhc_fused_post_pre_" #tname)]] kernel void               \
  dsv4_mhc_fused_post_pre_##tname(ARGS_FUSED(T)) {                           \
    threadgroup float shm[dsv4_mhc::SHM_SIZE];                               \
    dsv4_mhc::mhc_pre_body<T>(                                               \
        x, residual, post_mix, comb_mix, fn, scale, base, residual_out,      \
        next_post, next_comb, layer_input, H, rms_eps, pre_eps, sink_eps,    \
        post_mult, sinkhorn_repeat, tgid.x, tid, sg, lane, shm);             \
  }

// ---- split pre: dots + finalize --------------------------------------------
// The pre block runs as (a) a dots pass with one simdgroup per
// (token, row|sqsum) job -- 25*tokens threadgroups, which keeps a wide GPU
// occupied at decode token counts -- and (b) a small finalize pass that is
// mhc_pre_body's phase 2+3 with the threadgroup partials read from the
// scratch buffer instead. Both passes keep the per-lane stride-32
// accumulation order and the simd_sum tree of the fused body, so their
// outputs are bit-identical to it.

template <typename T>
inline void mhc_pre_dots_body(device const T* residual, device const float* fn,
                              device float* scratch, uint H, uint token,
                              uint job, uint lane) {
  const uint total = HC * H;
  device const T* res_t = residual + token * total;
  float acc = 0.0f;
  // Too few simdgroups exist at decode (25 jobs x tokens) for per-core
  // residency to hide the load->fma latency; batch loads eight strides
  // ahead instead. The fma chain itself stays strictly sequential,
  // keeping the reduction bit-identical.
  constexpr uint U = 8;
  if (job < MIXES) {
    device const float* fn_j = fn + job * total;
    for (uint stream = 0; stream < HC; ++stream) {
      const uint off = stream * H;
      uint d = lane;
      for (; d + (U - 1) * 32 < H; d += U * 32) {
        float rv[U];
        float fv[U];
#pragma unroll
        for (uint u = 0; u < U; ++u) {
          rv[u] = float(res_t[off + d + u * 32]);
          fv[u] = fn_j[off + d + u * 32];
        }
#pragma unroll
        for (uint u = 0; u < U; ++u) {
          acc += rv[u] * fv[u];
        }
      }
      for (; d < H; d += 32) {
        acc += float(res_t[off + d]) * fn_j[off + d];
      }
    }
  } else {
    for (uint stream = 0; stream < HC; ++stream) {
      const uint off = stream * H;
      uint d = lane;
      for (; d + (U - 1) * 32 < H; d += U * 32) {
        float rv[U];
#pragma unroll
        for (uint u = 0; u < U; ++u) {
          rv[u] = float(res_t[off + d + u * 32]);
        }
#pragma unroll
        for (uint u = 0; u < U; ++u) {
          acc += rv[u] * rv[u];
        }
      }
      for (; d < H; d += 32) {
        const float v = float(res_t[off + d]);
        acc += v * v;
      }
    }
  }
  acc = simd_sum(acc);
  if (lane == 0) scratch[token * (MIXES + 1) + job] = acc;
}

// Prefill-width dots: one threadgroup per token. The simdgroup-per-job
// dots kernel re-reads the token's [HC, H] residual from device once per
// (row|sqsum) job — 25x redundant traffic at chunk widths. Stage the
// residual through threadgroup memory one H-stream at a time (8 KB) and
// give each simdgroup the row set {sg, sg+8, sg+16} (+ sqsum on simdgroup
// 0). Per-lane fma order (stream-major, d stride 32) and the simd_sum
// tree match the simdgroup-per-job pass, so scratch is bit-identical to
// it. Decode widths keep that pass: at tokens <= ~32 this shape has too
// few threadgroups.
template <typename T>
inline void mhc_pre_dots_tg_body(device const T* residual,
                                 device const float* fn, device float* scratch,
                                 uint H, uint token, uint tid, uint sg,
                                 uint lane, threadgroup T* sres) {
  const uint total = HC * H;
  device const T* res_t = residual + token * total;
  const uint o0 = sg, o1 = sg + SIMDGROUPS, o2 = sg + 2 * SIMDGROUPS;
  device const float* fn0 = fn + o0 * total;
  device const float* fn1 = fn + o1 * total;
  device const float* fn2 = fn + o2 * total;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, sq = 0.0f;
  for (uint stream = 0; stream < HC; ++stream) {
    const uint off = stream * H;
    for (uint i = tid; i < H; i += THREADS) {
      sres[i] = res_t[off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < H; d += 32) {
      const float value = float(sres[d]);
      if (sg == 0) sq += value * value;
      acc0 += value * fn0[off + d];
      acc1 += value * fn1[off + d];
      acc2 += value * fn2[off + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  acc2 = simd_sum(acc2);
  if (sg == 0) sq = simd_sum(sq);
  if (lane == 0) {
    device float* sc_t = scratch + token * (MIXES + 1);
    sc_t[o0] = acc0;
    sc_t[o1] = acc1;
    sc_t[o2] = acc2;
    if (sg == 0) sc_t[MIXES] = sq;
  }
}

template <typename T>
inline void mhc_pre_finalize_body(
    device const T* residual, device const float* scratch,
    device const float* scale, device const float* base,
    device float* next_post, device float* next_comb, device T* layer_input,
    uint H, float rms_eps, float pre_eps, float sink_eps, float post_mult,
    int sinkhorn_repeat, uint token, uint tid, uint sg, uint lane,
    threadgroup float* shm) {
  const uint total = HC * H;
  device const T* res_t = residual + token * total;
  device const float* sc_t = scratch + token * (MIXES + 1);

  if (sg == 0) {
    const float inv_rms = rsqrt(sc_t[MIXES] / float(total) + rms_eps);
    const float mix = (lane < MIXES) ? sc_t[lane] * inv_rms : 0.0f;
    if (lane < HC) {
      shm[lane] = sigmoid_f(mix * scale[0] + base[lane]) + pre_eps;
    } else if (lane < 2 * HC) {
      next_post[token * HC + lane - HC] =
          sigmoid_f(mix * scale[1] + base[lane]) * post_mult;
    }
    float logit = (lane < HC * HC)
                      ? sc_t[2 * HC + lane] * inv_rms * scale[2] +
                            base[2 * HC + lane]
                      : 0.0f;
    const float row_max = row_reduce_max(logit);
    float m = exp(logit - row_max);
    m = m / row_reduce_sum(m) + sink_eps;
    for (int it = 0; it < sinkhorn_repeat; ++it) {
      if (it > 0) m = m / (row_reduce_sum(m) + sink_eps);
      m = m / (col_reduce_sum(m) + sink_eps);
    }
    if (lane < HC * HC) next_comb[token * HC * HC + lane] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float p0 = shm[0], p1 = shm[1], p2 = shm[2], p3 = shm[3];
  for (uint d = tid; d < H; d += THREADS) {
    const float v = p0 * float(res_t[d]) + p1 * float(res_t[H + d]) +
                    p2 * float(res_t[2 * H + d]) + p3 * float(res_t[3 * H + d]);
    layer_input[token * H + d] = T(v);
  }
}

#define instantiate_mhc_pre_dots(tname, T)                                    \
  [[host_name("dsv4_mhc_pre_dots_" #tname)]] kernel void                      \
  dsv4_mhc_pre_dots_##tname(                                                  \
      device const T* residual [[buffer(0)]],                                 \
      device const float* fn [[buffer(1)]],                                   \
      device float* scratch [[buffer(2)]],                                    \
      constant uint& H [[buffer(3)]],                                         \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    dsv4_mhc::mhc_pre_dots_body<T>(residual, fn, scratch, H, tgid.x, tgid.y,  \
                                   lane);                                     \
  }

#define instantiate_mhc_pre_dots_tg(tname, T)                                 \
  [[host_name("dsv4_mhc_pre_dots_tg_" #tname)]] kernel void                   \
  dsv4_mhc_pre_dots_tg_##tname(                                               \
      device const T* residual [[buffer(0)]],                                 \
      device const float* fn [[buffer(1)]],                                   \
      device float* scratch [[buffer(2)]],                                    \
      constant uint& H [[buffer(3)]],                                         \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]],                               \
      uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    threadgroup T sres[4096]; /* H <= 4096 host-checked */                    \
    dsv4_mhc::mhc_pre_dots_tg_body<T>(residual, fn, scratch, H, tgid.x, tid,  \
                                      sg, lane, sres);                        \
  }

#define instantiate_mhc_pre_finalize(tname, T)                                \
  [[host_name("dsv4_mhc_pre_finalize_" #tname)]] kernel void                  \
  dsv4_mhc_pre_finalize_##tname(                                              \
      device const T* residual [[buffer(0)]],                                 \
      device const float* scratch [[buffer(1)]],                              \
      device const float* scale [[buffer(2)]],                                \
      device const float* base [[buffer(3)]],                                 \
      device float* next_post [[buffer(4)]],                                  \
      device float* next_comb [[buffer(5)]],                                  \
      device T* layer_input [[buffer(6)]],                                    \
      constant uint& H [[buffer(7)]],                                         \
      constant float& rms_eps [[buffer(8)]],                                  \
      constant float& pre_eps [[buffer(9)]],                                  \
      constant float& sink_eps [[buffer(10)]],                                \
      constant float& post_mult [[buffer(11)]],                               \
      constant int& sinkhorn_repeat [[buffer(12)]],                           \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]],                               \
      uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    threadgroup float shm[dsv4_mhc::HC];                                      \
    dsv4_mhc::mhc_pre_finalize_body<T>(residual, scratch, scale, base,        \
                                       next_post, next_comb, layer_input, H,  \
                                       rms_eps, pre_eps, sink_eps, post_mult, \
                                       sinkhorn_repeat, tgid.x, tid, sg,      \
                                       lane, shm);                            \
  }

// ---- standalone post -------------------------------------------------------
// out[t,j,:] = post[t,j] * x[t,:] + sum_i comb[t,i,j] * residual[t,i,:]

template <typename T>
inline void mhc_post_body(device const T* x, device const T* residual,
                          device const float* post_mix,
                          device const float* comb_mix, device T* out, uint H,
                          uint token, uint slice, uint nslice, uint tid,
                          threadgroup float* coeff) {
  if (tid < HC) coeff[tid] = post_mix[token * HC + tid];
  if (tid < HC * HC) coeff[HC + tid] = comb_mix[token * HC * HC + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint total = HC * H;
  device const T* res_t = residual + token * total;
  device const T* x_t = x + token * H;
  device T* out_t = out + token * total;
  // Purely elementwise, so slicing the H walk across tgid.y threadgroups is
  // bit-exact; the extra grid width keeps decode token counts from
  // under-filling the GPU.
  for (uint d = slice * THREADS + tid; d < H; d += nslice * THREADS) {
    const float xv = float(x_t[d]);
    const float r0 = float(res_t[d]);
    const float r1 = float(res_t[H + d]);
    const float r2 = float(res_t[2 * H + d]);
    const float r3 = float(res_t[3 * H + d]);
    for (uint s = 0; s < HC; ++s) {
      const float v = coeff[s] * xv + coeff[HC + 0 * HC + s] * r0 +
                      coeff[HC + 1 * HC + s] * r1 +
                      coeff[HC + 2 * HC + s] * r2 +
                      coeff[HC + 3 * HC + s] * r3;
      out_t[s * H + d] = T(v);
    }
  }
}

#define instantiate_mhc_post(tname, T)                                       \
  [[host_name("dsv4_mhc_post_" #tname)]] kernel void dsv4_mhc_post_##tname(  \
      device const T* x [[buffer(0)]],                                       \
      device const T* residual [[buffer(1)]],                                \
      device const float* post_mix [[buffer(2)]],                            \
      device const float* comb_mix [[buffer(3)]],                            \
      device T* out [[buffer(4)]], constant uint& H [[buffer(5)]],           \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint3 tpg [[threadgroups_per_grid]],                                   \
      uint tid [[thread_index_in_threadgroup]]) {                            \
    threadgroup float coeff[dsv4_mhc::HC + dsv4_mhc::HC * dsv4_mhc::HC];     \
    dsv4_mhc::mhc_post_body<T>(x, residual, post_mix, comb_mix, out, H,      \
                               tgid.x, tgid.y, tpg.y, tid, coeff);           \
  }

// ---- hc head ---------------------------------------------------------------
// gate_s = sigmoid(dot(residual_flat, fn_s) * inv_rms * scale[0] + base[s])
//          + hc_eps;  out = sum_s gate_s * residual[s]

template <typename T>
inline void hc_head_body(device const T* residual, device const float* fn,
                         device const float* scale, device const float* base,
                         device T* out, uint H, float rms_eps, float hc_eps,
                         uint token, uint tid, uint sg, uint lane,
                         threadgroup float* shm) {
  const uint total = HC * H;
  device const T* res_t = residual + token * total;

  // simdgroups 0..3 each own one gate's projection; simdgroup 4 owns the
  // square-sum; 5..7 idle until the apply phase.
  float acc = 0.0f;
  if (sg < HC) {
    device const float* fn_s = fn + sg * total;
    for (uint flat = lane; flat < total; flat += 32)
      acc += float(res_t[flat]) * fn_s[flat];
  } else if (sg == HC) {
    for (uint flat = lane; flat < total; flat += 32) {
      const float v = float(res_t[flat]);
      acc += v * v;
    }
  }
  acc = simd_sum(acc);
  if (lane == 0 && sg <= HC) shm[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < HC) {
    const float inv_rms = rsqrt(shm[HC] / float(total) + rms_eps);
    shm[HC + 1 + tid] =
        sigmoid_f(shm[tid] * inv_rms * scale[0] + base[tid]) + hc_eps;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float g0 = shm[HC + 1], g1 = shm[HC + 2];
  const float g2 = shm[HC + 3], g3 = shm[HC + 4];
  for (uint d = tid; d < H; d += THREADS) {
    const float v = g0 * float(res_t[d]) + g1 * float(res_t[H + d]) +
                    g2 * float(res_t[2 * H + d]) + g3 * float(res_t[3 * H + d]);
    out[token * H + d] = T(v);
  }
}

#define instantiate_hc_head(tname, T)                                        \
  [[host_name("dsv4_hc_head_" #tname)]] kernel void dsv4_hc_head_##tname(    \
      device const T* residual [[buffer(0)]],                                \
      device const float* fn [[buffer(1)]],                                  \
      device const float* scale [[buffer(2)]],                               \
      device const float* base [[buffer(3)]],                                \
      device T* out [[buffer(4)]], constant uint& H [[buffer(5)]],           \
      constant float& rms_eps [[buffer(6)]],                                 \
      constant float& hc_eps [[buffer(7)]],                                  \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float shm[2 * dsv4_mhc::HC + 2];                             \
    dsv4_mhc::hc_head_body<T>(residual, fn, scale, base, out, H, rms_eps,    \
                              hc_eps, tgid.x, tid, sg, lane, shm);           \
  }

}  // namespace dsv4_mhc

instantiate_mhc_pre_dots(float16, half);
instantiate_mhc_pre_dots(bfloat16, bfloat);
instantiate_mhc_pre_dots_tg(float16, half);
instantiate_mhc_pre_dots_tg(bfloat16, bfloat);
instantiate_mhc_pre_finalize(float16, half);
instantiate_mhc_pre_finalize(bfloat16, bfloat);
instantiate_mhc_fused(float16, half);
instantiate_mhc_fused(bfloat16, bfloat);
instantiate_mhc_post(float16, half);
instantiate_mhc_post(bfloat16, bfloat);
instantiate_hc_head(float16, half);
instantiate_hc_head(bfloat16, bfloat);
