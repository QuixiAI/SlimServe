#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// ---------------------------------------------------------------------------
// GDN / GatedDeltaNet linear attention (Qwen3-Next / Kimi-Linear-class hybrid mixer):
// per-timestep delta-rule recurrence over a per-(head, dv) state row S[dv, 0:Dk]:
//
//   S      *= g[t, hv]                      (scalar gate/decay, a multiplier in (0,1])
//   kv_mem  = k[t] . S                      (simd_sum over Dk lanes)
//   delta   = (v[t, dv] - kv_mem) * beta[t, hv]
//   S      += k[t] * delta                  (rank-1 delta correction)
//   y[t,dv] = q[t] . S                      (simd_sum)
//
// One simdgroup per (request, hv, dv): grid (Dv, 1, R*Hv), 32 lanes partition Dk
// (DK/32 fp32 state elements per lane; DK in {64, 128} compile-time). Varlen packed
// inputs via cu_seqlens; persistent fp32 state pool (num_slots, Hv, Dv, Dk) indexed by
// slot_mapping[req] — each simdgroup owns its (hv, dv) row exclusively, so the in-place
// pool update is race-free. GQA: hk = hv / (Hv/Hk). load_initial == 0 starts fresh
// prefills at S = 0 (the pool row is still overwritten at the end).
// Port of metal-forge gdn_linear_attention with the state promoted to fp32.
// ---------------------------------------------------------------------------
template <typename T, int DK>
kernel void gdn_recur(device const T *q            [[buffer(0)]],
                      device const T *k            [[buffer(1)]],
                      device const T *v            [[buffer(2)]],
                      device const T *g            [[buffer(3)]],
                      device const T *beta         [[buffer(4)]],
                      device float   *state_pool   [[buffer(5)]],
                      device const int *cu_seqlens   [[buffer(6)]],
                      device const int *slot_mapping [[buffer(7)]],
                      device T       *y            [[buffer(8)]],
                      constant int   &num_requests [[buffer(9)]],
                      constant int   &Hk           [[buffer(10)]],
                      constant int   &Hv           [[buffer(11)]],
                      constant int   &Dv           [[buffer(12)]],
                      constant int   &load_initial [[buffer(13)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint  lane [[thread_index_in_simdgroup]]) {
    static_assert(DK == 64 || DK == 128, "gdn_recur supports Dk in {64, 128}");
    constexpr int N_PER_T = DK / 32;
    const int req_idx = (int)gid.z / Hv;
    const int hv_idx = (int)gid.z % Hv;
    const int dv_idx = (int)gid.x;
    const int dk0 = (int)lane * N_PER_T;
    if (req_idx >= num_requests || dv_idx >= Dv) { return; }

    const int hk_idx = hv_idx / (Hv / Hk);
    const int seq_start = cu_seqlens[req_idx];
    const int seq_len = cu_seqlens[req_idx + 1] - seq_start;
    const long slot = slot_mapping[req_idx];
    device float *state_ptr = state_pool + ((slot * Hv + hv_idx) * (long)Dv + dv_idx) * DK;

    float state[N_PER_T];
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
        state[i] = (load_initial != 0) ? state_ptr[dk0 + i] : 0.0f;
    }

    device const T *q_ = q + (long)seq_start * Hk * DK + hk_idx * DK;
    device const T *k_ = k + (long)seq_start * Hk * DK + hk_idx * DK;
    device const T *v_ = v + (long)seq_start * Hv * Dv + hv_idx * Dv;
    device const T *g_ = g + (long)seq_start * Hv;
    device const T *beta_ = beta + (long)seq_start * Hv;
    device T *y_ = y + (long)seq_start * Hv * Dv + hv_idx * Dv;

    using TN = metal::vec<T, N_PER_T>;   // lane owns N_PER_T CONTIGUOUS Dk elems -> vec load
    for (int t = 0; t < seq_len; ++t) {
        const float g_val = float(g_[hv_idx]);
        const TN kvec = ((device const TN*)(k_ + dk0))[0];   // k read once, reused below
        const TN qvec = ((device const TN*)(q_ + dk0))[0];
        float kv_mem = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < N_PER_T; ++i) {
            state[i] *= g_val;
            kv_mem += state[i] * float(kvec[i]);
        }
        kv_mem = metal::simd_sum(kv_mem);

        const float delta = (float(v_[dv_idx]) - kv_mem) * float(beta_[hv_idx]);

        float out = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < N_PER_T; ++i) {
            state[i] += float(kvec[i]) * delta;
            out += state[i] * float(qvec[i]);
        }
        out = metal::simd_sum(out);
        if (lane == 0) {
            y_[dv_idx] = T(out);
        }

        q_ += Hk * DK;
        k_ += Hk * DK;
        v_ += Hv * Dv;
        y_ += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
        state_ptr[dk0 + i] = state[i];
    }
}

#define instantiate_gdn_recur(type_name, T, DKVAL)                              \
  template [[host_name("gdn_recur_" #type_name "_d" #DKVAL)]] [[kernel]] void   \
  gdn_recur<T, DKVAL>(device const T *q [[buffer(0)]],                           \
                      device const T *k [[buffer(1)]],                           \
                      device const T *v [[buffer(2)]],                           \
                      device const T *g [[buffer(3)]],                           \
                      device const T *beta [[buffer(4)]],                        \
                      device float *state_pool [[buffer(5)]],                    \
                      device const int *cu_seqlens [[buffer(6)]],                \
                      device const int *slot_mapping [[buffer(7)]],              \
                      device T *y [[buffer(8)]],                                 \
                      constant int &num_requests [[buffer(9)]],                  \
                      constant int &Hk [[buffer(10)]],                           \
                      constant int &Hv [[buffer(11)]],                           \
                      constant int &Dv [[buffer(12)]],                           \
                      constant int &load_initial [[buffer(13)]],                 \
                      uint3 gid [[threadgroup_position_in_grid]],                \
                      uint lane [[thread_index_in_simdgroup]]);

instantiate_gdn_recur(float32, float, 64)
instantiate_gdn_recur(float32, float, 128)
instantiate_gdn_recur(float16, half, 64)
instantiate_gdn_recur(float16, half, 128)
instantiate_gdn_recur(bfloat16, bf16, 64)
instantiate_gdn_recur(bfloat16, bf16, 128)

// ---------------------------------------------------------------------------
// Reusable Gated DeltaNet preparation/output operations.
// ---------------------------------------------------------------------------

template <typename T>
kernel void gdn_short_conv(
    device const T *x [[buffer(0)]],
    device const T *weight [[buffer(1)]],
    device float *state_pool [[buffer(2)]],
    device const int *cu_seqlens [[buffer(3)]],
    device const int *slot_mapping [[buffer(4)]],
    device T *out [[buffer(5)]],
    constant int &num_requests [[buffer(6)]],
    constant int &channels [[buffer(7)]],
    constant int &kernel_size [[buffer(8)]],
    constant int &load_initial [[buffer(9)]],
    constant int &apply_silu [[buffer(10)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]]) {
  const int request = int(group_pos.y);
  const int channel = int(group_pos.x) * 256 + int(thread_pos.x);
  if (request >= num_requests || channel >= channels) {
    return;
  }

  const int start = cu_seqlens[request];
  const int end = cu_seqlens[request + 1];
  const int slot = slot_mapping[request];
  if (slot < 0) {
    for (int token = start; token < end; ++token) {
      out[(long)token * channels + channel] = T(0.0f);
    }
    return;
  }

  constexpr int MAX_HISTORY = 7;
  float history[MAX_HISTORY];
  device float *state = state_pool +
      ((long)slot * channels + channel) * (kernel_size - 1);
  for (int j = 0; j < kernel_size - 1; ++j) {
    history[j] = load_initial != 0 ? state[j] : 0.0f;
  }

  device const T *w = weight + (long)channel * kernel_size;
  for (int token = start; token < end; ++token) {
    const float current = float(x[(long)token * channels + channel]);
    float value = current * float(w[kernel_size - 1]);
    for (int j = 0; j < kernel_size - 1; ++j) {
      value += history[j] * float(w[j]);
    }
    if (apply_silu != 0) {
      value *= 1.0f / (1.0f + metal::exp(-value));
    }
    out[(long)token * channels + channel] = T(value);

    for (int j = 0; j < kernel_size - 2; ++j) {
      history[j] = history[j + 1];
    }
    history[kernel_size - 2] = current;
  }

  for (int j = 0; j < kernel_size - 1; ++j) {
    state[j] = history[j];
  }
}

template <typename T, int DK, int DV>
kernel void gdn_qkv_prepare(
    device const T *mixed [[buffer(0)]],
    device T *q [[buffer(1)]],
    device T *k [[buffer(2)]],
    device T *v [[buffer(3)]],
    constant int &tokens [[buffer(4)]],
    constant int &Hk [[buffer(5)]],
    constant int &Hv [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant float &q_scale [[buffer(8)]],
    constant float &k_scale [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int QK_PER_LANE = DK / 32;
  constexpr int V_PER_LANE = DV / 32;
  const int rows_per_token = 2 * Hk + Hv;
  const int token = int(row) / rows_per_token;
  const int logical_head = int(row) % rows_per_token;
  if (token >= tokens) {
    return;
  }
  const int channels = 2 * Hk * DK + Hv * DV;
  device const T *token_in = mixed + (long)token * channels;

  if (logical_head < 2 * Hk) {
    const bool is_k = logical_head >= Hk;
    const int head = is_k ? logical_head - Hk : logical_head;
    const long source_offset = (is_k ? (long)Hk * DK : 0) + (long)head * DK;
    const long output_offset = ((long)token * Hk + head) * DK;
    device T *dst = is_k ? k + output_offset : q + output_offset;
    float values[QK_PER_LANE];
    float sum_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < QK_PER_LANE; ++i) {
      const int d = int(lane) * QK_PER_LANE + i;
      const float value = float(token_in[source_offset + d]);
      values[i] = value;
      sum_sq += value * value;
    }
    sum_sq = metal::simd_sum(sum_sq);
    const float scale = (is_k ? k_scale : q_scale) *
        metal::rsqrt(sum_sq / float(DK) + eps);
    #pragma clang loop unroll(full)
    for (int i = 0; i < QK_PER_LANE; ++i) {
      const int d = int(lane) * QK_PER_LANE + i;
      dst[d] = T(values[i] * scale);
    }
    return;
  }

  const int head = logical_head - 2 * Hk;
  const long source_offset = (long)2 * Hk * DK + (long)head * DV;
  const long output_offset = ((long)token * Hv + head) * DV;
  #pragma clang loop unroll(full)
  for (int i = 0; i < V_PER_LANE; ++i) {
    const int d = int(lane) * V_PER_LANE + i;
    v[output_offset + d] = token_in[source_offset + d];
  }
}

template <typename T>
kernel void gdn_gate_beta(
    device const T *a [[buffer(0)]],
    device const T *b [[buffer(1)]],
    device const float *A_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device float *decay [[buffer(4)]],
    device float *beta [[buffer(5)]],
    constant uint &n [[buffer(6)]],
    constant int &heads [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= n) {
    return;
  }
  const int head = int(idx) % heads;
  const float alpha = float(a[idx]) + dt_bias[head];
  const float softplus = alpha > 20.0f ? alpha : metal::log(1.0f + metal::exp(alpha));
  decay[idx] = metal::exp(-metal::exp(A_log[head]) * softplus);
  const float beta_logit = float(b[idx]);
  beta[idx] = 1.0f / (1.0f + metal::exp(-beta_logit));
}

template <typename T, int D>
kernel void gdn_gated_rmsnorm(
    device const T *y [[buffer(0)]],
    device const T *z [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device T *out [[buffer(3)]],
    constant int &rows [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER_LANE = D / 32;
  if (int(row) >= rows) {
    return;
  }
  const long offset = (long)row * D;
  float values[PER_LANE];
  float sum_sq = 0.0f;
  #pragma clang loop unroll(full)
  for (int i = 0; i < PER_LANE; ++i) {
    const int d = int(lane) * PER_LANE + i;
    values[i] = float(y[offset + d]);
    sum_sq += values[i] * values[i];
  }
  sum_sq = metal::simd_sum(sum_sq);
  const float inv_rms = metal::rsqrt(sum_sq / float(D) + eps);
  #pragma clang loop unroll(full)
  for (int i = 0; i < PER_LANE; ++i) {
    const int d = int(lane) * PER_LANE + i;
    const float gate = float(z[offset + d]);
    const float silu_gate = gate / (1.0f + metal::exp(-gate));
    out[offset + d] = T(values[i] * inv_rms * float(weight[d]) * silu_gate);
  }
}

#define instantiate_gdn_qkv(type_name, T, DKVAL, DVVAL)                         \
  template [[host_name("gdn_qkv_prepare_" #type_name "_dk" #DKVAL "_dv" #DVVAL)]] \
  [[kernel]] void gdn_qkv_prepare<T, DKVAL, DVVAL>(                             \
      device const T *mixed [[buffer(0)]], device T *q [[buffer(1)]],           \
      device T *k [[buffer(2)]], device T *v [[buffer(3)]],                     \
      constant int &tokens [[buffer(4)]], constant int &Hk [[buffer(5)]],       \
      constant int &Hv [[buffer(6)]], constant float &eps [[buffer(7)]],        \
      constant float &q_scale [[buffer(8)]],                                    \
      constant float &k_scale [[buffer(9)]],                                    \
      uint row [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_gdn_norm(type_name, T, DVAL)                                \
  template [[host_name("gdn_gated_rmsnorm_" #type_name "_d" #DVAL)]]          \
  [[kernel]] void gdn_gated_rmsnorm<T, DVAL>(                                   \
      device const T *y [[buffer(0)]], device const T *z [[buffer(1)]],         \
      device const T *weight [[buffer(2)]], device T *out [[buffer(3)]],        \
      constant int &rows [[buffer(4)]], constant float &eps [[buffer(5)]],      \
      uint row [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_gdn_helpers(type_name, T)                                   \
  template [[host_name("gdn_short_conv_" #type_name)]] [[kernel]] void         \
  gdn_short_conv<T>(device const T *x [[buffer(0)]],                             \
                    device const T *weight [[buffer(1)]],                       \
                    device float *state_pool [[buffer(2)]],                     \
                    device const int *cu_seqlens [[buffer(3)]],                 \
                    device const int *slot_mapping [[buffer(4)]],               \
                    device T *out [[buffer(5)]],                                \
                    constant int &num_requests [[buffer(6)]],                   \
                    constant int &channels [[buffer(7)]],                       \
                    constant int &kernel_size [[buffer(8)]],                    \
                    constant int &load_initial [[buffer(9)]],                   \
                    constant int &apply_silu [[buffer(10)]],                    \
                    uint3 group_pos [[threadgroup_position_in_grid]],           \
                    uint3 thread_pos [[thread_position_in_threadgroup]]);       \
  template [[host_name("gdn_gate_beta_" #type_name)]] [[kernel]] void           \
  gdn_gate_beta<T>(device const T *a [[buffer(0)]],                              \
                   device const T *b [[buffer(1)]],                             \
                   device const float *A_log [[buffer(2)]],                     \
                   device const float *dt_bias [[buffer(3)]],                   \
                   device float *decay [[buffer(4)]],                           \
                   device float *beta [[buffer(5)]],                            \
                   constant uint &n [[buffer(6)]],                              \
                   constant int &heads [[buffer(7)]],                           \
                   uint idx [[thread_position_in_grid]]);                       \
  instantiate_gdn_qkv(type_name, T, 64, 64)                                     \
  instantiate_gdn_qkv(type_name, T, 64, 128)                                    \
  instantiate_gdn_qkv(type_name, T, 128, 64)                                    \
  instantiate_gdn_qkv(type_name, T, 128, 128)                                   \
  instantiate_gdn_norm(type_name, T, 64)                                        \
  instantiate_gdn_norm(type_name, T, 128)

instantiate_gdn_helpers(float32, float)
instantiate_gdn_helpers(float16, half)
instantiate_gdn_helpers(bfloat16, bf16)

#undef instantiate_gdn_helpers
#undef instantiate_gdn_qkv
#undef instantiate_gdn_norm

}
