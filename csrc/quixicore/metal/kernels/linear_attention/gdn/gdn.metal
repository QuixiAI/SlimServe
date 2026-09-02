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
// inputs via cu_seqlens; persistent fp32 state pool indexed by slot_mapping[req],
// slot rows state_stride floats apart (each row is (Hv, Dv, Dk) contiguous;
// vLLM's page-packed pools have state_stride > Hv*Dv*Dk) — each simdgroup owns
// its (hv, dv) row exclusively, so the in-place pool update is race-free. GQA: hk = hv / (Hv/Hk). load_initial == 0 starts fresh
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
                      constant int   &state_stride [[buffer(14)]],
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
    device float *state_ptr = state_pool + slot * (long)state_stride +
        ((long)hv_idx * Dv + dv_idx) * DK;

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
                      constant int &state_stride [[buffer(14)]],                 \
                      uint3 gid [[threadgroup_position_in_grid]],                \
                      uint lane [[thread_index_in_simdgroup]]);

instantiate_gdn_recur(float32, float, 64)
instantiate_gdn_recur(float32, float, 128)
instantiate_gdn_recur(float16, half, 64)
instantiate_gdn_recur(float16, half, 128)
instantiate_gdn_recur(bfloat16, bf16, 64)
instantiate_gdn_recur(bfloat16, bf16, 128)

// ---------------------------------------------------------------------------
// Speculative-verify variant of gdn_recur (fused_sigmoid_gating_delta_rule_
// update with IS_SPEC_DECODING): each request carries a row of num_spec+1
// state slots. The initial state loads from slot_table[r, num_accepted[r]-1]
// (the checkpoint at the last accepted token), and after every timestep the
// full updated state is stored to slot_table[r, t], so the next step can
// rewind to whichever draft position verification accepts. Slot ids <= 0 are
// the null block: a null initial slot skips the whole request (its y rows are
// padding), and null per-position slots skip only that checkpoint store. The
// per-timestep math is gdn_recur's verbatim.
// ---------------------------------------------------------------------------
template <typename T, int DK>
kernel void gdn_recur_spec(device const T *q            [[buffer(0)]],
                           device const T *k            [[buffer(1)]],
                           device const T *v            [[buffer(2)]],
                           device const T *g            [[buffer(3)]],
                           device const T *beta         [[buffer(4)]],
                           device float   *state_pool   [[buffer(5)]],
                           device const int *cu_seqlens   [[buffer(6)]],
                           device const int *slot_table   [[buffer(7)]],
                           device const int *num_accepted [[buffer(8)]],
                           device T       *y            [[buffer(9)]],
                           constant int   &num_requests [[buffer(10)]],
                           constant int   &Hk           [[buffer(11)]],
                           constant int   &Hv           [[buffer(12)]],
                           constant int   &Dv           [[buffer(13)]],
                           constant int   &state_stride [[buffer(14)]],
                           constant int   &table_stride [[buffer(15)]],
                           uint3 gid [[threadgroup_position_in_grid]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    static_assert(DK == 64 || DK == 128,
                  "gdn_recur_spec supports Dk in {64, 128}");
    constexpr int N_PER_T = DK / 32;
    const int req_idx = (int)gid.z / Hv;
    const int hv_idx = (int)gid.z % Hv;
    const int dv_idx = (int)gid.x;
    const int dk0 = (int)lane * N_PER_T;
    if (req_idx >= num_requests || dv_idx >= Dv) { return; }

    const int hk_idx = hv_idx / (Hv / Hk);
    const int seq_start = cu_seqlens[req_idx];
    const int seq_len = cu_seqlens[req_idx + 1] - seq_start;
    if (seq_len <= 0) { return; }
    device const int *slots = slot_table + (long)req_idx * table_stride;
    // num_accepted < 1 would index slots[-1] (an OOB read MPS does not
    // fault on); guard here since the hosts only validate dtype/length.
    const int na = num_accepted[req_idx];
    if (na < 1 || na > table_stride) { return; }
    const long init_slot = slots[na - 1];
    if (init_slot <= 0) { return; }
    device const float *init_ptr = state_pool + init_slot * (long)state_stride +
        ((long)hv_idx * Dv + dv_idx) * DK;

    float state[N_PER_T];
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
        state[i] = init_ptr[dk0 + i];
    }

    device const T *q_ = q + (long)seq_start * Hk * DK + hk_idx * DK;
    device const T *k_ = k + (long)seq_start * Hk * DK + hk_idx * DK;
    device const T *v_ = v + (long)seq_start * Hv * Dv + hv_idx * Dv;
    device const T *g_ = g + (long)seq_start * Hv;
    device const T *beta_ = beta + (long)seq_start * Hv;
    device T *y_ = y + (long)seq_start * Hv * Dv + hv_idx * Dv;

    using TN = metal::vec<T, N_PER_T>;
    for (int t = 0; t < seq_len; ++t) {
        const float g_val = float(g_[hv_idx]);
        const TN kvec = ((device const TN*)(k_ + dk0))[0];
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

        // slots holds table_stride entries per request; positions past
        // the table have no checkpoint rather than reading the next row
        const long ckpt_slot = (t < table_stride) ? slots[t] : 0;
        if (ckpt_slot > 0) {
            device float *ckpt = state_pool + ckpt_slot * (long)state_stride +
                ((long)hv_idx * Dv + dv_idx) * DK;
            #pragma clang loop unroll(full)
            for (int i = 0; i < N_PER_T; ++i) {
                ckpt[dk0 + i] = state[i];
            }
        }

        q_ += Hk * DK;
        k_ += Hk * DK;
        v_ += Hv * Dv;
        y_ += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }
}

#define instantiate_gdn_recur_spec(type_name, T, DKVAL)                          \
  template [[host_name("gdn_recur_spec_" #type_name "_d" #DKVAL)]]              \
  [[kernel]] void gdn_recur_spec<T, DKVAL>(                                      \
      device const T *q [[buffer(0)]],                                           \
      device const T *k [[buffer(1)]],                                           \
      device const T *v [[buffer(2)]],                                           \
      device const T *g [[buffer(3)]],                                           \
      device const T *beta [[buffer(4)]],                                        \
      device float *state_pool [[buffer(5)]],                                    \
      device const int *cu_seqlens [[buffer(6)]],                                \
      device const int *slot_table [[buffer(7)]],                                \
      device const int *num_accepted [[buffer(8)]],                              \
      device T *y [[buffer(9)]],                                                 \
      constant int &num_requests [[buffer(10)]],                                 \
      constant int &Hk [[buffer(11)]],                                           \
      constant int &Hv [[buffer(12)]],                                           \
      constant int &Dv [[buffer(13)]],                                           \
      constant int &state_stride [[buffer(14)]],                                 \
      constant int &table_stride [[buffer(15)]],                                 \
      uint3 gid [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_gdn_recur_spec(float32, float, 64)
instantiate_gdn_recur_spec(float32, float, 128)
instantiate_gdn_recur_spec(bfloat16, bf16, 64)
instantiate_gdn_recur_spec(bfloat16, bf16, 128)

#undef instantiate_gdn_recur_spec

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
    constant int &state_stride [[buffer(11)]],
    constant int &state_cols [[buffer(12)]],
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
  // Slot rows are state_stride floats apart; each row is (channels,
  // state_cols) contiguous. With speculation the pool carries
  // kernel_size-1+num_spec columns per channel, but the non-spec ring
  // occupies only the first kernel_size-1 of them (upstream forces
  // state_len = width-1 on the non-spec paths).
  device float *state = state_pool + (long)slot * state_stride +
      (long)channel * state_cols;
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
                    constant int &state_stride [[buffer(11)]],                  \
                    constant int &state_cols [[buffer(12)]],                    \
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

// ---------------------------------------------------------------------------
// Fused decode-side preparation: short conv + silu (history register-resident,
// state pool update included), q/k rms-form l2-norm + scale, v passthrough,
// and the decay/beta gate — one dispatch replacing gdn_short_conv +
// gdn_qkv_prepare + gdn_gate_beta plus the host-side fp32 cast and the b/a
// contiguous copies. Reads the projection outputs in place: qkvz rows hold
// [q|k|v|z] (only the first 2*Hk*DK + Hv*DV columns are touched) and ba rows
// hold [b|a], both with explicit element row strides. Outputs are the fp32
// serving-chain tensors gdn_recur consumes. One simdgroup per (request,
// logical row): rows 0..Hk-1 are q heads, Hk..2Hk-1 k heads, 2Hk..2Hk+Hv-1
// v heads, and one final row computes decay/beta for all Hv heads. The
// per-element math replicates the three source kernels operation for
// operation (same channel->lane mapping, same accumulate order, same
// simd_sum), so the fp32 outputs are intended bit-identical to the unfused
// chain. Routed for pure-decode batches; the token loop still handles short
// varlen runs (speculative verify) with the conv's history semantics.
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV>
kernel void gdn_fused_prepare(
    device const T *qkvz [[buffer(0)]],
    device const T *ba [[buffer(1)]],
    device const float *conv_w [[buffer(2)]],
    device float *conv_state_pool [[buffer(3)]],
    device const int *cu_seqlens [[buffer(4)]],
    device const int *slot_mapping [[buffer(5)]],
    device const float *A_log [[buffer(6)]],
    device const float *dt_bias [[buffer(7)]],
    device float *q [[buffer(8)]],
    device float *k [[buffer(9)]],
    device float *v [[buffer(10)]],
    device float *decay [[buffer(11)]],
    device float *beta_out [[buffer(12)]],
    constant int &num_requests [[buffer(13)]],
    constant int &Hk [[buffer(14)]],
    constant int &Hv [[buffer(15)]],
    constant int &kernel_size [[buffer(16)]],
    constant int &load_initial [[buffer(17)]],
    constant int &qkvz_stride [[buffer(18)]],
    constant int &ba_stride [[buffer(19)]],
    constant int &conv_state_stride [[buffer(20)]],
    constant float &eps [[buffer(21)]],
    constant float &q_scale [[buffer(22)]],
    constant float &k_scale [[buffer(23)]],
    constant int &state_cols [[buffer(24)]],
    device const int *num_accepted [[buffer(25)]],
    constant int &spec_mode [[buffer(26)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int QK_PER_LANE = DK / 32;
  constexpr int V_PER_LANE = DV / 32;
  constexpr int MAX_HISTORY = 7;
  const int rows_per_request = 2 * Hk + Hv + 1;
  const int request = int(row) / rows_per_request;
  const int lrow = int(row) % rows_per_request;
  if (request >= num_requests) {
    return;
  }
  const int start = cu_seqlens[request];
  const int end = cu_seqlens[request + 1];
  const long slot = slot_mapping[request];
  const int hist_len = kernel_size - 1;
  // Speculative rewind (causal_conv1d_update IS_SPEC_DECODING): the history
  // window starts num_accepted-1 columns into the state row, and the write
  // lays out the shifted old window followed by every new token so the next
  // step can rewind to any accepted point. Non-spec keeps the front ring.
  int read_off = 0;
  if (spec_mode != 0) {
    // num_accepted < 1 would start the history window before the state
    // row, and a window past state_cols would read the next channel's
    // columns; the hosts validate dtype/shape only, so bound it here.
    const int na = num_accepted[request];
    if (na < 1 || na - 1 + hist_len > state_cols) {
      return;
    }
    read_off = na - 1;
  }

  if (lrow == 2 * Hk + Hv) {
    // Gate row: gdn_gate_beta verbatim over the packed [b|a] rows. No slot
    // dependence (the unfused kernel computes these for padded requests too).
    for (int token = start; token < end; ++token) {
      device const T *ba_row = ba + (long)token * ba_stride;
      for (int hv = int(lane); hv < Hv; hv += 32) {
        const float alpha = float(ba_row[Hv + hv]) + dt_bias[hv];
        const float softplus =
            alpha > 20.0f ? alpha : metal::log(1.0f + metal::exp(alpha));
        const long idx = (long)token * Hv + hv;
        decay[idx] = metal::exp(-metal::exp(A_log[hv]) * softplus);
        const float beta_logit = float(ba_row[hv]);
        beta_out[idx] = 1.0f / (1.0f + metal::exp(-beta_logit));
      }
    }
    return;
  }

  const bool is_v = lrow >= 2 * Hk;

  if (!is_v) {
    const bool is_k = lrow >= Hk;
    const int head = is_k ? lrow - Hk : lrow;
    // Conv channel index == qkvz column for the [q|k|v] prefix.
    const int cbase = (is_k ? Hk * DK : 0) + head * DK;
    if (slot < 0 || (spec_mode != 0 && slot == 0)) {
      // Padded request: gdn_short_conv zero-fills, and the norm of a zero row
      // is zero. Spec mode also treats slot 0 (the null block) as padding,
      // matching the Triton kernel's null_block_id check.
      for (int token = start; token < end; ++token) {
        device float *dst = (is_k ? k : q) + ((long)token * Hk + head) * DK +
            (long)lane * QK_PER_LANE;
        #pragma clang loop unroll(full)
        for (int i = 0; i < QK_PER_LANE; ++i) {
          dst[i] = 0.0f;
        }
      }
      return;
    }
    device float *state_base = conv_state_pool + slot * (long)conv_state_stride;
    float history[QK_PER_LANE * MAX_HISTORY];
    #pragma clang loop unroll(full)
    for (int i = 0; i < QK_PER_LANE; ++i) {
      const int c = cbase + int(lane) * QK_PER_LANE + i;
      for (int j = 0; j < hist_len; ++j) {
        history[i * MAX_HISTORY + j] = load_initial != 0
            ? state_base[(long)c * state_cols + read_off + j]
            : 0.0f;
      }
    }
    if (spec_mode != 0) {
      // Shifted old window first (the initial registers still hold it):
      // new[j-1] = old[read_off + j] for j in [1, hist_len).
      #pragma clang loop unroll(full)
      for (int i = 0; i < QK_PER_LANE; ++i) {
        const int c = cbase + int(lane) * QK_PER_LANE + i;
        for (int j = 1; j < hist_len; ++j) {
          state_base[(long)c * state_cols + (j - 1)] =
              history[i * MAX_HISTORY + j];
        }
      }
    }
    for (int token = start; token < end; ++token) {
      device const T *x_row = qkvz + (long)token * qkvz_stride;
      float values[QK_PER_LANE];
      float sum_sq = 0.0f;
      #pragma clang loop unroll(full)
      for (int i = 0; i < QK_PER_LANE; ++i) {
        const int c = cbase + int(lane) * QK_PER_LANE + i;
        device const float *w = conv_w + (long)c * kernel_size;
        const float current = float(x_row[c]);
        float value = current * w[kernel_size - 1];
        for (int j = 0; j < hist_len; ++j) {
          value += history[i * MAX_HISTORY + j] * w[j];
        }
        value *= 1.0f / (1.0f + metal::exp(-value));
        values[i] = value;
        sum_sq += value * value;
        for (int j = 0; j < hist_len - 1; ++j) {
          history[i * MAX_HISTORY + j] = history[i * MAX_HISTORY + j + 1];
        }
        history[i * MAX_HISTORY + hist_len - 1] = current;
        if (spec_mode != 0) {
          // Every new token lands after the shifted window so the next
          // step's rewind can pick any accepted point.
          state_base[(long)c * state_cols + (hist_len - 1) + (token - start)] =
              current;
        }
      }
      sum_sq = metal::simd_sum(sum_sq);
      const float scale =
          (is_k ? k_scale : q_scale) * metal::rsqrt(sum_sq / float(DK) + eps);
      device float *dst = (is_k ? k : q) + ((long)token * Hk + head) * DK +
          (long)lane * QK_PER_LANE;
      #pragma clang loop unroll(full)
      for (int i = 0; i < QK_PER_LANE; ++i) {
        dst[i] = values[i] * scale;
      }
    }
    if (spec_mode == 0) {
      #pragma clang loop unroll(full)
      for (int i = 0; i < QK_PER_LANE; ++i) {
        const int c = cbase + int(lane) * QK_PER_LANE + i;
        for (int j = 0; j < hist_len; ++j) {
          state_base[(long)c * state_cols + j] = history[i * MAX_HISTORY + j];
        }
      }
    }
    return;
  }

  const int head = lrow - 2 * Hk;
  const int cbase = 2 * Hk * DK + head * DV;
  if (slot < 0 || (spec_mode != 0 && slot == 0)) {
    for (int token = start; token < end; ++token) {
      device float *dst =
          v + ((long)token * Hv + head) * DV + (long)lane * V_PER_LANE;
      #pragma clang loop unroll(full)
      for (int i = 0; i < V_PER_LANE; ++i) {
        dst[i] = 0.0f;
      }
    }
    return;
  }
  device float *state_base = conv_state_pool + slot * (long)conv_state_stride;
  float history[V_PER_LANE * MAX_HISTORY];
  #pragma clang loop unroll(full)
  for (int i = 0; i < V_PER_LANE; ++i) {
    const int c = cbase + int(lane) * V_PER_LANE + i;
    for (int j = 0; j < hist_len; ++j) {
      history[i * MAX_HISTORY + j] = load_initial != 0
          ? state_base[(long)c * state_cols + read_off + j]
          : 0.0f;
    }
  }
  if (spec_mode != 0) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < V_PER_LANE; ++i) {
      const int c = cbase + int(lane) * V_PER_LANE + i;
      for (int j = 1; j < hist_len; ++j) {
        state_base[(long)c * state_cols + (j - 1)] =
            history[i * MAX_HISTORY + j];
      }
    }
  }
  for (int token = start; token < end; ++token) {
    device const T *x_row = qkvz + (long)token * qkvz_stride;
    device float *dst =
        v + ((long)token * Hv + head) * DV + (long)lane * V_PER_LANE;
    #pragma clang loop unroll(full)
    for (int i = 0; i < V_PER_LANE; ++i) {
      const int c = cbase + int(lane) * V_PER_LANE + i;
      device const float *w = conv_w + (long)c * kernel_size;
      const float current = float(x_row[c]);
      float value = current * w[kernel_size - 1];
      for (int j = 0; j < hist_len; ++j) {
        value += history[i * MAX_HISTORY + j] * w[j];
      }
      value *= 1.0f / (1.0f + metal::exp(-value));
      dst[i] = value;
      for (int j = 0; j < hist_len - 1; ++j) {
        history[i * MAX_HISTORY + j] = history[i * MAX_HISTORY + j + 1];
      }
      history[i * MAX_HISTORY + hist_len - 1] = current;
      if (spec_mode != 0) {
        state_base[(long)c * state_cols + (hist_len - 1) + (token - start)] =
            current;
      }
    }
  }
  if (spec_mode == 0) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < V_PER_LANE; ++i) {
      const int c = cbase + int(lane) * V_PER_LANE + i;
      for (int j = 0; j < hist_len; ++j) {
        state_base[(long)c * state_cols + j] = history[i * MAX_HISTORY + j];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Gated RMSNorm over the fp32 recurrence output: rounds y to the activation
// dtype in-register first (bitwise the host chain's y.to(dtype) cast), reads
// the z gate in place from the projection row (explicit token stride), and
// writes the activation-dtype result — replacing the cast dispatch, the z
// gather copy, and the container copy around gdn_gated_rmsnorm. One simdgroup
// per (token, hv) row; the norm chain is gdn_gated_rmsnorm's verbatim.
// ---------------------------------------------------------------------------
template <typename T, int D>
kernel void gdn_gated_rmsnorm_f32(
    device const float *y [[buffer(0)]],
    device const T *z [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device T *out [[buffer(3)]],
    constant int &rows [[buffer(4)]],
    constant int &Hv [[buffer(5)]],
    constant int &z_stride [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER_LANE = D / 32;
  if (int(row) >= rows) {
    return;
  }
  const int token = int(row) / Hv;
  const int hv = int(row) % Hv;
  const long y_off = (long)row * D;
  const long z_off = (long)token * z_stride + (long)hv * D;
  float values[PER_LANE];
  float sum_sq = 0.0f;
  #pragma clang loop unroll(full)
  for (int i = 0; i < PER_LANE; ++i) {
    const int d = int(lane) * PER_LANE + i;
    values[i] = float(T(y[y_off + d]));
    sum_sq += values[i] * values[i];
  }
  sum_sq = metal::simd_sum(sum_sq);
  const float inv_rms = metal::rsqrt(sum_sq / float(D) + eps);
  #pragma clang loop unroll(full)
  for (int i = 0; i < PER_LANE; ++i) {
    const int d = int(lane) * PER_LANE + i;
    const float gate = float(z[z_off + d]);
    const float silu_gate = gate / (1.0f + metal::exp(-gate));
    out[y_off + d] = T(values[i] * inv_rms * float(weight[d]) * silu_gate);
  }
}

#define instantiate_gdn_fused_prepare(type_name, T, DKVAL, DVVAL)               \
  template [[host_name("gdn_fused_prepare_" #type_name "_dk" #DKVAL             \
                       "_dv" #DVVAL)]] [[kernel]] void                           \
  gdn_fused_prepare<T, DKVAL, DVVAL>(                                            \
      device const T *qkvz [[buffer(0)]],                                        \
      device const T *ba [[buffer(1)]],                                          \
      device const float *conv_w [[buffer(2)]],                                  \
      device float *conv_state_pool [[buffer(3)]],                               \
      device const int *cu_seqlens [[buffer(4)]],                                \
      device const int *slot_mapping [[buffer(5)]],                              \
      device const float *A_log [[buffer(6)]],                                   \
      device const float *dt_bias [[buffer(7)]],                                 \
      device float *q [[buffer(8)]], device float *k [[buffer(9)]],              \
      device float *v [[buffer(10)]], device float *decay [[buffer(11)]],        \
      device float *beta_out [[buffer(12)]],                                     \
      constant int &num_requests [[buffer(13)]],                                 \
      constant int &Hk [[buffer(14)]], constant int &Hv [[buffer(15)]],          \
      constant int &kernel_size [[buffer(16)]],                                  \
      constant int &load_initial [[buffer(17)]],                                 \
      constant int &qkvz_stride [[buffer(18)]],                                  \
      constant int &ba_stride [[buffer(19)]],                                    \
      constant int &conv_state_stride [[buffer(20)]],                            \
      constant float &eps [[buffer(21)]],                                        \
      constant float &q_scale [[buffer(22)]],                                    \
      constant float &k_scale [[buffer(23)]],                                    \
      constant int &state_cols [[buffer(24)]],                                   \
      device const int *num_accepted [[buffer(25)]],                             \
      constant int &spec_mode [[buffer(26)]],                                    \
      uint row [[threadgroup_position_in_grid]],                                 \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_gdn_norm_f32(type_name, T, DVAL)                            \
  template [[host_name("gdn_gated_rmsnorm_f32_" #type_name "_d" #DVAL)]]        \
  [[kernel]] void gdn_gated_rmsnorm_f32<T, DVAL>(                               \
      device const float *y [[buffer(0)]], device const T *z [[buffer(1)]],     \
      device const T *weight [[buffer(2)]], device T *out [[buffer(3)]],        \
      constant int &rows [[buffer(4)]], constant int &Hv [[buffer(5)]],         \
      constant int &z_stride [[buffer(6)]], constant float &eps [[buffer(7)]],  \
      uint row [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_gdn_fused(type_name, T)                                     \
  instantiate_gdn_fused_prepare(type_name, T, 64, 64)                           \
  instantiate_gdn_fused_prepare(type_name, T, 64, 128)                          \
  instantiate_gdn_fused_prepare(type_name, T, 128, 64)                          \
  instantiate_gdn_fused_prepare(type_name, T, 128, 128)                         \
  instantiate_gdn_norm_f32(type_name, T, 64)                                    \
  instantiate_gdn_norm_f32(type_name, T, 128)

instantiate_gdn_fused(float32, float)
instantiate_gdn_fused(float16, half)
instantiate_gdn_fused(bfloat16, bf16)

#undef instantiate_gdn_fused
#undef instantiate_gdn_fused_prepare
#undef instantiate_gdn_norm_f32

}
