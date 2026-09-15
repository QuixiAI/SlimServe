#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// ---------------------------------------------------------------------------
// KDA (Kimi Delta Attention, GLM-5.3-Flash / Kimi-Linear) serving chain.
//
// Three kernels replace the torch-native per-op path (kda_mps_fallback.py):
//
//   kda_fused_prepare   : short conv (silu) over the packed [q|k|v] channels
//                         with the persistent ring, L2-normalized q (scaled)
//                         and k, raw v, the per-CHANNEL log-decay gate
//                             decay[h, d] = exp(lb * sigmoid(exp(A_log[h]) *
//                                               (g[h, d] + dt_bias[h, d])))
//                         (lb = gate_lower_bound; lb == 0 selects the softplus
//                         form -exp(A_log) * softplus(x)) and beta = sigmoid(b).
//   kda_recur           : gdn_recur with a per-channel decay vector:
//                             S[dv, :] *= decay[t, h, :]
//                             kv_mem    = k . S[dv, :]
//                             delta     = (v[dv] - kv_mem) * beta
//                             S[dv, :] += k * delta
//                             y[dv]     = q . S[dv, :]
//                         varlen over cu_seqlens, fp32 state pool rows
//                         (H, DV, DK) contiguous at slot_mapping[req] *
//                         state_stride. Slot <= 0 is the null block: the
//                         request's y rows are zeroed and the pool untouched.
//   kda_gated_rmsnorm_f32: rmsnorm(y) * weight * sigmoid(z) per (token, head).
//
// Numerics contract == kda_mps_fallback.py (fp32 math throughout; conv state
// kept in the pool's own element type; l2norm x / sqrt(sum_sq + eps)).
// ---------------------------------------------------------------------------

template <typename T, int DK, int DV>
kernel void kda_fused_prepare(
    device const T *qkv [[buffer(0)]],            // [tokens, qkv_stride] (pre-conv)
    device const T *g_logits [[buffer(1)]],       // [tokens, g_stride]  (H*DK)
    device const T *beta_logits [[buffer(2)]],    // [tokens, beta_stride] (H)
    device const float *conv_w [[buffer(3)]],     // [3*H*DK, kernel_size]
    device T *conv_state_pool [[buffer(4)]],      // slot*conv_state_stride + c*chan_stride + j*col_stride
    device const int *cu_seqlens [[buffer(5)]],
    device const int *slot_mapping [[buffer(6)]],
    device const float *A_log [[buffer(7)]],      // [H]
    device const float *dt_bias [[buffer(8)]],    // [H*DK]
    device float *q [[buffer(9)]],                // [tokens, H*DK]
    device float *k [[buffer(10)]],               // [tokens, H*DK]
    device float *v [[buffer(11)]],               // [tokens, H*DV]
    device float *decay [[buffer(12)]],           // [tokens, H*DK]
    device float *beta_out [[buffer(13)]],        // [tokens, H]
    constant int &num_requests [[buffer(14)]],
    constant int &H [[buffer(15)]],
    constant int &kernel_size [[buffer(16)]],
    constant int &load_initial [[buffer(17)]],
    constant int &qkv_stride [[buffer(18)]],
    constant int &g_stride [[buffer(19)]],
    constant int &beta_stride [[buffer(20)]],
    constant int &conv_state_stride [[buffer(21)]],
    constant int &chan_stride [[buffer(22)]],
    constant float &l2_eps [[buffer(23)]],
    constant float &q_scale [[buffer(24)]],
    constant float &lower_bound [[buffer(25)]],
    constant int &has_dt_bias [[buffer(26)]],
    constant int &col_stride [[buffer(27)]],
    device const int *num_accepted [[buffer(28)]],   // [requests] (spec mode)
    constant int &spec_mode [[buffer(29)]],
    constant int &chunk_tokens [[buffer(30)]],       // tokens per grid-y chunk (>= kernel_size-1)
    uint2 gid [[threadgroup_position_in_grid]],      // x: (request, row); y: token chunk
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = gid.x;
  static_assert(DK == DV, "kda_fused_prepare shares its lane layout across q/k/v");
  constexpr int QK_PER_LANE = DK / 32;
  constexpr int V_PER_LANE = DV / 32;
  constexpr int MAX_HISTORY = 7;
  // Row kinds per request: [q_h x H][k_h x H][v_h x H][gate_h x H].
  const int rows_per_request = 4 * H;
  const int request = int(row) / rows_per_request;
  const int lrow = int(row) % rows_per_request;
  if (request >= num_requests) {
    return;
  }
  const int kind = lrow / H;  // 0 q, 1 k, 2 v, 3 gate
  const int head = lrow % H;
  const int start = cu_seqlens[request];
  const int end = cu_seqlens[request + 1];
  const long slot = slot_mapping[request];
  const int hist_len = kernel_size - 1;
  // Speculative rewind (kda_conv_spec_update_native / causal_conv1d_update
  // IS_SPEC_DECODING): the history window starts num_accepted-1 columns
  // into the state row; the row is rewritten from column 0 as the shifted
  // old window followed by every new token, so the next step can rewind to
  // any accepted point. Non-spec keeps the front ring (gdn.metal's contract).
  int read_off = 0;
  if (spec_mode != 0) {
    read_off = num_accepted[request] - 1;
    if (read_off < 0) { read_off = 0; }
  }
  // Token-parallel prefill: grid y walks the request in chunk_tokens-long
  // slices (chunk 0 seeds its conv history from the pool, later chunks from
  // the raw rows just before them; the chunk holding the last token writes
  // the pool). Per-token arithmetic is the serial walk's, so outputs and the
  // written state are bit-identical for any chunk size. Spec mode (per-token
  // window writes) is launched as a single chunk.
  const int chunk = int(gid.y);
  const int cstart = start + chunk * chunk_tokens;
  const int cend = metal::min(end, cstart + chunk_tokens);
  if (cstart >= end) {
    return;
  }

  if (kind == 3) {
    // Gate row: per-channel decay for this head + beta. No slot dependence.
    const float a = metal::exp(A_log[head]);
    for (int token = cstart; token < cend; ++token) {
      device const T *g_row = g_logits + (long)token * g_stride + (long)head * DK;
      device float *d_row = decay + ((long)token * H + head) * DK;
      #pragma clang loop unroll(full)
      for (int i = 0; i < QK_PER_LANE; ++i) {
        const int d = int(lane) * QK_PER_LANE + i;
        float x = float(g_row[d]);
        if (has_dt_bias != 0) {
          x += dt_bias[head * DK + d];
        }
        float log_decay;
        if (lower_bound != 0.0f) {
          log_decay = lower_bound / (1.0f + metal::exp(-a * x));
        } else {
          const float sp = x > 20.0f ? x : metal::log(1.0f + metal::exp(x));
          log_decay = -a * sp;
        }
        d_row[d] = metal::exp(log_decay);
      }
      if (lane == 0) {
        const float b = float(beta_logits[(long)token * beta_stride + head]);
        beta_out[(long)token * H + head] = 1.0f / (1.0f + metal::exp(-b));
      }
    }
    return;
  }

  const bool is_v = kind == 2;
  const int per_lane = is_v ? V_PER_LANE : QK_PER_LANE;
  const int dim = is_v ? DV : DK;
  // Conv channel index == packed [q|k|v] column.
  const int cbase = kind * H * DK + head * dim;
  device float *dst_base = (kind == 0 ? q : (kind == 1 ? k : v));
  if (slot <= 0) {
    // Null block / padding: zero outputs, pool untouched (the recurrence
    // skips the request as well).
    for (int token = cstart; token < cend; ++token) {
      device float *dst = dst_base + ((long)token * H + head) * dim +
          (long)lane * per_lane;
      for (int i = 0; i < per_lane; ++i) {
        dst[i] = 0.0f;
      }
    }
    return;
  }
  device T *state_base = conv_state_pool + slot * (long)conv_state_stride;
  float history[QK_PER_LANE * MAX_HISTORY];
  #pragma clang loop unroll(full)
  for (int i = 0; i < QK_PER_LANE; ++i) {
    const int c = cbase + int(lane) * per_lane + i;
    for (int j = 0; j < hist_len; ++j) {
      if (chunk == 0) {
        history[i * MAX_HISTORY + j] = load_initial != 0
            ? float(state_base[(long)c * chan_stride +
                               (long)(read_off + j) * col_stride])
            : 0.0f;
      } else {
        history[i * MAX_HISTORY + j] =
            float(qkv[(long)(cstart - hist_len + j) * qkv_stride + c]);
      }
    }
  }
  if (spec_mode != 0) {
    // Shifted old window first: new[j-1] = old[read_off + j], j in [1, hist).
    #pragma clang loop unroll(full)
    for (int i = 0; i < QK_PER_LANE; ++i) {
      const int c = cbase + int(lane) * per_lane + i;
      for (int j = 1; j < hist_len; ++j) {
        state_base[(long)c * chan_stride + (long)(j - 1) * col_stride] =
            T(history[i * MAX_HISTORY + j]);
      }
    }
  }
  for (int token = cstart; token < cend; ++token) {
    device const T *x_row = qkv + (long)token * qkv_stride;
    float values[QK_PER_LANE];
    float sum_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < QK_PER_LANE; ++i) {
      const int c = cbase + int(lane) * per_lane + i;
      device const float *w = conv_w + (long)c * kernel_size;
      const float current = float(x_row[c]);
      float value = current * w[kernel_size - 1];
      for (int j = 0; j < hist_len; ++j) {
        value += history[i * MAX_HISTORY + j] * w[j];
      }
      value *= 1.0f / (1.0f + metal::exp(-value));  // silu
      // The reference conv writes its output in the activation dtype and the
      // recurrence reads it back; round here to keep the state bit-close.
      value = float(T(value));
      values[i] = value;
      sum_sq += value * value;
      for (int j = 0; j < hist_len - 1; ++j) {
        history[i * MAX_HISTORY + j] = history[i * MAX_HISTORY + j + 1];
      }
      history[i * MAX_HISTORY + hist_len - 1] = current;
      if (spec_mode != 0) {
        // Every new token lands after the shifted window.
        state_base[(long)c * chan_stride +
                   (long)((hist_len - 1) + (token - start)) * col_stride] =
            T(current);
      }
    }
    device float *dst = dst_base + ((long)token * H + head) * dim +
        (long)lane * per_lane;
    if (is_v) {
      #pragma clang loop unroll(full)
      for (int i = 0; i < QK_PER_LANE; ++i) {
        dst[i] = values[i];
      }
    } else {
      sum_sq = metal::simd_sum(sum_sq);
      // kda_l2norm_native: x / sqrt(sum_sq + eps); q additionally * scale.
      const float scale =
          (kind == 0 ? q_scale : 1.0f) * metal::rsqrt(sum_sq + l2_eps);
      #pragma clang loop unroll(full)
      for (int i = 0; i < QK_PER_LANE; ++i) {
        dst[i] = values[i] * scale;
      }
    }
  }
  if (spec_mode == 0 && cend == end) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < QK_PER_LANE; ++i) {
      const int c = cbase + int(lane) * per_lane + i;
      for (int j = 0; j < hist_len; ++j) {
        state_base[(long)c * chan_stride + (long)j * col_stride] = T(history[i * MAX_HISTORY + j]);
      }
    }
  }
}

#define instantiate_kda_fused_prepare(type_name, T, DKVAL, DVVAL)                \
  template [[host_name("kda_fused_prepare_" #type_name "_dk" #DKVAL             \
                       "_dv" #DVVAL)]] [[kernel]] void                           \
  kda_fused_prepare<T, DKVAL, DVVAL>(                                            \
      device const T *qkv [[buffer(0)]], device const T *g_logits [[buffer(1)]], \
      device const T *beta_logits [[buffer(2)]],                                 \
      device const float *conv_w [[buffer(3)]],                                  \
      device T *conv_state_pool [[buffer(4)]],                                   \
      device const int *cu_seqlens [[buffer(5)]],                                \
      device const int *slot_mapping [[buffer(6)]],                              \
      device const float *A_log [[buffer(7)]],                                   \
      device const float *dt_bias [[buffer(8)]], device float *q [[buffer(9)]],  \
      device float *k [[buffer(10)]], device float *v [[buffer(11)]],            \
      device float *decay [[buffer(12)]], device float *beta_out [[buffer(13)]], \
      constant int &num_requests [[buffer(14)]], constant int &H [[buffer(15)]], \
      constant int &kernel_size [[buffer(16)]],                                  \
      constant int &load_initial [[buffer(17)]],                                 \
      constant int &qkv_stride [[buffer(18)]], constant int &g_stride [[buffer(19)]], \
      constant int &beta_stride [[buffer(20)]],                                  \
      constant int &conv_state_stride [[buffer(21)]],                            \
      constant int &chan_stride [[buffer(22)]], constant float &l2_eps [[buffer(23)]], \
      constant float &q_scale [[buffer(24)]],                                    \
      constant float &lower_bound [[buffer(25)]],                                \
      constant int &has_dt_bias [[buffer(26)]],                                  \
      constant int &col_stride [[buffer(27)]],                                   \
      device const int *num_accepted [[buffer(28)]],                             \
      constant int &spec_mode [[buffer(29)]],                                    \
      constant int &chunk_tokens [[buffer(30)]],                                 \
      uint2 gid [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_kda_fused_prepare(bfloat16, bf16, 128, 128)
instantiate_kda_fused_prepare(float16, half, 128, 128)
instantiate_kda_fused_prepare(float32, float, 128, 128)

// ---------------------------------------------------------------------------
// R value rows of the [DV, DK] state per simdgroup: q / k / decay rows are
// loaded once per timestep and reused across the R rows, and the grid
// shrinks R-fold (8192 single-row simdgroups per 64-head request thrash
// the per-core caches at prefill widths: R=1 89 ms/layer at 2500 tokens,
// R=2 44, R=4 5.7, R=8 4.8; w11-prefill/bench_kda_rows.log). Per-row arithmetic and
// the simd_sum reduction order are the single-row kernel's, so every R
// yields bit-identical y and state (decode stays bit-exact). The launcher
// picks R (VLLM_QC_KDA_RECUR_ROWS, see tk_launch.h kda_recur_rows_default).
template <int DK, int R>
kernel void kda_recur(device const float *q            [[buffer(0)]],  // [tokens, H*DK]
                      device const float *k            [[buffer(1)]],
                      device const float *v            [[buffer(2)]],  // [tokens, H*DV]
                      device const float *decay        [[buffer(3)]],  // [tokens, H*DK]
                      device const float *beta         [[buffer(4)]],  // [tokens, H]
                      device float       *state_pool   [[buffer(5)]],
                      device const int   *cu_seqlens   [[buffer(6)]],
                      device const int   *slot_mapping [[buffer(7)]],
                      device float       *y            [[buffer(8)]],  // [tokens, H*DV]
                      constant int       &num_requests [[buffer(9)]],
                      constant int       &H            [[buffer(10)]],
                      constant int       &DV           [[buffer(11)]],
                      constant int       &load_initial [[buffer(12)]],
                      constant int       &state_stride [[buffer(13)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint  lane [[thread_index_in_simdgroup]]) {
  static_assert(DK == 64 || DK == 128, "kda_recur supports Dk in {64, 128}");
  static_assert(R == 1 || R == 2 || R == 4 || R == 8, "kda_recur rows per simdgroup");
  constexpr int N_PER_T = DK / 32;
  const int req_idx = (int)gid.z / H;
  const int h = (int)gid.z % H;
  const int dv0 = (int)gid.x * R;
  const int dk0 = (int)lane * N_PER_T;
  if (req_idx >= num_requests || dv0 >= DV) { return; }

  const int seq_start = cu_seqlens[req_idx];
  const int seq_len = cu_seqlens[req_idx + 1] - seq_start;
  const long slot = slot_mapping[req_idx];
  device float *y_ = y + (long)seq_start * H * DV + h * DV;
  if (slot <= 0) {
    // Null block: zero output, pool untouched.
    if (lane == 0) {
      for (int t = 0; t < seq_len; ++t) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r) {
          y_[(long)t * H * DV + dv0 + r] = 0.0f;
        }
      }
    }
    return;
  }
  device float *state_ptr = state_pool + slot * (long)state_stride +
      ((long)h * DV + dv0) * DK;

  float state[R][N_PER_T];
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
      state[r][i] = (load_initial != 0) ? state_ptr[r * DK + dk0 + i] : 0.0f;
    }
  }

  device const float *q_ = q + (long)seq_start * H * DK + h * DK;
  device const float *k_ = k + (long)seq_start * H * DK + h * DK;
  device const float *v_ = v + (long)seq_start * H * DV + h * DV + dv0;
  device const float *d_ = decay + (long)seq_start * H * DK + h * DK;
  device const float *beta_ = beta + (long)seq_start * H;

  using FN = metal::vec<float, N_PER_T>;
  for (int t = 0; t < seq_len; ++t) {
    const FN kvec = ((device const FN*)(k_ + dk0))[0];
    const FN qvec = ((device const FN*)(q_ + dk0))[0];
    const FN dvec = ((device const FN*)(d_ + dk0))[0];
    float kv_mem[R];
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      kv_mem[r] = 0.0f;
      #pragma clang loop unroll(full)
      for (int i = 0; i < N_PER_T; ++i) {
        state[r][i] *= dvec[i];
        kv_mem[r] += state[r][i] * kvec[i];
      }
    }
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      kv_mem[r] = metal::simd_sum(kv_mem[r]);
    }
    const float b = beta_[h];
    float out[R];
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      const float delta = (v_[r] - kv_mem[r]) * b;
      out[r] = 0.0f;
      #pragma clang loop unroll(full)
      for (int i = 0; i < N_PER_T; ++i) {
        state[r][i] += kvec[i] * delta;
        out[r] += state[r][i] * qvec[i];
      }
    }
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      out[r] = metal::simd_sum(out[r]);
    }
    if (lane == 0) {
      #pragma clang loop unroll(full)
      for (int r = 0; r < R; ++r) {
        y_[dv0 + r] = out[r];
      }
    }

    q_ += H * DK;
    k_ += H * DK;
    d_ += H * DK;
    v_ += H * DV;
    y_ += H * DV;
    beta_ += H;
  }

  #pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
      state_ptr[r * DK + dk0 + i] = state[r][i];
    }
  }
}

#define instantiate_kda_recur(DKVAL, RVAL, NAME)                                 \
  template [[host_name(NAME)]] [[kernel]] void                                   \
  kda_recur<DKVAL, RVAL>(device const float *q [[buffer(0)]],                     \
                   device const float *k [[buffer(1)]],                           \
                   device const float *v [[buffer(2)]],                           \
                   device const float *decay [[buffer(3)]],                       \
                   device const float *beta [[buffer(4)]],                        \
                   device float *state_pool [[buffer(5)]],                        \
                   device const int *cu_seqlens [[buffer(6)]],                    \
                   device const int *slot_mapping [[buffer(7)]],                  \
                   device float *y [[buffer(8)]],                                 \
                   constant int &num_requests [[buffer(9)]],                      \
                   constant int &H [[buffer(10)]],                                \
                   constant int &DV [[buffer(11)]],                               \
                   constant int &load_initial [[buffer(12)]],                     \
                   constant int &state_stride [[buffer(13)]],                     \
                   uint3 gid [[threadgroup_position_in_grid]],                    \
                   uint lane [[thread_index_in_simdgroup]]);

instantiate_kda_recur(64, 1, "kda_recur_d64")
instantiate_kda_recur(128, 1, "kda_recur_d128")
instantiate_kda_recur(128, 2, "kda_recur_d128_r2")
instantiate_kda_recur(128, 4, "kda_recur_d128_r4")
instantiate_kda_recur(128, 8, "kda_recur_d128_r8")

// ---------------------------------------------------------------------------
// Speculative-verify variant of kda_recur (kda_recurrent_spec_native /
// fused_recurrent_kda with num_accepted_tokens): each request carries a row
// of num_spec+1 state slots. The initial state loads from
// slot_table[r, num_accepted[r]-1] (the checkpoint at the last accepted
// token) and after every timestep the running state is stored to
// slot_table[r, t] (slot > 0 only), so the next step can rewind to whichever
// draft position verification accepts. A null initial slot skips the request
// (zero y rows). Per-timestep math is kda_recur's verbatim.
// ---------------------------------------------------------------------------
template <int DK>
kernel void kda_recur_spec(device const float *q            [[buffer(0)]],
                           device const float *k            [[buffer(1)]],
                           device const float *v            [[buffer(2)]],
                           device const float *decay        [[buffer(3)]],
                           device const float *beta         [[buffer(4)]],
                           device float       *state_pool   [[buffer(5)]],
                           device const int   *cu_seqlens   [[buffer(6)]],
                           device const int   *slot_table   [[buffer(7)]],   // [R, table_stride]
                           device float       *y            [[buffer(8)]],
                           constant int       &num_requests [[buffer(9)]],
                           constant int       &H            [[buffer(10)]],
                           constant int       &DV           [[buffer(11)]],
                           constant int       &table_stride [[buffer(12)]],
                           constant int       &state_stride [[buffer(13)]],
                           device const int   *num_accepted [[buffer(14)]],
                           uint3 gid [[threadgroup_position_in_grid]],
                           uint  lane [[thread_index_in_simdgroup]]) {
  static_assert(DK == 64 || DK == 128, "kda_recur_spec supports Dk in {64, 128}");
  constexpr int N_PER_T = DK / 32;
  const int req_idx = (int)gid.z / H;
  const int h = (int)gid.z % H;
  const int dv_idx = (int)gid.x;
  const int dk0 = (int)lane * N_PER_T;
  if (req_idx >= num_requests || dv_idx >= DV) { return; }

  const int seq_start = cu_seqlens[req_idx];
  const int seq_len = cu_seqlens[req_idx + 1] - seq_start;
  if (seq_len <= 0) { return; }
  device const int *slots = slot_table + (long)req_idx * table_stride;
  device float *y_ = y + (long)seq_start * H * DV + h * DV;
  // num_accepted < 1 clamps to the first checkpoint (the torch reference's
  // clamp(min=1)); > table_stride is a host contract violation.
  int na = num_accepted[req_idx];
  if (na < 1) { na = 1; }
  if (na > table_stride) { return; }
  const long init_slot = slots[na - 1];
  if (init_slot <= 0) {
    if (lane == 0) {
      for (int t = 0; t < seq_len; ++t) {
        y_[(long)t * H * DV + dv_idx] = 0.0f;
      }
    }
    return;
  }
  device const float *init_ptr = state_pool + init_slot * (long)state_stride +
      ((long)h * DV + dv_idx) * DK;

  float state[N_PER_T];
  #pragma clang loop unroll(full)
  for (int i = 0; i < N_PER_T; ++i) {
    state[i] = init_ptr[dk0 + i];
  }

  device const float *q_ = q + (long)seq_start * H * DK + h * DK;
  device const float *k_ = k + (long)seq_start * H * DK + h * DK;
  device const float *v_ = v + (long)seq_start * H * DV + h * DV;
  device const float *d_ = decay + (long)seq_start * H * DK + h * DK;
  device const float *beta_ = beta + (long)seq_start * H;

  using FN = metal::vec<float, N_PER_T>;
  for (int t = 0; t < seq_len; ++t) {
    const FN kvec = ((device const FN*)(k_ + dk0))[0];
    const FN qvec = ((device const FN*)(q_ + dk0))[0];
    const FN dvec = ((device const FN*)(d_ + dk0))[0];
    float kv_mem = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
      state[i] *= dvec[i];
      kv_mem += state[i] * kvec[i];
    }
    kv_mem = metal::simd_sum(kv_mem);

    const float delta = (v_[dv_idx] - kv_mem) * beta_[h];

    float out = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
      state[i] += kvec[i] * delta;
      out += state[i] * qvec[i];
    }
    out = metal::simd_sum(out);
    if (lane == 0) {
      y_[dv_idx] = out;
    }

    if (t < table_stride) {
      const long ckpt_slot = slots[t];
      if (ckpt_slot > 0) {
        device float *ckpt = state_pool + ckpt_slot * (long)state_stride +
            ((long)h * DV + dv_idx) * DK;
        #pragma clang loop unroll(full)
        for (int i = 0; i < N_PER_T; ++i) {
          ckpt[dk0 + i] = state[i];
        }
      }
    }

    q_ += H * DK;
    k_ += H * DK;
    d_ += H * DK;
    v_ += H * DV;
    y_ += H * DV;
    beta_ += H;
  }
}

#define instantiate_kda_recur_spec(DKVAL)                                        \
  template [[host_name("kda_recur_spec_d" #DKVAL)]] [[kernel]] void              \
  kda_recur_spec<DKVAL>(device const float *q [[buffer(0)]],                      \
                   device const float *k [[buffer(1)]],                           \
                   device const float *v [[buffer(2)]],                           \
                   device const float *decay [[buffer(3)]],                       \
                   device const float *beta [[buffer(4)]],                        \
                   device float *state_pool [[buffer(5)]],                        \
                   device const int *cu_seqlens [[buffer(6)]],                    \
                   device const int *slot_table [[buffer(7)]],                    \
                   device float *y [[buffer(8)]],                                 \
                   constant int &num_requests [[buffer(9)]],                      \
                   constant int &H [[buffer(10)]],                                \
                   constant int &DV [[buffer(11)]],                               \
                   constant int &table_stride [[buffer(12)]],                     \
                   constant int &state_stride [[buffer(13)]],                     \
                   device const int *num_accepted [[buffer(14)]],                 \
                   uint3 gid [[threadgroup_position_in_grid]],                    \
                   uint lane [[thread_index_in_simdgroup]]);

instantiate_kda_recur_spec(64)
instantiate_kda_recur_spec(128)

// ---------------------------------------------------------------------------
// rmsnorm(y) * weight * sigmoid(z): FusedRMSNormGated(activation="sigmoid").
// y fp32 [rows, D] (row = token*H + h); z in T at token*z_stride + h*D.
// ---------------------------------------------------------------------------
template <typename T, int D>
kernel void kda_gated_rmsnorm_f32(
    device const float *y [[buffer(0)]],
    device const T *z [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device T *out [[buffer(3)]],
    constant int &rows [[buffer(4)]],
    constant int &H [[buffer(5)]],
    constant int &z_stride [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int PER_LANE = D / 32;
  if (int(row) >= rows) {
    return;
  }
  const int token = int(row) / H;
  const int h = int(row) % H;
  const long y_off = (long)row * D;
  const long z_off = (long)token * z_stride + (long)h * D;
  float values[PER_LANE];
  float sum_sq = 0.0f;
  #pragma clang loop unroll(full)
  for (int i = 0; i < PER_LANE; ++i) {
    const int d = int(lane) * PER_LANE + i;
    // The reference rounds the recurrence output to the activation dtype
    // before the norm (core_attn_out is a T tensor); match it.
    values[i] = float(T(y[y_off + d]));
    sum_sq += values[i] * values[i];
  }
  sum_sq = metal::simd_sum(sum_sq);
  const float inv_rms = metal::rsqrt(sum_sq / float(D) + eps);
  #pragma clang loop unroll(full)
  for (int i = 0; i < PER_LANE; ++i) {
    const int d = int(lane) * PER_LANE + i;
    const float gate = float(z[z_off + d]);
    const float sig = 1.0f / (1.0f + metal::exp(-gate));
    out[y_off + d] = T(values[i] * inv_rms * float(weight[d]) * sig);
  }
}

#define instantiate_kda_gated_rmsnorm_f32(type_name, T, DVAL)                    \
  template [[host_name("kda_gated_rmsnorm_f32_" #type_name "_d" #DVAL)]]         \
  [[kernel]] void kda_gated_rmsnorm_f32<T, DVAL>(                                \
      device const float *y [[buffer(0)]], device const T *z [[buffer(1)]],      \
      device const T *weight [[buffer(2)]], device T *out [[buffer(3)]],         \
      constant int &rows [[buffer(4)]], constant int &H [[buffer(5)]],           \
      constant int &z_stride [[buffer(6)]], constant float &eps [[buffer(7)]],   \
      uint row [[threadgroup_position_in_grid]],                                 \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_kda_gated_rmsnorm_f32(bfloat16, bf16, 128)
instantiate_kda_gated_rmsnorm_f32(float16, half, 128)
instantiate_kda_gated_rmsnorm_f32(float32, float, 128)

}  // namespace mittens
