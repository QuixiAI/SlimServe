// Fused Qwen3.5 gated-deltanet (GDN) decode step for Metal.
//
// Replaces the torch-native MPS core in
// vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py for the
// decode (S = 1) and spec-verify (S <= MAX_S positions per sequence) cases.
// The torch path is the verification oracle: _causal_conv1d_native +
// _gdn_recurrent_scan_native (decode) and _gdn_spec_state_step_native
// (verify). Its ~15 small MPS ops per position per layer become two
// dispatches per layer, with the per-position loop INSIDE the kernel.
//
// Two kernels, encoded back to back on the same command buffer:
//
//   qwen_gdn_conv_step  one thread per (sequence, conv channel): reads the
//                       conv window [off, off+width-1) of the slot, the S new
//                       inputs, writes act(conv) to an fp32 scratch
//                       (N, S, conv_dim) and rolls the window in place
//                       ([win[off+1:], x...] from column 0). Each channel is
//                       owned by exactly one thread, so read-then-write on
//                       the same slot columns is race-free.
//   qwen_gdn_scan_step  one simdgroup per (sequence, v-head, dv row): the 32
//                       lanes hold the 128-wide fp32 state row in registers
//                       (4 per lane); l2norm(q, k) + query scale, gating
//                       g = -exp(A_log) * softplus(a + dt_bias),
//                       beta = sigmoid(b), S *= exp(g), delta = (v - S.k) *
//                       beta, S += delta k, o = S.q, looped over positions
//                       with the running state stored to store_slots[n, t]
//                       after every position (the spec rollback contract;
//                       slots <= 0 skip the store). The initial state comes
//                       from resume_slot[n]; sequences with resume_slot <= 0
//                       are NULL entries: zero output, no state traffic.
//
// GQA head pairing: tiled (ggml layout, i_k = i_hv % Hk) or HF grouped
// (i_k = i_hv / (Hv / Hk)), selected per call.
//
// token_map[n * S + t] is the row of x / a / b / out for position t of
// sequence n (the spec path packs sequences through spec_token_indx; the
// decode path passes an identity map).
#include <metal_stdlib>

namespace mittens {

namespace gdn_step {

constexpr constant int MAX_WIDTH = 4;   // conv kernel size (Qwen3.5: 4)
constexpr constant int MAX_S = 16;      // positions per sequence per call
constexpr constant int DK = 128;        // head_k_dim (32 lanes x 4)
constexpr constant int CONV_TG = 256;
constexpr constant int SCAN_TG = 256;   // 8 simdgroups = 8 dv rows

// log1p(x) without the library call (Kahan), exact to fp32 rounding for the
// small values softplus feeds it; matches torch's softplus better than
// log(1 + e) when e is tiny.
inline float log1p_f(float x) {
    const float u = 1.0f + x;
    const float d = u - 1.0f;
    return (d == 0.0f) ? x : metal::precise::log(u) * x / d;
}

inline float softplus_f(float x) {
    // torch.nn.functional.softplus(beta=1, threshold=20)
    return (x > 20.0f) ? x : log1p_f(metal::precise::exp(x));
}

}  // namespace gdn_step

template <typename T, typename TC>
kernel void qwen_gdn_conv_step(
    device const T*     x            [[buffer(0)]],   // rows (x_rs), conv_dim wide
    device TC*          conv_state   [[buffer(1)]],   // (slot, chan, col) strided
    device const float* weight       [[buffer(2)]],   // (conv_dim, width)
    device const float* bias         [[buffer(3)]],   // (conv_dim) when has_bias
    device float*       conved       [[buffer(4)]],   // (N, S, conv_dim) fp32
    device const int*   token_map    [[buffer(5)]],   // (N * S) -> x row
    device const int*   conv_slot    [[buffer(6)]],   // (N); <= 0 skips
    device const int*   num_accepted [[buffer(7)]],   // (N) when use_accepted
    constant int&   conv_dim     [[buffer(8)]],
    constant int&   width        [[buffer(9)]],
    constant int&   S            [[buffer(10)]],
    constant long&  x_rs         [[buffer(11)]],      // x row stride (elements)
    constant long&  cs_slot      [[buffer(12)]],      // conv_state strides
    constant long&  cs_chan      [[buffer(13)]],
    constant long&  cs_col       [[buffer(14)]],
    constant int&   has_bias     [[buffer(15)]],
    constant int&   act_silu     [[buffer(16)]],
    constant int&   use_accepted [[buffer(17)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
    using namespace gdn_step;
    const int c = (int)(tgid.x * CONV_TG + tid);
    const int n = (int)tgid.y;
    if (c >= conv_dim) return;
    const int slot = conv_slot[n];
    if (slot <= 0) return;  // NULL entry: the scan kernel zeroes the output
    // S and width index the fixed local windows below. The host checks
    // 1 <= S <= MAX_S and 2 <= width <= MAX_WIDTH; a violation degrades to
    // a no-op here instead of writing past the stack arrays.
    if (S < 1 || S > MAX_S || width < 1 || width > MAX_WIDTH) return;
    const int off = use_accepted ? (num_accepted[n] - 1) : 0;
    if (off < 0) return;  // num_accepted < 1: no history window to read
    const int w1 = width - 1;

    device TC* st = conv_state + (long)slot * cs_slot + (long)c * cs_chan;

    // padded = [window(width-1), x(S)] -- read everything before writing.
    float padded[MAX_WIDTH - 1 + MAX_S];
    for (int j = 0; j < w1; ++j) {
        padded[j] = float(st[(long)(off + j) * cs_col]);
    }
    for (int t = 0; t < S; ++t) {
        const long row = (long)token_map[n * S + t];
        padded[w1 + t] = float(x[row * x_rs + c]);
    }

    float wt[MAX_WIDTH];
    for (int w = 0; w < width; ++w) wt[w] = weight[(long)c * width + w];
    const float bs = has_bias ? bias[c] : 0.0f;

    device float* o = conved + ((long)n * S) * conv_dim + c;
    for (int t = 0; t < S; ++t) {
        float acc = 0.0f;
        for (int w = 0; w < width; ++w) acc += wt[w] * padded[t + w];
        acc += bs;
        if (act_silu) acc = acc / (1.0f + metal::precise::exp(-acc));
        o[(long)t * conv_dim] = acc;
    }

    // Roll: new window = [padded[1 : w1], x(S)] written from column 0.
    for (int j = 0; j < w1 - 1 + S; ++j) {
        st[(long)j * cs_col] = TC(padded[1 + j]);
    }
}

template <typename T>
kernel void qwen_gdn_scan_step(
    device const float* conved       [[buffer(0)]],   // (N, S, conv_dim) fp32
    device const T*     a_in         [[buffer(1)]],   // rows (ab_rs), Hv wide
    device const T*     b_in         [[buffer(2)]],
    device const float* A_log        [[buffer(3)]],   // (Hv)
    device const float* dt_bias      [[buffer(4)]],   // (Hv)
    device float*       ssm_state    [[buffer(5)]],   // (slot, Hv, Dv, DK)
    device const int*   token_map    [[buffer(6)]],   // (N * S)
    device const int*   resume_slot  [[buffer(7)]],   // (N); <= 0 => NULL
    device const int*   store_slots  [[buffer(8)]],   // (N, >= S) rows (ss_rs)
    device T*           out          [[buffer(9)]],   // rows (out_rs), (Hv, Dv)
    constant int&   S         [[buffer(10)]],
    constant int&   Hk        [[buffer(11)]],
    constant int&   Hv        [[buffer(12)]],
    constant int&   Dv        [[buffer(13)]],
    constant int&   conv_dim  [[buffer(14)]],
    constant int&   tiled     [[buffer(15)]],
    constant long&  ab_rs     [[buffer(16)]],
    constant long&  st_slot   [[buffer(17)]],         // ssm_state slot stride
    constant long&  ss_rs     [[buffer(18)]],         // store_slots row stride
    constant long&  out_rs    [[buffer(19)]],
    constant float& scale     [[buffer(20)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  lane    [[thread_index_in_simdgroup]],
    uint  simd_id [[simdgroup_index_in_threadgroup]]) {
    using namespace gdn_step;
    constexpr int SIMDS = SCAN_TG / 32;
    const int dv = (int)(tgid.x * SIMDS + simd_id);
    const int hv = (int)tgid.y;
    const int n = (int)tgid.z;
    if (dv >= Dv) return;

    const int rs = resume_slot[n];
    if (rs <= 0) {
        if (lane == 0) {
            for (int t = 0; t < S; ++t) {
                const long row = (long)token_map[n * S + t];
                out[row * out_rs + (long)hv * Dv + dv] = T(0.0f);
            }
        }
        return;
    }

    const int hk = tiled ? (hv % Hk) : (hv / (Hv / Hk));
    const int key_dim = Hk * DK;
    const long head_off = ((long)hv * Dv + dv) * DK + (long)lane * 4;

    float4 st = *((device const float4*)(ssm_state + (long)rs * st_slot + head_off));

    const float neg_exp_a = -metal::precise::exp(A_log[hv]);
    const float dtb = dt_bias[hv];

    for (int t = 0; t < S; ++t) {
        device const float* row = conved + ((long)n * S + t) * conv_dim;
        float4 q = *((device const float4*)(row + hk * DK + lane * 4));
        float4 k = *((device const float4*)(row + key_dim + hk * DK + lane * 4));
        const float v = row[2 * key_dim + hv * Dv + dv];

        const float qs = metal::simd_sum(metal::dot(q, q));
        const float ks = metal::simd_sum(metal::dot(k, k));
        q = q * metal::precise::rsqrt(qs + 1e-6f) * scale;
        k = k * metal::precise::rsqrt(ks + 1e-6f);

        const long trow = (long)token_map[n * S + t];
        const float a = float(a_in[trow * ab_rs + hv]);
        const float b = float(b_in[trow * ab_rs + hv]);
        const float g = neg_exp_a * softplus_f(a + dtb);
        const float beta = 1.0f / (1.0f + metal::precise::exp(-b));
        const float decay = metal::precise::exp(g);

        st *= decay;
        const float kv = metal::simd_sum(metal::dot(st, k));
        const float delta = (v - kv) * beta;
        st += k * delta;
        const float o = metal::simd_sum(metal::dot(st, q));
        if (lane == 0) {
            out[trow * out_rs + (long)hv * Dv + dv] = T(o);
        }

        const int ss = store_slots[(long)n * ss_rs + t];
        if (ss > 0) {
            *((device float4*)(ssm_state + (long)ss * st_slot + head_off)) = st;
        }
    }
}

#define instantiate_qwen_gdn_conv_step(tname, T, cname, TC)                     \
  template [[host_name("qwen_gdn_conv_step_" #tname "_cs" #cname)]] [[kernel]] \
  void qwen_gdn_conv_step<T, TC>(                                               \
      device const T* x [[buffer(0)]], device TC* conv_state [[buffer(1)]],     \
      device const float* weight [[buffer(2)]],                                 \
      device const float* bias [[buffer(3)]],                                   \
      device float* conved [[buffer(4)]],                                       \
      device const int* token_map [[buffer(5)]],                                \
      device const int* conv_slot [[buffer(6)]],                                \
      device const int* num_accepted [[buffer(7)]],                             \
      constant int& conv_dim [[buffer(8)]], constant int& width [[buffer(9)]],  \
      constant int& S [[buffer(10)]], constant long& x_rs [[buffer(11)]],       \
      constant long& cs_slot [[buffer(12)]], constant long& cs_chan [[buffer(13)]], \
      constant long& cs_col [[buffer(14)]], constant int& has_bias [[buffer(15)]], \
      constant int& act_silu [[buffer(16)]],                                    \
      constant int& use_accepted [[buffer(17)]],                                \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint tid [[thread_index_in_threadgroup]]);

#define instantiate_qwen_gdn_scan_step(tname, T)                                \
  template [[host_name("qwen_gdn_scan_step_" #tname)]] [[kernel]] void          \
  qwen_gdn_scan_step<T>(                                                        \
      device const float* conved [[buffer(0)]],                                 \
      device const T* a_in [[buffer(1)]], device const T* b_in [[buffer(2)]],   \
      device const float* A_log [[buffer(3)]],                                  \
      device const float* dt_bias [[buffer(4)]],                                \
      device float* ssm_state [[buffer(5)]],                                    \
      device const int* token_map [[buffer(6)]],                                \
      device const int* resume_slot [[buffer(7)]],                              \
      device const int* store_slots [[buffer(8)]], device T* out [[buffer(9)]], \
      constant int& S [[buffer(10)]], constant int& Hk [[buffer(11)]],          \
      constant int& Hv [[buffer(12)]], constant int& Dv [[buffer(13)]],         \
      constant int& conv_dim [[buffer(14)]], constant int& tiled [[buffer(15)]], \
      constant long& ab_rs [[buffer(16)]], constant long& st_slot [[buffer(17)]], \
      constant long& ss_rs [[buffer(18)]], constant long& out_rs [[buffer(19)]], \
      constant float& scale [[buffer(20)]],                                     \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd_id [[simdgroup_index_in_threadgroup]]);

instantiate_qwen_gdn_conv_step(float32, float, float32, float)
instantiate_qwen_gdn_conv_step(float16, half, float16, half)
instantiate_qwen_gdn_conv_step(float16, half, float32, float)
instantiate_qwen_gdn_conv_step(bfloat16, bfloat, bfloat16, bfloat)
instantiate_qwen_gdn_conv_step(bfloat16, bfloat, float32, float)

instantiate_qwen_gdn_scan_step(float32, float)
instantiate_qwen_gdn_scan_step(float16, half)
instantiate_qwen_gdn_scan_step(bfloat16, bfloat)

// Gated RMS norm (RMSNormGated, norm_before_gate, silu gate) over the GDN
// core output: out[t, h, :] = x * rsqrt(mean(x^2) + eps) * w * silu(z).
// One simdgroup per (token, head) row; replaces ~11 torch ops per layer.
// x is (tokens, Hv, D) contiguous, z is the (tokens, Hv, D) view carved out
// of the fused qkvz projection (token stride z_ts, head stride D).
template <typename T>
kernel void qwen_gdn_gated_norm(
    device const T*     x     [[buffer(0)]],
    device const T*     z     [[buffer(1)]],
    device const float* w     [[buffer(2)]],   // (D)
    device T*           out   [[buffer(3)]],   // (tokens, Hv, D)
    constant int&   rows  [[buffer(4)]],       // tokens * Hv
    constant int&   Hv    [[buffer(5)]],
    constant int&   D     [[buffer(6)]],
    constant long&  z_ts  [[buffer(7)]],
    constant float& eps   [[buffer(8)]],
    uint3 tgid    [[threadgroup_position_in_grid]],
    uint  lane    [[thread_index_in_simdgroup]],
    uint  simd_id [[simdgroup_index_in_threadgroup]]) {
    const int r = (int)(tgid.x * 8 + simd_id);
    if (r >= rows) return;
    const int tok = r / Hv;
    const int h = r % Hv;
    device const T* xr = x + (long)r * D;
    device const T* zr = z + (long)tok * z_ts + (long)h * D;
    device T* o = out + (long)r * D;

    float ss = 0.0f;
    for (int i = (int)lane * 4; i < D; i += 128) {
        const float4 v = float4(*((device const metal::vec<T, 4>*)(xr + i)));
        ss += metal::dot(v, v);
    }
    ss = metal::simd_sum(ss);
    const float rstd = metal::precise::rsqrt(ss / (float)D + eps);
    for (int i = (int)lane * 4; i < D; i += 128) {
        const float4 v = float4(*((device const metal::vec<T, 4>*)(xr + i)));
        const float4 g = float4(*((device const metal::vec<T, 4>*)(zr + i)));
        const float4 wv = *((device const float4*)(w + i));
        const float4 sg = g / (1.0f + metal::precise::exp(-g));
        const float4 y = v * rstd * wv * sg;
        *((device metal::vec<T, 4>*)(o + i)) = metal::vec<T, 4>(y);
    }
}

#define instantiate_qwen_gdn_gated_norm(tname, T)                               \
  template [[host_name("qwen_gdn_gated_norm_" #tname)]] [[kernel]] void         \
  qwen_gdn_gated_norm<T>(                                                       \
      device const T* x [[buffer(0)]], device const T* z [[buffer(1)]],         \
      device const float* w [[buffer(2)]], device T* out [[buffer(3)]],         \
      constant int& rows [[buffer(4)]], constant int& Hv [[buffer(5)]],         \
      constant int& D [[buffer(6)]], constant long& z_ts [[buffer(7)]],         \
      constant float& eps [[buffer(8)]],                                        \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd_id [[simdgroup_index_in_threadgroup]]);

instantiate_qwen_gdn_gated_norm(float32, float)
instantiate_qwen_gdn_gated_norm(float16, half)
instantiate_qwen_gdn_gated_norm(bfloat16, bfloat)

}  // namespace mittens
