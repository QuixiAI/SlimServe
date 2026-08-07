// Shared, framework-agnostic launch logic for the ThunderMittens kernels.
//
// This header is the SINGLE SOURCE OF TRUTH for each kernel's host ABI — the kernel
// name, the buffer index mapping, the scalar parameters, and the grid/threadgroup
// geometry. Both backends drive it through a small "encoder" adapter:
//   - MLX  (<kernel>.cpp via MLXEncoder in tk_mlx_launch.h): binds with set_input_array
//     so MLX's residency/scheduling bookkeeping is preserved.
//   - Torch (tk_torch/torch_kernels.mm via TorchEncoder): binds the MTLBuffer directly.
//
// An adapter `E` must provide:
//   typedefs E::in_t, E::out_t                      (input / output buffer handle types)
//   void pipeline(const std::string& kernel_name)   (set the compute pipeline state)
//   void in(E::in_t, int index)                     (bind an input buffer)
//   void out(E::out_t, int index)                   (bind the output buffer)
//   template<class T> void bytes(const T&, int idx) (set inline scalar bytes)
//   void dispatch(int gx,int gy,int gz, int tx,int ty,int tz)  (dispatch threadgroups)
//
// Pure C++: depends on neither MLX nor Metal, so it compiles in both .cpp and .mm.

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "base_q_descriptor.h"

namespace tk {

// ----- kernel-name helpers (must match the [[host_name(...)]] in <kernel>.metal) -----
inline std::string layernorm_kernel_name(int D) { return "layernorm_" + std::to_string(D); }
inline std::string attn_fwd_kernel_name(int D) { return "attn_fwd_" + std::to_string(D); }
inline std::string add_rt_kernel_name(const std::string& t) { return "add_rt_" + t; }
inline std::string matmul_custom_kernel_name(const std::string& t) { return "matmul_custom_" + t; }
inline std::string rms_norm_kernel_name(int D) { return "rms_norm_" + std::to_string(D); }
inline std::string mean_pool_rms_l2_kernel_name(int D) { return "mean_pool_rms_l2_" + std::to_string(D); }
inline std::string rms_norm_add_kernel_name(int D) { return "rms_norm_add_" + std::to_string(D); }
inline std::string rms_norm_residual_next_kernel_name(int D) {
  return "rms_norm_residual_next_" + std::to_string(D);
}
inline std::string layernorm_add_kernel_name(int D) { return "layernorm_add_" + std::to_string(D); }
inline std::string rope_kv_insert_kernel_name(const std::string& t, int D) {
  return "rope_kv_insert_" + t + "_" + std::to_string(D);
}
inline std::string rope_kv_insert_norm_kernel_name(const std::string& t, int D) {
  return "rope_kv_insert_norm_" + t + "_" + std::to_string(D);
}
inline std::string rope_q_kernel_name(const std::string& t, int D) {
  return "rope_q_" + t + "_" + std::to_string(D);
}
inline std::string rms_norm_add_fp8_kernel_name(int D) { return "rms_norm_add_fp8_" + std::to_string(D); }
inline std::string rms_norm_add_fp8_dyn_kernel_name(int D) { return "rms_norm_add_fp8_dyn_" + std::to_string(D); }
inline std::string rms_norm_add_int8_dyn_kernel_name(int D) { return "rms_norm_add_int8_dyn_" + std::to_string(D); }
inline std::string layernorm_add_fp8_kernel_name(int D) { return "layernorm_add_fp8_" + std::to_string(D); }
inline std::string layernorm_add_fp8_dyn_kernel_name(int D) { return "layernorm_add_fp8_dyn_" + std::to_string(D); }
inline std::string layernorm_add_int8_dyn_kernel_name(int D) { return "layernorm_add_int8_dyn_" + std::to_string(D); }
inline std::string rms_norm_add_per_block_kernel_name(int D, bool i8) {
  return std::string("rms_norm_add_per_block_") + (i8 ? "int8_" : "fp8_") + std::to_string(D);
}
inline std::string layernorm_add_per_block_kernel_name(int D, bool i8) {
  return std::string("layernorm_add_per_block_") + (i8 ? "int8_" : "fp8_") + std::to_string(D);
}
inline std::string argmax_kernel_name(const std::string& t) { return "argmax_" + t; }
inline std::string moe_route_topk_kernel_name(const std::string& t) { return "moe_route_topk_" + t; }
inline std::string moe_finalize_kernel_name(const std::string& t) { return "moe_finalize_" + t; }
inline std::string moe_grouped_gemm_kernel_name(const std::string& t) { return "moe_grouped_gemm_" + t; }
inline std::string moe_grouped_gemm_rect_kernel_name(const std::string& t) { return "moe_grouped_gemm_rect_" + t; }
inline std::string moe_grouped_gemm_swiglu_kernel_name(const std::string& t) { return "moe_grouped_gemm_swiglu_" + t; }
inline std::string moe_route_grouped_kernel_name(const std::string& t) { return "moe_route_grouped_" + t; }
inline std::string qk_norm_rope_kernel_name(int D) { return "qk_norm_rope_" + std::to_string(D); }
inline std::string qk_norm_rope_positioned_kernel_name(int D) {
  return "qk_norm_rope_positioned_" + std::to_string(D);
}
inline std::string qk_norm_rope_kv_f16_kernel_name(int D) {
  return "qk_norm_rope_kv_f16_" + std::to_string(D);
}
inline std::string selective_scan_kernel_name(const std::string& variant, const std::string& t) {
  return "selective_scan_" + variant + "_" + t;
}
inline std::string gdn_recur_kernel_name(const std::string& t, int Dk) {
  return "gdn_recur_" + t + "_d" + std::to_string(Dk);
}
inline std::string gdn_short_conv_kernel_name(const std::string& t) {
  return "gdn_short_conv_" + t;
}
inline std::string gdn_qkv_prepare_kernel_name(
    const std::string& t, int Dk, int Dv) {
  return "gdn_qkv_prepare_" + t + "_dk" + std::to_string(Dk) +
         "_dv" + std::to_string(Dv);
}
inline std::string gdn_gate_beta_kernel_name(const std::string& t) {
  return "gdn_gate_beta_" + t;
}
inline std::string gdn_gated_rmsnorm_kernel_name(const std::string& t, int D) {
  return "gdn_gated_rmsnorm_" + t + "_d" + std::to_string(D);
}
inline std::string moe_grouped_gemm_rect_q_kernel_name(const std::string& fmt) { return "moe_grouped_gemm_rect_q_" + fmt; }
inline std::string moe_grouped_gemm_swiglu_q_kernel_name(const std::string& fmt) { return "moe_grouped_gemm_swiglu_q_" + fmt; }
inline std::string sample_categorical_kernel_name(const std::string& t) { return "sample_categorical_" + t; }
inline std::string top_k_sample_kernel_name(const std::string& t) { return "top_k_sample_" + t; }
inline std::string top_p_sample_kernel_name(const std::string& t) { return "top_p_sample_" + t; }
inline std::string apply_penalty_kernel_name(const std::string& t) { return "apply_penalty_" + t; }
inline std::string quant_tensor_absmax_kernel_name(const std::string& t) { return "quant_tensor_absmax_" + t; }
inline std::string quant_tensor_encode_fp8_kernel_name(const std::string& t) { return "quant_tensor_encode_fp8_" + t; }
inline std::string quant_tensor_encode_int8_kernel_name(const std::string& t) { return "quant_tensor_encode_int8_" + t; }
inline std::string quantize_per_token_fp8_kernel_name(const std::string& t) { return "quantize_per_token_fp8_" + t; }
inline std::string quantize_per_token_int8_kernel_name(const std::string& t) { return "quantize_per_token_int8_" + t; }
inline std::string weight_quant_ternary_kernel_name(const std::string& t) { return "weight_quant_ternary_" + t; }
inline std::string softmax_kernel_name(int D) { return "softmax_" + std::to_string(D); }
inline std::string rotary_kernel_name(int D) { return "rotary_" + std::to_string(D); }
inline std::string rotary_interleaved_kernel_name(int D) { return "rotary_interleaved_" + std::to_string(D); }
inline std::string rotary_positioned_kernel_name(int D) {
  return "rotary_positioned_" + std::to_string(D);
}
inline std::string mrope_positioned_kernel_name(int D) {
  return "mrope_positioned_" + std::to_string(D);
}
inline std::string mla_q_norm_rope_kernel_name(int D) { return "mla_q_norm_rope_" + std::to_string(D); }
inline std::string mla_kv_insert_kernel_name(int L) { return "mla_kv_insert_" + std::to_string(L); }
inline std::string mla_decode_kernel_name(int L, int R) {
  return "mla_decode_" + std::to_string(L) + "_" + std::to_string(R);
}
inline std::string gelu_kernel_name(int D) { return "gelu_" + std::to_string(D); }
inline std::string glu_kernel_name(const std::string& mode, const std::string& t) { return "glu_" + mode + "_" + t; }
inline std::string hadamard_kernel_name(const std::string& t, int D) {
  return "hadamard_" + t + "_" + std::to_string(D);
}
inline std::string kv_cache_kernel_name(const std::string& op, const std::string& t) {
  return "kv_cache_" + op + "_" + t;
}
inline std::string paged_attention_kernel_name(const std::string& t, int D) {
  return "paged_attention_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_gqa_staged_kernel_name(const std::string& t, int D) {
  return "paged_attention_gqa_staged_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_xcache_kernel_name(const std::string& t, int D) {
  return "paged_attention_xcache_" + t + "_" + std::to_string(D);
}
inline std::string kv_cache_scatter_fp8_kernel_name(const std::string& t) { return "kv_cache_scatter_fp8_" + t; }
inline std::string paged_attention_fp8_kernel_name(const std::string& t, int D, int fmt) {
  return "paged_attention_fp8_" + std::string(fmt == 1 ? "e5m2_" : "e4m3_") + t + "_" +
         std::to_string(D);
}
inline std::string paged_attention_partition_kernel_name(const std::string& t, int D) {
  return "paged_attention_partition_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_partition_fp8_kernel_name(const std::string& t, int D, int fmt) {
  return "paged_attention_partition_fp8_" +
         std::string(fmt == 1 ? "e5m2_" : "e4m3_") + t + "_" + std::to_string(D);
}
inline std::string paged_attention_reduce_kernel_name(const std::string& t, int D) {
  return "paged_attention_reduce_" + t + "_" + std::to_string(D);
}
inline std::string attn_causal_kernel_name(int D) { return "attn_causal_" + std::to_string(D); }
inline std::string flux_gelu_kernel_name(const std::string& t) { return "flux_gelu_" + t; }
inline std::string flux_gate_kernel_name(const std::string& t) { return "flux_gate_" + t; }
inline std::string gemm_staged_kernel_name(const std::string& t) { return "gemm_staged_" + t; }
inline std::string attn_multiwarp_kernel_name(int D) { return "attn_multiwarp_" + std::to_string(D); }
inline std::string attn_q_kernel_name(const std::string& fmt, int D, bool causal) {
  return std::string("attn_q_") + (causal ? "causal_" : "") + fmt + "_" + std::to_string(D);
}
inline std::string linear_attn_kernel_name(int D) { return "linear_attn_" + std::to_string(D); }
inline std::string hedgehog_kernel_name(int D) { return "hedgehog_" + std::to_string(D); }
inline std::string lin_attn_causal_kernel_name(int D) { return "lin_attn_causal_" + std::to_string(D); }
inline std::string mamba2_kernel_name(int D) { return "mamba2_" + std::to_string(D); }
inline std::string lin_attn_decay_kernel_name(int D) { return "lin_attn_decay_" + std::to_string(D); }
inline std::string based_kernel_name(int DQK, int DVO) {
  return "based_" + std::to_string(DQK) + "_" + std::to_string(DVO);
}
inline std::string cmplx_matmul_kernel_name(const std::string& t) { return "cmplx_matmul_" + t; }
inline std::string fftconv_kernel_name(int S) { return "fftconv_" + std::to_string(S); }
inline std::string qgemm_kernel_name(const std::string& fmt) { return "qgemm_" + fmt; }
inline std::string qgemv_kernel_name(const std::string& fmt) { return "qgemv_" + fmt; }
inline std::string qflux_gelu_kernel_name(const std::string& fmt) { return "qflux_gelu_" + fmt; }
inline std::string qgemm_frag_kernel_name(const std::string& fmt) { return "qgemm_frag_" + fmt; }
inline std::string qgemm_actorder_kernel_name(const std::string& fmt) { return "qgemm_actorder_" + fmt; }
inline std::string base_q_kernel_name(
    const std::string& operation, const std::string& type_name) {
  return "base_q" + operation + "_" + type_name;
}

// ----- LayerNorm: x@0 w@1 b@2 -> o@3 ; M@4(u32) eps@5(f32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_layernorm(E& e, typename E::in_t x, typename E::in_t w, typename E::in_t b,
                      typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline(layernorm_kernel_name(D));
  e.in(x, 0); e.in(w, 1); e.in(b, 2); e.out(o, 3);
  e.bytes(M, 4); e.bytes(eps, 5);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_dyn(E& e, typename E::in_t x, typename E::in_t w, typename E::in_t b,
                          typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline("layernorm_dyn_bfloat16");
  e.in(x, 0); e.in(w, 1); e.in(b, 2); e.out(o, 3);
  e.bytes(M, 4); e.bytes(eps, 5); e.bytes(D, 6);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_bwd_dx(E& e, typename E::in_t x, typename E::in_t w, typename E::in_t dy,
                             typename E::in_t mean, typename E::in_t rstd, typename E::out_t dx,
                             int rows, int D, const std::string& type_name) {
  e.pipeline("layernorm_bwd_dx_" + type_name);
  e.in(x, 0); e.in(w, 1); e.in(dy, 2); e.in(mean, 3); e.in(rstd, 4); e.out(dx, 5);
  e.bytes(D, 6);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
// fused LayerNorm backward: mean/rstd in-kernel + dX + atomic dweight & dbias (both (D,) fp32 zeroed).
template <class E>
void launch_layernorm_bwd_fused(E& e, typename E::in_t x, typename E::in_t w, typename E::in_t dy,
                                typename E::out_t dx, typename E::out_t dweight,
                                typename E::out_t dbias, int rows, int D, float eps,
                                const std::string& type_name) {
  e.pipeline("layernorm_bwd_fused_" + type_name);
  e.in(x, 0); e.in(w, 1); e.in(dy, 2); e.out(dx, 3); e.out(dweight, 4); e.out(dbias, 5);
  e.bytes(D, 6); e.bytes(eps, 7);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- add_rt: x@0 y@1 -> out@2 ; rows@3(i32) cols@4(i32) ; flat, one thread per 4 elements
// (vec4). The original 8x8-register-tile version measured 0.34x of mx add (64 elements per
// 32-thread group, 2-element gathers); the rt load/add/store path stays covered by the Xcode
// primitive tests and every MMA kernel. -----
template <class E>
void launch_add_rt(E& e, typename E::in_t x, typename E::in_t y, typename E::out_t o,
                   int rows, int cols, const std::string& type_name) {
  e.pipeline(add_rt_kernel_name(type_name));
  e.in(x, 0); e.in(y, 1); e.out(o, 2);
  e.bytes(rows, 3); e.bytes(cols, 4);
  const long n = (long)rows * cols;
  const long nthreads = (n + 7) / 8;
  e.dispatch(static_cast<int>((nthreads + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- matmul_custom: D(out)@0 A@1 B@2 ; N@3 K@4 M@5 (i32) ; grid (M/32, N/32, 1) -----
// A is (N,K), B is (K,M), out is (N,M).
template <class E>
void launch_matmul_custom(E& e, typename E::out_t o, typename E::in_t a, typename E::in_t b,
                          int N, int K, int M, const std::string& type_name) {
  e.pipeline(matmul_custom_kernel_name(type_name));
  e.out(o, 0); e.in(a, 1); e.in(b, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- gemm_v3: D(out)@0 A@1 B@2 ; N@3 K@4 M@5 ; grid (M/64,N/64,1), 128 threads. -----
template <class E>
void launch_gemm_v3(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                    int N, int K, int M, const std::string& type_name) {
  e.pipeline("gemm_v3_" + type_name);
  e.out(d, 0); e.in(a, 1); e.in(b, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 64, N / 64, 1, 128, 1, 1);
}

// ----- attn_fwd: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (N/8, H, B) group (32,1,1) -----
template <class E>
void launch_attn_fwd(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                     typename E::out_t o, unsigned N, unsigned H, int B, int D,
                     float softcap, typename E::in_t sinks, int has_sink) {
  // 16-row Q tile for D=128 when N allows: halves the passes over K/V (the 8-row tile fell
  // to 7.4 TFLOP/s at (2,16,4096,128) from K/V re-read pressure; q16 measured 1.3-1.6x).
  // D=64 was TRIED and reverted: its K/V stream is half the bytes and the doubled register
  // footprint made q16 ~1.4x slower there.
  // softcap <= 0 disables logit soft-capping. sinks is a per-head fp32 buffer read only when
  // has_sink != 0; callers without sinks pass q as the placeholder binding (never read).
  const bool q16 = (D == 128) && (N % 16 == 0);
  e.pipeline(q16 ? attn_fwd_kernel_name(D) + "_q16" : attn_fwd_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.bytes(softcap, 6); e.in(sinks, 7); e.bytes(has_sink, 8);
  e.dispatch(static_cast<int>(N) / (q16 ? 16 : 8), static_cast<int>(H), B, 32, 1, 1);
}

// ----- attn_fwd_sg_d256 (simdgroup_matrix flash attention, head-dim 256, GQA, f16 KV):
//        q@0(f32) k@1(f16) v@2(f16) o@3(f32) ; n_tokens@4 window@5 scale@6(f32) Hq@7 Hkv@8 ;
//        grid (ceil(T/8), Hq, 1), 128 threads (4 simdgroups). Bidirectional + optional window. -----
template <class E>
void launch_attn_fwd_sg_d256(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                             typename E::out_t o, uint32_t n_tokens, uint32_t window, float scale,
                             uint32_t Hq, uint32_t Hkv) {
  e.pipeline("mittens::attn_fwd_sg_d256");   // namespaced non-template kernel
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(n_tokens, 4); e.bytes(window, 5); e.bytes(scale, 6);
  e.bytes(Hq, 7); e.bytes(Hkv, 8);
  e.dispatch(static_cast<int>((n_tokens + 7) / 8), static_cast<int>(Hq), 1, 128, 1, 1);
}

// ----- attn_q (quantized-KV attention): q@0(bf16) Kq@1(uchar) Vq@2(uchar) o@3(bf16) ; N@4 H@5 ;
//        grid (N/8, H, B), 32 threads. Same online-softmax flow as attn_fwd, K/V dequantized. -----
template <class E>
void launch_attn_q(E& e, typename E::in_t q, typename E::in_t kq, typename E::in_t vq,
                   typename E::out_t o, unsigned N, unsigned H, int B, int D,
                   const std::string& fmt, bool causal, bool multiwarp) {
  const int NW = 4;  // attn_q_mw warps
  e.pipeline(multiwarp ? ("attn_q_mw_" + fmt + "_" + std::to_string(D))
                       : attn_q_kernel_name(fmt, D, causal));
  e.in(q, 0); e.in(kq, 1); e.in(vq, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  if (multiwarp)
    e.dispatch(static_cast<int>(N) / (8 * NW), static_cast<int>(H), B, 32 * NW, 1, 1);
  else
    e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- attn_decode: q@0(Hq,D) kc@1 vc@2(Tk,Hkv,D) -> out@3(Hq,D). -----
template <class E>
void launch_attn_decode(E& e, typename E::in_t q, typename E::in_t kc, typename E::in_t vc,
                        typename E::out_t out, int Tk, int Hq, int Hkv, int D,
                        const std::string& type_name) {
  e.pipeline("attn_decode_" + type_name);
  e.in(q, 0); e.in(kc, 1); e.in(vc, 2); e.out(out, 3);
  e.bytes(Tk, 4); e.bytes(Hq, 5); e.bytes(Hkv, 6); e.bytes(D, 7);
  e.dispatch(Hq, 1, 1, 32, 1, 1);
}

// Batched head-major decode: q (B,Hq,D), cache (B,Hkv,cache_T,D).
template <class E>
void launch_attn_decode_bh(E& e, typename E::in_t q, typename E::in_t kc,
                           typename E::in_t vc, typename E::out_t out,
                           int B, int Tk, int cache_T, int Hq, int Hkv, int D,
                           const std::string& type_name) {
  const int partitions = Tk >= 2048 ? 32 : (Tk >= 512 ? 8 : 1);
  e.pipeline(partitions == 1
      ? "attn_decode_bh_" + type_name
      : "attn_decode_bh_partitioned_" + std::to_string(partitions) + "_" + type_name);
  e.in(q, 0); e.in(kc, 1); e.in(vc, 2); e.out(out, 3);
  e.bytes(Tk, 4); e.bytes(Hq, 5); e.bytes(Hkv, 6); e.bytes(D, 7); e.bytes(cache_T, 8);
  e.dispatch(Hq, B, 1, partitions * 32, 1, 1);
}

template <class E>
void launch_decode_cache_attention(
    E& e, typename E::in_t q, typename E::in_t new_k, typename E::in_t new_v,
    typename E::in_t cos, typename E::in_t sin, typename E::in_t positions,
    typename E::in_t context_lengths, typename E::in_t q_weight,
    typename E::in_t k_weight, typename E::out_t key_cache,
    typename E::out_t value_cache, typename E::out_t output,
    int B, int Hq, int Hkv, int cache_T, int D, float eps,
    bool do_q_norm, bool do_k_norm, bool gemma, float softmax_scale,
    const std::string& type_name) {
  e.pipeline("decode_cache_attention_" + type_name);
  e.in(q, 0); e.in(new_k, 1); e.in(new_v, 2); e.in(cos, 3); e.in(sin, 4);
  e.in(positions, 5); e.in(context_lengths, 6); e.in(q_weight, 7); e.in(k_weight, 8);
  e.out(key_cache, 9); e.out(value_cache, 10); e.out(output, 11);
  e.bytes(B, 12); e.bytes(Hq, 13); e.bytes(Hkv, 14); e.bytes(cache_T, 15);
  e.bytes(D, 16); e.bytes(eps, 17); e.bytes(do_q_norm ? 1 : 0, 18);
  e.bytes(do_k_norm ? 1 : 0, 19); e.bytes(gemma ? 1 : 0, 20);
  e.bytes(softmax_scale, 21);
  e.dispatch(Hq, B, 1, 1024, 1, 1);
}

// Swin window attention over packed qkv (BW,N,3,H,32).
template <class E>
void launch_swin_attn_d32(E& e, typename E::in_t qkv, typename E::in_t relative_bias,
                          typename E::in_t mask, typename E::out_t output,
                          int BW, int N, int H, int windows_per_image, int has_mask,
                          const std::string& type_name) {
  e.pipeline("swin_attn_d32_" + type_name);
  e.in(qkv, 0); e.in(relative_bias, 1); e.in(mask, 2); e.out(output, 3);
  e.bytes(BW, 4); e.bytes(N, 5); e.bytes(H, 6);
  e.bytes(windows_per_image, 7); e.bytes(has_mask, 8);
  e.dispatch(N, H, BW, 32, 1, 1);
}

// Swin 2x2 gather + LayerNorm. One simdgroup per output patch.
template <class E>
void launch_patch_merge_layernorm(E& e, typename E::in_t input, typename E::in_t weight,
                                  typename E::in_t bias, typename E::out_t output,
                                  int B, int H, int W, int C, float eps) {
  e.pipeline("patch_merge_layernorm_bfloat16");
  e.in(input, 0); e.in(weight, 1); e.in(bias, 2); e.out(output, 3);
  e.bytes(B, 4); e.bytes(H, 5); e.bytes(W, 6); e.bytes(C, 7); e.bytes(eps, 8);
  e.dispatch(B * ((H + 1) / 2) * ((W + 1) / 2), 1, 1, 32, 1, 1);
}

template <class E>
void launch_space_to_depth_norm_linear(
    E& e, typename E::in_t input, typename E::in_t norm_weight,
    typename E::in_t norm_bias, typename E::in_t projection_weight,
    typename E::in_t projection_bias, typename E::out_t output,
    int B, int H, int W, int C, int O, int block_size, float eps,
    bool use_norm_bias, bool use_projection_bias,
    const std::string& type_name) {
  const int dimension = block_size * block_size * C;
  const int patches = B * ((H + block_size - 1) / block_size) *
      ((W + block_size - 1) / block_size);
  const bool use_tiled = type_name == "float32" && H % block_size == 0 &&
      W % block_size == 0 && dimension % 8 == 0 && O % 32 == 0 &&
      patches % 8 == 0;
  const bool use_group4 = !use_tiled && dimension <= 1024 && patches >= 256;
  e.pipeline("space_to_depth_norm_linear_" +
             std::string(use_tiled ? "tiled_" : (use_group4 ? "group4_" : "")) +
             type_name);
  e.in(input, 0); e.in(norm_weight, 1); e.in(norm_bias, 2);
  e.in(projection_weight, 3); e.in(projection_bias, 4); e.out(output, 5);
  e.bytes(B, 6); e.bytes(H, 7); e.bytes(W, 8); e.bytes(C, 9); e.bytes(O, 10);
  e.bytes(block_size, 11); e.bytes(eps, 12); e.bytes(use_norm_bias ? 1 : 0, 13);
  e.bytes(use_projection_bias ? 1 : 0, 14);
  if (use_tiled) {
    e.dispatch(O / 32, patches / 8, 1, 128, 1, 1);
  } else {
    e.dispatch(use_group4 ? (patches + 3) / 4 : patches,
               1, 1, 256, 1, 1);
  }
}

template <class E>
void launch_decode_linear(E& e, typename E::in_t input, typename E::in_t weight,
                          typename E::in_t bias, typename E::out_t output,
                          int B, int K, int N, bool gelu, const std::string& type_name) {
  e.pipeline("decode_linear_" + type_name);
  e.in(input, 0); e.in(weight, 1); e.in(bias, 2); e.out(output, 3);
  e.bytes(B, 4); e.bytes(K, 5); e.bytes(N, 6); e.bytes(gelu ? 1 : 0, 7);
  e.dispatch(N, B, 1, 32, 1, 1);
}

template <class E>
void launch_decode_linear_residual(
    E& e, typename E::in_t input, typename E::in_t weight,
    typename E::in_t bias, typename E::in_t residual, typename E::out_t output,
    int B, int K, int N, const std::string& type_name) {
  e.pipeline("decode_linear_residual_" + type_name);
  e.in(input, 0); e.in(weight, 1); e.in(bias, 2); e.in(residual, 3); e.out(output, 4);
  e.bytes(B, 5); e.bytes(K, 6); e.bytes(N, 7);
  e.dispatch(N, B, 1, 32, 1, 1);
}

template <class E>
void launch_decode_linear_q8(
    E& e, typename E::in_t input, typename E::in_t weight,
    typename E::in_t bias, typename E::in_t residual, typename E::out_t output,
    int B, int K, int N, bool gelu, bool use_residual,
    const std::string& type_name) {
  e.pipeline("decode_linear_q8_" + type_name);
  e.in(input, 0); e.in(weight, 1); e.in(bias, 2); e.in(residual, 3); e.out(output, 4);
  e.bytes(B, 5); e.bytes(K, 6); e.bytes(N, 7);
  e.bytes(gelu ? 1 : 0, 8); e.bytes(use_residual ? 1 : 0, 9);
  e.dispatch(N, B, 1, 32, 1, 1);
}

template <class E>
void launch_decode_linear_epilogue(
    E& e, typename E::in_t input, typename E::in_t weight,
    typename E::in_t bias, typename E::in_t residual, typename E::out_t output,
    int B, int K, int N, int activation, bool use_bias, bool use_residual,
    const std::string& format, const std::string& type_name) {
  const std::string layout = format.empty() ? "dense" : format;
  e.pipeline("decode_linear_epilogue_" + layout + "_" + type_name);
  e.in(input, 0); e.in(weight, 1); e.in(bias, 2); e.in(residual, 3); e.out(output, 4);
  e.bytes(B, 5); e.bytes(K, 6); e.bytes(N, 7); e.bytes(activation, 8);
  e.bytes(use_bias ? 1 : 0, 9); e.bytes(use_residual ? 1 : 0, 10);
  e.dispatch(N, B, 1, 32, 1, 1);
}

template <class E>
void launch_decode_swiglu(
    E& e, typename E::in_t input, typename E::in_t gate_weight,
    typename E::in_t up_weight, typename E::in_t gate_bias,
    typename E::in_t up_bias, typename E::out_t output,
    int B, int K, int N, bool use_bias, const std::string& format,
    const std::string& type_name) {
  const std::string layout = format.empty() ? "dense" : format;
  e.pipeline("decode_swiglu_" + layout + "_" + type_name);
  e.in(input, 0); e.in(gate_weight, 1); e.in(up_weight, 2);
  e.in(gate_bias, 3); e.in(up_bias, 4); e.out(output, 5);
  e.bytes(B, 6); e.bytes(K, 7); e.bytes(N, 8); e.bytes(use_bias ? 1 : 0, 9);
  e.dispatch(N, B, 1, 32, 1, 1);
}

template <class E>
void launch_edge_mlp_project_256(
    E& e, typename E::in_t hidden, typename E::in_t first_weight,
    typename E::in_t first_bias, typename E::out_t left_output,
    typename E::out_t right_output,
    int B, int L, const std::string& type_name) {
  e.pipeline("edge_mlp_project_256_" + type_name);
  e.in(hidden, 0); e.in(first_weight, 1); e.in(first_bias, 2);
  e.out(left_output, 3); e.out(right_output, 4);
  e.bytes(B, 5); e.bytes(L, 6);
  e.dispatch(B * L, 1, 1, 256, 1, 1);
}

template <class E>
void launch_edge_mlp_combine_256x7(
    E& e, typename E::in_t left_partial, typename E::in_t right_partial,
    typename E::in_t second_weight, typename E::in_t second_bias,
    typename E::out_t output, int B, int L, const std::string& type_name) {
  e.pipeline("edge_mlp_combine_256x7_" + type_name);
  e.in(left_partial, 0); e.in(right_partial, 1);
  e.in(second_weight, 2); e.in(second_bias, 3); e.out(output, 4);
  e.bytes(B, 5); e.bytes(L, 6);
  e.dispatch(L * L, B, 1, 256, 1, 1);
}

// ----- rms_norm: x@0 w@1 -> o@2 ; M@3(u32) eps@4(f32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_rms_norm(E& e, typename E::in_t x, typename E::in_t w,
                     typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_kernel_name(D));
  e.in(x, 0); e.in(w, 1); e.out(o, 2);
  e.bytes(M, 3); e.bytes(eps, 4);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- mean_pool_rms_l2: x@0 w@1 -> o@2 ; M@3(u32) eps@4(f32) ; grid (1,1,1) group (32,1,1) -----
// Pools M rows of width D into one D-vector (mean -> RMSNorm(w) -> L2-normalize).
template <class E>
void launch_mean_pool_rms_l2(E& e, typename E::in_t x, typename E::in_t w,
                             typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline(mean_pool_rms_l2_kernel_name(D));
  e.in(x, 0); e.in(w, 1); e.out(o, 2);
  e.bytes(M, 3); e.bytes(eps, 4);
  e.dispatch(1, 1, 1, 32, 1, 1);
}
template <class E>
void launch_masked_mean_pool_rms_l2(
    E& e, typename E::in_t x, typename E::in_t mask, typename E::in_t w,
    typename E::out_t o, uint32_t B, uint32_t T, int D, float eps) {
  e.pipeline("masked_mean_pool_rms_l2_" + std::to_string(D));
  e.in(x, 0); e.in(mask, 1); e.in(w, 2); e.out(o, 3);
  e.bytes(T, 4); e.bytes(eps, 5);
  e.dispatch(B, 1, 1, 32, 1, 1);
}

// ----- rms_norm_residual_next: x@0 post_w@1 residual@2 next_w@3 -> res_out@4 next_out@5 ;
//        M@6(u32) eps@7(f32) ; grid (M,1,1) group (32,1,1). Post-norm x, add to residual,
//        pre-norm the result for the next block — the residual-stream seam in one pass. -----
template <class E>
void launch_rms_norm_residual_next(E& e, typename E::in_t x, typename E::in_t post_w,
                                   typename E::in_t residual, typename E::in_t next_w,
                                   typename E::out_t res_out, typename E::out_t next_out,
                                   uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_residual_next_kernel_name(D));
  e.in(x, 0); e.in(post_w, 1); e.in(residual, 2); e.in(next_w, 3);
  e.out(res_out, 4); e.out(next_out, 5);
  e.bytes(M, 6); e.bytes(eps, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_rms_norm_dyn(E& e, typename E::in_t x, typename E::in_t w,
                         typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline("mittens::rms_norm_dyn");
  e.in(x, 0); e.in(w, 1); e.out(o, 2);
  e.bytes(M, 3); e.bytes(eps, 4); e.bytes(D, 5);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_rms_norm_bwd_dx(E& e, typename E::in_t x, typename E::in_t w, typename E::in_t dy,
                            typename E::in_t rstd, typename E::out_t dx, int rows, int D,
                            const std::string& type_name) {
  e.pipeline("rms_norm_bwd_dx_" + type_name);
  e.in(x, 0); e.in(w, 1); e.in(dy, 2); e.in(rstd, 3); e.out(dx, 4);
  e.bytes(D, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
// fused RMSNorm backward: rstd in-kernel + dX + atomic dweight (dweight (D,) fp32 zeroed first).
template <class E>
void launch_rms_norm_bwd_fused(E& e, typename E::in_t x, typename E::in_t w, typename E::in_t dy,
                               typename E::out_t dx, typename E::out_t dweight, int rows, int D,
                               float eps, const std::string& type_name) {
  e.pipeline("rms_norm_bwd_fused_" + type_name);
  e.in(x, 0); e.in(w, 1); e.in(dy, 2); e.out(dx, 3); e.out(dweight, 4);
  e.bytes(D, 5); e.bytes(eps, 6);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- rms_norm_add: x@0 residual@1 w@2 -> o@3 res_out@4 ; M@5(u32) eps@6(f32) ;
//        grid (M,1,1) group (32,1,1). o = rms_norm(x+residual)*w ; res_out = x+residual -----
template <class E>
void launch_rms_norm_add(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                         typename E::out_t o, typename E::out_t res_out,
                         uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_add_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(o, 3); e.out(res_out, 4);
  e.bytes(M, 5); e.bytes(eps, 6);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- layernorm_add: x@0 residual@1 w@2 b@3 -> o@4 res_out@5 ; M@6(u32) eps@7(f32) ;
//        grid (M,1,1) group (32,1,1). o = layernorm(x+residual)*w+b ; res_out = x+residual -----
template <class E>
void launch_layernorm_add(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                          typename E::in_t b, typename E::out_t o, typename E::out_t res_out,
                          uint32_t M, int D, float eps) {
  e.pipeline(layernorm_add_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(o, 4); e.out(res_out, 5);
  e.bytes(M, 6); e.bytes(eps, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

template <class E>
void launch_decode_layernorm_add(
    E& e, typename E::in_t input, typename E::in_t residual,
    typename E::in_t weight, typename E::in_t bias,
    typename E::out_t normalized, typename E::out_t summed,
    int rows, int dimension, float eps, const std::string& type_name) {
  e.pipeline("decode_layernorm_add_" + type_name);
  e.in(input, 0); e.in(residual, 1); e.in(weight, 2); e.in(bias, 3);
  e.out(normalized, 4); e.out(summed, 5);
  e.bytes(rows, 6); e.bytes(dimension, 7); e.bytes(eps, 8);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- rope_kv_insert: k@0 v@1 cos@2 sin@3 positions@4(i32) slot_mapping@5(i64) ->
//        key_cache@6 value_cache@7 ; num_kv_heads@8(i32) block_size@9(i32) ; grid (M,1,1).
//        M = num_tokens*num_kv_heads. caches must be pre-cloned (insert overwrites slot rows). -----
template <class E>
void launch_rope_kv_insert(E& e, typename E::in_t k, typename E::in_t v,
                           typename E::in_t cos, typename E::in_t sin,
                           typename E::in_t positions, typename E::in_t slot_mapping,
                           typename E::out_t key_cache, typename E::out_t value_cache,
                           int M, int num_kv_heads, int block_size, int D,
                           const std::string& type_name) {
  e.pipeline(rope_kv_insert_kernel_name(type_name, D));
  e.in(k, 0); e.in(v, 1); e.in(cos, 2); e.in(sin, 3);
  e.in(positions, 4); e.in(slot_mapping, 5);
  e.out(key_cache, 6); e.out(value_cache, 7);
  e.bytes(num_kv_heads, 8); e.bytes(block_size, 9);
  e.dispatch(M, 1, 1, 32, 1, 1);
}

// ----- rope_kv_insert_norm: adds K RMSNorm (weight@8, eps@11, gemma@12) before RoPE+insert. -----
template <class E>
void launch_rope_kv_insert_norm(E& e, typename E::in_t k, typename E::in_t v,
                                typename E::in_t cos, typename E::in_t sin,
                                typename E::in_t positions, typename E::in_t slot_mapping,
                                typename E::out_t key_cache, typename E::out_t value_cache,
                                typename E::in_t norm_weight, int M, int num_kv_heads,
                                int block_size, int D, float eps, int gemma,
                                const std::string& type_name) {
  e.pipeline(rope_kv_insert_norm_kernel_name(type_name, D));
  e.in(k, 0); e.in(v, 1); e.in(cos, 2); e.in(sin, 3);
  e.in(positions, 4); e.in(slot_mapping, 5);
  e.out(key_cache, 6); e.out(value_cache, 7); e.in(norm_weight, 8);
  e.bytes(num_kv_heads, 9); e.bytes(block_size, 10); e.bytes(eps, 11); e.bytes(gemma, 12);
  e.dispatch(M, 1, 1, 32, 1, 1);
}

// ----- rope_q: q@0 cos@1 sin@2 positions@3 -> q_out@4 ; num_heads@5 do_norm@6 gemma@7 eps@8 ;
//        norm_weight@9 ; grid (M=tokens*heads,1,1), 32 thr. Rotate (+opt norm) Q, out row = in row. -----
template <class E>
void launch_rope_q(E& e, typename E::in_t q, typename E::in_t cos, typename E::in_t sin,
                   typename E::in_t positions, typename E::out_t q_out, typename E::in_t norm_weight,
                   int M, int num_heads, int do_norm, int gemma, float eps, int D,
                   const std::string& type_name) {
  e.pipeline(rope_q_kernel_name(type_name, D));
  e.in(q, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.out(q_out, 4);
  e.bytes(num_heads, 5); e.bytes(do_norm, 6); e.bytes(gemma, 7); e.bytes(eps, 8);
  e.in(norm_weight, 9);
  e.dispatch(M, 1, 1, 32, 1, 1);
}

// ----- moe_route_topk: logits@0(T,E) -> topk_ids@1(i32) topk_weights@2(f32), both (T,K) ;
//        E@3(i32) K@4(i32) ; grid (num_tokens,1,1), 32 thr. Top-k experts + renormalized softmax. -----
template <class Enc>
void launch_moe_route_topk(Enc& e, typename Enc::in_t logits, typename Enc::out_t topk_ids,
                           typename Enc::out_t topk_weights, int num_tokens, int num_experts,
                           int k, const std::string& type_name) {
  e.pipeline(moe_route_topk_kernel_name(type_name));
  e.in(logits, 0); e.out(topk_ids, 1); e.out(topk_weights, 2);
  e.bytes(num_experts, 3); e.bytes(k, 4);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

// ----- MoE permute pipeline (all int32). Run in order in one (serial) encoder. -----
template <class Enc>
void launch_moe_zero_i32(Enc& e, typename Enc::out_t p, int n) {
  e.pipeline("moe_zero_i32");
  e.out(p, 0); e.bytes(n, 1);
  e.dispatch((n + 255) / 256, 1, 1, 256, 1, 1);
}
template <class Enc>
void launch_moe_histogram(Enc& e, typename Enc::in_t topk_ids, typename Enc::out_t counts, int TK) {
  e.pipeline("moe_histogram");
  e.in(topk_ids, 0); e.out(counts, 1); e.bytes(TK, 2);
  e.dispatch((TK + 255) / 256, 1, 1, 256, 1, 1);
}
template <class Enc>
void launch_moe_scan_offsets(Enc& e, typename Enc::in_t counts, typename Enc::out_t offsets,
                             typename Enc::out_t cursor, int num_experts) {
  e.pipeline("moe_scan_offsets");
  e.in(counts, 0); e.out(offsets, 1); e.out(cursor, 2); e.bytes(num_experts, 3);
  e.dispatch(1, 1, 1, 256, 1, 1);   // one threadgroup; parallel P2 scan
}
template <class Enc>
void launch_moe_scatter(Enc& e, typename Enc::in_t topk_ids, typename Enc::out_t cursor,
                        typename Enc::out_t sorted_row_idx, typename Enc::out_t inv_idx, int TK) {
  e.pipeline("moe_scatter");
  e.in(topk_ids, 0); e.out(cursor, 1); e.out(sorted_row_idx, 2); e.out(inv_idx, 3); e.bytes(TK, 4);
  e.dispatch((TK + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- moe padded schedule (GPU replacement for the host glue): -----
// moe_pad_offsets: offsets@0(E+1) -> off_pad@1(E+1) expert_of_tile@2(max_tiles)
//   gather_idx@3(init -1) ; E@4 max_tiles@5 total_pad_max@6 ; one threadgroup.
template <class Enc>
void launch_moe_pad_offsets(Enc& e, typename Enc::in_t offsets, typename Enc::out_t off_pad,
                            typename Enc::out_t expert_of_tile, typename Enc::out_t gather_idx,
                            int E, int max_tiles, int total_pad_max) {
  e.pipeline("moe_pad_offsets");
  e.in(offsets, 0); e.out(off_pad, 1); e.out(expert_of_tile, 2); e.out(gather_idx, 3);
  e.bytes(E, 4); e.bytes(max_tiles, 5); e.bytes(total_pad_max, 6);
  e.dispatch(1, 1, 1, 256, 1, 1);
}
// moe_pad_scatter: sorted_row_idx@0 offsets@1 off_pad@2 -> gather_idx@3 inv_pad@4 ;
//   TK@5 E@6 K@7 ; flat over TK.
template <class Enc>
void launch_moe_pad_scatter(Enc& e, typename Enc::in_t sorted_row_idx, typename Enc::in_t offsets,
                            typename Enc::in_t off_pad, typename Enc::out_t gather_idx,
                            typename Enc::out_t inv_pad, int TK, int E, int K) {
  e.pipeline("moe_pad_scatter");
  e.in(sorted_row_idx, 0); e.in(offsets, 1); e.in(off_pad, 2);
  e.out(gather_idx, 3); e.out(inv_pad, 4);
  e.bytes(TK, 5); e.bytes(E, 6); e.bytes(K, 7);
  e.dispatch((TK + 255) / 256, 1, 1, 256, 1, 1);
}
// moe_gather: x@0(T,H) gather_idx@1 -> out@2(total_pad_max,H) ; H@3 ;
//   grid (total_pad_max,) x 128 thr (vec4 row copies; gather_idx<0 rows zero-filled).
template <class Enc>
void launch_moe_gather(Enc& e, typename Enc::in_t x, typename Enc::in_t gather_idx,
                       typename Enc::out_t out, int H, int total_pad_max,
                       const std::string& type_name) {
  e.pipeline("moe_gather_" + type_name);
  e.in(x, 0); e.in(gather_idx, 1); e.out(out, 2); e.bytes(H, 3);
  e.dispatch(total_pad_max, 1, 1, 128, 1, 1);
}

// ----- moe_grouped_gemm: out@0 A@1(permuted_input) W@2(E,H,H) expert_of_tile@3(i32) ;
//        total_rows@4 H@5 ; grid (H/32, total_rows/32, 1), 32 thr. out = A @ W[expert]. -----
template <class Enc>
void launch_moe_grouped_gemm(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                             typename Enc::in_t W, typename Enc::in_t expert_of_tile,
                             int total_rows, int H, const std::string& type_name) {
  e.pipeline(moe_grouped_gemm_kernel_name(type_name));
  e.out(out, 0); e.in(A, 1); e.in(W, 2); e.in(expert_of_tile, 3);
  e.bytes(total_rows, 4); e.bytes(H, 5);
  e.dispatch(H / 32, total_rows / 32, 1, 32, 1, 1);
}

// Rectangular grouped GEMM: out(total_rows,N_out)=A(total_rows,K_dim)@W[e](K_dim,N_out). grid (N_out/32, rows/32).
template <class Enc>
void launch_moe_grouped_gemm_rect(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                                  typename Enc::in_t W, typename Enc::in_t expert_of_tile,
                                  int total_rows, int K_dim, int N_out, const std::string& type_name) {
  e.pipeline(moe_grouped_gemm_rect_kernel_name(type_name));
  e.out(out, 0); e.in(A, 1); e.in(W, 2); e.in(expert_of_tile, 3);
  e.bytes(total_rows, 4); e.bytes(K_dim, 5); e.bytes(N_out, 6);
  e.dispatch(N_out / 32, total_rows / 32, 1, 32, 1, 1);
}

// Fused SiLU-GLU GEMM1: out(total_rows,inter)=silu(A@W1_gate)*(A@W1_up), W1[e] (H,2*inter). grid (inter/32, rows/32).
template <class Enc>
void launch_moe_grouped_gemm_swiglu(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                                    typename Enc::in_t W1, typename Enc::in_t expert_of_tile,
                                    int total_rows, int H, int inter, const std::string& type_name) {
  e.pipeline(moe_grouped_gemm_swiglu_kernel_name(type_name));
  e.out(out, 0); e.in(A, 1); e.in(W1, 2); e.in(expert_of_tile, 3);
  e.bytes(total_rows, 4); e.bytes(H, 5); e.bytes(inter, 6);
  e.dispatch(inter / 32, total_rows / 32, 1, 32, 1, 1);
}

template <class Enc>
void launch_moe_grouped_gemm_bwd_dx(Enc& e, typename Enc::out_t dx, typename Enc::in_t dy,
                                    typename Enc::in_t w, typename Enc::in_t expert_of_tile,
                                    int total_rows, int K_dim, int N_out,
                                    const std::string& type_name) {
  e.pipeline("moe_grouped_gemm_bwd_dx_" + type_name);
  e.out(dx, 0); e.in(dy, 1); e.in(w, 2); e.in(expert_of_tile, 3);
  e.bytes(total_rows, 4); e.bytes(K_dim, 5); e.bytes(N_out, 6);
  e.dispatch(K_dim / 32, total_rows / 32, 1, 32, 1, 1);
}

template <class Enc>
void launch_moe_grouped_gemm_bwd_dw(Enc& e, typename Enc::out_t dw, typename Enc::in_t a,
                                    typename Enc::in_t dy, typename Enc::in_t off_pad,
                                    int NE, int total_rows, int K_dim, int N_out,
                                    const std::string& type_name) {
  e.pipeline("moe_grouped_gemm_bwd_dw_" + type_name);
  e.out(dw, 0); e.in(a, 1); e.in(dy, 2); e.in(off_pad, 3);
  e.bytes(total_rows, 4); e.bytes(K_dim, 5); e.bytes(N_out, 6);
  e.dispatch(N_out / 32, K_dim / 32, NE, 32, 1, 1);
}

// ----- gdn_recur (GatedDeltaNet): q@0 k@1 v@2 g@3 beta@4 state_pool@5(f32) cu_seqlens@6(i32)
//        slot_mapping@7(i32) -> y@8 ; R@9 Hk@10 Hv@11 Dv@12 load_initial@13 ;
//        grid (Dv, 1, R*Hv), 32 thr (lanes partition Dk; Dk in {64,128} via kernel name). -----
template <class E>
void launch_gdn_recur(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                      typename E::in_t g, typename E::in_t beta, typename E::out_t state_pool,
                      typename E::in_t cu_seqlens, typename E::in_t slot_mapping,
                      typename E::out_t y, int R, int Hk, int Hv, int Dv, int Dk,
                      int load_initial, const std::string& type_name) {
  e.pipeline(gdn_recur_kernel_name(type_name, Dk));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(g, 3); e.in(beta, 4);
  e.out(state_pool, 5); e.in(cu_seqlens, 6); e.in(slot_mapping, 7); e.out(y, 8);
  e.bytes(R, 9); e.bytes(Hk, 10); e.bytes(Hv, 11); e.bytes(Dv, 12); e.bytes(load_initial, 13);
  e.dispatch(Dv, 1, R * Hv, 32, 1, 1);
}

// ----- GDN preparation/output substrate. Short convolution uses one thread per channel
//        and one threadgroup row per request; QKV preparation and gated RMSNorm use one
//        simdgroup per logical head row. Gate/beta is flat elementwise fp32 output. -----
template <class E>
void launch_gdn_short_conv(
    E& e, typename E::in_t x, typename E::in_t weight,
    typename E::out_t state_pool, typename E::in_t cu_seqlens,
    typename E::in_t slot_mapping, typename E::out_t out,
    int R, int channels, int kernel_size, int load_initial, int apply_silu,
    const std::string& type_name) {
  e.pipeline(gdn_short_conv_kernel_name(type_name));
  e.in(x, 0); e.in(weight, 1); e.out(state_pool, 2); e.in(cu_seqlens, 3);
  e.in(slot_mapping, 4); e.out(out, 5); e.bytes(R, 6); e.bytes(channels, 7);
  e.bytes(kernel_size, 8); e.bytes(load_initial, 9); e.bytes(apply_silu, 10);
  constexpr int threads = 256;
  e.dispatch((channels + threads - 1) / threads, R, 1, threads, 1, 1);
}

template <class E>
void launch_gdn_qkv_prepare(
    E& e, typename E::in_t mixed, typename E::out_t q, typename E::out_t k,
    typename E::out_t v, int tokens, int Hk, int Hv, int Dk, int Dv,
    float eps, float q_scale, float k_scale, const std::string& type_name) {
  e.pipeline(gdn_qkv_prepare_kernel_name(type_name, Dk, Dv));
  e.in(mixed, 0); e.out(q, 1); e.out(k, 2); e.out(v, 3);
  e.bytes(tokens, 4); e.bytes(Hk, 5); e.bytes(Hv, 6); e.bytes(eps, 7);
  e.bytes(q_scale, 8); e.bytes(k_scale, 9);
  e.dispatch(tokens * (2 * Hk + Hv), 1, 1, 32, 1, 1);
}

template <class E>
void launch_gdn_gate_beta(
    E& e, typename E::in_t a, typename E::in_t b, typename E::in_t A_log,
    typename E::in_t dt_bias, typename E::out_t decay, typename E::out_t beta,
    uint32_t n, int heads, const std::string& type_name) {
  e.pipeline(gdn_gate_beta_kernel_name(type_name));
  e.in(a, 0); e.in(b, 1); e.in(A_log, 2); e.in(dt_bias, 3);
  e.out(decay, 4); e.out(beta, 5); e.bytes(n, 6); e.bytes(heads, 7);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

template <class E>
void launch_gdn_gated_rmsnorm(
    E& e, typename E::in_t y, typename E::in_t z, typename E::in_t weight,
    typename E::out_t out, int rows, int dim, float eps,
    const std::string& type_name) {
  e.pipeline(gdn_gated_rmsnorm_kernel_name(type_name, dim));
  e.in(y, 0); e.in(z, 1); e.in(weight, 2); e.out(out, 3);
  e.bytes(rows, 4); e.bytes(eps, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- Mamba-1 (S6) selective scan. Layouts channel-major (seqlen/total_tokens LAST);
//        state (…, dim, dstate) fp32 is the single in/out buffer (bound as out; the MLX path
//        clone-prepasses the pool). grid (batch, dim, 1); threads = round32(dstate) <= 256. -----
template <class E>
void launch_selective_scan_dense(E& e, typename E::in_t u, typename E::in_t delta,
                                 typename E::in_t A, typename E::in_t B, typename E::in_t C,
                                 typename E::in_t D, typename E::in_t delta_bias,
                                 typename E::in_t z, typename E::out_t out,
                                 typename E::out_t state, int batch, int dim, int seqlen,
                                 int dstate, int n_groups, int has_d, int has_delta_bias,
                                 int has_z, int delta_softplus, const std::string& type_name) {
  e.pipeline(selective_scan_kernel_name("dense", type_name));
  e.in(u, 0); e.in(delta, 1); e.in(A, 2); e.in(B, 3); e.in(C, 4);
  e.in(D, 5); e.in(delta_bias, 6); e.in(z, 7);
  e.out(out, 8); e.out(state, 9);
  e.bytes(batch, 10); e.bytes(dim, 11); e.bytes(seqlen, 12); e.bytes(dstate, 13);
  e.bytes(n_groups, 14); e.bytes(has_d, 15); e.bytes(has_delta_bias, 16);
  e.bytes(has_z, 17); e.bytes(delta_softplus, 18);
  const int threads = ((dstate + 31) / 32) * 32;
  e.dispatch(batch, dim, 1, threads, 1, 1);
}

template <class E>
void launch_selective_scan_varlen(E& e, typename E::in_t u, typename E::in_t delta,
                                  typename E::in_t A, typename E::in_t B, typename E::in_t C,
                                  typename E::in_t D, typename E::in_t delta_bias,
                                  typename E::in_t z, typename E::in_t query_start_loc,
                                  typename E::in_t cache_indices,
                                  typename E::in_t has_initial_state, typename E::out_t out,
                                  typename E::out_t state, int batch, int dim, int total_tokens,
                                  int dstate, int n_groups, int has_d, int has_delta_bias,
                                  int has_z, int delta_softplus, int use_cache_indices,
                                  int use_has_initial_state, int null_block_id,
                                  const std::string& type_name) {
  e.pipeline(selective_scan_kernel_name("varlen", type_name));
  e.in(u, 0); e.in(delta, 1); e.in(A, 2); e.in(B, 3); e.in(C, 4);
  e.in(D, 5); e.in(delta_bias, 6); e.in(z, 7);
  e.in(query_start_loc, 8); e.in(cache_indices, 9); e.in(has_initial_state, 10);
  e.out(out, 11); e.out(state, 12);
  e.bytes(batch, 13); e.bytes(dim, 14); e.bytes(total_tokens, 15); e.bytes(dstate, 16);
  e.bytes(n_groups, 17); e.bytes(has_d, 18); e.bytes(has_delta_bias, 19);
  e.bytes(has_z, 20); e.bytes(delta_softplus, 21); e.bytes(use_cache_indices, 22);
  e.bytes(use_has_initial_state, 23); e.bytes(null_block_id, 24);
  const int threads = ((dstate + 31) / 32) * 32;
  e.dispatch(batch, dim, 1, threads, 1, 1);
}

// varlen + APC: paged state checkpointing at chunk boundaries + prefix-cache initial state.
template <class E>
void launch_selective_scan_varlen_apc(
    E& e, typename E::in_t u, typename E::in_t delta, typename E::in_t A, typename E::in_t B,
    typename E::in_t C, typename E::in_t D, typename E::in_t delta_bias, typename E::in_t z,
    typename E::in_t query_start_loc, typename E::in_t cache_indices,
    typename E::in_t has_initial_state, typename E::out_t out, typename E::out_t state,
    typename E::in_t block_idx_first, typename E::in_t block_idx_last,
    typename E::in_t initial_state_idx, typename E::in_t cu_chunk_seqlen,
    typename E::in_t last_chunk_indices, int batch, int dim, int total_tokens, int dstate,
    int n_groups, int has_d, int has_delta_bias, int has_z, int delta_softplus,
    int null_block_id, int block_size, int cache_indices_stride, int use_chunk_metadata,
    const std::string& type_name) {
  e.pipeline(selective_scan_kernel_name("varlen_apc", type_name));
  e.in(u, 0); e.in(delta, 1); e.in(A, 2); e.in(B, 3); e.in(C, 4);
  e.in(D, 5); e.in(delta_bias, 6); e.in(z, 7);
  e.in(query_start_loc, 8); e.in(cache_indices, 9); e.in(has_initial_state, 10);
  e.out(out, 11); e.out(state, 12);
  e.in(block_idx_first, 13); e.in(block_idx_last, 14); e.in(initial_state_idx, 15);
  e.in(cu_chunk_seqlen, 16); e.in(last_chunk_indices, 17);
  e.bytes(batch, 18); e.bytes(dim, 19); e.bytes(total_tokens, 20); e.bytes(dstate, 21);
  e.bytes(n_groups, 22); e.bytes(has_d, 23); e.bytes(has_delta_bias, 24);
  e.bytes(has_z, 25); e.bytes(delta_softplus, 26); e.bytes(null_block_id, 27);
  e.bytes(block_size, 28); e.bytes(cache_indices_stride, 29); e.bytes(use_chunk_metadata, 30);
  const int threads = ((dstate + 31) / 32) * 32;
  e.dispatch(batch, dim, 1, threads, 1, 1);
}

// fp32 pool clone prepass (src@0 -> dst@1 ; n@2(u32)); grid-stride copy.
// (non-template kernel => the namespaced symbol survives, per the mittens:: gotcha)
template <class E>
void launch_sscan_pool_clone(E& e, typename E::in_t src, typename E::out_t dst, uint32_t n) {
  e.pipeline("mittens::sscan_pool_clone");
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- qk_norm_rope: qkv@0 q_weight@1 k_weight@2 cos@3 sin@4 positions@5(i32) -> out@6 ;
//        Hq@7 Hk@8 Hv@9 eps@10(f32) interleave@11 gemma@12 ; grid (Hq+Hk+Hv, T, 1), 32 thr.
//        Fused per-head QK-RMSNorm + RoPE on packed QKV; V heads copied through. -----
template <class E>
void launch_qk_norm_rope(E& e, typename E::in_t qkv, typename E::in_t q_weight,
                         typename E::in_t k_weight, typename E::in_t cosb, typename E::in_t sinb,
                         typename E::in_t positions, typename E::out_t out, int T, int Hq, int Hk,
                         int Hv, int D, float eps, int interleave, int gemma) {
  e.pipeline(qk_norm_rope_kernel_name(D));
  e.in(qkv, 0); e.in(q_weight, 1); e.in(k_weight, 2); e.in(cosb, 3); e.in(sinb, 4);
  e.in(positions, 5); e.out(out, 6);
  e.bytes(Hq, 7); e.bytes(Hk, 8); e.bytes(Hv, 9); e.bytes(eps, 10);
  e.bytes(interleave, 11); e.bytes(gemma, 12);
  e.dispatch(Hq + Hk + Hv, T, 1, 32, 1, 1);
}

// ----- qk_norm_rope_kv_f16: qkv@0 q_w@1 k_w@2 cos@3 sin@4 positions@5(i32) ->
//        q_out@6(bf16) k_out@7(half) v_out@8(half) ; Hq@9 Hk@10 Hv@11 eps@12(f32)
//        interleave@13 gemma@14 ; grid (Hq+Hk+Hv, T, 1), 32 thr. qk_norm_rope with a
//        fused f16 KV split-store: Q->bf16 q_out, K/V->contiguous half k_out/v_out. -----
template <class E>
void launch_qk_norm_rope_kv_f16(E& e, typename E::in_t qkv, typename E::in_t q_weight,
                                typename E::in_t k_weight, typename E::in_t cosb,
                                typename E::in_t sinb, typename E::in_t positions,
                                typename E::out_t q_out, typename E::out_t k_out,
                                typename E::out_t v_out, int T, int Hq, int Hk, int Hv, int D,
                                float eps, int interleave, int gemma) {
  e.pipeline(qk_norm_rope_kv_f16_kernel_name(D));
  e.in(qkv, 0); e.in(q_weight, 1); e.in(k_weight, 2); e.in(cosb, 3); e.in(sinb, 4);
  e.in(positions, 5); e.out(q_out, 6); e.out(k_out, 7); e.out(v_out, 8);
  e.bytes(Hq, 9); e.bytes(Hk, 10); e.bytes(Hv, 11); e.bytes(eps, 12);
  e.bytes(interleave, 13); e.bytes(gemma, 14);
  e.dispatch(Hq + Hk + Hv, T, 1, 32, 1, 1);
}

// ----- moe_route_grouped (DeepSeek noaux_tc): logits@0 bias@1(f32,(E,)) -> topk_ids@2(i32)
//        topk_weights@3(f32) ; E@4 n_group@5 topk_group@6 K@7 renormalize@8
//        routed_scaling_factor@9(f32) scoring_func@10 (0 softmax/1 sigmoid/2 sqrt-softplus)
//        has_bias@11 ; grid (T,1,1), 32 thr. bias read only when has_bias (pass any small
//        f32 buffer as placeholder otherwise). Output contract == moe_route_topk. -----
template <class Enc>
void launch_moe_route_grouped(Enc& e, typename Enc::in_t logits, typename Enc::in_t bias,
                              typename Enc::out_t topk_ids, typename Enc::out_t topk_weights,
                              int T, int E, int n_group, int topk_group, int K, int renormalize,
                              float routed_scaling_factor, int scoring_func, int has_bias,
                              const std::string& type_name) {
  e.pipeline(moe_route_grouped_kernel_name(type_name));
  e.in(logits, 0); e.in(bias, 1); e.out(topk_ids, 2); e.out(topk_weights, 3);
  e.bytes(E, 4); e.bytes(n_group, 5); e.bytes(topk_group, 6); e.bytes(K, 7);
  e.bytes(renormalize, 8); e.bytes(routed_scaling_factor, 9); e.bytes(scoring_func, 10);
  e.bytes(has_bias, 11);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

// ----- Quantized grouped expert GEMMs (weight-only quant, bf16 activations). Wq is the packed
// expert stack (E, N_out, K_dim/block_k, block_bytes) — (N, K) orientation, TRANSPOSED vs the
// dense kernels, contracted as A @ Wq^T. bias is (E, N_out) bf16 (rect) / (E, 2*inter) (swiglu);
// pass a 1-element dummy with has_bias=0 when absent. fmt selects the [[host_name]] variant
// (mxfp4 / mxfp8 / kU4 / fp8_e4m3 / q8_0 / nvfp4 / q4_K). -----
// rect_q: out@0 A@1 Wq@2(u8) expert_of_tile@3(i32) bias@4 ; rows@5 K_dim@6 N_out@7 has_bias@8 ;
//         grid (N_out/32, rows/32, 1), 32 thr.
template <class Enc>
void launch_moe_grouped_gemm_rect_q(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                                    typename Enc::in_t Wq, typename Enc::in_t expert_of_tile,
                                    typename Enc::in_t bias, int total_rows, int K_dim, int N_out,
                                    int has_bias, const std::string& fmt) {
  e.pipeline(moe_grouped_gemm_rect_q_kernel_name(fmt));
  e.out(out, 0); e.in(A, 1); e.in(Wq, 2); e.in(expert_of_tile, 3); e.in(bias, 4);
  e.bytes(total_rows, 5); e.bytes(K_dim, 6); e.bytes(N_out, 7); e.bytes(has_bias, 8);
  e.dispatch(N_out / 32, total_rows / 32, 1, 128, 1, 1);   // 4-warp split-K per tile
}

// swiglu_q: out@0 A@1 W1q@2(u8, [gate|up] along N) expert_of_tile@3(i32) bias@4 ; rows@5 H@6
//           inter@7 has_bias@8 act_mode@9 (0 swiglu, 1 swiglu_oai) alpha@10(f32) limit@11(f32) ;
//           grid (inter/32, rows/32, 1), 64 threads for byte-per-value FP8, 128 otherwise.
template <class Enc>
void launch_moe_grouped_gemm_swiglu_q(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                                      typename Enc::in_t W1q, typename Enc::in_t expert_of_tile,
                                      typename Enc::in_t bias, int total_rows, int H, int inter,
                                      int has_bias, int act_mode, float alpha, float limit,
                                      const std::string& fmt) {
  const bool fp8_two_warp = fmt == "mxfp8" || fmt == "fp8_e4m3";
  e.pipeline(moe_grouped_gemm_swiglu_q_kernel_name(fmt));
  e.out(out, 0); e.in(A, 1); e.in(W1q, 2); e.in(expert_of_tile, 3); e.in(bias, 4);
  e.bytes(total_rows, 5); e.bytes(H, 6); e.bytes(inter, 7); e.bytes(has_bias, 8);
  e.bytes(act_mode, 9); e.bytes(alpha, 10); e.bytes(limit, 11);
  const int threads = fp8_two_warp ? 64 : 128;
  e.dispatch(inter / 32, total_rows / 32, 1, threads, 1, 1);
}

// ----- moe_finalize: expert_out@0 inv_idx@1(i32) topk_weights@2(f32) -> out@3 ; K@4 Hdim@5 ;
//        grid (num_tokens,1,1), 32 thr. k-way weighted reduce via inv_idx (no atomics). -----
template <class Enc>
void launch_moe_finalize(Enc& e, typename Enc::in_t expert_out, typename Enc::in_t inv_idx,
                         typename Enc::in_t topk_weights, typename Enc::out_t out,
                         int num_tokens, int k, int Hdim, const std::string& type_name) {
  e.pipeline(moe_finalize_kernel_name(type_name));
  e.in(expert_out, 0); e.in(inv_idx, 1); e.in(topk_weights, 2); e.out(out, 3);
  e.bytes(k, 4); e.bytes(Hdim, 5);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

template <class Enc>
void launch_moe_finalize_bwd(Enc& e, typename Enc::in_t grad_out, typename Enc::in_t expert_out,
                             typename Enc::in_t inv_idx, typename Enc::out_t grad_eo,
                             typename Enc::out_t grad_w, typename Enc::in_t topk_weights,
                             int T, int K, int Hdim, const std::string& type_name) {
  e.pipeline("moe_finalize_bwd_" + type_name);
  e.in(grad_out, 0); e.in(expert_out, 1); e.in(inv_idx, 2); e.out(grad_eo, 3);
  e.out(grad_w, 4); e.in(topk_weights, 5);
  e.bytes(K, 6); e.bytes(Hdim, 7);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

template <class Enc>
void launch_moe_gather_bwd(Enc& e, typename Enc::in_t dA, typename Enc::in_t inv_idx,
                           typename Enc::out_t dx, int T, int K, int Hdim,
                           const std::string& type_name) {
  e.pipeline("moe_gather_bwd_" + type_name);
  e.in(dA, 0); e.in(inv_idx, 1); e.out(dx, 2);
  e.bytes(K, 3); e.bytes(Hdim, 4);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

// ----- argmax (greedy sampling): logits@0 -> out_idx@1(i32) ; V@2(i32) ; grid (rows,1,1), 32 thr.
//        One simdgroup per row finds the argmax token over the vocab dim V. -----
template <class E>
void launch_argmax(E& e, typename E::in_t logits, typename E::out_t out_idx,
                   int rows, int V, const std::string& type_name) {
  e.pipeline(argmax_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- sample_categorical: logits@0 -> out_idx@1(i32) ; V@2(i32) seed@3(u32) invtemp@4(f32) ;
//        grid (rows,1,1), 32 thr. Gumbel-max sampling from softmax(logits/temperature). -----
template <class E>
void launch_sample_categorical(E& e, typename E::in_t logits, typename E::out_t out_idx,
                               int rows, int V, uint32_t seed, float invtemp,
                               const std::string& type_name) {
  e.pipeline(sample_categorical_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(seed, 3); e.bytes(invtemp, 4);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- top_k_sample: logits@0 -> out_idx@1(i32) ; V@2 K@3(i32) seed@4(u32) invtemp@5(f32) ;
//        grid (rows,1,1), 32 thr. Gumbel-max sampling restricted to the top-k logits. -----
template <class E>
void launch_top_k_sample(E& e, typename E::in_t logits, typename E::out_t out_idx,
                         int rows, int V, int k, uint32_t seed, float invtemp,
                         const std::string& type_name) {
  e.pipeline(top_k_sample_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(k, 3); e.bytes(seed, 4); e.bytes(invtemp, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- beam_topk_partials: logits@0 cum_log_probs@1 -> cand_score@2 cand_token@3 ; V@4 two_bm@5
//        (i32) ; grid (B*BM,) × 32. Per-beam log-softmax top-2BM candidate scores. -----
template <class E>
void launch_beam_topk_partials(E& e, typename E::in_t logits, typename E::in_t cum_log_probs,
                               typename E::out_t cand_score, typename E::out_t cand_token,
                               int rows, int V, int two_bm, const std::string& type_name) {
  e.pipeline("beam_topk_partials_" + type_name);
  e.in(logits, 0); e.in(cum_log_probs, 1); e.out(cand_score, 2); e.out(cand_token, 3);
  e.bytes(V, 4); e.bytes(two_bm, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- beam_select: cand_score@0 cand_token@1 -> next_token@2 parent_beam@3 new_cum@4 ; BM@5
//        two_bm@6 (i32) ; grid (B,) × 32. Global top-BM over the batch's BM*2BM candidates. -----
template <class E>
void launch_beam_select(E& e, typename E::in_t cand_score, typename E::in_t cand_token,
                        typename E::out_t next_token, typename E::out_t parent_beam,
                        typename E::out_t new_cum, int B, int BM, int two_bm) {
  e.pipeline("beam_select");
  e.in(cand_score, 0); e.in(cand_token, 1);
  e.out(next_token, 2); e.out(parent_beam, 3); e.out(new_cum, 4);
  e.bytes(BM, 5); e.bytes(two_bm, 6);
  e.dispatch(B, 1, 1, 32, 1, 1);
}

// Speculative decoding: linear rejection-sampling verification. One simdgroup per request.
template <class E>
void launch_spec_verify_linear(E& e, typename E::in_t draft_tokens, typename E::in_t draft_probs,
                               typename E::in_t target_probs, typename E::in_t bonus_tokens,
                               typename E::in_t accept_u, typename E::out_t out_tokens,
                               typename E::out_t accepted_cnt, int B, int S, int V, unsigned seed) {
  e.pipeline("spec_verify_linear");
  e.in(draft_tokens, 0); e.in(draft_probs, 1); e.in(target_probs, 2); e.in(bonus_tokens, 3);
  e.in(accept_u, 4); e.out(out_tokens, 5); e.out(accepted_cnt, 6);
  e.bytes(S, 7); e.bytes(V, 8); e.bytes(seed, 9);
  e.dispatch(B, 1, 1, 32, 1, 1);
}
// spec_verify_tree: one simdgroup per request (grid B x 32) tree rejection verification.
template <class E>
void launch_spec_verify_tree(E& e, typename E::in_t draft_tokens, typename E::in_t target_probs,
                             typename E::in_t retrieve_next_token,
                             typename E::in_t retrieve_next_sibling, typename E::out_t accept_index,
                             typename E::out_t accept_token, typename E::out_t accept_num,
                             typename E::in_t tree_valid, int B, int N, int V, unsigned seed) {
  e.pipeline("spec_verify_tree");
  e.in(draft_tokens, 0); e.in(target_probs, 1); e.in(retrieve_next_token, 2);
  e.in(retrieve_next_sibling, 3); e.out(accept_index, 4); e.out(accept_token, 5);
  e.out(accept_num, 6); e.bytes(N, 7); e.bytes(V, 8); e.bytes(seed, 9);
  e.in(tree_valid, 10);
  e.dispatch(B, 1, 1, 32, 1, 1);
}
// build_dynamic_tree: one simdgroup per request (grid B x 32) builds the tree pointers on device.
template <class E>
void launch_build_dynamic_tree(E& e, typename E::in_t parents, typename E::out_t rt,
                               typename E::out_t rs, typename E::out_t positions, int B, int N) {
  e.pipeline("build_dynamic_tree");
  e.in(parents, 0); e.out(rt, 1); e.out(rs, 2); e.out(positions, 3); e.bytes(N, 4);
  e.dispatch(B, 1, 1, 32, 1, 1);
}
// spec_compact: single threadgroup (B<=256) exclusive-scan compaction of accepted tokens.
template <class E>
void launch_spec_compact(E& e, typename E::in_t out_tokens, typename E::in_t accepted_cnt,
                         typename E::in_t seq_lens, typename E::out_t packed_tokens,
                         typename E::out_t packed_pos, typename E::out_t cu_accepted, int B,
                         int Sp1) {
  e.pipeline("spec_compact");
  e.in(out_tokens, 0); e.in(accepted_cnt, 1); e.in(seq_lens, 2);
  e.out(packed_tokens, 3); e.out(packed_pos, 4); e.out(cu_accepted, 5);
  e.bytes(B, 6); e.bytes(Sp1, 7);
  int nthreads = ((B + 31) / 32) * 32;
  if (nthreads < 32) nthreads = 32;
  if (nthreads > 256) nthreads = 256;
  e.dispatch(1, 1, 1, nthreads, 1, 1);
}
template <class E>
void launch_spec_update_kv_meta(E& e, typename E::in_t seq_lens, typename E::in_t accepted_cnt,
                                typename E::out_t new_seq_lens, int B) {
  e.pipeline("spec_update_kv_meta");
  e.in(seq_lens, 0); e.in(accepted_cnt, 1); e.out(new_seq_lens, 2); e.bytes(B, 3);
  e.dispatch((B + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- vLLM v1 ragged rejection samplers. out (B, S1) int32; cu (B+1,); one thread/request
//        for greedy/random, one simdgroup/draft-token for recovered. -----
template <class E>
void launch_rejection_greedy_sample(E& e, typename E::out_t out, typename E::in_t cu,
                                    typename E::in_t draft_ids, typename E::in_t target_argmax,
                                    typename E::in_t bonus_ids, typename E::in_t is_greedy,
                                    int B, int S1, int has_is_greedy) {
  e.pipeline("rejection_greedy_sample");
  e.out(out, 0); e.in(cu, 1); e.in(draft_ids, 2); e.in(target_argmax, 3); e.in(bonus_ids, 4);
  e.in(is_greedy, 5); e.bytes(B, 6); e.bytes(S1, 7); e.bytes(has_is_greedy, 8);
  e.dispatch((B + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_rejection_random_sample(E& e, typename E::out_t out, typename E::in_t cu,
                                    typename E::in_t draft_ids, typename E::in_t draft_probs,
                                    typename E::in_t target_probs, typename E::in_t bonus_ids,
                                    typename E::in_t recovered_ids, typename E::in_t uniform_probs,
                                    typename E::in_t is_greedy, int B, int S1, int V,
                                    int no_draft_probs, int has_is_greedy) {
  e.pipeline("rejection_random_sample");
  e.out(out, 0); e.in(cu, 1); e.in(draft_ids, 2); e.in(draft_probs, 3); e.in(target_probs, 4);
  e.in(bonus_ids, 5); e.in(recovered_ids, 6); e.in(uniform_probs, 7); e.in(is_greedy, 8);
  e.bytes(B, 9); e.bytes(S1, 10); e.bytes(V, 11); e.bytes(no_draft_probs, 12);
  e.bytes(has_is_greedy, 13);
  e.dispatch((B + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_sample_recovered_tokens(E& e, typename E::out_t out, typename E::in_t cu,
                                    typename E::in_t draft_ids, typename E::in_t draft_probs,
                                    typename E::in_t target_probs, typename E::in_t inv_q,
                                    int B, int total, int V, int no_draft_probs) {
  e.pipeline("sample_recovered_tokens");
  e.out(out, 0); e.in(cu, 1); e.in(draft_ids, 2); e.in(draft_probs, 3); e.in(target_probs, 4);
  e.in(inv_q, 5); e.bytes(B, 6); e.bytes(total, 7); e.bytes(V, 8); e.bytes(no_draft_probs, 9);
  e.dispatch(total, 1, 1, 32, 1, 1);
}

// ----- EAGLE input-prep metadata builders. One thread/request; cu (B+1,). -----
template <class E>
void launch_eagle_prepare_inputs_padded(E& e, typename E::in_t cu, typename E::in_t valid_count,
                                        typename E::in_t query_start_loc,
                                        typename E::out_t token_indices, typename E::out_t num_rej,
                                        int num_reqs) {
  e.pipeline("eagle_prepare_inputs_padded");
  e.in(cu, 0); e.in(valid_count, 1); e.in(query_start_loc, 2); e.out(token_indices, 3);
  e.out(num_rej, 4); e.bytes(num_reqs, 5);
  e.dispatch((num_reqs + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_eagle_prepare_next_token_padded(E& e, typename E::in_t sampled_ids,
                                            typename E::in_t discard, typename E::in_t backup,
                                            typename E::out_t next_token_ids,
                                            typename E::out_t valid_count, int vocab_size,
                                            int num_sampled, int num_reqs) {
  e.pipeline("eagle_prepare_next_token_padded");
  e.in(sampled_ids, 0); e.in(discard, 1); e.in(backup, 2); e.out(next_token_ids, 3);
  e.out(valid_count, 4); e.bytes(vocab_size, 5); e.bytes(num_sampled, 6); e.bytes(num_reqs, 7);
  e.dispatch((num_reqs + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_eagle_step_slot_mapping_metadata(E& e, typename E::in_t positions,
                                             typename E::in_t block_table, typename E::in_t seq_lens,
                                             typename E::out_t out_clamped_pos,
                                             typename E::out_t out_slot_mapping,
                                             typename E::out_t new_seq_lens, int block_size,
                                             int max_model_len, int pad_id, int batch_size,
                                             int input_batch_size, int block_table_stride,
                                             int n_blocks_per_req) {
  e.pipeline("eagle_step_slot_mapping_metadata");
  e.in(positions, 0); e.in(block_table, 1); e.in(seq_lens, 2); e.out(out_clamped_pos, 3);
  e.out(out_slot_mapping, 4); e.out(new_seq_lens, 5); e.bytes(block_size, 6);
  e.bytes(max_model_len, 7); e.bytes(pad_id, 8); e.bytes(batch_size, 9);
  e.bytes(input_batch_size, 10); e.bytes(block_table_stride, 11); e.bytes(n_blocks_per_req, 12);
  e.dispatch((input_batch_size + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_eagle_expand_int32(E& e, typename E::out_t output, typename E::in_t input,
                               typename E::in_t cu, int replace_from, int replace_to,
                               int batch_size) {
  e.pipeline("eagle_expand_int32");
  e.out(output, 0); e.in(input, 1); e.in(cu, 2); e.bytes(replace_from, 3); e.bytes(replace_to, 4);
  e.bytes(batch_size, 5);
  e.dispatch((batch_size + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- top_p_sample: logits@0 -> out_idx@1(i32) ; V@2(i32) p@3(f32) seed@4(u32) invtemp@5(f32) ;
//        grid (rows,1,1), 32 thr. Gumbel-max sampling from the nucleus (cumulative prob >= p). -----
template <class E>
void launch_top_p_sample(E& e, typename E::in_t logits, typename E::out_t out_idx,
                         int rows, int V, float p, uint32_t seed, float invtemp,
                         const std::string& type_name) {
  e.pipeline(top_p_sample_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(p, 3); e.bytes(seed, 4); e.bytes(invtemp, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_min_p_sample(E& e, typename E::in_t logits, typename E::out_t out_idx, int rows, int V,
                         float min_p, uint32_t seed, float invtemp, const std::string& type_name) {
  e.pipeline("min_p_sample_" + type_name);
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(min_p, 3); e.bytes(seed, 4); e.bytes(invtemp, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_typical_p_sample(E& e, typename E::in_t logits, typename E::out_t out_idx, int rows,
                             int V, float typ_p, uint32_t seed, float invtemp,
                             const std::string& type_name) {
  e.pipeline("typical_p_sample_" + type_name);
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(typ_p, 3); e.bytes(seed, 4); e.bytes(invtemp, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_apply_token_bitmask(E& e, typename E::in_t logits, typename E::in_t bitmask,
                                typename E::out_t out, int rows, int V, int num_words,
                                const std::string& type_name) {
  e.pipeline("apply_token_bitmask_" + type_name);
  e.in(logits, 0); e.in(bitmask, 1); e.out(out, 2);
  e.bytes(V, 3); e.bytes(num_words, 4);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_apply_bad_words(E& e, typename E::in_t logits, typename E::in_t bad_ids,
                            typename E::in_t bad_lens, typename E::out_t out, int rows, int V,
                            int maxbad, const std::string& type_name) {
  e.pipeline("apply_bad_words_" + type_name);
  e.in(logits, 0); e.in(bad_ids, 1); e.in(bad_lens, 2); e.out(out, 3);
  e.bytes(V, 4); e.bytes(maxbad, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- token embedding + multimodal span merge: one threadgroup per token, threads stride D
//        (vec4 when D%4==0). Thread count = vec4-groups rounded to a warp, capped at 256. -----
static inline int tk_embed_threads(int D) {
  int t = (((D + 3) / 4 + 31) / 32) * 32;    // round vec4-group count up to a simdgroup
  return t < 32 ? 32 : (t > 256 ? 256 : t);
}
template <class E>
void launch_embedding_lookup(E& e, typename E::in_t token_ids, typename E::in_t table,
                             typename E::in_t pos_table, typename E::out_t out, int D, int vocab,
                             int n_tok, float scale, int use_pos, const std::string& type_name) {
  e.pipeline("embedding_lookup_" + type_name);
  e.in(token_ids, 0); e.in(table, 1); e.in(pos_table, 2); e.out(out, 3);
  e.bytes(D, 4); e.bytes(vocab, 5); e.bytes(n_tok, 6); e.bytes(scale, 7); e.bytes(use_pos, 8);
  e.dispatch(n_tok, 1, 1, tk_embed_threads(D), 1, 1);
}
template <class E>
void launch_embedding_lookup_types(
    E& e, typename E::in_t token_ids, typename E::in_t type_ids,
    typename E::in_t token_table, typename E::in_t type_table,
    typename E::out_t out, int D, int token_vocab, int type_vocab,
    int n_tok, float token_scale, const std::string& type_name) {
  e.pipeline("embedding_lookup_types_" + type_name);
  e.in(token_ids, 0); e.in(type_ids, 1); e.in(token_table, 2); e.in(type_table, 3);
  e.out(out, 4); e.bytes(D, 5); e.bytes(token_vocab, 6); e.bytes(type_vocab, 7);
  e.bytes(token_scale, 8);
  e.dispatch(n_tok, 1, 1, tk_embed_threads(D), 1, 1);
}
// build the multimodal src map on-device (one thread per token, scans the spans).
template <class E>
void launch_build_multimodal_src(E& e, typename E::in_t span_offsets, typename E::in_t span_lengths,
                                 typename E::in_t modal_starts, typename E::out_t src, int num_spans,
                                 int num_tok) {
  e.pipeline("build_multimodal_src");
  e.in(span_offsets, 0); e.in(span_lengths, 1); e.in(modal_starts, 2); e.out(src, 3);
  e.bytes(num_spans, 4); e.bytes(num_tok, 5);
  e.dispatch((num_tok + 255) / 256, 1, 1, 256, 1, 1);
}
// zero a float buffer of n elements (gradient accumulator init).
template <class E>
void launch_embedding_zero_f32(E& e, typename E::out_t p, int n) {
  e.pipeline("embedding_zero_f32");
  e.out(p, 0); e.bytes(n, 1);
  e.dispatch((n + 255) / 256, 1, 1, 256, 1, 1);
}
// embedding backward: atomic scatter-add dY rows into dtable (vocab,D) fp32 (zeroed first) by
// token id. One threadgroup per token; threads stride D.
template <class E>
void launch_embedding_backward(E& e, typename E::in_t token_ids, typename E::in_t dY,
                               typename E::out_t dtable, int D, int vocab, int n_tok, float scale,
                               const std::string& type_name) {
  e.pipeline("embedding_backward_" + type_name);
  e.in(token_ids, 0); e.in(dY, 1); e.out(dtable, 2);
  e.bytes(D, 3); e.bytes(vocab, 4); e.bytes(n_tok, 5); e.bytes(scale, 6);
  e.dispatch(n_tok, 1, 1, tk_embed_threads(D), 1, 1);
}
// embedding backward (sorted-segment, atomic-free): host pre-sorts tokens by id (perm/sorted_ids);
// the segment-start threadgroup of each id sums its run into dtable (zeroed first). One threadgroup
// per sorted position; threads stride D.
template <class E>
void launch_embedding_backward_sorted(E& e, typename E::in_t sorted_ids, typename E::in_t perm,
                                      typename E::in_t dY, typename E::out_t dtable, int D, int vocab,
                                      int n_tok, float scale, const std::string& type_name) {
  e.pipeline("embedding_backward_sorted_" + type_name);
  e.in(sorted_ids, 0); e.in(perm, 1); e.in(dY, 2); e.out(dtable, 3);
  e.bytes(D, 4); e.bytes(vocab, 5); e.bytes(n_tok, 6); e.bytes(scale, 7);
  e.dispatch(n_tok, 1, 1, tk_embed_threads(D), 1, 1);
}
template <class E>
void launch_merge_multimodal_spans(E& e, typename E::in_t text, typename E::in_t modal,
                                   typename E::in_t src, typename E::out_t out, int D, int n_tok,
                                   int n_modal, const std::string& type_name) {
  e.pipeline("merge_multimodal_spans_" + type_name);
  e.in(text, 0); e.in(modal, 1); e.in(src, 2); e.out(out, 3);
  e.bytes(D, 4); e.bytes(n_tok, 5); e.bytes(n_modal, 6);
  e.dispatch(n_tok, 1, 1, tk_embed_threads(D), 1, 1);
}

// ----- penalty_histogram: prev_tokens@0(i32) -> counts@1(atomic i32) ; V@2 L@3 TL@4 ; grid (TL).
//        counts[(row,tok)] += 1 for each valid history token. Zero counts first. -----
template <class E>
void launch_penalty_histogram(E& e, typename E::in_t prev_tokens, typename E::out_t counts,
                              int V, int L, int TL, typename E::in_t parent_ids) {
  e.pipeline("penalty_histogram");
  e.in(prev_tokens, 0); e.out(counts, 1);
  e.bytes(V, 2); e.bytes(L, 3); e.bytes(TL, 4); e.in(parent_ids, 5);
  e.dispatch((TL + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- apply_penalty: logits@0 counts@1(i32) -> out@2 ; V@3 invtemp@4 rep@5 presence@6 freq@7 ;
//        grid (rows,1,1), 32 thr. temperature + repetition/presence/frequency penalties. -----
template <class E>
void launch_apply_penalty(E& e, typename E::in_t logits, typename E::in_t counts,
                          typename E::out_t out, typename E::in_t bias, int rows, int V,
                          float invtemp, float rep, float presence, float freq,
                          int eos_id, int min_length, int gen_len, const std::string& type_name) {
  e.pipeline(apply_penalty_kernel_name(type_name));
  e.in(logits, 0); e.in(counts, 1); e.out(out, 2);
  e.bytes(V, 3); e.bytes(invtemp, 4); e.bytes(rep, 5); e.bytes(presence, 6); e.bytes(freq, 7);
  e.in(bias, 8); e.bytes(eos_id, 9); e.bytes(min_length, 10); e.bytes(gen_len, 11);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- per-tensor dynamic quant (2 passes): absmax@1(atomic_uint) reduce, then encode. -----
template <class E>
void launch_quant_tensor_absmax(E& e, typename E::in_t x, typename E::out_t scale_u, int n,
                                const std::string& type_name) {
  e.pipeline(quant_tensor_absmax_kernel_name(type_name));
  e.in(x, 0); e.out(scale_u, 1); e.bytes(n, 2);
  const int t16 = (n + 15) / 16;                       // 16 elements per thread
  e.dispatch((t16 + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_quant_tensor_encode(E& e, typename E::in_t x, typename E::in_t scale_u,
                                typename E::out_t codes, typename E::out_t scale_out, int n,
                                bool is_int8, const std::string& type_name) {
  e.pipeline(is_int8 ? quant_tensor_encode_int8_kernel_name(type_name)
                     : quant_tensor_encode_fp8_kernel_name(type_name));
  e.in(x, 0); e.in(scale_u, 1); e.out(codes, 2); e.out(scale_out, 3); e.bytes(n, 4);
  const int t4 = (n + 3) / 4;                          // 4 elements per thread
  e.dispatch((t4 + 255) / 256, 1, 1, 256, 1, 1);
}

template <class E>
void launch_calibration_absmax(
    E& e, typename E::in_t x, typename E::in_t running,
    typename E::out_t out, int tokens, int channels, int has_running,
    const std::string& type_name) {
  e.pipeline("calibration_absmax_" + type_name);
  e.in(x, 0); e.in(running, 1); e.out(out, 2);
  e.bytes(tokens, 3); e.bytes(channels, 4); e.bytes(has_running, 5);
  e.dispatch((channels + 31) / 32, 1, 1, 256, 1, 1);
}

// ----- quantize_per_token_fp8: x@0 -> codes@1(uint8) scale@2(f32) ; D@3(i32) ; grid (rows,1,1).
//        Per-row absmax -> scale=absmax/448 ; codes = e4m3(x/scale). -----
template <class E>
void launch_quantize_per_token_fp8(E& e, typename E::in_t x, typename E::out_t codes,
                                   typename E::out_t scale, int rows, int D,
                                   const std::string& type_name) {
  e.pipeline(quantize_per_token_fp8_kernel_name(type_name));
  e.in(x, 0); e.out(codes, 1); e.out(scale, 2);
  e.bytes(D, 3);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- quantize_per_token_int8: x@0 -> codes@1(int8) scale@2(f32) ; D@3(i32) ; grid (rows,1,1).
//        Per-row absmax -> scale=absmax/127 ; codes = round_clamp(x/scale). -----
template <class E>
void launch_quantize_per_token_int8(E& e, typename E::in_t x, typename E::out_t codes,
                                    typename E::out_t scale, int rows, int D,
                                    const std::string& type_name) {
  e.pipeline(quantize_per_token_int8_kernel_name(type_name));
  e.in(x, 0); e.out(codes, 1); e.out(scale, 2);
  e.bytes(D, 3);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- weight_quant_ternary: W@0(E,N,K) -> wq@1(u8 bitnet blocks), w_deq@2(bf16).
//        K@3 group_k@4 N@5; grid (N,E,1), one simdgroup per row. -----
template <class E>
void launch_weight_quant_ternary(E& e, typename E::in_t w, typename E::out_t wq,
                                 typename E::out_t w_deq, int NE, int N, int K, int group_k,
                                 const std::string& type_name) {
  e.pipeline(weight_quant_ternary_kernel_name(type_name));
  e.in(w, 0); e.out(wq, 1); e.out(w_deq, 2);
  e.bytes(K, 3); e.bytes(group_k, 4); e.bytes(N, 5);
  e.dispatch(N, NE, 1, 32, 1, 1);
}

template <class E>
void launch_quantize_tq2_0(E& e, typename E::in_t w, typename E::out_t wq,
                           typename E::out_t w_deq, int NE, int N, int K,
                           const std::string& type_name) {
  e.pipeline("quantize_tq2_0_" + type_name);
  e.in(w, 0); e.out(wq, 1); e.out(w_deq, 2);
  e.bytes(K, 3); e.bytes(N, 4);
  e.dispatch(N, NE, 1, 32, 1, 1);
}

template <class E>
void launch_weight_quant_zero_float(E& e, typename E::out_t p, int n) {
  e.pipeline("weight_quant_zero_float");
  e.out(p, 0); e.bytes(n, 1);
  e.dispatch((n + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- weight_quant_ternary_pt: one absmean scale per (N,K) slice. -----
template <class E>
void launch_weight_quant_ternary_abssum(E& e, typename E::in_t w, typename E::out_t abssum,
                                        int NE, int NK, const std::string& type_name) {
  e.pipeline("weight_quant_ternary_abssum_" + type_name);
  e.in(w, 0); e.out(abssum, 1); e.bytes(NK, 2);
  const int t16 = (NK + 15) / 16;
  e.dispatch((t16 + 255) / 256, NE, 1, 256, 1, 1);
}

template <class E>
void launch_weight_quant_ternary_pt_encode(E& e, typename E::in_t w, typename E::in_t abssum,
                                           typename E::out_t wq, typename E::out_t w_deq,
                                           int NE, int N, int K, const std::string& type_name) {
  e.pipeline("weight_quant_ternary_pt_encode_" + type_name);
  e.in(w, 0); e.in(abssum, 1); e.out(wq, 2); e.out(w_deq, 3);
  e.bytes(K, 4); e.bytes(N, 5);
  e.dispatch(N, NE, 1, 32, 1, 1);
}

// ----- kd_kl_topk sparse-teacher distillation loss. -----
template <class E>
void launch_kd_kl_topk_fwd(E& e, typename E::in_t logits, typename E::in_t t_idx,
                           typename E::in_t t_prob, typename E::out_t loss,
                           typename E::out_t lse, int rows, int V, int K, float invtemp,
                           int tail_mode, const std::string& type_name) {
  e.pipeline("kd_kl_topk_fwd_" + type_name);
  e.in(logits, 0); e.in(t_idx, 1); e.in(t_prob, 2); e.out(loss, 3); e.out(lse, 4);
  e.bytes(V, 5); e.bytes(K, 6); e.bytes(invtemp, 7); e.bytes(tail_mode, 8);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_kd_kl_topk_bwd(E& e, typename E::in_t logits, typename E::in_t t_idx,
                           typename E::in_t t_prob, typename E::in_t lse,
                           typename E::in_t grad_out, typename E::out_t grad_logits,
                           int rows, int V, int K, float invtemp, int tail_mode,
                           const std::string& type_name) {
  e.pipeline("kd_kl_topk_bwd_" + type_name);
  e.in(logits, 0); e.in(t_idx, 1); e.in(t_prob, 2); e.in(lse, 3); e.in(grad_out, 4);
  e.out(grad_logits, 5);
  e.bytes(V, 6); e.bytes(K, 7); e.bytes(invtemp, 8); e.bytes(tail_mode, 9);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_kd_kl_dense_fwd(E& e, typename E::in_t t_logits, typename E::in_t s_logits,
                            typename E::out_t loss, typename E::out_t lse_t,
                            typename E::out_t lse_s, int rows, int V, float invtemp,
                            const std::string& type_name) {
  e.pipeline("kd_kl_dense_fwd_" + type_name);
  e.in(t_logits, 0); e.in(s_logits, 1); e.out(loss, 2); e.out(lse_t, 3); e.out(lse_s, 4);
  e.bytes(V, 5); e.bytes(invtemp, 6);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_kd_kl_dense_bwd(E& e, typename E::in_t t_logits, typename E::in_t s_logits,
                            typename E::in_t lse_t, typename E::in_t lse_s,
                            typename E::in_t grad_out, typename E::out_t grad_s,
                            int rows, int V, float invtemp, const std::string& type_name) {
  e.pipeline("kd_kl_dense_bwd_" + type_name);
  e.in(t_logits, 0); e.in(s_logits, 1); e.in(lse_t, 2); e.in(lse_s, 3); e.in(grad_out, 4);
  e.out(grad_s, 5);
  e.bytes(V, 6); e.bytes(invtemp, 7);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_kd_ce_fused_fwd(E& e, typename E::in_t t_logits, typename E::in_t s_logits,
                            typename E::in_t targets, typename E::out_t ce,
                            typename E::out_t kd, typename E::out_t lse_sr,
                            typename E::out_t lse_st, typename E::out_t lse_t,
                            int rows, int V, float invtemp, const std::string& type_name) {
  e.pipeline("kd_ce_fused_fwd_" + type_name);
  e.in(t_logits, 0); e.in(s_logits, 1); e.in(targets, 2);
  e.out(ce, 3); e.out(kd, 4); e.out(lse_sr, 5); e.out(lse_st, 6); e.out(lse_t, 7);
  e.bytes(V, 8); e.bytes(invtemp, 9);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_kd_ce_fused_bwd(E& e, typename E::in_t t_logits, typename E::in_t s_logits,
                            typename E::in_t targets, typename E::in_t lse_sr,
                            typename E::in_t lse_st, typename E::in_t lse_t,
                            typename E::in_t go_ce, typename E::in_t go_kd,
                            typename E::out_t grad_s, int rows, int V, float invtemp,
                            const std::string& type_name) {
  e.pipeline("kd_ce_fused_bwd_" + type_name);
  e.in(t_logits, 0); e.in(s_logits, 1); e.in(targets, 2); e.in(lse_sr, 3);
  e.in(lse_st, 4); e.in(lse_t, 5); e.in(go_ce, 6); e.in(go_kd, 7);
  e.out(grad_s, 8);
  e.bytes(V, 9); e.bytes(invtemp, 10);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- fake_quant_int8: x@0 -> x_q@1(bf16) codes@2(i8) scale@3(f32). -----
template <class E>
void launch_fake_quant_int8(E& e, typename E::in_t x, typename E::out_t x_q,
                            typename E::out_t codes, typename E::out_t scale,
                            int rows, int D, const std::string& type_name) {
  e.pipeline("fake_quant_int8_" + type_name);
  e.in(x, 0); e.out(x_q, 1); e.out(codes, 2); e.out(scale, 3);
  e.bytes(D, 4);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_silu_mul_fake_quant_int8(E& e, typename E::in_t x, typename E::in_t gate,
                                     typename E::out_t x_q, typename E::out_t codes,
                                     typename E::out_t scale, int rows, int D, int mode,
                                     float alpha, float limit, const std::string& type_name) {
  e.pipeline("silu_mul_fake_quant_int8_" + type_name);
  e.in(x, 0); e.in(gate, 1); e.out(x_q, 2); e.out(codes, 3); e.out(scale, 4);
  e.bytes(D, 5); e.bytes(mode, 6); e.bytes(alpha, 7); e.bytes(limit, 8);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_fake_quant_fp8(E& e, typename E::in_t x, typename E::in_t scale_u,
                           typename E::out_t x_fq, typename E::out_t scale_out, int n,
                           const std::string& type_name) {
  e.pipeline("fake_quant_fp8_" + type_name);
  e.in(x, 0); e.in(scale_u, 1); e.out(x_fq, 2); e.out(scale_out, 3);
  e.bytes(n, 4);
  const int t4 = (n + 3) / 4;
  e.dispatch((t4 + 255) / 256, 1, 1, 256, 1, 1);
}

template <class E>
void launch_ternary_stats(E& e, typename E::in_t wq, typename E::out_t counts,
                          int rows, int nblocks) {
  e.pipeline("ternary_stats");
  e.in(wq, 0); e.out(counts, 1); e.bytes(nblocks, 2);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_code_flip_count(E& e, typename E::in_t wq_a, typename E::in_t wq_b,
                            typename E::out_t flips, int rows, int nblocks) {
  e.pipeline("code_flip_count");
  e.in(wq_a, 0); e.in(wq_b, 1); e.out(flips, 2); e.bytes(nblocks, 3);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- fp8 norm epilogues: codes=e4m3(norm(x+residual)*w[+b]/scale). res_out=x+residual (bf16).
//        static: inv_scale param; dyn: per-row absmax/448 -> scale output. grid (M,1,1). -----
template <class E>
void launch_rms_norm_add_fp8(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                             typename E::out_t codes, typename E::out_t res_out,
                             uint32_t M, int D, float eps, float inv_scale) {
  e.pipeline(rms_norm_add_fp8_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(codes, 3); e.out(res_out, 4);
  e.bytes(M, 5); e.bytes(eps, 6); e.bytes(inv_scale, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_rms_norm_add_fp8_dyn(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                 typename E::out_t codes, typename E::out_t res_out,
                                 typename E::out_t scale, uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_add_fp8_dyn_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(codes, 3); e.out(res_out, 4); e.out(scale, 5);
  e.bytes(M, 6); e.bytes(eps, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_rms_norm_add_int8_dyn(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                 typename E::out_t codes, typename E::out_t res_out,
                                 typename E::out_t scale, uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_add_int8_dyn_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(codes, 3); e.out(res_out, 4); e.out(scale, 5);
  e.bytes(M, 6); e.bytes(eps, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
// per-block (per-128-group) dynamic norm-quant: codes@3 res_out@4 scale@5(rows,D/128) ; ue8m0@8.
template <class E>
void launch_rms_norm_add_per_block(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                   typename E::out_t codes, typename E::out_t res_out,
                                   typename E::out_t scale, uint32_t M, int D, float eps, bool int8,
                                   int ue8m0) {
  e.pipeline(rms_norm_add_per_block_kernel_name(D, int8));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(codes, 3); e.out(res_out, 4); e.out(scale, 5);
  e.bytes(M, 6); e.bytes(eps, 7); e.bytes(ue8m0, 8);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_add_int8_dyn(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                   typename E::in_t b, typename E::out_t codes,
                                   typename E::out_t res_out, typename E::out_t scale, uint32_t M,
                                   int D, float eps) {
  e.pipeline(layernorm_add_int8_dyn_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(codes, 4); e.out(res_out, 5); e.out(scale, 6);
  e.bytes(M, 7); e.bytes(eps, 8);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_add_per_block(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                    typename E::in_t b, typename E::out_t codes,
                                    typename E::out_t res_out, typename E::out_t scale, uint32_t M,
                                    int D, float eps, bool int8, int ue8m0) {
  e.pipeline(layernorm_add_per_block_kernel_name(D, int8));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(codes, 4); e.out(res_out, 5); e.out(scale, 6);
  e.bytes(M, 7); e.bytes(eps, 8); e.bytes(ue8m0, 9);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_add_fp8(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                              typename E::in_t b, typename E::out_t codes, typename E::out_t res_out,
                              uint32_t M, int D, float eps, float inv_scale) {
  e.pipeline(layernorm_add_fp8_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(codes, 4); e.out(res_out, 5);
  e.bytes(M, 6); e.bytes(eps, 7); e.bytes(inv_scale, 8);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_add_fp8_dyn(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                  typename E::in_t b, typename E::out_t codes, typename E::out_t res_out,
                                  typename E::out_t scale, uint32_t M, int D, float eps) {
  e.pipeline(layernorm_add_fp8_dyn_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(codes, 4); e.out(res_out, 5); e.out(scale, 6);
  e.bytes(M, 7); e.bytes(eps, 8);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- softmax (last axis): x@0 -> o@1 ; M@2(u32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_softmax(E& e, typename E::in_t x, typename E::out_t o, uint32_t M, int D) {
  e.pipeline(softmax_kernel_name(D));
  e.in(x, 0); e.out(o, 1);
  e.bytes(M, 2);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- rotary (split-half RoPE): x@0 cos@1 sin@2 -> o@3 ; N@4(u32) M@5(u32) ;
//        FLAT grid: one thread per 4 rotation pairs (M*D/8 threads, 256/group).
//        x is (M=B*H*N, D) flattened; cos/sin are (N, D/2); row n = row % N. -----
template <class E>
void launch_rotary(E& e, typename E::in_t x, typename E::in_t cos, typename E::in_t sin,
                   typename E::out_t o, uint32_t M, unsigned N, int D, bool interleaved = false) {
  e.pipeline(interleaved ? rotary_interleaved_kernel_name(D) : rotary_kernel_name(D));
  e.in(x, 0); e.in(cos, 1); e.in(sin, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(M, 5);
  const uint32_t total = M * (uint32_t)(D / 8);       // threads = rows * quads-per-row
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- positioned/partial RoPE: x@0 cos@1 sin@2 positions@3 -> o@4;
//       M@5 N@6 H@7 rotary_dim@8 position_batch_stride@9 interleaved@10.
template <class E>
void launch_rotary_positioned(
    E& e, typename E::in_t x, typename E::in_t cos, typename E::in_t sin,
    typename E::in_t positions, typename E::out_t o, uint32_t M, uint32_t N,
    uint32_t heads, int D, uint32_t rotary_dim, uint32_t pos_bstride,
    uint32_t interleaved) {
  e.pipeline(rotary_positioned_kernel_name(D));
  e.in(x, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.out(o, 4);
  e.bytes(M, 5); e.bytes(N, 6); e.bytes(heads, 7); e.bytes(rotary_dim, 8);
  e.bytes(pos_bstride, 9); e.bytes(interleaved, 10);
  const uint32_t total = M * static_cast<uint32_t>(D / 8);
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- 3-axis M-RoPE: same common bindings, section pair counts@10..12 and
//       THW-interleaving flag@13. Pairing is always split-half/NeoX.
template <class E>
void launch_mrope(
    E& e, typename E::in_t x, typename E::in_t cos, typename E::in_t sin,
    typename E::in_t positions, typename E::out_t o, uint32_t M, uint32_t N,
    uint32_t heads, int D, uint32_t rotary_dim, uint32_t pos_bstride,
    uint32_t section_t, uint32_t section_h, uint32_t section_w,
    uint32_t section_interleaved) {
  e.pipeline(mrope_positioned_kernel_name(D));
  e.in(x, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.out(o, 4);
  e.bytes(M, 5); e.bytes(N, 6); e.bytes(heads, 7); e.bytes(rotary_dim, 8);
  e.bytes(pos_bstride, 9); e.bytes(section_t, 10); e.bytes(section_h, 11);
  e.bytes(section_w, 12); e.bytes(section_interleaved, 13);
  const uint32_t total = M * static_cast<uint32_t>(D / 8);
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}

template <class E>
void launch_vision_rope_2d(
    E& e, typename E::in_t x, typename E::in_t cos, typename E::in_t sin,
    typename E::in_t positions, typename E::out_t out,
    uint32_t B, uint32_t H, uint32_t N, int D, uint32_t max_position,
    uint32_t global_split) {
  e.pipeline("vision_rope_2d_D" + std::to_string(D));
  e.in(x, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.out(out, 4);
  const uint32_t rows = B * H * N;
  e.bytes(rows, 5); e.bytes(N, 6); e.bytes(H, 7); e.bytes(max_position, 8);
  e.bytes(global_split, 9);
  const uint32_t total = rows * static_cast<uint32_t>(D / 16);
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- explicit fused Q/K RMSNorm + positioned/partial/M-RoPE over packed QKV.
template <class E>
void launch_qk_norm_rope_positioned(
    E& e, typename E::in_t qkv, typename E::in_t qw, typename E::in_t kw,
    typename E::in_t cos, typename E::in_t sin, typename E::in_t positions,
    typename E::out_t out, int T, int hq, int hk, int hv, int D, float eps,
    int rotary_dim, int interleaved, float weight_offset, int position_mode,
    int section_t, int section_h, int section_w) {
  e.pipeline(qk_norm_rope_positioned_kernel_name(D));
  e.in(qkv, 0); e.in(qw, 1); e.in(kw, 2); e.in(cos, 3); e.in(sin, 4);
  e.in(positions, 5); e.out(out, 6);
  e.bytes(hq, 7); e.bytes(hk, 8); e.bytes(hv, 9); e.bytes(eps, 10);
  e.bytes(rotary_dim, 11); e.bytes(interleaved, 12); e.bytes(weight_offset, 13);
  e.bytes(position_mode, 14); e.bytes(section_t, 15); e.bytes(section_h, 16);
  e.bytes(section_w, 17); e.bytes(T, 18);
  e.dispatch(hq + hk + hv, T, 1, 32, 1, 1);
}

// ----- MLA Q-path: q@0 cos@1 sin@2 positions@3 -> out@4 ; num_heads@5 nope@6 rope@7 norm_mode@8
//        eps@9 ; norm_weight@10 (read iff mode 2) ; grid (M=tokens*heads,1,1) group (32,1,1). -----
template <class E>
void launch_mla_q_norm_rope(E& e, typename E::in_t q, typename E::in_t cos, typename E::in_t sin,
                            typename E::in_t positions, typename E::in_t norm_weight,
                            typename E::out_t out, int M, int num_heads, int nope_dim, int rope_dim,
                            int norm_mode, float eps, int head_dim) {
  e.pipeline(mla_q_norm_rope_kernel_name(head_dim));
  e.in(q, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.out(out, 4);
  e.bytes(num_heads, 5); e.bytes(nope_dim, 6); e.bytes(rope_dim, 7); e.bytes(norm_mode, 8);
  e.bytes(eps, 9); e.in(norm_weight, 10);
  e.dispatch(M, 1, 1, 32, 1, 1);
}

// ----- MLA classic KV insert: kv_c@0 k_pe@1 cos@2 sin@3 positions@4 slot_mapping@5 -> kv_cache@6 ;
//        block_size@7 rope_dim@8 norm_mode@9 eps@10 ; norm_weight@11 ; grid (T,1,1) group (32,1,1). -----
template <class E>
void launch_mla_cache_clone(E& e, typename E::in_t src, typename E::out_t dst, uint64_t n) {
  e.pipeline("mla_cache_clone");
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

template <class E>
void launch_mla_cache_clone_u8(E& e, typename E::in_t src, typename E::out_t dst, uint64_t n) {
  e.pipeline("mla_cache_clone_u8");
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- MLA V4 packed fp8 insert: kv@0(512) cos@1 sin@2 positions@3 slot@4 -> data_cache@5(u8,576)
//        scale_cache@6(u8,8) ; block_size@7 ; grid (T,1,1) group (32,1,1). -----
template <class E>
void launch_mla_kv_insert_fp8(E& e, typename E::in_t kv, typename E::in_t cos, typename E::in_t sin,
                              typename E::in_t positions, typename E::in_t slot_mapping,
                              typename E::out_t data_cache, typename E::out_t scale_cache,
                              int num_tokens, int block_size) {
  e.pipeline("mla_kv_insert_fp8");
  e.in(kv, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.in(slot_mapping, 4);
  e.out(data_cache, 5); e.out(scale_cache, 6); e.bytes(block_size, 7);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

template <class E>
void launch_mla_kv_insert(E& e, typename E::in_t kv_c, typename E::in_t k_pe, typename E::in_t cos,
                          typename E::in_t sin, typename E::in_t positions,
                          typename E::in_t slot_mapping, typename E::out_t kv_cache,
                          typename E::in_t norm_weight, int num_tokens, int block_size,
                          int rope_dim, int norm_mode, float eps, int latent) {
  e.pipeline(mla_kv_insert_kernel_name(latent));
  e.in(kv_c, 0); e.in(k_pe, 1); e.in(cos, 2); e.in(sin, 3); e.in(positions, 4);
  e.in(slot_mapping, 5); e.out(kv_cache, 6);
  e.bytes(block_size, 7); e.bytes(rope_dim, 8); e.bytes(norm_mode, 9); e.bytes(eps, 10);
  e.in(norm_weight, 11);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

// ----- MLA latent decode (MQA): q@0(B,N,QK) kv_cache@1(nb,bs,QK) block_table@2 context_lens@3 ->
//        out@4(B,N,LATENT) ; block_size@5 stride@6 scale@7 num_heads@8 ; grid (num_heads,B) 32 thr. -----
template <class E>
void launch_mla_decode(E& e, typename E::in_t q, typename E::in_t kv_cache,
                       typename E::in_t block_table, typename E::in_t context_lens,
                       typename E::out_t out, int batch, int num_heads, int block_size,
                       int block_table_stride, float scale, int latent, int rope) {
  e.pipeline(mla_decode_kernel_name(latent, rope));
  e.in(q, 0); e.in(kv_cache, 1); e.in(block_table, 2); e.in(context_lens, 3); e.out(out, 4);
  e.bytes(block_size, 5); e.bytes(block_table_stride, 6); e.bytes(scale, 7); e.bytes(num_heads, 8);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- MLA latent decode, partitioned (P4v2): emits paged-v2-style partials
//        tmp_out@4 (B,N,P,LATENT) max_logits@5 exp_sums@6 (both B,N,P) f32; combine with
//        launch_paged_attention_reduce(type "bfloat16", head_size=LATENT). Grid (H, B, P),
//        one simdgroup per (head, partition). -----
template <class E>
void launch_mla_decode_partition(E& e, typename E::in_t q, typename E::in_t kv_cache,
                                 typename E::in_t block_table, typename E::in_t context_lens,
                                 typename E::out_t tmp_out, typename E::out_t max_logits,
                                 typename E::out_t exp_sums, int batch, int num_heads,
                                 int block_size, int block_table_stride, float scale,
                                 int latent, int rope, int num_partitions, int partition_size) {
  e.pipeline("mla_decode_partition_" + std::to_string(latent) + "_" + std::to_string(rope));
  e.in(q, 0); e.in(kv_cache, 1); e.in(block_table, 2); e.in(context_lens, 3);
  e.out(tmp_out, 4); e.out(max_logits, 5); e.out(exp_sums, 6);
  e.bytes(block_size, 7); e.bytes(block_table_stride, 8); e.bytes(scale, 9);
  e.bytes(num_heads, 10); e.bytes(num_partitions, 11); e.bytes(partition_size, 12);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}

// ----- MLA V4 dense fp8 decode: q@0(B,N,512) data@1(u8,576) scale@2(u8,8) block_table@3
//        context_lens@4 -> out@5(B,N,512) ; block_size@6 stride@7 scale@8 num_heads@9 ; grid (N,B) 32 thr. -----
template <class E>
void launch_mla_decode_fp8(E& e, typename E::in_t q, typename E::in_t data_cache,
                           typename E::in_t scale_cache, typename E::in_t block_table,
                           typename E::in_t context_lens, typename E::out_t out, int batch,
                           int num_heads, int block_size, int block_table_stride, float scale) {
  e.pipeline("mla_decode_fp8");
  e.in(q, 0); e.in(data_cache, 1); e.in(scale_cache, 2); e.in(block_table, 3);
  e.in(context_lens, 4); e.out(out, 5);
  e.bytes(block_size, 6); e.bytes(block_table_stride, 7); e.bytes(scale, 8); e.bytes(num_heads, 9);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- MLA V4 sparse fp8 decode: q@0 data@1 scale@2 block_table@3 indices@4 topk_length@5 ->
//        out@6 ; block_size@7 stride@8 scale@9 num_heads@10 max_topk@11 ; grid (N,B) 32 thr. -----
template <class E>
void launch_mla_decode_fp8_sparse(E& e, typename E::in_t q, typename E::in_t data_cache,
                                  typename E::in_t scale_cache, typename E::in_t block_table,
                                  typename E::in_t indices, typename E::in_t topk_length,
                                  typename E::out_t out, int batch, int num_heads, int block_size,
                                  int block_table_stride, float scale, int max_topk) {
  e.pipeline("mla_decode_fp8_sparse");
  e.in(q, 0); e.in(data_cache, 1); e.in(scale_cache, 2); e.in(block_table, 3);
  e.in(indices, 4); e.in(topk_length, 5); e.out(out, 6);
  e.bytes(block_size, 7); e.bytes(block_table_stride, 8); e.bytes(scale, 9);
  e.bytes(num_heads, 10); e.bytes(max_topk, 11);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- MLA fp8 decode, PARTITIONED (P4a-v2/P4b-v2): paged-v2-style partials, combined with
//        launch_paged_attention_reduce("bfloat16", 512). Dense partitions the token range
//        (grid (H,B,P)); sparse partitions the top-k index list. -----
template <class E>
void launch_mla_decode_fp8_partition(E& e, typename E::in_t q, typename E::in_t data_cache,
                                     typename E::in_t scale_cache, typename E::in_t block_table,
                                     typename E::in_t context_lens, typename E::out_t tmp_out,
                                     typename E::out_t max_logits, typename E::out_t exp_sums,
                                     int batch, int num_heads, int block_size,
                                     int block_table_stride, float scale, int num_partitions,
                                     int partition_size) {
  e.pipeline("mla_decode_fp8_partition");
  e.in(q, 0); e.in(data_cache, 1); e.in(scale_cache, 2); e.in(block_table, 3);
  e.in(context_lens, 4); e.out(tmp_out, 5); e.out(max_logits, 6); e.out(exp_sums, 7);
  e.bytes(block_size, 8); e.bytes(block_table_stride, 9); e.bytes(scale, 10);
  e.bytes(num_heads, 11); e.bytes(num_partitions, 12); e.bytes(partition_size, 13);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}
template <class E>
void launch_mla_decode_fp8_sparse_partition(E& e, typename E::in_t q, typename E::in_t data_cache,
                                            typename E::in_t scale_cache, typename E::in_t block_table,
                                            typename E::in_t indices, typename E::in_t topk_length,
                                            typename E::out_t tmp_out, typename E::out_t max_logits,
                                            typename E::out_t exp_sums, int batch, int num_heads,
                                            int block_size, int block_table_stride, float scale,
                                            int max_topk, int num_partitions, int partition_size) {
  e.pipeline("mla_decode_fp8_sparse_partition");
  e.in(q, 0); e.in(data_cache, 1); e.in(scale_cache, 2); e.in(block_table, 3);
  e.in(indices, 4); e.in(topk_length, 5);
  e.out(tmp_out, 6); e.out(max_logits, 7); e.out(exp_sums, 8);
  e.bytes(block_size, 9); e.bytes(block_table_stride, 10); e.bytes(scale, 11);
  e.bytes(num_heads, 12); e.bytes(max_topk, 13); e.bytes(num_partitions, 14);
  e.bytes(partition_size, 15);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}

// ----- gelu (elementwise, last axis): x@0 -> o@1 ; M@2(u32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_gelu(E& e, typename E::in_t x, typename E::out_t o, uint32_t M, int D) {
  e.pipeline(gelu_kernel_name(D));
  e.in(x, 0); e.out(o, 1);
  e.bytes(M, 2);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
// AdamW step: param/grad/m/v -> param_out/m_out/v_out (bias-corr factors bc1,bc2 from the host).
template <class E>
void launch_adamw(E& e, typename E::in_t param, typename E::in_t grad, typename E::in_t m,
                  typename E::in_t v, typename E::out_t param_out, typename E::out_t m_out,
                  typename E::out_t v_out, float lr, float beta1, float beta2, float eps, float wd,
                  float bc1, float bc2, uint32_t n, const std::string& type_name) {
  e.pipeline("adamw_" + type_name);
  e.in(param, 0); e.in(grad, 1); e.in(m, 2); e.in(v, 3);
  e.out(param_out, 4); e.out(m_out, 5); e.out(v_out, 6);
  e.bytes(lr, 7); e.bytes(beta1, 8); e.bytes(beta2, 9); e.bytes(eps, 10); e.bytes(wd, 11);
  e.bytes(bc1, 12); e.bytes(bc2, 13); e.bytes(n, 14);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// Masked AdamW: mask[gid / seg_size] controls cold segments. mask_mode 0 skips the entire update;
// mask_mode 1 applies gradient/moment updates but skips decoupled weight decay on inactive segments.
template <class E>
void launch_adamw_masked(E& e, typename E::in_t param, typename E::in_t grad, typename E::in_t m,
                         typename E::in_t v, typename E::out_t param_out, typename E::out_t m_out,
                         typename E::out_t v_out, float lr, float beta1, float beta2, float eps,
                         float wd, float bc1, float bc2, uint32_t n, typename E::in_t mask,
                         uint32_t seg_size, int mask_mode, const std::string& type_name) {
  e.pipeline("adamw_masked_" + type_name);
  e.in(param, 0); e.in(grad, 1); e.in(m, 2); e.in(v, 3);
  e.out(param_out, 4); e.out(m_out, 5); e.out(v_out, 6);
  e.bytes(lr, 7); e.bytes(beta1, 8); e.bytes(beta2, 9); e.bytes(eps, 10); e.bytes(wd, 11);
  e.bytes(bc1, 12); e.bytes(bc2, 13); e.bytes(n, 14);
  e.in(mask, 15); e.bytes(seg_size, 16); e.bytes(mask_mode, 17);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}
// Dropout fwd/bwd (same signature; bwd selects the backward kernel). out = keep ? x/(1-p) : 0.
template <class E>
void launch_dropout(E& e, typename E::in_t x, typename E::out_t out, uint32_t seed, float p,
                    uint32_t n, bool bwd, const std::string& type_name) {
  e.pipeline((bwd ? "dropout_bwd_" : "dropout_fwd_") + type_name);
  e.in(x, 0); e.out(out, 1);
  e.bytes(seed, 2); e.bytes(p, 3);
  const float inv_keep = 1.0f / (1.0f - p);
  e.bytes(inv_keep, 4); e.bytes(n, 5);
  constexpr int threads = 256;
  const uint32_t nthreads = (n + 3) / 4;              // one thread per 4 elements (vec4)
  e.dispatch(static_cast<int>((nthreads + threads - 1) / threads), 1, 1, threads, 1, 1);
}
template <class E>
void launch_gelu_bwd(E& e, typename E::in_t x, typename E::in_t dy, typename E::out_t dx, int n,
                     const std::string& type_name) {
  e.pipeline("gelu_bwd_" + type_name);
  e.in(x, 0); e.in(dy, 1); e.out(dx, 2); e.bytes(n, 3);
  constexpr int threads = 256;
  const int nthreads = (n + 3) / 4;                 // one thread per 4 elements (vec4)
  e.dispatch((nthreads + threads - 1) / threads, 1, 1, threads, 1, 1);
}

// ----- glu family: x@0 gate@1 -> out@2 ; n@3(uint32) alpha@4 limit@5 ; flat, one thread per
// 4 elements (vec4). Modes mirror llama.cpp's ReGLU/GEGLU/SwiGLU kernels; alpha/limit are only
// used by swiglu_oai. -----
template <class E>
void launch_glu(E& e, typename E::in_t x, typename E::in_t gate, typename E::out_t o,
                uint32_t n, const std::string& mode, const std::string& type_name,
                float alpha, float limit) {
  e.pipeline(glu_kernel_name(mode, type_name));
  e.in(x, 0); e.in(gate, 1); e.out(o, 2);
  e.bytes(n, 3); e.bytes(alpha, 4); e.bytes(limit, 5);
  constexpr int threads = 256;
  const uint32_t nthreads = (n + 3) / 4;
  e.dispatch(static_cast<int>((nthreads + threads - 1) / threads), 1, 1, threads, 1, 1);
}
// GLU backward: da@3, db@4 outputs; da = dc*b*act'(a), db = dc*act(a). vec4 like the forward.
template <class E>
void launch_glu_bwd(E& e, typename E::in_t x, typename E::in_t gate, typename E::in_t dc,
                    typename E::out_t da, typename E::out_t db, uint32_t n, const std::string& mode,
                    const std::string& type_name, float alpha, float limit) {
  e.pipeline("glu_bwd_" + mode + "_" + type_name);
  e.in(x, 0); e.in(gate, 1); e.in(dc, 2); e.out(da, 3); e.out(db, 4);
  e.bytes(n, 5); e.bytes(alpha, 6); e.bytes(limit, 7);
  constexpr int threads = 256;
  const uint32_t nthreads = (n + 3) / 4;
  e.dispatch(static_cast<int>((nthreads + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- LoRA decode/small-batch path: both F16 adapter projections and the
// final scale/base add in one row-owned threadgroup. A is (R,K), B is (N,R). -----
template <class E>
void launch_lora_apply_direct(
    E& e, typename E::in_t x, typename E::in_t A, typename E::in_t B,
    typename E::in_t base, typename E::out_t out,
    int M, int K, int N, int R, float scale, int has_base,
    const std::string& type_name) {
  e.pipeline("lora_apply_direct_f16_" + type_name);
  e.in(x, 0); e.in(A, 1); e.in(B, 2); e.in(base, 3); e.out(out, 4);
  e.bytes(K, 5); e.bytes(N, 6); e.bytes(R, 7); e.bytes(scale, 8);
  e.bytes(has_base, 9);
  e.dispatch(M, 1, 1, 256, 1, 1);
}

// ----- reusable vision/audio 2-D tensor preparation. -----
template <class E>
void launch_extract_patches_2d(
    E& e, typename E::in_t x, typename E::out_t out,
    int B, int H, int W, int C, int KH, int KW, int SH, int SW, int PH, int PW,
    const std::string& type_name) {
  const int OH = (H + 2 * PH - KH) / SH + 1;
  const int OW = (W + 2 * PW - KW) / SW + 1;
  e.pipeline("extract_patches_2d_" + type_name);
  e.in(x, 0); e.out(out, 1); e.bytes(H, 2); e.bytes(W, 3); e.bytes(C, 4);
  e.bytes(OH, 5); e.bytes(OW, 6); e.bytes(KH, 7); e.bytes(KW, 8);
  e.bytes(SH, 9); e.bytes(SW, 10); e.bytes(PH, 11); e.bytes(PW, 12);
  e.dispatch(B * OH * OW, 1, 1, 256, 1, 1);
}
template <class E>
void launch_extract_patches_3d(
    E& e, typename E::in_t x, typename E::out_t out,
    int B, int T, int H, int W, int C, int KT, int KH, int KW,
    int ST, int SH, int SW, int PT, int PH, int PW,
    const std::string& type_name) {
  const int OT = (T + 2 * PT - KT) / ST + 1;
  const int OH = (H + 2 * PH - KH) / SH + 1;
  const int OW = (W + 2 * PW - KW) / SW + 1;
  e.pipeline("extract_patches_3d_" + type_name);
  e.in(x, 0); e.out(out, 1); e.bytes(T, 2); e.bytes(H, 3); e.bytes(W, 4);
  e.bytes(C, 5); e.bytes(OT, 6); e.bytes(OH, 7); e.bytes(OW, 8);
  e.bytes(KT, 9); e.bytes(KH, 10); e.bytes(KW, 11);
  e.bytes(ST, 12); e.bytes(SH, 13); e.bytes(SW, 14);
  e.bytes(PT, 15); e.bytes(PH, 16); e.bytes(PW, 17);
  e.dispatch(B * OT * OH * OW, 1, 1, 256, 1, 1);
}
template <class E>
void launch_interpolate_position_2d(
    E& e, typename E::in_t table, typename E::out_t out,
    int IH, int IW, int OH, int OW, int C, int align_corners,
    const std::string& type_name) {
  e.pipeline("interpolate_position_2d_" + type_name);
  e.in(table, 0); e.out(out, 1); e.bytes(IH, 2); e.bytes(IW, 3);
  e.bytes(OH, 4); e.bytes(OW, 5); e.bytes(C, 6); e.bytes(align_corners, 7);
  e.dispatch(OH * OW, 1, 1, tk_embed_threads(C), 1, 1);
}
template <class E>
void launch_avg_pool2d_tokens(
    E& e, typename E::in_t x, typename E::out_t out,
    int B, int H, int W, int C, int KH, int KW, int SH, int SW, bool ceil_mode,
    const std::string& type_name) {
  const int OH = ceil_mode ? (H - KH + SH - 1) / SH + 1 : (H - KH) / SH + 1;
  const int OW = ceil_mode ? (W - KW + SW - 1) / SW + 1 : (W - KW) / SW + 1;
  e.pipeline("avg_pool2d_tokens_" + type_name);
  e.in(x, 0); e.out(out, 1); e.bytes(H, 2); e.bytes(W, 3); e.bytes(C, 4);
  e.bytes(OH, 5); e.bytes(OW, 6); e.bytes(KH, 7); e.bytes(KW, 8);
  e.bytes(SH, 9); e.bytes(SW, 10);
  e.dispatch(B * OH * OW, 1, 1, tk_embed_threads(C), 1, 1);
}
template <class E>
void launch_factorized_position_2d(
    E& e, typename E::in_t position_ids, typename E::in_t table,
    typename E::in_t valid_mask, typename E::out_t out,
    int B, int N, int P, int D, const std::string& type_name) {
  e.pipeline("factorized_position_2d_" + type_name);
  e.in(position_ids, 0); e.in(table, 1); e.in(valid_mask, 2); e.out(out, 3);
  const int tokens = B * N;
  e.bytes(tokens, 4); e.bytes(P, 5); e.bytes(D, 6);
  const uint64_t total = static_cast<uint64_t>(tokens) * D;
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}
template <class E>
void launch_pool_tokens_by_position(
    E& e, typename E::in_t x, typename E::in_t position_ids,
    typename E::in_t valid_mask, typename E::out_t out,
    typename E::out_t out_mask, int B, int N, int D, int output_length,
    int kernel_size, int source_width, const std::string& type_name) {
  const int64_t output_values = static_cast<int64_t>(B) * output_length * D;
  const int64_t output_tokens = static_cast<int64_t>(B) * output_length;
  e.pipeline("pool_tokens_by_position_zero");
  e.out(out, 0); e.out(out_mask, 1); e.bytes(output_values, 2); e.bytes(output_tokens, 3);
  const int64_t zero_values = std::max(output_values, output_tokens);
  e.dispatch(static_cast<int>((zero_values + 255) / 256), 1, 1, 256, 1, 1);

  e.pipeline("pool_tokens_by_position_scatter_" + type_name);
  e.in(x, 0); e.in(position_ids, 1); e.in(valid_mask, 2); e.out(out, 3); e.out(out_mask, 4);
  e.bytes(B, 5); e.bytes(N, 6); e.bytes(D, 7); e.bytes(output_length, 8);
  e.bytes(kernel_size, 9); e.bytes(source_width, 10);
  const float scale = std::sqrt(static_cast<float>(D)) /
                      static_cast<float>(kernel_size * kernel_size);
  e.bytes(scale, 11);
  const uint64_t total = static_cast<uint64_t>(B) * N * D;
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- audio/sequence convolution. x is NWC; general weights O,K,C and
// depthwise weights C,K. FP32 accumulate, input-dtype output. -----
template <class E>
void launch_audio_conv1d(
    E& e, typename E::in_t x, typename E::in_t weight, typename E::in_t bias,
    typename E::out_t out, int B, int T, int C, int OT, int O, int K,
    int stride, int padding, int dilation, int has_bias,
    const std::string& type_name) {
  e.pipeline("audio_conv1d_" + type_name);
  e.in(x, 0); e.in(weight, 1); e.in(bias, 2); e.out(out, 3);
  e.bytes(T, 4); e.bytes(C, 5); e.bytes(OT, 6); e.bytes(O, 7); e.bytes(K, 8);
  e.bytes(stride, 9); e.bytes(padding, 10); e.bytes(dilation, 11); e.bytes(has_bias, 12);
  e.bytes(B, 13);
  const uint64_t n = static_cast<uint64_t>(B) * OT * O;
  e.dispatch(static_cast<int>((n + 255) / 256), 1, 1, 256, 1, 1);
}
template <class E>
void launch_audio_depthwise_conv1d(
    E& e, typename E::in_t x, typename E::in_t weight, typename E::in_t bias,
    typename E::out_t out, int B, int T, int C, int OT, int K,
    int stride, int padding, int dilation, int has_bias, int activation,
    const std::string& type_name) {
  e.pipeline("audio_depthwise_conv1d_" + type_name);
  e.in(x, 0); e.in(weight, 1); e.in(bias, 2); e.out(out, 3);
  e.bytes(T, 4); e.bytes(C, 5); e.bytes(OT, 6); e.bytes(K, 7); e.bytes(stride, 8);
  e.bytes(padding, 9); e.bytes(dilation, 10); e.bytes(has_bias, 11); e.bytes(activation, 12);
  e.bytes(B, 13);
  const uint64_t n = static_cast<uint64_t>(B) * OT * C;
  e.dispatch(static_cast<int>((n + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- independent-length GQA cross attention. q/out (B,Hq,Tq,D),
// k/v (B,Hkv,Tk,D), one simdgroup per query row. -----
template <class E>
void launch_cross_attention(
    E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
    typename E::in_t key_lengths, typename E::in_t bias, typename E::out_t out,
    int B, int Hq, int Tq, int Hkv, int Tk, int D, float scale, float softcap,
    int has_bias, const std::string& type_name) {
  e.pipeline("cross_attention_D" + std::to_string(D) + "_" + type_name);
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(key_lengths, 3); e.in(bias, 4); e.out(out, 5);
  e.bytes(Tq, 6); e.bytes(Tk, 7); e.bytes(Hq, 8); e.bytes(Hkv, 9);
  e.bytes(scale, 10); e.bytes(softcap, 11); e.bytes(has_bias, 12);
  e.dispatch(Tq, Hq, B, 32, 1, 1);
}

// ----- blocked relative-position attention for Conformer/Gemma audio.
// q/k/v/out are (B,T,H,D), relative_k is (P,H,D). -----
template <class E>
void launch_audio_relative_attention(
    E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
    typename E::in_t relative_k, typename E::in_t per_dim_scale,
    typename E::in_t lengths, typename E::out_t out,
    int B, int T, int H, int D, int P, int chunk_size,
    int left_context, int right_context, float q_scale, float k_scale,
    float softcap, const std::string& type_name) {
  e.pipeline("audio_relative_attention_D" + std::to_string(D) + "_" + type_name);
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(relative_k, 3);
  e.in(per_dim_scale, 4); e.in(lengths, 5); e.out(out, 6);
  e.bytes(T, 7); e.bytes(H, 8); e.bytes(P, 9); e.bytes(chunk_size, 10);
  e.bytes(left_context, 11); e.bytes(right_context, 12); e.bytes(q_scale, 13);
  e.bytes(k_scale, 14); e.bytes(softcap, 15);
  e.dispatch(T, H, B, 32, 1, 1);
}

// ----- Hadamard/FWHT over the final axis: x@0 -> out@1 ; scale@2 nrows@3. D in {64,128,256,512}.
//        One simdgroup per R rows (R=2 at D=64, else 1): in-register + simd_shuffle_xor
//        butterflies. -----
template <class E>
void launch_hadamard(
    E& e,
    typename E::in_t x,
    typename E::out_t out,
    int rows,
    int D,
    float scale,
    const std::string& type_name) {
  e.pipeline(hadamard_kernel_name(type_name, D));
  e.in(x, 0);
  e.out(out, 1);
  e.bytes(scale, 2);
  e.bytes(rows, 3);
  const int R = (D == 64) ? 4 : (D == 128) ? 2 : 1;   // rows per simdgroup = 32/LPR
  e.dispatch((rows + R - 1) / R, 1, 1, 32, 1, 1);
}

// ----- KV cache zero: key_cache@0 value_cache@1 ; n@2(ulong). Flat memset for fresh caches. -----
template <class E>
void launch_kv_cache_zero(
    E& e,
    typename E::out_t key_cache,
    typename E::out_t value_cache,
    uint64_t n,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("zero", type_name));
  e.out(key_cache, 0);
  e.out(value_cache, 1);
  e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- KV cache scatter: key@0 value@1 slot_mapping@2 -> key_cache@3 value_cache@4.
// key/value are (T,H,D); caches are (num_blocks, block_size, H, D). -----
template <class E>
void launch_kv_cache_scatter(
    E& e,
    typename E::in_t key,
    typename E::in_t value,
    typename E::in_t slot_mapping,
    typename E::out_t key_cache,
    typename E::out_t value_cache,
    int num_tokens,
    int num_heads,
    int head_size,
    int block_size,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("scatter", type_name));
  e.in(key, 0);
  e.in(value, 1);
  e.in(slot_mapping, 2);
  e.out(key_cache, 3);
  e.out(value_cache, 4);
  e.bytes(num_heads, 5);
  e.bytes(head_size, 6);
  e.bytes(block_size, 7);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

// ----- KV cache gather: key_cache@0 value_cache@1 -> key_out@2 value_out@3.
// block_table@4 cu_seq_lens@5; outputs are (num_tokens,H,D). -----
template <class E>
void launch_kv_cache_gather(
    E& e,
    typename E::in_t key_cache,
    typename E::in_t value_cache,
    typename E::out_t key_out,
    typename E::out_t value_out,
    typename E::in_t block_table,
    typename E::in_t cu_seq_lens,
    int num_tokens,
    int num_seqs,
    int block_size,
    int block_table_stride,
    int num_heads,
    int head_size,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("gather", type_name));
  e.in(key_cache, 0);
  e.in(value_cache, 1);
  e.out(key_out, 2);
  e.out(value_out, 3);
  e.in(block_table, 4);
  e.in(cu_seq_lens, 5);
  e.bytes(num_tokens, 6);
  e.bytes(num_seqs, 7);
  e.bytes(block_size, 8);
  e.bytes(block_table_stride, 9);
  e.bytes(num_heads, 10);
  e.bytes(head_size, 11);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

// ----- fp8 KV gather + upconvert: uchar caches@0,1 -> bf16 out@2,3 ; block_table@4 cu_seq_lens@5
//        k_scale@6 v_scale@7 (per-kv_head) ; scalars num_tokens@8..head_size@13 fmt@14. -----
template <class E>
void launch_kv_cache_gather_fp8(
    E& e, typename E::in_t key_cache, typename E::in_t value_cache, typename E::out_t key_out,
    typename E::out_t value_out, typename E::in_t block_table, typename E::in_t cu_seq_lens,
    typename E::in_t k_scale, typename E::in_t v_scale, int num_tokens, int num_seqs,
    int block_size, int block_table_stride, int num_heads, int head_size, int fmt,
    const std::string& out_type_name) {
  e.pipeline("kv_cache_gather_fp8_" + out_type_name);
  e.in(key_cache, 0); e.in(value_cache, 1); e.out(key_out, 2); e.out(value_out, 3);
  e.in(block_table, 4); e.in(cu_seq_lens, 5); e.in(k_scale, 6); e.in(v_scale, 7);
  e.bytes(num_tokens, 8); e.bytes(num_seqs, 9); e.bytes(block_size, 10);
  e.bytes(block_table_stride, 11); e.bytes(num_heads, 12); e.bytes(head_size, 13);
  e.bytes(fmt, 14);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

// ----- incremental KV scale update (running max): key@0 value@1 old_k@2 old_v@3 -> new_k@4
//        new_v@5 ; n@6. Single 256-thread threadgroup. -----
template <class E>
void launch_kv_cache_scale_update(
    E& e, typename E::in_t key, typename E::in_t value, typename E::in_t old_key_scale,
    typename E::in_t old_value_scale, typename E::out_t new_key_scale,
    typename E::out_t new_value_scale, uint64_t n, const std::string& type_name) {
  e.pipeline("kv_cache_scale_update_" + type_name);
  e.in(key, 0); e.in(value, 1); e.in(old_key_scale, 2); e.in(old_value_scale, 3);
  e.out(new_key_scale, 4); e.out(new_value_scale, 5); e.bytes(n, 6);
  e.dispatch(1, 1, 1, 256, 1, 1);
}

// ----- KV cache clone: key_cache@0 value_cache@1 -> key_out@2 value_out@3 ; n@4. -----
template <class E>
void launch_kv_cache_clone(
    E& e,
    typename E::in_t key_cache,
    typename E::in_t value_cache,
    typename E::out_t key_out,
    typename E::out_t value_out,
    uint64_t n,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("clone", type_name));
  e.in(key_cache, 0);
  e.in(value_cache, 1);
  e.out(key_out, 2);
  e.out(value_out, 3);
  e.bytes(n, 4);
  constexpr int threads = 256;
  const uint64_t n4 = (n + 3) / 4;   // one thread per vec4 group (kernel handles the scalar tail)
  e.dispatch(static_cast<int>((n4 + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- KV cache block copy: in-place over output caches. mapping is (num_pairs,2) int64. -----
template <class E>
void launch_kv_cache_copy_blocks(
    E& e,
    typename E::in_t key_src,
    typename E::in_t value_src,
    typename E::out_t key_dst,
    typename E::out_t value_dst,
    typename E::in_t block_mapping,
    int num_pairs,
    int numel_per_block,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("copy_blocks", type_name));
  e.in(key_src, 0);
  e.in(value_src, 1);
  e.out(key_dst, 2);
  e.out(value_dst, 3);
  e.in(block_mapping, 4);
  e.bytes(numel_per_block, 5);
  e.dispatch(num_pairs, 1, 1, 256, 1, 1);
}

// Beam KV reorder: build the (src,dst) copy pairs on-device (removes the host readback). Emits a
// fixed (B*BM*max_blocks, 2) long buffer of pairs/sentinels for kv_cache_copy_blocks to consume.
template <class E>
void launch_beam_build_copy_pairs(E& e, typename E::in_t parent_beam, typename E::in_t block_table,
                                  typename E::in_t seq_lens, typename E::out_t pairs, int BM,
                                  int max_blocks, int block_size, int n_slots) {
  e.pipeline("beam_build_copy_pairs");
  e.in(parent_beam, 0); e.in(block_table, 1); e.in(seq_lens, 2); e.out(pairs, 3);
  e.bytes(BM, 4); e.bytes(max_blocks, 5); e.bytes(block_size, 6); e.bytes(n_slots, 7);
  constexpr int threads = 256;
  e.dispatch((n_slots + threads - 1) / threads, 1, 1, threads, 1, 1);
}
// beam_remap_block_table: one threadgroup per beam row, threads copy the block columns (row gather).
template <class E>
void launch_beam_remap_block_table(E& e, typename E::in_t block_table, typename E::in_t parent_beam,
                                   typename E::out_t new_block_table, int nrows, int BM,
                                   int max_blocks) {
  e.pipeline("beam_remap_block_table");
  e.in(block_table, 0); e.in(parent_beam, 1); e.out(new_block_table, 2);
  e.bytes(BM, 3); e.bytes(max_blocks, 4);
  int threads = max_blocks < 32 ? 32 : (max_blocks > 256 ? 256 : max_blocks);
  e.dispatch(nrows, 1, 1, threads, 1, 1);
}

// ----- KV cache scales: key@0 value@1 -> key_scale@2 value_scale@3 ; n@4.
// Single threadgroup scans the arrays and emits absmax / 240, matching vLLM's fp8 scale convention. -----
template <class E>
void launch_kv_cache_scales(
    E& e,
    typename E::in_t key,
    typename E::in_t value,
    typename E::out_t key_scale,
    typename E::out_t value_scale,
    uint64_t n,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("scales", type_name));
  e.in(key, 0);
  e.in(value, 1);
  e.out(key_scale, 2);
  e.out(value_scale, 3);
  e.bytes(n, 4);
  e.dispatch(1, 1, 1, 256, 1, 1);
}

// ----- Paged decode attention: q@0 cacheK@1 cacheV@2 block_table@3 context_lens@4 -> out@5.
// q/out are (B, num_heads, D); caches are (num_blocks, block_size, num_kv_heads, D), D in {64,128}.
// GQA/MQA: num_heads may be a multiple of num_kv_heads (kv_head = head / (num_heads/num_kv_heads)). -----
template <class E>
void launch_paged_attention(
    E& e,
    typename E::in_t q,
    typename E::in_t key_cache,
    typename E::in_t value_cache,
    typename E::in_t block_table,
    typename E::in_t context_lens,
    typename E::out_t out,
    int batch,
    int num_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int block_table_stride,
    float scale,
    typename E::in_t alibi_slopes,
    int use_alibi,
    typename E::in_t block_mask,
    int use_mask,
    int window,
    int mask_heads,
    const std::string& type_name) {
  e.pipeline(paged_attention_kernel_name(type_name, head_size));
  e.in(q, 0);
  e.in(key_cache, 1);
  e.in(value_cache, 2);
  e.in(block_table, 3);
  e.in(context_lens, 4);
  e.out(out, 5);
  e.bytes(block_size, 6);
  e.bytes(block_table_stride, 7);
  e.bytes(scale, 8);
  e.bytes(num_heads, 9);
  e.bytes(num_kv_heads, 10);
  e.in(alibi_slopes, 11);
  e.bytes(use_alibi, 12);
  e.in(block_mask, 13);
  e.bytes(use_mask, 14);
  e.bytes(window, 15);
  e.bytes(mask_heads, 16);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// vLLM x-packed cache decode: same grid as paged_attention; caches use vLLM's memory order,
// x@11 (= 16/sizeof(dtype)) selects the packed head-dim stride.
template <class E>
void launch_paged_attention_xcache(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens, typename E::out_t out,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, int x, const std::string& type_name) {
  e.pipeline(paged_attention_xcache_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4); e.out(out, 5);
  e.bytes(block_size, 6); e.bytes(block_table_stride, 7); e.bytes(scale, 8);
  e.bytes(num_heads, 9); e.bytes(num_kv_heads, 10); e.bytes(x, 11);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// GQA KV-reuse staged decode: grid (num_kv_heads, batch, 1), threadgroup (32, group_size, 1) —
// group_size simdgroups share one staged KV vector. Same buffer ABI as launch_paged_attention.
template <class E>
void launch_paged_attention_gqa_staged(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens, typename E::out_t out,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, const std::string& type_name) {
  const int group_size = num_heads / num_kv_heads;
  e.pipeline(paged_attention_gqa_staged_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4); e.out(out, 5);
  e.bytes(block_size, 6); e.bytes(block_table_stride, 7); e.bytes(scale, 8);
  e.bytes(num_heads, 9); e.bytes(num_kv_heads, 10);
  e.dispatch(num_kv_heads, batch, 1, 32, group_size, 1);
}

// ----- fp8 KV cache: zero (uint8), scatter-with-encode, and dequant-on-read paged attention. -----
template <class E>
void launch_kv_cache_zero_u8(E& e, typename E::out_t key_cache, typename E::out_t value_cache,
                             uint64_t n) {
  e.pipeline("kv_cache_zero_u8");
  e.out(key_cache, 0); e.out(value_cache, 1); e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

template <class E>
void launch_kv_cache_scatter_fp8(E& e, typename E::in_t key, typename E::in_t value,
                                 typename E::in_t slot_mapping, typename E::out_t key_cache,
                                 typename E::out_t value_cache, int num_tokens, int num_heads,
                                 int head_size, int block_size, typename E::in_t k_scale,
                                 typename E::in_t v_scale, int fmt, const std::string& type_name) {
  e.pipeline(kv_cache_scatter_fp8_kernel_name(type_name));
  e.in(key, 0); e.in(value, 1); e.in(slot_mapping, 2);
  e.out(key_cache, 3); e.out(value_cache, 4);
  e.bytes(num_heads, 5); e.bytes(head_size, 6); e.bytes(block_size, 7);
  e.in(k_scale, 8); e.in(v_scale, 9); e.bytes(fmt, 10);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

template <class E>
void launch_paged_attention_fp8(E& e, typename E::in_t q, typename E::in_t key_cache,
                                typename E::in_t value_cache, typename E::in_t block_table,
                                typename E::in_t context_lens, typename E::out_t out,
                                int batch, int num_heads, int num_kv_heads, int head_size,
                                int block_size, int block_table_stride, float scale,
                                typename E::in_t k_scale, typename E::in_t v_scale, int fmt,
                                int window, const std::string& type_name) {
  e.pipeline(paged_attention_fp8_kernel_name(type_name, head_size, fmt));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4); e.out(out, 5);
  e.bytes(block_size, 6); e.bytes(block_table_stride, 7); e.bytes(scale, 8);
  e.bytes(num_heads, 9); e.bytes(num_kv_heads, 10);
  e.in(k_scale, 11); e.in(v_scale, 12); e.bytes(fmt, 13); e.bytes(window, 14);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- QuixiCore Q8_0 KV ABI: int8 code planes (nb,bs,H,D) plus fp16 scale
//       planes (nb,bs,H,D/32). -----
template <class E>
void launch_kv_cache_zero_q8_0(
    E& e, typename E::out_t key_codes, typename E::out_t key_scales,
    typename E::out_t value_codes, typename E::out_t value_scales,
    uint64_t code_n, uint64_t scale_n) {
  e.pipeline("kv_cache_zero_q8_0");
  e.out(key_codes, 0); e.out(key_scales, 1); e.out(value_codes, 2); e.out(value_scales, 3);
  e.bytes(code_n, 4); e.bytes(scale_n, 5);
  constexpr int threads = 256;
  const uint64_t n = code_n > scale_n ? code_n : scale_n;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

template <class E>
void launch_kv_cache_scatter_q8_0(
    E& e, typename E::in_t key, typename E::in_t value, typename E::in_t slot_mapping,
    typename E::out_t key_codes, typename E::out_t key_scales,
    typename E::out_t value_codes, typename E::out_t value_scales,
    int num_tokens, int num_heads, int head_size, int block_size,
    const std::string& type_name) {
  e.pipeline("kv_cache_scatter_q8_0_" + type_name);
  e.in(key, 0); e.in(value, 1); e.in(slot_mapping, 2);
  e.out(key_codes, 3); e.out(key_scales, 4); e.out(value_codes, 5); e.out(value_scales, 6);
  e.bytes(num_heads, 7); e.bytes(head_size, 8); e.bytes(block_size, 9);
  e.dispatch(head_size / 32, num_heads, num_tokens, 32, 1, 1);
}

template <class E>
void launch_kv_cache_gather_q8_0(
    E& e, typename E::in_t key_codes, typename E::in_t key_scales,
    typename E::in_t value_codes, typename E::in_t value_scales,
    typename E::out_t key_out, typename E::out_t value_out,
    typename E::in_t block_table, typename E::in_t cu_seq_lens,
    int num_tokens, int num_seqs, int block_size, int block_table_stride,
    int num_heads, int head_size, const std::string& out_type_name) {
  e.pipeline("kv_cache_gather_q8_0_" + out_type_name);
  e.in(key_codes, 0); e.in(key_scales, 1); e.in(value_codes, 2); e.in(value_scales, 3);
  e.out(key_out, 4); e.out(value_out, 5); e.in(block_table, 6); e.in(cu_seq_lens, 7);
  e.bytes(num_tokens, 8); e.bytes(num_seqs, 9); e.bytes(block_size, 10);
  e.bytes(block_table_stride, 11); e.bytes(num_heads, 12); e.bytes(head_size, 13);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

template <class E>
void launch_kv_cache_clone_q8_0(
    E& e, typename E::in_t key_codes, typename E::in_t key_scales,
    typename E::in_t value_codes, typename E::in_t value_scales,
    typename E::out_t key_codes_out, typename E::out_t key_scales_out,
    typename E::out_t value_codes_out, typename E::out_t value_scales_out,
    uint64_t code_n, uint64_t scale_n) {
  e.pipeline("kv_cache_clone_q8_0");
  e.in(key_codes, 0); e.in(key_scales, 1); e.in(value_codes, 2); e.in(value_scales, 3);
  e.out(key_codes_out, 4); e.out(key_scales_out, 5);
  e.out(value_codes_out, 6); e.out(value_scales_out, 7);
  e.bytes(code_n, 8); e.bytes(scale_n, 9);
  constexpr int threads = 256;
  const uint64_t n = code_n > scale_n ? code_n : scale_n;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

template <class E>
void launch_kv_cache_copy_blocks_q8_0(
    E& e, typename E::in_t key_codes, typename E::in_t key_scales,
    typename E::in_t value_codes, typename E::in_t value_scales,
    typename E::out_t key_codes_out, typename E::out_t key_scales_out,
    typename E::out_t value_codes_out, typename E::out_t value_scales_out,
    typename E::in_t block_mapping, int num_pairs, int codes_per_block,
    int scales_per_block) {
  e.pipeline("kv_cache_copy_blocks_q8_0");
  e.in(key_codes, 0); e.in(key_scales, 1); e.in(value_codes, 2); e.in(value_scales, 3);
  e.out(key_codes_out, 4); e.out(key_scales_out, 5);
  e.out(value_codes_out, 6); e.out(value_scales_out, 7); e.in(block_mapping, 8);
  e.bytes(codes_per_block, 9); e.bytes(scales_per_block, 10);
  e.dispatch(num_pairs, 1, 1, 256, 1, 1);
}

template <class E>
void launch_paged_attention_q8_0(
    E& e, typename E::in_t q, typename E::in_t key_codes, typename E::in_t key_scales,
    typename E::in_t value_codes, typename E::in_t value_scales,
    typename E::in_t block_table, typename E::in_t context_lens, typename E::out_t out,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, int window, const std::string& type_name) {
  e.pipeline("paged_attention_q8_0_" + type_name + "_" + std::to_string(head_size));
  e.in(q, 0); e.in(key_codes, 1); e.in(key_scales, 2);
  e.in(value_codes, 3); e.in(value_scales, 4); e.in(block_table, 5); e.in(context_lens, 6);
  e.out(out, 7); e.bytes(block_size, 8); e.bytes(block_table_stride, 9); e.bytes(scale, 10);
  e.bytes(num_heads, 11); e.bytes(num_kv_heads, 12); e.bytes(window, 13);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- Paged attention v2 partition: q@0 cacheK@1 cacheV@2 block_table@3 context_lens@4 ->
//        tmp_out@5 max_logits@6 exp_sums@7 (all fp32) ; scalars 8..14 ; grid (H, B, P), 32 thr.
//        Each (head,batch,partition) does a local softmax over its KV slice. GQA-aware. -----
template <class E>
void launch_paged_attention_partition(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens,
    typename E::out_t tmp_out, typename E::out_t max_logits, typename E::out_t exp_sums,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, int num_partitions, int partition_size,
    int window, float softcap, const std::string& type_name) {
  e.pipeline(paged_attention_partition_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4);
  e.out(tmp_out, 5); e.out(max_logits, 6); e.out(exp_sums, 7);
  e.bytes(block_size, 8); e.bytes(block_table_stride, 9); e.bytes(scale, 10);
  e.bytes(num_heads, 11); e.bytes(num_kv_heads, 12);
  e.bytes(num_partitions, 13); e.bytes(partition_size, 14); e.bytes(window, 15);
  e.bytes(softcap, 16);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}

// Cascade prefix partition: a shared CONTIGUOUS prefix KV (prefix_k/prefix_v (prefix_len,H_KV,D)),
// emits (m,l,o) partials in the same layout as paged_attention_partition. Reduce is reused.
template <class E>
void launch_cascade_prefix_partition(
    E& e, typename E::in_t q, typename E::in_t prefix_k, typename E::in_t prefix_v,
    typename E::out_t tmp_out, typename E::out_t max_logits, typename E::out_t exp_sums,
    int batch, int num_heads, int num_kv_heads, int head_size, int prefix_len, float scale,
    int num_partitions, int partition_size, const std::string& type_name) {
  e.pipeline("cascade_prefix_partition_" + type_name + "_" + std::to_string(head_size));
  e.in(q, 0); e.in(prefix_k, 1); e.in(prefix_v, 2);
  e.out(tmp_out, 3); e.out(max_logits, 4); e.out(exp_sums, 5);
  e.bytes(prefix_len, 6); e.bytes(scale, 7); e.bytes(num_heads, 8); e.bytes(num_kv_heads, 9);
  e.bytes(num_partitions, 10); e.bytes(partition_size, 11);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}

// fp8 partition: uint8 caches, per-head k_scale/v_scale@15,16 (in), fmt@17. Reduce is reused.
template <class E>
void launch_paged_attention_partition_fp8(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens,
    typename E::out_t tmp_out, typename E::out_t max_logits, typename E::out_t exp_sums,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, int num_partitions, int partition_size,
    typename E::in_t k_scale, typename E::in_t v_scale, int fmt, int window, float softcap,
    const std::string& type_name) {
  e.pipeline(paged_attention_partition_fp8_kernel_name(type_name, head_size, fmt));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4);
  e.out(tmp_out, 5); e.out(max_logits, 6); e.out(exp_sums, 7);
  e.bytes(block_size, 8); e.bytes(block_table_stride, 9); e.bytes(scale, 10);
  e.bytes(num_heads, 11); e.bytes(num_kv_heads, 12);
  e.bytes(num_partitions, 13); e.bytes(partition_size, 14);
  e.in(k_scale, 15); e.in(v_scale, 16); e.bytes(fmt, 17); e.bytes(window, 18);
  e.bytes(softcap, 19);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}
// fp8 cascade prefix partition (uint8 shared prefix KV + per-kv-head dequant on read).
template <class E>
void launch_cascade_prefix_partition_fp8(
    E& e, typename E::in_t q, typename E::in_t prefix_k, typename E::in_t prefix_v,
    typename E::out_t tmp_out, typename E::out_t max_logits, typename E::out_t exp_sums,
    int batch, int num_heads, int num_kv_heads, int head_size, int prefix_len, float scale,
    int num_partitions, int partition_size, typename E::in_t k_scale, typename E::in_t v_scale,
    int fmt, const std::string& type_name) {
  e.pipeline("cascade_prefix_partition_fp8_" + std::string(fmt == 1 ? "e5m2_" : "e4m3_") +
             type_name + "_" + std::to_string(head_size));
  e.in(q, 0); e.in(prefix_k, 1); e.in(prefix_v, 2);
  e.out(tmp_out, 3); e.out(max_logits, 4); e.out(exp_sums, 5);
  e.bytes(prefix_len, 6); e.bytes(scale, 7); e.bytes(num_heads, 8); e.bytes(num_kv_heads, 9);
  e.bytes(num_partitions, 10); e.bytes(partition_size, 11);
  e.in(k_scale, 12); e.in(v_scale, 13); e.bytes(fmt, 14);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}

// ----- Paged attention v2 reduce: tmp_out@0 max_logits@1 exp_sums@2 (fp32) -> out@3 ;
//        num_heads@4 num_partitions@5 sinks@6(f32*) has_sink@7(i32) ; grid (H, B, 1), 32 thr.
//        LSE merge over partitions. The attention sink (gpt-oss) enters the denominator HERE,
//        exactly once — never in the partition kernels. sinks read only when has_sink; pass
//        tmp_out as the placeholder binding otherwise. -----
template <class E>
void launch_paged_attention_reduce(
    E& e, typename E::in_t tmp_out, typename E::in_t max_logits, typename E::in_t exp_sums,
    typename E::out_t out, int batch, int num_heads, int head_size, int num_partitions,
    typename E::in_t sinks, int has_sink, const std::string& type_name) {
  e.pipeline(paged_attention_reduce_kernel_name(type_name, head_size));
  e.in(tmp_out, 0); e.in(max_logits, 1); e.in(exp_sums, 2); e.out(out, 3);
  e.bytes(num_heads, 4); e.bytes(num_partitions, 5);
  e.in(sinks, 6); e.bytes(has_sink, 7);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- attn_causal: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) softcap@6(f32) sinks@7(f32*)
//        has_sink@8(i32) ; grid (N/8, H, B) group (32,1,1) -----
// Same as attn_fwd but with causal masking (lower-triangular). softcap <= 0 off; sinks read
// only when has_sink (pass q as the placeholder binding otherwise).
template <class E>
void launch_attn_causal(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                        typename E::out_t o, unsigned N, unsigned H, int B, int D,
                        float softcap, typename E::in_t sinks, int has_sink) {
  e.pipeline(attn_causal_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.bytes(softcap, 6); e.in(sinks, 7); e.bytes(has_sink, 8);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- attn_window: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) window@6(i32) softcap@7(f32)
//        sinks@8(f32*) has_sink@9(i32) ; grid (N/8, H, B) group (32,1,1).
//        Sliding-window causal (W most recent keys incl self). -----
template <class E>
void launch_attn_window(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                        typename E::out_t o, unsigned N, unsigned H, int B, int D, int window,
                        float softcap, typename E::in_t sinks, int has_sink) {
  e.pipeline("attn_window_" + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5); e.bytes(window, 6);
  e.bytes(softcap, 7); e.in(sinks, 8); e.bytes(has_sink, 9);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- attn_varlen_prefill: q_hm@0 key_cache@1 value_cache@2 block_table@3 context_lens@4
//        tile_seq@5 tile_local0@6 seq_qlen@7 -> o_hm@8 ; total_padded@9 H@10 H_KV@11
//        block_size@12 bt_stride@13 (i32) scale@14 (f32) ; grid (n_tiles, H, 1) group (32,1,1).
//        Ragged causal prefill reading K/V from the paged cache; q/o are head-major (H,tp,D). -----
template <class E>
void launch_attn_varlen_prefill(E& e, typename E::in_t q_hm, typename E::in_t key_cache,
                                typename E::in_t value_cache, typename E::in_t block_table,
                                typename E::in_t context_lens, typename E::in_t tile_seq,
                                typename E::in_t tile_local0, typename E::in_t seq_qlen,
                                typename E::out_t o_hm, int n_tiles, int total_padded, int H,
                                int H_KV, int block_size, int bt_stride, float scale, int D,
                                float softcap, typename E::in_t sinks, int has_sink) {
  e.pipeline("attn_varlen_prefill_" + std::to_string(D));
  e.in(q_hm, 0); e.in(key_cache, 1); e.in(value_cache, 2); e.in(block_table, 3);
  e.in(context_lens, 4); e.in(tile_seq, 5); e.in(tile_local0, 6); e.in(seq_qlen, 7);
  e.out(o_hm, 8);
  e.bytes(total_padded, 9); e.bytes(H, 10); e.bytes(H_KV, 11);
  e.bytes(block_size, 12); e.bytes(bt_stride, 13); e.bytes(scale, 14);
  e.bytes(softcap, 15); e.in(sinks, 16); e.bytes(has_sink, 17);
  e.dispatch(n_tiles, H, 1, 32, 1, 1);
}
// Device varlen Q pad/gather (packed -> head-major) and output re-gather (head-major -> packed).
// bwd=false = pad/gather into q_hm; bwd=true = regather o_hm into o_packed. One threadgroup/token.
template <class E>
void launch_varlen_pack_gather(E& e, typename E::in_t src, typename E::in_t cu_seqlens,
                               typename E::in_t pad_off, typename E::out_t dst, int total_q, int B,
                               int H, int D, int total_padded, bool regather) {
  e.pipeline(regather ? "varlen_o_regather" : "varlen_q_pad_gather");
  e.in(src, 0); e.in(cu_seqlens, 1); e.in(pad_off, 2); e.out(dst, 3);
  e.bytes(B, 4); e.bytes(H, 5); e.bytes(D, 6); e.bytes(total_padded, 7);
  int threads = D < 32 ? 32 : (D > 256 ? 256 : D);
  // pad/gather grids over padded positions (writes every row incl. zero-pad); regather over tokens.
  e.dispatch(regather ? total_q : total_padded, 1, 1, threads, 1, 1);
}

// Build the varlen prefill worklist on-device from cu_seqlens. One threadgroup, round32(B) threads.
template <class E>
void launch_varlen_build_worklist(E& e, typename E::in_t cu_seqlens, typename E::out_t qlens,
                                  typename E::out_t pad_off, typename E::out_t tile_seq,
                                  typename E::out_t tile_local0, typename E::out_t n_tiles,
                                  int B, int max_tiles) {
  e.pipeline("varlen_build_worklist");
  e.in(cu_seqlens, 0); e.out(qlens, 1); e.out(pad_off, 2); e.out(tile_seq, 3);
  e.out(tile_local0, 4); e.out(n_tiles, 5); e.bytes(B, 6); e.bytes(max_tiles, 7);
  int nthreads = ((B + 31) / 32) * 32;
  if (nthreads < 32) nthreads = 32;
  if (nthreads > 256) nthreads = 256;
  e.dispatch(1, 1, 1, nthreads, 1, 1);
}

// ----- lm_head fused LM-head + sampling (two-stage partition/reduce) -----
// argcat partials: h@0 W@1 -> part_val@2 part_id@3 ; bias@4 ; V@5 K@6 TILE_V@7 num_vtiles@8 (i32)
//   invtemp@9 (f32) seed@10 (u32) use_gumbel@11 use_bias@12 (i32) ; grid (num_vtiles, T) × 32.
template <class E>
void launch_lm_head_argcat_partials(E& e, typename E::in_t h, typename E::in_t W,
                                    typename E::out_t part_val, typename E::out_t part_id,
                                    typename E::in_t bias, int V, int K, int TILE_V, int num_vtiles,
                                    float invtemp, unsigned seed, int use_gumbel, int use_bias,
                                    int T, const std::string& t) {
  e.pipeline("lm_head_argcat_partials_" + t);
  e.in(h, 0); e.in(W, 1); e.out(part_val, 2); e.out(part_id, 3); e.in(bias, 4);
  e.bytes(V, 5); e.bytes(K, 6); e.bytes(TILE_V, 7); e.bytes(num_vtiles, 8);
  e.bytes(invtemp, 9); e.bytes(seed, 10); e.bytes(use_gumbel, 11); e.bytes(use_bias, 12);
  e.dispatch(num_vtiles, T, 1, 32, 1, 1);
}

// argcat partials over QUANTIZED weights: Wq packed uchar@1 (dequant-on-read); pipeline
// "lm_head_argcat_partials_q_<fmt>_<htype>". Same buffers/grid as the dense argcat partials.
template <class E>
void launch_lm_head_argcat_partials_q(E& e, typename E::in_t h, typename E::in_t Wq,
                                      typename E::out_t part_val, typename E::out_t part_id,
                                      typename E::in_t bias, int V, int K, int TILE_V,
                                      int num_vtiles, float invtemp, unsigned seed, int use_gumbel,
                                      int use_bias, int T, const std::string& fmt,
                                      const std::string& htype) {
  e.pipeline("lm_head_argcat_partials_q_" + fmt + "_" + htype);
  e.in(h, 0); e.in(Wq, 1); e.out(part_val, 2); e.out(part_id, 3); e.in(bias, 4);
  e.bytes(V, 5); e.bytes(K, 6); e.bytes(TILE_V, 7); e.bytes(num_vtiles, 8);
  e.bytes(invtemp, 9); e.bytes(seed, 10); e.bytes(use_gumbel, 11); e.bytes(use_bias, 12);
  e.dispatch(num_vtiles, T, 1, 32, 1, 1);
}

// argcat reduce: part_val@0 part_id@1 -> out_idx@2 ; num_vtiles@3 (i32) ; grid (T,) × 32.
template <class E>
void launch_lm_head_argcat_reduce(E& e, typename E::in_t part_val, typename E::in_t part_id,
                                  typename E::out_t out_idx, int num_vtiles, int T) {
  e.pipeline("lm_head_argcat_reduce");
  e.in(part_val, 0); e.in(part_id, 1); e.out(out_idx, 2); e.bytes(num_vtiles, 3);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

template <class E>
void launch_lm_head_constrained_partials(
    E& e, typename E::in_t h, typename E::in_t W, typename E::in_t bias,
    typename E::in_t forbidden, typename E::in_t previous,
    typename E::out_t part_max, typename E::out_t part_sum,
    typename E::out_t part_best, typename E::out_t part_id,
    int V, int K, int TILE_V, int num_vtiles, int use_bias,
    int eos_id, int forbid_eos, int T, const std::string& type_name) {
  e.pipeline("lm_head_constrained_partials_" + type_name);
  e.in(h, 0); e.in(W, 1); e.in(bias, 2); e.in(forbidden, 3); e.in(previous, 4);
  e.out(part_max, 5); e.out(part_sum, 6); e.out(part_best, 7); e.out(part_id, 8);
  e.bytes(V, 9); e.bytes(K, 10); e.bytes(TILE_V, 11); e.bytes(num_vtiles, 12);
  e.bytes(use_bias, 13); e.bytes(eos_id, 14); e.bytes(forbid_eos, 15);
  e.dispatch(num_vtiles, T, 1, 32, 1, 1);
}

template <class E>
void launch_lm_head_constrained_reduce(
    E& e, typename E::in_t part_max, typename E::in_t part_sum,
    typename E::in_t part_best, typename E::in_t part_id,
    typename E::out_t out_token, typename E::out_t out_logprob,
    int num_vtiles, int T) {
  e.pipeline("lm_head_constrained_reduce");
  e.in(part_max, 0); e.in(part_sum, 1); e.in(part_best, 2); e.in(part_id, 3);
  e.out(out_token, 4); e.out(out_logprob, 5); e.bytes(num_vtiles, 6);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

template <class E>
void launch_lm_head_masked_partials(
    E& e, typename E::in_t h, typename E::in_t weight,
    typename E::in_t bias, typename E::in_t allow_mask,
    typename E::out_t part_val, typename E::out_t part_id,
    typename E::out_t part_max, typename E::out_t part_sum,
    int V, int K, int tile_v, int num_vtiles, int topk, bool use_bias,
    bool normalize_allowed, int mask_words, int T,
    const std::string& format, const std::string& type_name) {
  const std::string layout = format.empty() ? "dense" : format;
  e.pipeline("lm_head_masked_partials_" + layout + "_" + type_name);
  e.in(h, 0); e.in(weight, 1); e.in(bias, 2); e.in(allow_mask, 3);
  e.out(part_val, 4); e.out(part_id, 5); e.out(part_max, 6); e.out(part_sum, 7);
  e.bytes(V, 8); e.bytes(K, 9); e.bytes(tile_v, 10); e.bytes(num_vtiles, 11);
  e.bytes(topk, 12); e.bytes(use_bias ? 1 : 0, 13);
  e.bytes(normalize_allowed ? 1 : 0, 14); e.bytes(mask_words, 15);
  e.dispatch(num_vtiles, T, 1, 32, 1, 1);
}

template <class E>
void launch_lm_head_masked_reduce(
    E& e, typename E::in_t part_val, typename E::in_t part_id,
    typename E::in_t part_max, typename E::in_t part_sum,
    typename E::out_t out_id, typename E::out_t out_logprob,
    int num_vtiles, int topk, int T) {
  e.pipeline("lm_head_masked_reduce");
  e.in(part_val, 0); e.in(part_id, 1); e.in(part_max, 2); e.in(part_sum, 3);
  e.out(out_id, 4); e.out(out_logprob, 5);
  e.bytes(num_vtiles, 6); e.bytes(topk, 7);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

template <class E>
void launch_lm_head_candidates(
    E& e, typename E::in_t h, typename E::in_t weight,
    typename E::in_t candidate_ids, typename E::in_t offsets,
    typename E::in_t bias, typename E::out_t out_id,
    typename E::out_t out_logprob, int V, int K, int topk,
    bool use_bias, int T, const std::string& format,
    const std::string& type_name) {
  const std::string layout = format.empty() ? "dense" : format;
  e.pipeline("lm_head_candidates_" + layout + "_" + type_name);
  e.in(h, 0); e.in(weight, 1); e.in(candidate_ids, 2); e.in(offsets, 3);
  e.in(bias, 4); e.out(out_id, 5); e.out(out_logprob, 6);
  e.bytes(V, 7); e.bytes(K, 8); e.bytes(topk, 9); e.bytes(use_bias ? 1 : 0, 10);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

// topk partials: h@0 W@1 -> part_val@2 part_id@3 ; bias@4 ; V@5 K@6 TILE_V@7 num_vtiles@8 topk@9
//   use_bias@10 (i32) ; grid (num_vtiles, T) × 32.
template <class E>
void launch_lm_head_topk_partials(E& e, typename E::in_t h, typename E::in_t W,
                                  typename E::out_t part_val, typename E::out_t part_id,
                                  typename E::in_t bias, int V, int K, int TILE_V, int num_vtiles,
                                  int topk, int use_bias, int T, const std::string& t) {
  e.pipeline("lm_head_topk_partials_" + t);
  e.in(h, 0); e.in(W, 1); e.out(part_val, 2); e.out(part_id, 3); e.in(bias, 4);
  e.bytes(V, 5); e.bytes(K, 6); e.bytes(TILE_V, 7); e.bytes(num_vtiles, 8);
  e.bytes(topk, 9); e.bytes(use_bias, 10);
  e.dispatch(num_vtiles, T, 1, 32, 1, 1);
}
template <class E>
void launch_lm_head_topk_partials_q(E& e, typename E::in_t h, typename E::in_t Wq,
                                    typename E::out_t part_val, typename E::out_t part_id,
                                    typename E::in_t bias, int V, int K, int TILE_V, int num_vtiles,
                                    int topk, int use_bias, int T, const std::string& fmt,
                                    const std::string& htype) {
  e.pipeline("lm_head_topk_partials_q_" + fmt + "_" + htype);
  e.in(h, 0); e.in(Wq, 1); e.out(part_val, 2); e.out(part_id, 3); e.in(bias, 4);
  e.bytes(V, 5); e.bytes(K, 6); e.bytes(TILE_V, 7); e.bytes(num_vtiles, 8);
  e.bytes(topk, 9); e.bytes(use_bias, 10);
  e.dispatch(num_vtiles, T, 1, 32, 1, 1);
}

// quant top-p partials: like topk_partials_q + a per-tile tempered logsumexp (part_lse@12, invtemp@11).
template <class E>
void launch_lm_head_topp_partials_q(E& e, typename E::in_t h, typename E::in_t Wq,
                                    typename E::out_t part_val, typename E::out_t part_id,
                                    typename E::in_t bias, int V, int K, int TILE_V, int num_vtiles,
                                    int topk, int use_bias, float invtemp,
                                    typename E::out_t part_lse, int T, const std::string& fmt,
                                    const std::string& htype, int rows_per_tg = 1) {
  e.pipeline("lm_head_topp_partials_q_" + fmt + "_" + htype);
  e.in(h, 0); e.in(Wq, 1); e.out(part_val, 2); e.out(part_id, 3); e.in(bias, 4);
  e.bytes(V, 5); e.bytes(K, 6); e.bytes(TILE_V, 7); e.bytes(num_vtiles, 8);
  e.bytes(topk, 9); e.bytes(use_bias, 10); e.bytes(invtemp, 11); e.out(part_lse, 12);
  e.bytes(rows_per_tg, 13); e.bytes(T, 14);
  e.dispatch(num_vtiles, (T + rows_per_tg - 1) / rows_per_tg, 1,
             32 * rows_per_tg, 1, 1);
}

// topk reduce: part_val@0 part_id@1 -> out_idx@2 ; num_vtiles@3 topk@4 (i32) seed@5 (u32)
//   invtemp@6 (f32) ; grid (T,) × 32.
template <class E>
void launch_lm_head_topk_reduce(E& e, typename E::in_t part_val, typename E::in_t part_id,
                                typename E::out_t out_idx, int num_vtiles, int topk, unsigned seed,
                                float invtemp, int T) {
  e.pipeline("lm_head_topk_reduce");
  e.in(part_val, 0); e.in(part_id, 1); e.out(out_idx, 2);
  e.bytes(num_vtiles, 3); e.bytes(topk, 4); e.bytes(seed, 5); e.bytes(invtemp, 6);
  e.dispatch(T, 1, 1, 32, 1, 1);
}
template <class E>
void launch_lm_head_topp_reduce(E& e, typename E::in_t part_val, typename E::in_t part_id,
                                typename E::out_t out_idx, int num_vtiles, int topk, float p,
                                unsigned seed, float invtemp, typename E::in_t part_lse, int T) {
  e.pipeline("lm_head_topp_reduce");
  e.in(part_val, 0); e.in(part_id, 1); e.out(out_idx, 2);
  e.bytes(num_vtiles, 3); e.bytes(topk, 4); e.bytes(p, 5); e.bytes(seed, 6); e.bytes(invtemp, 7);
  e.in(part_lse, 8);
  e.dispatch(T, 1, 1, 32, 1, 1);
}

// Quantized LM-head beam merge: tile top candidates + tile lse + cumulative
// score -> exact per-row top-2BM child scores, grid (B*BM,) x 32.
template <class E>
void launch_lm_head_beam_reduce(
    E& e, typename E::in_t part_val, typename E::in_t part_id,
    typename E::in_t part_lse, typename E::in_t cum,
    typename E::out_t cand_score, typename E::out_t cand_token,
    int num_vtiles, int two_bm, int rows) {
  e.pipeline("lm_head_beam_reduce");
  e.in(part_val, 0); e.in(part_id, 1); e.in(part_lse, 2); e.in(cum, 3);
  e.out(cand_score, 4); e.out(cand_token, 5);
  e.bytes(num_vtiles, 6); e.bytes(two_bm, 7);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- cross_entropy fwd: logits@0 targets@1 -> loss@2 lse@3 ; V@4 ignore_index@5 (i32)
//        label_smoothing@6 z_loss@7 (f32) ; grid (T,) × 32 (one simdgroup per row). -----
template <class E>
void launch_cross_entropy_fwd(E& e, typename E::in_t logits, typename E::in_t targets,
                              typename E::out_t loss, typename E::out_t lse, int V,
                              int ignore_index, float label_smoothing, float z_loss, float softcap,
                              int T, bool mw, const std::string& t) {
  e.pipeline(mw ? "cross_entropy_fwd_mw_" + t : "cross_entropy_fwd_" + t);
  e.in(logits, 0); e.in(targets, 1); e.out(loss, 2); e.out(lse, 3);
  e.bytes(V, 4); e.bytes(ignore_index, 5); e.bytes(label_smoothing, 6); e.bytes(z_loss, 7);
  e.bytes(softcap, 8);
  e.dispatch(T, 1, 1, mw ? 128 : 32, 1, 1);
}

// ----- cross_entropy bwd: logits@0 targets@1 lse@2 grad_out@3 -> grad_logits@4 ; V@5
//        ignore_index@6 (i32) label_smoothing@7 z_loss@8 (f32) ; grid (T,) × 32. -----
template <class E>
void launch_cross_entropy_bwd(E& e, typename E::in_t logits, typename E::in_t targets,
                              typename E::in_t lse, typename E::in_t grad_out,
                              typename E::out_t grad_logits, int V, int ignore_index,
                              float label_smoothing, float z_loss, float softcap, int T,
                              bool mw, const std::string& t) {
  e.pipeline(mw ? "cross_entropy_bwd_mw_" + t : "cross_entropy_bwd_" + t);
  e.in(logits, 0); e.in(targets, 1); e.in(lse, 2); e.in(grad_out, 3); e.out(grad_logits, 4);
  e.bytes(V, 5); e.bytes(ignore_index, 6); e.bytes(label_smoothing, 7); e.bytes(z_loss, 8);
  e.bytes(softcap, 9);
  e.dispatch(T, 1, 1, mw ? 128 : 32, 1, 1);
}

// ----- flux_gelu: D@0 A@1 B@2 bias@3 ; N@4 K@5 M@6 (i32) ; grid (M/32, N/32, 1) -----
// out = gelu(A@B + bias); A (N,K), B (K,M), bias (M,).
template <class E>
void launch_flux_gelu(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                      typename E::in_t bias, int N, int K, int M, const std::string& t) {
  e.pipeline(flux_gelu_kernel_name(t));
  e.out(d, 0); e.in(a, 1); e.in(b, 2); e.in(bias, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

template <class E>
void launch_flux_gelu_erf(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                          typename E::in_t bias, int N, int K, int M,
                          const std::string& t) {
  e.pipeline("flux_gelu_erf_" + t);
  e.out(d, 0); e.in(a, 1); e.in(b, 2); e.in(bias, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- flux_gate: D@0 A@1 B@2 bias@3 gate@4 residual@5 ; N@6 K@7 M@8 ; grid (M/32, N/32, 1) -----
// out = (A@B + bias) * gate + residual.
template <class E>
void launch_flux_gate(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                      typename E::in_t bias, typename E::in_t gate, typename E::in_t resid,
                      int N, int K, int M, const std::string& t) {
  e.pipeline(flux_gate_kernel_name(t));
  e.out(d, 0); e.in(a, 1); e.in(b, 2); e.in(bias, 3); e.in(gate, 4); e.in(resid, 5);
  e.bytes(N, 6); e.bytes(K, 7); e.bytes(M, 8);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- gemm_staged: D@0 A@1 B@2 ; N@3 K@4 M@5 (i32) ; grid (M/32, N/32, 1), 64 threads
//        (2 simdgroups) per threadgroup. A (N,K), B (K,M), out (N,M). A bigger 4-simdgroup
//        BM=128 tile was benchmarked and is slower (see gemm_staged.metal). -----
template <class E>
void launch_gemm_staged(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                        int N, int K, int M, const std::string& t) {
  e.pipeline(gemm_staged_kernel_name(t));
  e.out(d, 0); e.in(a, 1); e.in(b, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 64, 1, 1);  // 64 threads = 2 simdgroups
}

// ----- attn_multiwarp: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (N/32, H, B),
//        128 threads (4 simdgroups) per threadgroup; shared K/V across warps. -----
template <class E>
void launch_attn_multiwarp(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                           typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  constexpr int NUM_WARPS = 4;  // 2 vs 4 benchmarked equivalent (both ~5% behind attn_fwd)
  e.pipeline(attn_multiwarp_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(static_cast<int>(N) / (8 * NUM_WARPS), static_cast<int>(H), B,
             32 * NUM_WARPS, 1, 1);
}

// ----- linear_attn: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (1, H, B) group (32,1,1).
//        Non-causal linear attention out = Q @ (K^T @ V). q,k,v,o (B,H,N,D), D=64. -----
template <class E>
void launch_linear_attn(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                        typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(linear_attn_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(1, static_cast<int>(H), B, 32, 1, 1);
}

// ----- hedgehog: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (1, H, B) group (32,1,1).
//        Feature-map linear attention out = phi(Q) @ (phi(K)^T @ V), D=64. -----
template <class E>
void launch_hedgehog(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                     typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(hedgehog_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(1, static_cast<int>(H), B, 32, 1, 1);
}

// ----- lin_attn_causal: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (1, H, B) group (32,1,1).
//        Causal linear attention (chunked running-KV scan), D=64. -----
template <class E>
void launch_lin_attn_causal(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                            typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(lin_attn_causal_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(1, static_cast<int>(H), B, 32, 1, 1);
}

// ----- chunked-parallel causal linear attention (3 kernels, chunk L=64). Scratch S is
//        (B,H,C,D,D) fp32, C = N/64. K1 per-chunk KV -> S; K2 exclusive chunk prefix
//        (Sin -> Sex); K3 per-chunk serial body seeded with Sex. Grids: (C,H,B) / (B*H,1,1)
//        x 256 / (C,H,B). -----
template <class E>
void launch_lin_chunk_kv(E& e, typename E::in_t k, typename E::in_t v, typename E::out_t s,
                         unsigned N, unsigned H, int B, int C, int D) {
  e.pipeline("lin_chunk_kv_" + std::to_string(D));
  e.in(k, 0); e.in(v, 1); e.out(s, 2);
  e.bytes(N, 3); e.bytes(H, 4);
  e.dispatch(C, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_lin_chunk_scan(E& e, typename E::in_t sin, typename E::out_t sex,
                           unsigned C, int BH, int D) {
  e.pipeline("lin_chunk_scan_" + std::to_string(D));
  e.in(sin, 0); e.out(sex, 1);
  e.bytes(C, 2);
  e.dispatch(BH, 1, 1, 256, 1, 1);
}
template <class E>
void launch_lin_chunk_out(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                          typename E::in_t sex, typename E::out_t o,
                          unsigned N, unsigned H, int B, int C, int D) {
  e.pipeline("lin_chunk_out_" + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(sex, 3); e.out(o, 4);
  e.bytes(N, 5); e.bytes(H, 6);
  e.dispatch(C, static_cast<int>(H), B, 32, 1, 1);
}

// ----- mamba2 (SSD): C@0 B@1 X@2 cumlog@3 -> Y@4 ; N@5(u32) H@6(u32) ;
//        grid (N/8, H, B) group (32,1,1). C,B,X,Y (B,H,N,D) bf16; cumlog (B,H,N) fp32. -----
template <class E>
void launch_mamba2(E& e, typename E::in_t C, typename E::in_t Bm, typename E::in_t X,
                   typename E::in_t cumlog, typename E::out_t Y, unsigned N, unsigned H,
                   int B, int D) {
  e.pipeline(mamba2_kernel_name(D));
  e.in(C, 0); e.in(Bm, 1); e.in(X, 2); e.in(cumlog, 3); e.out(Y, 4);
  e.bytes(N, 5); e.bytes(H, 6);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- mamba2_bwd_row: C@0 Bm@1 X@2 cl@3(f32) dY@4 -> dC@5 r@6(f32) ; N@7 H@8 (u32) ;
//        grid (N/8, H, B) × 32. Row-owned dC + rowsum(M). -----
template <class E>
void launch_mamba2_bwd_row(E& e, typename E::in_t C, typename E::in_t Bm, typename E::in_t X,
                           typename E::in_t cl, typename E::in_t dY, typename E::out_t dC,
                           typename E::out_t r, unsigned N, unsigned H, int B, int D) {
  e.pipeline("mamba2_bwd_row_" + std::to_string(D));
  e.in(C, 0); e.in(Bm, 1); e.in(X, 2); e.in(cl, 3); e.in(dY, 4); e.out(dC, 5); e.out(r, 6);
  e.bytes(N, 7); e.bytes(H, 8);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- mamba2_bwd_col: C@0 Bm@1 X@2 cl@3(f32) dY@4 -> dB@5 dX@6 cc@7(f32) ; N@8 H@9 (u32) ;
//        grid (N/8, H, B) × 32. Col-owned dB, dX + colsum(M). -----
template <class E>
void launch_mamba2_bwd_col(E& e, typename E::in_t C, typename E::in_t Bm, typename E::in_t X,
                           typename E::in_t cl, typename E::in_t dY, typename E::out_t dB,
                           typename E::out_t dX, typename E::out_t cc, unsigned N, unsigned H,
                           int B, int D) {
  e.pipeline("mamba2_bwd_col_" + std::to_string(D));
  e.in(C, 0); e.in(Bm, 1); e.in(X, 2); e.in(cl, 3); e.in(dY, 4);
  e.out(dB, 5); e.out(dX, 6); e.out(cc, 7);
  e.bytes(N, 8); e.bytes(H, 9);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- chunked linear-time SSD (3 kernels; shared by mamba2 and lin_attn_decay, chunk L=64;
//        D in {64,128} via 64x64 state QUADRANTS, QB = D/64). Scratch: s_raw (B,H,C,D,D) fp32,
//        s_ex (B,H,C,D,D) BF16 (the out mma consumes bf16 anyway — identical results, half the
//        state-read traffic), C = N/64. K1 per-chunk decayed KV (one simdgroup per quadrant);
//        K2 decayed exclusive chunk prefix; K3 intra(decay tiles) + inter(state), COOPERATIVE —
//        one threadgroup per chunk (256 threads), state quadrants staged once into threadgroup
//        memory and shared by the chunk's 8 query tiles. -----
template <class E>
void launch_ssd_chunk_kv(E& e, typename E::in_t bm, typename E::in_t x, typename E::in_t cl,
                         typename E::out_t s, unsigned N, unsigned H, int B, int C, int D) {
  e.pipeline("ssd_chunk_kv_" + std::to_string(D));
  e.in(bm, 0); e.in(x, 1); e.in(cl, 2); e.out(s, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  const int QB = D / 64;                       // 64x64 state quadrants per side
  e.dispatch(C * QB * QB, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_ssd_chunk_scan(E& e, typename E::in_t sin, typename E::in_t cl,
                           typename E::out_t sex, unsigned C, unsigned N, int BH, int D) {
  e.pipeline("ssd_chunk_scan_" + std::to_string(D));
  e.in(sin, 0); e.in(cl, 1); e.out(sex, 2);
  e.bytes(C, 3); e.bytes(N, 4);
  e.dispatch(BH, 1, 1, 256, 1, 1);
}
template <class E>
void launch_ssd_chunk_out(E& e, typename E::in_t cq, typename E::in_t bm, typename E::in_t x,
                          typename E::in_t cl, typename E::in_t sex, typename E::out_t y,
                          unsigned N, unsigned H, int B, int D) {
  e.pipeline("ssd_chunk_out_" + std::to_string(D));
  e.in(cq, 0); e.in(bm, 1); e.in(x, 2); e.in(cl, 3); e.in(sex, 4); e.out(y, 5);
  e.bytes(N, 6); e.bytes(H, 7);
  e.dispatch(static_cast<int>(N) / 64, static_cast<int>(H), B, 256, 1, 1);   // coop: tg per chunk
}

// ----- chunked SSD backward (D in {64,128}): recompute the forward Sex (kv -> scan), build the
//        gradient states (gstate -> rscan, ONE reverse chain — bwd_col loads dKV in both row and
//        col register layouts instead of materializing a transposed copy), then cooperative
//        row/col output kernels. dcl comes out in-kernel, split into intra + inter halves:
//        rowsum(M) = r + ri, colsum(M) = cc + ci; host combines (r+ri)-(cc+ci). -----
template <class E>
void launch_ssd_chunk_gstate(E& e, typename E::in_t cq, typename E::in_t dy, typename E::in_t cl,
                             typename E::out_t g, unsigned N, unsigned H, int B, int C, int D) {
  e.pipeline("ssd_chunk_gstate_" + std::to_string(D));
  e.in(cq, 0); e.in(dy, 1); e.in(cl, 2); e.out(g, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  const int QB = D / 64;
  e.dispatch(C * QB * QB, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_ssd_chunk_rscan(E& e, typename E::in_t gin, typename E::in_t cl,
                            typename E::out_t dkv, unsigned C, unsigned N, int BH, int D) {
  e.pipeline("ssd_chunk_rscan_" + std::to_string(D));
  e.in(gin, 0); e.in(cl, 1); e.out(dkv, 2);
  e.bytes(C, 3); e.bytes(N, 4);
  e.dispatch(BH, 1, 1, 256, 1, 1);
}
template <class E>
void launch_ssd_chunk_bwd_row(E& e, typename E::in_t cq, typename E::in_t bm, typename E::in_t x,
                              typename E::in_t cl, typename E::in_t dy, typename E::in_t sex,
                              typename E::out_t dc, typename E::out_t r, typename E::out_t ri,
                              unsigned N, unsigned H, int B, int D) {
  e.pipeline("ssd_chunk_bwd_row_" + std::to_string(D));
  e.in(cq, 0); e.in(bm, 1); e.in(x, 2); e.in(cl, 3); e.in(dy, 4); e.in(sex, 5);
  e.out(dc, 6); e.out(r, 7); e.out(ri, 8); e.bytes(N, 9); e.bytes(H, 10);
  e.dispatch(static_cast<int>(N) / 64, static_cast<int>(H), B, 256, 1, 1);   // cooperative
}
template <class E>
void launch_ssd_chunk_bwd_col(E& e, typename E::in_t cq, typename E::in_t bm, typename E::in_t x,
                              typename E::in_t cl, typename E::in_t dy, typename E::in_t dkv,
                              typename E::out_t db, typename E::out_t dx, typename E::out_t cc,
                              typename E::out_t ci, unsigned N, unsigned H, int B, int D) {
  e.pipeline("ssd_chunk_bwd_col_" + std::to_string(D));
  e.in(cq, 0); e.in(bm, 1); e.in(x, 2); e.in(cl, 3); e.in(dy, 4); e.in(dkv, 5);
  e.out(db, 6); e.out(dx, 7); e.out(cc, 8); e.out(ci, 9); e.bytes(N, 10); e.bytes(H, 11);
  e.dispatch(static_cast<int>(N) / 64, static_cast<int>(H), B, 256, 1, 1);   // cooperative
}

// ----- ssd_decode: single-token SSD decode step, S' = alpha*S + x⊗k ; y = S'·q.
//        Sin/Sout (B,H,D,D) fp32 (may alias for in-place); alpha (B,H); x/k/q/y (B,H,D).
//        One threadgroup per (batch, head), D threads (one per state row). -----
template <class E>
void launch_ssd_decode(E& e, typename E::in_t sin, typename E::in_t alpha, typename E::in_t x,
                       typename E::in_t k, typename E::in_t q, typename E::out_t sout,
                       typename E::out_t y, unsigned H, int B, int D) {
  e.pipeline("ssd_decode_" + std::to_string(D));
  e.in(sin, 0); e.in(alpha, 1); e.in(x, 2); e.in(k, 3); e.in(q, 4);
  e.out(sout, 5); e.out(y, 6);
  e.bytes(H, 7);
  e.dispatch(1, static_cast<int>(H), B, D, 1, 1);
}

// ----- attn backward family (FlashAttention-2 bwd). All grid (N/8, H, B) group (32,1,1). -----
template <class E>
void launch_attn_fwd_l(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                       typename E::out_t o, typename E::out_t L, unsigned N, unsigned H, int B, int D,
                       bool causal) {
  e.pipeline("attn_fwd_l_" + std::string(causal ? "causal_" : "noncausal_") + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3); e.out(L, 4);
  e.bytes(N, 5); e.bytes(H, 6);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_attn_bwd_prep(E& e, typename E::in_t o, typename E::in_t ddo, typename E::out_t delta,
                          unsigned N, unsigned H, int B, int D) {
  e.pipeline("attn_bwd_prep_" + std::to_string(D));
  e.in(o, 0); e.in(ddo, 1); e.out(delta, 2);
  e.bytes(N, 3); e.bytes(H, 4);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_attn_bwd_dq(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                        typename E::in_t ddo, typename E::in_t L, typename E::in_t delta,
                        typename E::out_t dq, unsigned N, unsigned H, int B, int D, bool causal) {
  e.pipeline("attn_bwd_dq_" + std::string(causal ? "causal_" : "noncausal_") + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(ddo, 3); e.in(L, 4); e.in(delta, 5); e.out(dq, 6);
  e.bytes(N, 7); e.bytes(H, 8);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_attn_bwd_dkv(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                         typename E::in_t ddo, typename E::in_t L, typename E::in_t delta,
                         typename E::out_t dk, typename E::out_t dv, unsigned N, unsigned H, int B,
                         int D, bool causal) {
  e.pipeline("attn_bwd_dkv_" + std::string(causal ? "causal_" : "noncausal_") + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(ddo, 3); e.in(L, 4); e.in(delta, 5); e.out(dk, 6); e.out(dv, 7);
  e.bytes(N, 8); e.bytes(H, 9);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- lin_attn_decay (retention): q@0 k@1 v@2 cl@3(=-slope*pos) -> o@4 ; N@5(u32) H@6(u32) ;
//        grid (N/8, H, B) group (32,1,1). q,k,v,o (B,H,N,D) bf16; cl (B,H,N) fp32. -----
template <class E>
void launch_lin_attn_decay(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                           typename E::in_t cl, typename E::out_t o, unsigned N, unsigned H,
                           int B, int D) {
  e.pipeline(lin_attn_decay_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(cl, 3); e.out(o, 4);
  e.bytes(N, 5); e.bytes(H, 6);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- based (Taylor feature-map linear attention): q@0 k@1 (D_QK) v@2 (D_VO) -> o@3 ; N@4 H@5 ;
//        grid (N/8, H, B) group (32,1,1). q,k (B,H,N,16) v,o (B,H,N,64) bf16. -----
template <class E>
void launch_based(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                  typename E::out_t o, unsigned N, unsigned H, int B, int DQK, int DVO) {
  e.pipeline(based_kernel_name(DQK, DVO));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- cmplx_matmul: D@0 A@1 B@2 ; N@3 K@4 M@5 (i32) ; grid (M/32, N/32, 1) group (32,1,1).
//        Complex GEMM D = A @ B; each operand has a leading size-2 (real,imag) axis:
//        A (2,N,K), B (2,K,M), D (2,N,M). Uses the complex_mma_AB primitive. -----
template <class E>
void launch_cmplx_matmul(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                         int N, int K, int M, const std::string& t) {
  const bool use_small = K < 512;
  e.pipeline(cmplx_matmul_kernel_name(t) + (use_small ? "_small" : ""));
  e.out(d, 0); e.in(a, 1); e.in(b, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- fftconv (Monarch FFT convolution): OUT@0 X@1 F@2 TWF@3 FINV@4 TWI@5 KF@6 ;
//        BH@7 H@8 (i32) ; grid (BH,1,1) group (32,1,1). N = S*S; S in {16,32}.
//        Complex arrays carry a leading size-2 (real,imag) axis; OUT is real (BH,S,S). -----
template <class E>
void launch_fftconv(E& e, typename E::out_t out, typename E::in_t x, typename E::in_t F,
                    typename E::in_t twf, typename E::in_t finv, typename E::in_t twi,
                    typename E::in_t kf, int BH, int H, int S) {
  e.pipeline(fftconv_kernel_name(S));
  e.out(out, 0); e.in(x, 1); e.in(F, 2); e.in(twf, 3);
  e.in(finv, 4); e.in(twi, 5); e.in(kf, 6);
  e.bytes(BH, 7); e.bytes(H, 8);
  e.dispatch(BH, 1, 1, 32, 1, 1);
}

// ----- qgemm (quantized GEMM, dequant-to-shared): D@0 Wq@1 X@2 ; N@3 K@4 M@5 (i32) ;
//        grid (M/tile_m, N/32, 1), 64 threads (2 simdgroups). D=W@X, W (N,K) quantized blocks
//        (format `fmt`), X (K,M) half, D (N,M) half. -----
template <class E>
void launch_qgemm(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                  int N, int K, int M, const std::string& fmt) {
  e.pipeline(qgemm_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  const int tile_m = fmt == "mxfp8" ? 64 : 32;
  e.dispatch(M / tile_m, N / 32, 1, 64, 1, 1);
}

// ----- qgemm_actorder: GPTQ act-order, in-kernel g_idx gather. D@0 Wq@1 X@2 perm@3(int) ; N@4 K@5
//        M@6 ; grid (M/32, N/32, 1), 32 threads. Gathers X K-rows by perm during the X load. -----
template <class E>
void launch_qgemm_actorder(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                           typename E::in_t perm, int N, int K, int M, const std::string& fmt) {
  e.pipeline(qgemm_actorder_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2); e.in(perm, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- qgemm_fp8_scaled: both operands fp8 e4m3, rank-1 scaled. D@0 Wq@1(N,K fp8) Xq@2(K,M fp8)
//        w_scale@3(N) a_scale@4(M) ; N@5 K@6 M@7 ; grid (M/32, N/32, 1), 32 threads. -----
template <class E>
void launch_qgemm_fp8_scaled(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                             typename E::in_t wscale, typename E::in_t ascale, int N, int K, int M) {
  e.pipeline("mittens::qgemm_fp8_scaled");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(wscale, 3); e.in(ascale, 4);
  e.bytes(N, 5); e.bytes(K, 6); e.bytes(M, 7);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- qgemm_blockscale (fp8_block2d): D@0 Wq@1(codes) X@2 scale2d@3 ; N@4 K@5 M@6 ; grid
//        (M/32, N/32, 1), 32 threads. Separate (N/128,K/128) tile scale. -----
template <class E>
void launch_qgemm_blockscale(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                             typename E::in_t scale2d, int N, int K, int M) {
  e.pipeline("qgemm_blockscale_fp8_raw");
  e.out(d, 0); e.in(wq, 1); e.in(x, 2); e.in(scale2d, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- qgemm_frag: dequant-direct-to-fragment. D@0 Wq@1 X@2 ; N@3 K@4 M@5 ; grid (M/32, N/32, 1),
//        32 threads (1 simdgroup) per 32x32 output tile. No shared staging / barrier. -----
template <class E>
void launch_qgemm_frag(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                       int N, int K, int M, const std::string& fmt) {
  e.pipeline(qgemm_frag_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);  // 32 threads = 1 simdgroup
}

// ----- qdequant_fp16: W@0(N,K half) Wq@1 ; N@2 K@3 (i32) ; flat, one thread per 8-col span.
//        Full-weight dequant backing the k-quant prefill route. -----
template <class E>
void launch_qdequant_fp16(E& e, typename E::out_t w, typename E::in_t wq,
                          int N, int K, const std::string& fmt) {
  e.pipeline("qdequant_" + fmt);
  e.out(w, 0); e.in(wq, 1);
  e.bytes(N, 2); e.bytes(K, 3);
  const long threads = (long)N * (K / 8);
  e.dispatch(static_cast<int>((threads + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- canonical BaseQN separate-plane contract. Codes are (N,K*bits/8) u8;
// scales/biases are (N,K/group_size); X is (K,M). The same descriptor drives
// MLX and PyTorch MPS, including symmetric no-bias tensors (bias buffer ignored).
template <class E>
void launch_base_qdequant(
    E& e, typename E::out_t output, typename E::in_t codes,
    typename E::in_t scales, typename E::in_t biases, int N, int K,
    const BaseQDescriptor& descriptor, const std::string& type_name) {
  e.pipeline(base_q_kernel_name("dequant", type_name));
  e.out(output, 0); e.in(codes, 1); e.in(scales, 2); e.in(biases, 3);
  const int scale_type = base_q_scale_type_code(descriptor);
  const int symmetric = descriptor.symmetric ? 1 : 0;
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(descriptor.group_size, 6);
  e.bytes(descriptor.bits, 7); e.bytes(scale_type, 8); e.bytes(symmetric, 9);
  const long total = static_cast<long>(N) * K;
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}

template <class E>
void launch_base_qmatmul(
    E& e, typename E::out_t output, typename E::in_t codes,
    typename E::in_t scales, typename E::in_t biases, typename E::in_t x,
    int N, int K, int M, const BaseQDescriptor& descriptor,
    const std::string& type_name) {
  e.pipeline(base_q_kernel_name(M == 1 ? "gemv" : "gemm", type_name));
  e.out(output, 0); e.in(codes, 1); e.in(scales, 2); e.in(biases, 3); e.in(x, 4);
  const int scale_type = base_q_scale_type_code(descriptor);
  const int symmetric = descriptor.symmetric ? 1 : 0;
  e.bytes(N, 5); e.bytes(K, 6); e.bytes(M, 7);
  e.bytes(descriptor.group_size, 8); e.bytes(descriptor.bits, 9);
  e.bytes(scale_type, 10); e.bytes(symmetric, 11);
  if (M == 1) {
    e.dispatch(N, 1, 1, 32, 1, 1);
  } else {
    e.dispatch((M + 31) / 32, N, 1, 32, 1, 1);
  }
}

template <class E>
void launch_base_qgemv_qkv(
    E& e, typename E::out_t q_output, typename E::out_t k_output,
    typename E::out_t v_output, typename E::in_t q_codes,
    typename E::in_t q_scales, typename E::in_t q_biases,
    typename E::in_t k_codes, typename E::in_t k_scales,
    typename E::in_t k_biases, typename E::in_t v_codes,
    typename E::in_t v_scales, typename E::in_t v_biases,
    typename E::in_t x, int q_rows, int k_rows, int v_rows, int K,
    const BaseQDescriptor& descriptor, const std::string& type_name) {
  e.pipeline(base_q_kernel_name("gemv_qkv", type_name));
  e.out(q_output, 0); e.out(k_output, 1); e.out(v_output, 2);
  e.in(q_codes, 3); e.in(q_scales, 4); e.in(q_biases, 5);
  e.in(k_codes, 6); e.in(k_scales, 7); e.in(k_biases, 8);
  e.in(v_codes, 9); e.in(v_scales, 10); e.in(v_biases, 11); e.in(x, 12);
  const int scale_type = base_q_scale_type_code(descriptor);
  const int symmetric = descriptor.symmetric ? 1 : 0;
  e.bytes(q_rows, 13); e.bytes(k_rows, 14); e.bytes(v_rows, 15);
  e.bytes(K, 16); e.bytes(descriptor.group_size, 17);
  e.bytes(descriptor.bits, 18); e.bytes(scale_type, 19); e.bytes(symmetric, 20);
  e.dispatch(q_rows + k_rows + v_rows, 1, 1, 32, 1, 1);
}

template <class E>
void launch_base_qgemv_swiglu(
    E& e, typename E::out_t output, typename E::in_t gate_codes,
    typename E::in_t gate_scales, typename E::in_t gate_biases,
    typename E::in_t up_codes, typename E::in_t up_scales,
    typename E::in_t up_biases, typename E::in_t x, int N, int K,
    const BaseQDescriptor& descriptor, const std::string& type_name) {
  e.pipeline(base_q_kernel_name("gemv_swiglu", type_name));
  e.out(output, 0);
  e.in(gate_codes, 1); e.in(gate_scales, 2); e.in(gate_biases, 3);
  e.in(up_codes, 4); e.in(up_scales, 5); e.in(up_biases, 6); e.in(x, 7);
  const int scale_type = base_q_scale_type_code(descriptor);
  const int symmetric = descriptor.symmetric ? 1 : 0;
  e.bytes(N, 8); e.bytes(K, 9); e.bytes(descriptor.group_size, 10);
  e.bytes(descriptor.bits, 11); e.bytes(scale_type, 12); e.bytes(symmetric, 13);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

template <class E>
void launch_base_qembedding(
    E& e, typename E::out_t output, typename E::in_t codes,
    typename E::in_t scales, typename E::in_t biases, typename E::in_t ids,
    int rows, int columns, int tokens, const BaseQDescriptor& descriptor,
    const std::string& type_name) {
  e.pipeline(base_q_kernel_name("embedding", type_name));
  e.out(output, 0); e.in(codes, 1); e.in(scales, 2); e.in(biases, 3); e.in(ids, 4);
  const int scale_type = base_q_scale_type_code(descriptor);
  const int symmetric = descriptor.symmetric ? 1 : 0;
  e.bytes(rows, 5); e.bytes(columns, 6); e.bytes(tokens, 7);
  e.bytes(descriptor.group_size, 8); e.bytes(descriptor.bits, 9);
  e.bytes(scale_type, 10); e.bytes(symmetric, 11);
  const long total = static_cast<long>(tokens) * columns;
  e.dispatch(static_cast<int>((total + 255) / 256), 1, 1, 256, 1, 1);
}

// ----- canonical BaseQN grouped expert projections. Expert stacks add a
// leading E dimension to the ordinary separate planes; expert_of_tile uses
// the existing padded QuixiCore MoE schedule (one expert per 32-row tile).
template <class E>
void launch_base_qmoe_gemm(
    E& e, typename E::out_t output, typename E::in_t input,
    typename E::in_t codes, typename E::in_t scales,
    typename E::in_t biases, typename E::in_t expert_of_tile,
    int total_rows, int inner, int output_rows,
    const BaseQDescriptor& descriptor, const std::string& type_name) {
  e.pipeline(base_q_kernel_name("moe_gemm", type_name));
  e.out(output, 0); e.in(input, 1); e.in(codes, 2); e.in(scales, 3);
  e.in(biases, 4); e.in(expert_of_tile, 5);
  const int scale_type = base_q_scale_type_code(descriptor);
  const int symmetric = descriptor.symmetric ? 1 : 0;
  e.bytes(total_rows, 6); e.bytes(inner, 7); e.bytes(output_rows, 8);
  e.bytes(descriptor.group_size, 9); e.bytes(descriptor.bits, 10);
  e.bytes(scale_type, 11); e.bytes(symmetric, 12);
  e.dispatch(output_rows / 32, total_rows / 32, 1, 32, 1, 1);
}

template <class E>
void launch_base_qmoe_swiglu(
    E& e, typename E::out_t output, typename E::in_t input,
    typename E::in_t codes, typename E::in_t scales,
    typename E::in_t biases, typename E::in_t expert_of_tile,
    int total_rows, int inner, int intermediate,
    const BaseQDescriptor& descriptor, const std::string& type_name) {
  e.pipeline(base_q_kernel_name("moe_swiglu", type_name));
  e.out(output, 0); e.in(input, 1); e.in(codes, 2); e.in(scales, 3);
  e.in(biases, 4); e.in(expert_of_tile, 5);
  const int scale_type = base_q_scale_type_code(descriptor);
  const int symmetric = descriptor.symmetric ? 1 : 0;
  e.bytes(total_rows, 6); e.bytes(inner, 7); e.bytes(intermediate, 8);
  e.bytes(descriptor.group_size, 9); e.bytes(descriptor.bits, 10);
  e.bytes(scale_type, 11); e.bytes(symmetric, 12);
  e.dispatch(intermediate / 32, total_rows / 32, 1, 128, 1, 1);
}

// ----- qgemv (quantized GEMV, batch-1 decode): D@0 Wq@1 X@2 ; N@3 K@4 (i32) ;
//        grid (N,1,1), 32 threads (1 simdgroup) per output row. d = W @ x, x (K,1) half. -----
template <class E>
void launch_qgemv(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                  int N, int K, const std::string& fmt,
                  const std::string& type_name = "float16") {
  const bool use_small = K <= 512 && (fmt == "q8_0" || fmt == "q4_0");
  if (type_name == "float32") {
    e.pipeline(qgemv_kernel_name(fmt) + "_float32");
  } else {
    e.pipeline(use_small ? qgemv_kernel_name(fmt) + "_small" : qgemv_kernel_name(fmt));
  }
  e.out(d, 0); e.in(wq, 1); e.in(x, 2);
  e.bytes(N, 3); e.bytes(K, 4);
  // one simdgroup per output row. Geometry experiments measured and REJECTED here: two ROWS
  // per simdgroup (1.9x better for the integer w8a8 path, 1.6-2.8x worse here — float dequant
  // doubles register pressure) and two-simdgroup split-K (3-4x better at the small BitNet
  // shapes but 2-3x worse at the K=4096 LLM shapes, both half-split and interleaved).
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- fused packed-Q4_0 decode GEMVs (batch-1), fp32 activation + output. -----
// One simdgroup per output row; a shared (K,1) activation is streamed once for
// the fused projections. Weights are GGUF Q4_0 packed blocks (N, K/32, 18) u8.

// up+gate+GELU: out@0 = gelu(gate@x) * (up@x). up@1 gate@2 x@3 ; N@4 K@5 (i32) ; grid (N,1,1).
template <class E>
void launch_qgemv_q4_0_f32_up_gate_gelu(E& e, typename E::out_t out, typename E::in_t up,
                                        typename E::in_t gate, typename E::in_t x, int N, int K) {
  e.pipeline("qgemv_q4_0_f32_up_gate_gelu");
  e.out(out, 0); e.in(up, 1); e.in(gate, 2); e.in(x, 3);
  e.bytes(N, 4); e.bytes(K, 5);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// up+gate: up_out@0 = up@x, gate_out@1 = gate@x. up@2 gate@3 x@4 ; N@5 K@6 ; grid (2N,1,1).
template <class E>
void launch_qgemv_q4_0_f32_up_gate(E& e, typename E::out_t up_out, typename E::out_t gate_out,
                                   typename E::in_t up, typename E::in_t gate, typename E::in_t x,
                                   int N, int K) {
  e.pipeline("qgemv_q4_0_f32_up_gate");
  e.out(up_out, 0); e.out(gate_out, 1); e.in(up, 2); e.in(gate, 3); e.in(x, 4);
  e.bytes(N, 5); e.bytes(K, 6);
  e.dispatch(2 * N, 1, 1, 32, 1, 1);
}

// QKV: q_out@0 k_out@1 v_out@2 ; qw@3 kw@4 vw@5 x@6 ; Nq@7 Nkv@8 K@9 ; grid (Nq+2*Nkv,1,1).
template <class E>
void launch_qgemv_q4_0_f32_qkv(E& e, typename E::out_t q_out, typename E::out_t k_out,
                               typename E::out_t v_out, typename E::in_t qw, typename E::in_t kw,
                               typename E::in_t vw, typename E::in_t x, int Nq, int Nkv, int K) {
  e.pipeline("qgemv_q4_0_f32_qkv");
  e.out(q_out, 0); e.out(k_out, 1); e.out(v_out, 2);
  e.in(qw, 3); e.in(kw, 4); e.in(vw, 5); e.in(x, 6);
  e.bytes(Nq, 7); e.bytes(Nkv, 8); e.bytes(K, 9);
  e.dispatch(Nq + 2 * Nkv, 1, 1, 32, 1, 1);
}

// Packed embedding/PLE gather: table@0 ids@1 -> fp16 output@2.
template <class E>
void launch_dequant_gather(E& e, typename E::in_t table, typename E::in_t ids,
                           typename E::out_t output, int rows, int columns, int tokens,
                           float scale, const std::string& fmt) {
  e.pipeline("dequant_gather_" + fmt);
  e.in(table, 0); e.in(ids, 1); e.out(output, 2);
  e.bytes(rows, 3); e.bytes(columns, 4); e.bytes(tokens, 5); e.bytes(scale, 6);
  const long spans = (long)tokens * (columns / 8);
  e.dispatch(static_cast<int>((spans + 255) / 256), 1, 1, 256, 1, 1);
}

// General packed embedding lookup with output dtype and optional additive
// epilogue.  One thread owns an aligned eight-column span.
template <class E>
void launch_quantized_embedding(
    E& e, typename E::in_t table, typename E::in_t ids, typename E::in_t add,
    typename E::out_t output, int rows, int columns, int tokens, float scale,
    bool use_add, const std::string& fmt, const std::string& type_name) {
  e.pipeline("quantized_embedding_" + fmt + "_" + type_name);
  e.in(table, 0); e.in(ids, 1); e.in(add, 2); e.out(output, 3);
  e.bytes(rows, 4); e.bytes(columns, 5); e.bytes(tokens, 6); e.bytes(scale, 7);
  e.bytes(use_add ? 1 : 0, 8);
  const long spans = static_cast<long>(tokens) * (columns / 8);
  e.dispatch(static_cast<int>((spans + 127) / 128), 1, 1, 128, 1, 1);
}

template <class E>
void launch_quantized_embedding_bag(
    E& e, typename E::in_t table, typename E::in_t ids,
    typename E::in_t offsets, typename E::in_t sample_weights,
    typename E::out_t output, int rows, int columns, int id_count, int bags,
    float scale, bool use_weights, bool mean_mode, const std::string& fmt,
    const std::string& type_name) {
  e.pipeline("quantized_embedding_bag_" + fmt + "_" + type_name);
  e.in(table, 0); e.in(ids, 1); e.in(offsets, 2); e.in(sample_weights, 3);
  e.out(output, 4); e.bytes(rows, 5); e.bytes(columns, 6);
  e.bytes(id_count, 7); e.bytes(bags, 8); e.bytes(scale, 9);
  e.bytes(use_weights ? 1 : 0, 10); e.bytes(mean_mode ? 1 : 0, 11);
  const long spans = static_cast<long>(bags) * (columns / 8);
  e.dispatch(static_cast<int>((spans + 127) / 128), 1, 1, 128, 1, 1);
}

// ----- qgemv_w8a8 (W8A8 int8xint8 decode): D@0 Wq@1(int8) Xq@2(int8) w_scale@3 a_scale@4 ;
//        N@5 K@6 (i32) ; grid (N,1,1) 32 threads. int32 accumulate then *w_scale[n]*a_scale. -----
template <class E>
void launch_qgemv_w8a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t wscale, typename E::in_t ascale, int N, int K) {
  e.pipeline("mittens::qgemv_w8a8");  // non-template kernel keeps its namespaced symbol
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(wscale, 3); e.in(ascale, 4);
  e.bytes(N, 5); e.bytes(K, 6);
  e.dispatch((N + 1) / 2, 1, 1, 32, 1, 1);   // two rows per simdgroup
}

// ----- qgemv_w2a8 (BitNet W2A8 int2xint8 decode): D@0 Wq@1(bitnet blocks) Xq@2(int8) a_scale@3 ;
//        N@4 K@5 (i32) ; grid (N,1,1) 32 threads. per-group int32 sums * absmean scale * a_scale. -----
template <class E>
void launch_qgemv_w2a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t ascale, int N, int K) {
  e.pipeline("mittens::qgemv_w2a8");  // non-template kernel keeps its namespaced symbol
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(ascale, 3);
  e.bytes(N, 4); e.bytes(K, 5);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

template <class E>
void launch_qgemv_w2a8_v2(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                          typename E::in_t ascale, int N, int K) {
  e.pipeline("mittens::qgemv_w2a8_v2");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(ascale, 3);
  e.bytes(N, 4); e.bytes(K, 5);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- marginal utilities (marginal.metal). tau_tail flat 1D grid; packbits 1 thr/byte;
//        segment_packbits 1 thr/output-byte; permute_cols 2D (cols, rows). -----
template <class E>
void launch_marginal_copy(E& e, typename E::in_t src, typename E::out_t dst, uint32_t n) {
  e.pipeline("mittens::tq_clone_bytes");   // shared 16-byte/thread byte clone
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  const uint32_t nthreads = (n + 15) / 16;
  e.dispatch((int)((nthreads + 255) / 256), 1, 1, 256, 1, 1);
}
template <class E>
void launch_tau_tail(E& e, typename E::out_t qkv, typename E::in_t tok_qv_lin,
                     typename E::in_t tau_pos_table, typename E::in_t positions, int elements,
                     int n_heads, int head_dim, int q_dim, const std::string& type_name) {
  e.pipeline("tau_tail_" + type_name);
  e.out(qkv, 0); e.in(tok_qv_lin, 1); e.in(tau_pos_table, 2); e.in(positions, 3);
  e.bytes(elements, 4); e.bytes(n_heads, 5); e.bytes(head_dim, 6); e.bytes(q_dim, 7);
  e.dispatch((elements + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_packbits(E& e, typename E::in_t input, typename E::out_t output, int num_elements,
                     int bit_order_big) {
  e.pipeline("mittens::packbits_uint8");
  e.in(input, 0); e.out(output, 1); e.bytes(num_elements, 2); e.bytes(bit_order_big, 3);
  const int nbytes = (num_elements + 7) / 8;
  e.dispatch((nbytes + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_segment_packbits(E& e, typename E::in_t input, typename E::in_t input_indptr,
                             typename E::in_t output_indptr, typename E::out_t output,
                             int num_segments, int total_output_bytes, int bit_order_big) {
  e.pipeline("mittens::segment_packbits_uint8");
  e.in(input, 0); e.in(input_indptr, 1); e.in(output_indptr, 2); e.out(output, 3);
  e.bytes(num_segments, 4); e.bytes(total_output_bytes, 5); e.bytes(bit_order_big, 6);
  e.dispatch((total_output_bytes + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_permute_cols(E& e, typename E::in_t input, typename E::in_t perm,
                         typename E::out_t output, int rows, int cols) {
  e.pipeline("mittens::permute_cols_16bit");
  e.in(input, 0); e.in(perm, 1); e.out(output, 2); e.bytes(rows, 3); e.bytes(cols, 4);
  e.dispatch((cols + 15) / 16, (rows + 15) / 16, 1, 16, 16, 1);
}

// ----- TurboQuant KV codec (turboquant.metal).// ----- TurboQuant KV codec (turboquant.metal). encode grid (tokens, Hkv), HS threads;
//        decode grid (n, Hkv), HS threads. tq_clone_bytes: 1D over bytes (16/thread). -----
// ----- DeepSeek-V3.2 indexer (indexer.metal). quant grid (tokens, nq); gather grid (n, nq);
//        both 32 threads. indexer_clone_bytes: 1D over bytes (16/thread). -----
template <class E>
void launch_indexer_clone_bytes(E& e, typename E::in_t src, typename E::out_t dst, uint32_t n) {
  e.pipeline("mittens::indexer_clone_bytes");
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  const uint32_t nthreads = (n + 15) / 16;
  e.dispatch((int)((nthreads + 255) / 256), 1, 1, 256, 1, 1);
}
template <class E>
void launch_indexer_k_quant_and_cache(E& e, typename E::in_t k, typename E::in_t slot_mapping,
                                      typename E::out_t code_cache, typename E::out_t scale_cache,
                                      int num_tokens, int head_dim, int nq, int quant_block_size,
                                      int use_ue8m0, const std::string& type_name) {
  e.pipeline("indexer_k_quant_and_cache_" + type_name);
  e.in(k, 0); e.in(slot_mapping, 1); e.out(code_cache, 2); e.out(scale_cache, 3);
  e.bytes(head_dim, 4); e.bytes(quant_block_size, 5); e.bytes(use_ue8m0, 6);
  e.dispatch(num_tokens, nq, 1, 32, 1, 1);
}
template <class E>
void launch_indexer_k_gather(E& e, typename E::in_t code_cache, typename E::in_t scale_cache,
                             typename E::in_t slots, typename E::out_t k_out, int n, int head_dim,
                             int nq, int quant_block_size, const std::string& type_name) {
  e.pipeline("indexer_k_gather_" + type_name);
  e.in(code_cache, 0); e.in(scale_cache, 1); e.in(slots, 2); e.out(k_out, 3);
  e.bytes(head_dim, 4); e.bytes(quant_block_size, 5);
  e.dispatch(n, nq, 1, 32, 1, 1);
}

template <class E>
void launch_tq_clone_bytes(E& e, typename E::in_t src, typename E::out_t dst, uint32_t n) {
  e.pipeline("mittens::tq_clone_bytes");
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  const uint32_t nthreads = (n + 15) / 16;
  e.dispatch((int)((nthreads + 255) / 256), 1, 1, 256, 1, 1);
}
template <class E>
void launch_tq_encode(E& e, typename E::in_t key, typename E::in_t value,
                      typename E::out_t key_cache, typename E::out_t value_cache,
                      typename E::out_t key_scale, typename E::out_t value_scale,
                      typename E::out_t key_zero, typename E::in_t slot_mapping,
                      typename E::in_t v_centroids, typename E::in_t signs,
                      int num_tokens, int num_kv_heads, int head_size, int block_size,
                      int k_bits, int k_signed, int v_bits, const std::string& type_name) {
  e.pipeline("tq_encode_" + type_name + "_hs" + std::to_string(head_size));
  e.in(key, 0); e.in(value, 1); e.out(key_cache, 2); e.out(value_cache, 3);
  e.out(key_scale, 4); e.out(value_scale, 5); e.out(key_zero, 6);
  e.in(slot_mapping, 7); e.in(v_centroids, 8); e.in(signs, 9);
  e.bytes(num_kv_heads, 10); e.bytes(block_size, 11);
  e.bytes(k_bits, 12); e.bytes(k_signed, 13); e.bytes(v_bits, 14);
  e.dispatch(num_tokens, num_kv_heads, 1, head_size, 1, 1);
}
template <class E>
void launch_tq_decode(E& e, typename E::in_t key_cache, typename E::in_t value_cache,
                      typename E::in_t key_scale, typename E::in_t value_scale,
                      typename E::in_t key_zero, typename E::in_t slots,
                      typename E::in_t v_centroids, typename E::in_t signs,
                      typename E::out_t k_out, typename E::out_t v_out,
                      int n, int num_kv_heads, int head_size, int block_size,
                      int k_bits, int k_signed, int v_bits, const std::string& type_name) {
  e.pipeline("tq_decode_" + type_name + "_hs" + std::to_string(head_size));
  e.in(key_cache, 0); e.in(value_cache, 1); e.in(key_scale, 2); e.in(value_scale, 3);
  e.in(key_zero, 4); e.in(slots, 5); e.in(v_centroids, 6); e.in(signs, 7);
  e.out(k_out, 8); e.out(v_out, 9);
  e.bytes(num_kv_heads, 10); e.bytes(block_size, 11);
  e.bytes(k_bits, 12); e.bytes(k_signed, 13); e.bytes(v_bits, 14);
  e.dispatch(n, num_kv_heads, 1, head_size, 1, 1);
}

// ----- minference_build_block_mask:// ----- minference_build_block_mask: vert@0 slash@1 (B,H,nnz i32 -1pad) lens@2 -> mask@3
//        (B,H,max_blocks i32); scalars H@4 nnz_v@5 nnz_s@6 vtopk@7 stopk@8 bs@9 mb@10
//        last_n@11 ; grid (H, B, 1), 32 thr. -----
template <class E>
void launch_minference_block_mask(E& e, typename E::in_t vert, typename E::in_t slash,
                                  typename E::in_t lens, typename E::out_t mask, int B, int H,
                                  int nnz_v, int nnz_s, int vertical_topk, int slash_topk,
                                  int block_size, int max_blocks, int last_n_blocks) {
  e.pipeline("mittens::minference_build_block_mask");
  e.in(vert, 0); e.in(slash, 1); e.in(lens, 2); e.out(mask, 3);
  e.bytes(H, 4); e.bytes(nnz_v, 5); e.bytes(nnz_s, 6);
  e.bytes(vertical_topk, 7); e.bytes(slash_topk, 8);
  e.bytes(block_size, 9); e.bytes(max_blocks, 10); e.bytes(last_n_blocks, 11);
  e.dispatch(H, B, 1, 32, 1, 1);
}

// ----- sampler-zoo transforms (sampling_transforms.metal): one simdgroup per row;
//        x@0 -> out@1 ; V@2 ; kind-specific scalars follow. grid (rows,1,1), 32 thr. -----
template <class E>
void launch_quadratic_transform(E& e, typename E::in_t x, typename E::out_t out, int rows,
                                int V, float factor, float curve, float invtemp,
                                const std::string& tn) {
  e.pipeline("quadratic_transform_" + tn);
  e.in(x, 0); e.out(out, 1); e.bytes(V, 2);
  e.bytes(factor, 3); e.bytes(curve, 4); e.bytes(invtemp, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
// shared shape for nsigma / top_a / epsilon / eta: one float param + invtemp
template <class E>
void launch_logit_mask1(E& e, const std::string& kernel, typename E::in_t x,
                        typename E::out_t out, int rows, int V, float p0, float invtemp) {
  e.pipeline(kernel);
  e.in(x, 0); e.out(out, 1); e.bytes(V, 2); e.bytes(p0, 3); e.bytes(invtemp, 4);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_xtc_mask(E& e, typename E::in_t x, typename E::out_t out, int rows, int V,
                     float threshold, float probability, float invtemp, uint32_t seed,
                     const std::string& tn) {
  e.pipeline("xtc_mask_" + tn);
  e.in(x, 0); e.out(out, 1); e.bytes(V, 2);
  e.bytes(threshold, 3); e.bytes(probability, 4); e.bytes(invtemp, 5); e.bytes(seed, 6);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
// shared shape for skew / top_p_renorm: one float param, probs domain
template <class E>
void launch_prob_transform1(E& e, const std::string& kernel, typename E::in_t x,
                            typename E::out_t out, int rows, int V, float p0) {
  e.pipeline(kernel);
  e.in(x, 0); e.out(out, 1); e.bytes(V, 2); e.bytes(p0, 3);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_logits_softcap(E& e, typename E::in_t x, typename E::out_t out,
                           uint32_t n, float cap, const std::string& tn) {
  e.pipeline("logits_softcap_" + tn);
  e.in(x, 0); e.out(out, 1); e.bytes(n, 2); e.bytes(cap, 3);
  constexpr int threads = 256;
  const uint32_t nthreads = (n + 3) / 4;
  e.dispatch(static_cast<int>((nthreads + threads - 1) / threads),
             1, 1, threads, 1, 1);
}
template <class E>
void launch_value_clip(E& e, typename E::in_t x, typename E::out_t out,
                       uint32_t n, float min_value, float max_value,
                       const std::string& tn) {
  e.pipeline("value_clip_" + tn);
  e.in(x, 0); e.out(out, 1); e.bytes(n, 2);
  e.bytes(min_value, 3); e.bytes(max_value, 4);
  constexpr int threads = 256;
  const uint32_t nthreads = (n + 3) / 4;
  e.dispatch(static_cast<int>((nthreads + threads - 1) / threads),
             1, 1, threads, 1, 1);
}
template <class E>
void launch_top_k_renorm(E& e, typename E::in_t x, typename E::out_t out, int rows, int V,
                         int K, const std::string& tn) {
  e.pipeline("top_k_renorm_probs_" + tn);
  e.in(x, 0); e.out(out, 1); e.bytes(V, 2); e.bytes(K, 3);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_no_repeat_ngram_mask(E& e, typename E::in_t x, typename E::in_t prev,
                                 typename E::in_t lens, typename E::out_t out, int rows,
                                 int V, int L, int ngram, float invtemp,
                                 const std::string& tn) {
  e.pipeline("no_repeat_ngram_mask_" + tn);
  e.in(x, 0); e.in(prev, 1); e.in(lens, 2); e.out(out, 3);
  e.bytes(V, 4); e.bytes(L, 5); e.bytes(ngram, 6); e.bytes(invtemp, 7);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_dry_penalty(E& e, typename E::in_t x, typename E::in_t prev, typename E::in_t lens,
                        typename E::in_t breakers, typename E::out_t out, int rows, int V,
                        int L, int NB, float multiplier, float base, int allowed, int range,
                        int max_ngram, int max_occ, int early_exit, float invtemp,
                        const std::string& tn) {
  e.pipeline("dry_penalty_" + tn);
  e.in(x, 0); e.in(prev, 1); e.in(lens, 2); e.in(breakers, 3); e.out(out, 4);
  e.bytes(V, 5); e.bytes(L, 6); e.bytes(NB, 7);
  e.bytes(multiplier, 8); e.bytes(base, 9); e.bytes(allowed, 10); e.bytes(range, 11);
  e.bytes(max_ngram, 12); e.bytes(max_occ, 13); e.bytes(early_exit, 14); e.bytes(invtemp, 15);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- fused act->quant epilogues (act_quant): x@0(activated) gate@1 -> codes@2 scale@3 ;
//        one simdgroup per row; mode 0 swiglu / 1 swiglu_oai (alpha, limit). -----
template <class E>
void launch_silu_mul_quant_fp8(E& e, typename E::in_t x, typename E::in_t gate,
                               typename E::out_t codes, typename E::out_t scale, int rows,
                               int D, int mode, float alpha, float limit,
                               const std::string& type_name) {
  e.pipeline("silu_mul_quant_fp8_" + type_name);
  e.in(x, 0); e.in(gate, 1); e.out(codes, 2); e.out(scale, 3);
  e.bytes(D, 4); e.bytes(mode, 5); e.bytes(alpha, 6); e.bytes(limit, 7);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_silu_mul_quant_int8(E& e, typename E::in_t x, typename E::in_t gate,
                                typename E::out_t codes, typename E::out_t scale, int rows,
                                int D, int mode, float alpha, float limit,
                                const std::string& type_name) {
  e.pipeline("silu_mul_quant_int8_" + type_name);
  e.in(x, 0); e.in(gate, 1); e.out(codes, 2); e.out(scale, 3);
  e.bytes(D, 4); e.bytes(mode, 5); e.bytes(alpha, 6); e.bytes(limit, 7);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_silu_mul_quant_fp8_group(E& e, typename E::in_t x, typename E::in_t gate,
                                     typename E::out_t codes, typename E::out_t scale,
                                     int rows, int D, int G, int ue8m0, int mode, float alpha,
                                     float limit, const std::string& type_name) {
  const std::string mode_name = mode == 1 ? "swiglu_oai_" : "swiglu_";
  e.pipeline("silu_mul_quant_fp8_group_" + mode_name + type_name);
  e.in(x, 0); e.in(gate, 1); e.out(codes, 2); e.out(scale, 3);
  e.bytes(D, 4); e.bytes(G, 5); e.bytes(ue8m0, 6); e.bytes(mode, 7);
  e.bytes(alpha, 8); e.bytes(limit, 9);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- per-group / azp activation quantizers (quant_rt): one simdgroup per row. -----
template <class E>
void launch_quantize_per_group_fp8(E& e, typename E::in_t x, typename E::out_t codes,
                                   typename E::out_t scale, int rows, int D, int G, int ue8m0,
                                   const std::string& type_name) {
  e.pipeline("quantize_per_group_fp8_" + type_name);
  e.in(x, 0); e.out(codes, 1); e.out(scale, 2);
  e.bytes(D, 3); e.bytes(G, 4); e.bytes(ue8m0, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_quantize_per_group_int8(E& e, typename E::in_t x, typename E::out_t codes,
                                    typename E::out_t scale, int rows, int D, int G,
                                    const std::string& type_name) {
  e.pipeline("quantize_per_group_int8_" + type_name);
  e.in(x, 0); e.out(codes, 1); e.out(scale, 2);
  e.bytes(D, 3); e.bytes(G, 4);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}
template <class E>
void launch_quantize_per_token_int8_azp(E& e, typename E::in_t x, typename E::out_t codes,
                                        typename E::out_t scale, typename E::out_t azp,
                                        int rows, int D, const std::string& type_name) {
  e.pipeline("quantize_per_token_int8_azp_" + type_name);
  e.in(x, 0); e.out(codes, 1); e.out(scale, 2); e.out(azp, 3);
  e.bytes(D, 4);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// azp-corrected W8A8: D@0 Wq@1 Xq@2 w_scale@3(h) a_scale@4(f32) w_rowsum@5(i32) azp@6(i32) ;
//        N@7 K@8 M@9 ; grid (N,1,1), 32 thr.
template <class E>
void launch_qgemm_w8a8_azp(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                           typename E::in_t w_scale, typename E::in_t a_scale,
                           typename E::in_t w_rowsum, typename E::in_t azp,
                           int N, int K, int M) {
  e.pipeline("mittens::qgemm_w8a8_azp");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(w_scale, 3); e.in(a_scale, 4);
  e.in(w_rowsum, 5); e.in(azp, 6);
  e.bytes(N, 7); e.bytes(K, 8); e.bytes(M, 9);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- qgemm_w8a8 (W8A8 int8xint8 PREFILL, M>1): D@0 Wq@1(int8 N,K) Xq@2(int8 M,K) w_scale@3
//        a_scale@4 ; N@5 K@6 M@7 ; grid (N,1,1) 32 threads. Exact int32, scaled once. -----
template <class E>
void launch_qgemm_w8a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t wscale, typename E::in_t ascale, int N, int K, int M) {
  e.pipeline("mittens::qgemm_w8a8");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(wscale, 3); e.in(ascale, 4);
  e.bytes(N, 5); e.bytes(K, 6); e.bytes(M, 7);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- qgemm_w2a8 (BitNet W2A8 prefill): D@0 Wq@1(blocks) Xq@2(int8 M,K) a_scale@3 ; N@4 K@5 M@6. ---
template <class E>
void launch_qgemm_w2a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t ascale, int N, int K, int M) {
  e.pipeline("mittens::qgemm_w2a8");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(ascale, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

template <class E>
void launch_qgemm_w2a8_fused(E& e, typename E::out_t d, typename E::in_t wq,
                             typename E::in_t x, int M, int N, int K,
                             const std::string& type_name) {
  e.pipeline("qgemm_w2a8_fused_" + type_name);
  e.out(d, 0); e.in(wq, 1); e.in(x, 2);
  e.bytes(N, 3); e.bytes(K, 4);
  e.dispatch(M, 1, 1, 128, 1, 1);
}

template <class E>
void launch_qgemm_bwd(E& e, typename E::out_t d, typename E::in_t g, typename E::in_t wq,
                      int M, int N, int K, const std::string& fmt) {
  e.pipeline("qgemm_bwd_" + fmt);
  e.out(d, 0); e.in(g, 1); e.in(wq, 2);
  e.bytes(M, 3); e.bytes(N, 4); e.bytes(K, 5);
  e.dispatch(K / 32, M / 32, 1, 64, 1, 1);
}

// ----- qflux_gelu (quantized fused GEMM+GELU): D@0 Wq@1 X@2 bias@3 ; N@4 K@5 M@6 (i32) ;
//        grid (M/32, N/32, 1), 32 threads (1 simdgroup, dequant-direct-to-fragment). -----
template <class E>
void launch_qflux_gelu(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                       typename E::in_t bias, int N, int K, int M, const std::string& fmt) {
  e.pipeline(qflux_gelu_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2); e.in(bias, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);  // 1 simdgroup per 32x32 tile
}

} // namespace tk
