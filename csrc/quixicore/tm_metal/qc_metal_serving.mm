// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// `_quixicore_C` on Apple Metal.
//
// Same module name and the same role as the CUDA and ROCm builds: the ops the
// serving path calls directly as Python functions rather than through
// torch.ops. The compute lives in the vendored MSL under
// csrc/quixicore/metal/, compiled to quixicore_metal.metallib at build time;
// this file is only the host glue that dispatches it onto PyTorch's MPS
// stream.
//
// The per-kernel host ABI -- function name, buffer indices, parameter order,
// grid and threadgroup geometry -- is not restated here. It lives in
// kernels/common/tk_launch.h, which is vendored unmodified, so a kernel
// resync cannot silently desynchronize this file from the shaders. All this
// file supplies is a Torch encoder adapter and the tensor/buffer plumbing.
//
// This file owns the single PYBIND11_MODULE; sibling .mm files add to it.

#include <torch/extension.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <unordered_map>

#include "tk_launch.h"

namespace {

// The MTLBuffer backing an MPS tensor's storage (the documented PyTorch
// pattern; storage().data() is the buffer, not a host pointer).
inline id<MTLBuffer> mtl_buffer(const at::Tensor& t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

inline NSUInteger byte_offset(const at::Tensor& t) {
  return static_cast<NSUInteger>(t.storage_offset()) * t.element_size();
}

void check_mps(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be an MPS tensor, got ",
              t.device());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_mps_strided(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be an MPS tensor, got ",
              t.device());
}

// The kernels are instantiated per element type; this is the suffix that
// selects the instantiation.
std::string activation_type_name(const at::Tensor& x) {
  switch (x.scalar_type()) {
    case at::kFloat:
      return "float32";
    case at::kHalf:
      return "float16";
    case at::kBFloat16:
      return "bfloat16";
    default:
      TORCH_CHECK(false, "quixicore(metal): unsupported dtype ",
                  x.scalar_type());
  }
}

// ---- metallib + pipeline-state cache -------------------------------------
//
// The metallib sits beside this extension inside the installed package. It is
// resolved once, lazily, from the module's own __file__ so the path survives
// wheel installs and editable checkouts alike.

std::string g_metallib_path;
id<MTLLibrary> g_library = nil;
std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

void set_library(const std::string& path) {
  g_metallib_path = path;
  g_library = nil;
  g_pipelines.clear();
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, NSString* name) {
  std::string key = name.UTF8String;
  auto it = g_pipelines.find(key);
  if (it != g_pipelines.end()) return it->second;

  NSError* err = nil;
  if (g_library == nil) {
    TORCH_CHECK(!g_metallib_path.empty(),
                "quixicore: metallib path unset; the Python wrapper sets it at "
                "import. Loading the extension directly is unsupported.");
    NSString* p = [NSString stringWithUTF8String:g_metallib_path.c_str()];
    g_library = [device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&err];
    TORCH_CHECK(g_library != nil, "quixicore: failed to load metallib at ",
                g_metallib_path);
  }
  id<MTLFunction> fn = [g_library newFunctionWithName:name];
  TORCH_CHECK(fn != nil,
              "quixicore: kernel not found in metallib: ", name.UTF8String);
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso != nil, "quixicore: failed to build pipeline for ",
              name.UTF8String);
  g_pipelines[key] = pso;
  return pso;
}

// ---- the encoder adapter tk_launch.h drives ------------------------------
struct TorchEncoder {
  using in_t = const at::Tensor&;
  using out_t = const at::Tensor&;
  id<MTLComputeCommandEncoder> enc;
  id<MTLDevice> device;

  void pipeline(const std::string& name) {
    [enc setComputePipelineState: ::pipeline(
                                      device,
                                      [NSString
                                          stringWithUTF8String:name.c_str()])];
  }
  void in(const at::Tensor& t, int i) {
    [enc setBuffer:mtl_buffer(t) offset:byte_offset(t) atIndex:i];
  }
  void out(const at::Tensor& t, int i) {
    [enc setBuffer:mtl_buffer(t) offset:byte_offset(t) atIndex:i];
  }
  template <class T>
  void bytes(const T& v, int i) {
    [enc setBytes:&v length:sizeof(T) atIndex:i];
  }
  void dispatch(int gx, int gy, int gz, int tx, int ty, int tz) {
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
        threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  }
};

// Encode onto torch's current MPS command buffer. Torch commits it at the next
// stream sync, so the work is ordered against surrounding torch ops without an
// explicit wait here.
template <class F>
void encode(F fn) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    id<MTLDevice> dev = cb.device;
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      TorchEncoder e{enc, dev};
      fn(e);
      [enc endEncoding];
    });
  }
}

// ---- sparse MLA decode ---------------------------------------------------
//
// The Metal counterpart of the CUDA `mla_decode_fp8_sparse`. Same contract:
// top-k-gathered sparse MQA decode over the paged fp8 MLA latent, with the
// value being the leading 512 lanes of each 576-wide slot.

at::Tensor mla_decode_fp8_sparse(const at::Tensor& q, const at::Tensor& kv_data,
                                 const at::Tensor& kv_scale,
                                 const at::Tensor& block_table,
                                 const at::Tensor& topk_indices,
                                 const at::Tensor& topk_length,
                                 int64_t block_size, double sm_scale,
                                 int64_t partition_size) {
  check_mps(q, "q");
  check_mps(kv_data, "kv_data");
  check_mps(kv_scale, "kv_scale");
  check_mps(block_table, "block_table");
  check_mps(topk_indices, "topk_indices");
  check_mps(topk_length, "topk_length");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, heads, 576], got ", q.sizes());

  const int batch = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int max_topk = static_cast<int>(topk_indices.size(-1));
  const int block_table_stride = static_cast<int>(block_table.stride(0));

  at::Tensor out = at::empty({batch, num_heads, 512}, q.options());

  TORCH_CHECK(partition_size == 0,
              "quixicore(metal): the partitioned sparse MLA decode is not "
              "wired yet; pass partition_size=0.");

  encode([&](TorchEncoder& e) {
    tk::launch_mla_decode_fp8_sparse(
        e, q, kv_data, kv_scale, block_table, topk_indices, topk_length, out,
        batch, num_heads, static_cast<int>(block_size), block_table_stride,
        static_cast<float>(sm_scale), max_topk);
  });
  return out;
}

// ---- dense/GQA paged attention ------------------------------------------
//
// The decode fast path behind vllm/v1/attention/backends/metal_attn.py. Reads
// the (num_blocks, block_size, H_kv, D) halves of the KV cache in place; the
// GQA head ratio is handled inside the kernel.

at::Tensor paged_attention(const at::Tensor& q, const at::Tensor& key_cache,
                           const at::Tensor& value_cache,
                           const at::Tensor& block_table,
                           const at::Tensor& context_lens, double scale) {
  check_mps(q, "q");
  check_mps(key_cache, "key_cache");
  check_mps(value_cache, "value_cache");
  check_mps(block_table, "block_table");
  check_mps(context_lens, "context_lens");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, heads, head_size], got ",
              q.sizes());
  TORCH_CHECK(
      key_cache.dim() == 4,
      "key_cache must be [blocks, block_size, kv_heads, head_size], got ",
      key_cache.sizes());

  const int batch = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int head_size = static_cast<int>(q.size(2));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int num_kv_heads = static_cast<int>(key_cache.size(2));
  const int block_table_stride = static_cast<int>(block_table.stride(0));

  at::Tensor out = at::empty_like(q);

  // The kernel takes alibi/mask buffers unconditionally; neither is wired, so
  // pass q as an inert stand-in with its use flag off rather than allocate.
  encode([&](TorchEncoder& e) {
    tk::launch_paged_attention(e, q, key_cache, value_cache, block_table,
                               context_lens, out, batch, num_heads,
                               num_kv_heads, head_size, block_size,
                               block_table_stride, static_cast<float>(scale),
                               /*alibi_slopes=*/q, /*use_alibi=*/0,
                               /*block_mask=*/q, /*use_mask=*/0, /*window=*/0,
                               /*mask_heads=*/0, activation_type_name(q));
  });
  return out;
}

// ---- DeepSeek-V4 packed sparse MLA ---------------------------------------

void deepseek_v4_save_partial_states(
    const at::Tensor& kv_in, const at::Tensor& score_in,
    const at::Tensor& ape_in, const at::Tensor& positions_in,
    const at::Tensor& state_cache, const at::Tensor& slot_mapping_in,
    int64_t block_size, int64_t state_width, int64_t compress_ratio) {
  check_mps(kv_in, "kv");
  check_mps(score_in, "score");
  check_mps(ape_in, "ape");
  check_mps_strided(state_cache, "state_cache");
  TORCH_CHECK(kv_in.scalar_type() == at::kBFloat16 &&
                  score_in.scalar_type() == at::kBFloat16 &&
                  ape_in.scalar_type() == at::kBFloat16 &&
                  state_cache.scalar_type() == at::kFloat,
              "DeepSeek-V4 partial-state inputs must be bfloat16 and the "
              "state cache must be float32");
  TORCH_CHECK(kv_in.dim() == 2 && score_in.sizes() == kv_in.sizes(),
              "kv and score must be [tokens, head_size]");
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.stride(2) == 1,
              "state_cache must have contiguous state rows");
  const int tokens = static_cast<int>(slot_mapping_in.size(0));
  const int head_size = static_cast<int>(kv_in.size(1));
  auto kv = kv_in.narrow(0, 0, tokens).contiguous();
  auto score = score_in.narrow(0, 0, tokens).contiguous();
  auto ape = ape_in.to(at::kBFloat16).contiguous();
  auto positions = positions_in.narrow(0, 0, tokens).to(at::kInt).contiguous();
  auto slots = slot_mapping_in.to(at::kLong).contiguous();
  encode([&](TorchEncoder& e) {
    tk::launch_dsv4_save_partial_states(
        e, kv, score, ape, positions, state_cache, slots, tokens, head_size,
        static_cast<int>(block_size), static_cast<int>(state_cache.stride(0)),
        static_cast<int>(state_cache.stride(1)), static_cast<int>(state_width),
        static_cast<int>(compress_ratio));
  });
}

at::Tensor deepseek_v4_qnorm_rope_kv_insert(
    const at::Tensor& q_in, const at::Tensor& kv_in, const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping_in, const at::Tensor& positions_in,
    const at::Tensor& cos_sin_cache, double eps, int64_t block_size) {
  check_mps(q_in, "q");
  check_mps(kv_in, "kv");
  check_mps_strided(kv_cache, "kv_cache");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16,
              "DeepSeek-V4 Metal q must be bfloat16");
  TORCH_CHECK(kv_in.scalar_type() == at::kBFloat16,
              "DeepSeek-V4 Metal kv must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte,
              "DeepSeek-V4 packed cache must be uint8");
  TORCH_CHECK(q_in.dim() == 3 && q_in.size(2) == 512,
              "q must be [tokens, heads, 512], got ", q_in.sizes());
  TORCH_CHECK(kv_in.dim() == 2 && kv_in.size(1) == 512,
              "kv must be [tokens, 512], got ", kv_in.sizes());
  TORCH_CHECK(kv_cache.size(-1) == 584,
              "packed DeepSeek-V4 cache slots must be 584 bytes, got ",
              kv_cache.sizes());
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.stride(2) == 1 &&
                  kv_cache.stride(1) == 584,
              "packed DeepSeek-V4 cache must have contiguous token slots, got "
              "strides ",
              kv_cache.strides());

  const int tokens = static_cast<int>(q_in.size(0));
  const int heads = static_cast<int>(q_in.size(1));
  auto positions = positions_in.to(at::kInt).contiguous();
  auto slots = slot_mapping_in.to(at::kLong).contiguous();
  auto cs = cos_sin_cache.to(at::kBFloat16).contiguous();
  TORCH_CHECK(cs.dim() == 2 && cs.size(1) == 64,
              "cos_sin_cache must be [positions, 64], got ", cs.sizes());
  auto cos = cs.slice(1, 0, 32).contiguous();
  auto sin = cs.slice(1, 32, 64).contiguous();
  auto q = q_in.contiguous();
  auto kv = kv_in.contiguous();
  auto out = at::empty_like(q);

  encode([&](TorchEncoder& e) {
    tk::launch_mla_q_norm_rope(
        e, q, cos, sin, positions, q, out, tokens * heads, heads,
        /*nope_dim=*/448, /*rope_dim=*/64, /*norm_mode=*/1,
        static_cast<float>(eps), /*head_dim=*/512);
    tk::launch_mla_kv_insert_fp8_packed(
        e, kv, cos, sin, positions, slots, kv_cache, tokens,
        static_cast<int>(block_size), static_cast<int>(kv_cache.stride(0)));
  });
  return out;
}

void deepseek_v4_kv_insert(const at::Tensor& kv_in, const at::Tensor& kv_cache,
                           const at::Tensor& slot_mapping_in,
                           const at::Tensor& positions_in,
                           const at::Tensor& cos_sin_cache,
                           int64_t block_size) {
  check_mps(kv_in, "kv");
  check_mps_strided(kv_cache, "kv_cache");
  TORCH_CHECK(kv_in.scalar_type() == at::kBFloat16 && kv_in.size(-1) == 512,
              "DeepSeek-V4 Metal kv must be [..., 512] bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte && kv_cache.size(-1) == 584,
              "DeepSeek-V4 packed cache must have 584-byte uint8 slots");
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.stride(2) == 1 &&
                  kv_cache.stride(1) == 584,
              "packed DeepSeek-V4 cache must have contiguous token slots");
  auto kv = kv_in.reshape({-1, 512}).contiguous();
  auto slots = slot_mapping_in.to(at::kLong).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto cs = cos_sin_cache.to(at::kBFloat16).contiguous();
  auto cos = cs.slice(1, 0, 32).contiguous();
  auto sin = cs.slice(1, 32, 64).contiguous();
  const int tokens = static_cast<int>(kv.size(0));
  encode([&](TorchEncoder& e) {
    tk::launch_mla_kv_insert_fp8_packed(
        e, kv, cos, sin, positions, slots, kv_cache, tokens,
        static_cast<int>(block_size), static_cast<int>(kv_cache.stride(0)));
  });
}

at::Tensor deepseek_v4_sparse_attention(
    const at::Tensor& q_in, const at::Tensor& compressed_cache,
    const at::Tensor& compressed_slots_in, const at::Tensor& compressed_lens_in,
    const at::Tensor& swa_cache, const at::Tensor& swa_slots_in,
    const at::Tensor& swa_lens_in, const at::Tensor& sinks_in, double scale) {
  check_mps(q_in, "q");
  check_mps_strided(compressed_cache, "compressed_cache");
  check_mps_strided(swa_cache, "swa_cache");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16 && q_in.dim() == 3 &&
                  q_in.size(2) == 512,
              "q must be [tokens, heads, 512] bfloat16");
  TORCH_CHECK(compressed_cache.scalar_type() == at::kByte &&
                  compressed_cache.size(-1) == 584,
              "compressed cache must use 584-byte uint8 slots");
  TORCH_CHECK(swa_cache.scalar_type() == at::kByte && swa_cache.size(-1) == 584,
              "SWA cache must use 584-byte uint8 slots");
  TORCH_CHECK(compressed_cache.dim() == 3 && swa_cache.dim() == 3 &&
                  compressed_cache.stride(2) == 1 &&
                  compressed_cache.stride(1) == 584 &&
                  swa_cache.stride(2) == 1 && swa_cache.stride(1) == 584,
              "DeepSeek-V4 cache token slots must be contiguous");
  const int batch = static_cast<int>(q_in.size(0));
  const int heads = static_cast<int>(q_in.size(1));
  auto q = q_in.contiguous();
  auto compressed_slots = compressed_slots_in.to(at::kInt).contiguous();
  auto compressed_lens = compressed_lens_in.to(at::kInt).contiguous();
  auto swa_slots = swa_slots_in.to(at::kInt).contiguous();
  auto swa_lens = swa_lens_in.to(at::kInt).contiguous();
  auto sinks = sinks_in.to(at::kFloat).contiguous();
  TORCH_CHECK(compressed_slots.dim() == 2 && swa_slots.dim() == 2,
              "sparse slot lists must be rank 2");
  auto out = at::empty_like(q);
  encode([&](TorchEncoder& e) {
    tk::launch_mla_decode_fp8_sparse_two_cache_packed(
        e, q, compressed_cache, compressed_slots, compressed_lens, swa_cache,
        swa_slots, swa_lens, sinks, out, batch, heads,
        static_cast<int>(compressed_slots.size(1)),
        static_cast<int>(swa_slots.size(1)),
        static_cast<int>(compressed_cache.size(1)),
        static_cast<int>(compressed_cache.stride(0)),
        static_cast<int>(swa_cache.size(1)),
        static_cast<int>(swa_cache.stride(0)), static_cast<float>(scale));
  });
  return out;
}

// ---- TurboQuant combined paged cache -------------------------------------

void turboquant_encode_metal(const at::Tensor& key_in,
                             const at::Tensor& value_in,
                             const at::Tensor& kv_cache,
                             const at::Tensor& slot_mapping_in,
                             const at::Tensor& centroids_in,
                             const at::Tensor& signs_in, int64_t k_bits,
                             bool k_signed, int64_t v_bits) {
  check_mps(key_in, "key");
  check_mps(value_in, "value");
  check_mps_strided(kv_cache, "kv_cache");
  TORCH_CHECK(key_in.dim() == 3 && value_in.sizes() == key_in.sizes(),
              "TurboQuant key/value must be [tokens, kv_heads, head_size]");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte,
              "TurboQuant Metal cache must be uint8");
  const int tokens = static_cast<int>(key_in.size(0));
  const int kv_heads = static_cast<int>(key_in.size(1));
  const int head_size = static_cast<int>(key_in.size(2));
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256 ||
                  head_size == 512,
              "TurboQuant Metal head_size must be 64/128/256/512");
  auto key = key_in.contiguous();
  auto value = value_in.to(key_in.scalar_type()).contiguous();
  auto slots = slot_mapping_in.to(at::kInt).contiguous();
  auto centroids = centroids_in.to(at::kFloat).contiguous();
  auto signs = signs_in.to(at::kFloat).contiguous();
  const int block_size = static_cast<int>(kv_cache.size(1));
  const int block_stride = static_cast<int>(kv_cache.stride(0));
  const int token_stride = static_cast<int>(kv_cache.stride(1));
  const int head_stride = static_cast<int>(kv_cache.stride(2));
  const int slot_size = static_cast<int>(kv_cache.size(3));
  encode([&](TorchEncoder& e) {
    tk::launch_tq_encode_combined(
        e, key, value, kv_cache, slots, centroids, signs, tokens, kv_heads,
        head_size, block_size, block_stride, token_stride, head_stride,
        slot_size, static_cast<int>(k_bits), k_signed ? 1 : 0,
        static_cast<int>(v_bits), activation_type_name(key));
  });
}

at::Tensor turboquant_attention_metal(
    const at::Tensor& q_in, const at::Tensor& kv_cache,
    const at::Tensor& slots_in, const at::Tensor& lengths_in,
    const at::Tensor& centroids_in, const at::Tensor& signs_in,
    const at::Tensor& sinks_in, double scale, int64_t num_kv_heads,
    int64_t k_bits, bool k_signed, int64_t v_bits) {
  check_mps(q_in, "q");
  check_mps_strided(kv_cache, "kv_cache");
  TORCH_CHECK(q_in.dim() == 3,
              "TurboQuant q must be [batch, heads, head_size]");
  const int batch = static_cast<int>(q_in.size(0));
  const int heads = static_cast<int>(q_in.size(1));
  const int head_size = static_cast<int>(q_in.size(2));
  auto q = q_in.contiguous();
  auto slots = slots_in.to(at::kInt).contiguous();
  auto lengths = lengths_in.to(at::kInt).contiguous();
  auto centroids = centroids_in.to(at::kFloat).contiguous();
  auto signs = signs_in.to(at::kFloat).contiguous();
  auto sinks = sinks_in.to(at::kFloat).contiguous();
  auto out = at::empty_like(q);
  encode([&](TorchEncoder& e) {
    tk::launch_tq_attention_combined(
        e, q, kv_cache, slots, lengths, centroids, signs, sinks, out, batch,
        heads, static_cast<int>(num_kv_heads), head_size,
        static_cast<int>(slots.size(1)), static_cast<int>(kv_cache.size(1)),
        static_cast<int>(kv_cache.stride(0)),
        static_cast<int>(kv_cache.stride(1)),
        static_cast<int>(kv_cache.stride(2)),
        static_cast<int>(kv_cache.size(3)), static_cast<int>(k_bits),
        k_signed ? 1 : 0, static_cast<int>(v_bits), static_cast<float>(scale),
        activation_type_name(q));
  });
  return out;
}

// ---- GGUF quantized matmul ----------------------------------------------
//
// The GGUF layer speaks GGML type ids; the kernels are named by QuixiCore's
// format strings. This is the only place the two vocabularies meet.
//
// Ids come from GGMLQuantizationType in the `gguf` package. Only formats with
// a compiled qgemv/qgemm kernel appear here -- an id that is missing raises
// rather than silently selecting the wrong shader, because a wrong format
// reads the weight bytes with the wrong block layout and returns plausible
// garbage instead of failing.

const char* ggml_type_to_format(int64_t quant_type) {
  switch (quant_type) {
    case 2:
      return "q4_0";
    case 3:
      return "q4_1";
    case 6:
      return "q5_0";
    case 7:
      return "q5_1";
    case 8:
      return "q8_0";
    case 10:
      return "q2_K";
    case 11:
      return "q3_K";
    case 12:
      return "q4_K";
    case 13:
      return "q5_K";
    case 14:
      return "q6_K";
    case 16:
      return "iq2_xxs";
    case 17:
      return "iq2_xs";
    case 18:
      return "iq3_xxs";
    case 19:
      return "iq1_s";
    case 20:
      return "iq4_nl";
    case 23:
      return "iq4_xs";
    case 39:
      return "mxfp4";
    default:
      TORCH_CHECK(false,
                  "quixicore(metal): no Metal kernel for GGML quant type ",
                  quant_type);
  }
}

// Weight-only GEMV: one row of output per simdgroup. `w` holds raw GGUF blocks.
at::Tensor ggml_mul_mat_vec_a8(const at::Tensor& w, const at::Tensor& x,
                               int64_t quant_type, int64_t row) {
  check_mps(w, "w");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(-1));

  at::Tensor out = at::empty({x.size(0), N}, x.options());
  const std::string fmt = ggml_type_to_format(quant_type);
  const std::string type_name = activation_type_name(x);

  encode([&](TorchEncoder& e) {
    // QuixiCore's qgemv shader is intentionally batch-1: its grid is N and
    // the input/output pointers name one row.  vLLM sends decode batches up
    // to max_num_seqs here, so bind each row's storage offset and enqueue all
    // rows into the same command encoder.  The previous single launch only
    // initialized out[0] and returned garbage for requests 1..M-1.
    const int batch = static_cast<int>(x.size(0));
    for (int b = 0; b < batch; ++b) {
      const at::Tensor x_row = x.select(0, b);
      const at::Tensor out_row = out.select(0, b);
      tk::launch_qgemv(e, out_row, w, x_row, N, K, fmt, type_name);
    }
  });
  return out;
}

at::Tensor ggml_moe_a8_vec(const at::Tensor& x, const at::Tensor& w,
                           const at::Tensor& topk_ids_in, int64_t top_k,
                           int64_t quant_type, int64_t row, int64_t tokens) {
  check_mps(w, "w");
  check_mps(x, "x");
  check_mps(topk_ids_in, "topk_ids");
  TORCH_CHECK(x.dim() == 2, "MoE input must be [tokens, K], got ", x.sizes());
  TORCH_CHECK(w.dim() >= 3, "MoE weights must be [experts, N, packed-K], got ",
              w.sizes());
  TORCH_CHECK(topk_ids_in.dim() == 2, "topk_ids must be rank 2, got ",
              topk_ids_in.sizes());
  TORCH_CHECK(tokens > 0 && top_k > 0,
              "tokens and top_k must be positive, got ", tokens, " and ",
              top_k);
  TORCH_CHECK(x.size(0) == tokens,
              "MoE input row count must equal tokens, got ", x.size(0), " and ",
              tokens);
  TORCH_CHECK(topk_ids_in.numel() == tokens * top_k,
              "topk_ids must contain tokens * top_k entries, got ",
              topk_ids_in.numel(), " for ", tokens, " * ", top_k);
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "Metal GGUF MoE supports float16/bfloat16 activations");

  const int num_tokens = static_cast<int>(tokens);
  const int topk = static_cast<int>(top_k);
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(1));
  auto topk_ids = topk_ids_in.to(at::kInt).contiguous();
  auto input = x.contiguous();
  auto out = at::empty({num_tokens * topk, N}, x.options());
  const std::string fmt = ggml_type_to_format(quant_type);
  encode([&](TorchEncoder& e) {
    tk::launch_qgemv_moe(e, out, w, input, topk_ids, N, K, num_tokens, topk,
                         fmt, activation_type_name(input));
  });
  return out;
}

// Weight-only GEMM for the batched path.
at::Tensor ggml_mul_mat_a8(const at::Tensor& w, const at::Tensor& x,
                           int64_t quant_type, int64_t row) {
  check_mps(w, "w");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(-1));
  const int output_rows = static_cast<int>(x.size(0));
  const std::string fmt = ggml_type_to_format(quant_type);
  const int tile_m = (fmt == "mxfp8") ? 64 : 32;
  const int M = ((output_rows + tile_m - 1) / tile_m) * tile_m;

  TORCH_CHECK(N % 32 == 0,
              "quixicore(metal): qgemm needs N % 32 == 0, got N=", N);

  // The Metal tile consumes X as column-major [K, M] fp16 and emits [N, M]
  // fp16. vLLM supplies row-major [rows, K] BF16, and decode/spec batches do
  // not generally end on a 32-column tile. Convert, transpose, and pad here so
  // the public op preserves the ordinary [rows, N] dtype/layout contract.
  auto input = x.to(at::kHalf).transpose(0, 1).contiguous();
  auto input_padded = at::zeros({K, M}, x.options().dtype(at::kHalf));
  input_padded.narrow(1, 0, output_rows).copy_(input);
  auto output_padded = at::empty({N, M}, x.options().dtype(at::kHalf));

  encode([&](TorchEncoder& e) {
    tk::launch_qgemm(e, output_padded, w, input_padded, N, K, M, fmt);
  });
  return output_padded.narrow(1, 0, output_rows)
      .transpose(0, 1)
      .to(x.scalar_type())
      .contiguous();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "QuixiCore Metal kernels for the Apple Silicon serving path";

  m.def("_set_library", &set_library,
        "Point the extension at the compiled quixicore_metal.metallib",
        pybind11::arg("path"));

  m.def("mla_decode_fp8_sparse", &mla_decode_fp8_sparse,
        "Top-k sparse MQA decode over the paged fp8 MLA latent",
        pybind11::arg("q"), pybind11::arg("kv_data"), pybind11::arg("kv_scale"),
        pybind11::arg("block_table"), pybind11::arg("topk_indices"),
        pybind11::arg("topk_length"), pybind11::arg("block_size"),
        pybind11::arg("sm_scale"), pybind11::arg("partition_size") = 0);

  m.def("paged_attention", &paged_attention,
        "Dense/GQA paged attention decode over the block-table KV cache",
        pybind11::arg("q"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_table"),
        pybind11::arg("context_lens"), pybind11::arg("scale"));

  m.def("deepseek_v4_qnorm_rope_kv_insert", &deepseek_v4_qnorm_rope_kv_insert,
        "DeepSeek-V4 Q norm/RoPE plus packed FP8 KV insert", pybind11::arg("q"),
        pybind11::arg("kv"), pybind11::arg("kv_cache"),
        pybind11::arg("slot_mapping"), pybind11::arg("positions"),
        pybind11::arg("cos_sin_cache"), pybind11::arg("eps"),
        pybind11::arg("block_size"));

  m.def("deepseek_v4_save_partial_states", &deepseek_v4_save_partial_states,
        "Store DeepSeek-V4 compressor partial states without an MPS sync",
        pybind11::arg("kv"), pybind11::arg("score"), pybind11::arg("ape"),
        pybind11::arg("positions"), pybind11::arg("state_cache"),
        pybind11::arg("slot_mapping"), pybind11::arg("block_size"),
        pybind11::arg("state_width"), pybind11::arg("compress_ratio"));

  m.def("deepseek_v4_kv_insert", &deepseek_v4_kv_insert,
        "Insert compressed DeepSeek-V4 rows into the packed FP8 cache",
        pybind11::arg("kv"), pybind11::arg("kv_cache"),
        pybind11::arg("slot_mapping"), pybind11::arg("positions"),
        pybind11::arg("cos_sin_cache"), pybind11::arg("block_size"));

  m.def("deepseek_v4_sparse_attention", &deepseek_v4_sparse_attention,
        "Two-cache sparse DeepSeek-V4 attention over packed FP8 slots",
        pybind11::arg("q"), pybind11::arg("compressed_cache"),
        pybind11::arg("compressed_slots"), pybind11::arg("compressed_lens"),
        pybind11::arg("swa_cache"), pybind11::arg("swa_slots"),
        pybind11::arg("swa_lens"), pybind11::arg("sinks"),
        pybind11::arg("scale"));

  m.def("turboquant_encode_metal", &turboquant_encode_metal,
        "TurboQuant encode into vLLM's combined paged cache",
        pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("kv_cache"),
        pybind11::arg("slot_mapping"), pybind11::arg("centroids"),
        pybind11::arg("signs"), pybind11::arg("k_bits"),
        pybind11::arg("k_signed"), pybind11::arg("v_bits"));

  m.def("turboquant_attention_metal", &turboquant_attention_metal,
        "Fused TurboQuant paged attention on Metal", pybind11::arg("q"),
        pybind11::arg("kv_cache"), pybind11::arg("slots"),
        pybind11::arg("lengths"), pybind11::arg("centroids"),
        pybind11::arg("signs"), pybind11::arg("sinks"), pybind11::arg("scale"),
        pybind11::arg("num_kv_heads"), pybind11::arg("k_bits"),
        pybind11::arg("k_signed"), pybind11::arg("v_bits"));

  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8,
        "GGUF weight-only GEMV over raw quantized blocks", pybind11::arg("w"),
        pybind11::arg("x"), pybind11::arg("quant_type"), pybind11::arg("row"));

  m.def("ggml_moe_a8_vec", &ggml_moe_a8_vec,
        "Device-selected GGUF MoE GEMV over raw quantized expert blocks",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("topk_ids"),
        pybind11::arg("top_k"), pybind11::arg("quant_type"),
        pybind11::arg("row"), pybind11::arg("tokens"));

  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8,
        "GGUF weight-only GEMM over raw quantized blocks", pybind11::arg("w"),
        pybind11::arg("x"), pybind11::arg("quant_type"), pybind11::arg("row"));
}
