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

// The kernels are instantiated per element type; this is the suffix that
// selects the instantiation.
std::string activation_type_name(const at::Tensor& x) {
  switch (x.scalar_type()) {
    case at::kFloat: return "float32";
    case at::kHalf: return "float16";
    case at::kBFloat16: return "bfloat16";
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
  TORCH_CHECK(fn != nil, "quixicore: kernel not found in metallib: ",
              name.UTF8String);
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
    [enc setComputePipelineState:
             ::pipeline(device, [NSString stringWithUTF8String:name.c_str()])];
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
                                 const at::Tensor& topk_length, int64_t block_size,
                                 double sm_scale, int64_t partition_size) {
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
  TORCH_CHECK(key_cache.dim() == 4,
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
    case 2:  return "q4_0";
    case 3:  return "q4_1";
    case 6:  return "q5_0";
    case 7:  return "q5_1";
    case 8:  return "q8_0";
    case 10: return "q2_K";
    case 11: return "q3_K";
    case 12: return "q4_K";
    case 13: return "q5_K";
    case 14: return "q6_K";
    case 16: return "iq2_xxs";
    case 17: return "iq2_xs";
    case 18: return "iq3_xxs";
    case 19: return "iq1_s";
    case 20: return "iq4_nl";
    case 23: return "iq4_xs";
    case 39: return "mxfp4";
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

// Weight-only GEMM for the batched path.
at::Tensor ggml_mul_mat_a8(const at::Tensor& w, const at::Tensor& x,
                           int64_t quant_type, int64_t row) {
  check_mps(w, "w");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(-1));
  const int M = static_cast<int>(x.size(0));

  at::Tensor out = at::empty({M, N}, x.options());
  const std::string fmt = ggml_type_to_format(quant_type);

  // The tile kernel derives its grid by integer division, so a partial tile is
  // not computed at all -- it would return an uninitialized tail rather than
  // fail. Callers pad, or take the GEMV path.
  const int tile_m = (fmt == "mxfp8") ? 64 : 32;
  TORCH_CHECK(M % tile_m == 0 && N % 32 == 0,
              "quixicore(metal): qgemm needs M % ", tile_m, " == 0 and "
              "N % 32 == 0, got M=", M, " N=", N);

  encode([&](TorchEncoder& e) {
    tk::launch_qgemm(e, out, w, x, N, K, M, fmt);
  });
  return out;
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

  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8,
        "GGUF weight-only GEMV over raw quantized blocks",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("quant_type"),
        pybind11::arg("row"));

  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8,
        "GGUF weight-only GEMM over raw quantized blocks",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("quant_type"),
        pybind11::arg("row"));
}
