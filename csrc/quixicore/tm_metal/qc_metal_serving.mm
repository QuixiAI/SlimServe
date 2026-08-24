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
// grid and threadgroup geometry -- lives in kernels/common/tk_launch.h
// (co-maintained with the shaders) for the vendored kernel families. A few
// DSV4 serving-local kernels (mhc, rms_norm, indexer q/topk, o_inv_rope)
// encode inline here; their ABI is stated at each encode site. Beyond that
// this file supplies a Torch encoder adapter and the tensor/buffer plumbing.
//
// This file owns the single PYBIND11_MODULE; sibling .mm files add to it.

#include <torch/extension.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <objc/runtime.h>

#include "tk_launch.h"

namespace {

// ---- transient-output ring --------------------------------------------
// Per-(tag, shape, dtype) ring of reused output tensors for the per-step
// op outputs on the decode hot path. The MPS caching allocator recycles
// freed buffers across UNRELATED ops, which manufactures false cross-op
// hazards between resident command buffers. A stable ring pins buffer
// identities so the only hazards left are real producer->consumer edges.
// Depth 4 covers cross-step slack (async consumers, aux-stream reads) and
// must exceed the max number of same-shape, same-tag outputs live at once;
// all producers and consumers encode on the same MPS stream, so in-order
// reuse is safe. Values are bit-identical (same kernels, same math).
constexpr int kOutRingDepth = 4;
struct OutRing {
  std::array<at::Tensor, kOutRingDepth> bufs;
  uint32_t next = 0;
};

inline at::Tensor ring_out(const char* tag, at::IntArrayRef sizes,
                           const at::TensorOptions& opts) {
  // Decode-scale outputs only: prefill-width tensors (tens of MB) stay on
  // the caching allocator rather than being pinned 4-deep per shape.
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  if (numel * static_cast<int64_t>(opts.dtype().itemsize()) > (8LL << 20)) {
    return at::empty(sizes, opts);
  }
  static std::unordered_map<std::string, OutRing> rings;
  std::string key(tag);
  key += '|';
  key +=
      std::to_string(static_cast<int>(c10::typeMetaToScalarType(opts.dtype())));
  for (auto s : sizes) {
    key += ',';
    key += std::to_string(s);
  }
  OutRing& r = rings[key];
  at::Tensor& t = r.bufs[r.next % kOutRingDepth];
  r.next++;
  if (!t.defined()) t = at::empty(sizes, opts);
  return t;
}

inline at::Tensor ring_out_like(const char* tag, const at::Tensor& x) {
  if (x.is_contiguous()) return ring_out(tag, x.sizes(), x.options());
  return at::empty_like(x);
}

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
void encode(const char* label, F fn) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    id<MTLDevice> dev = cb.device;
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      // Names the encoder in Metal System Trace GPU intervals.
      enc.label = [NSString stringWithUTF8String:label];
      TorchEncoder e{enc, dev};
      fn(e);
      [enc endEncoding];
    });
  }
}

template <class F>
void encode(F fn) {
  encode("quixicore", fn);
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

  at::Tensor out = ring_out("mla_out", {batch, num_heads, 512}, q.options());

  TORCH_CHECK(partition_size == 0,
              "quixicore(metal): the partitioned sparse MLA decode is not "
              "wired yet; pass partition_size=0.");

  encode("qc_mla_decode_fp8_sparse", [&](TorchEncoder& e) {
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
                           const at::Tensor& context_lens, double scale,
                           int64_t window) {
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
  encode("qc_paged_attention", [&](TorchEncoder& e) {
    tk::launch_paged_attention(e, q, key_cache, value_cache, block_table,
                               context_lens, out, batch, num_heads,
                               num_kv_heads, head_size, block_size,
                               block_table_stride, static_cast<float>(scale),
                               /*alibi_slopes=*/q, /*use_alibi=*/0,
                               /*block_mask=*/q, /*use_mask=*/0,
                               /*window=*/static_cast<int>(window),
                               /*mask_heads=*/0, activation_type_name(q));
  });
  return out;
}

std::tuple<at::Tensor, at::Tensor> kv_cache_gather_range(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table_in, int64_t token_start, int64_t num_tokens) {
  check_mps_strided(key_cache, "key_cache");
  check_mps_strided(value_cache, "value_cache");
  check_mps(block_table_in, "block_table");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(),
              "KV caches must both be [blocks, block_size, heads, dim]");
  TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(),
              "key/value cache dtype mismatch");
  TORCH_CHECK(
      key_cache.stride(1) == key_cache.size(2) * key_cache.size(3) &&
          value_cache.stride(1) == value_cache.size(2) * value_cache.size(3),
      "KV cache token rows must be contiguous");
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0),
              "key/value cache block strides must match");
  TORCH_CHECK(block_table_in.dim() == 1, "block_table must be one request row");
  TORCH_CHECK(token_start >= 0 && num_tokens >= 0,
              "token range must be non-negative");

  const int nblocks = static_cast<int>(key_cache.size(0));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int heads = static_cast<int>(key_cache.size(2));
  const int head_size = static_cast<int>(key_cache.size(3));
  const int start = static_cast<int>(token_start);
  const int count = static_cast<int>(num_tokens);
  TORCH_CHECK(count == 0 || (start + count + block_size - 1) / block_size <=
                                block_table_in.numel(),
              "token range exceeds block table");

  auto block_table = block_table_in.to(at::kInt).contiguous();
  auto key_out = at::empty({count, heads, head_size}, key_cache.options());
  auto value_out = at::empty_like(key_out);
  if (count == 0) return {key_out, value_out};

  const int64_t cache_block_stride = key_cache.stride(0);
  encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_gather_range(
        e, key_cache, value_cache, key_out, value_out, block_table, start,
        count, nblocks, block_size, heads, head_size, cache_block_stride,
        activation_type_name(key_cache));
  });
  return {key_out, value_out};
}

// Multi-query verify attention (bench-first): the m expanded rows share
// each K/V read via cooperative tiles. Same partition/reduce math as the
// v2 kernels; per-row causal boundary and window applied in-kernel.
at::Tensor paged_attention_verify(const at::Tensor& q,
                                  const at::Tensor& key_cache,
                                  const at::Tensor& value_cache,
                                  const at::Tensor& block_table,
                                  const at::Tensor& context_lens, double scale,
                                  int64_t window) {
  check_mps(q, "q");
  check_mps(key_cache, "key_cache");
  const int m = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int head_size = static_cast<int>(q.size(2));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int num_kv_heads = static_cast<int>(key_cache.size(2));
  const int bt_stride = static_cast<int>(block_table.stride(0));
  TORCH_CHECK(head_size == 128 && m >= 1 && m <= 32,
              "quixicore(metal): paged_attention_verify wants D=128, m<=32");
  constexpr int kPartitionSize = 512;
  const int ctx_hint = static_cast<int>(context_lens.max().item<int64_t>());
  const int num_partitions =
      std::max(1, (ctx_hint + kPartitionSize - 1) / kPartitionSize);
  auto opt = q.options().dtype(at::kFloat);
  auto tmp = at::empty({m, num_heads, num_partitions, head_size}, opt);
  auto mlog = at::empty({m, num_heads, num_partitions}, opt);
  auto esum = at::empty({m, num_heads, num_partitions}, opt);
  auto out = at::empty_like(q);
  encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_verify(
        e, q, key_cache, value_cache, block_table, context_lens, tmp, mlog,
        esum, m, num_heads, num_kv_heads, head_size, block_size, bt_stride,
        static_cast<float>(scale), num_partitions, kPartitionSize,
        static_cast<int>(window), activation_type_name(q));
    tk::launch_paged_attention_reduce(
        e, tmp, mlog, esum, out, m, num_heads, head_size, num_partitions,
        /*sinks=*/q, /*has_sink=*/0, activation_type_name(q));
  });
  return out;
}

// ---- Muse-Glimmer fused decode step --------------------------------------
//
// The deep fix for eager-mode dispatch overhead: the whole 52-layer dense
// decode forward is encoded into ONE command buffer from a C++ loop, so the
// per-op Python/ATen/torch-MPS cost (measured ~40 ms of an ~80 ms step)
// collapses to microseconds of encoding. Weights and scratch are registered
// once; each step passes only the tensors that change (hidden rows,
// positions, block tables, sequence lengths, slot mappings).

const char* ggml_type_to_format(int64_t quant_type);

namespace muse_step {

struct Proj {
  at::Tensor w;
  int type = 0;
  int rows = 0;
};

struct Layer {
  bool is_local = false;
  std::vector<Proj> qkv;  // q, k, v
  Proj gate, o, down;
  std::vector<Proj> gate_up;  // gate, up
  at::Tensor norm1, qn, kn, post_attn, norm2, post_ffn;
  at::Tensor kv_cache;  // (2, blocks, block_size, kv_heads, head_dim)
};

struct State {
  bool ready = false;
  int hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0;
  int window = 0, max_rows = 0;
  float theta = 0.f, eps = 0.f, post_eps = 0.f, scale = 0.f;
  std::vector<Layer> layers;
  at::Tensor h, q, k, v, attn_out, gate_out, o_out, g_out, u_out, mlp_mid;
  at::Tensor mq_tmp, mq_mlog, mq_esum;
  // Rotating split-K partial buffers: independent matmuls (q/k/v/gate,
  // gate/up) would otherwise serialize on a shared buffer's write-write
  // hazard in Metal's tracker.
  std::array<at::Tensor, 4> gemm_partials;
  int gemm_partials_idx = 0;
  // Rotating (K_max, 32) half scratch for the deep-K transpose-first route.
  std::array<at::Tensor, 2> xpose;
  int xpose_idx = 0;
};

State g;

void emit_matvec(TorchEncoder& e, const Proj& p, const at::Tensor& x,
                 const at::Tensor& out, int m) {
  const int K = static_cast<int>(x.size(-1));
  const std::string fmt = ggml_type_to_format(p.type);
  const bool sm_fmt = fmt == "q4_0" || fmt == "q8_0" || fmt == "q4_K" ||
                      fmt == "q5_K" || fmt == "q6_K";
  if (m == 1) {
    tk::launch_qgemv(e, out, p.w, x, p.rows, K, fmt, "bfloat16");
  } else if (m >= 9 && sm_fmt && p.rows % 16 == 0 && K % 32 == 0) {
    // verify band: weight-streaming split-K MMA. One contiguous transpose
    // pass puts X in the (K, 32) half layout every variant stages
    // contiguously; the paired-plane kernels take q4_K/q5_K, BK=64 takes
    // q6_K, and the BK=32 kernel covers the rest / unaligned K.
    const at::Tensor& partials = g.gemm_partials[g.gemm_partials_idx];
    g.gemm_partials_idx = (g.gemm_partials_idx + 1) % 4;
    const at::Tensor& xp = g.xpose[g.xpose_idx];
    g.xpose_idx = (g.xpose_idx + 1) % 2;
    int variant = 9;
    if (K % 64 == 0) {
      const bool kquant = fmt == "q4_K" || fmt == "q5_K" || fmt == "q6_K";
      if (kquant && p.rows % 32 == 0) {
        // mirror the eager route (quixicore/ops.py): tensor-ops kernels,
        // per-shape 15/16/17 selection (the rested-A/B winner)
        if (p.rows % 64 == 0 && p.rows >= 16384) {
          variant = 16;
        } else if (fmt == "q5_K" && p.rows < 16384 && K % 128 == 0) {
          variant = 17;
        } else {
          variant = 15;
        }
      } else if (fmt == "q4_K" || fmt == "q5_K") {
        variant = (p.rows > 8192 || K > 8192) ? 12 : 11;
      } else if (fmt == "q6_K") {
        variant = 10;
      }
    }
    tk::launch_muse_xpose32(e, xp, x, K, m);
    tk::launch_qgemm_sm(e, partials, p.w, xp, p.rows, K, variant, fmt);
    tk::launch_qgemm_sm_reduce_rm(e, out, partials, p.rows, m);
  } else {
    tk::launch_qgemv_mm(e, out, p.w, x, p.rows, K, m, fmt, "bfloat16");
  }
}

void emit_rms(TorchEncoder& e, const at::Tensor& x, const at::Tensor& w,
              const at::Tensor& o, int rows, int d, float eps) {
  tk::launch_rms_norm_dyn(e, x, w, o, static_cast<uint32_t>(rows), d, eps);
}

void emit_elemwise(TorchEncoder& e, const std::string& name, int n4) {
  e.pipeline(name);
  e.dispatch((n4 + 255) / 256, 1, 1, 256, 1, 1);
}

}  // namespace muse_step

void muse_step_init(int64_t num_layers, int64_t hidden, int64_t heads,
                    int64_t kv_heads, int64_t head_dim, int64_t inter,
                    int64_t window, double theta, double eps, double post_eps,
                    int64_t max_rows, const at::Tensor& ref) {
  using namespace muse_step;
  g = State{};
  g.hidden = static_cast<int>(hidden);
  g.heads = static_cast<int>(heads);
  g.kv_heads = static_cast<int>(kv_heads);
  g.head_dim = static_cast<int>(head_dim);
  g.inter = static_cast<int>(inter);
  g.window = static_cast<int>(window);
  g.max_rows = static_cast<int>(max_rows);
  g.theta = static_cast<float>(theta);
  g.eps = static_cast<float>(eps);
  g.post_eps = static_cast<float>(post_eps);
  g.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  g.layers.resize(static_cast<size_t>(num_layers));
  const auto opt = ref.options();
  const int64_t m = max_rows;
  g.h = at::empty({m, hidden}, opt);
  g.q = at::empty({m, heads * head_dim}, opt);
  g.k = at::empty({m, kv_heads * head_dim}, opt);
  g.v = at::empty({m, kv_heads * head_dim}, opt);
  g.attn_out = at::empty({m, heads * head_dim}, opt);
  g.gate_out = at::empty({m, heads * head_dim}, opt);
  g.o_out = at::empty({m, hidden}, opt);
  g.g_out = at::empty({m, inter}, opt);
  g.u_out = at::empty({m, inter}, opt);
  g.mlp_mid = at::empty({m, inter}, opt);
  const int64_t max_n =
      std::max<int64_t>(inter, std::max<int64_t>(heads * head_dim, hidden));
  for (auto& buf : g.gemm_partials) {
    buf = at::empty({4, max_n, 32}, opt.dtype(at::kFloat));
  }
  const int64_t max_k = std::max<int64_t>(inter, hidden);
  for (auto& buf : g.xpose) {
    buf = at::empty({max_k, 32}, opt.dtype(at::kHalf));
  }
  // multi-query attention partials for long-context global layers
  // (P_max 64 partitions x 512 = 32k ctx; ~18 MB at m=17)
  g.mq_tmp = at::empty({m, heads, 64, head_dim}, opt.dtype(at::kFloat));
  g.mq_mlog = at::empty({m, heads, 64}, opt.dtype(at::kFloat));
  g.mq_esum = at::empty({m, heads, 64}, opt.dtype(at::kFloat));
}

void muse_step_layer(
    int64_t idx, bool is_local, const std::vector<at::Tensor>& qkv_w,
    const std::vector<int64_t>& qkv_t, const at::Tensor& gate_w, int64_t gate_t,
    const at::Tensor& o_w, int64_t o_t, const std::vector<at::Tensor>& gu_w,
    const std::vector<int64_t>& gu_t, const at::Tensor& down_w, int64_t down_t,
    const at::Tensor& norm1, const at::Tensor& qn, const at::Tensor& kn,
    const at::Tensor& post_attn, const at::Tensor& norm2,
    const at::Tensor& post_ffn, const at::Tensor& kv_cache) {
  using namespace muse_step;
  TORCH_CHECK(idx >= 0 && idx < (int64_t)g.layers.size(), "bad layer idx");
  TORCH_CHECK(qkv_w.size() == 3 && qkv_t.size() == 3, "qkv wants 3 shards");
  TORCH_CHECK(gu_w.size() == 2 && gu_t.size() == 2, "gate_up wants 2 shards");
  Layer& L = g.layers[static_cast<size_t>(idx)];
  L.is_local = is_local;
  const int hd = g.head_dim;
  L.qkv = {{qkv_w[0], (int)qkv_t[0], g.heads * hd},
           {qkv_w[1], (int)qkv_t[1], g.kv_heads * hd},
           {qkv_w[2], (int)qkv_t[2], g.kv_heads * hd}};
  L.gate = {gate_w, (int)gate_t, g.heads * hd};
  L.o = {o_w, (int)o_t, g.hidden};
  L.gate_up = {{gu_w[0], (int)gu_t[0], g.inter},
               {gu_w[1], (int)gu_t[1], g.inter}};
  L.down = {down_w, (int)down_t, g.hidden};
  L.norm1 = norm1;
  L.qn = qn;
  L.kn = kn;
  L.post_attn = post_attn;
  L.norm2 = norm2;
  L.post_ffn = post_ffn;
  L.kv_cache = kv_cache;
  if (idx == (int64_t)g.layers.size() - 1) {
    g.ready = true;
  }
}

void muse_step_run_impl(const at::Tensor& x, const at::Tensor& positions,
                        const at::Tensor& bt_local, const at::Tensor& sl_local,
                        const at::Tensor& slot_local, const at::Tensor& bt_full,
                        const at::Tensor& sl_full, const at::Tensor& slot_full,
                        const at::Tensor* aux_out,
                        const std::vector<int64_t>& aux_layers, int ctx_len,
                        bool rows_are_one_request) {
  using namespace muse_step;
  TORCH_CHECK(g.ready, "muse_step not initialized");
  const int m = static_cast<int>(x.size(0));
  TORCH_CHECK(m >= 1 && m <= g.max_rows, "row count out of range");
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16,
              "x must be contiguous bf16");
  const int hidden = g.hidden;
  const int hd = g.head_dim;
  const int n4_hidden = m * hidden / 4;
  const int n4_attn = m * g.heads * hd / 4;
  const int n4_inter = m * g.inter / 4;

  encode([&](TorchEncoder& e) {
    size_t aux_j = 0;
    for (size_t li = 0; li < g.layers.size(); ++li) {
      const Layer& L = g.layers[li];
      // snapshot the residual stream entering this layer for the drafter
      if (aux_out != nullptr && aux_j < aux_layers.size() &&
          (int64_t)li == aux_layers[aux_j]) {
        e.pipeline("mittens::muse_copy");
        e.out(aux_out->select(0, (int64_t)aux_j), 0);
        e.in(x, 1);
        e.bytes(n4_hidden, 2);
        e.dispatch((n4_hidden + 255) / 256, 1, 1, 256, 1, 1);
        aux_j++;
      }
      const at::Tensor& bt = L.is_local ? bt_local : bt_full;
      const at::Tensor& sl = L.is_local ? sl_local : sl_full;
      const at::Tensor& slots = L.is_local ? slot_local : slot_full;
      const int window = L.is_local ? g.window : 0;

      // h = rms(x, norm1)
      emit_rms(e, x, L.norm1, g.h, m, hidden, g.eps);
      // q/k/v projections
      emit_matvec(e, L.qkv[0], g.h, g.q, m);
      emit_matvec(e, L.qkv[1], g.h, g.k, m);
      emit_matvec(e, L.qkv[2], g.h, g.v, m);
      // per-head QK-RMSNorm (in place)
      emit_rms(e, g.q, L.qn, g.q, m * g.heads, hd, g.eps);
      emit_rms(e, g.k, L.kn, g.k, m * g.kv_heads, hd, g.eps);
      // interleaved rope on local layers
      if (L.is_local) {
        e.pipeline("mittens::muse_rope_qk");
        e.out(g.q, 0);
        e.in(positions, 1);
        e.bytes(hd, 2);
        e.bytes(g.heads, 3);
        e.bytes(g.theta, 4);
        e.dispatch(m * g.heads, 1, 1, 32, 1, 1);
        e.pipeline("mittens::muse_rope_qk");
        e.out(g.k, 0);
        e.in(positions, 1);
        e.bytes(hd, 2);
        e.bytes(g.kv_heads, 3);
        e.bytes(g.theta, 4);
        e.dispatch(m * g.kv_heads, 1, 1, 32, 1, 1);
      }
      // write K/V into the paged cache
      {
        const long half = L.kv_cache.numel() / 2;
        e.pipeline("mittens::muse_kv_store");
        e.out(L.kv_cache, 0);
        e.in(g.k, 1);
        e.in(g.v, 2);
        e.in(slots, 3);
        e.bytes(static_cast<int>(L.kv_cache.size(2)), 4);
        e.bytes(g.kv_heads, 5);
        e.bytes(hd, 6);
        e.bytes(half, 7);
        e.bytes(m, 8);
        const int total4 = m * g.kv_heads * hd / 4;
        e.dispatch((total4 + 255) / 256, 1, 1, 256, 1, 1);
      }
      // paged attention over the cache. Global layers at length use the
      // multi-query kernel: one shared K/V pass for the m rows (3.9x per
      // layer at 9.9k ctx) with the per-row causal boundary in-kernel.
      constexpr int kMqPartition = 512;
      const int mq_parts = (ctx_len + kMqPartition - 1) / kMqPartition;
      // verify blocks (rows_are_one_request): the MQ kernel shares one
      // request's KV pass across the m rows. Decode batches (independent
      // requests, 1 query each): the partition/reduce pair splits each
      // row's long scan across chunk-parallel threadgroups instead.
      const bool use_mq = rows_are_one_request && !L.is_local && m > 1 &&
                          ctx_len > 1024 && mq_parts <= 64;
      const bool use_part = !rows_are_one_request && !L.is_local &&
                            ctx_len > 2048 && mq_parts <= 64;
      if (use_part) {
        tk::launch_paged_attention_partition(
            e, g.q, L.kv_cache.select(0, 0), L.kv_cache.select(0, 1), bt, sl,
            g.mq_tmp, g.mq_mlog, g.mq_esum, m, g.heads, g.kv_heads, hd,
            static_cast<int>(L.kv_cache.size(2)),
            static_cast<int>(bt.stride(0)), g.scale, mq_parts, kMqPartition,
            /*window=*/0, /*softcap=*/0.0f, "bfloat16");
        tk::launch_paged_attention_reduce(e, g.mq_tmp, g.mq_mlog, g.mq_esum,
                                          g.attn_out, m, g.heads, hd, mq_parts,
                                          /*sinks=*/g.q,
                                          /*has_sink=*/0, "bfloat16");
      } else if (use_mq) {
        tk::launch_paged_attention_verify(
            e, g.q, L.kv_cache.select(0, 0), L.kv_cache.select(0, 1), bt,
            sl.narrow(0, m - 1, 1), g.mq_tmp, g.mq_mlog, g.mq_esum, m, g.heads,
            g.kv_heads, hd, static_cast<int>(L.kv_cache.size(2)),
            static_cast<int>(bt.stride(0)), g.scale, mq_parts, kMqPartition,
            /*window=*/0, "bfloat16");
        tk::launch_paged_attention_reduce(e, g.mq_tmp, g.mq_mlog, g.mq_esum,
                                          g.attn_out, m, g.heads, hd, mq_parts,
                                          /*sinks=*/g.q,
                                          /*has_sink=*/0, "bfloat16");
      } else {
        tk::launch_paged_attention(
            e, g.q, L.kv_cache.select(0, 0), L.kv_cache.select(0, 1), bt, sl,
            g.attn_out, m, g.heads, g.kv_heads, hd,
            static_cast<int>(L.kv_cache.size(2)),
            static_cast<int>(bt.stride(0)), g.scale,
            /*alibi=*/g.q, /*use_alibi=*/0, /*mask=*/g.q, /*use_mask=*/0,
            window, /*mask_heads=*/0, "bfloat16");
      }
      // gated output: attn_out *= sigmoid(gate_proj(h)); then o_proj
      emit_matvec(e, L.gate, g.h, g.gate_out, m);
      e.pipeline("mittens::muse_sigmoid_mul");
      e.out(g.attn_out, 0);
      e.in(g.gate_out, 1);
      e.bytes(n4_attn, 2);
      e.dispatch((n4_attn + 255) / 256, 1, 1, 256, 1, 1);
      emit_matvec(e, L.o, g.attn_out, g.o_out, m);
      // post-attention norm (tighter eps) + residual add
      emit_rms(e, g.o_out, L.post_attn, g.o_out, m, hidden, g.post_eps);
      e.pipeline("mittens::muse_add_inplace");
      e.out(x, 0);
      e.in(g.o_out, 1);
      e.bytes(n4_hidden, 2);
      e.dispatch((n4_hidden + 255) / 256, 1, 1, 256, 1, 1);
      // MLP half
      emit_rms(e, x, L.norm2, g.h, m, hidden, g.eps);
      emit_matvec(e, L.gate_up[0], g.h, g.g_out, m);
      emit_matvec(e, L.gate_up[1], g.h, g.u_out, m);
      e.pipeline("mittens::muse_silu_mul");
      e.out(g.mlp_mid, 0);
      e.in(g.g_out, 1);
      e.in(g.u_out, 2);
      e.bytes(n4_inter, 3);
      e.dispatch((n4_inter + 255) / 256, 1, 1, 256, 1, 1);
      emit_matvec(e, L.down, g.mlp_mid, g.o_out, m);
      emit_rms(e, g.o_out, L.post_ffn, g.o_out, m, hidden, g.post_eps);
      e.pipeline("mittens::muse_add_inplace");
      e.out(x, 0);
      e.in(g.o_out, 1);
      e.bytes(n4_hidden, 2);
      e.dispatch((n4_hidden + 255) / 256, 1, 1, 256, 1, 1);
    }
  });
}

// ---- fused DFlash drafter step (gated; VLLM_MUSE_FUSED_DRAFTER) ------------
// Same emit style as muse_step, specialized to the drafter: 5 layers, no
// attention gate, standard pre-norm residuals (no sandwich norms), NeoX
// rope, all-SWA attention that is BIDIRECTIONAL within the query block
// (the caller passes per-row seq_lens of base + m for every row).
namespace dflash_step {

struct Layer {
  std::array<muse_step::Proj, 3> qkv;
  muse_step::Proj o;
  std::array<muse_step::Proj, 2> gate_up;
  muse_step::Proj down;
  at::Tensor norm1, qn, kn, norm2;
  at::Tensor kv_cache;
};

struct State {
  int hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0;
  int window = 0, max_rows = 0;
  float theta = 0.f, eps = 0.f, scale = 0.f;
  bool ready = false;
  std::vector<Layer> layers;
  at::Tensor h, mlp_h, q, k, v, attn_out, o_out, g_out, u_out, mlp_mid;
};

State d;

}  // namespace dflash_step

void dflash_step_init(int64_t num_layers, int64_t hidden, int64_t heads,
                      int64_t kv_heads, int64_t head_dim, int64_t inter,
                      int64_t window, double theta, double eps,
                      int64_t max_rows, const at::Tensor& ref) {
  using namespace dflash_step;
  d = State{};
  d.hidden = static_cast<int>(hidden);
  d.heads = static_cast<int>(heads);
  d.kv_heads = static_cast<int>(kv_heads);
  d.head_dim = static_cast<int>(head_dim);
  d.inter = static_cast<int>(inter);
  d.window = static_cast<int>(window);
  d.max_rows = static_cast<int>(max_rows);
  d.theta = static_cast<float>(theta);
  d.eps = static_cast<float>(eps);
  d.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  d.layers.resize(static_cast<size_t>(num_layers));
  const auto opt = ref.options();
  const int64_t m = max_rows;
  d.h = at::empty({m, hidden}, opt);
  d.mlp_h = at::empty({m, hidden}, opt);
  d.q = at::empty({m, heads * head_dim}, opt);
  d.k = at::empty({m, kv_heads * head_dim}, opt);
  d.v = at::empty({m, kv_heads * head_dim}, opt);
  d.attn_out = at::empty({m, heads * head_dim}, opt);
  d.o_out = at::empty({m, hidden}, opt);
  d.g_out = at::empty({m, inter}, opt);
  d.u_out = at::empty({m, inter}, opt);
  d.mlp_mid = at::empty({m, inter}, opt);
}

void dflash_step_layer(int64_t idx, const std::vector<at::Tensor>& qkv_w,
                       const std::vector<int64_t>& qkv_t, const at::Tensor& o_w,
                       int64_t o_t, const std::vector<at::Tensor>& gu_w,
                       const std::vector<int64_t>& gu_t,
                       const at::Tensor& down_w, int64_t down_t,
                       const at::Tensor& norm1, const at::Tensor& qn,
                       const at::Tensor& kn, const at::Tensor& norm2,
                       const at::Tensor& kv_cache) {
  using namespace dflash_step;
  TORCH_CHECK(idx >= 0 && idx < (int64_t)d.layers.size(), "bad layer idx");
  TORCH_CHECK(qkv_w.size() == 3 && qkv_t.size() == 3, "qkv wants 3 shards");
  TORCH_CHECK(gu_w.size() == 2 && gu_t.size() == 2, "gate_up wants 2 shards");
  Layer& L = d.layers[static_cast<size_t>(idx)];
  const int hd = d.head_dim;
  L.qkv = {{{qkv_w[0], (int)qkv_t[0], d.heads * hd},
            {qkv_w[1], (int)qkv_t[1], d.kv_heads * hd},
            {qkv_w[2], (int)qkv_t[2], d.kv_heads * hd}}};
  L.o = {o_w, (int)o_t, d.hidden};
  L.gate_up = {
      {{gu_w[0], (int)gu_t[0], d.inter}, {gu_w[1], (int)gu_t[1], d.inter}}};
  L.down = {down_w, (int)down_t, d.hidden};
  L.norm1 = norm1;
  L.qn = qn;
  L.kn = kn;
  L.norm2 = norm2;
  L.kv_cache = kv_cache;
  if (idx == (int64_t)d.layers.size() - 1) {
    d.ready = true;
  }
}

void dflash_step_run(const at::Tensor& x, const at::Tensor& positions,
                     const at::Tensor& bt, const at::Tensor& sl,
                     const at::Tensor& slots) {
  using namespace dflash_step;
  using muse_step::emit_matvec;
  using muse_step::emit_rms;
  TORCH_CHECK(d.ready, "dflash_step not initialized");
  const int m = static_cast<int>(x.size(0));
  TORCH_CHECK(m >= 1 && m <= d.max_rows, "row count out of range");
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16,
              "x must be contiguous bf16");
  const int hidden = d.hidden;
  const int hd = d.head_dim;
  const int n4_hidden = m * hidden / 4;
  const int n4_inter = m * d.inter / 4;

  encode([&](TorchEncoder& e) {
    for (size_t li = 0; li < d.layers.size(); ++li) {
      const Layer& L = d.layers[li];
      emit_rms(e, x, L.norm1, d.h, m, hidden, d.eps);
      emit_matvec(e, L.qkv[0], d.h, d.q, m);
      emit_matvec(e, L.qkv[1], d.h, d.k, m);
      emit_matvec(e, L.qkv[2], d.h, d.v, m);
      emit_rms(e, d.q, L.qn, d.q, m * d.heads, hd, d.eps);
      emit_rms(e, d.k, L.kn, d.k, m * d.kv_heads, hd, d.eps);
      e.pipeline("mittens::muse_rope_qk_neox");
      e.out(d.q, 0);
      e.in(positions, 1);
      e.bytes(hd, 2);
      e.bytes(d.heads, 3);
      e.bytes(d.theta, 4);
      e.dispatch(m * d.heads, 1, 1, 32, 1, 1);
      e.pipeline("mittens::muse_rope_qk_neox");
      e.out(d.k, 0);
      e.in(positions, 1);
      e.bytes(hd, 2);
      e.bytes(d.kv_heads, 3);
      e.bytes(d.theta, 4);
      e.dispatch(m * d.kv_heads, 1, 1, 32, 1, 1);
      {
        const long half = L.kv_cache.numel() / 2;
        e.pipeline("mittens::muse_kv_store");
        e.out(L.kv_cache, 0);
        e.in(d.k, 1);
        e.in(d.v, 2);
        e.in(slots, 3);
        e.bytes(static_cast<int>(L.kv_cache.size(2)), 4);
        e.bytes(d.kv_heads, 5);
        e.bytes(hd, 6);
        e.bytes(half, 7);
        e.bytes(m, 8);
        const int total4 = m * d.kv_heads * hd / 4;
        e.dispatch((total4 + 255) / 256, 1, 1, 256, 1, 1);
      }
      tk::launch_paged_attention(
          e, d.q, L.kv_cache.select(0, 0), L.kv_cache.select(0, 1), bt, sl,
          d.attn_out, m, d.heads, d.kv_heads, hd,
          static_cast<int>(L.kv_cache.size(2)), static_cast<int>(bt.stride(0)),
          d.scale,
          /*alibi=*/d.q, /*use_alibi=*/0, /*mask=*/d.q, /*use_mask=*/0,
          d.window, /*mask_heads=*/0, "bfloat16");
      emit_matvec(e, L.o, d.attn_out, d.o_out, m);
      e.pipeline("mittens::muse_add_inplace");
      e.out(x, 0);
      e.in(d.o_out, 1);
      e.bytes(n4_hidden, 2);
      e.dispatch((n4_hidden + 255) / 256, 1, 1, 256, 1, 1);
      emit_rms(e, x, L.norm2, d.mlp_h, m, hidden, d.eps);
      emit_matvec(e, L.gate_up[0], d.mlp_h, d.g_out, m);
      emit_matvec(e, L.gate_up[1], d.mlp_h, d.u_out, m);
      e.pipeline("mittens::muse_silu_mul");
      e.out(d.mlp_mid, 0);
      e.in(d.g_out, 1);
      e.in(d.u_out, 2);
      e.bytes(n4_inter, 3);
      e.dispatch((n4_inter + 255) / 256, 1, 1, 256, 1, 1);
      emit_matvec(e, L.down, d.mlp_mid, d.o_out, m);
      e.pipeline("mittens::muse_add_inplace");
      e.out(x, 0);
      e.in(d.o_out, 1);
      e.bytes(n4_hidden, 2);
      e.dispatch((n4_hidden + 255) / 256, 1, 1, 256, 1, 1);
    }
  });
}

// Fused greedy draft sampling: one command buffer runs the shared lm_head
// GEMM (weight-streaming sm route) and a row-wise argmax. Softcap and
// logit scale are argmax-invariant and skipped; greedy only.
at::Tensor dflash_sample_greedy(const at::Tensor& hidden,
                                const at::Tensor& lm_w, int64_t lm_type,
                                int64_t vocab_rows) {
  check_mps(hidden, "hidden");
  check_mps(lm_w, "lm_w");
  const int m = static_cast<int>(hidden.size(0));
  const int K = static_cast<int>(hidden.size(1));
  const int N = static_cast<int>(vocab_rows);
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 && hidden.is_contiguous(),
              "quixicore(metal): dflash_sample_greedy wants contiguous bf16");
  TORCH_CHECK(m >= 9 && m <= 17 && N % 64 == 0 && K % 64 == 0,
              "quixicore(metal): dflash_sample_greedy m 9..17, N%64, K%64");
  const std::string fmt = ggml_type_to_format(lm_type);
  auto opt = hidden.options();
  auto xp = at::empty({K, 32}, opt.dtype(at::kHalf));
  auto partials = at::empty({4, N, 32}, opt.dtype(at::kFloat));
  auto logits = at::empty({m, N}, opt);
  auto tokens = at::empty({m}, opt.dtype(at::kLong));
  const int variant = 16;  // wide-N 8-warp tensor kernel
  encode([&](TorchEncoder& e) {
    tk::launch_muse_xpose32(e, xp, hidden, K, m);
    tk::launch_qgemm_sm(e, partials, lm_w, xp, N, K, variant, fmt);
    tk::launch_qgemm_sm_reduce_rm(e, logits, partials, N, m);
    e.pipeline("mittens::muse_argmax");
    e.out(tokens, 0);
    e.in(logits, 1);
    e.bytes(N, 2);
    e.dispatch(m, 1, 1, 256, 1, 1);
  });
  return tokens;
}

void muse_step_run(const at::Tensor& x, const at::Tensor& positions,
                   const at::Tensor& bt_local, const at::Tensor& sl_local,
                   const at::Tensor& slot_local, const at::Tensor& bt_full,
                   const at::Tensor& sl_full, const at::Tensor& slot_full,
                   int64_t ctx_len) {
  muse_step_run_impl(x, positions, bt_local, sl_local, slot_local, bt_full,
                     sl_full, slot_full, nullptr, {}, static_cast<int>(ctx_len),
                     /*rows_are_one_request=*/false);
}

// Verify-step variant: also snapshots the residual stream entering each
// layer listed in aux_layers (ascending) into aux_out[j] for the DFlash
// drafter. aux_out is (len(aux_layers), rows, hidden) bf16.
void muse_step_run_aux(const at::Tensor& x, const at::Tensor& positions,
                       const at::Tensor& bt_local, const at::Tensor& sl_local,
                       const at::Tensor& slot_local, const at::Tensor& bt_full,
                       const at::Tensor& sl_full, const at::Tensor& slot_full,
                       const at::Tensor& aux_out,
                       const std::vector<int64_t>& aux_layers,
                       int64_t ctx_len) {
  using namespace muse_step;
  check_mps(aux_out, "aux_out");
  TORCH_CHECK(aux_out.scalar_type() == at::kBFloat16 && aux_out.dim() == 3 &&
                  aux_out.size(0) == (int64_t)aux_layers.size() &&
                  aux_out.size(1) == x.size(0) && aux_out.size(2) == g.hidden,
              "aux_out must be (n_aux, rows, hidden) bf16");
  for (size_t j = 1; j < aux_layers.size(); ++j) {
    TORCH_CHECK(aux_layers[j] > aux_layers[j - 1] && aux_layers[j] >= 0 &&
                    aux_layers[j] < (int64_t)g.layers.size(),
                "aux_layers must be ascending in-range layer indices");
  }
  muse_step_run_impl(x, positions, bt_local, sl_local, slot_local, bt_full,
                     sl_full, slot_full, &aux_out, aux_layers,
                     static_cast<int>(ctx_len),
                     /*rows_are_one_request=*/true);
}

// Standalone rm-variant probe: row-major bf16 in/out, fresh partials per
// call (mirrors how the fused step would behave with per-op buffers).
at::Tensor ggml_mul_mat_sm_rm_pre(const at::Tensor& w, const at::Tensor& x,
                                  int64_t quant_type, int64_t row) {
  check_mps(w, "w");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(-1));
  const int M = static_cast<int>(x.size(0));
  const std::string fmt = ggml_type_to_format(quant_type);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && M <= 32,
              "sm_rm_pre wants contiguous (M<=32, K) bf16");
  auto partials = at::empty({4, N, 32}, x.options().dtype(at::kFloat));
  auto out = at::empty({M, N}, x.options());
  encode([&](TorchEncoder& e) {
    tk::launch_qgemm_sm_rm(e, partials, w, x, N, K, M, fmt);
    tk::launch_qgemm_sm_reduce_rm(e, out, partials, N, M);
  });
  return out;
}

// Bench probe: qgemm_sm structure with raw half weights (no dequant ALU).
at::Tensor ggml_mul_mat_sm_f16probe(const at::Tensor& w, const at::Tensor& x,
                                    int64_t row) {
  check_mps(w, "w");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(0));
  auto partials = at::empty({4, N, 32}, x.options().dtype(at::kFloat));
  auto out = at::empty({N, 32}, x.options());
  encode([&](TorchEncoder& e) {
    e.pipeline("qgemm_sm_r8sk4_f16_raw");
    e.out(partials, 0);
    e.in(w, 1);
    e.in(x, 2);
    e.bytes(N, 3);
    e.bytes(K, 4);
    e.dispatch(1, N / 16, 4, 64, 1, 1);
    tk::launch_qgemm_sm_reduce(e, out, partials, N, 4);
  });
  return out;
}

// ---- RMSNorm -------------------------------------------------------------
//
// One dispatch per norm call. The eager torch-native RMSNorm decomposes into
// ~five MPS ops; with six norms per Muse-Glimmer layer that dominates the
// non-matvec dispatch budget of a decode step.

// bf16 contiguous fast path (Muse-Glimmer): fixed-D single-dispatch kernels.
// The bound rms_norm below routes here when the preconditions hold.
at::Tensor rms_norm_bf16_contig(const at::Tensor& x, const at::Tensor& weight,
                                double eps) {
  check_mps(x, "x");
  check_mps(weight, "weight");
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
      "rms_norm(metal) is bf16-only");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "rms_norm(metal) wants a contiguous [rows, D] input");
  const int D = static_cast<int>(x.size(1));
  TORCH_CHECK(weight.numel() == D, "weight/D mismatch");
  TORCH_CHECK(D % 4 == 0, "rms_norm(metal) needs D % 4 == 0");
  const auto M = static_cast<uint32_t>(x.size(0));
  at::Tensor out = at::empty_like(x);
  const bool fixed = D == 256 || D == 512 || D == 768 || D == 1024;
  encode("qc_rms_norm", [&](TorchEncoder& e) {
    if (fixed) {
      tk::launch_rms_norm(e, x, weight, out, M, D, static_cast<float>(eps));
    } else {
      tk::launch_rms_norm_dyn(e, x, weight, out, M, D, static_cast<float>(eps));
    }
  });
  return out;
}

// ---- DeepSeek-V4 packed sparse MLA ---------------------------------------

// Step-constant marshalling memo: the bf16 cos/sin halves of a RoPE
// cos_sin_cache, keyed by the cache's data pointer. The cache is a fixed
// model weight, so the cast+split+contiguous chain (6 eager kernels) runs
// once per server lifetime instead of once per layer per step. EngineCore
// encodes single-threaded; no locking needed.
std::pair<at::Tensor, at::Tensor> cos_sin_bf16_halves(
    const at::Tensor& cos_sin_cache) {
  static std::unordered_map<const void*, std::pair<at::Tensor, at::Tensor>>
      memo;
  const void* key = cos_sin_cache.data_ptr();
  auto it = memo.find(key);
  if (it != memo.end()) return it->second;
  auto cs = cos_sin_cache.to(at::kBFloat16).contiguous();
  TORCH_CHECK(cs.dim() == 2 && cs.size(1) == 64,
              "cos_sin_cache must be [positions, 64], got ", cs.sizes());
  auto pr = std::make_pair(cs.slice(1, 0, 32).contiguous(),
                           cs.slice(1, 32, 64).contiguous());
  memo.emplace(key, pr);
  return pr;
}

void deepseek_v4_save_partial_states(
    const at::Tensor& kv_in, const at::Tensor& score_in,
    const at::Tensor& ape_in, const at::Tensor& positions_in,
    const at::Tensor& state_cache, const at::Tensor& slot_mapping_in,
    int64_t block_size, int64_t state_width, int64_t compress_ratio) {
  check_mps_strided(kv_in, "kv");
  check_mps_strided(score_in, "score");
  check_mps(ape_in, "ape");
  check_mps_strided(state_cache, "state_cache");
  // fp16 kv/score (the raw kv_score GEMM output under fp16 serving) round to
  // bf16 in-register — bit-identical to the eager .float()+.to(bfloat16)
  // chain this replaces.
  const bool half_input = kv_in.scalar_type() == at::kHalf;
  TORCH_CHECK((half_input ? score_in.scalar_type() == at::kHalf
                          : (kv_in.scalar_type() == at::kBFloat16 &&
                             score_in.scalar_type() == at::kBFloat16)) &&
                  ape_in.scalar_type() == at::kBFloat16 &&
                  state_cache.scalar_type() == at::kFloat,
              "DeepSeek-V4 partial-state kv/score must both be bfloat16 or "
              "both float16, ape bfloat16, state cache float32");
  TORCH_CHECK(kv_in.dim() == 2 && score_in.sizes() == kv_in.sizes(),
              "kv and score must be [tokens, head_size]");
  TORCH_CHECK(state_cache.dim() == 3 && state_cache.stride(2) == 1,
              "state_cache must have contiguous state rows");
  const int tokens = static_cast<int>(slot_mapping_in.size(0));
  const int head_size = static_cast<int>(kv_in.size(1));
  // Row-strided views (e.g. halves of the fused kv_score GEMM output) bind
  // directly — the kernel takes the row stride, so no eager copies. Only a
  // non-unit column stride forces the packed fallback.
  auto kv = kv_in.narrow(0, 0, tokens);
  auto score = score_in.narrow(0, 0, tokens);
  if (kv.stride(1) != 1 || score.stride(1) != 1 ||
      kv.stride(0) != score.stride(0)) {
    kv = kv.contiguous();
    score = score.contiguous();
  }
  const int in_stride = static_cast<int>(kv.stride(0));
  auto ape = ape_in.to(at::kBFloat16).contiguous();
  auto positions = positions_in.narrow(0, 0, tokens).to(at::kInt).contiguous();
  auto slots = slot_mapping_in.to(at::kLong).contiguous();
  encode("qc_deepseek_v4_save_partial_states", [&](TorchEncoder& e) {
    tk::launch_dsv4_save_partial_states(
        e, kv, score, ape, positions, state_cache, slots, tokens, head_size,
        static_cast<int>(block_size), static_cast<int>(state_cache.stride(0)),
        static_cast<int>(state_cache.stride(1)), static_cast<int>(state_width),
        static_cast<int>(compress_ratio), in_stride, half_input);
  });
}

at::Tensor deepseek_v4_qnorm_rope_kv_insert(
    const at::Tensor& q_in, const at::Tensor& kv_in, const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping_in, const at::Tensor& positions_in,
    const at::Tensor& cos_sin_cache, double eps, int64_t block_size) {
  check_mps_strided(q_in, "q");
  check_mps_strided(kv_in, "kv");
  check_mps_strided(kv_cache, "kv_cache");
  // fp16 serving passes q/kv straight from the projections; the half-input
  // kernel variants round each element to bf16 in-register (RNE), which is
  // bit-identical to the eager .to(bfloat16) casts they replace.
  const bool q_half = q_in.scalar_type() == at::kHalf;
  const bool kv_half = kv_in.scalar_type() == at::kHalf;
  TORCH_CHECK(q_half || q_in.scalar_type() == at::kBFloat16,
              "DeepSeek-V4 Metal q must be bfloat16 or float16");
  TORCH_CHECK(kv_half || kv_in.scalar_type() == at::kBFloat16,
              "DeepSeek-V4 Metal kv must be bfloat16 or float16");
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
  auto [cos, sin] = cos_sin_bf16_halves(cos_sin_cache);
  auto q = q_in.contiguous();
  // The half kv kernel reads a row-strided view (unit inner stride), so the
  // fused-projection slice binds with no eager copy at all.
  const bool kv_strided_ok = kv_half && kv_in.stride(1) == 1 &&
                             kv_in.stride(0) >= kv_in.size(1) &&
                             kv_in.stride(0) <= std::numeric_limits<int>::max();
  auto kv = kv_strided_ok ? kv_in : kv_in.contiguous();
  auto out = q_half ? ring_out("qnorm_out", q.sizes(),
                               q.options().dtype(at::kBFloat16))
                    : ring_out_like("qnorm_out_l", q);

  encode("qc_deepseek_v4_qnorm_rope_kv_insert", [&](TorchEncoder& e) {
    tk::launch_mla_q_norm_rope(
        e, q, cos, sin, positions, q, out, tokens * heads, heads,
        /*nope_dim=*/448, /*rope_dim=*/64, /*norm_mode=*/1,
        static_cast<float>(eps), /*head_dim=*/512, /*half_input=*/q_half);
    if (kv_half) {
      tk::launch_mla_kv_insert_fp8_packed_half(
          e, kv, cos, sin, positions, slots, kv_cache, tokens,
          static_cast<int>(block_size), static_cast<int>(kv_cache.stride(0)),
          static_cast<int>(kv.stride(0)));
    } else {
      tk::launch_mla_kv_insert_fp8_packed(
          e, kv, cos, sin, positions, slots, kv_cache, tokens,
          static_cast<int>(block_size), static_cast<int>(kv_cache.stride(0)));
    }
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
  auto [cos, sin] = cos_sin_bf16_halves(cos_sin_cache);
  const int tokens = static_cast<int>(kv.size(0));
  encode("qc_deepseek_v4_kv_insert", [&](TorchEncoder& e) {
    tk::launch_mla_kv_insert_fp8_packed(
        e, kv, cos, sin, positions, slots, kv_cache, tokens,
        static_cast<int>(block_size), static_cast<int>(kv_cache.stride(0)));
  });
}

void deepseek_v4_indexer_kv_insert(const at::Tensor& kv_in,
                                   const at::Tensor& kv_cache,
                                   const at::Tensor& slot_mapping_in,
                                   const at::Tensor& positions_in,
                                   const at::Tensor& cos_sin_cache,
                                   int64_t block_size) {
  check_mps(kv_in, "kv");
  check_mps_strided(kv_cache, "kv_cache");
  TORCH_CHECK(kv_in.scalar_type() == at::kBFloat16 && kv_in.size(-1) == 128,
              "DeepSeek-V4 Metal indexer kv must be [..., 128] bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte && kv_cache.size(-1) == 132,
              "DeepSeek-V4 indexer cache must have 132-byte uint8 slots");
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.stride(2) == 1 &&
                  kv_cache.stride(1) == 132,
              "DeepSeek-V4 indexer cache must have contiguous token slots");
  auto kv = kv_in.reshape({-1, 128}).contiguous();
  auto slots = slot_mapping_in.to(at::kLong).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto [cos, sin] = cos_sin_bf16_halves(cos_sin_cache);
  const int tokens = static_cast<int>(kv.size(0));
  encode("qc_deepseek_v4_indexer_kv_insert", [&](TorchEncoder& e) {
    tk::launch_dsv4_indexer_kv_insert(
        e, kv, cos, sin, positions, slots, kv_cache, tokens,
        static_cast<int>(block_size), static_cast<int>(kv_cache.stride(0)));
  });
}

void qc_swiglu(const at::Tensor& x, at::Tensor& y,
               const std::optional<double>& clamp_limit, bool oai_form,
               double alpha, double beta) {
  check_mps(x, "x");
  check_mps(y, "y");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "swiglu input must be contiguous [tokens, 2*d]");
  TORCH_CHECK(y.dim() == 2 && y.is_contiguous() &&
                  y.scalar_type() == x.scalar_type() &&
                  y.size(0) == x.size(0) && y.size(1) * 2 == x.size(1),
              "swiglu output must be contiguous [tokens, d], same dtype");
  const int n_out = static_cast<int>(y.size(1));
  const long total_l = y.numel();
  TORCH_CHECK(total_l <= INT32_MAX, "swiglu output too large");
  const int total = static_cast<int>(total_l);
  if (total == 0) {
    return;
  }
  encode("qc_swiglu", [&](TorchEncoder& e) {
    tk::launch_qc_swiglu(
        e, x, y, n_out, clamp_limit.has_value() ? 1 : 0,
        clamp_limit.has_value() ? static_cast<float>(*clamp_limit) : 0.0f,
        oai_form ? 1 : 0, static_cast<float>(alpha), static_cast<float>(beta),
        total, activation_type_name(x));
  });
}

void moe_weighted_sum(const at::Tensor& x, const at::Tensor& w, at::Tensor& y) {
  check_mps(x, "x");
  check_mps(w, "w");
  check_mps(y, "y");
  TORCH_CHECK(x.dim() == 3 && x.is_contiguous(),
              "x must be contiguous [tokens, topk, dim]");
  TORCH_CHECK(
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
          y.scalar_type() == x.scalar_type(),
      "x/y must both be fp16 or bf16");
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() &&
                  w.dim() == 2 && w.size(0) == x.size(0) &&
                  w.size(1) == x.size(1),
              "w must be contiguous fp32 [tokens, topk]");
  const int tokens = static_cast<int>(x.size(0));
  const int topk = static_cast<int>(x.size(1));
  const int dim = static_cast<int>(x.size(2));
  TORCH_CHECK(topk <= 8, "moe_weighted_sum supports topk <= 8");
  TORCH_CHECK(y.dim() == 2 && y.is_contiguous() && y.size(0) == tokens &&
                  y.size(1) == dim,
              "y must be contiguous [tokens, dim]");
  if (tokens == 0) {
    return;
  }
  const std::string tname =
      x.scalar_type() == at::kHalf ? "float16" : "bfloat16";
  encode("qc_moe_weighted_sum", [&](TorchEncoder& e) {
    tk::launch_qc_moe_weighted_sum(e, x, w, y, tokens, topk, dim, tname);
  });
}

// gating must already hold softplus(logits) — the bitwise-safe route; see
// the kernel header for why softplus cannot move in-kernel.
void dsv4_router_topk(const at::Tensor& gating, at::Tensor& out_w,
                      at::Tensor& out_ids, bool renormalize, double scaling,
                      const std::optional<at::Tensor>& bias,
                      const std::optional<at::Tensor>& hash_table,
                      const std::optional<at::Tensor>& input_ids) {
  check_mps(gating, "gating");
  TORCH_CHECK(gating.scalar_type() == at::kFloat && gating.dim() == 2 &&
                  gating.is_contiguous(),
              "gating must be contiguous fp32 [tokens, experts]");
  const int tokens = static_cast<int>(gating.size(0));
  const int experts = static_cast<int>(gating.size(1));
  TORCH_CHECK(experts <= 1024, "router kernel supports at most 1024 experts");
  TORCH_CHECK(out_w.scalar_type() == at::kFloat &&
                  out_ids.scalar_type() == at::kInt && out_w.dim() == 2 &&
                  out_ids.dim() == 2 && out_w.is_contiguous() &&
                  out_ids.is_contiguous() && out_w.size(0) == tokens &&
                  out_ids.size(0) == tokens,
              "outputs must be contiguous [tokens, topk] fp32 / int32");
  const int topk = static_cast<int>(out_w.size(1));
  TORCH_CHECK(topk <= 8 && out_ids.size(1) == topk,
              "router kernel supports topk <= 8");
  const bool has_bias = bias.has_value();
  if (has_bias) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
                    bias->numel() == experts,
                "bias must be contiguous fp32 [experts]");
  }
  const bool has_hash = hash_table.has_value();
  if (has_hash) {
    TORCH_CHECK(input_ids.has_value(), "hash routing requires input_ids");
    TORCH_CHECK(hash_table->scalar_type() == at::kInt &&
                    hash_table->is_contiguous() && hash_table->dim() == 2 &&
                    hash_table->size(1) == topk,
                "hash_table must be contiguous int32 [vocab, topk]");
    TORCH_CHECK(input_ids->scalar_type() == at::kInt &&
                    input_ids->is_contiguous() && input_ids->numel() == tokens,
                "input_ids must be contiguous int32 [tokens]");
  }
  if (tokens == 0) {
    return;
  }
  // Metal requires every declared buffer bound; absent optionals bind an
  // arbitrary live tensor the kernel never reads (has_bias/has_hash gate).
  const at::Tensor bias_t = has_bias ? *bias : gating;
  const at::Tensor table_t = has_hash ? *hash_table : out_ids;
  const at::Tensor ids_t = has_hash ? *input_ids : out_ids;
  encode("qc_dsv4_router_topk", [&](TorchEncoder& e) {
    tk::launch_dsv4_router_topk(
        e, gating, bias_t, table_t, ids_t, out_w, out_ids, tokens, experts,
        topk, has_bias ? 1 : 0, has_hash ? 1 : 0, renormalize ? 1 : 0,
        static_cast<float>(scaling));
  });
}

void dsv4_indexer_compress_insert(
    const at::Tensor& state_cache, const at::Tensor& positions_in,
    const at::Tensor& state_slots_in, const at::Tensor& token_to_req_in,
    const at::Tensor& block_table, const at::Tensor& kv_slots_in,
    const at::Tensor& rms_w_in, const at::Tensor& cos_in,
    const at::Tensor& sin_in, const at::Tensor& kv_cache,
    int64_t state_block_size, int64_t state_width, int64_t compress_ratio,
    double eps) {
  check_mps_strided(state_cache, "state_cache");
  check_mps_strided(kv_cache, "kv_cache");
  TORCH_CHECK(state_cache.scalar_type() == at::kFloat &&
                  state_cache.dim() == 3 && state_cache.stride(2) == 1,
              "state_cache must be fp32 [blocks, bs, width] with contiguous "
              "rows");
  TORCH_CHECK(
      compress_ratio == 4 && state_width == 256 && state_cache.size(2) == 512,
      "indexer compress kernel expects the ratio-4 overlap layout "
      "(512-wide fp32 state rows)");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte && kv_cache.dim() == 3 &&
                  kv_cache.size(-1) == 132 && kv_cache.stride(2) == 1 &&
                  kv_cache.stride(1) == 132,
              "DeepSeek-V4 indexer cache must have contiguous 132-byte slots");
  TORCH_CHECK(cos_in.scalar_type() == at::kBFloat16 &&
                  sin_in.scalar_type() == at::kBFloat16 && cos_in.dim() == 2 &&
                  cos_in.size(1) == 32 && cos_in.is_contiguous() &&
                  sin_in.is_contiguous(),
              "cos/sin must be contiguous bf16 [positions, 32]");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 &&
                  block_table.stride(1) == 1,
              "block_table must be int32 [reqs, cols] with contiguous rows");
  auto positions = positions_in.to(at::kInt).contiguous();
  auto sslots = state_slots_in.to(at::kLong).contiguous();
  auto t2r = token_to_req_in.to(at::kInt).contiguous();
  auto kvslots = kv_slots_in.to(at::kLong).contiguous();
  auto rms_w = rms_w_in.to(at::kFloat).contiguous();
  TORCH_CHECK(rms_w.numel() == 128, "rms weight must have 128 elements");
  const int tokens = static_cast<int>(positions.size(0));
  TORCH_CHECK(sslots.size(0) == tokens && t2r.size(0) == tokens &&
                  kvslots.size(0) == tokens,
              "positions/state_slots/token_to_req/kv_slots must agree");
  if (tokens == 0) {
    return;
  }
  encode("qc_dsv4_indexer_compress_insert", [&](TorchEncoder& e) {
    tk::launch_dsv4_indexer_compress_insert(
        e, state_cache, cos_in, sin_in, positions, sslots, t2r, block_table,
        rms_w, kvslots, kv_cache, tokens, static_cast<int>(state_block_size),
        static_cast<int>(state_cache.stride(0)),
        static_cast<int>(state_cache.stride(1)), static_cast<int>(state_width),
        static_cast<int>(compress_ratio),
        static_cast<int>(block_table.stride(0)),
        static_cast<int>(block_table.size(1)),
        static_cast<int>(kv_cache.size(1)),
        static_cast<int>(kv_cache.stride(0)), static_cast<float>(eps));
  });
}

// Fused head=512 compressor front (see dsv4_compress_front[_c128] in
// indexer.metal): returns the normed bf16 rows the native
// deepseek_v4_kv_insert consumes. Rows for non-boundary / slot<0 tokens
// are left uninitialized — the insert masks them via output_slots = -1.
// cr=4 overlap rows are [kv 2x512 | score 2x512] (width 1024); cr=128
// no-overlap rows are [kv 512 | score 512] (width 512).
at::Tensor dsv4_compress_front(const at::Tensor& state_cache,
                               const at::Tensor& positions_in,
                               const at::Tensor& state_slots_in,
                               const at::Tensor& token_to_req_in,
                               const at::Tensor& block_table,
                               const at::Tensor& rms_w_in, int64_t num_tokens,
                               int64_t state_block_size, int64_t state_width,
                               int64_t compress_ratio, double eps) {
  check_mps_strided(state_cache, "state_cache");
  check_mps(block_table, "block_table");
  TORCH_CHECK(state_cache.scalar_type() == at::kFloat &&
                  state_cache.dim() == 3 && state_cache.stride(2) == 1,
              "state cache must be fp32 with contiguous rows");
  TORCH_CHECK(state_cache.size(2) == 2 * state_width &&
                  ((compress_ratio == 4 && state_width == 1024) ||
                   (compress_ratio == 128 && state_width == 512)),
              "dsv4_compress_front expects cr=4 [kv 2x512 | score 2x512] or "
              "cr=128 [kv 512 | score 512] rows, got cr=",
              compress_ratio, " width=", state_width,
              " row=", state_cache.size(2));
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2,
              "block_table must be int32 rank 2");
  const int tokens = static_cast<int>(num_tokens);
  auto positions = positions_in.narrow(0, 0, tokens).to(at::kInt).contiguous();
  auto sslots = state_slots_in.narrow(0, 0, tokens).to(at::kLong).contiguous();
  auto t2r = token_to_req_in.narrow(0, 0, tokens).to(at::kInt).contiguous();
  auto rms_w = rms_w_in.to(at::kFloat).contiguous();
  TORCH_CHECK(rms_w.numel() == 512, "rms weight must be 512-wide");
  auto out = ring_out("cfront_out", {tokens, 512},
                      state_cache.options().dtype(at::kBFloat16));
  encode("qc_dsv4_compress_front", [&](TorchEncoder& e) {
    tk::launch_dsv4_compress_front(
        e, state_cache, positions, sslots, t2r, block_table, rms_w, out, tokens,
        static_cast<int>(state_block_size),
        static_cast<int>(state_cache.stride(0)),
        static_cast<int>(state_cache.stride(1)), static_cast<int>(state_width),
        static_cast<int>(compress_ratio),
        static_cast<int>(block_table.stride(0)),
        static_cast<int>(block_table.size(1)), static_cast<float>(eps));
  });
  return out;
}

// Dense-causal prefill FA (see mla.metal): decode a slot list into a
// contiguous half [n, 512] scratch. Slots must be pre-padded with -1 to a
// multiple of 32 rows plus one spare block so the FA tile loads never run
// off the scratch.
at::Tensor deepseek_v4_prefill_dequant(const at::Tensor& cache,
                                       const at::Tensor& slots_in) {
  check_mps_strided(cache, "cache");
  check_mps(slots_in, "slots");
  TORCH_CHECK(cache.scalar_type() == at::kByte && cache.dim() == 3 &&
                  cache.size(-1) == 584 && cache.stride(2) == 1 &&
                  cache.stride(1) == 584,
              "cache must be [blocks, block_size, 584] uint8 contiguous");
  auto slots = slots_in.to(at::kInt).contiguous();
  const int n = static_cast<int>(slots.numel());
  TORCH_CHECK(n > 0, "empty slot list");
  auto out = at::empty(
      {n, 512}, at::TensorOptions().dtype(at::kHalf).device(cache.device()));
  encode("qc_mla_prefill_dequant", [&](TorchEncoder& e) {
    tk::launch_mla_prefill_dequant_slots(e, cache, slots, out, n,
                                         static_cast<int>(cache.size(1)),
                                         static_cast<long>(cache.stride(0)));
  });
  return out;
}

// Dense-causal prefill MMA FA over pre-decoded axes (ULP class vs the
// decode walk: P rounds to half, block-level reassociation).
at::Tensor deepseek_v4_prefill_fa(
    const at::Tensor& q_in, const at::Tensor& kc, const at::Tensor& ks,
    const at::Tensor& lens_c_in, const at::Tensor& lo_s_in,
    const at::Tensor& hi_s_in, const at::Tensor& sinks_in, double scale,
    const std::optional<at::Tensor>& out_opt = std::nullopt) {
  check_mps(q_in, "q");
  check_mps(kc, "kc");
  check_mps(ks, "ks");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16 && q_in.dim() == 3 &&
                  q_in.size(2) == 512,
              "q must be [tokens, heads, 512] bfloat16");
  TORCH_CHECK(kc.scalar_type() == at::kHalf && kc.dim() == 2 &&
                  kc.size(1) == 512 && kc.is_contiguous() &&
                  ks.scalar_type() == at::kHalf && ks.dim() == 2 &&
                  ks.size(1) == 512 && ks.is_contiguous(),
              "kc/ks must be contiguous half [n, 512]");
  // The FA tail loads complete 8x8 fragments; the dequant scratches must be
  // padded to 32-row multiples (metal.py _pad_slots adds the -1 padding).
  TORCH_CHECK(kc.size(0) % 32 == 0 && ks.size(0) % 32 == 0,
              "kc/ks rows must be padded to a multiple of 32, got ", kc.size(0),
              " and ", ks.size(0));
  const int T = static_cast<int>(q_in.size(0));
  const int heads = static_cast<int>(q_in.size(1));
  auto q = q_in.contiguous();
  auto lens_c = lens_c_in.to(at::kInt).contiguous();
  auto lo_s = lo_s_in.to(at::kInt).contiguous();
  auto hi_s = hi_s_in.to(at::kInt).contiguous();
  auto sinks = sinks_in.to(at::kFloat).contiguous();
  TORCH_CHECK(lens_c.numel() == T && lo_s.numel() == T && hi_s.numel() == T,
              "per-token tables must have T entries");
  bool half_out = false;
  at::Tensor out;
  if (out_opt.has_value()) {
    out = *out_opt;
    check_mps(out, "out");
    TORCH_CHECK(out.sizes() == q.sizes() && out.is_contiguous(),
                "out must match q's shape");
    half_out = out.scalar_type() == at::kHalf;
    TORCH_CHECK(half_out || out.scalar_type() == at::kBFloat16,
                "out must be float16 or bfloat16");
  } else {
    out = at::empty_like(q);
  }
  encode("qc_deepseek_v4_prefill_fa", [&](TorchEncoder& e) {
    tk::launch_mla_prefill_fa_mma(e, q, kc, ks, lens_c, lo_s, hi_s, sinks, out,
                                  T, heads, static_cast<int>(kc.size(0)),
                                  static_cast<int>(ks.size(0)),
                                  static_cast<float>(scale), half_out);
  });
  return out;
}

at::Tensor deepseek_v4_sparse_attention(
    const at::Tensor& q_in, const at::Tensor& compressed_cache,
    const at::Tensor& compressed_slots_in, const at::Tensor& compressed_lens_in,
    const at::Tensor& swa_cache, const at::Tensor& swa_slots_in,
    const at::Tensor& swa_lens_in, const at::Tensor& sinks_in, double scale,
    const std::optional<at::Tensor>& out_opt = std::nullopt) {
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
  // With a caller buffer the kernel writes attention output in place; the
  // fp16 (serving) variant rounds through bf16 first, bit-identical to the
  // bf16-allocate + .copy_() chain it replaces.
  bool half_out = false;
  at::Tensor out;
  if (out_opt.has_value()) {
    out = *out_opt;
    check_mps(out, "out");
    TORCH_CHECK(out.sizes() == q.sizes(),
                "out must match q's [tokens, heads, 512], got ", out.sizes());
    half_out = out.scalar_type() == at::kHalf;
    TORCH_CHECK(half_out || out.scalar_type() == at::kBFloat16,
                "out must be float16 or bfloat16");
  } else {
    out = at::empty_like(q);
  }
  // Split-K policy: the fused kernel walks the whole candidate list with one
  // simdgroup per (head, batch); the serial walk is latency-bound at decode
  // widths. Partitioning slices the walk across P grid layers, with the
  // existing paged-v2 reduce (sink applied there, exactly once) folding the
  // partials. ULP class: the LSE merge reassociates the softmax vs the fused
  // kernel's sequential order — VLLM_QC_MLA_SPLITK=0 restores the unsplit
  // walk as the reversion sentinel. The 768-simdgroup target keeps a 64-core
  // GPU's cores fed at decode grid sizes.
  static const int splitk_target = [] {
    const char* off = std::getenv("VLLM_QC_MLA_SPLITK");
    return (off != nullptr && std::string(off) == "0") ? 0 : 768;
  }();
  const int cw = static_cast<int>(compressed_slots.size(1));
  const int sw = static_cast<int>(swa_slots.size(1));
  // Prefill widths: the decode kernel re-decodes every 584-byte slot once
  // per head. The staged twin shares one fp32 decode across a 16-head group
  // and is bit-identical to the fused decode kernel (same per-candidate
  // order, lane mapping, simd_sum tree).
  if (batch >= 64 && heads % 16 == 0) {
    encode("qc_deepseek_v4_sparse_attention", [&](TorchEncoder& e) {
      tk::launch_mla_prefill_fp8_sparse_two_cache_packed(
          e, q, compressed_cache, compressed_slots, compressed_lens, swa_cache,
          swa_slots, swa_lens, sinks, out, batch, heads, cw, sw,
          static_cast<int>(compressed_cache.size(1)),
          static_cast<int>(compressed_cache.stride(0)),
          static_cast<int>(swa_cache.size(1)),
          static_cast<int>(swa_cache.stride(0)), static_cast<float>(scale),
          half_out);
    });
    return out;
  }
  const int total_width = cw + sw;
  int num_partitions = 1;
  if (splitk_target > 0 && total_width >= 2) {
    const int units = batch * heads;
    num_partitions = std::min((splitk_target + units - 1) / units, 16);
    num_partitions = std::min(num_partitions, total_width);
  }
  if (num_partitions > 1) {
    const int partition_size =
        (total_width + num_partitions - 1) / num_partitions;
    num_partitions = (total_width + partition_size - 1) / partition_size;
    auto fopts = q.options().dtype(at::kFloat);
    auto tmp = ring_out("mla2_tmp", {batch, heads, num_partitions, 512}, fopts);
    auto ml = ring_out("mla2_ml", {batch, heads, num_partitions}, fopts);
    auto es = ring_out("mla2_es", {batch, heads, num_partitions}, fopts);
    encode("qc_deepseek_v4_sparse_attention_splitk", [&](TorchEncoder& e) {
      tk::launch_mla_decode_fp8_sparse_two_cache_packed_partition(
          e, q, compressed_cache, compressed_slots, compressed_lens, swa_cache,
          swa_slots, swa_lens, tmp, ml, es, batch, heads, cw, sw,
          static_cast<int>(compressed_cache.size(1)),
          static_cast<int>(compressed_cache.stride(0)),
          static_cast<int>(swa_cache.size(1)),
          static_cast<int>(swa_cache.stride(0)), static_cast<float>(scale),
          num_partitions, partition_size);
    });
    encode("qc_deepseek_v4_sparse_attention_reduce", [&](TorchEncoder& e) {
      tk::launch_paged_attention_reduce(
          e, tmp, ml, es, out, batch, heads, /*head_size=*/512, num_partitions,
          sinks, /*has_sink=*/1, half_out ? "float16" : "bfloat16");
    });
    return out;
  }
  encode("qc_deepseek_v4_sparse_attention", [&](TorchEncoder& e) {
    tk::launch_mla_decode_fp8_sparse_two_cache_packed(
        e, q, compressed_cache, compressed_slots, compressed_lens, swa_cache,
        swa_slots, swa_lens, sinks, out, batch, heads,
        static_cast<int>(compressed_slots.size(1)),
        static_cast<int>(swa_slots.size(1)),
        static_cast<int>(compressed_cache.size(1)),
        static_cast<int>(compressed_cache.stride(0)),
        static_cast<int>(swa_cache.size(1)),
        static_cast<int>(swa_cache.stride(0)), static_cast<float>(scale),
        half_out);
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
  encode("qc_turboquant_encode_metal", [&](TorchEncoder& e) {
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
  encode("qc_turboquant_attention_metal", [&](TorchEncoder& e) {
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
    case 21:
      return "iq3_s";
    case 22:
      return "iq2_s";
    case 23:
      return "iq4_xs";
    case 29:
      return "iq1_m";
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

  at::Tensor out = ring_out("gemv_out", {x.size(0), N}, x.options());
  const std::string fmt = ggml_type_to_format(quant_type);
  const std::string type_name = activation_type_name(x);
  const int batch = static_cast<int>(x.size(0));

  // Verify/decode widths 2..8: one weight-stationary launch that reads each
  // weight block once for all rows. Formats limited to the instantiated mb
  // set; per-row results are bit-identical to the looped batch-1 kernels.
  const bool mb_ok =
      batch >= 2 && batch <= 8 &&
      (fmt == "q8_0" || fmt == "q2_K" || fmt == "iq2_xxs" || fmt == "q6_K") &&
      (type_name == "float16" || type_name == "bfloat16") &&
      // K <= 512 q8_0 fp16 routes to the generic-walk "_small" batch-1
      // kernel; the specialized q8_0 mb twin would change summation order.
      !(K <= 512 && fmt == "q8_0" && type_name == "float16");
  if (mb_ok) {
    auto input = x.contiguous();
    encode("qc_mmvq_mb", [&](TorchEncoder& e) {
      tk::launch_qgemv_mb(e, out, w, input, N, K, batch, fmt, type_name);
    });
    return out;
  }

  // Multi-row blocks ride the weight-stationary qgemv_mm variants so the
  // quantized bytes are read once per block instead of once per row.
  // Instantiated row counts, largest-first; 17 is the speculative-verify
  // width (k+1) and gets a single dispatch.
  static const int kMMRows[] = {17, 16, 8, 4, 2};
  static const char* const kMMFormats[] = {
      "q4_0",   "q8_0",  "q4_K",    "q5_K",  "q6_K",
      "q2_K",   "q3_K",  "iq1_s",   "iq1_m", "iq2_xxs",
      "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs"};
  bool has_mm = false;
  // Keep the q8_0-small carve-out from the DSV4 path: its batch-1 kernel has
  // a different summation order. All other formats supported by the merged
  // qgemv_mm shader can use the multi-row route added for Qwen verification.
  if (type_name != "float32" &&
      !(K <= 512 && fmt == "q8_0" && type_name == "float16")) {
    for (const char* supported : kMMFormats) {
      if (fmt == supported) {
        has_mm = true;
        break;
      }
    }
  }
  encode("qc_mmvq_loop", [&](TorchEncoder& e) {
    int b = 0;
    while (b < batch) {
      const int rem = batch - b;
      int chunk = 1;
      if (has_mm) {
        for (int s : kMMRows) {
          if (s <= rem) {
            chunk = s;
            break;
          }
        }
      }
      if (chunk > 1) {
        const at::Tensor x_rows = x.narrow(0, b, chunk);
        const at::Tensor out_rows = out.narrow(0, b, chunk);
        tk::launch_qgemv_mm(e, out_rows, w, x_rows, N, K, chunk, fmt,
                            type_name);
      } else {
        const at::Tensor x_row = x.select(0, b);
        const at::Tensor out_row = out.select(0, b);
        tk::launch_qgemv(e, out_row, w, x_row, N, K, fmt, type_name);
      }
      b += chunk;
    }
  });
  return out;
}

at::Tensor ggml_moe_a8_vec(const at::Tensor& x, const at::Tensor& w,
                           const at::Tensor& topk_ids_in, int64_t top_k,
                           int64_t quant_type, int64_t row, int64_t tokens,
                           bool soa) {
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
  auto out = ring_out("moe_out", {num_tokens * topk, N}, x.options());
  const std::string fmt = ggml_type_to_format(quant_type);
  // SoA-repacked experts (the load-time permutation in gguf/fused_moe.py)
  // are only readable by the multi-row SoA kernels; the AoS fallback would
  // decode garbage, so fail loudly instead of falling through.
  if (soa) {
    TORCH_CHECK(fmt == "q2_K",
                "SoA MoE repack covers q2_K only (iq2_xxs is AoS), got ", fmt);
    TORCH_CHECK(K % 256 == 0,
                "SoA-repacked experts need the multi-row kernels (K % 256)");
    TORCH_CHECK((w.size(1) * w.size(2)) % 8 == 0,
                "SoA expert stride must stay 8-byte aligned, got ", w.size(1),
                " x ", w.size(2));
  }
  // Multi-row kernel (see qgemv_moe_mr_*): ULP-level output changes vs the
  // one-simdgroup-per-row kernel, which stays the route for every other
  // format and for K % 256 != 0. The mr grid ceil-divides N over nsg*nr0
  // rows per threadgroup, and tail simdgroups read weight rows past N
  // before the store guards run — so a non-multiple N (never the case for
  // the DSV4 dims) also stays on the safe one-row route.
  const int mr_rows =
      (fmt == "q2_K" && x.scalar_type() != at::kBFloat16) ? 32 : 8;
  if ((fmt == "iq2_xxs" || fmt == "q2_K") && K % 256 == 0 && N % mr_rows == 0) {
    encode("qc_moe_vec_mr", [&](TorchEncoder& e) {
      tk::launch_qgemv_moe_mr(e, out, w, input, topk_ids, N, K, num_tokens,
                              topk, fmt, activation_type_name(input), soa);
    });
    return out;
  }
  encode("qc_moe_vec", [&](TorchEncoder& e) {
    tk::launch_qgemv_moe(e, out, w, input, topk_ids, N, K, num_tokens, topk,
                         fmt, activation_type_name(input));
  });
  return out;
}

// Multi-row iq2_xxs MoE GEMV with fused SwiGLU epilogue: returns the
// activated (tokens*topk, row/2) tensor. Bit-exact vs
// ggml_moe_a8_vec + qc_swiglu (oai_form 0) — see the kernel comment.
at::Tensor ggml_moe_a8_vec_swiglu(const at::Tensor& x, const at::Tensor& w,
                                  const at::Tensor& topk_ids_in, int64_t top_k,
                                  int64_t quant_type, int64_t row,
                                  int64_t tokens,
                                  std::optional<double> clamp_limit, bool soa) {
  check_mps(w, "w");
  check_mps(x, "x");
  check_mps(topk_ids_in, "topk_ids");
  const std::string fmt = ggml_type_to_format(quant_type);
  TORCH_CHECK(fmt == "iq2_xxs",
              "ggml_moe_a8_vec_swiglu supports iq2_xxs only, got ", fmt);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == tokens,
              "MoE input must be [tokens, K]");
  TORCH_CHECK(row % 2 == 0, "gate|up row count must be even");
  // The mr swiglu grid ceil-divides N/2 over 4 gate/up pairs per
  // threadgroup; tail simdgroups would read weight rows past N otherwise.
  TORCH_CHECK(row % 8 == 0,
              "multi-row swiglu kernel needs N divisible by 8, got ", row);
  TORCH_CHECK(x.size(1) % 256 == 0, "iq2_xxs needs K % 256 == 0");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "Metal GGUF MoE supports float16/bfloat16 activations");
  const int num_tokens = static_cast<int>(tokens);
  const int topk = static_cast<int>(top_k);
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(1));
  auto topk_ids = topk_ids_in.to(at::kInt).contiguous();
  auto input = x.contiguous();
  auto out = ring_out("moes_out", {num_tokens * topk, N / 2}, x.options());
  const int has_clamp = clamp_limit.has_value() ? 1 : 0;
  const float limit =
      clamp_limit.has_value() ? static_cast<float>(*clamp_limit) : 0.0f;
  TORCH_CHECK(!soa, "iq2_xxs gate|up weights are AoS-only (no SoA repack)");
  // No expert-grouped twin: measured negative, 123 -> 230 ms/step (see
  // optimization_status 2026-08-13, expert-grouped w13 entry).
  encode("qc_moe_vec_mr_swiglu", [&](TorchEncoder& e) {
    tk::launch_qgemv_moe_mr_swiglu(e, out, w, input, topk_ids, N, K, num_tokens,
                                   topk, has_clamp, limit,
                                   activation_type_name(input));
  });
  return out;
}

// Sum-folded Q2_K down projection: the multi-row down GEMV with the
// qc_moe_weighted_sum reduce folded into its epilogue, writing (tokens, N)
// into `out` directly — deletes the (tokens*topk, N) intermediate
// round-trip and the separate weighted-sum dispatch. x holds the per-slot
// activations in flat (token, k) order (the swiglu kernel's output). q2_K
// only (the DSV4 routed down format); AoS and SoA expert stacks both
// supported. Rounding points match the unfused chain (see the kernel
// comment); bitwise equivalence is oracle-gated.
at::Tensor ggml_moe_a8_vec_sum(const at::Tensor& x, const at::Tensor& w,
                               const at::Tensor& topk_ids_in,
                               const at::Tensor& topk_w_in, int64_t top_k,
                               int64_t quant_type, int64_t row, int64_t tokens,
                               const at::Tensor& out, bool soa) {
  check_mps(w, "w");
  check_mps(x, "x");
  check_mps(topk_ids_in, "topk_ids");
  check_mps(topk_w_in, "topk_w");
  check_mps(out, "out");
  const std::string fmt = ggml_type_to_format(quant_type);
  TORCH_CHECK(fmt == "q2_K", "ggml_moe_a8_vec_sum supports q2_K only, got ",
              fmt);
  TORCH_CHECK(tokens > 0 && top_k > 0 && top_k <= 8,
              "tokens must be positive and top_k in [1, 8], got ", tokens,
              " and ", top_k);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == tokens * top_k,
              "MoE input must be [tokens*top_k, K], got ", x.sizes());
  TORCH_CHECK(x.size(1) % 256 == 0, "q2_K needs K % 256 == 0");
  TORCH_CHECK(topk_ids_in.numel() == tokens * top_k,
              "topk_ids must contain tokens * top_k entries, got ",
              topk_ids_in.numel());
  TORCH_CHECK(topk_w_in.numel() == tokens * top_k,
              "topk_w must contain tokens * top_k entries, got ",
              topk_w_in.numel());
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "Metal GGUF MoE supports float16/bfloat16 activations");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == tokens && out.size(1) == row &&
                  out.scalar_type() == x.scalar_type() && out.is_contiguous(),
              "out must be contiguous [tokens, row] with the activation "
              "dtype, got ",
              out.sizes(), " ", out.scalar_type());
  if (soa) {
    TORCH_CHECK((w.size(1) * w.size(2)) % 8 == 0,
                "SoA expert stride must stay 8-byte aligned, got ", w.size(1),
                " x ", w.size(2));
  }
  const int num_tokens = static_cast<int>(tokens);
  const int topk = static_cast<int>(top_k);
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(1));
  // Tail simdgroups of the mr grid read weight rows past a non-multiple N
  // before the store guards run; there is no one-row fallback here.
  const int sum_rows = x.scalar_type() != at::kBFloat16 ? 32 : 8;
  TORCH_CHECK(N % sum_rows == 0,
              "sum-folded q2_K kernel needs N divisible "
              "by ",
              sum_rows, ", got ", N);
  auto topk_ids = topk_ids_in.to(at::kInt).contiguous();
  auto topk_w = topk_w_in.scalar_type() == at::kFloat
                    ? topk_w_in.contiguous()
                    : topk_w_in.to(at::kFloat).contiguous();
  auto input = x.contiguous();
  encode("qc_moe_vec_mr_sum", [&](TorchEncoder& e) {
    tk::launch_qgemv_moe_mr_q2k_sum(e, out, w, input, topk_ids, topk_w, N, K,
                                    num_tokens, topk,
                                    activation_type_name(input), soa);
  });
  return out;
}

// Tiled MoE GEMM for prefill widths (llama.cpp mul_mm_id port): two-phase
// map0 (per-expert dense slot lists) + 64x32 simdgroup-MMA tiles over the
// raw iq2_xxs expert blocks. Returns (tokens*top_k, row) in the vec kernels'
// flat (token, slot) row order, so the existing act() and down paths consume
// it unchanged. AoS weights only (the resident w13 layout); the GEMV path
// stays the decode answer below the host token threshold.
at::Tensor ggml_moe_mm_id(const at::Tensor& x, const at::Tensor& w,
                          const at::Tensor& topk_ids_in, int64_t top_k,
                          int64_t quant_type, int64_t row, int64_t tokens,
                          bool soa) {
  check_mps(w, "w");
  check_mps(x, "x");
  check_mps(topk_ids_in, "topk_ids");
  const std::string fmt = ggml_type_to_format(quant_type);
  TORCH_CHECK(fmt == "iq2_xxs" || fmt == "q2_K",
              "ggml_moe_mm_id supports iq2_xxs (w13) and q2_K (down), got ",
              fmt);
  const bool down = fmt == "q2_K";
  // w13 consumes per-token rows (B row = id/topk); down consumes the
  // per-slot activations the w13 path produced (B row = id).
  TORCH_CHECK(x.dim() == 2 && x.size(0) == (down ? tokens * top_k : tokens),
              "MoE input must be [", down ? "tokens*top_k" : "tokens",
              ", K], got ", x.sizes());
  TORCH_CHECK(x.scalar_type() == at::kHalf,
              "ggml_moe_mm_id needs float16 activations, got ",
              x.scalar_type());
  TORCH_CHECK(!soa || down, "SoA expert stacks are q2_K-only here (iq2_xxs "
                            "w13 is AoS by measurement)");
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(row);
  const int64_t block_bytes = down ? 84 : 66;
  TORCH_CHECK(K % 256 == 0, "K % 256 == 0 required, got ", K);
  TORCH_CHECK(N % 64 == 0, "tile kernel needs row % 64 == 0, got ", N);
  TORCH_CHECK(w.dim() == 3 && w.size(1) == row &&
                  w.size(2) == (int64_t)(K / 256) * block_bytes,
              "expert stack must be [E, row, K/256*", block_bytes, "] raw ",
              fmt, ", got ", w.sizes());
  const int E = static_cast<int>(w.size(0));
  TORCH_CHECK(E <= 256,
              "map0 runs one thread per expert in one threadgroup, E <= 256, "
              "got ",
              E);
  TORCH_CHECK(top_k == 2 || top_k == 4 || top_k == 6 || top_k == 8,
              "qc_moe_mm_map0 is instantiated for top_k in {2,4,6,8}, got ",
              top_k);
  TORCH_CHECK(tokens > 0 && topk_ids_in.numel() == tokens * top_k,
              "topk_ids must contain tokens * top_k entries, got ",
              topk_ids_in.numel(), " for ", tokens, " * ", top_k);
  TORCH_CHECK(tokens < 65536,
              "map0 packs slot indices into 16 bits, tokens < 65536");
  const int num_tokens = static_cast<int>(tokens);
  const int topk = static_cast<int>(top_k);
  auto topk_ids = topk_ids_in.to(at::kInt).contiguous();
  auto input = x.contiguous();
  // int32 scratch is bit-compatible with the kernel's u32 counts. The output
  // is a plain allocation on purpose: ~100 MB at chunk width dwarfs the ring's
  // 8 MiB bypass anyway.
  auto tpe = at::empty({E}, topk_ids.options());
  auto ids = at::empty({(int64_t)E * num_tokens}, topk_ids.options());
  // Tile-queue caps: sum of per-expert ceil(count/32) <= slots/32 + E, and
  // ceil(count/64) <= slots/64 + E for the dual-half queue.
  const int work_cap = (num_tokens * topk + 31) / 32 + E;
  const int work_cap64 = (num_tokens * topk + 63) / 64 + E;
  auto work = at::empty({work_cap}, topk_ids.options());
  auto wcount = at::empty({1}, topk_ids.options());
  auto work64 = at::empty({work_cap64}, topk_ids.options());
  auto wcount64 = at::empty({1}, topk_ids.options());
  // Caller contract: every topk_id is >= 0. map0 drops negative router
  // ids, so their slot rows would appear in no expert's ids list, neither
  // tile kernel would write them, and this pooled allocation would surface
  // them as stale memory (the vec kernels write T(0) for expert < 0).
  // Negative ids only arise via expert_map indexing, and both Python
  // callers gate this route on expert_map is None; a zeros fill here was
  // measured at ~107 MB per call (~8.6 GB per 2176-token chunk step across
  // layers) to defend that unreachable case, so the contract stays with
  // the caller.
  auto out = at::empty({tokens * top_k, (int64_t)N}, x.options());
  // Dual-half 64-slot tiles amortize one weight-dequant pass over two slot
  // halves (per-slot bit-identical to the 32-wide kernels). iq2_xxs only:
  // the q2_K twin is slower on the SoA layout (optimization_status
  // 2026-08-14 v8a), so q2_K stays on the 32-wide kernels and its unused
  // 64-slot queue below is allocate-and-discard (map0 writes both queues
  // unconditionally).
  const bool w64 = fmt == "iq2_xxs";
  encode("qc_moe_mm_map0", [&](TorchEncoder& e) {
    tk::launch_moe_mm_map0(e, topk_ids, tpe, ids, work, wcount, work64,
                           wcount64, num_tokens, topk, E);
  });
  encode("qc_moe_mm_id", [&](TorchEncoder& e) {
    tk::launch_moe_mm_id(e, out, w, input, tpe, ids, w64 ? work64 : work,
                         w64 ? wcount64 : wcount, w64 ? work_cap64 : work_cap,
                         N, K, num_tokens, topk, fmt, soa, w64);
  });
  return out;
}

// Tile policy for the GGUF weight-only GEMM.
//  - wide: 64x64 tile with a transposed [M, N] row-major store (q8_0 only;
//    needs N % 64 == 0). Wins on N-heavy shapes from 512 rows and on any
//    shape from 1024 rows; loses below that on narrow-N/tall-K shapes.
//    Per-element bit-identical to the narrow path.
//  - narrow: 32x32 tile (64 for mxfp8), [N, M] store + host transpose.
struct MmqPlan {
  bool wide;
  int tile_m;
  int m_padded;
};
inline MmqPlan mmq_plan(const std::string& fmt, int N, int output_rows) {
  MmqPlan p;
  p.wide = fmt == "q8_0" && N % 64 == 0 &&
           (output_rows >= 1024 || (output_rows >= 512 && N >= 8192));
  p.tile_m = (p.wide || fmt == "mxfp8") ? 64 : 32;
  p.m_padded = ((output_rows + p.tile_m - 1) / p.tile_m) * p.tile_m;
  return p;
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
  const MmqPlan plan = mmq_plan(fmt, N, output_rows);
  const int M = plan.m_padded;

  TORCH_CHECK(N % 32 == 0,
              "quixicore(metal): qgemm needs N % 32 == 0, got N=", N);

  // The Metal tile consumes X as column-major [K, M] fp16 and emits [N, M]
  // fp16. vLLM supplies row-major [rows, K] BF16, and decode/spec batches do
  // not generally end on a 32-column tile. Convert, transpose, and pad here so
  // the public op preserves the ordinary [rows, N] dtype/layout contract.
  auto input = x.to(at::kHalf).transpose(0, 1).contiguous();
  at::Tensor input_padded;
  if (M == output_rows) {
    input_padded = input;
  } else {
    input_padded = at::zeros({K, M}, x.options().dtype(at::kHalf));
    input_padded.narrow(1, 0, output_rows).copy_(input);
  }

  if (plan.wide) {
    // Both mul_mat_a8 outputs bypass the ring on purpose: prefill-width
    // GEMM outputs are far past the ring's 8 MiB cap.
    auto out_t = at::empty({M, N}, x.options().dtype(at::kHalf));
    encode("qc_mmq", [&](TorchEncoder& e) {
      tk::launch_qgemm_wide_t(e, out_t, w, input_padded, N, K, M);
    });
    auto out = out_t.narrow(0, 0, output_rows);
    return x.scalar_type() == at::kHalf ? out : out.to(x.scalar_type());
  }

  auto output_padded = at::empty({N, M}, x.options().dtype(at::kHalf));

  encode("qc_mmq", [&](TorchEncoder& e) {
    tk::launch_qgemm(e, output_padded, w, input_padded, N, K, M, fmt);
  });
  return output_padded.narrow(1, 0, output_rows)
      .transpose(0, 1)
      .to(x.scalar_type())
      .contiguous();
}

// ---- DeepSeek-V4 mHC (hyper-connections) ----------------------------------
//
// Metal counterparts of the Ampere `dsv4_mhc_*` ops (mhc_ampere.cuh), shaped
// for the decode/verify token counts this box serves: one threadgroup per
// token runs projection, RMS statistic, gates, and the full Sinkhorn loop in
// a single dispatch. The torch decomposition this replaces costs ~230 aten
// launches per call. Kernels live in kernels/serving/dsv4_mhc/dsv4_mhc.metal.

void check_mhc_activation(const at::Tensor& t, const char* name) {
  check_mps(t, name);
  TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16,
              name, " must be fp16 or bf16, got ", t.scalar_type());
}

void check_mhc_f32(const at::Tensor& t, const char* name) {
  check_mps(t, name);
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be fp32, got ",
              t.scalar_type());
}

constexpr int kMhcThreads = 256;

std::tuple<at::Tensor, at::Tensor, at::Tensor> dsv4_mhc_pre(
    const at::Tensor& residual, const at::Tensor& fn,
    const at::Tensor& hc_scale, const at::Tensor& hc_base, double rms_eps,
    double pre_eps, double sinkhorn_eps, double post_multiplier,
    int64_t sinkhorn_repeat, const std::optional<at::Tensor>& norm_weight,
    double norm_eps) {
  TORCH_CHECK(!norm_weight.has_value(),
              "quixicore(metal): dsv4_mhc_pre norm fusion is not wired");
  check_mhc_activation(residual, "residual");
  check_mhc_f32(fn, "fn");
  check_mhc_f32(hc_scale, "hc_scale");
  check_mhc_f32(hc_base, "hc_base");
  TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4,
              "residual must be [tokens, 4, hidden], got ", residual.sizes());
  const auto tokens = residual.size(0);
  const auto hidden = residual.size(2);
  TORCH_CHECK(fn.size(0) == 24 && fn.size(1) == 4 * hidden,
              "fn must be [24, 4*hidden], got ", fn.sizes());

  auto opts_f32 = residual.options().dtype(at::kFloat);
  at::Tensor post = ring_out("mhc_post4", {tokens, 4}, opts_f32);
  at::Tensor comb = ring_out("mhc_comb", {tokens, 4, 4}, opts_f32);
  at::Tensor layer_input =
      ring_out("mhc_li", {tokens, hidden}, residual.options());

  const uint32_t h = static_cast<uint32_t>(hidden);
  const float f_rms = static_cast<float>(rms_eps);
  const float f_pre = static_cast<float>(pre_eps);
  const float f_sink = static_cast<float>(sinkhorn_eps);
  const float f_mult = static_cast<float>(post_multiplier);
  const int32_t repeat = static_cast<int32_t>(sinkhorn_repeat);

  // Two-pass pre: a dots pass fills the [tokens, 25] scratch, then a small
  // finalize runs gates + Sinkhorn. The dots pass has two bit-identical
  // shapes (same per-lane fma order and simd_sum tree): the threadgroup
  // variant stages the residual once per token (prefill widths), the
  // simdgroup-per-job variant keeps 25x-threadgroup occupancy at decode
  // widths.
  at::Tensor scratch = ring_out("mhc_scratch", {tokens, 25}, opts_f32);
  const bool dots_tg = tokens >= 64 && hidden <= 4096;
  encode("qc_dsv4_mhc_pre_dots", [&](TorchEncoder& e) {
    e.pipeline((dots_tg ? "dsv4_mhc_pre_dots_tg_" : "dsv4_mhc_pre_dots_") +
               activation_type_name(residual));
    e.in(residual, 0);
    e.in(fn, 1);
    e.out(scratch, 2);
    e.bytes(h, 3);
    if (dots_tg) {
      e.dispatch(static_cast<int>(tokens), 1, 1, kMhcThreads, 1, 1);
    } else {
      e.dispatch(static_cast<int>(tokens), 25, 1, 32, 1, 1);
    }
  });
  encode("qc_dsv4_mhc_pre_fin", [&](TorchEncoder& e) {
    e.pipeline("dsv4_mhc_pre_finalize_" + activation_type_name(residual));
    e.in(residual, 0);
    e.in(scratch, 1);
    e.in(hc_scale, 2);
    e.in(hc_base, 3);
    e.out(post, 4);
    e.out(comb, 5);
    e.out(layer_input, 6);
    e.bytes(h, 7);
    e.bytes(f_rms, 8);
    e.bytes(f_pre, 9);
    e.bytes(f_sink, 10);
    e.bytes(f_mult, 11);
    e.bytes(repeat, 12);
    e.dispatch(static_cast<int>(tokens), 1, 1, kMhcThreads, 1, 1);
  });
  return {post, comb, layer_input};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dsv4_mhc_fused_post_pre(const at::Tensor& x, const at::Tensor& residual,
                        const at::Tensor& post_mix, const at::Tensor& comb_mix,
                        const at::Tensor& fn, const at::Tensor& hc_scale,
                        const at::Tensor& hc_base, double rms_eps,
                        double pre_eps, double sinkhorn_eps,
                        double post_multiplier, int64_t sinkhorn_repeat,
                        const std::optional<at::Tensor>& norm_weight,
                        double norm_eps) {
  TORCH_CHECK(!norm_weight.has_value(),
              "quixicore(metal): dsv4_mhc_fused_post_pre norm fusion is not "
              "wired");
  check_mhc_activation(x, "x");
  check_mhc_activation(residual, "residual");
  check_mhc_f32(post_mix, "post_mix");
  check_mhc_f32(comb_mix, "comb_mix");
  check_mhc_f32(fn, "fn");
  check_mhc_f32(hc_scale, "hc_scale");
  check_mhc_f32(hc_base, "hc_base");
  TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4,
              "residual must be [tokens, 4, hidden], got ", residual.sizes());
  TORCH_CHECK(x.scalar_type() == residual.scalar_type(),
              "x and residual dtypes must match");
  const auto tokens = residual.size(0);
  const auto hidden = residual.size(2);
  TORCH_CHECK(x.dim() == 2 && x.size(0) == tokens && x.size(1) == hidden,
              "x must be [tokens, hidden], got ", x.sizes());
  TORCH_CHECK(fn.size(0) == 24 && fn.size(1) == 4 * hidden,
              "fn must be [24, 4*hidden], got ", fn.sizes());

  auto opts_f32 = residual.options().dtype(at::kFloat);
  at::Tensor residual_out = ring_out_like("mhc_resout", residual);
  at::Tensor post = ring_out("mhcf_post4", {tokens, 4}, opts_f32);
  at::Tensor comb = ring_out("mhcf_comb", {tokens, 4, 4}, opts_f32);
  at::Tensor layer_input =
      ring_out("mhcf_li", {tokens, hidden}, residual.options());

  const uint32_t h = static_cast<uint32_t>(hidden);
  const float f_rms = static_cast<float>(rms_eps);
  const float f_pre = static_cast<float>(pre_eps);
  const float f_sink = static_cast<float>(sinkhorn_eps);
  const float f_mult = static_cast<float>(post_multiplier);
  const int32_t repeat = static_cast<int32_t>(sinkhorn_repeat);

  encode("qc_dsv4_mhc_fused_post_pre", [&](TorchEncoder& e) {
    e.pipeline("dsv4_mhc_fused_post_pre_" + activation_type_name(residual));
    e.in(x, 0);
    e.in(residual, 1);
    e.in(post_mix, 2);
    e.in(comb_mix, 3);
    e.in(fn, 4);
    e.in(hc_scale, 5);
    e.in(hc_base, 6);
    e.out(residual_out, 7);
    e.out(post, 8);
    e.out(comb, 9);
    e.out(layer_input, 10);
    e.bytes(h, 11);
    e.bytes(f_rms, 12);
    e.bytes(f_pre, 13);
    e.bytes(f_sink, 14);
    e.bytes(f_mult, 15);
    e.bytes(repeat, 16);
    e.dispatch(static_cast<int>(tokens), 1, 1, kMhcThreads, 1, 1);
  });
  return {residual_out, post, comb, layer_input};
}

at::Tensor dsv4_mhc_post(const at::Tensor& x, const at::Tensor& residual,
                         const at::Tensor& post_mix,
                         const at::Tensor& comb_mix) {
  check_mhc_activation(x, "x");
  check_mhc_activation(residual, "residual");
  check_mhc_f32(post_mix, "post_mix");
  check_mhc_f32(comb_mix, "comb_mix");
  TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4,
              "residual must be [tokens, 4, hidden], got ", residual.sizes());
  const auto tokens = residual.size(0);
  const auto hidden = residual.size(2);

  at::Tensor out = ring_out_like("mhcp_out", residual);
  const uint32_t h = static_cast<uint32_t>(hidden);
  encode("qc_dsv4_mhc_post", [&](TorchEncoder& e) {
    e.pipeline("dsv4_mhc_post_" + activation_type_name(residual));
    e.in(x, 0);
    e.in(residual, 1);
    e.in(post_mix, 2);
    e.in(comb_mix, 3);
    e.out(out, 4);
    e.bytes(h, 5);
    // Elementwise kernel slices its H walk across tgid.y (bit-exact for any
    // width); 8 slices lifts decode occupancy above `tokens` threadgroups.
    e.dispatch(static_cast<int>(tokens), 8, 1, kMhcThreads, 1, 1);
  });
  return out;
}

at::Tensor dsv4_hc_head(const at::Tensor& residual, const at::Tensor& fn,
                        const at::Tensor& hc_scale, const at::Tensor& hc_base,
                        double rms_eps, double hc_eps) {
  check_mhc_activation(residual, "residual");
  check_mhc_f32(fn, "fn");
  check_mhc_f32(hc_scale, "hc_scale");
  check_mhc_f32(hc_base, "hc_base");
  TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4,
              "residual must be [tokens, 4, hidden], got ", residual.sizes());
  const auto tokens = residual.size(0);
  const auto hidden = residual.size(2);
  TORCH_CHECK(fn.size(0) == 4 && fn.size(1) == 4 * hidden,
              "fn must be [4, 4*hidden], got ", fn.sizes());

  at::Tensor out = ring_out("mhcp_out2", {tokens, hidden}, residual.options());
  const uint32_t h = static_cast<uint32_t>(hidden);
  const float f_rms = static_cast<float>(rms_eps);
  const float f_hc = static_cast<float>(hc_eps);
  encode("qc_dsv4_hc_head", [&](TorchEncoder& e) {
    e.pipeline("dsv4_hc_head_" + activation_type_name(residual));
    e.in(residual, 0);
    e.in(fn, 1);
    e.in(hc_scale, 2);
    e.in(hc_base, 3);
    e.out(out, 4);
    e.bytes(h, 5);
    e.bytes(f_rms, 6);
    e.bytes(f_hc, 7);
    e.dispatch(static_cast<int>(tokens), 1, 1, kMhcThreads, 1, 1);
  });
  return out;
}

// ---------------------------------------------------------------------------
// Command-buffer census (diagnostic). Swizzles the factory methods of the
// concrete MTLCommandQueue class torch's MPS stream uses, counting every
// command buffer created and accumulating GPU busy time from the completion
// handlers. Answers "how many command buffers per decode step, and what
// fraction of wall time is the GPU actually executing".
// ---------------------------------------------------------------------------

std::atomic<uint64_t> g_cbc_created{0};
std::atomic<uint64_t> g_cbc_completed{0};
std::atomic<uint64_t> g_cbc_gpu_busy_ns{0};
std::atomic<bool> g_cbc_installed{false};

void cbc_attach(id<MTLCommandBuffer> cb) {
  g_cbc_created.fetch_add(1, std::memory_order_relaxed);
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
    const double s = done.GPUStartTime;
    const double e = done.GPUEndTime;
    if (e > s) {
      g_cbc_gpu_busy_ns.fetch_add(static_cast<uint64_t>((e - s) * 1e9),
                                  std::memory_order_relaxed);
    }
    g_cbc_completed.fetch_add(1, std::memory_order_relaxed);
  }];
}

using CbcFactory0 = id (*)(id, SEL);
using CbcFactory1 = id (*)(id, SEL, id);
CbcFactory0 g_cbc_orig_plain = nullptr;
CbcFactory0 g_cbc_orig_unretained = nullptr;
CbcFactory1 g_cbc_orig_descriptor = nullptr;

id cbc_commandBuffer(id self, SEL _cmd) {
  id cb = g_cbc_orig_plain(self, _cmd);
  if (cb) cbc_attach(cb);
  return cb;
}

id cbc_commandBufferWithUnretainedReferences(id self, SEL _cmd) {
  id cb = g_cbc_orig_unretained(self, _cmd);
  if (cb) cbc_attach(cb);
  return cb;
}

id cbc_commandBufferWithDescriptor(id self, SEL _cmd, id desc) {
  id cb = g_cbc_orig_descriptor(self, _cmd, desc);
  if (cb) cbc_attach(cb);
  return cb;
}

bool cb_census_install() {
  if (g_cbc_installed.load(std::memory_order_acquire)) {
    return true;
  }
  id<MTLCommandBuffer> cur = torch::mps::get_command_buffer();
  TORCH_CHECK(cur != nil, "cb_census_install: no MPS command buffer");
  id<MTLCommandQueue> queue = [cur commandQueue];
  TORCH_CHECK(queue != nil, "cb_census_install: command buffer has no queue");
  Class cls = object_getClass((id)queue);

  auto swizzle = [&](SEL sel, IMP replacement, void* orig_slot) {
    Method method = class_getInstanceMethod(cls, sel);
    if (method == nullptr) {
      return;
    }
    *static_cast<IMP*>(orig_slot) = method_getImplementation(method);
    class_replaceMethod(cls, sel, replacement, method_getTypeEncoding(method));
  };
  swizzle(@selector(commandBuffer), (IMP)cbc_commandBuffer,
          (void*)&g_cbc_orig_plain);
  swizzle(@selector(commandBufferWithUnretainedReferences),
          (IMP)cbc_commandBufferWithUnretainedReferences,
          (void*)&g_cbc_orig_unretained);
  swizzle(@selector(commandBufferWithDescriptor:),
          (IMP)cbc_commandBufferWithDescriptor, (void*)&g_cbc_orig_descriptor);
  // Latch only on success so a factory-less first call can be retried.
  const bool ok = g_cbc_orig_plain != nullptr;
  if (ok) {
    g_cbc_installed.store(true, std::memory_order_release);
  }
  return ok;
}

pybind11::tuple cb_census_read() {
  return pybind11::make_tuple(
      g_cbc_created.load(std::memory_order_relaxed),
      g_cbc_completed.load(std::memory_order_relaxed),
      static_cast<double>(g_cbc_gpu_busy_ns.load(std::memory_order_relaxed)) /
          1e9);
}

// ---------------------------------------------------------------------------
// Weight residency pinning (macOS 15+). torch's MPS buffers are pageable
// anonymous memory whose GPU residency is only declared per command buffer;
// between touches macOS proactively compresses them in tens-of-GiB waves and
// the GPU stalls faulting them back mid-kernel (measured: compressor
// sawtooth 1->46->1 GiB per step, decode at ~0.1 tok/s). An MTLResidencySet
// attached to the command queue keeps the weight allocations permanently
// resident so the compressor never takes them. Residency is heap-granular:
// heap-backed buffers pin their backing MTLHeap (deduplicated).
// ---------------------------------------------------------------------------

id g_residency_set = nil;  // id<MTLResidencySet> under @available(macOS 15)

pybind11::tuple residency_pin(const std::vector<at::Tensor>& tensors) {
  if (@available(macOS 15.0, *)) {
    id<MTLCommandBuffer> cur = torch::mps::get_command_buffer();
    TORCH_CHECK(cur != nil, "residency_pin: no MPS command buffer");
    id<MTLCommandQueue> queue = [cur commandQueue];
    id<MTLDevice> device = cur.device;

    NSUInteger added = 0;
    uint64_t total_bytes = 0;
    std::unordered_set<void*> seen;

    if (g_residency_set == nil) {
      MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
      desc.label = @"quixicore-weight-residency";
      desc.initialCapacity = tensors.size();
      NSError* err = nil;
      id<MTLResidencySet> set = [device newResidencySetWithDescriptor:desc
                                                                error:&err];
      TORCH_CHECK(set != nil, "residency_pin: newResidencySet failed: ",
                  err ? err.localizedDescription.UTF8String : "unknown");
      g_residency_set = set;
    }
    id<MTLResidencySet> set = (id<MTLResidencySet>)g_residency_set;

    for (const auto& t : tensors) {
      if (!t.defined() || t.device().type() != at::kMPS) {
        continue;
      }
      id<MTLBuffer> buf = mtl_buffer(t);
      if (buf == nil) {
        continue;
      }
      id<MTLHeap> heap = buf.heap;
      id<MTLAllocation> alloc =
          heap != nil ? (id<MTLAllocation>)heap : (id<MTLAllocation>)buf;
      void* key = (__bridge void*)alloc;
      if (seen.insert(key).second) {
        [set addAllocation:alloc];
        added += 1;
        total_bytes +=
            heap != nil ? heap.currentAllocatedSize : buf.allocatedSize;
      }
    }
    [set commit];
    [set requestResidency];
    [queue addResidencySet:set];
    return pybind11::make_tuple(static_cast<uint64_t>(added), total_bytes);
  }
  return pybind11::make_tuple(static_cast<uint64_t>(0),
                              static_cast<uint64_t>(0));
}

// Indexer Q RoPE + fp8-domain quantize (torch mirror:
// _fused_indexer_q_rope_quant_metal in fused_indexer_q.py). Returns
// (q_out same dtype as x holding value / q_scale, weights_out fp32 with
// q_scale folded). One simdgroup per head; replaces ~20 MPS dispatches per
// layer call. Kernels in kernels/serving/indexer/indexer.metal.
std::tuple<at::Tensor, at::Tensor> dsv4_indexer_q_rope_quant(
    const at::Tensor& index_q, const at::Tensor& positions,
    const at::Tensor& cos_sin_cache, const at::Tensor& index_weights,
    double softmax_scale, double head_scale) {
  check_mps(index_q, "index_q");
  check_mps(positions, "positions");
  check_mps(cos_sin_cache, "cos_sin_cache");
  check_mps(index_weights, "index_weights");
  TORCH_CHECK(index_q.dim() == 3, "index_q must be [tokens, heads, dim]");
  TORCH_CHECK(index_q.scalar_type() == at::kHalf ||
                  index_q.scalar_type() == at::kBFloat16,
              "index_q must be fp16 or bf16, got ", index_q.scalar_type());
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
  TORCH_CHECK(index_weights.scalar_type() == at::kFloat,
              "index_weights must be fp32");
  const auto tokens = index_q.size(0);
  const auto H = index_q.size(1);
  const auto D = index_q.size(2);
  const auto half_rot = cos_sin_cache.size(-1) / 2;
  const auto nope = D - 2 * half_rot;
  TORCH_CHECK(H % 8 == 0, "heads must be a multiple of 8, got ", H);
  TORCH_CHECK(half_rot <= 32, "half_rot must fit one simdgroup, got ",
              half_rot);
  TORCH_CHECK(nope >= 0, "cos_sin cache wider than head_dim");
  TORCH_CHECK(index_weights.numel() == tokens * H,
              "index_weights must be [tokens, heads]");
  const bool cs_f32 = cos_sin_cache.scalar_type() == at::kFloat;
  TORCH_CHECK(cs_f32 || cos_sin_cache.scalar_type() == index_q.scalar_type(),
              "cos_sin_cache must be fp32 or match index_q, got ",
              cos_sin_cache.scalar_type());

  at::Tensor q_out = ring_out_like("idxq_out", index_q);
  at::Tensor w_out =
      at::empty({tokens, H}, index_weights.options().dtype(at::kFloat));
  const int32_t h = static_cast<int32_t>(H);
  const int32_t d = static_cast<int32_t>(D);
  const int32_t nope_i = static_cast<int32_t>(nope);
  const int32_t hr = static_cast<int32_t>(half_rot);
  const float ss = static_cast<float>(softmax_scale);
  const float hs = static_cast<float>(head_scale);
  encode("qc_dsv4_indexer_q", [&](TorchEncoder& e) {
    e.pipeline("dsv4_indexer_q_" + activation_type_name(index_q) +
               (cs_f32 ? "_csf32" : ""));
    e.in(index_q, 0);
    e.in(cos_sin_cache, 1);
    e.in(positions, 2);
    e.in(index_weights, 3);
    e.out(q_out, 4);
    e.out(w_out, 5);
    e.bytes(h, 6);
    e.bytes(d, 7);
    e.bytes(nope_i, 8);
    e.bytes(hr, 9);
    e.bytes(ss, 10);
    e.bytes(hs, 11);
    e.dispatch(static_cast<int>(tokens), static_cast<int>(H / 8), 1, 256, 1, 1);
  });
  return {q_out, w_out};
}

// Decode-path indexer producer (torch mirror: metal_sparse_attn_indexer's
// num_decode branch). Computes relu-weighted MQA logits over the e4m3 K
// cache and writes top-k request-local indices (-1 padded) straight into
// topk_indices_buffer rows [0, tokens). One threadgroup per token.
void dsv4_indexer_topk_decode(const at::Tensor& q, const at::Tensor& weights,
                              const at::Tensor& kv_cache,
                              const at::Tensor& block_table,
                              const at::Tensor& cand, at::Tensor& out,
                              int64_t width, int64_t k_eff) {
  check_mps(q, "q");
  check_mps(weights, "weights");
  check_mps_strided(kv_cache, "kv_cache");
  check_mps(block_table, "block_table");
  check_mps(cand, "cand");
  check_mps_strided(out, "out");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 128,
              "q must be [tokens, 64, 128], got ", q.sizes());
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "q must be fp16 or bf16");
  TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be fp32");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte, "kv_cache must be uint8");
  TORCH_CHECK(kv_cache.size(-1) == 132, "kv_cache slot must be 132 bytes");
  TORCH_CHECK(block_table.scalar_type() == at::kInt,
              "block_table must be int32");
  TORCH_CHECK(cand.scalar_type() == at::kInt, "cand must be int32");
  TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32");
  TORCH_CHECK(out.stride(1) == 1, "out rows must be dense");
  TORCH_CHECK(width >= 1 && width <= block_table.size(1) * kv_cache.size(1),
              "width exceeds the block-table candidate capacity: ", width);
  TORCH_CHECK(width <= 1024 || k_eff <= 512,
              "streaming top-k supports at most 512 outputs, got ", k_eff);
  const auto tokens = q.size(0);
  TORCH_CHECK(k_eff <= out.size(1), "k_eff exceeds buffer width");
  TORCH_CHECK(tokens <= out.size(0), "more tokens than buffer rows");
  const int32_t w_i = static_cast<int32_t>(width);
  const int32_t k_i = static_cast<int32_t>(k_eff);
  const int32_t out_stride = static_cast<int32_t>(out.stride(0));
  const int32_t bs = static_cast<int32_t>(kv_cache.size(1));
  const int32_t bt_stride = static_cast<int32_t>(block_table.stride(0));
  const int64_t kv_stride = kv_cache.stride(0);
  encode("qc_dsv4_indexer_topk", [&](TorchEncoder& e) {
    e.pipeline("dsv4_indexer_topk_decode_" + activation_type_name(q));
    e.in(q, 0);
    e.in(weights, 1);
    e.in(kv_cache, 2);
    e.in(block_table, 3);
    e.in(cand, 4);
    e.out(out, 5);
    e.bytes(w_i, 6);
    e.bytes(k_i, 7);
    e.bytes(out_stride, 8);
    e.bytes(bs, 9);
    e.bytes(bt_stride, 10);
    e.bytes(kv_stride, 11);
    e.dispatch(static_cast<int>(tokens), 1, 1, 256, 1, 1);
  });
}

// Prefill twin of dsv4_indexer_topk_decode: the block-table row comes
// from the token's request (tok_req); candidate windows are request-local
// starting at 0, so outputs mirror the eager metal_indexer.py prefill
// chain's rebased indices to the decode kernel's ULP class.
void dsv4_indexer_topk_prefill(const at::Tensor& q, const at::Tensor& weights,
                               const at::Tensor& kv_cache,
                               const at::Tensor& block_table,
                               const at::Tensor& tok_req,
                               const at::Tensor& cand, at::Tensor& out,
                               int64_t width, int64_t k_eff) {
  check_mps(q, "q");
  check_mps(weights, "weights");
  check_mps_strided(kv_cache, "kv_cache");
  check_mps(block_table, "block_table");
  check_mps(tok_req, "tok_req");
  check_mps(cand, "cand");
  check_mps_strided(out, "out");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 128,
              "q must be [tokens, 64, 128], got ", q.sizes());
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "q must be fp16 or bf16");
  TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be fp32");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte, "kv_cache must be uint8");
  TORCH_CHECK(kv_cache.size(-1) == 132, "kv_cache slot must be 132 bytes");
  TORCH_CHECK(block_table.scalar_type() == at::kInt,
              "block_table must be int32");
  TORCH_CHECK(tok_req.scalar_type() == at::kInt, "tok_req must be int32");
  TORCH_CHECK(cand.scalar_type() == at::kInt, "cand must be int32");
  TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32");
  TORCH_CHECK(out.stride(1) == 1, "out rows must be dense");
  TORCH_CHECK(width >= 1 && width <= block_table.size(1) * kv_cache.size(1),
              "width exceeds the block-table candidate capacity: ", width);
  TORCH_CHECK(width <= 1024 || k_eff <= 512,
              "streaming top-k supports at most 512 outputs, got ", k_eff);
  const auto tokens = q.size(0);
  TORCH_CHECK(tok_req.numel() == tokens, "tok_req must have one entry/token");
  TORCH_CHECK(k_eff <= out.size(1), "k_eff exceeds buffer width");
  TORCH_CHECK(tokens <= out.size(0), "more tokens than buffer rows");
  const int32_t w_i = static_cast<int32_t>(width);
  const int32_t k_i = static_cast<int32_t>(k_eff);
  const int32_t out_stride = static_cast<int32_t>(out.stride(0));
  const int32_t bs = static_cast<int32_t>(kv_cache.size(1));
  const int32_t bt_stride = static_cast<int32_t>(block_table.stride(0));
  const int64_t kv_stride = kv_cache.stride(0);
  encode("qc_dsv4_indexer_topk_prefill", [&](TorchEncoder& e) {
    e.pipeline("dsv4_indexer_topk_prefill_" + activation_type_name(q));
    e.in(q, 0);
    e.in(weights, 1);
    e.in(kv_cache, 2);
    e.in(block_table, 3);
    e.in(cand, 4);
    e.out(out, 5);
    e.bytes(w_i, 6);
    e.bytes(k_i, 7);
    e.bytes(out_stride, 8);
    e.bytes(bs, 9);
    e.bytes(bt_stride, 10);
    e.bytes(kv_stride, 11);
    e.in(tok_req, 12);
    e.dispatch(static_cast<int>(tokens), 1, 1, 256, 1, 1);
  });
}

// Attention-output inverse RoPE (torch mirror: DeepseekScalingRotary
// .forward_native inverse=True, GPT-J style). Accepts a strided head-slice
// of the padded attention output; returns a contiguous [tokens, H*D] tensor
// ready for the grouped WO_A GEMVs. Replaces ~13 MPS dispatches per layer.
at::Tensor dsv4_o_inv_rope(const at::Tensor& o, const at::Tensor& positions,
                           const at::Tensor& cos_sin_cache) {
  check_mps_strided(o, "o");
  check_mps(positions, "positions");
  check_mps(cos_sin_cache, "cos_sin_cache");
  TORCH_CHECK(o.dim() == 3, "o must be [tokens, heads, dim]");
  TORCH_CHECK(o.scalar_type() == at::kHalf || o.scalar_type() == at::kBFloat16,
              "o must be fp16 or bf16, got ", o.scalar_type());
  TORCH_CHECK(o.stride(2) == 1, "o innermost dim must be dense");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
  const auto tokens = o.size(0);
  const auto H = o.size(1);
  const auto D = o.size(2);
  const auto half_rot = cos_sin_cache.size(-1) / 2;
  TORCH_CHECK(H % 8 == 0, "heads must be a multiple of 8, got ", H);
  TORCH_CHECK(half_rot <= 32, "half_rot must fit one simdgroup, got ",
              half_rot);
  TORCH_CHECK(2 * half_rot <= D, "cos_sin cache wider than head_dim");
  const bool cs_f32 = cos_sin_cache.scalar_type() == at::kFloat;
  TORCH_CHECK(cs_f32 || cos_sin_cache.scalar_type() == o.scalar_type(),
              "cos_sin_cache must be fp32 or match o, got ",
              cos_sin_cache.scalar_type());

  at::Tensor out = ring_out("oinv_out", {tokens, H * D}, o.options());
  const int32_t h = static_cast<int32_t>(H);
  const int32_t d = static_cast<int32_t>(D);
  const int32_t hr = static_cast<int32_t>(half_rot);
  const int64_t tok_stride = o.stride(0);
  const int64_t head_stride = o.stride(1);
  encode("qc_dsv4_o_inv_rope", [&](TorchEncoder& e) {
    e.pipeline("dsv4_o_inv_rope_" + activation_type_name(o) +
               (cs_f32 ? "_csf32" : ""));
    e.in(o, 0);
    e.in(cos_sin_cache, 1);
    e.in(positions, 2);
    e.out(out, 3);
    e.bytes(h, 4);
    e.bytes(d, 5);
    e.bytes(hr, 6);
    e.bytes(tok_stride, 7);
    e.bytes(head_stride, 8);
    e.dispatch(static_cast<int>(tokens), static_cast<int>(H / 8), 1, 256, 1, 1);
  });
  return out;
}

// Weighted RMS norm (vllm ir.ops.rms_norm numerics: fp32 statistic, final
// multiply in the weight dtype). One threadgroup per row; replaces the ~6
// dispatch eager MPS decomposition that the post-mHC census counted ~239x
// per engine step. Kernels in kernels/serving/rms_norm/rms_norm.metal.
at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight,
                    double epsilon) {
  // Muse-Glimmer's bf16 contiguous inputs keep their dedicated fixed-D
  // kernels (exact main-branch numerics); everything else - including the
  // DSV4 fp16 strided q/k splits - takes the path below.
  if (x.scalar_type() == at::kBFloat16 &&
      weight.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
      x.is_contiguous() && weight.numel() == x.size(1) && x.size(1) % 4 == 0) {
    return rms_norm_bf16_contig(x, weight, epsilon);
  }
  check_mps_strided(x, "x");  // prefill q/k splits pass strided views
  check_mps(weight, "weight");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "rms_norm: x must be fp16 or bf16, got ", x.scalar_type());
  // GGUF stores norm weights as F32; the reference keeps the weight multiply
  // in fp32 for that case (qc_rms_norm_w32_*), else in the shared dtype.
  const bool w32 = weight.scalar_type() == at::kFloat;
  TORCH_CHECK(w32 || weight.scalar_type() == x.scalar_type(),
              "rms_norm: weight must be fp32 or match x, got ",
              weight.scalar_type());
  const auto D = x.size(-1);
  TORCH_CHECK(weight.numel() == D, "rms_norm: weight must have ", D,
              " elements, got ", weight.numel());
  TORCH_CHECK(weight.is_contiguous(), "rms_norm: weight must be contiguous");
  // Row-strided 2-D views (fused-GEMM q/kv split halves) bind directly to
  // the *_strided_* kernel variants instead of paying an eager .contiguous()
  // clone per call (~285/step at the qk-norm sites). Identical element
  // reads — bit-exact.
  const bool strided_ok = x.dim() == 2 && !x.is_contiguous() &&
                          x.stride(1) == 1 && x.stride(0) >= D;
  auto input = strided_ok ? x : x.contiguous().view({-1, D});
  const auto tokens = input.size(0);
  at::Tensor out = strided_ok ? ring_out("rms_out", {tokens, D}, x.options())
                              : ring_out_like("rms_out_l", input);
  const uint32_t d = static_cast<uint32_t>(D);
  const float eps = static_cast<float>(epsilon);
  const uint64_t in_stride =
      strided_ok ? static_cast<uint64_t>(x.stride(0)) : 0;
  encode("qc_rms_norm", [&](TorchEncoder& e) {
    e.pipeline(
        (w32 ? std::string("qc_rms_norm_w32_") : std::string("qc_rms_norm_")) +
        (strided_ok ? "strided_" : "") + activation_type_name(input));
    e.in(input, 0);
    e.in(weight, 1);
    e.out(out, 2);
    e.bytes(d, 3);
    e.bytes(eps, 4);
    if (strided_ok) e.bytes(in_stride, 5);
    e.dispatch(static_cast<int>(tokens), 1, 1, 256, 1, 1);
  });
  return out.view(x.sizes());
}

// ---- native step tape -----------------------------------------------------
//
// One C++ call per decoder layer replaces the Python/torch encode of the
// uniform (non-indexer) layer body: mhc_pre -> rms_norm -> attention
// projections -> qnorm/RoPE/KV-insert -> compressor save -> sparse MQA ->
// o_proj -> mhc_post -> mhc_pre -> rms_norm -> MoE -> mhc_post.
//
// Stage-1 contract: BIT-EXACT with the Python path. Every step below calls
// the same quixicore host op or the same aten op, in the same order, with
// the same dtypes/casts, as the Python sites referenced in the comments.
// Do not "optimize" an op here without moving it through the trajectory
// lottery -- sha identity is the gate for this code.
//
// Python side: vllm/models/deepseek_v4/metal_tape.py registers per-layer
// persistent tensors once (lazily, first eligible step) and per-step shared
// products (slot tables built by the same helpers forward_mqa uses), then
// calls qc_tape_layer_forward per covered layer. Uncovered layers/steps
// (indexer layers, c128 boundary steps, tokens > 8) run Python.

struct TapeLayer {
  bool valid = false;
  int64_t kind = 0;  // 0 = compressor-only (c128), 1 = swa-only
  // mhc (amd/model.py:626-666)
  at::Tensor hc_attn_fn, hc_attn_scale, hc_attn_base;
  at::Tensor hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
  // norms
  at::Tensor attn_norm_w, ffn_norm_w;  // fp16
  at::Tensor q_norm_w, kv_norm_w;  // pre-.float()ed (fused_qk_rmsnorm.py:89-90)
  // attention
  at::Tensor wqa_wkv_qw, wq_b_qw, wo_b_qw;  // gguf qweights
  at::Tensor comp_w;  // fused_wkv_wgate.weight [2*coff*512,4096] fp16
  at::Tensor
      ape_bf16;  // pre-cast .to(bf16).contiguous() (save_partial_states.py:45)
  at::Tensor state_cache, swa_kv_cache, comp_kv_cache;
  at::Tensor attn_sink, cos_sin_cache;
  at::Tensor
      wo_a_w;  // wo_a.weight.reshape(8,1024,4096) view (metal.py:120-122)
  // moe
  at::Tensor gate_w;       // [n_experts,4096] fp16
  at::Tensor router_bias;  // e_score_correction_bias [n_experts] f32 (optional)
  at::Tensor hash_table;   // [vocab, top_k] i32 (hash layers, optional)
  at::Tensor w13_qw, w2_qw, sh_gateup_qw, sh_down_qw;
  // scalars
  int64_t wqa_wkv_qt = 0, wq_b_qt = 0, wo_b_qt = 0;
  int64_t w13_qt = 0, w2_qt = 0, sh_gateup_qt = 0, sh_down_qt = 0;
  int64_t w13_row = 0, w2_row = 0;
  bool w13_soa = false, w2_soa = false;  // SoA-repacked expert layouts
  int64_t q_lora = 0, kv_dim = 0, n_heads = 0, head_dim = 0, o_groups = 0;
  int64_t sinkhorn_iters = 0, top_k = 0;
  int64_t state_block_size = 0, state_width = 0, compress_ratio = 0;
  double rms_eps = 0, hc_eps = 0, hc_post_alpha = 0, qk_eps = 0;
  double sm_scale = 0, swiglu_limit = 0, routed_scaling = 0;
  bool renormalize = true;
};

std::array<TapeLayer, 96> g_tape_layers;

// Per-call step tensors. Passed per layer (not per step) because vLLM's KV
// group unification can split identical cache specs into multiple groups
// (uneven page-size tails), so slot mappings and slot tables are a property
// of the LAYER's group, not of the step: sharing one canonical layer's
// tables mis-slotted layer 0's KV insert (found by tape verify 2026-08-11).
struct TapeStepArgs {
  at::Tensor swa_slot_mapping;       // this layer's swa metadata slot_mapping
  at::Tensor swa_slots, swa_lens;    // metal.py builders' products
  at::Tensor comp_slots, comp_lens;  // dense cr128 tables or comp_none pair
  at::Tensor
      comp_state_slot_mapping;    // CompressorMetadata.slot_mapping (kind 0)
  int64_t insert_block_size = 0;  // swa metadata block_size (metal.py:60)
};

at::Tensor tape_dict_tensor(const pybind11::dict& d, const char* key,
                            bool required) {
  if (d.contains(key)) return d[key].cast<at::Tensor>();
  TORCH_CHECK(!required, "qc_tape: missing required tensor '", key, "'");
  return at::Tensor();
}

void qc_tape_register_layer(int64_t idx, const pybind11::dict& tensors,
                            const pybind11::dict& scalars) {
  TORCH_CHECK(idx >= 0 && idx < static_cast<int64_t>(g_tape_layers.size()),
              "qc_tape: layer index out of range: ", idx);
  TapeLayer& L = g_tape_layers[idx];
  L.kind = scalars["kind"].cast<int64_t>();
  L.hc_attn_fn = tape_dict_tensor(tensors, "hc_attn_fn", true);
  L.hc_attn_scale = tape_dict_tensor(tensors, "hc_attn_scale", true);
  L.hc_attn_base = tape_dict_tensor(tensors, "hc_attn_base", true);
  L.hc_ffn_fn = tape_dict_tensor(tensors, "hc_ffn_fn", true);
  L.hc_ffn_scale = tape_dict_tensor(tensors, "hc_ffn_scale", true);
  L.hc_ffn_base = tape_dict_tensor(tensors, "hc_ffn_base", true);
  L.attn_norm_w = tape_dict_tensor(tensors, "attn_norm_w", true);
  L.ffn_norm_w = tape_dict_tensor(tensors, "ffn_norm_w", true);
  L.q_norm_w = tape_dict_tensor(tensors, "q_norm_w", true);
  L.kv_norm_w = tape_dict_tensor(tensors, "kv_norm_w", true);
  L.wqa_wkv_qw = tape_dict_tensor(tensors, "wqa_wkv_qw", true);
  L.wq_b_qw = tape_dict_tensor(tensors, "wq_b_qw", true);
  L.wo_b_qw = tape_dict_tensor(tensors, "wo_b_qw", true);
  L.comp_w = tape_dict_tensor(tensors, "comp_w", L.kind == 0);
  L.ape_bf16 = tape_dict_tensor(tensors, "ape_bf16", L.kind == 0);
  L.state_cache = tape_dict_tensor(tensors, "state_cache", L.kind == 0);
  L.comp_kv_cache = tape_dict_tensor(tensors, "comp_kv_cache", L.kind == 0);
  L.swa_kv_cache = tape_dict_tensor(tensors, "swa_kv_cache", true);
  L.attn_sink = tape_dict_tensor(tensors, "attn_sink", true);
  L.cos_sin_cache = tape_dict_tensor(tensors, "cos_sin_cache", true);
  L.wo_a_w = tape_dict_tensor(tensors, "wo_a_w", true);
  L.gate_w = tape_dict_tensor(tensors, "gate_w", true);
  L.router_bias = tape_dict_tensor(tensors, "router_bias", false);
  L.hash_table = tape_dict_tensor(tensors, "hash_table", false);
  L.w13_qw = tape_dict_tensor(tensors, "w13_qw", true);
  L.w2_qw = tape_dict_tensor(tensors, "w2_qw", true);
  L.sh_gateup_qw = tape_dict_tensor(tensors, "sh_gateup_qw", true);
  L.sh_down_qw = tape_dict_tensor(tensors, "sh_down_qw", true);
  L.wqa_wkv_qt = scalars["wqa_wkv_qt"].cast<int64_t>();
  L.wq_b_qt = scalars["wq_b_qt"].cast<int64_t>();
  L.wo_b_qt = scalars["wo_b_qt"].cast<int64_t>();
  L.w13_qt = scalars["w13_qt"].cast<int64_t>();
  L.w2_qt = scalars["w2_qt"].cast<int64_t>();
  L.sh_gateup_qt = scalars["sh_gateup_qt"].cast<int64_t>();
  L.sh_down_qt = scalars["sh_down_qt"].cast<int64_t>();
  L.w13_row = scalars["w13_row"].cast<int64_t>();
  L.w2_row = scalars["w2_row"].cast<int64_t>();
  L.w13_soa =
      scalars.contains("w13_soa") ? scalars["w13_soa"].cast<bool>() : false;
  L.w2_soa =
      scalars.contains("w2_soa") ? scalars["w2_soa"].cast<bool>() : false;
  L.q_lora = scalars["q_lora"].cast<int64_t>();
  L.kv_dim = scalars["kv_dim"].cast<int64_t>();
  L.n_heads = scalars["n_heads"].cast<int64_t>();
  L.head_dim = scalars["head_dim"].cast<int64_t>();
  L.o_groups = scalars["o_groups"].cast<int64_t>();
  L.sinkhorn_iters = scalars["sinkhorn_iters"].cast<int64_t>();
  L.top_k = scalars["top_k"].cast<int64_t>();
  L.state_block_size = scalars["state_block_size"].cast<int64_t>();
  L.state_width = scalars["state_width"].cast<int64_t>();
  L.compress_ratio = scalars["compress_ratio"].cast<int64_t>();
  L.rms_eps = scalars["rms_eps"].cast<double>();
  L.hc_eps = scalars["hc_eps"].cast<double>();
  L.hc_post_alpha = scalars["hc_post_alpha"].cast<double>();
  L.qk_eps = scalars["qk_eps"].cast<double>();
  L.sm_scale = scalars["sm_scale"].cast<double>();
  L.swiglu_limit = scalars["swiglu_limit"].cast<double>();
  L.routed_scaling = scalars["routed_scaling"].cast<double>();
  L.renormalize = scalars["renormalize"].cast<bool>();
  L.valid = true;
}

at::Tensor qc_tape_layer_forward(int64_t idx, const at::Tensor& x,
                                 const at::Tensor& positions,
                                 const std::optional<at::Tensor>& input_ids,
                                 const pybind11::dict& step,
                                 int64_t insert_block_size) {
  TORCH_CHECK(idx >= 0 && idx < static_cast<int64_t>(g_tape_layers.size()) &&
                  g_tape_layers[idx].valid,
              "qc_tape: layer ", idx, " is not registered");
  const TapeLayer& L = g_tape_layers[idx];
  const int64_t T = x.size(0);
  TapeStepArgs step_args;
  step_args.swa_slot_mapping = tape_dict_tensor(step, "swa_slot_mapping", true);
  step_args.swa_slots = tape_dict_tensor(step, "swa_slots", true);
  step_args.swa_lens = tape_dict_tensor(step, "swa_lens", true);
  step_args.comp_slots = tape_dict_tensor(step, "comp_slots", true);
  step_args.comp_lens = tape_dict_tensor(step, "comp_lens", true);
  step_args.comp_state_slot_mapping =
      tape_dict_tensor(step, "comp_state_slot_mapping", L.kind == 0);
  step_args.insert_block_size = insert_block_size;

  // -- mhc pre, attention side (amd/model.py:1075-1079, mhc.py:325-360) -----
  auto pre = dsv4_mhc_pre(x, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base,
                          L.rms_eps, L.hc_eps, L.hc_eps, L.hc_post_alpha,
                          L.sinkhorn_iters, std::nullopt, 0.0);
  at::Tensor post = std::get<0>(pre);
  at::Tensor comb = std::get<1>(pre);
  at::Tensor h = std::get<2>(pre);

  // -- attn_norm (amd/model.py:1081, quixicore_metal_ops.py:44) -------------
  h = rms_norm(h, L.attn_norm_w, L.rms_eps);

  // -- attention (attention.py:418-470) -------------------------------------
  at::Tensor o_padded = at::empty({T, L.n_heads, L.head_dim}, h.options());
  // projections, serial order (multi_stream_utils.py:99-102):
  // fused_wqa_wkv (gguf/linear.py:214 -> ggml_mul_mat_vec_a8) then kv_score
  // (attention.py:544-548: mm + .float()).
  at::Tensor qr_kv =
      ggml_mul_mat_vec_a8(L.wqa_wkv_qw, h, L.wqa_wkv_qt, L.wqa_wkv_qw.size(0));
  at::Tensor kv_score;
  if (L.kind == 0) kv_score = at::mm(h, L.comp_w.t()).to(at::kFloat);
  // qr/kv split + fused_q_kv_rmsnorm (attention.py:444-451,
  // fused_qk_rmsnorm.py:89-90 -- weights pre-.float()ed at registration).
  auto qk = at::split_with_sizes(qr_kv, {L.q_lora, L.kv_dim}, -1);
  at::Tensor qr = rms_norm(qk[0], L.q_norm_w, L.qk_eps);
  at::Tensor kv = rms_norm(qk[1], L.kv_norm_w, L.qk_eps);
  // wq_b + qnorm/RoPE/KV-insert (attention.py:721-723, metal.py:42-61).
  at::Tensor q =
      ggml_mul_mat_vec_a8(L.wq_b_qw, qr, L.wq_b_qt, L.wq_b_qw.size(0))
          .view({T, L.n_heads, L.head_dim});
  q = deepseek_v4_qnorm_rope_kv_insert(
      q.to(at::kBFloat16).contiguous(), kv.to(at::kBFloat16).contiguous(),
      L.swa_kv_cache, step_args.swa_slot_mapping, positions, L.cos_sin_cache,
      L.qk_eps, step_args.insert_block_size);
  // compressor head (compressor.py:325-395): split + save_partial_states;
  // the c128 tail is skipped off-boundary (boundary steps run Python).
  if (L.kind == 0) {
    const int64_t half = kv_score.size(-1) / 2;
    auto ks = at::split_with_sizes(kv_score, {half, half}, -1);
    const int64_t n_actual = step_args.comp_state_slot_mapping.size(0);
    deepseek_v4_save_partial_states(
        ks[0].narrow(0, 0, n_actual).to(at::kBFloat16),
        ks[1].narrow(0, 0, n_actual).to(at::kBFloat16), L.ape_bf16, positions,
        L.state_cache, step_args.comp_state_slot_mapping, L.state_block_size,
        L.state_width, L.compress_ratio);
  }
  // sparse MQA (metal.py:311-324): c128 layers read the dense compressed
  // table; swa-only layers pass the comp_none placeholder + swa cache.
  at::Tensor att = deepseek_v4_sparse_attention(
      q, L.kind == 0 ? L.comp_kv_cache : L.swa_kv_cache, step_args.comp_slots,
      step_args.comp_lens, L.swa_kv_cache, step_args.swa_slots,
      step_args.swa_lens, L.attn_sink, L.sm_scale);
  o_padded.copy_(att);
  // o_proj (metal.py:80-124): inverse RoPE -> grouped einsum (wo_a dense on
  // Metal) -> wo_b gguf gemv.
  at::Tensor o_flat = dsv4_o_inv_rope(o_padded, positions, L.cos_sin_cache);
  at::Tensor grouped =
      o_flat.view({T, L.o_groups, (L.n_heads / L.o_groups) * L.head_dim});
  at::Tensor z = at::einsum("tgd,grd->tgr", {grouped, L.wo_a_w}).flatten(1);
  at::Tensor attn_out =
      ggml_mul_mat_vec_a8(L.wo_b_qw, z, L.wo_b_qt, L.wo_b_qw.size(0));

  // -- mhc post + pre, ffn side (amd/model.py:1092-1097) --------------------
  at::Tensor x2 = dsv4_mhc_post(attn_out, x, post, comb);
  auto pre2 = dsv4_mhc_pre(x2, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base,
                           L.rms_eps, L.hc_eps, L.hc_eps, L.hc_post_alpha,
                           L.sinkhorn_iters, std::nullopt, 0.0);
  post = std::get<0>(pre2);
  comb = std::get<1>(pre2);
  h = std::get<2>(pre2);

  // -- ffn_norm (amd/model.py:1099) ------------------------------------------
  h = rms_norm(h, L.ffn_norm_w, L.rms_eps);

  // -- MoE (moe_runner.py:946-974, order preserved) --------------------------
  // router linear (gate_linear.py:241-246): F.linear then .to(f32).
  at::Tensor gating = at::linear(h, L.gate_w).to(at::kFloat);
  // shared experts run first (moe_runner.py:609-611; amd/model.py:225-227):
  // gate_up gemv -> qc_swiglu (activation.py:261-282) -> down gemv.
  at::Tensor gu = ggml_mul_mat_vec_a8(L.sh_gateup_qw, h, L.sh_gateup_qt,
                                      L.sh_gateup_qw.size(0));
  at::Tensor sh = at::empty({T, gu.size(1) / 2}, gu.options());
  qc_swiglu(gu, sh, L.swiglu_limit, /*oai_form=*/true, /*alpha=*/1.0,
            /*beta=*/0.0);
  sh =
      ggml_mul_mat_vec_a8(L.sh_down_qw, sh, L.sh_down_qt, L.sh_down_qw.size(0));
  // routing (fused_topk_bias_router.py:305-352, 140-193): topk buffers,
  // pre-softplus, single-dispatch router kernel. Hash layers pass the table
  // + input_ids with bias forced None (fused_topk_bias_router.py:186-192).
  at::Tensor topk_w = at::empty({T, L.top_k}, gating.options());
  at::Tensor topk_ids =
      at::empty({T, L.top_k}, gating.options().dtype(at::kInt));
  at::Tensor scores = at::softplus(gating);
  std::optional<at::Tensor> bias_opt, hash_opt, ids_opt;
  if (L.hash_table.defined()) {
    TORCH_CHECK(input_ids.has_value(),
                "qc_tape: hash-routed layer needs input_ids");
    hash_opt = L.hash_table;
    ids_opt = input_ids->scalar_type() == at::kInt ? *input_ids
                                                   : input_ids->to(at::kInt);
  } else if (L.router_bias.defined()) {
    bias_opt = L.router_bias;
  }
  dsv4_router_topk(scores, topk_w, topk_ids, L.renormalize, L.routed_scaling,
                   bias_opt, hash_opt, ids_opt);
  // routed experts (gguf/fused_moe.py:209, 564-573, 600-621): fused SwiGLU
  // vec kernel, down vec kernel, weighted sum into empty_like(x).
  at::Tensor mo =
      ggml_moe_a8_vec_swiglu(h, L.w13_qw, topk_ids, L.top_k, L.w13_qt,
                             L.w13_row, T, L.swiglu_limit, L.w13_soa);
  // Sum-folded down projection (mirrors the Python route in
  // gguf/fused_moe.py): one kernel instead of down vec + reshape +
  // weighted sum. Shapes outside the folded kernel take the unfused chain.
  at::Tensor fused;
  const bool tape_sum_dtype =
      h.scalar_type() == at::kHalf || h.scalar_type() == at::kBFloat16;
  const int tape_sum_rows = h.scalar_type() == at::kBFloat16 ? 8 : 32;
  if (std::string(ggml_type_to_format(L.w2_qt)) == "q2_K" && tape_sum_dtype &&
      L.top_k <= 8 && L.w2_row % tape_sum_rows == 0) {
    fused = at::empty_like(h);
    ggml_moe_a8_vec_sum(mo, L.w2_qw, topk_ids, topk_w, L.top_k, L.w2_qt,
                        L.w2_row, T, fused, L.w2_soa);
  } else {
    mo = ggml_moe_a8_vec(mo, L.w2_qw, topk_ids, 1, L.w2_qt, L.w2_row,
                         T * L.top_k, L.w2_soa);
    mo = mo.reshape({T, L.top_k, L.w2_row});
    fused = at::empty_like(h);
    moe_weighted_sum(mo, topk_w, fused);
  }
  // shared + routed (moe_runner.py:796).
  at::Tensor moe_out = sh + fused;

  // -- mhc post, close the layer (amd/model.py:1109) -------------------------
  return dsv4_mhc_post(moe_out, x2, post, comb);
}

// Small-M weight-streaming MMA GEMM: every warp computes all (padded-to-32)
// columns for its own 16 weight rows, so no streamed weight byte feeds idle
// padding lanes. The verify/draft band (2..32 rows) routes here.
at::Tensor ggml_mul_mat_sm(const at::Tensor& w, const at::Tensor& x,
                           int64_t quant_type, int64_t row, int64_t n_warps) {
  check_mps(w, "w");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(-1));
  const int output_rows = static_cast<int>(x.size(0));
  const std::string fmt = ggml_type_to_format(quant_type);
  constexpr int kMPad = 32;

  TORCH_CHECK(output_rows <= kMPad,
              "quixicore(metal): qgemm_sm covers <= 32 rows, got ",
              output_rows);
  TORCH_CHECK((n_warps >= 2 && n_warps <= 17) || n_warps == 31,
              "quixicore(metal): unknown qgemm_sm variant ", n_warps);
  TORCH_CHECK(N % tk::qgemm_sm_tg_rows(static_cast<int>(n_warps)) == 0,
              "quixicore(metal): qgemm_sm needs N % ",
              tk::qgemm_sm_tg_rows(static_cast<int>(n_warps)),
              " == 0, got N=", N);
  TORCH_CHECK(K % 32 == 0,
              "quixicore(metal): qgemm_sm needs K % 32 == 0, got K=", K);
  TORCH_CHECK(fmt == "q4_0" || fmt == "q8_0" || fmt == "q4_K" ||
                  fmt == "q5_K" || fmt == "q6_K",
              "quixicore(metal): qgemm_sm unsupported format ", fmt);

  auto input = x.to(at::kHalf).transpose(0, 1).contiguous();
  auto input_padded = at::zeros({K, kMPad}, x.options().dtype(at::kHalf));
  input_padded.narrow(1, 0, output_rows).copy_(input);
  auto output_padded = at::empty({N, kMPad}, x.options().dtype(at::kHalf));

  const int variant = static_cast<int>(n_warps);
  const int split_k = tk::qgemm_sm_split_k(variant);
  if (variant == 31) {
    // split-K=1 tensor kernel: one float slice + SK=1 reduce (the cast)
    auto partials = at::empty({1, N, kMPad}, x.options().dtype(at::kFloat));
    encode([&](TorchEncoder& e) {
      tk::launch_qgemm_sm(e, partials, w, input_padded, N, K, variant, fmt);
      tk::launch_qgemm_sm_reduce(e, output_padded, partials, N, 1);
    });
  } else if (split_k == 1) {
    encode([&](TorchEncoder& e) {
      tk::launch_qgemm_sm(e, output_padded, w, input_padded, N, K, variant,
                          fmt);
    });
  } else {
    auto partials =
        at::empty({split_k, N, kMPad}, x.options().dtype(at::kFloat));
    encode([&](TorchEncoder& e) {
      tk::launch_qgemm_sm(e, partials, w, input_padded, N, K, variant, fmt);
      tk::launch_qgemm_sm_reduce(e, output_padded, partials, N, split_k);
    });
  }
  return output_padded.narrow(1, 0, output_rows)
      .transpose(0, 1)
      .to(x.scalar_type())
      .contiguous();
}

// uint4-native q4_K GEMM: weights repacked to tile-major packed uint4 with
// per-32-group scale/min half planes (see qgemm_sm_u4 kernel notes). X is
// (32, K) half row-major (M rows used, rest zero-padded). Returns (N, 32)
// half; deterministic split-K reduce.
at::Tensor ggml_mul_mat_sm_u4(const at::Tensor& wu, const at::Tensor& x,
                              const at::Tensor& sc, const at::Tensor& mn,
                              int64_t row) {
  check_mps(wu, "wu");
  check_mps(x, "x");
  check_mps(sc, "sc");
  check_mps(mn, "mn");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(1));
  TORCH_CHECK(
      x.scalar_type() == at::kHalf && x.size(0) == 32 && x.is_contiguous(),
      "quixicore(metal): qgemm_sm_u4 wants contiguous (32, K) half X");
  TORCH_CHECK(N % 64 == 0 && K % 128 == 0,
              "quixicore(metal): qgemm_sm_u4 needs N % 64 == 0, K % 128 == 0");
  // every partials element is written by the scatter store (coop capacity
  // x threads == tile size, all elements valid -- parity-verified), so no
  // zero-fill is needed
  auto xs = at::empty({K / 32, 32}, x.options());
  auto partials = at::empty({4, N, 32}, x.options().dtype(at::kFloat));
  auto out = at::empty({N, 32}, x.options());
  encode([&](TorchEncoder& e) {
    tk::launch_qgemm_sm_u4_xsum(e, xs, x, K);
    tk::launch_qgemm_sm_u4(e, partials, wu, x, sc, mn, xs, N, K);
    tk::launch_qgemm_sm_reduce(e, out, partials, N, 4);
  });
  return out;
}

// Glue-free u4: takes the original (M, K) bf16 activations, returns
// (M, N) bf16 via the transposed-store reduce. No pad/cast/transpose ops
// on either side.
at::Tensor ggml_mul_mat_sm_u4_rm(const at::Tensor& wu, const at::Tensor& x,
                                 const at::Tensor& sc, const at::Tensor& mn,
                                 int64_t row) {
  check_mps(wu, "wu");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(1));
  const int M = static_cast<int>(x.size(0));
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "quixicore(metal): qgemm_sm_u4b wants contiguous (M, K) bf16");
  TORCH_CHECK(M <= 32 && N % 64 == 0 && K % 128 == 0,
              "quixicore(metal): qgemm_sm_u4b needs M <= 32, N % 64 == 0, "
              "K % 128 == 0");
  auto xs = at::empty({K / 32, 32}, x.options().dtype(at::kHalf));
  auto partials = at::empty({4, N, 32}, x.options().dtype(at::kFloat));
  auto out = at::empty({M, N}, x.options());
  encode([&](TorchEncoder& e) {
    tk::launch_qgemm_sm_u4b_xsum(e, xs, x, K, M);
    tk::launch_qgemm_sm_u4b(e, partials, wu, x, sc, mn, xs, N, K, M);
    tk::launch_qgemm_sm_reduce_rm(e, out, partials, N, M);
  });
  return out;
}

// int8-native q6_K GEMM: (K, N) row-major int8 weights (the -32 folded at
// repack) + (K/16, N) half scale plane. X is (32, K) half row-major.
at::Tensor ggml_mul_mat_sm_u8(const at::Tensor& wq8, const at::Tensor& x,
                              const at::Tensor& sc, int64_t row) {
  check_mps(wq8, "wq8");
  check_mps(x, "x");
  check_mps(sc, "sc");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(1));
  TORCH_CHECK(
      x.scalar_type() == at::kHalf && x.size(0) == 32 && x.is_contiguous(),
      "quixicore(metal): qgemm_sm_u8 wants contiguous (32, K) half X");
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0,
              "quixicore(metal): qgemm_sm_u8 needs N % 64 == 0, K % 64 == 0");
  auto partials = at::empty({4, N, 32}, x.options().dtype(at::kFloat));
  auto out = at::empty({N, 32}, x.options());
  encode([&](TorchEncoder& e) {
    tk::launch_qgemm_sm_u8(e, partials, wq8, x, sc, N, K);
    tk::launch_qgemm_sm_reduce(e, out, partials, N, 4);
  });
  return out;
}

// Native DFlash input prep: one dispatch replaces the MPS Python loop
// (two GPU->CPU syncs + ~25 small ops per propose). Mirrors the Triton
// kernel; argument order matches the python-side native call exactly.
void prepare_dflash_inputs(
    const at::Tensor& input_ids, const at::Tensor& positions,
    const at::Tensor& query_start_loc, const at::Tensor& seq_lens,
    const at::Tensor& query_slot_mapping, const at::Tensor& context_positions,
    const at::Tensor& context_slot_mapping, const at::Tensor& sample_indices,
    const at::Tensor& sample_pos, const at::Tensor& sample_idx_mapping,
    const at::Tensor& target_positions, const at::Tensor& target_qsl,
    const at::Tensor& idx_mapping, const at::Tensor& last_sampled,
    const at::Tensor& next_prefill_tokens, const at::Tensor& num_sampled,
    const at::Tensor& num_rejected, const at::Tensor& block_table,
    int64_t bt_stride, int64_t parallel_drafting_token_id, int64_t block_size,
    int64_t num_query_per_req, int64_t num_speculative_steps,
    int64_t max_num_reqs, int64_t max_num_tokens, int64_t max_model_len,
    bool sample_from_anchor, int64_t pad_slot_id, int64_t num_reqs,
    int64_t max_tokens_per_req) {
  TORCH_CHECK(input_ids.scalar_type() == at::kInt &&
                  query_start_loc.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt &&
                  sample_idx_mapping.scalar_type() == at::kInt &&
                  idx_mapping.scalar_type() == at::kInt &&
                  next_prefill_tokens.scalar_type() == at::kInt &&
                  num_sampled.scalar_type() == at::kInt &&
                  num_rejected.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt,
              "quixicore(metal): prepare_dflash_inputs int32 dtype mismatch");
  TORCH_CHECK(positions.scalar_type() == at::kLong &&
                  query_slot_mapping.scalar_type() == at::kLong &&
                  context_positions.scalar_type() == at::kLong &&
                  context_slot_mapping.scalar_type() == at::kLong &&
                  sample_indices.scalar_type() == at::kLong &&
                  sample_pos.scalar_type() == at::kLong &&
                  target_positions.scalar_type() == at::kLong &&
                  last_sampled.scalar_type() == at::kLong,
              "quixicore(metal): prepare_dflash_inputs int64 dtype mismatch");
  const int grid_x = static_cast<int>((max_tokens_per_req + 255) / 256);
  encode([&](TorchEncoder& e) {
    e.pipeline("mittens::prepare_dflash_inputs");
    e.out(input_ids, 0);
    e.out(positions, 1);
    e.out(query_start_loc, 2);
    e.out(seq_lens, 3);
    e.out(query_slot_mapping, 4);
    e.out(context_positions, 5);
    e.out(context_slot_mapping, 6);
    e.out(sample_indices, 7);
    e.out(sample_pos, 8);
    e.out(sample_idx_mapping, 9);
    e.in(target_positions, 10);
    e.in(target_qsl, 11);
    e.in(idx_mapping, 12);
    e.in(last_sampled, 13);
    e.in(next_prefill_tokens, 14);
    e.in(num_sampled, 15);
    e.in(num_rejected, 16);
    e.in(block_table, 17);
    const int v_bt = static_cast<int>(bt_stride);
    const int v_pdt = static_cast<int>(parallel_drafting_token_id);
    const int v_bs = static_cast<int>(block_size);
    const int v_nq = static_cast<int>(num_query_per_req);
    const int v_ns = static_cast<int>(num_speculative_steps);
    const int v_mr = static_cast<int>(max_num_reqs);
    const int v_mt = static_cast<int>(max_num_tokens);
    const int v_ml = static_cast<int>(max_model_len);
    const int v_sa = sample_from_anchor ? 1 : 0;
    const int v_ps = static_cast<int>(pad_slot_id);
    const int v_nr = static_cast<int>(num_reqs);
    e.bytes(v_bt, 18);
    e.bytes(v_pdt, 19);
    e.bytes(v_bs, 20);
    e.bytes(v_nq, 21);
    e.bytes(v_ns, 22);
    e.bytes(v_mr, 23);
    e.bytes(v_mt, 24);
    e.bytes(v_ml, 25);
    e.bytes(v_sa, 26);
    e.bytes(v_ps, 27);
    e.bytes(v_nr, 28);
    e.dispatch(grid_x, static_cast<int>(num_reqs), 1, 256, 1, 1);
  });
}

// Layout-native qgemm_sm: X already (K, 32) half, D returned (N, 32) half.
// No transpose/pad/cast glue -- the shape the fused muse_step verify uses,
// and the honest way to benchmark the kernel itself.
at::Tensor ggml_mul_mat_sm_pre(const at::Tensor& w, const at::Tensor& x,
                               int64_t quant_type, int64_t row,
                               int64_t n_warps) {
  check_mps(w, "w");
  check_mps(x, "x");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(0));
  const std::string fmt = ggml_type_to_format(quant_type);
  TORCH_CHECK(
      x.scalar_type() == at::kHalf && x.size(1) == 32 && x.is_contiguous(),
      "quixicore(metal): qgemm_sm_pre wants contiguous (K, 32) half");
  auto out = at::empty({N, 32}, x.options());
  const int variant = static_cast<int>(n_warps);
  const int split_k = tk::qgemm_sm_split_k(variant);
  if (variant == 31) {
    // split-K=1 tensor kernel: one float slice + SK=1 reduce (the cast)
    auto partials = at::empty({1, N, 32}, x.options().dtype(at::kFloat));
    encode([&](TorchEncoder& e) {
      tk::launch_qgemm_sm(e, partials, w, x, N, K, variant, fmt);
      tk::launch_qgemm_sm_reduce(e, out, partials, N, 1);
    });
  } else if (split_k == 1) {
    encode([&](TorchEncoder& e) {
      tk::launch_qgemm_sm(e, out, w, x, N, K, variant, fmt);
    });
  } else {
    auto partials = at::empty({split_k, N, 32}, x.options().dtype(at::kFloat));
    encode([&](TorchEncoder& e) {
      tk::launch_qgemm_sm(e, partials, w, x, N, K, variant, fmt);
      tk::launch_qgemm_sm_reduce(e, out, partials, N, split_k);
    });
  }
  return out;
}

// DFlash 2 dynamic block-local two-tap convolution.  One dispatch replaces
// repeat_interleave + roll + clone/zero + the elementwise multiply/add chain.
at::Tensor dflash2_two_tap_conv(const at::Tensor& x, const at::Tensor& coeffs,
                                const at::Tensor& base, int64_t side,
                                int64_t block_size, int64_t group_size) {
  check_mps(x, "x");
  check_mps(coeffs, "coeffs");
  check_mps(base, "base");
  TORCH_CHECK(x.dim() == 2,
              "quixicore(metal): dflash2 conv x must be (tokens, hidden)");
  TORCH_CHECK(
      coeffs.dim() == 4 && coeffs.size(1) == 2 && coeffs.size(2) == 2 &&
          coeffs.size(0) == x.size(0),
      "quixicore(metal): dflash2 coeffs must be (tokens, 2, 2, groups)");
  TORCH_CHECK(base.dim() == 3 && base.size(0) == 2 && base.size(1) == 2 &&
                  base.size(2) == x.size(1),
              "quixicore(metal): dflash2 base must be (2, 2, hidden)");
  TORCH_CHECK(x.scalar_type() == coeffs.scalar_type() &&
                  x.scalar_type() == base.scalar_type(),
              "quixicore(metal): dflash2 x/coeffs/base dtype mismatch");
  TORCH_CHECK(side == 0 || side == 1,
              "quixicore(metal): dflash2 side must be 0 or 1");
  TORCH_CHECK(group_size > 0 && x.size(1) % group_size == 0,
              "quixicore(metal): dflash2 hidden must divide by group_size");

  const int tokens = static_cast<int>(x.size(0));
  const int hidden = static_cast<int>(x.size(1));
  const int groups = static_cast<int>(coeffs.size(3));
  TORCH_CHECK(groups * group_size == hidden,
              "quixicore(metal): dflash2 group geometry mismatch");
  auto out = at::empty_like(x);
  if (tokens == 0) return out;
  const int bs = block_size > 0 && tokens % block_size == 0
                     ? static_cast<int>(block_size)
                     : tokens;
  const int gs = static_cast<int>(group_size);
  const int sd = static_cast<int>(side);
  const int total = tokens * hidden;
  const std::string act = activation_type_name(x);
  encode([&](TorchEncoder& e) {
    e.pipeline("dflash2_two_tap_conv_" + act);
    e.in(x, 0);
    e.in(coeffs, 1);
    e.in(base, 2);
    e.out(out, 3);
    e.bytes(tokens, 4);
    e.bytes(hidden, 5);
    e.bytes(groups, 6);
    e.bytes(gs, 7);
    e.bytes(bs, 8);
    e.bytes(sd, 9);
    e.dispatch((total + 255) / 256, 1, 1, 256, 1, 1);
  });
  return out;
}

std::tuple<at::Tensor, at::Tensor> qwen38_rejection_sample(
    const at::Tensor& target, const std::optional<at::Tensor>& draft,
    const at::Tensor& draft_sampled, const at::Tensor& cu,
    const at::Tensor& pos, const at::Tensor& idx_mapping,
    const at::Tensor& temperature, const at::Tensor& seeds,
    int64_t num_speculative_steps, int64_t vocab_size) {
  check_mps(target, "target");
  check_mps(draft_sampled, "draft_sampled");
  check_mps(cu, "cu");
  check_mps(pos, "pos");
  check_mps(idx_mapping, "idx_mapping");
  check_mps(temperature, "temperature");
  check_mps(seeds, "seeds");
  TORCH_CHECK(target.dim() == 2 && target.scalar_type() == at::kFloat,
              "quixicore(metal): fused rejection target must be fp32 (L,V)");
  TORCH_CHECK(draft_sampled.scalar_type() == at::kInt &&
                  cu.scalar_type() == at::kInt &&
                  idx_mapping.scalar_type() == at::kInt,
              "quixicore(metal): fused rejection ids/cu/mapping must be int32");
  TORCH_CHECK(
      pos.scalar_type() == at::kLong && seeds.scalar_type() == at::kLong,
      "quixicore(metal): fused rejection pos/seeds must be int64");
  TORCH_CHECK(temperature.scalar_type() == at::kFloat,
              "quixicore(metal): fused rejection temperature must be fp32");
  TORCH_CHECK(num_speculative_steps >= 1 && num_speculative_steps <= 16,
              "quixicore(metal): fused rejection supports 1..16 steps");
  TORCH_CHECK(vocab_size > 0 && vocab_size <= target.size(1),
              "quixicore(metal): fused rejection invalid vocab size");
  const bool has_draft = draft.has_value();
  if (has_draft) {
    check_mps(*draft, "draft");
    TORCH_CHECK(draft->dim() == 3 && draft->scalar_type() == at::kFloat &&
                    draft->size(1) >= num_speculative_steps &&
                    draft->size(2) >= vocab_size,
                "quixicore(metal): fused rejection draft must be fp32 "
                "(states, >=steps, >=vocab)");
  }
  const int requests = static_cast<int>(cu.numel() - 1);
  TORCH_CHECK(requests >= 0 && idx_mapping.numel() >= requests,
              "quixicore(metal): fused rejection request metadata mismatch");
  auto sampled = at::empty({requests, num_speculative_steps + 1},
                           target.options().dtype(at::kLong));
  auto num_sampled = at::empty({requests}, target.options().dtype(at::kInt));
  if (requests == 0) return {sampled, num_sampled};
  const int steps = static_cast<int>(num_speculative_steps);
  const int vocab = static_cast<int>(vocab_size);
  const long target_stride = static_cast<long>(target.stride(0));
  const long draft_stride0 =
      has_draft ? static_cast<long>(draft->stride(0)) : 0;
  const long draft_stride1 =
      has_draft ? static_cast<long>(draft->stride(1)) : 0;
  const int has_draft_i = has_draft ? 1 : 0;
  encode([&](TorchEncoder& e) {
    e.pipeline("mittens::qwen38_rejection_sample");
    e.out(sampled, 0);
    e.out(num_sampled, 1);
    e.in(target, 2);
    e.in(has_draft ? *draft : target, 3);
    e.in(draft_sampled, 4);
    e.in(cu, 5);
    e.in(pos, 6);
    e.in(idx_mapping, 7);
    e.in(temperature, 8);
    e.in(seeds, 9);
    e.bytes(requests, 10);
    e.bytes(steps, 11);
    e.bytes(vocab, 12);
    e.bytes(target_stride, 13);
    e.bytes(draft_stride0, 14);
    e.bytes(draft_stride1, 15);
    e.bytes(has_draft_i, 16);
    e.dispatch(requests, 1, 1, 256, 1, 1);
  });
  return {sampled, num_sampled};
}

// ---- Qwen3.5 fused GDN decode / verify step -------------------------------
//
// Two dispatches per layer (conv window update + delta-rule scan) replacing
// the per-position torch-native MPS loop. Kernel contract and layout notes in
// kernels/serving_glue/gdn_step.metal; the torch path in
// qwen_gdn_linear_attn.py stays the oracle and the fallback.

void qwen_gdn_step(const at::Tensor& x, const at::Tensor& a,
                   const at::Tensor& b, const at::Tensor& conv_state,
                   const at::Tensor& ssm_state, const at::Tensor& conv_weight,
                   const std::optional<at::Tensor>& conv_bias,
                   const at::Tensor& A_log, const at::Tensor& dt_bias,
                   const at::Tensor& token_map, const at::Tensor& conv_slot,
                   const at::Tensor& resume_slot, const at::Tensor& store_slots,
                   const std::optional<at::Tensor>& num_accepted,
                   const at::Tensor& out, int64_t num_seqs, int64_t S,
                   int64_t num_k_heads, bool tiled, bool act_silu,
                   double scale) {
  check_mps_strided(x, "x");
  check_mps_strided(a, "a");
  check_mps_strided(b, "b");
  check_mps_strided(conv_state, "conv_state");
  check_mps_strided(ssm_state, "ssm_state");
  check_mps(conv_weight, "conv_weight");
  check_mps(A_log, "A_log");
  check_mps(dt_bias, "dt_bias");
  check_mps(token_map, "token_map");
  check_mps(conv_slot, "conv_slot");
  check_mps(resume_slot, "resume_slot");
  check_mps_strided(store_slots, "store_slots");
  check_mps_strided(out, "out");

  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1,
              "quixicore(metal): qwen_gdn_step x must be (tokens, conv_dim) "
              "with unit inner stride");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.stride(1) == 1 &&
                  b.stride(1) == 1 && a.stride(0) == b.stride(0),
              "quixicore(metal): qwen_gdn_step a/b must be (tokens, Hv) views "
              "with unit inner stride and a common row stride");
  TORCH_CHECK(
      conv_state.dim() == 3,
      "quixicore(metal): conv_state must be a (slots, conv_dim, L) view");
  TORCH_CHECK(ssm_state.dim() == 4 && ssm_state.scalar_type() == at::kFloat &&
                  ssm_state.stride(3) == 1 &&
                  ssm_state.stride(2) == ssm_state.size(3) &&
                  ssm_state.stride(1) == ssm_state.size(2) * ssm_state.size(3),
              "quixicore(metal): ssm_state must be fp32 (slots, Hv, Dv, Dk) "
              "with a contiguous per-slot block");
  TORCH_CHECK(
      out.dim() == 3 && out.stride(2) == 1 && out.stride(1) == out.size(2),
      "quixicore(metal): out must be (tokens, Hv, Dv) rows");
  TORCH_CHECK(conv_weight.dim() == 2 && conv_weight.scalar_type() == at::kFloat,
              "quixicore(metal): conv_weight must be fp32 (conv_dim, width)");
  TORCH_CHECK(
      A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat,
      "quixicore(metal): A_log/dt_bias must be fp32");
  TORCH_CHECK(token_map.scalar_type() == at::kInt &&
                  conv_slot.scalar_type() == at::kInt &&
                  resume_slot.scalar_type() == at::kInt &&
                  store_slots.scalar_type() == at::kInt,
              "quixicore(metal): qwen_gdn_step index tensors must be int32");
  TORCH_CHECK(x.scalar_type() == a.scalar_type() &&
                  x.scalar_type() == b.scalar_type() &&
                  x.scalar_type() == out.scalar_type(),
              "quixicore(metal): x/a/b/out must share the activation dtype");

  const int conv_dim = static_cast<int>(x.size(1));
  const int width = static_cast<int>(conv_weight.size(1));
  const int Hv = static_cast<int>(ssm_state.size(1));
  const int Dv = static_cast<int>(ssm_state.size(2));
  const int Dk = static_cast<int>(ssm_state.size(3));
  const int Hk = static_cast<int>(num_k_heads);
  const int N = static_cast<int>(num_seqs);
  const int Sv = static_cast<int>(S);
  TORCH_CHECK(Dk == 128,
              "quixicore(metal): qwen_gdn_step wants head_k_dim 128");
  TORCH_CHECK(width >= 2 && width <= 4,
              "quixicore(metal): qwen_gdn_step wants conv width in [2, 4]");
  TORCH_CHECK(Sv >= 1 && Sv <= 16,
              "quixicore(metal): qwen_gdn_step wants 1 <= S <= 16");
  TORCH_CHECK(Dv % 8 == 0,
              "quixicore(metal): head_v_dim must be a multiple of 8");
  TORCH_CHECK(conv_dim == 2 * Hk * Dk + Hv * Dv,
              "quixicore(metal): conv_dim mismatch: ", conv_dim, " vs ",
              2 * Hk * Dk + Hv * Dv);
  TORCH_CHECK(conv_state.size(1) == conv_dim,
              "quixicore(metal): conv_state channel dim mismatch");
  TORCH_CHECK(conv_state.size(2) >= width - 1 + Sv - 1,
              "quixicore(metal): conv_state too narrow for S=", Sv);
  TORCH_CHECK(token_map.numel() >= static_cast<int64_t>(N) * Sv,
              "quixicore(metal): token_map too short");
  TORCH_CHECK(conv_slot.numel() >= N && resume_slot.numel() >= N,
              "quixicore(metal): slot tensors too short");
  TORCH_CHECK(store_slots.dim() == 2 && store_slots.size(0) >= N &&
                  store_slots.size(1) >= Sv && store_slots.stride(1) == 1,
              "quixicore(metal): store_slots must be (N, >= S) rows");
  TORCH_CHECK(a.size(1) >= Hv && b.size(1) >= Hv,
              "quixicore(metal): a/b narrower than Hv");
  const bool has_bias = conv_bias.has_value();
  if (has_bias) check_mps(*conv_bias, "conv_bias");
  const bool use_acc = num_accepted.has_value();
  if (use_acc) {
    check_mps(*num_accepted, "num_accepted");
    TORCH_CHECK(num_accepted->scalar_type() == at::kInt,
                "quixicore(metal): num_accepted must be int32");
  }
  if (N == 0) return;

  const std::string act = activation_type_name(x);
  const std::string cs = activation_type_name(conv_state);
  TORCH_CHECK(cs == act || cs == "float32",
              "quixicore(metal): conv_state dtype must match x or be fp32");

  auto conved = at::empty({N, Sv, conv_dim}, x.options().dtype(at::kFloat));

  const long x_rs = static_cast<long>(x.stride(0));
  const long cs_slot = static_cast<long>(conv_state.stride(0));
  const long cs_chan = static_cast<long>(conv_state.stride(1));
  const long cs_col = static_cast<long>(conv_state.stride(2));
  const long ab_rs = static_cast<long>(a.stride(0));
  const long st_slot = static_cast<long>(ssm_state.stride(0));
  const long ss_rs = static_cast<long>(store_slots.stride(0));
  const long out_rs = static_cast<long>(out.stride(0));
  const int v_has_bias = has_bias ? 1 : 0;
  const int v_act = act_silu ? 1 : 0;
  const int v_use_acc = use_acc ? 1 : 0;
  const int v_tiled = tiled ? 1 : 0;
  const float v_scale = static_cast<float>(scale);
  const int conv_grid = (conv_dim + 255) / 256;
  const int scan_grid = Dv / 8;

  encode([&](TorchEncoder& e) {
    e.pipeline("qwen_gdn_conv_step_" + act + "_cs" + cs);
    e.in(x, 0);
    e.out(conv_state, 1);
    e.in(conv_weight, 2);
    e.in(has_bias ? *conv_bias : conv_weight, 3);
    e.out(conved, 4);
    e.in(token_map, 5);
    e.in(conv_slot, 6);
    e.in(use_acc ? *num_accepted : conv_slot, 7);
    e.bytes(conv_dim, 8);
    e.bytes(width, 9);
    e.bytes(Sv, 10);
    e.bytes(x_rs, 11);
    e.bytes(cs_slot, 12);
    e.bytes(cs_chan, 13);
    e.bytes(cs_col, 14);
    e.bytes(v_has_bias, 15);
    e.bytes(v_act, 16);
    e.bytes(v_use_acc, 17);
    e.dispatch(conv_grid, N, 1, 256, 1, 1);

    e.pipeline("qwen_gdn_scan_step_" + act);
    e.in(conved, 0);
    e.in(a, 1);
    e.in(b, 2);
    e.in(A_log, 3);
    e.in(dt_bias, 4);
    e.out(ssm_state, 5);
    e.in(token_map, 6);
    e.in(resume_slot, 7);
    e.in(store_slots, 8);
    e.out(out, 9);
    e.bytes(Sv, 10);
    e.bytes(Hk, 11);
    e.bytes(Hv, 12);
    e.bytes(Dv, 13);
    e.bytes(conv_dim, 14);
    e.bytes(v_tiled, 15);
    e.bytes(ab_rs, 16);
    e.bytes(st_slot, 17);
    e.bytes(ss_rs, 18);
    e.bytes(out_rs, 19);
    e.bytes(v_scale, 20);
    e.dispatch(scan_grid, Hv, N, 256, 1, 1);
  });
}

// Gated RMS norm over the GDN core output (RMSNormGated norm_before_gate +
// silu gate), one simdgroup per (token, head) row.
at::Tensor qwen_gdn_gated_norm(const at::Tensor& x, const at::Tensor& z,
                               const at::Tensor& w, double eps) {
  check_mps(x, "x");
  check_mps_strided(z, "z");
  check_mps(w, "w");
  TORCH_CHECK(x.dim() == 3 && z.dim() == 3 && x.sizes() == z.sizes(),
              "quixicore(metal): gated_norm wants x/z as (tokens, Hv, D)");
  const int tokens = static_cast<int>(x.size(0));
  const int Hv = static_cast<int>(x.size(1));
  const int D = static_cast<int>(x.size(2));
  TORCH_CHECK(D % 4 == 0 && D >= 4, "quixicore(metal): gated_norm D % 4 == 0");
  TORCH_CHECK(z.stride(2) == 1 && z.stride(1) == D,
              "quixicore(metal): gated_norm z must be head-contiguous");
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.numel() == D,
              "quixicore(metal): gated_norm weight must be fp32 (D)");
  TORCH_CHECK(x.scalar_type() == z.scalar_type(),
              "quixicore(metal): gated_norm x/z dtype mismatch");
  auto out = at::empty_like(x);
  const int rows = tokens * Hv;
  if (rows == 0) return out;
  const long z_ts = static_cast<long>(z.stride(0));
  const float v_eps = static_cast<float>(eps);
  const std::string act = activation_type_name(x);
  encode([&](TorchEncoder& e) {
    e.pipeline("qwen_gdn_gated_norm_" + act);
    e.in(x, 0);
    e.in(z, 1);
    e.in(w, 2);
    e.out(out, 3);
    e.bytes(rows, 4);
    e.bytes(Hv, 5);
    e.bytes(D, 6);
    e.bytes(z_ts, 7);
    e.bytes(v_eps, 8);
    e.dispatch((rows + 7) / 8, 1, 1, 256, 1, 1);
  });
  return out;
}

// Full-tensor dequant to fp16 through the same tk_dequant8 span decoders the
// GEMV/GEMM kernels use (decoder verification and load-time dequant).
at::Tensor ggml_dequantize_fp16(const at::Tensor& w, int64_t quant_type,
                                int64_t row, int64_t k) {
  check_mps(w, "w");
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(k);
  TORCH_CHECK(K % 8 == 0, "quixicore(metal): dequant wants K % 8 == 0");
  const std::string fmt = ggml_type_to_format(quant_type);
  auto out = at::empty({N, K}, w.options().dtype(at::kHalf));
  encode(
      [&](TorchEncoder& e) { tk::launch_qdequant_fp16(e, out, w, N, K, fmt); });
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "QuixiCore Metal kernels for the Apple Silicon serving path";

  m.def("_set_library", &set_library,
        "Point the extension at the compiled quixicore_metal.metallib",
        pybind11::arg("path"));

  m.def("cb_census_install", &cb_census_install,
        "Diagnostic: hook the MPS command queue to count command buffers "
        "and accumulate GPU busy time. Idempotent; call from the thread "
        "that owns the MPS stream.");
  m.def("cb_census_read", &cb_census_read,
        "Diagnostic: (created, completed, gpu_busy_seconds) since install.");
  m.def("residency_pin", &residency_pin,
        "Pin the given MPS tensors' allocations (heap-granular) into an "
        "MTLResidencySet attached to torch's command queue. Returns "
        "(allocations_added, bytes_pinned); (0, 0) before macOS 15.",
        pybind11::arg("tensors"));

  m.def("mla_decode_fp8_sparse", &mla_decode_fp8_sparse,
        "Top-k sparse MQA decode over the paged fp8 MLA latent",
        pybind11::arg("q"), pybind11::arg("kv_data"), pybind11::arg("kv_scale"),
        pybind11::arg("block_table"), pybind11::arg("topk_indices"),
        pybind11::arg("topk_length"), pybind11::arg("block_size"),
        pybind11::arg("sm_scale"), pybind11::arg("partition_size") = 0);

  m.def("muse_step_init", &muse_step_init,
        "Register geometry and allocate scratch for the fused decode step",
        pybind11::arg("num_layers"), pybind11::arg("hidden"),
        pybind11::arg("heads"), pybind11::arg("kv_heads"),
        pybind11::arg("head_dim"), pybind11::arg("inter"),
        pybind11::arg("window"), pybind11::arg("theta"), pybind11::arg("eps"),
        pybind11::arg("post_eps"), pybind11::arg("max_rows"),
        pybind11::arg("ref"));
  m.def("muse_step_layer", &muse_step_layer,
        "Register one decoder layer's weights for the fused decode step");
  m.def(
      "dflash_step_debug",
      []() {
        using namespace dflash_step;
        return std::vector<at::Tensor>{d.h,        d.q,     d.k,     d.v,
                                       d.attn_out, d.o_out, d.g_out, d.u_out,
                                       d.mlp_mid,  d.mlp_h};
      },
      "fused drafter scratch buffers (values from the LAST emitted layer)");
  m.def("paged_attention_verify", &paged_attention_verify,
        "multi-query verify attention: m rows share each K/V read",
        pybind11::arg("q"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_table"),
        pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("window"));
  m.def("dflash_sample_greedy", &dflash_sample_greedy,
        "fused greedy draft sampling: shared lm_head GEMM + row argmax",
        pybind11::arg("hidden"), pybind11::arg("lm_w"),
        pybind11::arg("lm_type"), pybind11::arg("vocab_rows"));
  m.def("dflash_step_init", &dflash_step_init,
        "register fused DFlash drafter geometry and scratch");
  m.def("dflash_step_layer", &dflash_step_layer,
        "register one fused DFlash drafter layer");
  m.def("dflash_step_run", &dflash_step_run,
        "single-command-buffer DFlash drafter forward (block-bidirectional)");
  m.def("muse_step_run_aux", &muse_step_run_aux,
        "Fused verify step: muse_step_run plus residual-stream snapshots "
        "entering each aux layer (for the DFlash drafter)",
        pybind11::arg("x"), pybind11::arg("positions"),
        pybind11::arg("bt_local"), pybind11::arg("sl_local"),
        pybind11::arg("slot_local"), pybind11::arg("bt_full"),
        pybind11::arg("sl_full"), pybind11::arg("slot_full"),
        pybind11::arg("aux_out"), pybind11::arg("aux_layers"),
        pybind11::arg("ctx_len") = 0);
  m.def("muse_step_run", &muse_step_run,
        "Encode the whole decoder stack for one decode step into a single "
        "command buffer; x is updated in place",
        pybind11::arg("x"), pybind11::arg("positions"),
        pybind11::arg("bt_local"), pybind11::arg("sl_local"),
        pybind11::arg("slot_local"), pybind11::arg("bt_full"),
        pybind11::arg("sl_full"), pybind11::arg("slot_full"),
        pybind11::arg("ctx_len") = 0);

  m.def("paged_attention", &paged_attention,
        "Dense/GQA paged attention decode over the block-table KV cache. "
        "window > 0 limits each query to the last `window` positions.",
        pybind11::arg("q"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_table"),
        pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("window") = 0);

  m.def("kv_cache_gather_range", &kv_cache_gather_range,
        "64-bit range gather from a strided dense/GQA paged KV cache",
        pybind11::arg("key_cache"), pybind11::arg("value_cache"),
        pybind11::arg("block_table"), pybind11::arg("token_start"),
        pybind11::arg("num_tokens"));

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

  m.def("deepseek_v4_indexer_kv_insert", &deepseek_v4_indexer_kv_insert,
        "RoPE + e4m3 quant + insert for the DeepSeek-V4 indexer K cache",
        pybind11::arg("kv"), pybind11::arg("kv_cache"),
        pybind11::arg("slot_mapping"), pybind11::arg("positions"),
        pybind11::arg("cos_sin_cache"), pybind11::arg("block_size"));

  m.def("qc_swiglu", &qc_swiglu,
        "Fused SwiGLU activation (silu*up or OAI gate*sigmoid*(up+beta), "
        "optional clamp), bitwise vs the eager Metal chain",
        pybind11::arg("x"), pybind11::arg("y"),
        pybind11::arg("clamp_limit") = pybind11::none(),
        pybind11::arg("oai_form") = false, pybind11::arg("alpha") = 1.0,
        pybind11::arg("beta") = 0.0);

  m.def("moe_weighted_sum", &moe_weighted_sum,
        "Weighted sum of MoE expert rows into the output hidden states "
        "(one dispatch, torch-eager numerics)",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("y"));

  m.def("dsv4_router_topk", &dsv4_router_topk,
        "Fused router top-k over pre-softplussed scores: sqrt + bias/hash "
        "select + top-k + renorm + scale (one dispatch)",
        pybind11::arg("gating"), pybind11::arg("out_w"),
        pybind11::arg("out_ids"), pybind11::arg("renormalize"),
        pybind11::arg("scaling"), pybind11::arg("bias") = pybind11::none(),
        pybind11::arg("hash_table") = pybind11::none(),
        pybind11::arg("input_ids") = pybind11::none());

  m.def("dsv4_indexer_compress_insert", &dsv4_indexer_compress_insert,
        "Fused indexer compressor tail: state gather + softmax compress + "
        "RMSNorm + RoPE + e4m3 quant + insert (one dispatch)",
        pybind11::arg("state_cache"), pybind11::arg("positions"),
        pybind11::arg("state_slots"), pybind11::arg("token_to_req"),
        pybind11::arg("block_table"), pybind11::arg("kv_slots"),
        pybind11::arg("rms_weight"), pybind11::arg("cos"), pybind11::arg("sin"),
        pybind11::arg("kv_cache"), pybind11::arg("state_block_size"),
        pybind11::arg("state_width"), pybind11::arg("compress_ratio"),
        pybind11::arg("eps"));

  m.def("dsv4_compress_front", &dsv4_compress_front,
        "Fused head=512 cr=4/cr=128 compressor front: state gather + "
        "softmax compress + RMSNorm, bf16 rows for deepseek_v4_kv_insert",
        pybind11::arg("state_cache"), pybind11::arg("positions"),
        pybind11::arg("state_slots"), pybind11::arg("token_to_req"),
        pybind11::arg("block_table"), pybind11::arg("rms_weight"),
        pybind11::arg("num_tokens"), pybind11::arg("state_block_size"),
        pybind11::arg("state_width"), pybind11::arg("compress_ratio"),
        pybind11::arg("eps"));

  m.def("deepseek_v4_prefill_dequant", &deepseek_v4_prefill_dequant,
        "Decode fp8 cache slots into a half [n, 512] scratch",
        pybind11::arg("cache"), pybind11::arg("slots"));
  m.def("deepseek_v4_prefill_fa", &deepseek_v4_prefill_fa,
        "Dense-causal prefill MMA FA over pre-decoded axes", pybind11::arg("q"),
        pybind11::arg("kc"), pybind11::arg("ks"), pybind11::arg("lens_c"),
        pybind11::arg("lo_s"), pybind11::arg("hi_s"), pybind11::arg("sinks"),
        pybind11::arg("scale"),
        pybind11::arg("out") = std::optional<at::Tensor>());
  m.def("deepseek_v4_sparse_attention", &deepseek_v4_sparse_attention,
        "Two-cache sparse DeepSeek-V4 attention over packed FP8 slots",
        pybind11::arg("q"), pybind11::arg("compressed_cache"),
        pybind11::arg("compressed_slots"), pybind11::arg("compressed_lens"),
        pybind11::arg("swa_cache"), pybind11::arg("swa_slots"),
        pybind11::arg("swa_lens"), pybind11::arg("sinks"),
        pybind11::arg("scale"),
        pybind11::arg("out") = std::optional<at::Tensor>());

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
        pybind11::arg("row"), pybind11::arg("tokens"),
        pybind11::arg("soa") = false);

  m.def("ggml_moe_a8_vec_swiglu", &ggml_moe_a8_vec_swiglu,
        "iq2_xxs MoE GEMV with fused SwiGLU epilogue (bit-exact vs "
        "ggml_moe_a8_vec + qc_swiglu)",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("topk_ids"),
        pybind11::arg("top_k"), pybind11::arg("quant_type"),
        pybind11::arg("row"), pybind11::arg("tokens"),
        pybind11::arg("clamp_limit") = pybind11::none(),
        pybind11::arg("soa") = false);

  m.def("ggml_moe_a8_vec_sum", &ggml_moe_a8_vec_sum,
        "q2_K MoE down GEMV with the weighted expert-slot sum folded into "
        "the epilogue; writes (tokens, N) into out",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("topk_ids"),
        pybind11::arg("topk_w"), pybind11::arg("top_k"),
        pybind11::arg("quant_type"), pybind11::arg("row"),
        pybind11::arg("tokens"), pybind11::arg("out"),
        pybind11::arg("soa") = false);

  m.def("ggml_moe_mm_id", &ggml_moe_mm_id,
        "MoE tiled GEMM for prefill widths (two-phase map0 + 64x32 simdgroup "
        "MMA): iq2_xxs w13 over (tokens, K) or q2_K down over the per-slot "
        "(tokens*top_k, K); returns (tokens*top_k, row) in flat slot order",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("topk_ids"),
        pybind11::arg("top_k"), pybind11::arg("quant_type"),
        pybind11::arg("row"), pybind11::arg("tokens"),
        pybind11::arg("soa") = false);

  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8,
        "GGUF weight-only GEMM over raw quantized blocks", pybind11::arg("w"),
        pybind11::arg("x"), pybind11::arg("quant_type"), pybind11::arg("row"));

  m.def("dsv4_mhc_pre", &dsv4_mhc_pre,
        "DeepSeek-V4 mHC pre block (projection, gates, Sinkhorn, mix)",
        pybind11::arg("residual"), pybind11::arg("fn"),
        pybind11::arg("hc_scale"), pybind11::arg("hc_base"),
        pybind11::arg("rms_eps"), pybind11::arg("pre_eps"),
        pybind11::arg("sinkhorn_eps"), pybind11::arg("post_multiplier"),
        pybind11::arg("sinkhorn_repeat"),
        pybind11::arg("norm_weight") = pybind11::none(),
        pybind11::arg("norm_eps") = 0.0);

  m.def("dsv4_mhc_fused_post_pre", &dsv4_mhc_fused_post_pre,
        "DeepSeek-V4 mHC post block fused with the next pre block",
        pybind11::arg("x"), pybind11::arg("residual"),
        pybind11::arg("post_mix"), pybind11::arg("comb_mix"),
        pybind11::arg("fn"), pybind11::arg("hc_scale"),
        pybind11::arg("hc_base"), pybind11::arg("rms_eps"),
        pybind11::arg("pre_eps"), pybind11::arg("sinkhorn_eps"),
        pybind11::arg("post_multiplier"), pybind11::arg("sinkhorn_repeat"),
        pybind11::arg("norm_weight") = pybind11::none(),
        pybind11::arg("norm_eps") = 0.0);

  m.def("dsv4_mhc_post", &dsv4_mhc_post,
        "DeepSeek-V4 mHC post block (residual stream remix)",
        pybind11::arg("x"), pybind11::arg("residual"),
        pybind11::arg("post_mix"), pybind11::arg("comb_mix"));

  m.def("dsv4_hc_head", &dsv4_hc_head,
        "DeepSeek-V4 hc head reduction (gated stream collapse)",
        pybind11::arg("residual"), pybind11::arg("fn"),
        pybind11::arg("hc_scale"), pybind11::arg("hc_base"),
        pybind11::arg("rms_eps"), pybind11::arg("hc_eps"));

  m.def("rms_norm", &rms_norm,
        "Weighted RMS norm (vllm ir.ops.rms_norm numerics)", pybind11::arg("x"),
        pybind11::arg("weight"), pybind11::arg("epsilon"));

  m.def("dsv4_indexer_q_rope_quant", &dsv4_indexer_q_rope_quant,
        "Indexer Q GPT-J RoPE + fp8-domain quantize with weight-folded scale",
        pybind11::arg("index_q"), pybind11::arg("positions"),
        pybind11::arg("cos_sin_cache"), pybind11::arg("index_weights"),
        pybind11::arg("softmax_scale"), pybind11::arg("head_scale"));

  m.def("dsv4_o_inv_rope", &dsv4_o_inv_rope,
        "Attention-output inverse GPT-J RoPE, flattened for grouped WO_A",
        pybind11::arg("o"), pybind11::arg("positions"),
        pybind11::arg("cos_sin_cache"));

  m.def("dsv4_indexer_topk_prefill", &dsv4_indexer_topk_prefill,
        "Prefill indexer top-k with per-request block-table rows",
        pybind11::arg("q"), pybind11::arg("weights"), pybind11::arg("kv_cache"),
        pybind11::arg("block_table"), pybind11::arg("tok_req"),
        pybind11::arg("cand"), pybind11::arg("out"), pybind11::arg("width"),
        pybind11::arg("k_eff"));
  m.def("dsv4_indexer_topk_decode", &dsv4_indexer_topk_decode,
        "Decode-path indexer logits + top-k into topk_indices_buffer",
        pybind11::arg("q"), pybind11::arg("weights"), pybind11::arg("kv_cache"),
        pybind11::arg("block_table"), pybind11::arg("cand"),
        pybind11::arg("out"), pybind11::arg("width"), pybind11::arg("k_eff"));

  m.def("qc_tape_register_layer", &qc_tape_register_layer,
        "Step tape: register a decoder layer's persistent tensors + scalars",
        pybind11::arg("idx"), pybind11::arg("tensors"),
        pybind11::arg("scalars"));
  m.def("qc_tape_layer_forward", &qc_tape_layer_forward,
        "Step tape: run one registered decoder layer body natively; step "
        "tensors are per-layer (KV group split safety)",
        pybind11::arg("idx"), pybind11::arg("x"), pybind11::arg("positions"),
        pybind11::arg("input_ids"), pybind11::arg("step"),
        pybind11::arg("insert_block_size"));
  m.def("ggml_mul_mat_sm", &ggml_mul_mat_sm,
        "GGUF small-M weight-streaming MMA GEMM (verify/draft band)",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("quant_type"),
        pybind11::arg("row"), pybind11::arg("n_warps") = 4);

  m.def("ggml_mul_mat_sm_f16probe", &ggml_mul_mat_sm_f16probe,
        "structure probe: sm tile flow over raw half weights",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("row"));
  m.def("ggml_mul_mat_sm_rm_pre", &ggml_mul_mat_sm_rm_pre,
        "row-major bf16 qgemm_sm_rm probe (fused-step layout)",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("quant_type"),
        pybind11::arg("row"));
  m.def("ggml_mul_mat_sm_pre", &ggml_mul_mat_sm_pre,
        "qgemm_sm without layout glue: X (K,32) half in, D (N,32) half out",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("quant_type"),
        pybind11::arg("row"), pybind11::arg("n_warps") = 4);
  m.def("ggml_mul_mat_sm_u4", &ggml_mul_mat_sm_u4,
        "uint4-native q4_K GEMM: tile-major packed weights + scale/min "
        "planes, X (32,K) half row-major, D (N,32) half out",
        pybind11::arg("wu"), pybind11::arg("x"), pybind11::arg("sc"),
        pybind11::arg("mn"), pybind11::arg("row"));
  m.def("qwen_gdn_step", &qwen_gdn_step,
        "Fused Qwen3.5 GDN decode/verify step: conv window update + gated "
        "delta-rule scan over S positions per sequence, fp32 state in place.",
        py::arg("x"), py::arg("a"), py::arg("b"), py::arg("conv_state"),
        py::arg("ssm_state"), py::arg("conv_weight"), py::arg("conv_bias"),
        py::arg("A_log"), py::arg("dt_bias"), py::arg("token_map"),
        py::arg("conv_slot"), py::arg("resume_slot"), py::arg("store_slots"),
        py::arg("num_accepted"), py::arg("out"), py::arg("num_seqs"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("tiled"),
        py::arg("act_silu"), py::arg("scale"));
  m.def("qwen_gdn_gated_norm", &qwen_gdn_gated_norm,
        "Gated RMS norm (norm_before_gate, silu) over (tokens, Hv, D) GDN "
        "core output; z is the qkvz view.",
        py::arg("x"), py::arg("z"), py::arg("w"), py::arg("eps"));
  m.def("dflash2_two_tap_conv", &dflash2_two_tap_conv,
        "DFlash 2 block-local dynamic two-tap convolution.", py::arg("x"),
        py::arg("coeffs"), py::arg("base"), py::arg("side"),
        py::arg("block_size"), py::arg("group_size"));
  m.def("qwen38_rejection_sample", &qwen38_rejection_sample,
        "Single-dispatch lossless probabilistic rejection sampler on Metal.",
        py::arg("target"), py::arg("draft"), py::arg("draft_sampled"),
        py::arg("cu"), py::arg("pos"), py::arg("idx_mapping"),
        py::arg("temperature"), py::arg("seeds"),
        py::arg("num_speculative_steps"), py::arg("vocab_size"));
  m.def("ggml_dequantize_fp16", &ggml_dequantize_fp16,
        "Dequantize a GGUF tensor to fp16 (N, K) via the tk_dequant8 decoders.",
        py::arg("w"), py::arg("quant_type"), py::arg("row"), py::arg("k"));
  m.def("prepare_dflash_inputs", &prepare_dflash_inputs,
        "native DFlash input prep (one dispatch, no host syncs)");
  m.def("ggml_mul_mat_sm_u4_rm", &ggml_mul_mat_sm_u4_rm,
        "glue-free uint4-native q4_K GEMM: X (M,K) bf16 in, (M,N) bf16 out",
        pybind11::arg("wu"), pybind11::arg("x"), pybind11::arg("sc"),
        pybind11::arg("mn"), pybind11::arg("row"));
  m.def("ggml_mul_mat_sm_u8", &ggml_mul_mat_sm_u8,
        "int8-native q6_K GEMM: (K,N) int8 weights + (K/16,N) scale plane, "
        "X (32,K) half row-major, D (N,32) half out",
        pybind11::arg("wq8"), pybind11::arg("x"), pybind11::arg("sc"),
        pybind11::arg("row"));
}
