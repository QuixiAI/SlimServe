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
  if (numel * static_cast<int64_t>(opts.dtype().itemsize()) >
      (8LL << 20)) {
    return at::empty(sizes, opts);
  }
  static std::unordered_map<std::string, OutRing> rings;
  std::string key(tag);
  key += '|';
  key += std::to_string(static_cast<int>(
      c10::typeMetaToScalarType(opts.dtype())));
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
                           int64_t window, int64_t max_context) {
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

  // D=256 rides the split-K partition/reduce pair: the monolithic
  // one-simdgroup-per-(head,seq) walk starves the GPU at this head size
  // (24 heads x batch 1 = 24 simdgroups, measured 0.9 GB/s), while the
  // partition grid (H, B, P) restores occupancy exactly like the MLA
  // decode split-K. D=64/128 keep the monolithic route (existing serving
  // paths, numerics untouched). `max_context` sizes the partitions
  // host-side (callers pass the batch max, no device sync); 0 falls back
  // to the block-table width upper bound — empty partitions early-out.
  if (head_size == 256) {
    int width = static_cast<int>(max_context);
    if (width <= 0) width = block_table_stride * block_size;
    width = std::max(width, 1);
    const int units = batch * num_heads;
    // Simdgroup occupancy target for split-K sizing; VLLM_QC_PA256_SPLITK
    // overrides for tuning (read once).
    static const int kSplitKTarget = [] {
      const char* v = getenv("VLLM_QC_PA256_SPLITK");
      return v ? std::max(1, atoi(v)) : 1536;
    }();
    int num_partitions =
        std::max(1, std::min((kSplitKTarget + units - 1) / units, 64));
    num_partitions = std::min(num_partitions, width);
    const int partition_size =
        (width + num_partitions - 1) / num_partitions;
    num_partitions = (width + partition_size - 1) / partition_size;
    auto fopts = q.options().dtype(at::kFloat);
    auto tmp = ring_out("pa256_tmp",
                        {batch, num_heads, num_partitions, head_size}, fopts);
    auto ml = ring_out("pa256_ml", {batch, num_heads, num_partitions}, fopts);
    auto es = ring_out("pa256_es", {batch, num_heads, num_partitions}, fopts);
    // One serial-dispatch encoder for both kernels: the implicit barrier
    // between dispatches orders reduce after partition, and one encoder
    // transition per layer is measurably cheaper than two on the c1 step.
    encode("qc_paged_attention_splitk", [&](TorchEncoder& e) {
      tk::launch_paged_attention_partition(
          e, q, key_cache, value_cache, block_table, context_lens, tmp, ml,
          es, batch, num_heads, num_kv_heads, head_size, block_size,
          block_table_stride, static_cast<float>(scale), num_partitions,
          partition_size, static_cast<int>(window), /*softcap=*/0.0f,
          activation_type_name(q));
      tk::launch_paged_attention_reduce(e, tmp, ml, es, out, batch, num_heads,
                                        head_size, num_partitions,
                                        /*sinks=*/q, /*has_sink=*/0,
                                        activation_type_name(q));
    });
    return out;
  }

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
};

State g;

void emit_matvec(TorchEncoder& e, const Proj& p, const at::Tensor& x,
                 const at::Tensor& out, int m) {
  const int K = static_cast<int>(x.size(-1));
  const std::string fmt = ggml_type_to_format(p.type);
  if (m == 1) {
    tk::launch_qgemv(e, out, p.w, x, p.rows, K, fmt, "bfloat16");
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

void muse_step_run(const at::Tensor& x, const at::Tensor& positions,
                   const at::Tensor& bt_local, const at::Tensor& sl_local,
                   const at::Tensor& slot_local, const at::Tensor& bt_full,
                   const at::Tensor& sl_full, const at::Tensor& slot_full) {
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

  encode("muse_step", [&](TorchEncoder& e) {
    for (const Layer& L : g.layers) {
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
      // paged attention over the cache
      tk::launch_paged_attention(
          e, g.q, L.kv_cache.select(0, 0), L.kv_cache.select(0, 1), bt, sl,
          g.attn_out, m, g.heads, g.kv_heads, hd,
          static_cast<int>(L.kv_cache.size(2)), static_cast<int>(bt.stride(0)),
          g.scale,
          /*alibi=*/g.q, /*use_alibi=*/0, /*mask=*/g.q, /*use_mask=*/0, window,
          /*mask_heads=*/0, "bfloat16");
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

// ---- Muse-qwen38 fused decode step ---------------------------------------
//
// Single-command-buffer emit of the Qwen3.8 hybrid decode forward: 48 GDN +
// 16 gated full-attention layers, gemma-norm seams, compressed-tensors
// NVFP4/FP8 projections. Same architecture as the Muse-Glimmer loop above —
// weights registered once, one encode() closure per step — but over the
// serving kernels of the qwen38 path (qgemv_fp8ch/nvfp4, gdn_fused_prepare/
// recur_spec/gated_rmsnorm_f32, qk_norm_rope_gate, kv_cache_scatter,
// paged_attention split-K). Eligibility, shadow validation and the
// trajectory re-pin plan live in csrc/quixicore/metal/muse_qwen38_design.md.
// NOT bit-identical to the eager layer loop by construction (torch-MPS
// eager elementwise numerics are size-dependent).

namespace muse_q38 {

struct QProj {
  at::Tensor w;       // packed weight bytes (fp8ch (N,K) / nvfp4 (N,K/2))
  at::Tensor scale;   // fp8ch per-row scales / nvfp4 group scales
  at::Tensor gscale;  // nvfp4 fp32 global multiplier; undefined for fp8ch
  int fmt = 0;        // 0 = fp8ch, 1 = nvfp4 planar
  int N = 0;
  int K = 0;
};

struct Layer {
  bool is_gdn = false;
  at::Tensor in_norm_w, post_norm_w;
  QProj gate_up, down;
  // gdn
  QProj qkvz, ba, gdn_out;
  at::Tensor conv_w, A_log, dt_bias, gated_norm_w;
  at::Tensor conv_state, ssm_state;
  int conv_state_stride = 0, conv_state_cols = 0, conv_kernel = 0;
  int ssm_stride = 0;
  float prep_eps = 0.f, gq_scale = 1.f, gk_scale = 1.f, norm_eps = 0.f;
  // attn
  QProj qkv, o;
  at::Tensor q_norm_w, k_norm_w, cos_sin, kc, vc;
  int block_mult = 1;
};

struct State {
  bool ready = false;
  int num_layers = 0, hidden = 0;
  // full-attention geometry
  int heads = 0, kv_heads = 0, head_dim = 0, rot_dim = 0, block_size = 0;
  int q_sz = 0, kv_sz = 0, qkv_n = 0, max_blocks = 0;
  float attn_scale = 0.f, eps = 0.f;
  // gdn geometry
  int gHk = 0, gHv = 0, gDk = 0, gDv = 0, gdn_qkv_ch = 0;
  int inter = 0, max_rows = 0;
  at::Tensor final_norm_w;
  // Drafter aux taps: POST-layer boundary indices (the eager loop's
  // aux_hidden_state_layers). aux_out slice i receives hidden + residual
  // after layer aux_layers[i]-1.
  std::vector<int> aux_layers;
  std::vector<Layer> layers;
  // bf16 scratch, all [max_rows, *]. res/res2 ping-pong across the fused
  // add-norm seams: the kernel must NEVER see residual and res_out alias
  // (an in-place update corrupts the normed output while leaving res_out
  // correct — found the hard way in P2 shadow round 4).
  at::Tensor h, res, res2, t_hidden;
  at::Tensor qkvz, ba_s, gnormed;      // gdn projections
  at::Tensor attn_qkv, q, k, v, gate, attn_out;
  at::Tensor gu, mid;
  at::Tensor z_view;                   // [max_rows, gHv, gDv] view into qkvz
  // fp32 gdn scratch
  at::Tensor gq, gk, gv, gdecay, gbeta, gy;
  // fp32 split-K attention scratch (>= max_rows*heads*64 partitions)
  at::Tensor pa_tmp, pa_ml, pa_es;
  // expanded-PA metadata, one slice per attention KV GROUP (hybrid
  // models split the attention layers across KV groups with separate
  // block pools — group-local block ids; found as P3's layer-7 bug).
  static constexpr int kMaxAttnGroups = 8;
  at::Tensor exp_seq, exp_bt;  // [kMaxAttnGroups, ...]
};

State g;

void set_qproj(QProj& p, const std::vector<at::Tensor>& t, int64_t fmt,
               int64_t N, int64_t K, const char* what) {
  TORCH_CHECK(fmt >= 0 && fmt <= 2, what,
              ": fmt must be 0 (fp8ch), 1 (nvfp4) or 2 (dense bf16)");
  const size_t want = fmt == 0 ? 2u : fmt == 1 ? 3u : 1u;
  TORCH_CHECK(t.size() == want, what,
              ": wants {w, scale} fp8ch / {w, scale, gscale} nvfp4 / {w} "
              "dense");
  p.w = t[0];
  if (fmt != 2) p.scale = t[1];
  if (fmt == 1) p.gscale = t[2];
  p.fmt = static_cast<int>(fmt);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  if (fmt == 2) {
    TORCH_CHECK(p.w.scalar_type() == at::kBFloat16 && p.w.dim() == 2 &&
                    p.w.size(0) == N && p.w.size(1) == K &&
                    p.w.is_contiguous() && K % 4 == 0,
                what, ": dense proj wants contiguous bf16 [N, K], K % 4");
  } else {
    TORCH_CHECK(p.N % 8 == 0 && p.K % 16 == 0, what,
                ": N % 8 and K % 16 required by the GEMV family, got N=", N,
                " K=", K);
  }
}

// Emit one quantized projection over the first m rows of x into out,
// mirroring the serving hosts' routing EXACTLY (fp8ch_mul_mat_vec /
// nvfp4_mul_mat_vec): m==1 base kernel, 3..8 the mv4r multi-row twin,
// everything else (m==2, and any m>8 a caller-configured max_rows could
// admit) the column-pair _mb batch twin — so muse projections are
// bit-identical to the eager step's.
void emit_qproj(TorchEncoder& e, const QProj& p, const at::Tensor& x,
                const at::Tensor& out, int m) {
  if (p.fmt == 2) {
    e.pipeline("mittens::muse_dense_gemv");
    e.out(out, 0);
    e.in(p.w, 1);
    e.in(x, 2);
    e.bytes(p.N, 3);
    e.bytes(p.K, 4);
    e.dispatch(p.N, m, 1, 32, 1, 1);
    return;
  }
  if (p.fmt == 0) {
    if (m == 1) {
      tk::launch_qgemv_fp8ch(e, out, p.w, x, p.scale, p.N, p.K, "bfloat16");
    } else if (m >= 3 && m <= 8) {
      tk::launch_qgemv_fp8ch_mv4r(e, out, p.w, x, p.scale, p.N, p.K, m,
                                  "bfloat16");
    } else {
      tk::launch_qgemv_fp8ch_mb(e, out, p.w, x, p.scale, p.N, p.K, m,
                                "bfloat16");
    }
  } else {
    if (m == 1) {
      tk::launch_qgemv_nvfp4_planar(e, out, p.w, x, p.scale, p.gscale, p.N,
                                    p.K, "bfloat16");
    } else if (m >= 3 && m <= 8) {
      tk::launch_qgemv_nvfp4_mv4r(e, out, p.w, x, p.scale, p.gscale, p.N,
                                  p.K, m, "bfloat16");
    } else {
      tk::launch_qgemv_nvfp4_planar_mb(e, out, p.w, x, p.scale, p.gscale,
                                       p.N, p.K, m, "bfloat16");
    }
  }
}

void emit_elem(TorchEncoder& e, int n4) {
  e.dispatch((n4 + 255) / 256, 1, 1, 256, 1, 1);
}

}  // namespace muse_q38

void muse_q38_init(int64_t num_layers, int64_t hidden, int64_t heads,
                   int64_t kv_heads, int64_t head_dim, int64_t rot_dim,
                   double attn_scale, int64_t gdn_k_heads, int64_t gdn_v_heads,
                   int64_t gdn_k_dim, int64_t gdn_v_dim, int64_t inter,
                   double eps, int64_t max_rows, int64_t max_blocks,
                   int64_t block_size, const at::Tensor& final_norm_w,
                   const std::vector<int64_t>& aux_layers,
                   const at::Tensor& ref) {
  using namespace muse_q38;
  check_mps(final_norm_w, "final_norm_w");
  TORCH_CHECK(ref.scalar_type() == at::kBFloat16,
              "muse_q38 is bf16-only; ref must be bf16");
  g = State{};
  g.num_layers = static_cast<int>(num_layers);
  g.hidden = static_cast<int>(hidden);
  g.heads = static_cast<int>(heads);
  g.kv_heads = static_cast<int>(kv_heads);
  g.head_dim = static_cast<int>(head_dim);
  g.rot_dim = static_cast<int>(rot_dim);
  g.attn_scale = static_cast<float>(attn_scale);
  g.gHk = static_cast<int>(gdn_k_heads);
  g.gHv = static_cast<int>(gdn_v_heads);
  g.gDk = static_cast<int>(gdn_k_dim);
  g.gDv = static_cast<int>(gdn_v_dim);
  g.inter = static_cast<int>(inter);
  g.eps = static_cast<float>(eps);
  g.max_rows = static_cast<int>(max_rows);
  g.max_blocks = static_cast<int>(max_blocks);
  g.block_size = static_cast<int>(block_size);
  g.q_sz = g.heads * g.head_dim;
  g.kv_sz = g.kv_heads * g.head_dim;
  g.qkv_n = 2 * g.q_sz + 2 * g.kv_sz;
  g.gdn_qkv_ch = 2 * g.gHk * g.gDk + g.gHv * g.gDv;
  g.final_norm_w = final_norm_w;
  g.aux_layers.clear();
  for (int64_t a : aux_layers) {
    TORCH_CHECK(a >= 1 && a <= num_layers, "aux layer index out of range: ",
                a);
    g.aux_layers.push_back(static_cast<int>(a));
  }
  g.layers.resize(static_cast<size_t>(num_layers));

  const auto opt = ref.options();
  const auto f32 = opt.dtype(at::kFloat);
  const auto i32 = opt.dtype(at::kInt);
  const int64_t m = max_rows;
  g.h = at::empty({m, hidden}, opt);
  g.res = at::empty({m, hidden}, opt);
  g.res2 = at::empty({m, hidden}, opt);
  g.t_hidden = at::empty({m, hidden}, opt);
  const int64_t gdn_w = g.gdn_qkv_ch + (int64_t)g.gHv * g.gDv;  // + z lanes
  g.qkvz = at::empty({m, gdn_w}, opt);
  g.ba_s = at::empty({m, (int64_t)2 * g.gHv}, opt);
  g.gnormed = at::empty({m, (int64_t)g.gHv * g.gDv}, opt);
  // Strided slice of the projection rows; the gated-norm kernel reads it
  // via its base pointer + explicit z_stride (a .view() would throw on the
  // non-contiguous narrow).
  g.z_view = g.qkvz.narrow(1, g.gdn_qkv_ch, (int64_t)g.gHv * g.gDv);
  g.attn_qkv = at::empty({m, g.qkv_n}, opt);
  g.q = at::empty({m, g.q_sz}, opt);
  g.k = at::empty({m, g.kv_sz}, opt);
  g.v = at::empty({m, g.kv_sz}, opt);
  g.gate = at::empty({m, g.q_sz}, opt);
  g.attn_out = at::empty({m, g.q_sz}, opt);
  g.gu = at::empty({m, (int64_t)2 * g.inter}, opt);
  g.mid = at::empty({m, g.inter}, opt);
  g.gq = at::empty({m, (int64_t)g.gHk * g.gDk}, f32);
  g.gk = at::empty({m, (int64_t)g.gHk * g.gDk}, f32);
  g.gv = at::empty({m, (int64_t)g.gHv * g.gDv}, f32);
  g.gdecay = at::empty({m, g.gHv}, f32);
  g.gbeta = at::empty({m, g.gHv}, f32);
  g.gy = at::empty({m, (int64_t)g.gHv * g.gDv}, f32);
  // Split-K scratch sized for the 64-partition ceiling.
  g.pa_tmp = at::empty({m, heads, 64, head_dim}, f32);
  g.pa_ml = at::empty({m, heads, 64}, f32);
  g.pa_es = at::empty({m, heads, 64}, f32);
  g.exp_seq = at::empty({muse_q38::State::kMaxAttnGroups, m}, i32);
  g.exp_bt = at::empty({muse_q38::State::kMaxAttnGroups, m, max_blocks}, i32);
}

void muse_q38_layer_gdn(
    int64_t idx, const at::Tensor& in_norm_w, const at::Tensor& post_norm_w,
    const std::vector<at::Tensor>& qkvz_t, int64_t qkvz_fmt,
    const std::vector<at::Tensor>& ba_t, int64_t ba_fmt,
    const at::Tensor& conv_w, const at::Tensor& A_log,
    const at::Tensor& dt_bias, const at::Tensor& gated_norm_w,
    const std::vector<at::Tensor>& out_t, int64_t out_fmt,
    const at::Tensor& conv_state, const at::Tensor& ssm_state,
    double prep_eps, double q_scale, double k_scale, double norm_eps,
    const std::vector<at::Tensor>& gu_t, int64_t gu_fmt,
    const std::vector<at::Tensor>& down_t, int64_t down_fmt) {
  using namespace muse_q38;
  TORCH_CHECK(idx >= 0 && idx < (int64_t)g.layers.size(), "bad layer idx");
  Layer& L = g.layers[static_cast<size_t>(idx)];
  L.is_gdn = true;
  L.in_norm_w = in_norm_w;
  L.post_norm_w = post_norm_w;
  const int64_t gdn_w = g.gdn_qkv_ch + (int64_t)g.gHv * g.gDv;
  set_qproj(L.qkvz, qkvz_t, qkvz_fmt, gdn_w, g.hidden, "gdn qkvz");
  set_qproj(L.ba, ba_t, ba_fmt, 2 * g.gHv, g.hidden, "gdn ba");
  L.conv_w = conv_w;
  L.A_log = A_log;
  L.dt_bias = dt_bias;
  L.gated_norm_w = gated_norm_w;
  set_qproj(L.gdn_out, out_t, out_fmt, g.hidden, (int64_t)g.gHv * g.gDv,
            "gdn out");
  TORCH_CHECK(conv_state.dim() == 3 && conv_state.scalar_type() == at::kFloat,
              "conv_state must be fp32 [slots, channels, cols]");
  TORCH_CHECK(ssm_state.dim() == 4 && ssm_state.scalar_type() == at::kFloat,
              "ssm_state must be fp32 [slots, Hv, Dv, Dk]");
  L.conv_state = conv_state;
  L.ssm_state = ssm_state;
  L.conv_state_cols = static_cast<int>(conv_state.size(2));
  L.conv_state_stride = static_cast<int>(conv_state.stride(0));
  L.conv_kernel = static_cast<int>(conv_w.size(1));
  L.ssm_stride = static_cast<int>(ssm_state.stride(0));
  L.prep_eps = static_cast<float>(prep_eps);
  L.gq_scale = static_cast<float>(q_scale);
  L.gk_scale = static_cast<float>(k_scale);
  L.norm_eps = static_cast<float>(norm_eps);
  set_qproj(L.gate_up, gu_t, gu_fmt, 2 * (int64_t)g.inter, g.hidden,
            "gate_up");
  set_qproj(L.down, down_t, down_fmt, g.hidden, g.inter, "down");
}

void muse_q38_layer_attn(
    int64_t idx, const at::Tensor& in_norm_w, const at::Tensor& post_norm_w,
    const std::vector<at::Tensor>& qkv_t, int64_t qkv_fmt,
    const at::Tensor& q_norm_w, const at::Tensor& k_norm_w,
    const at::Tensor& cos_sin, const std::vector<at::Tensor>& o_t,
    int64_t o_fmt, const at::Tensor& key_cache, const at::Tensor& value_cache,
    int64_t block_mult, const std::vector<at::Tensor>& gu_t, int64_t gu_fmt,
    const std::vector<at::Tensor>& down_t, int64_t down_fmt) {
  using namespace muse_q38;
  TORCH_CHECK(idx >= 0 && idx < (int64_t)g.layers.size(), "bad layer idx");
  Layer& L = g.layers[static_cast<size_t>(idx)];
  L.is_gdn = false;
  L.in_norm_w = in_norm_w;
  L.post_norm_w = post_norm_w;
  set_qproj(L.qkv, qkv_t, qkv_fmt, g.qkv_n, g.hidden, "attn qkv");
  L.q_norm_w = q_norm_w;
  L.k_norm_w = k_norm_w;
  L.cos_sin = cos_sin;
  set_qproj(L.o, o_t, o_fmt, g.hidden, g.q_sz, "attn o");
  L.kc = key_cache;
  L.vc = value_cache;
  TORCH_CHECK(block_mult == 1 || block_mult == 2, "block_mult must be 1|2");
  L.block_mult = static_cast<int>(block_mult);
  set_qproj(L.gate_up, gu_t, gu_fmt, 2 * (int64_t)g.inter, g.hidden,
            "gate_up");
  set_qproj(L.down, down_t, down_fmt, g.hidden, g.inter, "down");
}

// Encode the decode-step forward for the first `layers_cap` layers (<=0 =
// all) into one command buffer. x [m, hidden] bf16 is consumed and, when
// the cap covers the whole stack, overwritten with the FINAL-NORMED hidden
// states; residual_out receives the running residual (the prefix-capture
// boundary state — with a partial cap, x gets the raw layer output and
// eager layers continue from (x, residual_out)).
// GDN spec metadata is PER LAYER: the state pools are shared across the
// GDN layers with per-layer slot windows, so every GDN layer carries its
// own spec_cu/conv_slots/slot_table/num_accepted (vectors in GDN layer
// order). Using one layer's slots for all layers makes every layer stomp
// that layer's state — found via the P2 interval-attribution probe.
void muse_q38_run(const at::Tensor& x, const at::Tensor& residual_out,
                  const at::Tensor& positions,
                  const std::vector<at::Tensor>& block_table,
                  const std::vector<at::Tensor>& seq_lens,
                  const std::vector<at::Tensor>& attn_slots,
                  const std::vector<int64_t>& attn_max_context,
                  const std::vector<int64_t>& attn_group,
                  const std::vector<at::Tensor>& spec_cu,
                  const std::vector<at::Tensor>& conv_slots,
                  const std::vector<at::Tensor>& slot_table,
                  const std::vector<at::Tensor>& num_accepted,
                  int64_t q_len, int64_t layers_cap,
                  const c10::optional<at::Tensor>& aux_out,
                  const c10::optional<at::Tensor>& debug_out,
                  int64_t debug_layer) {
  using namespace muse_q38;
  TORCH_CHECK(g.num_layers > 0, "muse_q38 not initialized");
  const int m = static_cast<int>(x.size(0));
  TORCH_CHECK(m >= 1 && m <= g.max_rows, "row count out of range: ", m);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == g.hidden && x.is_contiguous() &&
                  x.scalar_type() == at::kBFloat16,
              "x must be contiguous bf16 [m, hidden]");
  TORCH_CHECK(residual_out.sizes() == x.sizes() &&
                  residual_out.is_contiguous() &&
                  residual_out.scalar_type() == at::kBFloat16,
              "residual_out must match x");
  TORCH_CHECK(q_len >= 1 && m % q_len == 0, "m must be reqs*q_len");
  const int reqs = m / static_cast<int>(q_len);
  const int cap = layers_cap <= 0
                      ? g.num_layers
                      : std::min<int>(static_cast<int>(layers_cap),
                                      g.num_layers);
  const bool full = cap == g.num_layers;
  size_t num_gdn = 0;
  for (const Layer& L : g.layers) num_gdn += L.is_gdn ? 1 : 0;
  TORCH_CHECK(spec_cu.size() == num_gdn && conv_slots.size() == num_gdn &&
                  slot_table.size() == num_gdn &&
                  num_accepted.size() == num_gdn,
              "per-layer gdn metadata vectors must have one entry per GDN "
              "layer (", num_gdn, ")");
  for (size_t i = 0; i < num_gdn; ++i) {
    TORCH_CHECK(spec_cu[i].scalar_type() == at::kInt &&
                    conv_slots[i].scalar_type() == at::kInt &&
                    slot_table[i].scalar_type() == at::kInt &&
                    num_accepted[i].scalar_type() == at::kInt,
                "gdn metadata must be i32");
  }
  const int num_groups = static_cast<int>(block_table.size());
  TORCH_CHECK(num_groups >= 1 &&
                  num_groups <= muse_q38::State::kMaxAttnGroups,
              "attention group count out of range: ", num_groups);
  TORCH_CHECK(seq_lens.size() == (size_t)num_groups &&
                  attn_slots.size() == (size_t)num_groups &&
                  attn_max_context.size() == (size_t)num_groups,
              "per-group attention metadata vectors must align");
  for (int gi = 0; gi < num_groups; ++gi) {
    TORCH_CHECK(seq_lens[gi].scalar_type() == at::kInt &&
                    block_table[gi].scalar_type() == at::kInt,
                "attention metadata must be i32");
    TORCH_CHECK(attn_slots[gi].scalar_type() == at::kLong,
                "attn slot_mapping must be i64");
  }
  TORCH_CHECK(positions.is_contiguous() && positions.numel() >= m,
              "positions must be contiguous with >= m entries");
  TORCH_CHECK(positions.scalar_type() == at::kLong ||
                  positions.scalar_type() == at::kInt,
              "positions must be int32 or int64");
  const std::string idx_name =
      positions.scalar_type() == at::kLong ? "i64" : "i32";

  // Per-group width, split-K sizing (mirrored from paged_attention) and
  // expanded-metadata slices.
  static const int kSplitKTarget = [] {
    const char* v = getenv("VLLM_QC_PA256_SPLITK");
    return v ? std::max(1, atoi(v)) : 1536;
  }();
  const int units = m * g.heads;
  std::vector<int> bt_widths(num_groups), n_parts(num_groups),
      part_sizes(num_groups);
  std::vector<at::Tensor> exp_seq_g(num_groups), exp_bt_g(num_groups);
  const int exp_bt_stride = static_cast<int>(g.exp_bt.stride(1));
  for (int gi = 0; gi < num_groups; ++gi) {
    int width = static_cast<int>(attn_max_context[gi]);
    if (width <= 0)
      width = static_cast<int>(block_table[gi].size(1)) * g.block_size;
    width = std::max(width, 1);
    int bw = (width + g.block_size - 1) / g.block_size;
    bt_widths[gi] = std::min(
        bw, std::min(static_cast<int>(block_table[gi].size(1)),
                     g.max_blocks));
    int np = std::max(1, std::min((kSplitKTarget + units - 1) / units, 64));
    np = std::min(np, width);
    part_sizes[gi] = (width + np - 1) / np;
    n_parts[gi] = (width + part_sizes[gi] - 1) / part_sizes[gi];
    exp_seq_g[gi] = g.exp_seq.select(0, gi);
    exp_bt_g[gi] = g.exp_bt.select(0, gi);
  }

  const int n4_hidden = m * g.hidden / 4;
  const int n4_attn = m * g.q_sz / 4;

  // Drafter aux taps: slice i of aux_out receives hidden + residual after
  // layer aux_layers[i]-1 (rows [:m]; the caller pre-zeroes padding).
  // Taps are emitted only when the caller wants them (serve mode always
  // passes aux_out; capped shadow replays may skip it).
  std::vector<at::Tensor> aux_slices;
  if (!g.aux_layers.empty() && aux_out.has_value()) {
    const at::Tensor& a = aux_out.value();
    TORCH_CHECK(a.dim() == 3 && a.size(0) == (int64_t)g.aux_layers.size() &&
                    a.size(1) >= m && a.size(2) == g.hidden &&
                    a.is_contiguous() && a.scalar_type() == at::kBFloat16,
                "aux_out must be contiguous bf16 [n_aux, >=m, hidden]");
    for (size_t i = 0; i < g.aux_layers.size(); ++i) {
      aux_slices.push_back(a.select(0, static_cast<int64_t>(i)));
    }
  }

  // Stage-dump instrument (bringup): for layer `debug_layer`, slot i of
  // debug_out [8, m, 2*inter] receives, flat at each buffer's own width:
  // 0 = h after entry seam, 1 = qkvz (GDN layers), 2 = ba, 3 = gnormed
  // (GDN core out), 4 = t_hidden after mixer, 5 = residual after the post
  // seam, 6 = gu, 7 = mid.
  const bool dbg = debug_out.has_value();
  if (dbg) {
    const at::Tensor& d = debug_out.value();
    TORCH_CHECK(d.dim() == 3 && d.size(0) == 8 && d.size(1) >= m &&
                    d.size(2) == 2 * (int64_t)g.inter && d.is_contiguous() &&
                    d.scalar_type() == at::kBFloat16,
                "debug_out must be contiguous bf16 [8, >=m, 2*inter]");
    // The widest dumped buffer (qkvz) must fit the 2*inter slot width —
    // dump() packs rows flat at each buffer's own width.
    TORCH_CHECK(g.gdn_qkv_ch + (int64_t)g.gHv * g.gDv <= 2 * (int64_t)g.inter,
                "debug_out slot width cannot hold the qkvz dump");
  }

  bool any_attn = false;
  int attn_mult = 0;
  for (int l = 0; l < cap; ++l) {
    if (g.layers[l].is_gdn) continue;
    any_attn = true;
    if (attn_mult == 0) attn_mult = g.layers[l].block_mult;
    TORCH_CHECK(g.layers[l].block_mult == attn_mult,
                "all attention layers must share one block layout");
  }

  encode("muse_q38_step", [&](TorchEncoder& e) {
    // Residual ping-pong: each fused add-norm reads *rcur and writes the
    // OTHER buffer — residual/res_out must never alias (see State note).
    // Layer 0 assigns rcur before any seam dereferences it.
    const at::Tensor* rcur = nullptr;
    int rflip = 0;
    auto dump = [&](int slot, const at::Tensor& src, int width) {
      if (!dbg) return;
      e.pipeline("mittens::muse_copy_rows");
      e.out(debug_out.value().select(0, slot), 0);
      e.in(src, 1);
      e.bytes(width / 4, 2);
      e.bytes(width, 3);
      e.bytes(m * width / 4, 4);
      e.dispatch((m * width / 4 + 255) / 256, 1, 1, 256, 1, 1);
    };
    auto seam = [&](const at::Tensor& xin, const at::Tensor& w) {
      const at::Tensor& rout = rflip ? g.res2 : g.res;
      rflip ^= 1;
      tk::launch_gemma_rms_norm_add_dyn(e, xin, *rcur, w, g.h, rout,
                                        static_cast<uint32_t>(m), g.hidden,
                                        g.eps);
      rcur = &rout;
    };
    if (any_attn) {
      // One expansion per attention KV group.
      for (int gi = 0; gi < num_groups; ++gi) {
        e.pipeline("mittens::muse_expand_meta");
        e.in(seq_lens[gi], 0);
        e.in(block_table[gi], 1);
        e.out(exp_seq_g[gi], 2);
        e.out(exp_bt_g[gi], 3);
        e.bytes(static_cast<int>(q_len), 4);
        e.bytes(static_cast<int>(block_table[gi].stride(0)), 5);
        e.bytes(exp_bt_stride, 6);
        e.bytes(bt_widths[gi], 7);
        e.bytes(attn_mult, 8);
        e.dispatch(m, 1, 1, 256, 1, 1);
      }
    }

    size_t gdn_idx = 0;
    size_t attn_idx = 0;
    for (int l = 0; l < cap; ++l) {
      const Layer& L = g.layers[static_cast<size_t>(l)];
      const bool dl = dbg && l == (int)debug_layer;
      if (l == 0) {
        // Layer 0: the eager path calls the 1-arg gemma norm (no fused
        // add) — emit the SAME kernel for bit-identity, and initialize
        // the residual with a plain copy of x.
        tk::launch_gemma_rms_norm_dyn(e, x, L.in_norm_w, g.h,
                                      static_cast<uint32_t>(m), g.hidden,
                                      g.eps);
        e.pipeline("mittens::muse_copy_rows");
        e.out(g.res, 0);
        e.in(x, 1);
        e.bytes(g.hidden / 4, 2);
        e.bytes(g.hidden, 3);
        e.bytes(n4_hidden, 4);
        emit_elem(e, n4_hidden);  // res_init
        rcur = &g.res;
        rflip = 1;  // next seam writes res2
      } else {
        seam(g.t_hidden, L.in_norm_w);
      }
      if (dl) dump(0, g.h, g.hidden);

      if (L.is_gdn) {
        const at::Tensor& l_cu = spec_cu[gdn_idx];
        const at::Tensor& l_conv_slots = conv_slots[gdn_idx];
        const at::Tensor& l_table = slot_table[gdn_idx];
        const at::Tensor& l_accepted = num_accepted[gdn_idx];
        ++gdn_idx;
        emit_qproj(e, L.qkvz, g.h, g.qkvz, m);
        emit_qproj(e, L.ba, g.h, g.ba_s, m);
        if (dl) {
          const int gdn_w = g.gdn_qkv_ch + g.gHv * g.gDv;
          dump(1, g.qkvz, gdn_w);
          dump(2, g.ba_s, 2 * g.gHv);
        }
        tk::launch_gdn_fused_prepare(
            e, g.qkvz, g.ba_s, L.conv_w, L.conv_state, l_cu, l_conv_slots,
            L.A_log, L.dt_bias, g.gq, g.gk, g.gv, g.gdecay, g.gbeta, reqs,
            g.gHk, g.gHv, g.gDk, g.gDv, L.conv_kernel, /*load_initial=*/1,
            /*qkvz_stride=*/static_cast<int>(g.qkvz.stride(0)),
            /*ba_stride=*/static_cast<int>(g.ba_s.stride(0)),
            L.conv_state_stride, L.prep_eps, L.gq_scale, L.gk_scale,
            L.conv_state_cols, l_accepted, /*spec_mode=*/1, "bfloat16");
        tk::launch_gdn_recur_spec(e, g.gq, g.gk, g.gv, g.gdecay, g.gbeta,
                                  L.ssm_state, l_cu, l_table,
                                  l_accepted, g.gy, reqs, g.gHk, g.gHv,
                                  g.gDv, g.gDk, L.ssm_stride,
                                  static_cast<int>(l_table.stride(0)),
                                  "float32");
        tk::launch_gdn_gated_rmsnorm_f32(
            e, g.gy, g.z_view, L.gated_norm_w, g.gnormed, m * g.gHv, g.gHv,
            g.gDv, static_cast<int>(g.qkvz.stride(0)), L.norm_eps,
            "bfloat16");
        if (dl) dump(3, g.gnormed, g.gHv * g.gDv);
        emit_qproj(e, L.gdn_out, g.gnormed, g.t_hidden, m);
      } else {
        const int gi = static_cast<int>(attn_group[attn_idx]);
        TORCH_CHECK(gi >= 0 && gi < num_groups, "bad attn group idx");
        ++attn_idx;
        emit_qproj(e, L.qkv, g.h, g.attn_qkv, m);
        tk::launch_qk_norm_rope_gate(
            e, g.attn_qkv, L.q_norm_w, L.k_norm_w, L.cos_sin, positions, g.q,
            g.gate, g.k, g.heads, g.kv_heads, g.head_dim, g.rot_dim, g.eps,
            g.qkv_n, m, "bfloat16", idx_name);
        {
          // V lanes: strided slice of the fused projection -> contiguous.
          const int width4 = g.kv_sz / 4;
          e.pipeline("mittens::muse_copy_rows");
          e.out(g.v, 0);
          e.in(g.attn_qkv.narrow(1, 2 * g.q_sz + g.kv_sz, g.kv_sz), 1);
          e.bytes(width4, 2);
          e.bytes(g.qkv_n, 3);
          e.bytes(m * width4, 4);
          emit_elem(e, m * width4);  // v_copy
        }
        tk::launch_kv_cache_scatter(e, g.k, g.v, attn_slots[gi], L.kc,
                                    L.vc, m, g.kv_heads, g.head_dim,
                                    g.block_size, L.block_mult, "bfloat16");
        if (g.head_dim == 256) {
          tk::launch_paged_attention_partition(
              e, g.q, L.kc, L.vc, exp_bt_g[gi], exp_seq_g[gi], g.pa_tmp,
              g.pa_ml, g.pa_es, m, g.heads, g.kv_heads, g.head_dim,
              g.block_size, exp_bt_stride, g.attn_scale, n_parts[gi],
              part_sizes[gi], /*window=*/0, /*softcap=*/0.0f, "bfloat16");
          tk::launch_paged_attention_reduce(e, g.pa_tmp, g.pa_ml, g.pa_es,
                                            g.attn_out, m, g.heads,
                                            g.head_dim, n_parts[gi],
                                            /*sinks=*/g.q, /*has_sink=*/0,
                                            "bfloat16");
        } else {
          tk::launch_paged_attention(
              e, g.q, L.kc, L.vc, exp_bt_g[gi], exp_seq_g[gi], g.attn_out,
              m, g.heads, g.kv_heads, g.head_dim, g.block_size,
              exp_bt_stride, g.attn_scale, /*alibi=*/g.q, /*use_alibi=*/0,
              /*mask=*/g.q, /*use_mask=*/0, /*window=*/0, /*mask_heads=*/0,
              "bfloat16");
        }
        e.pipeline("mittens::muse_sigmoid_mul_exact");
        e.out(g.attn_out, 0);
        e.in(g.gate, 1);
        e.bytes(n4_attn, 2);
        emit_elem(e, n4_attn);  // sigmoid_mul
        emit_qproj(e, L.o, g.attn_out, g.t_hidden, m);
      }

      // Post-attention seam + MLP.
      if (dl) dump(4, g.t_hidden, g.hidden);
      seam(g.t_hidden, L.post_norm_w);
      if (dl) dump(5, *rcur, g.hidden);
      emit_qproj(e, L.gate_up, g.h, g.gu, m);
      if (dl) dump(6, g.gu, 2 * g.inter);
      // Same kernel + args as the eager SiluAndMul.forward_mps
      // (_metal_swiglu -> qc_swiglu): bit-identical MLP activation.
      tk::launch_qc_swiglu(e, g.gu, g.mid, g.inter, /*has_clamp=*/0,
                           /*limit=*/0.0f, /*oai_form=*/0, /*alpha=*/1.0f,
                           /*beta=*/0.0f, m * g.inter, "bfloat16");
      if (dl) dump(7, g.mid, g.inter);
      emit_qproj(e, L.down, g.mid, g.t_hidden, m);

      // Aux tap: the boundary value after this layer is t_hidden + res —
      // the same hidden + residual the eager loop hands the drafter.
      for (size_t i = 0; i < aux_slices.size(); ++i) {
        if (g.aux_layers[i] != l + 1) continue;
        e.pipeline("mittens::muse_add_out");
        e.out(aux_slices[i], 0);
        e.in(g.t_hidden, 1);
        e.in(*rcur, 2);
        e.bytes(n4_hidden, 3);
        emit_elem(e, n4_hidden);  // aux_add
      }
    }

    if (full) {
      // Final seam: fused add + final gemma norm, written straight into x.
      tk::launch_gemma_rms_norm_add_dyn(e, g.t_hidden, *rcur,
                                        g.final_norm_w, x, residual_out,
                                        static_cast<uint32_t>(m), g.hidden,
                                        g.eps);
    } else {
      // Prefix-capture boundary: hand (raw layer output, residual) back so
      // the eager layers continue exactly where muse stopped.
      e.pipeline("mittens::muse_copy_rows");
      e.out(x, 0);
      e.in(g.t_hidden, 1);
      e.bytes(g.hidden / 4, 2);
      e.bytes(g.hidden, 3);
      e.bytes(n4_hidden, 4);
      emit_elem(e, n4_hidden);  // x_out
      e.pipeline("mittens::muse_copy_rows");
      e.out(residual_out, 0);
      e.in(*rcur, 1);
      e.bytes(g.hidden / 4, 2);
      e.bytes(g.hidden, 3);
      e.bytes(n4_hidden, 4);
      emit_elem(e, n4_hidden);  // res_out
    }
  });
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

// Fused residual add + RMSNorm (the 128-per-step decoder-seam pattern):
// returns (normed, summed_residual), bit-identical to the eager
// `residual = residual + x; rms_norm(residual)` chain.
std::tuple<at::Tensor, at::Tensor> add_rms_norm(const at::Tensor& x,
                                                const at::Tensor& residual,
                                                const at::Tensor& weight,
                                                double eps) {
  check_mps(x, "x");
  check_mps(residual, "residual");
  check_mps(weight, "weight");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  residual.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "add_rms_norm(metal) is bf16-only");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && residual.dim() == 2 &&
                  residual.is_contiguous() && x.sizes() == residual.sizes(),
              "add_rms_norm(metal) wants matching contiguous [rows, D]");
  const int D = static_cast<int>(x.size(1));
  TORCH_CHECK(weight.numel() == D, "weight/D mismatch");
  TORCH_CHECK(D % 4 == 0, "add_rms_norm(metal) needs D % 4 == 0");
  const auto M = static_cast<uint32_t>(x.size(0));
  at::Tensor out = at::empty_like(x);
  at::Tensor res_out = at::empty_like(x);
  encode("qc_add_rms_norm", [&](TorchEncoder& e) {
    tk::launch_rms_norm_add_dyn(e, x, residual, weight, out, res_out, M, D,
                                static_cast<float>(eps));
  });
  return {out, res_out};
}

// Gemma-semantics RMSNorm (vllm GemmaRMSNorm: y = bf16(x_hat32 * (w32 + 1)),
// fp32 weight multiply, single final round). Takes the raw bf16 module
// weight; the (1 + w) promote happens in-kernel.
at::Tensor gemma_rms_norm(const at::Tensor& x, const at::Tensor& weight,
                          double eps) {
  check_mps(x, "x");
  check_mps(weight, "weight");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "gemma_rms_norm(metal) is bf16-only");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "gemma_rms_norm(metal) wants contiguous [rows, D]");
  const int D = static_cast<int>(x.size(1));
  TORCH_CHECK(weight.numel() == D, "weight/D mismatch");
  TORCH_CHECK(D % 4 == 0, "gemma_rms_norm(metal) needs D % 4 == 0");
  const auto M = static_cast<uint32_t>(x.size(0));
  at::Tensor out = at::empty_like(x);
  encode("qc_gemma_rms_norm", [&](TorchEncoder& e) {
    tk::launch_gemma_rms_norm_dyn(e, x, weight, out, M, D,
                                  static_cast<float>(eps));
  });
  return out;
}

// Gemma-semantics fused residual add + RMSNorm (the 126-per-step decoder-seam
// pattern of the Qwen3.5 target): statistic and normed value from the
// UNROUNDED fp32 sum, summed residual rounded once — exactly
// ir.ops.fused_add_rms_norm with weight = float(w) + 1.
std::tuple<at::Tensor, at::Tensor> gemma_add_rms_norm(
    const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight,
    double eps) {
  check_mps(x, "x");
  check_mps(residual, "residual");
  check_mps(weight, "weight");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  residual.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "gemma_add_rms_norm(metal) is bf16-only");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && residual.dim() == 2 &&
                  residual.is_contiguous() && x.sizes() == residual.sizes(),
              "gemma_add_rms_norm(metal) wants matching contiguous [rows, D]");
  const int D = static_cast<int>(x.size(1));
  TORCH_CHECK(weight.numel() == D, "weight/D mismatch");
  TORCH_CHECK(D % 4 == 0, "gemma_add_rms_norm(metal) needs D % 4 == 0");
  const auto M = static_cast<uint32_t>(x.size(0));
  at::Tensor out = at::empty_like(x);
  at::Tensor res_out = at::empty_like(x);
  encode("qc_gemma_add_rms_norm", [&](TorchEncoder& e) {
    tk::launch_gemma_rms_norm_add_dyn(e, x, residual, weight, out, res_out, M,
                                      D, static_cast<float>(eps));
  });
  return {out, res_out};
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
  const bool kv_strided_ok =
      kv_half && kv_in.stride(1) == 1 &&
      kv_in.stride(0) >= kv_in.size(1) &&
      kv_in.stride(0) <= std::numeric_limits<int>::max();
  auto kv = kv_strided_ok ? kv_in : kv_in.contiguous();
  auto out = q_half
      ? ring_out("qnorm_out", q.sizes(), q.options().dtype(at::kBFloat16))
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
        oai_form ? 1 : 0, static_cast<float>(alpha),
        static_cast<float>(beta), total, activation_type_name(x));
  });
}

void qc_kv_cache_scatter(const at::Tensor& key, const at::Tensor& value,
                         const at::Tensor& slot_mapping,
                         const at::Tensor& key_cache,
                         const at::Tensor& value_cache, int64_t num_heads,
                         int64_t head_size, int64_t block_size,
                         int64_t block_mult) {
  check_mps(key, "key");
  check_mps(value, "value");
  check_mps(slot_mapping, "slot_mapping");
  // Caches may be layout views (page-local dense base + one-block-shifted V
  // alias); the kernel's flat math + block_mult is the layout contract.
  check_mps_strided(key_cache, "key_cache");
  check_mps_strided(value_cache, "value_cache");
  TORCH_CHECK(key.is_contiguous() && value.is_contiguous(),
              "kv_cache_scatter key/value must be contiguous");
  const int64_t T = key.size(0);
  TORCH_CHECK(value.size(0) == T, "key/value token mismatch");
  TORCH_CHECK(key.numel() == T * num_heads * head_size &&
                  value.numel() == T * num_heads * head_size,
              "key/value shape mismatch");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong &&
                  slot_mapping.is_contiguous() && slot_mapping.numel() >= T,
              "slot_mapping must be contiguous int64 [tokens]");
  TORCH_CHECK(key_cache.scalar_type() == key.scalar_type() &&
                  value_cache.scalar_type() == key.scalar_type() &&
                  value.scalar_type() == key.scalar_type(),
              "kv_cache_scatter dtype mismatch");
  if (T == 0) {
    return;
  }
  encode("qc_kv_cache_scatter", [&](TorchEncoder& e) {
    tk::launch_kv_cache_scatter(
        e, key, value, slot_mapping, key_cache, value_cache,
        static_cast<int>(T), static_cast<int>(num_heads),
        static_cast<int>(head_size), static_cast<int>(block_size),
        static_cast<int>(block_mult), activation_type_name(key));
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qc_qk_norm_rope_gate(
    const at::Tensor& qkv, const at::Tensor& q_w, const at::Tensor& k_w,
    const at::Tensor& cos_sin, const at::Tensor& positions,
    int64_t num_q_heads, int64_t num_k_heads, int64_t head_dim,
    int64_t rot_dim, double eps) {
  check_mps(qkv, "qkv");
  check_mps(q_w, "q_w");
  check_mps(k_w, "k_w");
  check_mps(cos_sin, "cos_sin");
  check_mps(positions, "positions");
  TORCH_CHECK(qkv.dim() == 2 && qkv.is_contiguous(),
              "qk_norm_rope_gate qkv must be contiguous [tokens, width]");
  const int64_t T = qkv.size(0);
  const int64_t D = head_dim;
  TORCH_CHECK(D > 0 && D <= 256, "head_dim must be in (0, 256]");
  TORCH_CHECK(rot_dim > 0 && rot_dim <= D && rot_dim % 2 == 0,
              "rot_dim must be even and <= head_dim");
  TORCH_CHECK(qkv.size(1) >= num_q_heads * 2 * D + num_k_heads * D,
              "qkv width too small for gated q + k spans");
  TORCH_CHECK(q_w.numel() == D && k_w.numel() == D && q_w.is_contiguous() &&
                  k_w.is_contiguous(),
              "q_w/k_w must be contiguous [head_dim]");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.is_contiguous() &&
                  cos_sin.size(1) == rot_dim,
              "cos_sin must be contiguous [max_pos, rot_dim]");
  TORCH_CHECK(q_w.scalar_type() == qkv.scalar_type() &&
                  k_w.scalar_type() == qkv.scalar_type() &&
                  cos_sin.scalar_type() == qkv.scalar_type(),
              "qk_norm_rope_gate dtype mismatch");
  TORCH_CHECK(positions.dim() == 1 && positions.is_contiguous() &&
                  positions.numel() >= T,
              "positions must be contiguous [tokens]");
  std::string idx_name;
  if (positions.scalar_type() == at::kLong) {
    idx_name = "i64";
  } else if (positions.scalar_type() == at::kInt) {
    idx_name = "i32";
  } else {
    TORCH_CHECK(false, "positions must be int32 or int64");
  }
  auto opts = qkv.options();
  auto q_out = at::empty({T, num_q_heads * D}, opts);
  auto gate_out = at::empty({T, num_q_heads * D}, opts);
  auto k_out = at::empty({T, num_k_heads * D}, opts);
  if (T == 0) {
    return {q_out, gate_out, k_out};
  }
  encode("qc_qk_norm_rope_gate", [&](TorchEncoder& e) {
    tk::launch_qk_norm_rope_gate(
        e, qkv, q_w, k_w, cos_sin, positions, q_out, gate_out, k_out,
        static_cast<int>(num_q_heads), static_cast<int>(num_k_heads),
        static_cast<int>(D), static_cast<int>(rot_dim),
        static_cast<float>(eps), static_cast<int>(qkv.size(1)),
        static_cast<int>(T), activation_type_name(qkv), idx_name);
  });
  return {q_out, gate_out, k_out};
}

at::Tensor qc_dflash_conv(const at::Tensor& x, const at::Tensor& delta,
                          const at::Tensor& base, int64_t block_size) {
  check_mps(x, "x");
  // delta is a side slice of the [T, 2, taps, G] projection view — strided
  // over rows but inner-contiguous; the explicit stride checks below are
  // the layout contract.
  check_mps_strided(delta, "delta");
  check_mps(base, "base");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "dflash_conv x must be contiguous [tokens, hidden]");
  TORCH_CHECK(delta.dim() == 3 && delta.stride(2) == 1 &&
                  delta.stride(1) == delta.size(2) &&
                  delta.size(0) == x.size(0),
              "dflash_conv delta must be [tokens, taps, groups] with "
              "contiguous inner dims (side slices of the projection view "
              "pass via their row stride)");
  TORCH_CHECK(base.dim() == 2 && base.is_contiguous() &&
                  base.size(0) == delta.size(1) && base.size(1) == x.size(1),
              "dflash_conv base must be contiguous [taps, hidden]");
  TORCH_CHECK(delta.scalar_type() == x.scalar_type() &&
                  base.scalar_type() == x.scalar_type(),
              "dflash_conv dtype mismatch");
  const int H = static_cast<int>(x.size(1));
  const int num_groups = static_cast<int>(delta.size(2));
  TORCH_CHECK(num_groups > 0 && H % num_groups == 0,
              "dflash_conv hidden must be divisible by groups");
  const int group_size = H / num_groups;
  const int taps = static_cast<int>(delta.size(1));
  TORCH_CHECK(taps >= 1 && block_size >= 1, "dflash_conv bad taps/block");
  auto out = at::empty_like(x);
  const long total_l = x.numel();
  TORCH_CHECK(total_l <= INT32_MAX, "dflash_conv input too large");
  const int total = static_cast<int>(total_l);
  if (total == 0) {
    return out;
  }
  const int delta_row_stride = static_cast<int>(delta.stride(0));
  encode("qc_dflash_conv", [&](TorchEncoder& e) {
    tk::launch_qc_dflash_conv(e, x, delta, base, out, H, num_groups,
                              group_size, taps, static_cast<int>(block_size),
                              delta_row_stride, total,
                              activation_type_name(x));
  });
  return out;
}

void moe_weighted_sum(const at::Tensor& x, const at::Tensor& w,
                      at::Tensor& y) {
  check_mps(x, "x");
  check_mps(w, "w");
  check_mps(y, "y");
  TORCH_CHECK(x.dim() == 3 && x.is_contiguous(),
              "x must be contiguous [tokens, topk, dim]");
  TORCH_CHECK((x.scalar_type() == at::kHalf ||
               x.scalar_type() == at::kBFloat16) &&
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
                    input_ids->is_contiguous() &&
                    input_ids->numel() == tokens,
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
  TORCH_CHECK(compress_ratio == 4 && state_width == 256 &&
                  state_cache.size(2) == 512,
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
        static_cast<int>(state_cache.stride(1)),
        static_cast<int>(state_width), static_cast<int>(compress_ratio),
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
at::Tensor dsv4_compress_front(
    const at::Tensor& state_cache, const at::Tensor& positions_in,
    const at::Tensor& state_slots_in, const at::Tensor& token_to_req_in,
    const at::Tensor& block_table, const at::Tensor& rms_w_in,
    int64_t num_tokens, int64_t state_block_size, int64_t state_width,
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
              "cr=128 [kv 512 | score 512] rows, got cr=", compress_ratio,
              " width=", state_width, " row=", state_cache.size(2));
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
        e, state_cache, positions, sslots, t2r, block_table, rms_w, out,
        tokens, static_cast<int>(state_block_size),
        static_cast<int>(state_cache.stride(0)),
        static_cast<int>(state_cache.stride(1)),
        static_cast<int>(state_width), static_cast<int>(compress_ratio),
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
  auto out = at::empty({n, 512},
                       at::TensorOptions().dtype(at::kHalf).device(cache.device()));
  encode("qc_mla_prefill_dequant", [&](TorchEncoder& e) {
    tk::launch_mla_prefill_dequant_slots(
        e, cache, slots, out, n, static_cast<int>(cache.size(1)),
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
              "kc/ks rows must be padded to a multiple of 32, got ",
              kc.size(0), " and ", ks.size(0));
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
    tk::launch_mla_prefill_fa_mma(
        e, q, kc, ks, lens_c, lo_s, hi_s, sinks, out, T, heads,
        static_cast<int>(kc.size(0)), static_cast<int>(ks.size(0)),
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
    num_partitions =
        std::min((splitk_target + units - 1) / units, 16);
    num_partitions = std::min(num_partitions, total_width);
  }
  if (num_partitions > 1) {
    const int partition_size =
        (total_width + num_partitions - 1) / num_partitions;
    num_partitions = (total_width + partition_size - 1) / partition_size;
    auto fopts = q.options().dtype(at::kFloat);
    auto tmp = ring_out("mla2_tmp", {batch, heads, num_partitions, 512},
                        fopts);
    auto ml = ring_out("mla2_ml", {batch, heads, num_partitions}, fopts);
    auto es = ring_out("mla2_es", {batch, heads, num_partitions}, fopts);
    encode("qc_deepseek_v4_sparse_attention_splitk", [&](TorchEncoder& e) {
      tk::launch_mla_decode_fp8_sparse_two_cache_packed_partition(
          e, q, compressed_cache, compressed_slots, compressed_lens,
          swa_cache, swa_slots, swa_lens, tmp, ml, es, batch, heads, cw, sw,
          static_cast<int>(compressed_cache.size(1)),
          static_cast<int>(compressed_cache.stride(0)),
          static_cast<int>(swa_cache.size(1)),
          static_cast<int>(swa_cache.stride(0)), static_cast<float>(scale),
          num_partitions, partition_size);
    });
    encode("qc_deepseek_v4_sparse_attention_reduce", [&](TorchEncoder& e) {
      tk::launch_paged_attention_reduce(
          e, tmp, ml, es, out, batch, heads, /*head_size=*/512,
          num_partitions, sinks, /*has_sink=*/1,
          half_out ? "float16" : "bfloat16");
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

// Split-K TQ decode attention: the (H, B, P) partition grid restores GPU
// occupancy exactly like the bf16 PA256 split-K (the monolithic
// tq_attention_combined runs one threadgroup per (head, batch) with two
// barriers per token — measured ~7 GB/s at the qwen38 c1 shape). The kernel
// reads the block table directly, so callers skip the per-layer B x width
// slots-matrix build. `max_context` sizes partitions host-side (batch max
// from CPU seq-lens, no device sync); 0 falls back to the block-table width
// bound — empty partitions early-out in-kernel.
at::Tensor turboquant_attention_splitk_metal(
    const at::Tensor& q_in, const at::Tensor& kv_cache,
    const at::Tensor& block_table_in, const at::Tensor& lengths_in,
    const at::Tensor& centroids_in, const at::Tensor& signs_in,
    const at::Tensor& sinks_in, double scale, int64_t num_kv_heads,
    int64_t k_bits, bool k_signed, int64_t v_bits, int64_t max_context) {
  check_mps(q_in, "q");
  check_mps_strided(kv_cache, "kv_cache");
  TORCH_CHECK(q_in.dim() == 3,
              "TurboQuant q must be [batch, heads, head_size]");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte,
              "TurboQuant Metal cache must be uint8");
  TORCH_CHECK(block_table_in.dim() == 2,
              "TurboQuant block_table must be [batch, max_blocks]");
  const int batch = static_cast<int>(q_in.size(0));
  const int heads = static_cast<int>(q_in.size(1));
  const int head_size = static_cast<int>(q_in.size(2));
  const int block_size = static_cast<int>(kv_cache.size(1));
  auto q = q_in.contiguous();
  auto block_table = block_table_in.to(at::kInt).contiguous();
  auto lengths = lengths_in.to(at::kInt).contiguous();
  auto centroids = centroids_in.to(at::kFloat).contiguous();
  auto signs = signs_in.to(at::kFloat).contiguous();
  auto sinks = sinks_in.to(at::kFloat).contiguous();
  const int bt_stride = static_cast<int>(block_table.stride(0));
  int width = static_cast<int>(max_context);
  const int width_bound = static_cast<int>(block_table.size(1)) * block_size;
  if (width <= 0 || width > width_bound) width = width_bound;
  width = std::max(width, 1);
  const int units = batch * heads;
  static const int kSplitKTarget = [] {
    const char* v = getenv("VLLM_QC_TQ_SPLITK_TARGET");
    return v ? std::max(1, atoi(v)) : 1536;
  }();
  int num_partitions =
      std::max(1, std::min((kSplitKTarget + units - 1) / units, 64));
  num_partitions = std::min(num_partitions, width);
  const int partition_size = (width + num_partitions - 1) / num_partitions;
  num_partitions = (width + partition_size - 1) / partition_size;
  auto fopts = q.options().dtype(at::kFloat);
  auto tmp = ring_out("tq_sk_tmp", {batch, heads, num_partitions, head_size},
                      fopts);
  auto ml = ring_out("tq_sk_ml", {batch, heads, num_partitions}, fopts);
  auto es = ring_out("tq_sk_es", {batch, heads, num_partitions}, fopts);
  auto out = at::empty_like(q);
  // One serial-dispatch encoder orders reduce after partition (same pattern
  // as the PA256 split-K pair).
  encode("qc_turboquant_attention_splitk", [&](TorchEncoder& e) {
    tk::launch_tq_attention_splitk(
        e, q, kv_cache, block_table, lengths, centroids, tmp, ml, es, batch,
        heads, static_cast<int>(num_kv_heads), head_size, bt_stride,
        block_size, static_cast<int>(kv_cache.stride(0)),
        static_cast<int>(kv_cache.stride(1)),
        static_cast<int>(kv_cache.stride(2)), num_partitions, partition_size,
        static_cast<int>(k_bits), k_signed ? 1 : 0, static_cast<int>(v_bits),
        static_cast<float>(scale), activation_type_name(q));
    tk::launch_tq_attention_reduce(e, tmp, ml, es, sinks, signs, out, batch,
                                   heads, head_size, num_partitions,
                                   activation_type_name(q));
  });
  return out;
}

void turboquant_dequant_kv_metal(const at::Tensor& kv_cache,
                                 const at::Tensor& slots_in,
                                 const at::Tensor& centroids_in,
                                 const at::Tensor& signs_in, at::Tensor& k_out,
                                 at::Tensor& v_out, int64_t k_bits,
                                 bool k_signed, int64_t v_bits) {
  check_mps_strided(kv_cache, "kv_cache");
  check_mps(k_out, "k_out");
  check_mps(v_out, "v_out");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte,
              "TurboQuant Metal cache must be uint8");
  TORCH_CHECK(k_out.dim() == 4 && v_out.sizes() == k_out.sizes() &&
                  k_out.scalar_type() == v_out.scalar_type(),
              "TurboQuant dequant outputs must be matching "
              "[1, kv_heads, rows, head_size] tensors");
  const int kv_heads = static_cast<int>(k_out.size(1));
  const int n_rows = static_cast<int>(k_out.size(2));
  const int head_size = static_cast<int>(k_out.size(3));
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256 ||
                  head_size == 512,
              "TurboQuant Metal head_size must be 64/128/256/512");
  auto slots = slots_in.to(at::kInt).contiguous();
  TORCH_CHECK(slots.numel() == n_rows,
              "TurboQuant dequant slots must have one entry per output row");
  auto centroids = centroids_in.to(at::kFloat).contiguous();
  auto signs = signs_in.to(at::kFloat).contiguous();
  encode("qc_turboquant_dequant_kv_metal", [&](TorchEncoder& e) {
    tk::launch_tq_decode_combined(
        e, kv_cache, slots, centroids, signs, k_out, v_out, n_rows, kv_heads,
        head_size, static_cast<int>(kv_cache.size(1)),
        static_cast<int>(kv_cache.stride(0)),
        static_cast<int>(kv_cache.stride(1)),
        static_cast<int>(kv_cache.stride(2)), static_cast<int>(k_bits),
        k_signed ? 1 : 0, static_cast<int>(v_bits),
        activation_type_name(k_out));
  });
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
  const bool has_mm = (fmt == "q4_0" || fmt == "q8_0" || fmt == "q4_K" ||
                       fmt == "q5_K" || fmt == "q6_K") &&
                      type_name != "float32" &&
                      // Same q8_0-small carve-out as mb: keep the "_small"
                      // batch-1 summation order for K <= 512 fp16.
                      !(K <= 512 && fmt == "q8_0" && type_name == "float16");
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

// Compressed-tensors FP8-per-channel W8A16 GEMV (the NVFP4 checkpoint's FP8
// side). `w` is the planar row-major (N, K) e4m3 byte tensor exactly as
// loaded (uint8 storage; torch-MPS has no fp8 dtype), `w_scale` one float
// per output row. An even batch with contiguous rows rides the
// weight-stationary column-pair mb twin (rows bit-identical to the loop);
// batch 1 and odd batches loop the batch-1 kernel inside one command
// buffer (mirrors the q4_K sequence).
at::Tensor fp8ch_mul_mat_vec(const at::Tensor& w, const at::Tensor& x,
                             const at::Tensor& w_scale) {
  check_mps(w, "w");
  check_mps(x, "x");
  check_mps(w_scale, "w_scale");
  TORCH_CHECK(w.scalar_type() == at::kByte,
              "quixicore(metal): fp8ch weight must be uint8 e4m3 bytes");
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous(),
              "quixicore(metal): fp8ch weight must be contiguous (N, K)");
  TORCH_CHECK(w_scale.scalar_type() == at::kFloat && w_scale.is_contiguous(),
              "quixicore(metal): fp8ch w_scale must be contiguous float32");
  const int N = static_cast<int>(w.size(0));
  const int K = static_cast<int>(w.size(1));
  TORCH_CHECK(x.size(-1) == K, "quixicore(metal): fp8ch x/K mismatch");
  TORCH_CHECK(w_scale.numel() == N, "quixicore(metal): fp8ch scale/N mismatch");
  TORCH_CHECK(N % 4 == 0 && K % 16 == 0,
              "quixicore(metal): fp8ch needs N % 4 == 0 and K % 16 == 0");

  at::Tensor out = ring_out("gemv_out", {x.size(0), N}, x.options());
  const std::string type_name = activation_type_name(x);
  const int batch = static_cast<int>(x.size(0));
  // mv_ext route (N5b): batches 3..8 (any parity) decode weights once per
  // ceil(M/4) column pass — measured 1.67x vs the column-pair twin at
  // M=8 and ~2x vs the dense fallback at odd M. Batch 2 stays on the
  // pair twin (890 vs 783 GB/s at qkv). Boot-time kill switch.
  static const bool use_mv4r = [] {
    const char* e = getenv("VLLM_QC_FP8CH_MV4R");
    return !(e && e[0] == '0');
  }();
  encode("qc_fp8ch_mmvq", [&](TorchEncoder& e) {
    if (use_mv4r && batch >= 3 && batch <= 8 && N % 8 == 0 &&
        x.dim() == 2 && x.is_contiguous()) {
      tk::launch_qgemv_fp8ch_mv4r(e, out, w, x, w_scale, N, K, batch,
                                  type_name);
      return;
    }
    if (batch > 1 && batch % 2 == 0 && x.dim() == 2 && x.is_contiguous()) {
      tk::launch_qgemv_fp8ch_mb(e, out, w, x, w_scale, N, K, batch,
                                type_name);
      return;
    }
    for (int b = 0; b < batch; ++b) {
      const at::Tensor x_row = x.select(0, b);
      const at::Tensor out_row = out.select(0, b);
      tk::launch_qgemv_fp8ch(e, out_row, w, x_row, w_scale, N, K, type_name);
    }
  });
  return out;
}

// Compressed-tensors NVFP4 W4A16 GEMV (the NVFP4 checkpoint's MLP side,
// layers 0-55). Planar checkpoint buffers, no repack: `w` (N, K/2) packed
// e2m1 nibble pairs, `w_scale` (N, K/16) raw e4m3 bytes (uint8 storage),
// `global_scale` the fp32 per-tensor multiplier (already inverted from the
// CT divisor by the scheme). Contiguous batches 3..8 ride the mv_ext
// twin, batch 2 the weight-stationary column-pair mb twin (both
// bit-identical per row to the loop); batch 1 and non-contiguous batches
// loop the batch-1 kernel.
at::Tensor nvfp4_mul_mat_vec(const at::Tensor& w, const at::Tensor& x,
                             const at::Tensor& w_scale,
                             const at::Tensor& global_scale) {
  check_mps(w, "w");
  check_mps(x, "x");
  check_mps(w_scale, "w_scale");
  check_mps(global_scale, "global_scale");
  TORCH_CHECK(w.scalar_type() == at::kByte && w_scale.scalar_type() == at::kByte,
              "quixicore(metal): nvfp4 weight and scale must be uint8 bytes");
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous() && w_scale.is_contiguous(),
              "quixicore(metal): nvfp4 weight/scale must be contiguous");
  TORCH_CHECK(global_scale.scalar_type() == at::kFloat,
              "quixicore(metal): nvfp4 global scale must be float32");
  const int N = static_cast<int>(w.size(0));
  const int K = static_cast<int>(w.size(1)) * 2;
  TORCH_CHECK(x.size(-1) == K, "quixicore(metal): nvfp4 x/K mismatch");
  TORCH_CHECK(w_scale.size(0) == N && w_scale.size(1) * 16 == K,
              "quixicore(metal): nvfp4 scale shape mismatch");
  TORCH_CHECK(N % 4 == 0 && K % 16 == 0,
              "quixicore(metal): nvfp4 needs N % 4 == 0 and K % 16 == 0");

  at::Tensor out = ring_out("gemv_out", {x.size(0), N}, x.options());
  const std::string type_name = activation_type_name(x);
  const int batch = static_cast<int>(x.size(0));
  // mv_ext route (N5b): batches 3..8 (any parity), NR=4 rows x R1=2
  // columns per pass — measured 1.2-1.3x vs the column-pair twin at
  // M=4/8. Batch 2 stays on the pair twin (578 vs 455 GB/s at gate_up).
  static const bool use_mv4r = [] {
    const char* e = getenv("VLLM_QC_NVFP4_MV4R");
    return !(e && e[0] == '0');
  }();
  encode("qc_nvfp4_mmvq", [&](TorchEncoder& e) {
    if (use_mv4r && batch >= 3 && batch <= 8 && N % 8 == 0 &&
        x.dim() == 2 && x.is_contiguous()) {
      tk::launch_qgemv_nvfp4_mv4r(e, out, w, x, w_scale, global_scale, N, K,
                                  batch, type_name);
      return;
    }
    if (batch > 1 && batch % 2 == 0 && x.dim() == 2 && x.is_contiguous()) {
      tk::launch_qgemv_nvfp4_planar_mb(e, out, w, x, w_scale, global_scale,
                                       N, K, batch, type_name);
      return;
    }
    for (int b = 0; b < batch; ++b) {
      const at::Tensor x_row = x.select(0, b);
      const at::Tensor out_row = out.select(0, b);
      tk::launch_qgemv_nvfp4_planar(e, out_row, w, x_row, w_scale,
                                    global_scale, N, K, type_name);
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
                "SoA MoE repack covers q2_K only (iq2_xxs is AoS), got ",
                fmt);
    TORCH_CHECK(K % 256 == 0,
                "SoA-repacked experts need the multi-row kernels (K % 256)");
    TORCH_CHECK((w.size(1) * w.size(2)) % 8 == 0,
                "SoA expert stride must stay 8-byte aligned, got ",
                w.size(1), " x ", w.size(2));
  }
  // Multi-row kernel (see qgemv_moe_mr_*): ULP-level output changes vs the
  // one-simdgroup-per-row kernel, which stays the route for every other
  // format and for K % 256 != 0. The mr grid ceil-divides N over nsg*nr0
  // rows per threadgroup, and tail simdgroups read weight rows past N
  // before the store guards run — so a non-multiple N (never the case for
  // the DSV4 dims) also stays on the safe one-row route.
  const int mr_rows =
      (fmt == "q2_K" && x.scalar_type() != at::kBFloat16) ? 32 : 8;
  if ((fmt == "iq2_xxs" || fmt == "q2_K") && K % 256 == 0 &&
      N % mr_rows == 0) {
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
                                  const at::Tensor& topk_ids_in,
                                  int64_t top_k, int64_t quant_type,
                                  int64_t row, int64_t tokens,
                                  std::optional<double> clamp_limit,
                                  bool soa) {
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
    tk::launch_qgemv_moe_mr_swiglu(e, out, w, input, topk_ids, N, K,
                                   num_tokens, topk, has_clamp, limit,
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
                               int64_t quant_type, int64_t row,
                               int64_t tokens, const at::Tensor& out,
                               bool soa) {
  check_mps(w, "w");
  check_mps(x, "x");
  check_mps(topk_ids_in, "topk_ids");
  check_mps(topk_w_in, "topk_w");
  check_mps(out, "out");
  const std::string fmt = ggml_type_to_format(quant_type);
  TORCH_CHECK(fmt == "q2_K",
              "ggml_moe_a8_vec_sum supports q2_K only, got ", fmt);
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
              "dtype, got ", out.sizes(), " ", out.scalar_type());
  if (soa) {
    TORCH_CHECK((w.size(1) * w.size(2)) % 8 == 0,
                "SoA expert stride must stay 8-byte aligned, got ",
                w.size(1), " x ", w.size(2));
  }
  const int num_tokens = static_cast<int>(tokens);
  const int topk = static_cast<int>(top_k);
  const int N = static_cast<int>(row);
  const int K = static_cast<int>(x.size(1));
  // Tail simdgroups of the mr grid read weight rows past a non-multiple N
  // before the store guards run; there is no one-row fallback here.
  const int sum_rows = x.scalar_type() != at::kBFloat16 ? 32 : 8;
  TORCH_CHECK(N % sum_rows == 0, "sum-folded q2_K kernel needs N divisible "
              "by ", sum_rows, ", got ", N);
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
  TORCH_CHECK(x.dim() == 2 &&
                  x.size(0) == (down ? tokens * top_k : tokens),
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
              "expert stack must be [E, row, K/256*", block_bytes,
              "] raw ", fmt, ", got ", w.sizes());
  const int E = static_cast<int>(w.size(0));
  TORCH_CHECK(E <= 256,
              "map0 runs one thread per expert in one threadgroup, E <= 256, "
              "got ", E);
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
    tk::launch_moe_mm_id(e, out, w, input, tpe, ids,
                         w64 ? work64 : work,
                         w64 ? wcount64 : wcount,
                         w64 ? work_cap64 : work_cap, N, K, num_tokens,
                         topk, fmt, soa, w64);
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
    class_replaceMethod(cls, sel, replacement,
                        method_getTypeEncoding(method));
  };
  swizzle(@selector(commandBuffer), (IMP)cbc_commandBuffer,
          (void*)&g_cbc_orig_plain);
  swizzle(@selector(commandBufferWithUnretainedReferences),
          (IMP)cbc_commandBufferWithUnretainedReferences,
          (void*)&g_cbc_orig_unretained);
  swizzle(@selector(commandBufferWithDescriptor:),
          (IMP)cbc_commandBufferWithDescriptor,
          (void*)&g_cbc_orig_descriptor);
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
      id<MTLResidencySet> set =
          [device newResidencySetWithDescriptor:desc error:&err];
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
  TORCH_CHECK(
      index_q.scalar_type() == at::kHalf ||
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
    e.dispatch(static_cast<int>(tokens), static_cast<int>(H / 8), 1, 256, 1,
               1);
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
  TORCH_CHECK(width >= 1 && width <= 1024, "width must be in [1, 1024], got ",
              width);
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
  TORCH_CHECK(width >= 1 && width <= 1024, "width must be in [1, 1024], got ",
              width);
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
    e.dispatch(static_cast<int>(tokens), static_cast<int>(H / 8), 1, 256, 1,
               1);
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
  at::Tensor out =
      strided_ok ? ring_out("rms_out", {tokens, D}, x.options())
                 : ring_out_like("rms_out_l", input);
  const uint32_t d = static_cast<uint32_t>(D);
  const float eps = static_cast<float>(epsilon);
  const uint64_t in_stride =
      strided_ok ? static_cast<uint64_t>(x.stride(0)) : 0;
  encode("qc_rms_norm", [&](TorchEncoder& e) {
    e.pipeline((w32 ? std::string("qc_rms_norm_w32_")
                    : std::string("qc_rms_norm_")) +
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
  at::Tensor attn_norm_w, ffn_norm_w;      // fp16
  at::Tensor q_norm_w, kv_norm_w;          // pre-.float()ed (fused_qk_rmsnorm.py:89-90)
  // attention
  at::Tensor wqa_wkv_qw, wq_b_qw, wo_b_qw; // gguf qweights
  at::Tensor comp_w;                       // fused_wkv_wgate.weight [2*coff*512,4096] fp16
  at::Tensor ape_bf16;                     // pre-cast .to(bf16).contiguous() (save_partial_states.py:45)
  at::Tensor state_cache, swa_kv_cache, comp_kv_cache;
  at::Tensor attn_sink, cos_sin_cache;
  at::Tensor wo_a_w;                       // wo_a.weight.reshape(8,1024,4096) view (metal.py:120-122)
  // moe
  at::Tensor gate_w;                       // [n_experts,4096] fp16
  at::Tensor router_bias;                  // e_score_correction_bias [n_experts] f32 (optional)
  at::Tensor hash_table;                   // [vocab, top_k] i32 (hash layers, optional)
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
  at::Tensor swa_slot_mapping;             // this layer's swa metadata slot_mapping
  at::Tensor swa_slots, swa_lens;          // metal.py builders' products
  at::Tensor comp_slots, comp_lens;        // dense cr128 tables or comp_none pair
  at::Tensor comp_state_slot_mapping;      // CompressorMetadata.slot_mapping (kind 0)
  int64_t insert_block_size = 0;           // swa metadata block_size (metal.py:60)
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
  at::Tensor q = ggml_mul_mat_vec_a8(L.wq_b_qw, qr, L.wq_b_qt, L.wq_b_qw.size(0))
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
      q, L.kind == 0 ? L.comp_kv_cache : L.swa_kv_cache,
      step_args.comp_slots, step_args.comp_lens, L.swa_kv_cache,
      step_args.swa_slots, step_args.swa_lens, L.attn_sink, L.sm_scale);
  o_padded.copy_(att);
  // o_proj (metal.py:80-124): inverse RoPE -> grouped einsum (wo_a dense on
  // Metal) -> wo_b gguf gemv.
  at::Tensor o_flat = dsv4_o_inv_rope(o_padded, positions, L.cos_sin_cache);
  at::Tensor grouped = o_flat.view(
      {T, L.o_groups, (L.n_heads / L.o_groups) * L.head_dim});
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
  sh = ggml_mul_mat_vec_a8(L.sh_down_qw, sh, L.sh_down_qt, L.sh_down_qw.size(0));
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
    ids_opt = input_ids->scalar_type() == at::kInt
                  ? *input_ids
                  : input_ids->to(at::kInt);
  } else if (L.router_bias.defined()) {
    bias_opt = L.router_bias;
  }
  dsv4_router_topk(scores, topk_w, topk_ids, L.renormalize, L.routed_scaling,
                   bias_opt, hash_opt, ids_opt);
  // routed experts (gguf/fused_moe.py:209, 564-573, 600-621): fused SwiGLU
  // vec kernel, down vec kernel, weighted sum into empty_like(x).
  at::Tensor mo = ggml_moe_a8_vec_swiglu(h, L.w13_qw, topk_ids, L.top_k,
                                         L.w13_qt, L.w13_row, T,
                                         L.swiglu_limit, L.w13_soa);
  // Sum-folded down projection (mirrors the Python route in
  // gguf/fused_moe.py): one kernel instead of down vec + reshape +
  // weighted sum. Shapes outside the folded kernel take the unfused chain.
  at::Tensor fused;
  if (std::string(ggml_type_to_format(L.w2_qt)) == "q2_K" &&
      L.top_k <= 8) {
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

// ---- Gated DeltaNet (Qwen3.5 GDN mixer) ----------------------------------
//
// Thin hosts over the vendored gdn.metal kernels; ABI in tk_launch.h. The
// serving layer runs the post-conv chain on the float32 instantiations
// (state pools are fp32 by contract; fp32 decay avoids bf16 rounding of
// exp(g) ~ 0.99x). The five un-fused hosts (short_conv, qkv_prepare,
// gate_beta, recur, gated_rmsnorm) accept any instantiated dtype and
// dispatch on the activation tensor's dtype; the fused trio is narrower by
// contract (fused_prepare reads projection rows in place, recur_spec runs
// float32, gated_rmsnorm_f32 takes an fp32 y). Per-request fresh-state
// handling is the caller's job (pre-zero the fresh rows, then
// load_initial=true).

at::Tensor gdn_short_conv(const at::Tensor& x, const at::Tensor& weight,
                          const at::Tensor& state_pool,
                          const at::Tensor& cu_seqlens,
                          const at::Tensor& slot_mapping, bool load_initial,
                          bool apply_silu) {
  check_mps(x, "x");
  check_mps(weight, "weight");
  check_mps_strided(state_pool, "state_pool");
  check_mps(cu_seqlens, "cu_seqlens");
  check_mps(slot_mapping, "slot_mapping");
  TORCH_CHECK(x.dim() == 2, "x must be [tokens, channels], got ", x.sizes());
  const int channels = static_cast<int>(x.size(1));
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels,
              "weight must be [channels, kernel], got ", weight.sizes());
  const int kernel_size = static_cast<int>(weight.size(1));
  TORCH_CHECK(kernel_size >= 2 && kernel_size <= 8,
              "gdn_short_conv supports kernel sizes 2..8 (MAX_HISTORY)");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(),
              "weight dtype must match x");
  TORCH_CHECK(state_pool.scalar_type() == at::kFloat,
              "conv state pool must be fp32");
  // With speculation the pool carries kernel-1+num_spec columns per channel;
  // the non-spec ring occupies only the first kernel-1 of them (upstream
  // forces state_len = width-1 on the non-spec paths).
  TORCH_CHECK(
      state_pool.dim() == 3 && state_pool.size(1) == channels &&
          state_pool.size(2) >= kernel_size - 1,
      "conv state pool must be [slots, channels, >=kernel-1] (DS layout), "
      "got ",
      state_pool.sizes());
  const int state_cols = static_cast<int>(state_pool.size(2));
  // Page-packed pools (vLLM bind_kv_cache) are contiguous within a slot with
  // a wider slot stride; the kernel takes that stride explicitly.
  TORCH_CHECK(state_pool.stride(2) == 1 &&
                  state_pool.stride(1) == state_cols,
              "conv state pool rows must be (channels, state_cols) "
              "contiguous");
  const auto conv_stride64 = state_pool.stride(0);
  TORCH_CHECK(conv_stride64 >= (int64_t)channels * state_cols &&
                  conv_stride64 <= std::numeric_limits<int>::max(),
              "conv state slot stride out of range: ", conv_stride64);
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be i32");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kInt,
              "slot_mapping must be i32");
  const int R = static_cast<int>(slot_mapping.size(0));
  TORCH_CHECK(cu_seqlens.numel() >= R + 1, "cu_seqlens must be [R+1]");
  at::Tensor out = ring_out_like("qc_gdn_conv", x);
  encode("qc_gdn_short_conv", [&](TorchEncoder& e) {
    tk::launch_gdn_short_conv(e, x, weight, state_pool, cu_seqlens,
                              slot_mapping, out, R, channels, kernel_size,
                              load_initial ? 1 : 0, apply_silu ? 1 : 0,
                              static_cast<int>(conv_stride64), state_cols,
                              activation_type_name(x));
  });
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gdn_qkv_prepare(
    const at::Tensor& mixed, int64_t num_k_heads, int64_t num_v_heads,
    int64_t head_k_dim, int64_t head_v_dim, double eps, double q_scale,
    double k_scale) {
  check_mps(mixed, "mixed");
  const int Hk = static_cast<int>(num_k_heads);
  const int Hv = static_cast<int>(num_v_heads);
  const int Dk = static_cast<int>(head_k_dim);
  const int Dv = static_cast<int>(head_v_dim);
  TORCH_CHECK((Dk == 64 || Dk == 128) && (Dv == 64 || Dv == 128),
              "gdn_qkv_prepare instantiations cover head dims {64,128}");
  TORCH_CHECK(mixed.dim() == 2 && mixed.size(1) == 2 * Hk * Dk + Hv * Dv,
              "mixed must be [tokens, 2*Hk*Dk + Hv*Dv], got ", mixed.sizes());
  const int tokens = static_cast<int>(mixed.size(0));
  at::Tensor q =
      ring_out("qc_gdn_q", {tokens, (int64_t)Hk * Dk}, mixed.options());
  at::Tensor k =
      ring_out("qc_gdn_k", {tokens, (int64_t)Hk * Dk}, mixed.options());
  at::Tensor v =
      ring_out("qc_gdn_v", {tokens, (int64_t)Hv * Dv}, mixed.options());
  encode("qc_gdn_qkv_prepare", [&](TorchEncoder& e) {
    tk::launch_gdn_qkv_prepare(e, mixed, q, k, v, tokens, Hk, Hv, Dk, Dv,
                               static_cast<float>(eps),
                               static_cast<float>(q_scale),
                               static_cast<float>(k_scale),
                               activation_type_name(mixed));
  });
  return {q, k, v};
}

std::tuple<at::Tensor, at::Tensor> gdn_gate_beta(const at::Tensor& a,
                                                 const at::Tensor& b,
                                                 const at::Tensor& A_log,
                                                 const at::Tensor& dt_bias) {
  check_mps(a, "a");
  check_mps(b, "b");
  check_mps(A_log, "A_log");
  check_mps(dt_bias, "dt_bias");
  TORCH_CHECK(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type(),
              "a and b must match");
  const int heads = static_cast<int>(a.size(-1));
  TORCH_CHECK(A_log.scalar_type() == at::kFloat &&
                  dt_bias.scalar_type() == at::kFloat,
              "A_log and dt_bias must be fp32");
  TORCH_CHECK(A_log.numel() == heads && dt_bias.numel() == heads,
              "A_log/dt_bias must be [heads]");
  const auto n = static_cast<uint32_t>(a.numel());
  at::Tensor decay =
      ring_out("qc_gdn_decay", a.sizes(), a.options().dtype(at::kFloat));
  at::Tensor beta =
      ring_out("qc_gdn_beta", a.sizes(), a.options().dtype(at::kFloat));
  encode("qc_gdn_gate_beta", [&](TorchEncoder& e) {
    tk::launch_gdn_gate_beta(e, a, b, A_log, dt_bias, decay, beta, n, heads,
                             activation_type_name(a));
  });
  return {decay, beta};
}

at::Tensor gdn_recur(const at::Tensor& q, const at::Tensor& k,
                     const at::Tensor& v, const at::Tensor& g,
                     const at::Tensor& beta, const at::Tensor& state_pool,
                     const at::Tensor& cu_seqlens,
                     const at::Tensor& slot_mapping, int64_t num_k_heads,
                     int64_t num_v_heads, int64_t head_k_dim,
                     int64_t head_v_dim, bool load_initial) {
  check_mps(q, "q");
  check_mps(k, "k");
  check_mps(v, "v");
  check_mps(g, "g");
  check_mps(beta, "beta");
  check_mps_strided(state_pool, "state_pool");
  check_mps(cu_seqlens, "cu_seqlens");
  check_mps(slot_mapping, "slot_mapping");
  const int Hk = static_cast<int>(num_k_heads);
  const int Hv = static_cast<int>(num_v_heads);
  const int Dk = static_cast<int>(head_k_dim);
  const int Dv = static_cast<int>(head_v_dim);
  TORCH_CHECK(Dk == 64 || Dk == 128, "gdn_recur supports Dk in {64,128}");
  TORCH_CHECK(Hv % Hk == 0, "Hv must be a multiple of Hk (GQA broadcast)");
  const int tokens = static_cast<int>(q.size(0));
  TORCH_CHECK(q.dim() == 2 && q.size(1) == (int64_t)Hk * Dk, "q must be ",
              "[tokens, Hk*Dk], got ", q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && k.scalar_type() == q.scalar_type(),
              "k must match q");
  TORCH_CHECK(v.dim() == 2 && v.size(0) == tokens &&
                  v.size(1) == (int64_t)Hv * Dv &&
                  v.scalar_type() == q.scalar_type(),
              "v must be [tokens, Hv*Dv] of q's dtype, got ", v.sizes());
  // g/beta are read through the activation type in the kernel: on the fp32
  // serving chain they are gate_beta's fp32 outputs verbatim.
  TORCH_CHECK(g.numel() == (int64_t)tokens * Hv &&
                  g.scalar_type() == q.scalar_type(),
              "g must be [tokens, Hv] of q's dtype");
  TORCH_CHECK(beta.numel() == (int64_t)tokens * Hv &&
                  beta.scalar_type() == q.scalar_type(),
              "beta must be [tokens, Hv] of q's dtype");
  TORCH_CHECK(state_pool.scalar_type() == at::kFloat,
              "ssm state pool must be fp32");
  TORCH_CHECK(state_pool.dim() == 4 && state_pool.size(1) == Hv &&
                  state_pool.size(2) == Dv && state_pool.size(3) == Dk,
              "ssm state pool must be [slots, Hv, Dv, Dk], got ",
              state_pool.sizes());
  // Page-packed pools are contiguous within a slot, wider between slots.
  TORCH_CHECK(state_pool.stride(3) == 1 && state_pool.stride(2) == Dk &&
                  state_pool.stride(1) == (int64_t)Dv * Dk,
              "ssm state pool rows must be (Hv, Dv, Dk) contiguous");
  const auto ssm_stride64 = state_pool.stride(0);
  TORCH_CHECK(ssm_stride64 >= (int64_t)Hv * Dv * Dk &&
                  ssm_stride64 <= std::numeric_limits<int>::max(),
              "ssm state slot stride out of range: ", ssm_stride64);
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be i32");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kInt,
              "slot_mapping must be i32");
  const int R = static_cast<int>(slot_mapping.size(0));
  TORCH_CHECK(cu_seqlens.numel() >= R + 1, "cu_seqlens must be [R+1]");
  at::Tensor y = ring_out("qc_gdn_y", {tokens, (int64_t)Hv * Dv}, q.options());
  encode("qc_gdn_recur", [&](TorchEncoder& e) {
    tk::launch_gdn_recur(e, q, k, v, g, beta, state_pool, cu_seqlens,
                         slot_mapping, y, R, Hk, Hv, Dv, Dk,
                         load_initial ? 1 : 0, static_cast<int>(ssm_stride64),
                         activation_type_name(q));
  });
  return y;
}

at::Tensor gdn_gated_rmsnorm(const at::Tensor& y, const at::Tensor& z,
                             const at::Tensor& weight, double eps) {
  check_mps(y, "y");
  check_mps(z, "z");
  check_mps(weight, "weight");
  TORCH_CHECK(y.dim() == 2, "y must be [rows, D], got ", y.sizes());
  const int D = static_cast<int>(y.size(1));
  TORCH_CHECK(D == 64 || D == 128,
              "gdn_gated_rmsnorm instantiations cover D in {64,128}");
  TORCH_CHECK(z.sizes() == y.sizes() && z.scalar_type() == y.scalar_type(),
              "z must match y");
  TORCH_CHECK(weight.numel() == D && weight.scalar_type() == y.scalar_type(),
              "weight must be [D] of y's dtype");
  const int rows = static_cast<int>(y.size(0));
  at::Tensor out = ring_out_like("qc_gdn_norm", y);
  encode("qc_gdn_gated_rmsnorm", [&](TorchEncoder& e) {
    tk::launch_gdn_gated_rmsnorm(e, y, z, weight, out, rows, D,
                                 static_cast<float>(eps),
                                 activation_type_name(y));
  });
  return out;
}

// Fused decode-side preparation: one dispatch covering gdn_short_conv +
// gdn_qkv_prepare + gdn_gate_beta, reading the projection outputs in place
// (qkvz rows [q|k|v|z], ba rows [b|a], both strided). Returns the fp32
// serving-chain tensors (q, k, v, decay, beta) that gdn_recur consumes.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
gdn_fused_prepare(const at::Tensor& qkvz, const at::Tensor& ba,
                  const at::Tensor& conv_w, const at::Tensor& conv_state_pool,
                  const at::Tensor& cu_seqlens, const at::Tensor& slot_mapping,
                  const at::Tensor& A_log, const at::Tensor& dt_bias,
                  int64_t num_k_heads, int64_t num_v_heads,
                  int64_t head_k_dim, int64_t head_v_dim, double eps,
                  double q_scale, double k_scale, bool load_initial,
                  const c10::optional<at::Tensor>& num_accepted) {
  check_mps_strided(qkvz, "qkvz");
  check_mps_strided(ba, "ba");
  check_mps(conv_w, "conv_w");
  check_mps_strided(conv_state_pool, "conv_state_pool");
  check_mps(cu_seqlens, "cu_seqlens");
  check_mps(slot_mapping, "slot_mapping");
  check_mps(A_log, "A_log");
  check_mps(dt_bias, "dt_bias");
  const int Hk = static_cast<int>(num_k_heads);
  const int Hv = static_cast<int>(num_v_heads);
  const int Dk = static_cast<int>(head_k_dim);
  const int Dv = static_cast<int>(head_v_dim);
  TORCH_CHECK((Dk == 64 || Dk == 128) && (Dv == 64 || Dv == 128),
              "gdn_fused_prepare instantiations cover head dims {64,128}");
  const int channels = 2 * Hk * Dk + Hv * Dv;
  TORCH_CHECK(qkvz.dim() == 2 && qkvz.size(1) >= channels &&
                  qkvz.stride(1) == 1,
              "qkvz must be [tokens, >=2*Hk*Dk+Hv*Dv] with unit column "
              "stride, got ",
              qkvz.sizes());
  const auto qkvz_stride64 = qkvz.stride(0);
  TORCH_CHECK(qkvz_stride64 >= channels &&
                  qkvz_stride64 <= std::numeric_limits<int>::max(),
              "qkvz row stride out of range: ", qkvz_stride64);
  const int tokens = static_cast<int>(qkvz.size(0));
  TORCH_CHECK(ba.dim() == 2 && ba.size(0) == tokens &&
                  ba.size(1) >= 2 * Hv && ba.stride(1) == 1 &&
                  ba.scalar_type() == qkvz.scalar_type(),
              "ba must be [tokens, >=2*Hv] of qkvz's dtype with unit column "
              "stride, got ",
              ba.sizes());
  const auto ba_stride64 = ba.stride(0);
  TORCH_CHECK(ba_stride64 >= 2 * Hv &&
                  ba_stride64 <= std::numeric_limits<int>::max(),
              "ba row stride out of range: ", ba_stride64);
  TORCH_CHECK(conv_w.scalar_type() == at::kFloat && conv_w.dim() == 2 &&
                  conv_w.size(0) == channels && conv_w.is_contiguous(),
              "conv_w must be contiguous fp32 [channels, kernel], got ",
              conv_w.sizes());
  const int kernel_size = static_cast<int>(conv_w.size(1));
  TORCH_CHECK(kernel_size >= 2 && kernel_size <= 8,
              "gdn_fused_prepare supports kernel sizes 2..8 (MAX_HISTORY)");
  TORCH_CHECK(conv_state_pool.scalar_type() == at::kFloat,
              "conv state pool must be fp32");
  // With speculation the pool carries kernel-1+num_spec columns per channel.
  TORCH_CHECK(conv_state_pool.dim() == 3 &&
                  conv_state_pool.size(1) == channels &&
                  conv_state_pool.size(2) >= kernel_size - 1,
              "conv state pool must be [slots, channels, >=kernel-1] (DS "
              "layout), got ",
              conv_state_pool.sizes());
  const int state_cols = static_cast<int>(conv_state_pool.size(2));
  TORCH_CHECK(conv_state_pool.stride(2) == 1 &&
                  conv_state_pool.stride(1) == state_cols,
              "conv state pool rows must be (channels, state_cols) "
              "contiguous");
  const auto conv_stride64 = conv_state_pool.stride(0);
  TORCH_CHECK(conv_stride64 >= (int64_t)channels * state_cols &&
                  conv_stride64 <= std::numeric_limits<int>::max(),
              "conv state slot stride out of range: ", conv_stride64);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat &&
                  dt_bias.scalar_type() == at::kFloat &&
                  A_log.numel() == Hv && dt_bias.numel() == Hv,
              "A_log/dt_bias must be fp32 [Hv]");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be i32");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kInt,
              "slot_mapping must be i32");
  const int R = static_cast<int>(slot_mapping.size(0));
  TORCH_CHECK(cu_seqlens.numel() >= R + 1, "cu_seqlens must be [R+1]");
  const bool spec_mode = num_accepted.has_value();
  at::Tensor accepted;
  if (spec_mode) {
    accepted = num_accepted.value();
    check_mps(accepted, "num_accepted");
    TORCH_CHECK(accepted.scalar_type() == at::kInt &&
                    accepted.is_contiguous() && accepted.numel() >= R,
                "num_accepted must be contiguous i32 [R]");
    // The speculative window (shifted old history + every new token) must
    // fit in the state row for the longest request.
    TORCH_CHECK(state_cols >= kernel_size - 1,
                "spec conv needs kernel-1+num_spec state columns");
  } else {
    // The kernel only dereferences num_accepted in spec mode; bind a
    // persistent 1-element dummy so the pipeline's buffer table is complete.
    static at::Tensor dummy = at::zeros(
        {1}, at::TensorOptions().dtype(at::kInt).device(at::kMPS));
    accepted = dummy;
  }
  const auto f32 = qkvz.options().dtype(at::kFloat);
  at::Tensor q = ring_out("qc_gdn_q", {tokens, (int64_t)Hk * Dk}, f32);
  at::Tensor k = ring_out("qc_gdn_k", {tokens, (int64_t)Hk * Dk}, f32);
  at::Tensor v = ring_out("qc_gdn_v", {tokens, (int64_t)Hv * Dv}, f32);
  at::Tensor decay = ring_out("qc_gdn_decay", {tokens, (int64_t)Hv}, f32);
  at::Tensor beta = ring_out("qc_gdn_beta", {tokens, (int64_t)Hv}, f32);
  encode("qc_gdn_fused_prepare", [&](TorchEncoder& e) {
    tk::launch_gdn_fused_prepare(
        e, qkvz, ba, conv_w, conv_state_pool, cu_seqlens, slot_mapping, A_log,
        dt_bias, q, k, v, decay, beta, R, Hk, Hv, Dk, Dv, kernel_size,
        load_initial ? 1 : 0, static_cast<int>(qkvz_stride64),
        static_cast<int>(ba_stride64), static_cast<int>(conv_stride64),
        static_cast<float>(eps), static_cast<float>(q_scale),
        static_cast<float>(k_scale), state_cols, accepted, spec_mode ? 1 : 0,
        activation_type_name(qkvz));
  });
  return {q, k, v, decay, beta};
}

// Speculative-verify recurrence: per-request rows of num_spec+1 state slots.
// Initial state loads from slot_table[r, num_accepted[r]-1]; the state after
// every timestep t is checkpointed to slot_table[r, t] (slot ids <= 0 are the
// null block). The per-timestep math is gdn_recur's verbatim.
at::Tensor gdn_recur_spec(const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, const at::Tensor& g,
                          const at::Tensor& beta, const at::Tensor& state_pool,
                          const at::Tensor& cu_seqlens,
                          const at::Tensor& slot_table,
                          const at::Tensor& num_accepted, int64_t num_k_heads,
                          int64_t num_v_heads, int64_t head_k_dim,
                          int64_t head_v_dim) {
  check_mps(q, "q");
  check_mps(k, "k");
  check_mps(v, "v");
  check_mps(g, "g");
  check_mps(beta, "beta");
  check_mps_strided(state_pool, "state_pool");
  check_mps(cu_seqlens, "cu_seqlens");
  check_mps_strided(slot_table, "slot_table");
  check_mps(num_accepted, "num_accepted");
  const int Hk = static_cast<int>(num_k_heads);
  const int Hv = static_cast<int>(num_v_heads);
  const int Dk = static_cast<int>(head_k_dim);
  const int Dv = static_cast<int>(head_v_dim);
  TORCH_CHECK(Dk == 64 || Dk == 128,
              "gdn_recur_spec supports Dk in {64,128}");
  TORCH_CHECK(Hv % Hk == 0, "Hv must be a multiple of Hk (GQA broadcast)");
  const int tokens = static_cast<int>(q.size(0));
  TORCH_CHECK(q.dim() == 2 && q.size(1) == (int64_t)Hk * Dk, "q must be ",
              "[tokens, Hk*Dk], got ", q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && k.scalar_type() == q.scalar_type(),
              "k must match q");
  TORCH_CHECK(v.dim() == 2 && v.size(0) == tokens &&
                  v.size(1) == (int64_t)Hv * Dv &&
                  v.scalar_type() == q.scalar_type(),
              "v must be [tokens, Hv*Dv] of q's dtype, got ", v.sizes());
  TORCH_CHECK(g.numel() == (int64_t)tokens * Hv &&
                  g.scalar_type() == q.scalar_type(),
              "g must be [tokens, Hv] of q's dtype");
  TORCH_CHECK(beta.numel() == (int64_t)tokens * Hv &&
                  beta.scalar_type() == q.scalar_type(),
              "beta must be [tokens, Hv] of q's dtype");
  TORCH_CHECK(state_pool.scalar_type() == at::kFloat,
              "ssm state pool must be fp32");
  TORCH_CHECK(state_pool.dim() == 4 && state_pool.size(1) == Hv &&
                  state_pool.size(2) == Dv && state_pool.size(3) == Dk,
              "ssm state pool must be [slots, Hv, Dv, Dk], got ",
              state_pool.sizes());
  TORCH_CHECK(state_pool.stride(3) == 1 && state_pool.stride(2) == Dk &&
                  state_pool.stride(1) == (int64_t)Dv * Dk,
              "ssm state pool rows must be (Hv, Dv, Dk) contiguous");
  const auto ssm_stride64 = state_pool.stride(0);
  TORCH_CHECK(ssm_stride64 >= (int64_t)Hv * Dv * Dk &&
                  ssm_stride64 <= std::numeric_limits<int>::max(),
              "ssm state slot stride out of range: ", ssm_stride64);
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be i32");
  TORCH_CHECK(slot_table.scalar_type() == at::kInt && slot_table.dim() == 2 &&
                  slot_table.stride(1) == 1,
              "slot_table must be i32 [R, S] with unit column stride");
  const int R = static_cast<int>(slot_table.size(0));
  const auto table_stride64 = slot_table.stride(0);
  TORCH_CHECK(table_stride64 >= slot_table.size(1) &&
                  table_stride64 <= std::numeric_limits<int>::max(),
              "slot_table row stride out of range: ", table_stride64);
  TORCH_CHECK(num_accepted.scalar_type() == at::kInt &&
                  num_accepted.is_contiguous() && num_accepted.numel() >= R,
              "num_accepted must be contiguous i32 [R]");
  TORCH_CHECK(cu_seqlens.numel() >= R + 1, "cu_seqlens must be [R+1]");
  at::Tensor y =
      ring_out("qc_gdn_y_spec", {tokens, (int64_t)Hv * Dv}, q.options());
  encode("qc_gdn_recur_spec", [&](TorchEncoder& e) {
    tk::launch_gdn_recur_spec(e, q, k, v, g, beta, state_pool, cu_seqlens,
                              slot_table, num_accepted, y, R, Hk, Hv, Dv, Dk,
                              static_cast<int>(ssm_stride64),
                              static_cast<int>(table_stride64),
                              activation_type_name(q));
  });
  return y;
}

// Gated RMSNorm over the fp32 recurrence output with the z gate read in
// place from the projection rows: y fp32 [tokens*Hv, D], z strided
// [tokens, Hv, D] activation-dtype view. Returns [tokens, Hv*D] in z's dtype.
at::Tensor gdn_gated_rmsnorm_f32(const at::Tensor& y, const at::Tensor& z,
                                 const at::Tensor& weight, double eps) {
  check_mps(y, "y");
  check_mps_strided(z, "z");
  check_mps(weight, "weight");
  TORCH_CHECK(y.scalar_type() == at::kFloat && y.dim() == 2,
              "y must be fp32 [rows, D], got ", y.sizes());
  const int D = static_cast<int>(y.size(1));
  TORCH_CHECK(D == 64 || D == 128,
              "gdn_gated_rmsnorm_f32 instantiations cover D in {64,128}");
  TORCH_CHECK(z.dim() == 3 && z.size(2) == D && z.stride(2) == 1 &&
                  z.stride(1) == D,
              "z must be [tokens, Hv, D] with contiguous (Hv, D) rows, got ",
              z.sizes());
  const int tokens = static_cast<int>(z.size(0));
  const int Hv = static_cast<int>(z.size(1));
  const int rows = static_cast<int>(y.size(0));
  TORCH_CHECK(rows == tokens * Hv, "y rows must equal tokens*Hv, got ", rows,
              " vs ", tokens, "*", Hv);
  const auto z_stride64 = z.stride(0);
  TORCH_CHECK(z_stride64 >= (int64_t)Hv * D &&
                  z_stride64 <= std::numeric_limits<int>::max(),
              "z token stride out of range: ", z_stride64);
  TORCH_CHECK(weight.numel() == D && weight.scalar_type() == z.scalar_type(),
              "weight must be [D] of z's dtype");
  at::Tensor out =
      ring_out("qc_gdn_norm", {(int64_t)tokens, (int64_t)Hv * D}, z.options());
  encode("qc_gdn_gated_rmsnorm_f32", [&](TorchEncoder& e) {
    tk::launch_gdn_gated_rmsnorm_f32(e, y, z, weight, out, rows, Hv, D,
                                     static_cast<int>(z_stride64),
                                     static_cast<float>(eps),
                                     activation_type_name(z));
  });
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
  m.def("muse_step_run", &muse_step_run,
        "Encode the whole decoder stack for one decode step into a single "
        "command buffer; x is updated in place",
        pybind11::arg("x"), pybind11::arg("positions"),
        pybind11::arg("bt_local"), pybind11::arg("sl_local"),
        pybind11::arg("slot_local"), pybind11::arg("bt_full"),
        pybind11::arg("sl_full"), pybind11::arg("slot_full"));

  m.def("muse_q38_init", &muse_q38_init,
        "Register geometry and allocate scratch for the Qwen3.8 hybrid "
        "single-CB decode step",
        pybind11::arg("num_layers"), pybind11::arg("hidden"),
        pybind11::arg("heads"), pybind11::arg("kv_heads"),
        pybind11::arg("head_dim"), pybind11::arg("rot_dim"),
        pybind11::arg("attn_scale"), pybind11::arg("gdn_k_heads"),
        pybind11::arg("gdn_v_heads"), pybind11::arg("gdn_k_dim"),
        pybind11::arg("gdn_v_dim"), pybind11::arg("inter"),
        pybind11::arg("eps"), pybind11::arg("max_rows"),
        pybind11::arg("max_blocks"), pybind11::arg("block_size"),
        pybind11::arg("final_norm_w"), pybind11::arg("aux_layers"),
        pybind11::arg("ref"));
  m.def("muse_q38_layer_gdn", &muse_q38_layer_gdn,
        "Register one GDN decoder layer (mixer + MLP + norm seams)",
        pybind11::arg("idx"), pybind11::arg("in_norm_w"),
        pybind11::arg("post_norm_w"), pybind11::arg("qkvz_t"),
        pybind11::arg("qkvz_fmt"), pybind11::arg("ba_t"),
        pybind11::arg("ba_fmt"), pybind11::arg("conv_w"),
        pybind11::arg("A_log"), pybind11::arg("dt_bias"),
        pybind11::arg("gated_norm_w"), pybind11::arg("out_t"),
        pybind11::arg("out_fmt"), pybind11::arg("conv_state"),
        pybind11::arg("ssm_state"), pybind11::arg("prep_eps"),
        pybind11::arg("q_scale"), pybind11::arg("k_scale"),
        pybind11::arg("norm_eps"), pybind11::arg("gu_t"),
        pybind11::arg("gu_fmt"), pybind11::arg("down_t"),
        pybind11::arg("down_fmt"));
  m.def("muse_q38_layer_attn", &muse_q38_layer_attn,
        "Register one gated full-attention decoder layer (mixer + MLP + "
        "norm seams)",
        pybind11::arg("idx"), pybind11::arg("in_norm_w"),
        pybind11::arg("post_norm_w"), pybind11::arg("qkv_t"),
        pybind11::arg("qkv_fmt"), pybind11::arg("q_norm_w"),
        pybind11::arg("k_norm_w"), pybind11::arg("cos_sin"),
        pybind11::arg("o_t"), pybind11::arg("o_fmt"),
        pybind11::arg("key_cache"), pybind11::arg("value_cache"),
        pybind11::arg("block_mult"), pybind11::arg("gu_t"),
        pybind11::arg("gu_fmt"), pybind11::arg("down_t"),
        pybind11::arg("down_fmt"));
  m.def("muse_q38_run", &muse_q38_run,
        "Encode the qwen38 decode-step forward (first layers_cap layers; "
        "<=0 = all) into one command buffer; x/residual_out updated in "
        "place",
        pybind11::arg("x"), pybind11::arg("residual_out"),
        pybind11::arg("positions"), pybind11::arg("block_table"),
        pybind11::arg("seq_lens"), pybind11::arg("attn_slots"),
        pybind11::arg("attn_max_context"), pybind11::arg("attn_group"),
        pybind11::arg("spec_cu"), pybind11::arg("conv_slots"),
        pybind11::arg("slot_table"), pybind11::arg("num_accepted"),
        pybind11::arg("q_len"), pybind11::arg("layers_cap") = 0,
        pybind11::arg("aux_out") = pybind11::none(),
        pybind11::arg("debug_out") = pybind11::none(),
        pybind11::arg("debug_layer") = 0);

  m.def("paged_attention", &paged_attention,
        "Dense/GQA paged attention decode over the block-table KV cache. "
        "window > 0 limits each query to the last `window` positions. "
        "max_context > 0 is a host-side bound on the batch's longest "
        "context, used to size the D=256 split-K partitions without a "
        "device sync.",
        pybind11::arg("q"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_table"),
        pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("window") = 0, pybind11::arg("max_context") = 0);

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

  m.def("qc_kv_cache_scatter", &qc_kv_cache_scatter,
        "Paged KV insert: one dispatch writes K and V rows by slot "
        "(block_mult=2 for the page-local dense layout; slot<0 rows "
        "skipped)",
        pybind11::arg("key"), pybind11::arg("value"),
        pybind11::arg("slot_mapping"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("num_heads"),
        pybind11::arg("head_size"), pybind11::arg("block_size"),
        pybind11::arg("block_mult"));

  m.def("qc_qk_norm_rope_gate", &qc_qk_norm_rope_gate,
        "Fused Qwen3-Next attention prep: gated-q split + gemma QK-RMSNorm "
        "+ partial NeoX RoPE + gate de-interleave (bit-exact vs the eager "
        "Metal chain)",
        pybind11::arg("qkv"), pybind11::arg("q_w"), pybind11::arg("k_w"),
        pybind11::arg("cos_sin"), pybind11::arg("positions"),
        pybind11::arg("num_q_heads"), pybind11::arg("num_k_heads"),
        pybind11::arg("head_dim"), pybind11::arg("rot_dim"),
        pybind11::arg("eps"));

  m.def("qc_dflash_conv", &qc_dflash_conv,
        "Fused DFlash2 grouped dynamic conv (block-local taps, position "
        "mask); one dispatch replacing the eager pad/mask/mul chain",
        pybind11::arg("x"), pybind11::arg("delta"), pybind11::arg("base"),
        pybind11::arg("block_size"));

  m.def("gdn_short_conv", &gdn_short_conv,
        "GDN causal short conv over varlen packed tokens with a persistent "
        "fp32 [slots, channels, kernel-1] history pool; slot<0 zeroes the "
        "output rows and skips the state",
        pybind11::arg("x"), pybind11::arg("weight"),
        pybind11::arg("state_pool"), pybind11::arg("cu_seqlens"),
        pybind11::arg("slot_mapping"), pybind11::arg("load_initial"),
        pybind11::arg("apply_silu"));
  m.def("gdn_qkv_prepare", &gdn_qkv_prepare,
        "Split the packed post-conv q|k|v row and q/k-normalize in one "
        "dispatch. eps/q_scale/k_scale are in the kernel's rms form; see the "
        "serving layer for the exact l2norm recovery constants",
        pybind11::arg("mixed"), pybind11::arg("num_k_heads"),
        pybind11::arg("num_v_heads"), pybind11::arg("head_k_dim"),
        pybind11::arg("head_v_dim"), pybind11::arg("eps"),
        pybind11::arg("q_scale"), pybind11::arg("k_scale"));
  m.def("gdn_gate_beta", &gdn_gate_beta,
        "decay = exp(-exp(A_log)*softplus(a + dt_bias)), beta = sigmoid(b); "
        "fp32 outputs",
        pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("A_log"),
        pybind11::arg("dt_bias"));
  m.def("gdn_recur", &gdn_recur,
        "Gated delta-rule recurrence over varlen packed tokens against the "
        "persistent fp32 [slots, Hv, Dv, Dk] state pool (in-place update)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("state_pool"),
        pybind11::arg("cu_seqlens"), pybind11::arg("slot_mapping"),
        pybind11::arg("num_k_heads"), pybind11::arg("num_v_heads"),
        pybind11::arg("head_k_dim"), pybind11::arg("head_v_dim"),
        pybind11::arg("load_initial"));
  m.def("gdn_gated_rmsnorm", &gdn_gated_rmsnorm,
        "out = rmsnorm(y)*weight*silu(z) per [rows, D] row "
        "(norm-before-gate)",
        pybind11::arg("y"), pybind11::arg("z"), pybind11::arg("weight"),
        pybind11::arg("eps"));
  m.def("gdn_fused_prepare", &gdn_fused_prepare,
        "Fused GDN decode preparation: short conv + silu, q/k normalize, v, "
        "and decay/beta in one dispatch, reading the qkvz/ba projection rows "
        "in place (strided); returns fp32 (q, k, v, decay, beta)",
        pybind11::arg("qkvz"), pybind11::arg("ba"), pybind11::arg("conv_w"),
        pybind11::arg("conv_state_pool"), pybind11::arg("cu_seqlens"),
        pybind11::arg("slot_mapping"), pybind11::arg("A_log"),
        pybind11::arg("dt_bias"), pybind11::arg("num_k_heads"),
        pybind11::arg("num_v_heads"), pybind11::arg("head_k_dim"),
        pybind11::arg("head_v_dim"), pybind11::arg("eps"),
        pybind11::arg("q_scale"), pybind11::arg("k_scale"),
        pybind11::arg("load_initial"),
        pybind11::arg("num_accepted") = pybind11::none());
  m.def("gdn_recur_spec", &gdn_recur_spec,
        "Speculative-verify gated delta-rule recurrence: initial state from "
        "slot_table[r, num_accepted[r]-1], per-timestep state checkpoints "
        "into slot_table[r, t]",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("state_pool"),
        pybind11::arg("cu_seqlens"), pybind11::arg("slot_table"),
        pybind11::arg("num_accepted"), pybind11::arg("num_k_heads"),
        pybind11::arg("num_v_heads"), pybind11::arg("head_k_dim"),
        pybind11::arg("head_v_dim"));
  m.def("gdn_gated_rmsnorm_f32", &gdn_gated_rmsnorm_f32,
        "Gated RMSNorm over the fp32 recurrence output (rounded to z's dtype "
        "in-register first) with z read in place from the strided "
        "[tokens, Hv, D] projection view; returns [tokens, Hv*D] in z's dtype",
        pybind11::arg("y"), pybind11::arg("z"), pybind11::arg("weight"),
        pybind11::arg("eps"));

  m.def("moe_weighted_sum", &moe_weighted_sum,
        "Weighted sum of MoE expert rows into the output hidden states "
        "(one dispatch, torch-eager numerics)",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("y"));

  m.def("dsv4_router_topk", &dsv4_router_topk,
        "Fused router top-k over pre-softplussed scores: sqrt + bias/hash "
        "select + top-k + renorm + scale (one dispatch)",
        pybind11::arg("gating"), pybind11::arg("out_w"),
        pybind11::arg("out_ids"), pybind11::arg("renormalize"),
        pybind11::arg("scaling"),
        pybind11::arg("bias") = pybind11::none(),
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
        "Dense-causal prefill MMA FA over pre-decoded axes",
        pybind11::arg("q"), pybind11::arg("kc"), pybind11::arg("ks"),
        pybind11::arg("lens_c"), pybind11::arg("lo_s"),
        pybind11::arg("hi_s"), pybind11::arg("sinks"),
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

  m.def("turboquant_attention_splitk_metal", &turboquant_attention_splitk_metal,
        "Split-K TurboQuant paged decode attention on Metal",
        pybind11::arg("q"), pybind11::arg("kv_cache"),
        pybind11::arg("block_table"), pybind11::arg("lengths"),
        pybind11::arg("centroids"), pybind11::arg("signs"),
        pybind11::arg("sinks"), pybind11::arg("scale"),
        pybind11::arg("num_kv_heads"), pybind11::arg("k_bits"),
        pybind11::arg("k_signed"), pybind11::arg("v_bits"),
        pybind11::arg("max_context"));

  m.def("turboquant_dequant_kv_metal", &turboquant_dequant_kv_metal,
        "Dequant cached TurboQuant K/V into dense buffers on Metal",
        pybind11::arg("kv_cache"), pybind11::arg("slots"),
        pybind11::arg("centroids"), pybind11::arg("signs"),
        pybind11::arg("k_out"), pybind11::arg("v_out"), pybind11::arg("k_bits"),
        pybind11::arg("k_signed"), pybind11::arg("v_bits"));

  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8,
        "GGUF weight-only GEMV over raw quantized blocks", pybind11::arg("w"),
        pybind11::arg("x"), pybind11::arg("quant_type"), pybind11::arg("row"));

  m.def("fp8ch_mul_mat_vec", &fp8ch_mul_mat_vec,
        "Compressed-tensors FP8-per-channel W8A16 GEMV (planar e4m3 rows)",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("w_scale"));

  m.def("nvfp4_mul_mat_vec", &nvfp4_mul_mat_vec,
        "Compressed-tensors NVFP4 W4A16 GEMV (planar packed e2m1 + e4m3 "
        "group scales + fp32 global)",
        pybind11::arg("w"), pybind11::arg("x"), pybind11::arg("w_scale"),
        pybind11::arg("global_scale"));

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
        "Weighted RMS norm (vllm ir.ops.rms_norm numerics)",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("epsilon"));

  m.def("add_rms_norm", &add_rms_norm,
        "Fused residual add + weighted RMS norm: returns (normed, "
        "summed_residual); bit-identical to the eager add + rms_norm chain.",
        pybind11::arg("x"), pybind11::arg("residual"), pybind11::arg("weight"),
        pybind11::arg("epsilon"));

  m.def("gemma_rms_norm", &gemma_rms_norm,
        "Gemma-semantics RMS norm: y = bf16(x_hat32 * (w32 + 1)), fp32 weight "
        "multiply, single final round (ir.ops.rms_norm with w + 1).",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("epsilon"));

  m.def("gemma_add_rms_norm", &gemma_add_rms_norm,
        "Gemma-semantics fused residual add + RMS norm: statistic over the "
        "unrounded fp32 sum, residual rounded once; returns (normed, "
        "summed_residual) (ir.ops.fused_add_rms_norm with w + 1).",
        pybind11::arg("x"), pybind11::arg("residual"), pybind11::arg("weight"),
        pybind11::arg("epsilon"));

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
}
