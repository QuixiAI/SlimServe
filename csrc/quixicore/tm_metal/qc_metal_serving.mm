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
  encode([&](TorchEncoder& e) {
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

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
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
  encode([&](TorchEncoder& e) {
    if (fixed) {
      tk::launch_rms_norm(e, x, weight, out, M, D, static_cast<float>(eps));
    } else {
      tk::launch_rms_norm_dyn(e, x, weight, out, M, D, static_cast<float>(eps));
    }
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

  // Multi-row blocks ride the weight-stationary qgemv_mm variants so the
  // quantized bytes are read once per block instead of once per row.
  // Instantiated row counts, largest-first; 17 is the speculative-verify
  // width (k+1) and gets a single dispatch.
  static const int kMMRows[] = {17, 16, 8, 4, 2};
  const bool has_mm = (fmt == "q4_0" || fmt == "q8_0" || fmt == "q4_K" ||
                       fmt == "q5_K" || fmt == "q6_K") &&
                      type_name != "float32";
  encode([&](TorchEncoder& e) {
    const int batch = static_cast<int>(x.size(0));
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

  m.def("rms_norm", &rms_norm,
        "Single-dispatch bf16 RMSNorm over contiguous [rows, D]",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("eps"));

  m.def("paged_attention", &paged_attention,
        "Dense/GQA paged attention decode over the block-table KV cache. "
        "window > 0 limits each query to the last `window` positions.",
        pybind11::arg("q"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("block_table"),
        pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("window") = 0);

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
