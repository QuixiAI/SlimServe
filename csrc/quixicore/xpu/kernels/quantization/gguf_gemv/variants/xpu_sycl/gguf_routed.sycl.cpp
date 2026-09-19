// GGUF routed GEMV + dequantize for the DeepSeek-V4 serving path (SlimServe).
//
// Native decoders for the three formats the DSV4 IQ2_XXS artifact ships:
//   ggml Q8_0    (type  8): 34-byte block  = fp16 d + 32 int8
//   ggml Q2_K    (type 10): 84-byte super-block = scales[16] qs[64] d dmin
//   ggml IQ2_XXS (type 16): 66-byte super-block = fp16 d + qs[uint16*32]
// laid out row-major exactly as on disk (row = K/block_elems blocks).
//
// Two kernels, one decode body per format:
//   * gguf_routed_gemv: y[r, n] = dot(W[expert(r)][n, :], x[r / top_k, :]) for
//     r in [0, R). With `expert_ids == nullptr` W is a single [N, K] matrix and
//     top_k = 1 (dense GEMV, batched over R activations). With expert ids
//     (MoE), W is [E, N, K] and row r reads expert expert_ids[r]; a negative
//     id (expert-parallel skip) writes zeros. Rows are flat (token, k) order,
//     the layout vLLM's GGUF MoE method expects.
//   * gguf_dequantize: out[n, k] = decode(W[n, k]) in fp16/bf16/f32.
//
// Work decomposition: one subgroup per output value; lanes stride over 32-wide
// sub-units of K (K/32 units), so a 4096-deep row keeps all 32 lanes busy
// (the upstream per-type GEMV strides lanes over whole super-blocks, which
// idles half the subgroup at K=4096). Correctness-first: reads the on-disk
// interleaved layout directly. The measured next steps are a GPU repack of
// the expert stacks and a tile GEMM for prefill; see perf notebook.
//
// Numerics: fp32 accumulate; decoders follow ggml dequantize_row_* exactly.

#include <cstdint>

#include "quantization/gguf_gemv/gguf_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kSGPerWG = 8;
constexpr int kWG = kSG * kSGPerWG;

inline float half_at(const std::uint8_t* p) {
  const std::uint16_t bits =
      static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

inline std::uint32_t u32_at(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

// ksigns_iq2xs[i] == i | (parity(i) << 7): the sign byte carries odd parity.
inline std::uint8_t iq2_ksign(std::uint32_t i) {
  return static_cast<std::uint8_t>(i | ((sycl::popcount(i) & 1u) << 7));
}

// ---- per-format 32-element decode ------------------------------------------
// Each functor: given the row base pointer and the 32-wide unit index u
// (k in [32u, 32u+32)), write the 32 dequantized values into v[32].

struct Q8_0 {
  static constexpr int kType = 8;
  static constexpr std::size_t kBlockElems = 32;
  static constexpr std::size_t kBlockBytes = 34;
  const std::uint64_t* grid;  // unused
  static void decode(const std::uint8_t* row, std::size_t u, float* v,
                     const std::uint64_t*) {
    const std::uint8_t* blk = row + u * kBlockBytes;
    const float d = half_at(blk);
    const std::uint8_t* qs = blk + 2;
#pragma unroll
    for (int i = 0; i < 32; ++i)
      v[i] = d * static_cast<float>(static_cast<std::int8_t>(qs[i]));
  }
};

struct Q2_K {
  static constexpr int kType = 10;
  static constexpr std::size_t kBlockElems = 256;
  static constexpr std::size_t kBlockBytes = 84;
  static void decode(const std::uint8_t* row, std::size_t u, float* v,
                     const std::uint64_t*) {
    // super-block s, 32-wide unit j in [0, 8): elements [32j, 32j+32) of the
    // super-block are two 16-wide sub-blocks 2j and 2j+1, sharing the
    // 2-bit plane j%4 of the qs half j/4 (ggml dequantize_row_q2_K).
    const std::size_t s = u / 8;
    const int j = static_cast<int>(u % 8);
    const std::uint8_t* blk = row + s * kBlockBytes;
    const std::uint8_t* scales = blk;
    const std::uint8_t* qs = blk + 16;
    const float d = half_at(blk + 80);
    const float dmin = half_at(blk + 82);
    const int h = j / 4;          // which 128-element half
    const int plane = j % 4;      // which 2-bit plane
    const int shift = 2 * plane;
    const std::uint8_t* qb = qs + h * 32;
    const std::uint8_t sc1 = scales[h * 8 + plane * 2];
    const std::uint8_t sc2 = scales[h * 8 + plane * 2 + 1];
    const float dl1 = d * static_cast<float>(sc1 & 0xF);
    const float ml1 = dmin * static_cast<float>(sc1 >> 4);
    const float dl2 = d * static_cast<float>(sc2 & 0xF);
    const float ml2 = dmin * static_cast<float>(sc2 >> 4);
#pragma unroll
    for (int l = 0; l < 16; ++l) {
      v[l] = dl1 * static_cast<float>((qb[l] >> shift) & 3) - ml1;
      v[16 + l] = dl2 * static_cast<float>((qb[l + 16] >> shift) & 3) - ml2;
    }
  }
};

struct IQ2_XXS {
  static constexpr int kType = 16;
  static constexpr std::size_t kBlockElems = 256;
  static constexpr std::size_t kBlockBytes = 66;
  static void decode(const std::uint8_t* row, std::size_t u, float* v,
                     const std::uint64_t* grid) {
    const std::size_t s = u / 8;
    const int ib = static_cast<int>(u % 8);
    const std::uint8_t* blk = row + s * kBlockBytes;
    const float d = half_at(blk);
    const std::uint8_t* aux = blk + 2 + 8 * ib;  // 4 grid bytes + 4 sign/scale bytes
    const std::uint32_t a1 = u32_at(aux + 4);
    const float db = d * (0.5f + static_cast<float>(a1 >> 28)) * 0.25f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const std::uint64_t g = grid[aux[l]];
      const std::uint8_t signs = iq2_ksign((a1 >> (7 * l)) & 127u);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float gv = static_cast<float>(
            static_cast<std::int8_t>((g >> (8 * j)) & 0xFFu));
        v[l * 8 + j] = (signs & (1u << j)) ? -db * gv : db * gv;
      }
    }
  }
};

// ---- routed GEMV -----------------------------------------------------------

// Dense decode batches reuse one activation-independent Q8 weight row across
// every active token. The generic routed kernel below assigns one subgroup
// to each (token, output) pair, which is necessary for independently routed
// experts but rereads a dense matrix R times. Keep one subgroup on an output
// row here and accumulate a small row bucket while each packed block is live.
// On B70 Qwen3.8 TP4 this removes the concurrency-8 serving cliff: the matched
// 742-in/128-out sweep improved from 22.5 to 67.0 aggregate output tok/s.
template <typename T, int RowBucket>
sycl::event q8_dense_weight_stationary(sycl::queue& q, const std::uint8_t* w,
                                       const T* x, T* y, std::size_t R,
                                       std::size_t N, std::size_t K) {
  const std::size_t units = K / 32;
  const std::size_t row_bytes = units * Q8_0::kBlockBytes;
  const std::size_t nwg = (N + kSGPerWG - 1) / kSGPerWG;
  return q.parallel_for(
      sycl::nd_range<1>(nwg * kWG, kWG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const std::size_t n =
            it.get_group(0) * kSGPerWG + sg.get_group_linear_id();
        if (n >= N) return;
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const std::uint8_t* wrow = w + n * row_bytes;
        float acc[RowBucket];
#pragma unroll
        for (int r = 0; r < RowBucket; ++r) acc[r] = 0.0f;

        for (std::size_t u = lane; u < units; u += kSG) {
          float v[32];
          Q8_0::decode(wrow, u, v, nullptr);
#pragma unroll
          for (int r = 0; r < RowBucket; ++r) {
            if (static_cast<std::size_t>(r) < R) {
              const T* xs = x + static_cast<std::size_t>(r) * K + u * 32;
#pragma unroll
              for (int i = 0; i < 32; ++i) {
                acc[r] += v[i] * static_cast<float>(xs[i]);
              }
            }
          }
        }

#pragma unroll
        for (int r = 0; r < RowBucket; ++r) {
          if (static_cast<std::size_t>(r) < R) {
            const float sum =
                sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
            if (lane == 0) {
              y[static_cast<std::size_t>(r) * N + n] = static_cast<T>(sum);
            }
          }
        }
      });
}

template <typename T>
sycl::event q8_dense_weight_stationary_dispatch(
    sycl::queue& q, const std::uint8_t* w, const T* x, T* y, std::size_t R,
    std::size_t N, std::size_t K) {
  if (R <= 2) return q8_dense_weight_stationary<T, 2>(q, w, x, y, R, N, K);
  if (R <= 4) return q8_dense_weight_stationary<T, 4>(q, w, x, y, R, N, K);
  if (R <= 8) return q8_dense_weight_stationary<T, 8>(q, w, x, y, R, N, K);
  return q8_dense_weight_stationary<T, 16>(q, w, x, y, R, N, K);
}

template <typename T, typename Fmt>
sycl::event routed_gemv(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                        const std::int32_t* expert_ids, std::size_t R,
                        std::size_t N, std::size_t K, std::size_t top_k,
                        const std::uint64_t* grid) {
  const std::size_t units = K / 32;
  const std::size_t row_bytes = (K / Fmt::kBlockElems) * Fmt::kBlockBytes;
  const std::size_t expert_bytes = N * row_bytes;
  const std::size_t outputs = R * N;
  const std::size_t nwg = (outputs + kSGPerWG - 1) / kSGPerWG;
  return q.parallel_for(
      sycl::nd_range<1>(nwg * kWG, kWG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const std::size_t o = it.get_group(0) * kSGPerWG + sg.get_group_linear_id();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        if (o >= outputs) return;
        const std::size_t r = o / N;
        const std::size_t n = o % N;
        const int e = expert_ids ? expert_ids[r] : 0;
        if (e < 0) {
          if (lane == 0) y[o] = static_cast<T>(0.0f);
          return;
        }
        const std::uint8_t* wrow =
            w + static_cast<std::size_t>(e) * expert_bytes + n * row_bytes;
        const T* xrow = x + (r / top_k) * K;
        float acc = 0.0f;
        float v[32];
        for (std::size_t u = lane; u < units; u += kSG) {
          Fmt::decode(wrow, u, v, grid);
          const T* xs = xrow + u * 32;
#pragma unroll
          for (int i = 0; i < 32; ++i) acc += v[i] * static_cast<float>(xs[i]);
        }
        const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) y[o] = static_cast<T>(sum);
      });
}

template <typename T>
sycl::event routed_gemv_dispatch(sycl::queue& q, const std::uint8_t* w, const T* x,
                                 T* y, const std::int32_t* ids, std::size_t R,
                                 std::size_t N, std::size_t K, std::size_t top_k,
                                 int ggml_type, const std::uint64_t* grid) {
  if (ids == nullptr && ggml_type == Q8_0::kType && R >= 2 && R <= 16) {
    return q8_dense_weight_stationary_dispatch(q, w, x, y, R, N, K);
  }
  switch (ggml_type) {
    case Q8_0::kType:
      return routed_gemv<T, Q8_0>(q, w, x, y, ids, R, N, K, top_k, grid);
    case Q2_K::kType:
      return routed_gemv<T, Q2_K>(q, w, x, y, ids, R, N, K, top_k, grid);
    case IQ2_XXS::kType:
      return routed_gemv<T, IQ2_XXS>(q, w, x, y, ids, R, N, K, top_k, grid);
    default:
      return {};
  }
}

// ---- dequantize ------------------------------------------------------------

template <typename T, typename Fmt>
sycl::event dequant(sycl::queue& q, const std::uint8_t* w, T* out, std::size_t N,
                    std::size_t K, const std::uint64_t* grid) {
  const std::size_t units = K / 32;
  const std::size_t row_bytes = (K / Fmt::kBlockElems) * Fmt::kBlockBytes;
  return q.parallel_for(sycl::range<1>(N * units), [=](sycl::id<1> id) {
    const std::size_t n = id[0] / units;
    const std::size_t u = id[0] % units;
    float v[32];
    Fmt::decode(w + n * row_bytes, u, v, grid);
    T* o = out + n * K + u * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = static_cast<T>(v[i]);
  });
}

template <typename T>
sycl::event dequant_dispatch(sycl::queue& q, const std::uint8_t* w, T* out,
                             std::size_t N, std::size_t K, int ggml_type,
                             const std::uint64_t* grid) {
  switch (ggml_type) {
    case Q8_0::kType:
      return dequant<T, Q8_0>(q, w, out, N, K, grid);
    case Q2_K::kType:
      return dequant<T, Q2_K>(q, w, out, N, K, grid);
    case IQ2_XXS::kType:
      return dequant<T, IQ2_XXS>(q, w, out, N, K, grid);
    default:
      return {};
  }
}

// ---- Q8_0 split-plane layout ----------------------------------------------

using Q8Wide = sycl::vec<std::uint32_t, 4>;

template <typename T, int RowBucket>
sycl::event q8_split_weight_stationary(
    sycl::queue& q, const std::uint8_t* packed, const T* x, T* y,
    std::size_t R, std::size_t N, std::size_t K) {
  const std::size_t blocks = K / 32;
  const auto* scales = reinterpret_cast<const half_t*>(packed);
  const auto* quants = reinterpret_cast<const std::int8_t*>(
      packed + N * blocks * sizeof(half_t));
  const std::size_t nwg = (N + kSGPerWG - 1) / kSGPerWG;
  return q.parallel_for(
      sycl::nd_range<1>(nwg * kWG, kWG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const std::size_t n =
            it.get_group(0) * kSGPerWG + sg.get_group_linear_id();
        if (n >= N) return;
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const half_t* scale_row = scales + n * blocks;
        const std::int8_t* quant_row = quants + n * K;
        float acc[RowBucket];
#pragma unroll
        for (int r = 0; r < RowBucket; ++r) acc[r] = 0.0f;

        for (std::size_t block = lane; block < blocks; block += kSG) {
          const float scale = static_cast<float>(scale_row[block]);
          const auto* words =
              reinterpret_cast<const Q8Wide*>(quant_row + block * 32);
          const Q8Wide lo = words[0];
          const Q8Wide hi = words[1];
#pragma unroll
          for (int r = 0; r < RowBucket; ++r) {
            if (static_cast<std::size_t>(r) < R) {
              const T* xb = x + static_cast<std::size_t>(r) * K + block * 32;
              float partial[8];
#pragma unroll
              for (int wi = 0; wi < 8; ++wi) {
                const std::uint32_t word = wi < 4 ? lo[wi] : hi[wi - 4];
                const sycl::vec<T, 4> xv =
                    *reinterpret_cast<const sycl::vec<T, 4>*>(xb + wi * 4);
                float dot = 0.0f;
#pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                  const auto qv = static_cast<std::int8_t>(
                      (word >> (byte * 8)) & 0xffu);
                  dot += static_cast<float>(qv) *
                         static_cast<float>(xv[byte]);
                }
                partial[wi] = dot;
              }
              acc[r] += scale * (((partial[0] + partial[1]) +
                                  (partial[2] + partial[3])) +
                                 ((partial[4] + partial[5]) +
                                  (partial[6] + partial[7])));
            }
          }
        }
#pragma unroll
        for (int r = 0; r < RowBucket; ++r) {
          if (static_cast<std::size_t>(r) < R) {
            const float sum =
                sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
            if (lane == 0)
              y[static_cast<std::size_t>(r) * N + n] =
                  static_cast<T>(sum);
          }
        }
      });
}

template <typename T>
sycl::event q8_split_gemv_dispatch(sycl::queue& q, const std::uint8_t* packed,
                                   const T* x, T* y, std::size_t R,
                                   std::size_t N, std::size_t K) {
  if (R == 1) return q8_split_weight_stationary<T, 1>(q, packed, x, y, R, N, K);
  if (R <= 2) return q8_split_weight_stationary<T, 2>(q, packed, x, y, R, N, K);
  if (R <= 4) return q8_split_weight_stationary<T, 4>(q, packed, x, y, R, N, K);
  if (R <= 8) return q8_split_weight_stationary<T, 8>(q, packed, x, y, R, N, K);
  return q8_split_weight_stationary<T, 16>(q, packed, x, y, R, N, K);
}

template <typename T>
sycl::event q8_split_dequant(sycl::queue& q, const std::uint8_t* packed,
                             T* out, std::size_t N, std::size_t K) {
  const std::size_t blocks = K / 32;
  const auto* scales = reinterpret_cast<const half_t*>(packed);
  const auto* quants = reinterpret_cast<const std::int8_t*>(
      packed + N * blocks * sizeof(half_t));
  return q.parallel_for(sycl::range<1>(N * blocks), [=](sycl::id<1> id) {
    const std::size_t block = id[0];
    const float scale = static_cast<float>(scales[block]);
    const std::size_t base = block * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i)
      out[base + i] =
          static_cast<T>(scale * static_cast<float>(quants[base + i]));
  });
}

}  // namespace

bool gguf_routed_supports(int ggml_type) {
  return ggml_type == Q8_0::kType || ggml_type == Q2_K::kType ||
         ggml_type == IQ2_XXS::kType;
}

sycl::event gguf_routed_gemv_sycl(sycl::queue& q, const void* w, const void* x,
                                  void* y, const std::int32_t* expert_ids,
                                  std::size_t R, std::size_t N, std::size_t K,
                                  std::size_t top_k, int ggml_type, DType act_dt,
                                  const std::uint64_t* iq2xxs_grid_dev) {
  const auto* wb = static_cast<const std::uint8_t*>(w);
  switch (act_dt) {
    case DType::f32:
      return routed_gemv_dispatch<float>(q, wb, static_cast<const float*>(x),
                                         static_cast<float*>(y), expert_ids, R, N,
                                         K, top_k, ggml_type, iq2xxs_grid_dev);
    case DType::f16:
      return routed_gemv_dispatch<half_t>(q, wb, static_cast<const half_t*>(x),
                                          static_cast<half_t*>(y), expert_ids, R,
                                          N, K, top_k, ggml_type, iq2xxs_grid_dev);
    case DType::bf16:
      return routed_gemv_dispatch<bf16_t>(q, wb, static_cast<const bf16_t*>(x),
                                          static_cast<bf16_t*>(y), expert_ids, R,
                                          N, K, top_k, ggml_type, iq2xxs_grid_dev);
  }
  return {};
}

sycl::event gguf_dequantize_sycl(sycl::queue& q, const void* w, void* out,
                                 std::size_t N, std::size_t K, int ggml_type,
                                 DType out_dt, const std::uint64_t* iq2xxs_grid_dev) {
  const auto* wb = static_cast<const std::uint8_t*>(w);
  switch (out_dt) {
    case DType::f32:
      return dequant_dispatch<float>(q, wb, static_cast<float*>(out), N, K,
                                     ggml_type, iq2xxs_grid_dev);
    case DType::f16:
      return dequant_dispatch<half_t>(q, wb, static_cast<half_t*>(out), N, K,
                                      ggml_type, iq2xxs_grid_dev);
    case DType::bf16:
      return dequant_dispatch<bf16_t>(q, wb, static_cast<bf16_t*>(out), N, K,
                                      ggml_type, iq2xxs_grid_dev);
  }
  return {};
}

sycl::event q8_split_repack_sycl(sycl::queue& q, const void* raw, void* split,
                                 std::size_t N, std::size_t K) {
  const auto* src = static_cast<const std::uint8_t*>(raw);
  auto* dst = static_cast<std::uint8_t*>(split);
  const std::size_t blocks_per_row = K / 32;
  const std::size_t total_blocks = N * blocks_per_row;
  const std::size_t scale_bytes = total_blocks * sizeof(half_t);
  return q.parallel_for(sycl::range<1>(total_blocks), [=](sycl::id<1> id) {
    const std::size_t block = id[0];
    const std::uint8_t* in = src + block * 34;
    std::uint8_t* scale = dst + block * 2;
    std::uint8_t* quant = dst + scale_bytes + block * 32;
    scale[0] = in[0];
    scale[1] = in[1];
#pragma unroll
    for (int i = 0; i < 32; ++i) quant[i] = in[i + 2];
  });
}

sycl::event q8_split_gemv_sycl(sycl::queue& q, const void* split,
                               const void* x, void* y, std::size_t R,
                               std::size_t N, std::size_t K, DType act_dt) {
  const auto* packed = static_cast<const std::uint8_t*>(split);
  switch (act_dt) {
    case DType::f32:
      return q8_split_gemv_dispatch(q, packed, static_cast<const float*>(x),
                                    static_cast<float*>(y), R, N, K);
    case DType::f16:
      return q8_split_gemv_dispatch(q, packed, static_cast<const half_t*>(x),
                                    static_cast<half_t*>(y), R, N, K);
    case DType::bf16:
      return q8_split_gemv_dispatch(q, packed, static_cast<const bf16_t*>(x),
                                    static_cast<bf16_t*>(y), R, N, K);
  }
  return {};
}

sycl::event q8_split_dequantize_sycl(sycl::queue& q, const void* split,
                                     void* out, std::size_t N, std::size_t K,
                                     DType out_dt) {
  const auto* packed = static_cast<const std::uint8_t*>(split);
  switch (out_dt) {
    case DType::f32:
      return q8_split_dequant(q, packed, static_cast<float*>(out), N, K);
    case DType::f16:
      return q8_split_dequant(q, packed, static_cast<half_t*>(out), N, K);
    case DType::bf16:
      return q8_split_dequant(q, packed, static_cast<bf16_t*>(out), N, K);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
