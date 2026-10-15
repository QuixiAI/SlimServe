// RMSNorm over the last axis, native SYCL variant.
//
// One work-group per row; the group cooperatively reduces sum(x^2) via
// sycl::reduce_over_group (fp32 accumulation), then rescales. This is the
// reduction-pattern reference for the backend (contrast the elementwise GELU
// reference). Deterministic within a fixed launch geometry.
//
// Launch geometry (work-group size, rows-per-group, SLM staging) is a perf lever
// recorded in perf/optimization_status.md before any speedup is claimed.

#include "norms/norms_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

// Work-items per row. Each item strides over `dim`, so any dim is handled; 256
// balances occupancy against reduction cost for typical hidden sizes.
constexpr std::size_t kRowThreads = 256;

// Elements per vector load. V*sizeof(T) = 16 bytes (8x bf16/f16, 4x f32) so each
// thread issues one wide load and ADJACENT threads read ADJACENT vectors
// (coalesced). Reinterpreting the row as sycl::vec<T,V> forces the compiler to
// emit the wide load; an unrolled scalar loop instead compiled to V strided
// scalar loads and regressed (perf/optimization_status.md 2026-07-06).
template <typename T>
constexpr int vec_width() {
  return sizeof(T) == 2 ? 8 : 4;
}

template <typename T>
sycl::event rms_norm_typed(sycl::queue& q, const T* x, const T* w, T* out,
                           std::size_t rows, std::size_t dim, float eps) {
  constexpr int V = vec_width<T>();
  using Vec = sycl::vec<T, V>;
  const std::size_t nvec = dim / V;      // whole vectors per row
  const std::size_t tail0 = nvec * V;    // first scalar-tail index
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t row = it.get_group(0);
    const std::size_t lid = it.get_local_id(0);
    const T* xr = x + row * dim;
    const T* wr = w;
    T* outr = out + row * dim;
    const Vec* xv = reinterpret_cast<const Vec*>(xr);

    float partial = 0.0f;
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec v = xv[vi];
#pragma unroll
      for (int k = 0; k < V; ++k) {
        const float f = static_cast<float>(v[k]);
        partial += f * f;
      }
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      const float f = static_cast<float>(xr[i]);
      partial += f * f;
    }
    const float sumsq =
        sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
    const float inv = sycl::rsqrt(sumsq / static_cast<float>(dim) + eps);

    const Vec* wv = reinterpret_cast<const Vec*>(wr);
    Vec* ov = reinterpret_cast<Vec*>(outr);
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec xvec = xv[vi];
      const Vec wvec = wv[vi];
      Vec ovec;
#pragma unroll
      for (int k = 0; k < V; ++k) {
        ovec[k] = static_cast<T>(static_cast<float>(xvec[k]) * inv *
                                 static_cast<float>(wvec[k]));
      }
      ov[vi] = ovec;
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      outr[i] = static_cast<T>(static_cast<float>(xr[i]) * inv *
                               static_cast<float>(wr[i]));
    }
  });
}

template <typename T> class FusedAddRmsNormKernel;

template <typename T>
sycl::event fused_add_rms_norm_typed(sycl::queue &q, const T *x, T *residual, const T *weight,
                                     T *out, std::size_t rows, std::size_t dim, float eps) {
  const sycl::nd_range<1> range(sycl::range<1>(rows * kRowThreads), sycl::range<1>(kRowThreads));
  const float inv_dim = 1.0f / static_cast<float>(dim);
  return q.parallel_for<FusedAddRmsNormKernel<T>>(range, [=](sycl::nd_item<1> item) {
    const std::size_t row = item.get_group(0);
    const std::size_t lane = item.get_local_id(0);
    const T *x_row = x + row * dim;
    T *residual_row = residual + row * dim;
    T *out_row = out + row * dim;

    float partial = 0.0f;
    for (std::size_t i = lane; i < dim; i += kRowThreads) {
      const float value = static_cast<float>(x_row[i]) + static_cast<float>(residual_row[i]);
      partial += value * value;
    }
    const float sum_squares =
        sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
    const float inverse_rms = sycl::rsqrt(sum_squares * inv_dim + eps);

    for (std::size_t i = lane; i < dim; i += kRowThreads) {
      const float value = static_cast<float>(x_row[i]) + static_cast<float>(residual_row[i]);
      residual_row[i] = static_cast<T>(value);
      const float normalized = static_cast<float>(static_cast<T>(value * inverse_rms));
      out_row[i] = static_cast<T>(normalized * static_cast<float>(weight[i]));
    }
  });
}

template <typename T> class RmsResidualNextKernel;

// Fused residual-add + double RMSNorm -> f16 (extends the single-norm
// fused_add_rms_norm above). One work-group per row; two group reductions (RMS
// of the projection, then RMS of the updated residual). The residual store uses
// the fp32 sum while its square feeds the second reduction unrounded, exactly as
// the embeddinggemma.c launch_rms_residual_next_half reference does; the final
// pre-norm re-reads the (storage-dtype) residual and converts to f16.
template <typename T>
sycl::event rms_residual_next_typed(sycl::queue &q, const T *projection,
                                    const T *post_weight, T *residual,
                                    const T *next_weight, half_t *next_out,
                                    std::size_t rows, std::size_t dim, float eps) {
  const sycl::nd_range<1> range(sycl::range<1>(rows * kRowThreads),
                                sycl::range<1>(kRowThreads));
  const float inv_dim = 1.0f / static_cast<float>(dim);
  return q.parallel_for<RmsResidualNextKernel<T>>(range, [=](sycl::nd_item<1> item) {
    const std::size_t row = item.get_group(0);
    const std::size_t lane = item.get_local_id(0);
    const T *proj = projection + row * dim;
    T *res = residual + row * dim;
    half_t *out = next_out + row * dim;

    // Pass 1: RMS of the sublayer output for its post-norm scale.
    float proj_ss = 0.0f;
    for (std::size_t i = lane; i < dim; i += kRowThreads) {
      const float v = static_cast<float>(proj[i]);
      proj_ss += v * v;
    }
    proj_ss = sycl::reduce_over_group(item.get_group(), proj_ss, sycl::plus<float>());
    const float proj_inv = sycl::rsqrt(proj_ss * inv_dim + eps);

    // Pass 2: post-norm the projection into the residual stream (in place) and
    // accumulate the RMS of the updated residual for the next layers pre-norm.
    float res_ss = 0.0f;
    for (std::size_t i = lane; i < dim; i += kRowThreads) {
      const float value = static_cast<float>(res[i]) +
          static_cast<float>(proj[i]) * static_cast<float>(post_weight[i]) * proj_inv;
      res[i] = static_cast<T>(value);
      res_ss += value * value;
    }
    res_ss = sycl::reduce_over_group(item.get_group(), res_ss, sycl::plus<float>());
    const float res_inv = sycl::rsqrt(res_ss * inv_dim + eps);

    // Pass 3: next layers pre-norm, converted to f16.
    for (std::size_t i = lane; i < dim; i += kRowThreads) {
      out[i] = static_cast<half_t>(
          static_cast<float>(res[i]) * static_cast<float>(next_weight[i]) * res_inv);
    }
  });
}

}  // namespace

sycl::event rms_norm_sycl(sycl::queue& q, const void* x, const void* weight,
                          void* out, std::size_t rows, std::size_t dim,
                          float eps, DType dt) {
  switch (dt) {
    case DType::f32:
      return rms_norm_typed(q, static_cast<const float*>(x),
                            static_cast<const float*>(weight),
                            static_cast<float*>(out), rows, dim, eps);
    case DType::f16:
      return rms_norm_typed(q, static_cast<const half_t*>(x),
                            static_cast<const half_t*>(weight),
                            static_cast<half_t*>(out), rows, dim, eps);
    case DType::bf16:
      return rms_norm_typed(q, static_cast<const bf16_t*>(x),
                            static_cast<const bf16_t*>(weight),
                            static_cast<bf16_t*>(out), rows, dim, eps);
  }
  return {};
}

sycl::event fused_add_rms_norm_sycl(sycl::queue &q, const void *x, void *residual,
                                    const void *weight, void *out, std::size_t rows,
                                    std::size_t dim, float eps, DType dt) {
  switch (dt) {
  case DType::f32:
    return fused_add_rms_norm_typed(
        q, static_cast<const float *>(x), static_cast<float *>(residual),
        static_cast<const float *>(weight), static_cast<float *>(out), rows, dim, eps);
  case DType::f16:
    return fused_add_rms_norm_typed(
        q, static_cast<const half_t *>(x), static_cast<half_t *>(residual),
        static_cast<const half_t *>(weight), static_cast<half_t *>(out), rows, dim, eps);
  case DType::bf16:
    return fused_add_rms_norm_typed(
        q, static_cast<const bf16_t *>(x), static_cast<bf16_t *>(residual),
        static_cast<const bf16_t *>(weight), static_cast<bf16_t *>(out), rows, dim, eps);
  }
  return {};
}


sycl::event rms_residual_next_sycl(sycl::queue &q, const void *projection,
                                   const void *post_weight, void *residual,
                                   const void *next_weight, void *next_out,
                                   std::size_t rows, std::size_t dim, float eps,
                                   DType dt) {
  auto *out = static_cast<half_t *>(next_out);
  switch (dt) {
  case DType::f32:
    return rms_residual_next_typed(
        q, static_cast<const float *>(projection),
        static_cast<const float *>(post_weight), static_cast<float *>(residual),
        static_cast<const float *>(next_weight), out, rows, dim, eps);
  case DType::f16:
    return rms_residual_next_typed(
        q, static_cast<const half_t *>(projection),
        static_cast<const half_t *>(post_weight), static_cast<half_t *>(residual),
        static_cast<const half_t *>(next_weight), out, rows, dim, eps);
  case DType::bf16:
    return rms_residual_next_typed(
        q, static_cast<const bf16_t *>(projection),
        static_cast<const bf16_t *>(post_weight), static_cast<bf16_t *>(residual),
        static_cast<const bf16_t *>(next_weight), out, rows, dim, eps);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
