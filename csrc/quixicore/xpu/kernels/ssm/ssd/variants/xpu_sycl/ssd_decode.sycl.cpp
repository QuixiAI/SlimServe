// Mamba-2 SSD decode (selective-state-update, scalar-A-per-head variant).
//
// Per (batch, head): update the [headdim, dstate] SSM state one step and emit
// [headdim] output. One work-group per (batch, head), one lane per headdim row
// -- each lane owns a full state row of dstate, updates it, and computes its
// own output dot(state_row, C). No cross-lane reduction. fp32 recurrence:
//   dt_h  = softplus(dt_raw[b,h] + dt_bias[h])        (if dt_softplus)
//   dA    = exp(dt_h * A[h])
//   s[d,n]= state[d,n]*dA + dt_h * x[b,h,d] * B[b,g,n]   g = h / (nheads/ngroups)
//   y[d]  = sum_n s[d,n] * C[b,g,n] + D[h,d] * x[b,h,d]
// State is read from the src slot and written to the dst slot (copy-on-write);
// invalid src slots emit zeros and touch no state.
//
// Adapted from the QuixiAI MIT decode kernel set proven in vLLM XPU serving
// (vllm-xpu-kernels csrc/xpu/sycl/decode/mamba2_ssd_decode_kernel.hpp,
// commit dffcab7), re-homed on the QuixiCore DType/ops ABI.

#include "ssm/ssd/ssd_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

inline float ssd_softplus(float x) {
  return x <= 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
}

template <typename T, typename StateT>
class SsdDecodeKernel;

// state: [nslots, nheads, headdim, dstate] StateT, element strides s0..s3
// x, out: [batch, nheads, headdim] T
// dt_raw: [batch, nheads] f32; A, dt_bias: [nheads] f32; D: [nheads, headdim]
// f32 (nullptr = skip); B, C: [batch, ngroups, dstate] T
template <typename T, typename StateT>
sycl::event ssd_decode_typed(sycl::queue& q, StateT* state, const T* x,
                             const float* dt_raw, const float* A, const T* B,
                             const T* C, const float* D, const float* dt_bias,
                             const std::int32_t* src_indices,
                             const std::int32_t* dst_indices, T* out,
                             bool dt_softplus, std::size_t batch,
                             std::size_t nheads, std::size_t headdim,
                             std::size_t dstate, std::size_t ngroups,
                             std::size_t nslots, std::int64_t s0,
                             std::int64_t s1, std::int64_t s2,
                             std::int64_t s3) {
  const std::size_t hpg = nheads / ngroups;
  const sycl::nd_range<2> ndr(sycl::range<2>(batch * nheads, headdim),
                              sycl::range<2>(1, headdim));
  return q.submit([&](sycl::handler& hnd) {
    hnd.parallel_for<SsdDecodeKernel<T, StateT>>(ndr, [=](sycl::nd_item<2> it) {
      const std::size_t bh = it.get_global_id(0);
      const std::size_t d = it.get_local_id(1);
      const std::size_t b = bh / nheads;
      const std::size_t h = bh - b * nheads;

      const std::int32_t src = src_indices[b];
      const std::int32_t dst = dst_indices[b];
      T* out_ptr = out + (b * nheads + h) * headdim + d;
      if (src < 0 || static_cast<std::size_t>(src) >= nslots) {
        *out_ptr = T(0);
        return;
      }
      const std::size_t g = h / hpg;

      float dt_h = dt_raw[b * nheads + h] + dt_bias[h];
      if (dt_softplus) dt_h = ssd_softplus(dt_h);
      const float dA = sycl::exp(dt_h * A[h]);
      const float x_d = static_cast<float>(x[(b * nheads + h) * headdim + d]);

      const StateT* srow = state + static_cast<std::int64_t>(src) * s0 +
                           static_cast<std::int64_t>(h) * s1 +
                           static_cast<std::int64_t>(d) * s2;
      const bool dst_ok = dst >= 0 && static_cast<std::size_t>(dst) < nslots;
      StateT* drow = dst_ok ? state + static_cast<std::int64_t>(dst) * s0 +
                                  static_cast<std::int64_t>(h) * s1 +
                                  static_cast<std::int64_t>(d) * s2
                            : nullptr;
      const T* Bp = B + (b * ngroups + g) * dstate;
      const T* Cp = C + (b * ngroups + g) * dstate;

      const float dtx = dt_h * x_d;
      float y = 0.0f;
      for (std::size_t n = 0; n < dstate; ++n) {
        const std::int64_t off = static_cast<std::int64_t>(n) * s3;
        const float s = static_cast<float>(srow[off]) * dA +
                        dtx * static_cast<float>(Bp[n]);
        if (dst_ok) drow[off] = static_cast<StateT>(s);
        y += s * static_cast<float>(Cp[n]);
      }
      if (D != nullptr) y += D[h * headdim + d] * x_d;
      *out_ptr = static_cast<T>(y);
    });
  });
}

}  // namespace

sycl::event ssd_decode_sycl(sycl::queue& q, void* state, const void* x,
                            const float* dt_raw, const float* A, const void* B,
                            const void* C, const float* D, const float* dt_bias,
                            const std::int32_t* src_indices,
                            const std::int32_t* dst_indices, void* out,
                            bool dt_softplus, std::size_t batch,
                            std::size_t nheads, std::size_t headdim,
                            std::size_t dstate, std::size_t ngroups,
                            std::size_t nslots, std::int64_t s0,
                            std::int64_t s1, std::int64_t s2, std::int64_t s3,
                            DType act_dt, DType state_dt) {
#define QX_SSD_DISP(T, ST)                                                     \
  return ssd_decode_typed<T, ST>(                                              \
      q, static_cast<ST*>(state), static_cast<const T*>(x), dt_raw, A,         \
      static_cast<const T*>(B), static_cast<const T*>(C), D, dt_bias,          \
      src_indices, dst_indices, static_cast<T*>(out), dt_softplus, batch,      \
      nheads, headdim, dstate, ngroups, nslots, s0, s1, s2, s3)
#define QX_SSD_BY_STATE(T)          \
  switch (state_dt) {               \
    case DType::f32:                \
      QX_SSD_DISP(T, float);        \
    case DType::f16:                \
      QX_SSD_DISP(T, half_t);       \
    case DType::bf16:               \
      QX_SSD_DISP(T, bf16_t);       \
  }
  switch (act_dt) {
    case DType::f32:
      QX_SSD_BY_STATE(float);
      break;
    case DType::f16:
      QX_SSD_BY_STATE(half_t);
      break;
    case DType::bf16:
      QX_SSD_BY_STATE(bf16_t);
      break;
  }
#undef QX_SSD_BY_STATE
#undef QX_SSD_DISP
  return {};
}

}  // namespace quixicore::xpu::kernels
