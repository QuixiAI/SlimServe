// Varlen Mamba-2 SSD prefill selective scan (sequential-over-tokens variant).
//
// One work-group per (seq, head), one lane per headdim row d. Each lane owns a
// full SSM state row of dstate in shared local memory and runs the recurrence
// sequentially over the sequence's tokens:
//   dt_h  = clamp(softplus(dt_raw[t,h] + dt_bias[h]), dt_lo, dt_hi)
//   dA    = exp(dt_h * A[h])
//   s[d,n]= s[d,n]*dA + dt_h * x[t,h,d] * B[t,g,n]    g = h / (nheads/ngroups)
//   y[d]  = sum_n s[d,n] * C[t,g,n] + D[h,d] * x[t,h,d]
// then writes the final s[d,:] to varlen_states[seq,h,d,:]. Initial state is
// read from initial_states (pre-gathered by the caller) when non-null, else
// zero; the caller scatters varlen_states into the SSM cache. That gather /
// scatter split is what keeps the kernel free of slot indirection and
// graph-capture-safe. A chunked parallel-scan variant is the planned
// throughput upgrade (contract op unchanged).
//
// Adapted from the QuixiAI MIT decode kernel set proven in vLLM XPU serving
// (vllm-xpu-kernels csrc/xpu/sycl/decode/mamba2_ssd_prefill_kernel.hpp,
// commit dffcab7), re-homed on the QuixiCore DType/ops ABI.

#include "ssm/ssd/ssd_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

inline float ssd_pf_softplus(float x) {
  return x <= 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
}

template <typename T, typename StateT>
class SsdPrefillKernel;

// x, out:         [T, nheads, headdim] T
// dt_raw:         [T, nheads] f32; A, dt_bias: [nheads] f32
// D:              [nheads, headdim] f32 (nullptr = skip)
// B, C:           [T, ngroups, dstate] T
// initial_states: [batch, nheads, headdim, dstate] StateT (nullptr = zero)
// varlen_states:  [batch, nheads, headdim, dstate] StateT (final state out)
// cu_seqlens:     [batch+1] int32 (packed token offsets)
template <typename T, typename StateT>
sycl::event ssd_prefill_typed(sycl::queue& q, const T* x, const float* dt_raw,
                              const float* A, const T* B, const T* C,
                              const float* D, const float* dt_bias,
                              const std::int32_t* cu_seqlens,
                              const StateT* initial_states, T* out,
                              StateT* varlen_states, bool dt_softplus,
                              float dt_lo, float dt_hi, std::size_t batch,
                              std::size_t nheads, std::size_t headdim,
                              std::size_t dstate, std::size_t ngroups) {
  const std::size_t hpg = nheads / ngroups;
  const std::size_t slm_elems = headdim * dstate;
  const sycl::nd_range<2> ndr(sycl::range<2>(batch * nheads, headdim),
                              sycl::range<2>(1, headdim));
  return q.submit([&](sycl::handler& hnd) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>(slm_elems), hnd);
    hnd.parallel_for<SsdPrefillKernel<T, StateT>>(
        ndr, [=](sycl::nd_item<2> it) {
          const std::size_t bh = it.get_global_id(0);
          const std::size_t d = it.get_local_id(1);
          const std::size_t s = bh / nheads;
          const std::size_t h = bh - s * nheads;
          const std::size_t g = h / hpg;

          float* srow =
              slm.template get_multi_ptr<sycl::access::decorated::no>()
                  .get_raw() +
              d * dstate;

          const std::int64_t st_base =
              ((static_cast<std::int64_t>(s) * nheads +
                static_cast<std::int64_t>(h)) *
                   headdim +
               static_cast<std::int64_t>(d)) *
              dstate;
          if (initial_states != nullptr) {
            const StateT* isrc = initial_states + st_base;
            for (std::size_t n = 0; n < dstate; ++n)
              srow[n] = static_cast<float>(isrc[n]);
          } else {
            for (std::size_t n = 0; n < dstate; ++n) srow[n] = 0.0f;
          }

          const std::size_t lo = static_cast<std::size_t>(cu_seqlens[s]);
          const std::size_t hi = static_cast<std::size_t>(cu_seqlens[s + 1]);
          const float A_h = A[h];
          const float bias_h = dt_bias[h];
          const float D_hd = D != nullptr ? D[h * headdim + d] : 0.0f;

          for (std::size_t t = lo; t < hi; ++t) {
            float dt_h = dt_raw[t * nheads + h] + bias_h;
            if (dt_softplus) dt_h = ssd_pf_softplus(dt_h);
            dt_h = sycl::fmin(sycl::fmax(dt_h, dt_lo), dt_hi);
            const float dA = sycl::exp(dt_h * A_h);
            const float x_hd =
                static_cast<float>(x[(t * nheads + h) * headdim + d]);
            const float dtx = dt_h * x_hd;

            const T* Bp = B + (t * ngroups + g) * dstate;
            const T* Cp = C + (t * ngroups + g) * dstate;
            float y = 0.0f;
            for (std::size_t n = 0; n < dstate; ++n) {
              const float sn = srow[n] * dA + dtx * static_cast<float>(Bp[n]);
              srow[n] = sn;
              y += sn * static_cast<float>(Cp[n]);
            }
            if (D != nullptr) y += D_hd * x_hd;
            out[(t * nheads + h) * headdim + d] = static_cast<T>(y);
          }

          StateT* vdst = varlen_states + st_base;
          for (std::size_t n = 0; n < dstate; ++n)
            vdst[n] = static_cast<StateT>(srow[n]);
        });
  });
}

}  // namespace

sycl::event ssd_prefill_sycl(sycl::queue& q, const void* x, const float* dt_raw,
                             const float* A, const void* B, const void* C,
                             const float* D, const float* dt_bias,
                             const std::int32_t* cu_seqlens,
                             const void* initial_states, void* out,
                             void* varlen_states, bool dt_softplus, float dt_lo,
                             float dt_hi, std::size_t batch, std::size_t nheads,
                             std::size_t headdim, std::size_t dstate,
                             std::size_t ngroups, DType act_dt,
                             DType state_dt) {
  const auto dev = q.get_device();
  const std::size_t slm_bytes = headdim * dstate * sizeof(float);
  if (slm_bytes > dev.get_info<sycl::info::device::local_mem_size>() ||
      headdim > dev.get_info<sycl::info::device::max_work_group_size>()) {
    return {};  // outside the SLM/work-group envelope: reject, don't corrupt
  }
#define QX_SP_DISP(T, ST)                                                      \
  return ssd_prefill_typed<T, ST>(                                             \
      q, static_cast<const T*>(x), dt_raw, A, static_cast<const T*>(B),        \
      static_cast<const T*>(C), D, dt_bias, cu_seqlens,                        \
      static_cast<const ST*>(initial_states), static_cast<T*>(out),            \
      static_cast<ST*>(varlen_states), dt_softplus, dt_lo, dt_hi, batch,       \
      nheads, headdim, dstate, ngroups)
#define QX_SP_BY_STATE(T)          \
  switch (state_dt) {              \
    case DType::f32:               \
      QX_SP_DISP(T, float);        \
    case DType::f16:               \
      QX_SP_DISP(T, half_t);       \
    case DType::bf16:              \
      QX_SP_DISP(T, bf16_t);       \
  }
  switch (act_dt) {
    case DType::f32:
      QX_SP_BY_STATE(float);
      break;
    case DType::f16:
      QX_SP_BY_STATE(half_t);
      break;
    case DType::bf16:
      QX_SP_BY_STATE(bf16_t);
      break;
  }
#undef QX_SP_BY_STATE
#undef QX_SP_DISP
  return {};
}

}  // namespace quixicore::xpu::kernels
