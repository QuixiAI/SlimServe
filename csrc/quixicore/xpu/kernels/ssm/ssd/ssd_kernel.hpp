#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Mamba-2 SSD decode (selective-state-update, scalar-A-per-head). Shapes and
// slot semantics are documented on ops::ssd_decode.
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
                            DType act_dt, DType state_dt);

// Varlen Mamba-2 SSD prefill scan (sequential-over-tokens variant). Shapes and
// the SLM envelope are documented on ops::ssd_prefill.
sycl::event ssd_prefill_sycl(sycl::queue& q, const void* x, const float* dt_raw,
                             const float* A, const void* B, const void* C,
                             const float* D, const float* dt_bias,
                             const std::int32_t* cu_seqlens,
                             const void* initial_states, void* out,
                             void* varlen_states, bool dt_softplus, float dt_lo,
                             float dt_hi, std::size_t batch, std::size_t nheads,
                             std::size_t headdim, std::size_t dstate,
                             std::size_t ngroups, DType act_dt, DType state_dt);

}  // namespace quixicore::xpu::kernels
