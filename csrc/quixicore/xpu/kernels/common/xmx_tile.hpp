#pragma once

// Native DPAS (XMX) GEMM building block for the Xe tensor engine, generalizing
// the pattern proven bit-exact on Arc Pro B60 by
// kernels/quantization/w4a16_gemm (feasibility probe worst_abs=0):
//
//   * one subgroup (16 lanes) owns one 8x16 fp32 output tile;
//   * operands are staged through SLM by a caller lambda — bounds masking,
//     transposition, and on-the-fly dequantization all live in that lambda,
//     which sidesteps every joint_matrix operand-layout restriction (B is
//     always presented as a [TK, TN] row-major SLM tile no matter how the
//     weight is packed in memory);
//   * accumulation is fp32 via joint_matrix_mad; the epilogue spills the
//     accumulator to SLM and hands each element to a caller lambda for
//     masked/converted/scaled stores.
//
// No cutlass, no cute — raw SYCL joint_matrix only (house dependency policy).
// Everything here is compile-time-shaped and allocation-free, so kernels
// built on it stay graph-capture-safe. Consumers: mqa_logits (proving
// ground), the paged-attention and grouped-MoE-GEMM rewrites, and the MHC
// split-K path.

#include <cstddef>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace quixicore::xpu::xmx {

namespace jm = sycl::ext::oneapi::experimental::matrix;

inline constexpr int kTM = 8;   // DPAS tile rows (M)
inline constexpr int kTN = 16;  // DPAS tile cols (N)
inline constexpr int kTK = 16;  // DPAS tile depth (K)
inline constexpr int kSG = 16;  // subgroup width

template <typename T>
using MatA =
    jm::joint_matrix<sycl::sub_group, T, jm::use::a, kTM, kTK,
                     jm::layout::row_major>;
template <typename T>
using MatB =
    jm::joint_matrix<sycl::sub_group, T, jm::use::b, kTK, kTN,
                     jm::layout::row_major>;
using MatC =
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, kTM, kTN>;

// Cooperatively fill a ROWS x COLS SLM tile: fn(r, c) -> T supplies each
// element (out-of-bounds positions must return T(0) — zero-padding is what
// makes edge tiles correct). All kSG lanes must call this.
template <int ROWS, int COLS, typename T, typename Fn>
inline void stage_tile(const sycl::local_accessor<T, 1>& slm, int lane,
                       Fn&& fn) {
  for (int e = lane; e < ROWS * COLS; e += kSG) {
    slm[e] = fn(e / COLS, e % COLS);
  }
}

// One K-step: load the staged [kTM,kTK] A and [kTK,kTN] B tiles and
// accumulate into acc. Call between local barriers (the stage writes must be
// visible; the next stage must not overwrite early).
template <typename T>
inline void mad_step(const sycl::sub_group& sg,
                     const sycl::local_accessor<T, 1>& As,
                     const sycl::local_accessor<T, 1>& Bs, MatC& acc) {
  MatA<T> ma;
  MatB<T> mb;
  jm::joint_matrix_load(
      sg, ma, As.template get_multi_ptr<sycl::access::decorated::no>(), kTK);
  jm::joint_matrix_load(
      sg, mb, Bs.template get_multi_ptr<sycl::access::decorated::no>(), kTN);
  jm::joint_matrix_mad(sg, acc, ma, mb, acc);
}

// Epilogue: spill acc to the [kTM,kTN] fp32 SLM tile, barrier, then hand each
// element to fn(r, c, value) for the masked/converted store. All lanes call.
template <typename Group, typename Fn>
inline void store_tile(const Group& group, const sycl::sub_group& sg,
                       const sycl::local_accessor<float, 1>& Cs, int lane,
                       MatC& acc, Fn&& fn) {
  jm::joint_matrix_store(
      sg, acc, Cs.template get_multi_ptr<sycl::access::decorated::no>(), kTN,
      jm::layout::row_major);
  sycl::group_barrier(group);
  for (int e = lane; e < kTM * kTN; e += kSG) {
    fn(e / kTN, e % kTN, Cs[e]);
  }
}

}  // namespace quixicore::xpu::xmx
