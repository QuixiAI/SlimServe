// GELU activation, vendor variant via oneDNN eltwise.
//
// This is a co-equal shipped implementation (not merely a benchmark baseline):
// callers may select Variant::vendor to route GELU through oneDNN. It runs on
// the caller's own sycl::queue via oneDNN SYCL interop, so it composes with the
// native SYCL ops on the same in-order stream and USM allocations.
//
// Compiled only when oneDNN is found at configure time (QUIXICORE_XPU_HAS_ONEDNN).

#include "activations/gelu/gelu_kernel.hpp"

#if defined(QUIXICORE_XPU_HAS_ONEDNN)

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace quixicore::xpu::kernels {
namespace {

dnnl::memory::data_type to_dnnl(DType dt) {
  switch (dt) {
    case DType::f32:
      return dnnl::memory::data_type::f32;
    case DType::f16:
      return dnnl::memory::data_type::f16;
    case DType::bf16:
      return dnnl::memory::data_type::bf16;
  }
  return dnnl::memory::data_type::f32;
}

}  // namespace

sycl::event gelu_onednn(sycl::queue& q, const void* in, void* out, std::size_t n,
                        DType dt, bool tanh_approx) {
  // Build a oneDNN engine + stream bound to the caller's SYCL device/queue so
  // the primitive executes on the same stream as native kernels.
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

  const dnnl::memory::dims dims = {static_cast<dnnl::memory::dim>(n)};
  const dnnl::memory::desc md(dims, to_dnnl(dt), dnnl::memory::format_tag::a);

  // Wrap the existing USM pointers as oneDNN memory (no copy).
  dnnl::memory src_mem = dnnl::sycl_interop::make_memory(
      md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(in));
  dnnl::memory dst_mem = dnnl::sycl_interop::make_memory(
      md, eng, dnnl::sycl_interop::memory_kind::usm, out);

  const dnnl::algorithm algo =
      tanh_approx ? dnnl::algorithm::eltwise_gelu_tanh
                  : dnnl::algorithm::eltwise_gelu_erf;

  dnnl::eltwise_forward::primitive_desc pd(
      eng, dnnl::prop_kind::forward_inference, algo, md, md, /*alpha=*/0.0f,
      /*beta=*/0.0f);
  dnnl::eltwise_forward prim(pd);

  return dnnl::sycl_interop::execute(
      prim, strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
}

}  // namespace quixicore::xpu::kernels

#endif  // QUIXICORE_XPU_HAS_ONEDNN
