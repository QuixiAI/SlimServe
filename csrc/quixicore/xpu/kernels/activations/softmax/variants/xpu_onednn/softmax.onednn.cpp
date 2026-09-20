// Softmax over the last axis, vendor variant via oneDNN softmax_forward.
//
// Co-equal shipped implementation selectable as Variant::vendor. Runs on the
// caller's sycl::queue via oneDNN SYCL interop. algorithm::softmax_accurate is
// the numerically stable (max-subtracting) formulation, matching the SYCL
// variant's semantics. Compiled only when oneDNN is found.

#include "activations/softmax/softmax_kernel.hpp"

#if defined(QUIXICORE_XPU_HAS_ONEDNN)

#include <unordered_map>

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

sycl::event softmax_onednn(sycl::queue& q, const void* x, void* out,
                           std::size_t rows, std::size_t dim, DType dt) {
  dnnl::engine eng =
      dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

  const dnnl::memory::data_type d = to_dnnl(dt);
  const dnnl::memory::dims xdims = {static_cast<dnnl::memory::dim>(rows),
                                    static_cast<dnnl::memory::dim>(dim)};
  const dnnl::memory::desc src_md(xdims, d, dnnl::memory::format_tag::ab);
  const dnnl::memory::desc dst_md(xdims, d, dnnl::memory::format_tag::ab);

  dnnl::softmax_forward::primitive_desc pd(
      eng, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::softmax_accurate, src_md, dst_md, /*axis=*/1);
  dnnl::softmax_forward prim(pd);

  auto usm = [&](const dnnl::memory::desc& md, void* p) {
    return dnnl::sycl_interop::make_memory(
        md, eng, dnnl::sycl_interop::memory_kind::usm, p);
  };
  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, usm(src_md, const_cast<void*>(x))},
      {DNNL_ARG_DST, usm(dst_md, out)},
  };
  return dnnl::sycl_interop::execute(prim, strm, args);
}

}  // namespace quixicore::xpu::kernels

#endif  // QUIXICORE_XPU_HAS_ONEDNN
