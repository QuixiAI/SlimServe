// Dense GEMM, vendor variant via oneDNN matmul (XMX/DPAS-backed on Intel GPU).
// Co-equal shipped implementation selectable as Variant::vendor. Runs on the
// caller's sycl::queue via oneDNN SYCL interop. Compiled only when oneDNN is found.

#include "matmul/dense_gemm/dense_gemm_kernel.hpp"

#if defined(QUIXICORE_XPU_HAS_ONEDNN)

#include <unordered_map>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace quixicore::xpu::kernels {
namespace {

dnnl::memory::data_type to_dnnl(DType dt) {
  switch (dt) {
    case DType::f32: return dnnl::memory::data_type::f32;
    case DType::f16: return dnnl::memory::data_type::f16;
    case DType::bf16: return dnnl::memory::data_type::bf16;
  }
  return dnnl::memory::data_type::f32;
}

}  // namespace

sycl::event dense_gemm_onednn(sycl::queue& q, const void* a, const void* b,
                              void* c, std::size_t M, std::size_t N,
                              std::size_t K, DType dt) {
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

  using tag = dnnl::memory::format_tag;
  const dnnl::memory::data_type d = to_dnnl(dt);
  const dnnl::memory::desc a_md({static_cast<dnnl::memory::dim>(M),
                                 static_cast<dnnl::memory::dim>(K)}, d, tag::ab);
  const dnnl::memory::desc b_md({static_cast<dnnl::memory::dim>(K),
                                 static_cast<dnnl::memory::dim>(N)}, d, tag::ab);
  const dnnl::memory::desc c_md({static_cast<dnnl::memory::dim>(M),
                                 static_cast<dnnl::memory::dim>(N)}, d, tag::ab);

  dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md);
  dnnl::matmul prim(pd);

  auto usm = [&](const dnnl::memory::desc& md, void* p) {
    return dnnl::sycl_interop::make_memory(
        md, eng, dnnl::sycl_interop::memory_kind::usm, p);
  };
  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, usm(a_md, const_cast<void*>(a))},
      {DNNL_ARG_WEIGHTS, usm(b_md, const_cast<void*>(b))},
      {DNNL_ARG_DST, usm(c_md, c)},
  };
  return dnnl::sycl_interop::execute(prim, strm, args);
}

}  // namespace quixicore::xpu::kernels

#endif  // QUIXICORE_XPU_HAS_ONEDNN
