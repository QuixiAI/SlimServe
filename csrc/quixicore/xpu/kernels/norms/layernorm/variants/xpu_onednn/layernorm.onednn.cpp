// LayerNorm, vendor variant via oneDNN layer_normalization.
//
// Co-equal shipped implementation selectable as Variant::vendor. Runs on the
// caller's sycl::queue via oneDNN SYCL interop.
//
// oneDNN's layer_normalization takes a user-provided scale/shift data type (the
// basic primitive_desc overloads merely DEFAULT it to f32; oneDNN >= 3.x also
// exposes an overload with an explicit scale_shift_data_type). We pass the input
// dtype so weight/bias are consumed in their native dtype with no conversion.
// If a given (src, scale/shift) dtype combination is unsupported by the GPU
// implementation, primitive construction throws and we fall back to native SYCL.
//
// Compiled only when oneDNN is found (QUIXICORE_XPU_HAS_ONEDNN).

#include "norms/norms_kernel.hpp"

#if defined(QUIXICORE_XPU_HAS_ONEDNN)

#include <cstdio>
#include <cstdlib>
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

sycl::event layernorm_onednn(sycl::queue& q, const void* x, const void* weight,
                             const void* bias, void* out, std::size_t rows,
                             std::size_t dim, float eps, DType dt) {
  try {
    dnnl::engine eng =
        dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

    using tag = dnnl::memory::format_tag;
    const dnnl::memory::data_type d = to_dnnl(dt);
    const dnnl::memory::dims xdims = {static_cast<dnnl::memory::dim>(rows),
                                      static_cast<dnnl::memory::dim>(dim)};
    const dnnl::memory::dims sdims = {static_cast<dnnl::memory::dim>(dim)};
    const dnnl::memory::desc src_md(xdims, d, tag::ab);
    const dnnl::memory::desc dst_md(xdims, d, tag::ab);
    // scale/shift carried in the input dtype (see file header).
    const dnnl::memory::desc ss_md(sdims, d, tag::a);

    auto flags = dnnl::normalization_flags::use_scale;
    if (bias != nullptr) flags |= dnnl::normalization_flags::use_shift;

    // Explicit scale_shift_data_type overload (not the f32-defaulting one).
    dnnl::layer_normalization_forward::primitive_desc pd(
        eng, dnnl::prop_kind::forward_inference, src_md, dst_md, d, eps, flags);
    dnnl::layer_normalization_forward prim(pd);

    auto usm = [&](const dnnl::memory::desc& md, void* p) {
      return dnnl::sycl_interop::make_memory(
          md, eng, dnnl::sycl_interop::memory_kind::usm, p);
    };
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, usm(src_md, const_cast<void*>(x))},
        {DNNL_ARG_DST, usm(dst_md, out)},
        {DNNL_ARG_SCALE, usm(ss_md, const_cast<void*>(weight))},
    };
    if (bias != nullptr) {
      args.emplace(DNNL_ARG_SHIFT, usm(ss_md, const_cast<void*>(bias)));
    }

    return dnnl::sycl_interop::execute(prim, strm, args);
  } catch (const dnnl::error& e) {
    // Unsupported dtype combination on this device: fall back to native SYCL.
    if (std::getenv("QUIXICORE_XPU_TRACE_FALLBACK")) {
      std::fprintf(stderr, "[xpu] layernorm oneDNN fallback (dt=%s): %s\n",
                   dtype_name(dt), e.message);
    }
    return layernorm_sycl(q, x, weight, bias, out, rows, dim, eps, dt);
  }
}

}  // namespace quixicore::xpu::kernels

#endif  // QUIXICORE_XPU_HAS_ONEDNN
