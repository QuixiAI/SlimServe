// int8 w8a8 GEMM, vendor variant via oneDNN int8 matmul (XMX/DPAS int8).
// Per-row (src) and per-col (weights) fp32 scales via primitive attributes.
// Compiled only when oneDNN is found.

#include "quantization/qgemm/qgemm_kernel.hpp"

#if defined(QUIXICORE_XPU_HAS_ONEDNN)

#include <unordered_map>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace quixicore::xpu::kernels {
namespace {

dnnl::memory::data_type out_dnnl(DType dt) {
  switch (dt) {
    case DType::f32: return dnnl::memory::data_type::f32;
    case DType::f16: return dnnl::memory::data_type::f16;
    case DType::bf16: return dnnl::memory::data_type::bf16;
  }
  return dnnl::memory::data_type::f32;
}

}  // namespace

sycl::event qgemm_int8_onednn(sycl::queue& q, const void* a, const void* b,
                              const void* a_scale, const void* b_scale, void* c,
                              std::size_t M, std::size_t N, std::size_t K,
                              DType out_dt) {
  dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
  dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;
  const auto md = [](std::size_t r, std::size_t cc, dt t, tag tg) {
    return dnnl::memory::desc({static_cast<dnnl::memory::dim>(r),
                               static_cast<dnnl::memory::dim>(cc)}, t, tg);
  };
  const dnnl::memory::desc a_md = md(M, K, dt::s8, tag::ab);
  const dnnl::memory::desc b_md = md(K, N, dt::s8, tag::ab);
  const dnnl::memory::desc c_md = md(M, N, out_dnnl(out_dt), tag::ab);
  const dnnl::memory::desc bs_md({static_cast<dnnl::memory::dim>(N)}, dt::f32, tag::a);
  // Per-row (per-token) activation scale: oneDNN GPU int8 matmul does NOT accept
  // a per-M src scale attribute (only per-tensor src / per-channel weights), so
  // apply it as a binary-mul post-op with an [M,1] tensor that broadcasts over N.
  const dnnl::memory::desc as_md({static_cast<dnnl::memory::dim>(M), 1}, dt::f32, tag::ab);

  try {
    dnnl::primitive_attr attr;
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);  // per-col (N) weight scale
    dnnl::post_ops po;
    po.append_binary(dnnl::algorithm::binary_mul, as_md);  // * a_scale[m]
    attr.set_post_ops(po);

    dnnl::matmul::primitive_desc pd(eng, a_md, b_md, c_md, attr);
    dnnl::matmul prim(pd);

    auto usm = [&](const dnnl::memory::desc& m, void* p) {
      return dnnl::sycl_interop::make_memory(
          m, eng, dnnl::sycl_interop::memory_kind::usm, p);
    };
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, usm(a_md, const_cast<void*>(a))},
        {DNNL_ARG_WEIGHTS, usm(b_md, const_cast<void*>(b))},
        {DNNL_ARG_DST, usm(c_md, c)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, usm(bs_md, const_cast<void*>(b_scale))},
        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
         usm(as_md, const_cast<void*>(a_scale))},
    };
    return dnnl::sycl_interop::execute(prim, strm, args);
  } catch (const dnnl::error&) {
    return qgemm_int8_sycl(q, a, b, a_scale, b_scale, c, M, N, K, out_dt);
  }
}

}  // namespace quixicore::xpu::kernels

#endif  // QUIXICORE_XPU_HAS_ONEDNN
