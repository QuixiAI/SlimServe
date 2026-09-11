// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cmath>
#include <mutex>
#include <set>
#include "glm53_sparse_swapab.cuh"

namespace {
torch::Tensor prefill(torch::Tensor q, torch::Tensor cache, torch::Tensor bt,
                      torch::Tensor ids, torch::Tensor lengths,
                      int64_t block_size, double scale, int64_t page_stride_bytes) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16,
              "SM120 sparse prefill requires CUDA BF16 queries");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 12 && props->minor == 0, "requires SM120");
  for (const auto& t : {cache, bt, ids, lengths})
    TORCH_CHECK(t.device() == q.device(), "all tensors must share a device");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == slimserve::glm53_swapab::HEADS &&
              q.size(2) == 512 && q.is_contiguous() && q.size(0) <= INT_MAX,
              "expected [B,16,512] Q");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0,
              "queries must be 4-byte-aligned for paired BF16 loads");
  TORCH_CHECK(block_size > 0 && block_size <= INT_MAX && std::isfinite(scale),
              "invalid block size or attention scale");
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.dim() == 3 &&
              cache.size(0) > 0 && cache.size(0) <= INT_MAX &&
              cache.size(1) == block_size && cache.size(2) == 512 &&
              cache.stride(1) == 512 && cache.stride(2) == 1 &&
              cache.stride(0) >= block_size * 512 && cache.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0,
              "expected 16-byte-aligned BF16 cache pages with packed 512-wide rows");
  TORCH_CHECK(page_stride_bytes == 0 || page_stride_bytes == cache.stride(0) * 2,
              "page stride must match the cache view");
  TORCH_CHECK(bt.scalar_type() == at::kInt && ids.scalar_type() == at::kInt &&
              lengths.scalar_type() == at::kInt && bt.is_contiguous() &&
              ids.is_contiguous() && lengths.is_contiguous() &&
              bt.dim() == 2 && ids.dim() == 2 && lengths.dim() == 1 &&
              bt.size(0) == q.size(0) && ids.size(0) == q.size(0) &&
              lengths.size(0) == q.size(0) && bt.size(1) <= INT_MAX &&
              ids.size(1) <= INT_MAX - 31,
              "invalid sparse index/table/length metadata");
  auto out = torch::empty_like(q);
  if (q.size(0) == 0) return out;
  namespace impl = slimserve::glm53_swapab;
  auto kernel = impl::sparse_nope;
  // Cache launch setup per device, not per process: CUDA function attributes
  // belong to the current device's context. No setup on repeated graph calls.
  static std::mutex mutex;
  static std::set<int> configured;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!configured.count(q.get_device())) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(impl::Shared)));
      configured.insert(q.get_device());
    }
  }
  kernel<<<q.size(0), impl::THREADS, sizeof(impl::Shared),
           at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const impl::BF*>(q.data_ptr()),
      static_cast<const impl::BF*>(cache.data_ptr()), bt.data_ptr<int>(),
      ids.data_ptr<int>(), lengths.data_ptr<int>(),
      static_cast<impl::BF*>(out.data_ptr()), ids.size(1), bt.size(1),
      block_size, cache.stride(0), float(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace

void init_glm53_sparse(pybind11::module_& m) {
  m.def("mla_prefill_bf16_sparse_nope_sm120", &prefill,
        pybind11::arg("q"), pybind11::arg("cache"), pybind11::arg("block_table"),
        pybind11::arg("indices"), pybind11::arg("lengths"),
        pybind11::arg("block_size"), pybind11::arg("scale"),
        pybind11::arg("page_stride_bytes") = 0);
}
