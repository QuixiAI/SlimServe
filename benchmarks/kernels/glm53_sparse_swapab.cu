// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "quixicore/serving/glm53_sparse_swapab.cuh"

void run(at::Tensor q, at::Tensor cache, at::Tensor bt, at::Tensor ids,
         at::Tensor lengths, at::Tensor out, int block_size, double scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16);
  const c10::cuda::CUDAGuard guard(q.device());
  for (const auto& t : {q, cache, bt, ids, lengths, out})
    TORCH_CHECK(t.device() == q.device());
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && bt.is_contiguous() &&
              ids.is_contiguous() && lengths.is_contiguous());
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 32 && q.size(2) == 512);
  TORCH_CHECK(cache.scalar_type() == q.scalar_type() && out.scalar_type() == q.scalar_type());
  TORCH_CHECK(out.sizes() == q.sizes() && cache.dim() == 3 && cache.size(0) > 0 &&
              cache.size(1) == block_size && cache.size(2) == 512 &&
              cache.stride(1) == 512 && cache.stride(2) == 1);
  TORCH_CHECK(bt.scalar_type() == at::kInt && ids.scalar_type() == at::kInt &&
              lengths.scalar_type() == at::kInt && bt.dim() == 2 && ids.dim() == 2 &&
              bt.size(0) == q.size(0) && ids.size(0) == q.size(0) &&
              lengths.numel() == q.size(0));
  if (q.size(0) == 0) return;
  namespace impl = slimserve::glm53_swapab;
  auto kernel = impl::sparse_nope;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sizeof(impl::Shared)));
  kernel<<<q.size(0), impl::THREADS, sizeof(impl::Shared), at::cuda::getCurrentCUDAStream()>>>(
      (const impl::BF*)q.data_ptr(), (const impl::BF*)cache.data_ptr(),
      bt.data_ptr<int>(), ids.data_ptr<int>(), lengths.data_ptr<int>(),
      (impl::BF*)out.data_ptr(), ids.size(1), bt.size(1), block_size,
      cache.stride(0), float(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict info() {
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, slimserve::glm53_swapab::sparse_nope));
  pybind11::dict d;
  d["registers"] = attr.numRegs; d["local_bytes"] = attr.localSizeBytes;
  d["shared"] = sizeof(slimserve::glm53_swapab::Shared);
  return d;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); m.def("info", &info); }
