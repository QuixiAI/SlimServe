// Collectives family. The core is the capturable P2P sum all-reduce in
// kernels/collectives/all_reduce (one-shot + two-shot over peer-visible
// [ArSignal | staging] regions). `all_reduce` is the device-side API for
// callers that own regions (in-process shared-context USM, or IPC-mapped
// cross-process — the fd exchange itself is integration-owned, see
// quixicore/xpu/ipc.hpp). `all_reduce_sum` remains the host-buffer
// demonstrator, now orchestrating the real collective across every visible
// GPU in one shared context. Capability-gated: >1 GPU required.

#include <cstring>
#include <vector>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/ops.hpp"
#include "quixicore/xpu/runtime.hpp"

#include "collectives/all_reduce/all_reduce_kernel.hpp"

namespace quixicore::xpu::ops {

std::size_t all_reduce_region_bytes(std::size_t staging_bytes) {
  return kernels::all_reduce_region_bytes(staging_bytes);
}

void all_reduce(sycl::queue& q, const void* inp, void* out,
                void* const* regions, int rank, int world, std::size_t numel,
                DType dt, std::size_t twoshot_min_bytes, Variant variant,
                bool blocking) {
  (void)variant;  // native only; oneCCL vendor variant deferred
  if (world < 2 || world > kernels::kArMaxWorld || rank < 0 || rank >= world) {
    return;  // outside the rendezvous envelope: reject, don't corrupt
  }
  sycl::event ev = kernels::all_reduce_sycl(q, inp, out, regions, rank, world,
                                            numel, dt, twoshot_min_bytes);
  if (blocking) ev.wait();
}

std::size_t all_reduce_sum(const float* in_per_gpu, float* out,
                           std::size_t count) {
  // The B60s are exposed on multiple backends (Level Zero + OpenCL); a single
  // context cannot span platforms, so pick the single platform exposing the
  // most GPUs.
  std::vector<sycl::device> devices;
  for (const auto& p : sycl::platform::get_platforms()) {
    std::vector<sycl::device> gpus;
    for (auto& d : p.get_devices())
      if (d.is_gpu()) gpus.push_back(d);
    if (gpus.size() > devices.size()) devices = std::move(gpus);
  }
  std::size_t ng = devices.size();
  if (ng == 0) return 0;
  if (ng > static_cast<std::size_t>(kernels::kArMaxWorld))
    ng = kernels::kArMaxWorld;
  if (ng == 1) {
    std::memcpy(out, in_per_gpu, count * sizeof(float));
    return 1;
  }

  for (std::size_t a = 0; a < ng; ++a) {
    for (std::size_t b = 0; b < ng; ++b) {
      if (a == b) continue;
      try {
        devices[a].ext_oneapi_enable_peer_access(devices[b]);
      } catch (const sycl::exception&) {
        // Already enabled (or unsupported — the collective will then read
        // garbage and the caller's check fails loudly rather than hanging).
      }
    }
  }

  sycl::context ctx(devices);
  std::vector<sycl::queue> qs;
  qs.reserve(ng);
  for (std::size_t g = 0; g < ng; ++g) qs.emplace_back(ctx, devices[g]);

  const std::size_t region_bytes =
      kernels::all_reduce_region_bytes(count * sizeof(float));
  void* regions[kernels::kArMaxWorld] = {};
  std::vector<float*> in_dev(ng), out_dev(ng);
  for (std::size_t g = 0; g < ng; ++g) {
    regions[g] = sycl::malloc_device(region_bytes, qs[g]);
    in_dev[g] = sycl::malloc_device<float>(count, qs[g]);
    out_dev[g] = sycl::malloc_device<float>(count, qs[g]);
    qs[g].memset(regions[g], 0, region_bytes);
    qs[g].memcpy(in_dev[g], in_per_gpu + g * count, count * sizeof(float));
  }
  for (auto& q : qs) q.wait();

  std::vector<sycl::event> done(ng);
  for (std::size_t g = 0; g < ng; ++g) {
    done[g] = kernels::all_reduce_sycl(qs[g], in_dev[g], out_dev[g], regions,
                                       static_cast<int>(g),
                                       static_cast<int>(ng), count, DType::f32,
                                       /*twoshot_min_bytes=*/65536);
  }
  for (auto& ev : done) ev.wait();
  qs[0].memcpy(out, out_dev[0], count * sizeof(float)).wait();

  for (std::size_t g = 0; g < ng; ++g) {
    sycl::free(regions[g], qs[g]);
    sycl::free(in_dev[g], qs[g]);
    sycl::free(out_dev[g], qs[g]);
  }
  return ng;
}

}  // namespace quixicore::xpu::ops
