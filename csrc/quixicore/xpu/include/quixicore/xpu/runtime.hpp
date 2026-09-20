#pragma once

// QuixiCore XPU runtime layer.
//
// Device and queue management for the Intel GPU backend. This header requires a
// SYCL toolchain (icpx / -fsycl) and is only compiled into the library when
// QUIXICORE_XPU_ENABLE_SYCL is ON. The framework-agnostic op ABI in ops.hpp is
// built on top of the primitives declared here.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace quixicore::xpu {

// Element types supported at the op ABI boundary. Compute always accumulates in
// fp32 regardless of storage type (see docs/perf.md numeric conventions).
enum class DType : std::uint8_t {
  f32,
  f16,
  bf16,
};

// Storage type aliases used by the SYCL kernels.
using half_t = sycl::half;
using bf16_t = sycl::ext::oneapi::bfloat16;

// Size in bytes of one element of the given dtype.
std::size_t dtype_size(DType dt) noexcept;

// Human-readable dtype name (matches the contract vocabulary: "f32"/"f16"/"bf16").
const char* dtype_name(DType dt) noexcept;

// Enumerate Intel GPUs from one SYCL platform, preferring Level Zero on ties so
// OpenCL aliases do not duplicate physical devices. Ordered as the selected
// platform reports them.
std::vector<sycl::device> gpu_devices();

// Build a queue on a GPU device. `index` selects among gpu_devices(); it is
// clamped to the available range. `enable_profiling` turns on queue profiling
// events, required by the benchmark harness for on-device kernel timing.
//
// The queue is in-order so that op sequences without explicit dependencies
// still execute in submission order (simpler correctness story for bring-up;
// revisit for overlap once multi-stream kernels land).
sycl::queue make_gpu_queue(std::size_t index = 0, bool enable_profiling = false);

// One-line description of a device: name + backend + a few key limits. Used by
// the device probe and by benchmark run metadata.
std::string describe_device(const sycl::device& dev);

}  // namespace quixicore::xpu
