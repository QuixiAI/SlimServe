#include "quixicore/xpu/runtime.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace quixicore::xpu {

std::size_t dtype_size(const DType dt) noexcept {
  switch (dt) {
    case DType::f32:
      return 4;
    case DType::f16:
      return 2;
    case DType::bf16:
      return 2;
  }
  return 0;
}

const char* dtype_name(const DType dt) noexcept {
  switch (dt) {
    case DType::f32:
      return "f32";
    case DType::f16:
      return "f16";
    case DType::bf16:
      return "bf16";
  }
  return "unknown";
}

std::vector<sycl::device> gpu_devices() {
  std::vector<sycl::device> devices;
  bool selected_level_zero = false;
  for (const auto& platform : sycl::platform::get_platforms()) {
    std::vector<sycl::device> platform_devices;
    for (auto& device : platform.get_devices()) {
      if (device.is_gpu() && device.get_info<sycl::info::device::vendor_id>() == 0x8086u) {
        platform_devices.push_back(device);
      }
    }
    const bool level_zero = platform.get_backend() == sycl::backend::ext_oneapi_level_zero;
    if (platform_devices.size() > devices.size() ||
        (platform_devices.size() == devices.size() && level_zero && !selected_level_zero)) {
      devices = std::move(platform_devices);
      selected_level_zero = level_zero;
    }
  }
  return devices;
}

sycl::queue make_gpu_queue(const std::size_t index, const bool enable_profiling) {
  const auto devices = gpu_devices();
  if (devices.empty()) {
    throw std::runtime_error(
        "QuixiCore XPU: no SYCL GPU device found. Source the oneAPI environment "
        "(setvars.sh) and check `sycl-ls`.");
  }
  const std::size_t pick = std::min(index, devices.size() - 1);

  sycl::property_list props =
      enable_profiling
          ? sycl::property_list{sycl::property::queue::in_order(),
                                sycl::property::queue::enable_profiling()}
          : sycl::property_list{sycl::property::queue::in_order()};

  return sycl::queue{devices[pick], props};
}

std::string describe_device(const sycl::device& dev) {
  std::ostringstream os;
  os << dev.get_info<sycl::info::device::name>() << " | vendor="
     << dev.get_info<sycl::info::device::vendor>() << " | driver="
     << dev.get_info<sycl::info::device::driver_version>() << " | compute_units="
     << dev.get_info<sycl::info::device::max_compute_units>()
     << " | max_wg=" << dev.get_info<sycl::info::device::max_work_group_size>();

  const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  os << " | sub_group_sizes=";
  for (std::size_t i = 0; i < sg_sizes.size(); ++i) {
    os << (i ? "," : "") << sg_sizes[i];
  }
  return os.str();
}

}  // namespace quixicore::xpu
