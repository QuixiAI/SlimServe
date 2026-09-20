// Level Zero IPC primitives (see quixicore/xpu/ipc.hpp). On Intel L0/Linux
// the IPC handle embeds a dma_buf fd in its first bytes; export copies it
// out, open copies it back in. Compiled to stubs when Level Zero headers are
// unavailable so the scaffold build stays green.

#include "quixicore/xpu/ipc.hpp"

#ifdef QUIXICORE_XPU_HAS_LEVEL_ZERO

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <cstring>

namespace quixicore::xpu::ipc {

bool available(sycl::queue& q) {
  return q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}

int export_region_fd(sycl::queue& q, void* region) {
  if (!available(q)) return -1;
  auto ze_ctx =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  ze_ipc_mem_handle_t handle{};
  if (zeMemGetIpcHandle(ze_ctx, region, &handle) != ZE_RESULT_SUCCESS)
    return -1;
  int fd = -1;
  std::memcpy(&fd, handle.data, sizeof(int));
  return fd;
}

void* open_region_fd(sycl::queue& q, int fd) {
  if (!available(q) || fd < 0) return nullptr;
  auto ze_ctx =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  auto ze_dev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  ze_ipc_mem_handle_t handle{};
  std::memcpy(handle.data, &fd, sizeof(int));
  void* peer = nullptr;
  if (zeMemOpenIpcHandle(ze_ctx, ze_dev, handle, 0, &peer) !=
      ZE_RESULT_SUCCESS)
    return nullptr;
  return peer;
}

void close_region(sycl::queue& q, void* peer_region) {
  if (!available(q) || peer_region == nullptr) return;
  auto ze_ctx =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  zeMemCloseIpcHandle(ze_ctx, peer_region);
}

bool can_access_peer(const sycl::device& self, const sycl::device& peer,
                     bool atomics) {
  if (self == peer) return true;
  const auto capability =
      atomics ? sycl::ext::oneapi::peer_access::atomics_supported
              : sycl::ext::oneapi::peer_access::access_supported;
  return const_cast<sycl::device&>(self).ext_oneapi_can_access_peer(peer,
                                                                    capability);
}

void enable_peer_access(sycl::device self, const sycl::device& peer) {
  if (self == peer) return;
  try {
    self.ext_oneapi_enable_peer_access(peer);
  } catch (const sycl::exception&) {
    // Already enabled.
  }
}

}  // namespace quixicore::xpu::ipc

#else  // !QUIXICORE_XPU_HAS_LEVEL_ZERO

namespace quixicore::xpu::ipc {

bool available(sycl::queue&) { return false; }
int export_region_fd(sycl::queue&, void*) { return -1; }
void* open_region_fd(sycl::queue&, int) { return nullptr; }
void close_region(sycl::queue&, void*) {}
bool can_access_peer(const sycl::device& self, const sycl::device& peer,
                     bool atomics) {
  if (self == peer) return true;
  const auto capability =
      atomics ? sycl::ext::oneapi::peer_access::atomics_supported
              : sycl::ext::oneapi::peer_access::access_supported;
  return const_cast<sycl::device&>(self).ext_oneapi_can_access_peer(peer,
                                                                    capability);
}
void enable_peer_access(sycl::device self, const sycl::device& peer) {
  if (self == peer) return;
  try {
    self.ext_oneapi_enable_peer_access(peer);
  } catch (const sycl::exception&) {
  }
}

}  // namespace quixicore::xpu::ipc

#endif
