#pragma once

// Level Zero IPC primitives for sharing device-USM regions across PROCESSES
// (the in-process path needs none of this — same-context USM pointers are
// already peer-visible once peer access is enabled).
//
// An exported fd is process-local: unlike a CUDA IPC blob it cannot be passed
// by value through shared memory or an environment variable — it must be
// duplicated into the peer process by the kernel, i.e. sent over a Unix
// domain socket with SCM_RIGHTS. That rendezvous is integration-owned and
// deliberately NOT part of this library (examples/p2p_all_reduce_mp.cpp
// shows a reference socketpair exchange).
//
// Requires a Level Zero toolchain at build time; when absent the backend
// still builds and available() reports false.

#include <sycl/sycl.hpp>

namespace quixicore::xpu::ipc {

// True when the backend was built with Level Zero IPC support and the queue's
// backend is Level Zero.
bool available(sycl::queue& q);

// Export a device-USM allocation (e.g. an all_reduce region) as a dma_buf fd.
// Returns -1 on failure. The allocation must stay alive while exported.
int export_region_fd(sycl::queue& q, void* region);

// Map a peer's exported region fd (already duplicated into THIS process) into
// this process's address space. Returns nullptr on failure.
void* open_region_fd(sycl::queue& q, int fd);

// Unmap a peer region previously returned by open_region_fd.
void close_region(sycl::queue& q, void* peer_region);

// Peer capability probe / enable. `atomics` selects the peer-atomics
// capability (false = plain access; the B60 PCIe pairing reports access
// without atomics, which is exactly what the all_reduce flag protocol needs).
bool can_access_peer(const sycl::device& self, const sycl::device& peer,
                     bool atomics);
void enable_peer_access(sycl::device self, const sycl::device& peer);

}  // namespace quixicore::xpu::ipc
