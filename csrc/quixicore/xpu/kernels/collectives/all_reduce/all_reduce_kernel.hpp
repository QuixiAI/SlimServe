#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Fixed launch geometry so capture/replay and every rank agree exactly.
inline constexpr int kArBlocks = 64;   // launched work-groups
inline constexpr int kArWgSize = 256;  // work-items per group
inline constexpr int kArMaxWorld = 8;  // flag arrays sized for this bound
static_assert(kArWgSize >= kArMaxWorld, "one work-item per rank in the barrier");

// Per-rank rendezvous state at the head of each rank's region. Three separate
// flag phases (arrive / mid / depart) so a peer racing ahead to a later
// rendezvous can never clobber a counter still being polled; the generation
// counter lives in device memory and self-increments each execution, which is
// what lets the handshake survive graph replay.
struct alignas(128) ArSignal {
  std::uint32_t sync[3][kArBlocks][kArMaxWorld];
  std::uint32_t generation[kArBlocks];
};

// A rank's region is [ArSignal | staging]; every region must be zero-filled
// once before its first collective.
constexpr std::size_t all_reduce_region_bytes(std::size_t staging_bytes) {
  return sizeof(ArSignal) + staging_bytes;
}

// Capturable sum all-reduce over peer-visible regions. regions[r] is rank r's
// [ArSignal | staging] base pointer as mapped in THIS rank's address space
// (same-context USM in-process, or zeMemOpenIpcHandle-mapped cross-process);
// regions[rank] is our own. Staging in each region must hold numel elements of
// dt. Payloads below twoshot_min_bytes use the one-shot algorithm, larger ones
// reduce-scatter + all-gather. Returns the reduce kernel's event.
sycl::event all_reduce_sycl(sycl::queue& q, const void* inp, void* out,
                            void* const* regions, int rank, int world,
                            std::size_t numel, DType dt,
                            std::size_t twoshot_min_bytes);

}  // namespace quixicore::xpu::kernels
