// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace quixicore {

// Quarantined helper: not included by the serving sampler yet. All lanes in
// a warp address the same shared-memory histogram. Combining identical bins
// changes only the number of atomics, not the integer histogram counts.
__device__ __forceinline__ void warp_histogram_add(int* histogram, int bin) {
  const unsigned active = __activemask();
  const unsigned peers = __match_any_sync(active, bin);
  const int leader = __ffs(peers) - 1;
  if ((threadIdx.x & 31) == leader) {
    atomicAdd(histogram + bin, __popc(peers));
  }
}

}  // namespace quixicore
