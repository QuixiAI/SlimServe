// SPDX-License-Identifier: Apache-2.0
// Isolated GLM53 SM120 candidate; no serving dispatch uses this header yet.
#pragma once

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

namespace tms::glm_stable_align {

constexpr int EXPERTS = 288;
constexpr int THREADS = 256;
constexpr int MAX_WORDS = 8192 * 8 / 32;

// Integer histograms determine the same padded expert ranges as Marlin's
// alignment. A second CTA initializes all sorted capacity, including tails.
template<int COUNT_THREADS, bool AGGREGATE = true>
__global__ void count_prefix(const int* ids, int* sorted, int* experts,
                             int* padded, int* offsets, int numel, int block,
                             int capacity, int max_blocks) {
    static_assert(COUNT_THREADS == 256 || COUNT_THREADS == 1024);
    constexpr int WARPS = COUNT_THREADS / 32;
    constexpr int ITEMS = (EXPERTS + COUNT_THREADS - 1) / COUNT_THREADS;
    const int tid = threadIdx.x;
    if (blockIdx.x == 1) {
        for (int i = tid; i < capacity; i += COUNT_THREADS) sorted[i] = numel;
        return;
    }
    __shared__ int histogram[WARPS * EXPERTS];
    using Scan = cub::BlockScan<int, COUNT_THREADS>;
    __shared__ typename Scan::TempStorage scan;
    for (int i = tid; i < WARPS * EXPERTS; i += COUNT_THREADS) histogram[i] = 0;
    __syncthreads();

    const int lane = tid % 32, warp = tid / 32;
    // All lanes execute every match, including a partially populated last tile.
    for (int base = 0; base < numel; base += COUNT_THREADS) {
        const int i = base + tid;
        const int expert = i < numel ? ids[i] : -1;
        if constexpr (AGGREGATE) {
            const unsigned peers = __match_any_sync(0xffffffffu, expert);
            if (expert >= 0 && expert < EXPERTS && lane == __ffs(peers) - 1) {
                atomicAdd(histogram + warp * EXPERTS + expert, __popc(peers));
            }
        } else {
            if (expert >= 0 && expert < EXPERTS) {
                atomicAdd(histogram + warp * EXPERTS + expert, 1);
            }
        }
    }
    __syncthreads();

    int counts[ITEMS], starts[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int expert = ITEMS * tid + j;
        int count = 0;
        if (expert < EXPERTS) {
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) count += histogram[w * EXPERTS + expert];
        }
        counts[j] = (count + block - 1) / block * block;
    }
    Scan(scan).ExclusiveSum(counts, starts);
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int expert = ITEMS * tid + j;
        if (expert <= EXPERTS) offsets[expert] = starts[j];
        if (expert < EXPERTS) {
            for (int i = 0; i < counts[j]; i += block) {
                experts[(starts[j] + i) / block] = expert;
            }
        } else if (expert == EXPERTS) {
            *padded = starts[j];
            for (int i = starts[j] / block; i < max_blocks; ++i) experts[i] = -1;
        }
    }
}

// One CTA per expert. Ballots encode membership in flattened assignment order;
// scanning popcounts gives stable output positions without atomics or a sort.
// At the largest shape all CTAs collectively read 72 MiB of route IDs. This
// deliberate L2-traffic tradeoff must be measured; it is not a bandwidth claim.
__global__ void scatter_bitmap(const int* ids, int* sorted, const int* offsets,
                               int numel) {
    const int expert = blockIdx.x, tid = threadIdx.x;
    if (offsets[expert] == offsets[expert + 1]) return;  // CTA-uniform
    __shared__ unsigned bitmap[MAX_WORDS];
    using Scan = cub::BlockScan<int, THREADS>;
    __shared__ typename Scan::TempStorage scan;
    const int words = (numel + 31) / 32;
    for (int base = 0; base < numel; base += THREADS) {
        const int i = base + tid;
        const bool selected = i < numel && ids[i] == expert;
        const unsigned mask = __ballot_sync(0xffffffffu, selected);
        if (tid % 32 == 0 && i / 32 < words) bitmap[i / 32] = mask;
    }
    __syncthreads();

    constexpr int ITEMS = MAX_WORDS / THREADS;
    int counts[ITEMS], starts[ITEMS];
    unsigned masks[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int word = tid * ITEMS + j;
        masks[j] = word < words ? bitmap[word] : 0;
        counts[j] = __popc(masks[j]);
    }
    Scan(scan).ExclusiveSum(counts, starts);
    const int begin = offsets[expert];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        unsigned mask = masks[j];
        int out = begin + starts[j];
        while (mask) {
            sorted[out++] = (tid * ITEMS + j) * 32 + __ffs(mask) - 1;
            mask &= mask - 1;
        }
    }
}

}  // namespace tms::glm_stable_align
