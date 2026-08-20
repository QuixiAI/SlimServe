// Capturable P2P sum all-reduce over PCIe peer-visible staging regions.
//
// Each rank copies its input into its own staging buffer, rendezvouses with
// every peer through device-resident flags, then reduces. Two algorithms:
//   * one-shot (small payloads): every rank reads all `world` staging buffers
//     and writes the full sum — peer traffic (world-1) x payload, one
//     rendezvous round-trip; wins while latency dominates.
//   * two-shot (>= twoshot_min_bytes): reduce-scatter into each rank's own
//     chunk, mid rendezvous, then all-gather — 2(world-1)/world x payload.
//
// Two invariants shape this code (semantics shared with vLLM's proven
// custom-all-reduce design; independently expressed here):
//   * BITWISE CONSISTENCY: IEEE float addition is commutative but NOT
//     associative, so every rank accumulates in fixed rank order 0..world-1.
//     Any "self first" or tree order would silently desynchronize replicas.
//   * BLOCK-ALIGNED TWO-SHOT PARTITIONING: the rendezvous only synchronizes
//     work-group b with work-group b on every peer — it is not device-wide.
//     The all-gather may therefore only read bytes the same-numbered peer
//     group produced, which the identical per-block slicing guarantees.
//
// Flag protocol: the B60 PCIe pairing reports peer access_supported but NOT
// atomics_supported, so peer atomic RMW is invalid; arrival is a plain
// volatile store ordered by system-scope fences. Spins are bounded so a
// pathological handshake degrades to a wrong-but-caught result instead of
// hanging the device. -DQUIXICORE_XPU_AR_SYSTEM_ATOMICS opts into atomic
// flags on hardware that reports peer-atomic support.

#include "collectives/all_reduce/all_reduce_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

inline constexpr std::uint64_t kSpinCap = 2000000000ull;
inline constexpr int kArPhaseArrive = 0;
inline constexpr int kArPhaseMid = 1;
inline constexpr int kArPhaseDepart = 2;

struct RegionSet {
  ArSignal* sig[kArMaxWorld];
};
template <typename T>
struct StagingSet {
  T* base[kArMaxWorld];
};

inline void publish_flag(std::uint32_t* slot, std::uint32_t value) {
#ifdef QUIXICORE_XPU_AR_SYSTEM_ATOMICS
  sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                   sycl::memory_scope::system,
                   sycl::access::address_space::global_space>
      ref(*slot);
  ref.store(value, sycl::memory_order::release);
#else
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
  *const_cast<volatile std::uint32_t*>(slot) = value;
#endif
}

inline std::uint32_t observe_flag(std::uint32_t* slot) {
#ifdef QUIXICORE_XPU_AR_SYSTEM_ATOMICS
  sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                   sycl::memory_scope::system,
                   sycl::access::address_space::global_space>
      ref(*slot);
  return ref.load(sycl::memory_order::acquire);
#else
  const std::uint32_t value = *const_cast<volatile std::uint32_t*>(slot);
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  return value;
#endif
}

// Work-item `lid` owns rank `lid`: it publishes our arrival into that rank's
// region and waits for that rank's arrival in ours, so one pass covers every
// peer. `leading`/`trailing` control the group barriers around the exchange
// (the arrive phase needs none before it; the final depart needs none after).
template <bool leading, bool trailing>
inline void rendezvous(int phase, const RegionSet& regions, ArSignal* self,
                       int block, int rank, int world,
                       const sycl::nd_item<1>& item) {
  if constexpr (leading) sycl::group_barrier(item.get_group());
  const std::uint32_t target = self->generation[block] + 1;
  const int lid = static_cast<int>(item.get_local_linear_id());
  if (lid < world) {
    publish_flag(&regions.sig[lid]->sync[phase][block][rank], target);
    std::uint32_t* mine = &self->sync[phase][block][lid];
    std::uint64_t spins = 0;
    while (observe_flag(mine) < target && ++spins < kSpinCap) {
    }
  }
  if constexpr (trailing) sycl::group_barrier(item.get_group());
  if (lid == 0) self->generation[block] = target;
}

template <typename T>
class ArStageKernel;
template <typename T>
class ArOneShotKernel;
template <typename T>
class ArTwoShotKernel;

template <typename T>
sycl::event launch_typed(sycl::queue& q, const T* inp, T* out,
                         void* const* region_ptrs, int rank, int world,
                         std::size_t numel, std::size_t twoshot_min_bytes) {
  RegionSet regions{};
  StagingSet<T> staging{};
  for (int r = 0; r < world; ++r) {
    char* base = static_cast<char*>(region_ptrs[r]);
    regions.sig[r] = reinterpret_cast<ArSignal*>(base);
    staging.base[r] = reinterpret_cast<T*>(base + sizeof(ArSignal));
  }
  ArSignal* self_sig = regions.sig[rank];
  T* self_staging = staging.base[rank];

  const sycl::nd_range<1> ndr(sycl::range<1>(kArBlocks * kArWgSize),
                              sycl::range<1>(kArWgSize));
  constexpr std::size_t stride = kArBlocks * kArWgSize;

  // Stage first; the reduce depends on this event explicitly (correct on both
  // in-order and out-of-order queues). Each peer's staging is complete before
  // that peer flags arrival, so the arrive rendezvous publishes it.
  sycl::event staged = q.parallel_for<ArStageKernel<T>>(
      ndr, [=](sycl::nd_item<1> item) {
        for (std::size_t i = item.get_global_linear_id(); i < numel;
             i += stride) {
          self_staging[i] = inp[i];
        }
      });

  if (numel * sizeof(T) < twoshot_min_bytes) {
    return q.submit([&](sycl::handler& h) {
      h.depends_on(staged);
      h.parallel_for<ArOneShotKernel<T>>(ndr, [=](sycl::nd_item<1> item) {
        const int block = static_cast<int>(item.get_group(0));
        rendezvous<false, true>(kArPhaseArrive, regions, self_sig, block, rank,
                                world, item);
        for (std::size_t i = item.get_global_linear_id(); i < numel;
             i += stride) {
          float acc = 0.0f;
          for (int r = 0; r < world; ++r) {  // fixed rank order — see header
            acc += static_cast<float>(staging.base[r][i]);
          }
          out[i] = static_cast<T>(acc);
        }
        rendezvous<true, false>(kArPhaseDepart, regions, self_sig, block, rank,
                                world, item);
      });
    });
  }

  // Two-shot. Per-rank chunk, sliced per block with identical offsets on every
  // rank (see the partitioning invariant in the header comment).
  const std::size_t chunk = (numel + world - 1) / world;
  const std::size_t per_block = (chunk + kArBlocks - 1) / kArBlocks;

  return q.submit([&](sycl::handler& h) {
    h.depends_on(staged);
    h.parallel_for<ArTwoShotKernel<T>>(ndr, [=](sycl::nd_item<1> item) {
      const int block = static_cast<int>(item.get_group(0));
      const std::size_t lid = item.get_local_linear_id();

      auto slice_lo = [=](int owner) {
        return static_cast<std::size_t>(owner) * chunk +
               static_cast<std::size_t>(block) * per_block;
      };
      auto slice_hi = [=](int owner) {
        std::size_t within = static_cast<std::size_t>(block + 1) * per_block;
        if (within > chunk) within = chunk;
        std::size_t end = static_cast<std::size_t>(owner) * chunk + within;
        return end < numel ? end : numel;
      };

      rendezvous<false, true>(kArPhaseArrive, regions, self_sig, block, rank,
                              world, item);

      // Reduce-scatter: sum our own chunk across all ranks in fixed order and
      // publish it back into our staging (nobody else reads chunk `rank` in
      // this phase, so the overwrite is safe).
      for (std::size_t i = slice_lo(rank) + lid; i < slice_hi(rank);
           i += kArWgSize) {
        float acc = 0.0f;
        for (int r = 0; r < world; ++r) {
          acc += static_cast<float>(staging.base[r][i]);
        }
        self_staging[i] = static_cast<T>(acc);
      }

      rendezvous<true, true>(kArPhaseMid, regions, self_sig, block, rank, world,
                             item);

      // All-gather: copy every owner's reduced slice for THIS block.
      for (int owner = 0; owner < world; ++owner) {
        const T* src = staging.base[owner];
        for (std::size_t i = slice_lo(owner) + lid; i < slice_hi(owner);
             i += kArWgSize) {
          out[i] = src[i];
        }
      }

      rendezvous<true, false>(kArPhaseDepart, regions, self_sig, block, rank,
                              world, item);
    });
  });
}

}  // namespace

sycl::event all_reduce_sycl(sycl::queue& q, const void* inp, void* out,
                            void* const* regions, int rank, int world,
                            std::size_t numel, DType dt,
                            std::size_t twoshot_min_bytes) {
  switch (dt) {
    case DType::f32:
      return launch_typed(q, static_cast<const float*>(inp),
                          static_cast<float*>(out), regions, rank, world, numel,
                          twoshot_min_bytes);
    case DType::f16:
      return launch_typed(q, static_cast<const half_t*>(inp),
                          static_cast<half_t*>(out), regions, rank, world,
                          numel, twoshot_min_bytes);
    case DType::bf16:
      return launch_typed(q, static_cast<const bf16_t*>(inp),
                          static_cast<bf16_t*>(out), regions, rank, world,
                          numel, twoshot_min_bytes);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
