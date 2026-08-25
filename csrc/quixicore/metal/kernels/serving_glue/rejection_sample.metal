#include <metal_stdlib>

using namespace metal;

namespace mittens {

// SplitMix64 stream used by the torch MPS fallback in sample/gumbel.py.
// Keeping the same seed/position/column keys makes the fused verifier a
// dispatch optimization, not a sampling-policy change.
METAL_FUNC ulong rejection_splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}

METAL_FUNC float rejection_uniform(long seed, long pos, int column) {
    constexpr ulong gamma = 0x9E3779B97F4A7C15UL;
    const ulong row = rejection_splitmix64(
        rejection_splitmix64(as_type<ulong>(seed)) ^ as_type<ulong>(pos));
    const ulong z = rejection_splitmix64(row + ulong(column) * gamma);
    const uint top24 = uint(z >> 40);
    return metal::max(float(top24) * (1.0f / 16777216.0f),
                      1.0f / 16777216.0f);
}

// Full-row fp32 logsumexp + argmax. Each of the 256 threads owns one
// contiguous vocabulary interval. Contiguous ownership is also what lets the
// caller perform inverse-CDF sampling in vocabulary order without storing a
// full probability/CDF tensor.
METAL_FUNC void rejection_row_stats(
    device const float* row, int vocab, uint tid,
    threadgroup float* scratch_value, threadgroup int* scratch_index,
    threadgroup float* out_lse, threadgroup int* out_argmax) {
    constexpr int NT = 256;
    const int chunk = (vocab + NT - 1) / NT;
    const int lo = int(tid) * chunk;
    const int hi = metal::min(vocab, lo + chunk);
    float local_max = -INFINITY;
    int local_id = vocab;
    for (int v = lo; v < hi; ++v) {
        float x = row[v];
        if (x != x) x = -INFINITY;
        if (x > local_max || (x == local_max && v < local_id)) {
            local_max = x;
            local_id = v;
        }
    }
    scratch_value[tid] = local_max;
    scratch_index[tid] = local_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = NT / 2; stride > 0; stride >>= 1) {
        if (int(tid) < stride) {
            const float other = scratch_value[tid + stride];
            const int other_id = scratch_index[tid + stride];
            if (other > scratch_value[tid] ||
                (other == scratch_value[tid] &&
                 other_id < scratch_index[tid])) {
                scratch_value[tid] = other;
                scratch_index[tid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float global_max = scratch_value[0];
    float local_sum = 0.0f;
    if (global_max > -INFINITY) {
        for (int v = lo; v < hi; ++v) {
            float x = row[v];
            if (x == x) local_sum += metal::exp(x - global_max);
        }
    }
    scratch_value[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = NT / 2; stride > 0; stride >>= 1) {
        if (int(tid) < stride) {
            scratch_value[tid] += scratch_value[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        *out_lse = scratch_value[0] > 0.0f
            ? global_max + metal::log(scratch_value[0]) : -INFINITY;
        *out_argmax = scratch_index[0] < vocab ? scratch_index[0] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Lossless Leviathan rejection sampling for the Metal serving path.
// target_logits are already temperature/top-k/top-p processed. draft_logits
// are either the DFlash selector's sparse temperature-applied distribution or
// absent (one-hot draft). One threadgroup handles a request, verifies its
// prefix, and samples the rejected residual / all-accepted bonus directly.
kernel void qwen38_rejection_sample(
    device long* sampled [[buffer(0)]],             // (R, steps+1)
    device int* num_sampled [[buffer(1)]],           // (R,)
    device const float* target [[buffer(2)]],        // (L,Vt)
    device const float* draft [[buffer(3)]],         // (states,steps,Vd)
    device const int* draft_sampled [[buffer(4)]],   // (L,)
    device const int* cu [[buffer(5)]],              // (R+1,)
    device const long* pos [[buffer(6)]],            // (L,)
    device const int* idx_mapping [[buffer(7)]],     // (R,)
    device const float* temperature [[buffer(8)]],   // (states,)
    device const long* seeds [[buffer(9)]],          // (states,)
    constant int& requests [[buffer(10)]],
    constant int& steps [[buffer(11)]],
    constant int& vocab [[buffer(12)]],
    constant long& target_stride [[buffer(13)]],
    constant long& draft_stride0 [[buffer(14)]],
    constant long& draft_stride1 [[buffer(15)]],
    constant int& has_draft [[buffer(16)]],
    uint req [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr int NT = 256;
    constexpr long ACCEPT_KEY = 1L << 40;
    constexpr long EMIT_KEY = 1L << 41;
    if (int(req) >= requests) return;

    threadgroup float scratch_value[NT];
    threadgroup float chunk_mass[NT];
    threadgroup float chunk_prefix[NT];
    threadgroup int scratch_index[NT];
    threadgroup float stat_lse;
    threadgroup int stat_argmax;
    threadgroup float target_lse;
    threadgroup float draft_lse;
    threadgroup int target_argmax;
    threadgroup int accepted;
    threadgroup int rejected;
    threadgroup int emit_row;
    threadgroup int emit_draft_step;
    threadgroup int emitted;
    threadgroup float total_mass;
    threadgroup float cutoff;

    const int out_stride = steps + 1;
    if (int(tid) < out_stride) {
        sampled[long(req) * out_stride + int(tid)] = -1;
    }
    if (tid == 0) {
        accepted = 0;
        rejected = -1;
        total_mass = 0.0f;
        cutoff = 0.0f;
        emitted = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int start = cu[req];
    const int nd = cu[req + 1] - start - 1;
    const int state = idx_mapping[req];
    const float temp = temperature[state];
    const long seed = seeds[state];
    const long pos0 = pos[start];

    // Verify until the first rejection. The branch is threadgroup-uniform.
    for (int step = 0; step < nd; ++step) {
        device const float* trow = target + long(start + step) * target_stride;
        rejection_row_stats(trow, vocab, tid, scratch_value, scratch_index,
                            &stat_lse, &stat_argmax);
        if (tid == 0) {
            target_lse = stat_lse;
            target_argmax = stat_argmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (has_draft != 0) {
            device const float* drow = draft + long(state) * draft_stride0 +
                                       long(step) * draft_stride1;
            rejection_row_stats(drow, vocab, tid, scratch_value,
                                scratch_index, &stat_lse, &stat_argmax);
            if (tid == 0) draft_lse = stat_lse;
        } else if (tid == 0) {
            draft_lse = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            const int tok = draft_sampled[start + step + 1];
            bool take;
            if (temp == 0.0f) {
                take = tok == target_argmax;
            } else {
                const float lp = trow[tok] - target_lse;
                const float lq = has_draft != 0
                    ? draft[long(state) * draft_stride0 +
                            long(step) * draft_stride1 + tok] - draft_lse
                    : 0.0f;
                float ratio = metal::exp(metal::min(lp - lq, 0.0f));
                if (ratio != ratio) ratio = 1.0f;
                const float u = rejection_uniform(seed, pos0 ^ ACCEPT_KEY,
                                                  step);
                take = u < ratio;
            }
            if (take) {
                sampled[long(req) * out_stride + step] = long(tok);
                accepted = step + 1;
            } else {
                rejected = step;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rejected >= 0) break;
    }

    if (tid == 0) {
        emit_row = start + accepted;
        emit_draft_step = metal::min(accepted, steps - 1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A rejected row's stats are still live from the loop. The all-accepted
    // bonus row has not been visited yet, so compute it here.
    if (accepted == nd) {
        device const float* trow = target + long(emit_row) * target_stride;
        rejection_row_stats(trow, vocab, tid, scratch_value, scratch_index,
                            &stat_lse, &stat_argmax);
        if (tid == 0) {
            target_lse = stat_lse;
            target_argmax = stat_argmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (temp == 0.0f) {
        if (tid == 0) emitted = target_argmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        const int chunk = (vocab + NT - 1) / NT;
        const int lo = int(tid) * chunk;
        const int hi = metal::min(vocab, lo + chunk);
        device const float* trow = target + long(emit_row) * target_stride;
        device const float* drow = draft + long(state) * draft_stride0 +
                                   long(emit_draft_step) * draft_stride1;
        const bool all_accepted = accepted == nd;
        const int one_hot = all_accepted || has_draft != 0
            ? -1 : draft_sampled[start + accepted + 1];

        float local_mass = 0.0f;
        for (int v = lo; v < hi; ++v) {
            float p = target_lse > -INFINITY
                ? metal::exp(trow[v] - target_lse) : 0.0f;
            if (p != p) p = 0.0f;
            float mass = p;
            if (!all_accepted) {
                if (has_draft != 0) {
                    float q = draft_lse > -INFINITY
                        ? metal::exp(drow[v] - draft_lse) : 0.0f;
                    if (q != q) q = 0.0f;
                    mass = metal::max(p - q, 0.0f);
                } else if (v == one_hot) {
                    mass = 0.0f;
                }
            }
            local_mass += mass;
        }
        chunk_mass[tid] = local_mass;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float run = 0.0f;
            for (int i = 0; i < NT; ++i) {
                chunk_prefix[i] = run;
                run += chunk_mass[i];
            }
            total_mass = run;
            cutoff = rejection_uniform(seed, pos0 ^ EMIT_KEY, 0) * run;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int found = vocab;
        if (total_mass > 0.0f && chunk_mass[tid] > 0.0f &&
            cutoff > chunk_prefix[tid] &&
            cutoff <= chunk_prefix[tid] + chunk_mass[tid]) {
            float run = chunk_prefix[tid];
            for (int v = lo; v < hi; ++v) {
                float p = target_lse > -INFINITY
                    ? metal::exp(trow[v] - target_lse) : 0.0f;
                if (p != p) p = 0.0f;
                float mass = p;
                if (!all_accepted) {
                    if (has_draft != 0) {
                        float q = draft_lse > -INFINITY
                            ? metal::exp(drow[v] - draft_lse) : 0.0f;
                        if (q != q) q = 0.0f;
                        mass = metal::max(p - q, 0.0f);
                    } else if (v == one_hot) {
                        mass = 0.0f;
                    }
                }
                run += mass;
                if (found == vocab && run >= cutoff) found = v;
            }
        }
        scratch_index[tid] = found;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int stride = NT / 2; stride > 0; stride >>= 1) {
            if (int(tid) < stride) {
                scratch_index[tid] = metal::min(scratch_index[tid],
                                                scratch_index[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            emitted = total_mass > 0.0f && scratch_index[0] < vocab
                ? scratch_index[0] : target_argmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        sampled[long(req) * out_stride + accepted] = long(emitted);
        num_sampled[req] = accepted + 1;
    }
}

} // namespace mittens
