#include "tk.metal"
#include <metal_stdlib>

using namespace metal;

namespace mittens {

// ---------------------------------------------------------------------------
// Marginal layout/bit utilities (metal-forge cache/tau + sampling/bitpack + layout).
//  - tau_tail: scale Q and V slices of a packed (T, 3*q_dim) QKV by tanh(gate)+tau_pos.
//  - packbits / segment_packbits: pack bool/uint8 -> bits, big/little order (np.packbits).
//  - permute_cols_16bit: dtype-agnostic 16-bit column gather x[:, perm].
// tau_tail uses int32 positions (TM convention; ref used int64). Flat-grid elementwise v1
// for tau_tail (any head_dim); the _d64 float2 variant is a bench-gated follow-up.
// ---------------------------------------------------------------------------

template <typename T>
kernel void tau_tail(device T *qkv_out            [[buffer(0)]],   // (T, 3*q_dim), edited
                     device const T *tok_qv_lin   [[buffer(1)]],   // (T, 2*n_heads)
                     device const T *tau_pos_table [[buffer(2)]],  // (max_pos, n_heads)
                     device const int *positions  [[buffer(3)]],   // (T,)
                     constant int &elements       [[buffer(4)]],
                     constant int &n_heads        [[buffer(5)]],
                     constant int &head_dim       [[buffer(6)]],
                     constant int &q_dim          [[buffer(7)]],
                     uint tid [[thread_position_in_grid]]) {
    if (int(tid) >= elements) { return; }
    const int dim = int(tid) % head_dim;
    const int head = (int(tid) / head_dim) % n_heads;
    const int tok = int(tid) / (head_dim * n_heads);
    const float tok_q = metal::tanh(float(tok_qv_lin[tok * 2 * n_heads + head]));
    const float tok_v = metal::tanh(float(tok_qv_lin[tok * 2 * n_heads + n_heads + head]));
    const float tau_pos = float(tau_pos_table[(long)positions[tok] * n_heads + head]);
    const float q_scale = tok_q + tau_pos;
    const float v_scale = tok_v + tau_pos;
    const long q_idx = (long)tok * (3 * q_dim) + (long)head * head_dim + dim;
    const long v_idx = (long)tok * (3 * q_dim) + (long)(2 * q_dim) + (long)head * head_dim + dim;
    qkv_out[q_idx] = T(float(qkv_out[q_idx]) * q_scale);
    qkv_out[v_idx] = T(float(qkv_out[v_idx]) * v_scale);
}

// one thread per output byte; bit b of output[i] = (input[i*8+b] != 0), big or little order
kernel void packbits_uint8(device const uchar *input [[buffer(0)]],
                           device uchar *output [[buffer(1)]],
                           constant int &num_elements [[buffer(2)]],
                           constant int &bit_order_big [[buffer(3)]],
                           uint out_idx [[thread_position_in_grid]]) {
    const int base = int(out_idx) * 8;
    if (base >= num_elements) { return; }
    uchar packed = 0;
    for (int bit = 0; bit < 8; ++bit) {
        const int idx = base + bit;
        if (idx < num_elements && input[idx] != 0) {
            const int shift = bit_order_big != 0 ? 7 - bit : bit;
            packed |= uchar(1u << uint(shift));
        }
    }
    output[out_idx] = packed;
}

// ragged rows: output_indptr[seg] = first output byte of segment seg (host cumsum of
// ceil(len/8)); binary-search which segment owns each output byte.
kernel void segment_packbits_uint8(device const uchar *input [[buffer(0)]],
                                   device const int *input_indptr [[buffer(1)]],
                                   device const int *output_indptr [[buffer(2)]],
                                   device uchar *output [[buffer(3)]],
                                   constant int &num_segments [[buffer(4)]],
                                   constant int &total_output_bytes [[buffer(5)]],
                                   constant int &bit_order_big [[buffer(6)]],
                                   uint out_idx [[thread_position_in_grid]]) {
    if (int(out_idx) >= total_output_bytes) { return; }
    int lo = 0, hi = num_segments;
    while (lo + 1 < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (output_indptr[mid] <= int(out_idx)) lo = mid; else hi = mid;
    }
    const int seg = lo;
    const int input_start = input_indptr[seg];
    const int input_end = input_indptr[seg + 1];
    const int local_byte = int(out_idx) - output_indptr[seg];
    const int base = input_start + local_byte * 8;
    uchar packed = 0;
    for (int bit = 0; bit < 8; ++bit) {
        const int idx = base + bit;
        if (idx < input_end && input[idx] != 0) {
            const int shift = bit_order_big != 0 ? 7 - bit : bit;
            packed |= uchar(1u << uint(shift));
        }
    }
    output[out_idx] = packed;
}

// dtype-agnostic 16-bit column gather: output[r, c] = input[r, perm[c]]
kernel void permute_cols_16bit(device const ushort *input [[buffer(0)]],
                               device const int *perm [[buffer(1)]],
                               device ushort *output [[buffer(2)]],
                               constant int &rows [[buffer(3)]],
                               constant int &cols [[buffer(4)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x, row = gid.y;
    if (row >= uint(rows) || col >= uint(cols)) { return; }
    output[row * uint(cols) + col] = input[row * uint(cols) + uint(perm[col])];
}

#define instantiate_tau(type_name, T)                                            \
  template [[host_name("tau_tail_" #type_name)]] [[kernel]] void                 \
  tau_tail<T>(device T *qkv_out [[buffer(0)]], device const T *tok_qv_lin [[buffer(1)]], \
      device const T *tau_pos_table [[buffer(2)]], device const int *positions [[buffer(3)]], \
      constant int &elements [[buffer(4)]], constant int &n_heads [[buffer(5)]], \
      constant int &head_dim [[buffer(6)]], constant int &q_dim [[buffer(7)]],   \
      uint tid [[thread_position_in_grid]]);

instantiate_tau(float32, float)
instantiate_tau(float16, half)
instantiate_tau(bfloat16, bf16)

} // namespace mittens
