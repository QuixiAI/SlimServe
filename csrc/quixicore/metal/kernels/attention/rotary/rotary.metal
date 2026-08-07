#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Rotary positional embedding (RoPE), split-half / GPT-NeoX convention,
// matching mx.fast.rope(..., traditional=False). bf16 I/O, fp32 compute.
//
// With halves x1 = x[..., :D/2], x2 = x[..., D/2:] and per-position cos/sin of
// shape (N, D/2):
//     o1 = x1*cos - x2*sin
//     o2 = x2*cos + x1*sin
//
// cos/sin are precomputed and passed in (the kernel needs no trig op).
// Geometry: FLAT — one thread per 4 rotation pairs (vectorized bf16_4
// loads/stores), 256-thread groups. The previous one-simdgroup-per-row layout
// gave each threadgroup only D elements of work and scalar substrate loads,
// measuring ~2.3x slower than mx.fast.rope; this shape matches it.
// x is flattened to (M, D) with M = B*H*N; the sequence position is row % N.
// ---------------------------------------------------------------------------
template <int D>
kernel void rotary(device   bf16 *x    [[buffer(0)]],
                   device   bf16 *cosb [[buffer(1)]],
                   device   bf16 *sinb [[buffer(2)]],
                   device   bf16 *o    [[buffer(3)]],
                   constant uint &N    [[buffer(4)]],   // sequence length
                   constant uint &M    [[buffer(5)]],   // rows = B*H*N
                   uint tid [[thread_position_in_grid]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % 4 == 0, "D/2 must be divisible by 4");
    constexpr int QPR = D2 / 4;                  // 4-pair quads per row
    if (tid >= M * (uint)QPR) return;
    const int row = (int)(tid / QPR);
    const int p4  = (int)(tid % QPR) * 4;        // first pair index of this quad
    const long xb = (long)row * D;
    const long cs = (long)(row % (int)N) * D2 + p4;

    const float4 a = float4(((device const bf16_4*)(x + xb + p4))[0]);
    const float4 b = float4(((device const bf16_4*)(x + xb + D2 + p4))[0]);
    const float4 c = float4(((device const bf16_4*)(cosb + cs))[0]);
    const float4 s = float4(((device const bf16_4*)(sinb + cs))[0]);
    ((device bf16_4*)(o + xb + p4))[0]      = bf16_4(a * c - b * s);
    ((device bf16_4*)(o + xb + D2 + p4))[0] = bf16_4(b * c + a * s);
}

#define instantiate_rotary(DVAL)                                              \
  template [[host_name("rotary_" #DVAL)]] [[kernel]] void                     \
  rotary<DVAL>(device   bf16 *x    [[buffer(0)]],                             \
               device   bf16 *cosb [[buffer(1)]],                            \
               device   bf16 *sinb [[buffer(2)]],                            \
               device   bf16 *o    [[buffer(3)]],                            \
               constant uint &N    [[buffer(4)]],                            \
               constant uint &M    [[buffer(5)]],                            \
               uint tid [[thread_position_in_grid]]);

instantiate_rotary(64);
instantiate_rotary(128);

// ---------------------------------------------------------------------------
// Rotary, GPT-J *interleaved* convention, matching mx.fast.rope(traditional=True).
// Rotates adjacent pairs (x[2p], x[2p+1]) rather than the two halves:
//     o[2p]   = x[2p]*cos[p] - x[2p+1]*sin[p]
//     o[2p+1] = x[2p]*sin[p] + x[2p+1]*cos[p]
// cos/sin are (N, D/2) (one entry per pair). Same flat geometry: one thread
// per 4 pairs = 8 contiguous elements (two bf16_4 loads), pairs stay in-lane.
// ---------------------------------------------------------------------------
template <int D>
kernel void rotary_interleaved(device   bf16 *x    [[buffer(0)]],
                               device   bf16 *cosb [[buffer(1)]],
                               device   bf16 *sinb [[buffer(2)]],
                               device   bf16 *o    [[buffer(3)]],
                               constant uint &N    [[buffer(4)]],
                               constant uint &M    [[buffer(5)]],
                               uint tid [[thread_position_in_grid]]) {
    constexpr int D2 = D / 2;
    static_assert(D % 8 == 0, "interleaved rotary needs D divisible by 8");
    constexpr int QPR = D2 / 4;
    if (tid >= M * (uint)QPR) return;
    const int row = (int)(tid / QPR);
    const int p4  = (int)(tid % QPR) * 4;
    const long xb = (long)row * D + 2 * p4;      // 8 contiguous elements
    const long cs = (long)(row % (int)N) * D2 + p4;

    const float4 e0 = float4(((device const bf16_4*)(x + xb))[0]);      // pairs p4, p4+1
    const float4 e1 = float4(((device const bf16_4*)(x + xb + 4))[0]);  // pairs p4+2, p4+3
    const float4 c = float4(((device const bf16_4*)(cosb + cs))[0]);
    const float4 s = float4(((device const bf16_4*)(sinb + cs))[0]);
    ((device bf16_4*)(o + xb))[0] = bf16_4(float4(
        e0.x * c.x - e0.y * s.x, e0.x * s.x + e0.y * c.x,
        e0.z * c.y - e0.w * s.y, e0.z * s.y + e0.w * c.y));
    ((device bf16_4*)(o + xb + 4))[0] = bf16_4(float4(
        e1.x * c.z - e1.y * s.z, e1.x * s.z + e1.y * c.z,
        e1.z * c.w - e1.w * s.w, e1.z * s.w + e1.w * c.w));
}

#define instantiate_rotary_interleaved(DVAL)                                   \
  template [[host_name("rotary_interleaved_" #DVAL)]] [[kernel]] void          \
  rotary_interleaved<DVAL>(device   bf16 *x    [[buffer(0)]],                  \
                           device   bf16 *cosb [[buffer(1)]],                  \
                           device   bf16 *sinb [[buffer(2)]],                  \
                           device   bf16 *o    [[buffer(3)]],                  \
                           constant uint &N    [[buffer(4)]],                  \
                           constant uint &M    [[buffer(5)]],                  \
                           uint tid [[thread_position_in_grid]]);

instantiate_rotary_interleaved(64);
instantiate_rotary_interleaved(128);

// ---------------------------------------------------------------------------
// Positioned/partial RoPE.  This is the generic complement to the optimized
// implicit-position kernels above:
//   * positions may be shared across the batch or supplied per batch item;
//   * rotary_dim may cover only a prefix of the head (the tail is copied);
//   * split-half and adjacent-pair layouts are both explicit;
//   * D extends through 512 for multimodal and heterogeneous decoder heads.
//
// One thread handles four logical pairs.  For split-half partial rotation, a
// logical pair p addresses (p, p + rotary_dim/2).  Tail work copies two
// adjacent values so every output element is written exactly once.
// ---------------------------------------------------------------------------
template <int D>
kernel void rotary_positioned(device const bf16 *x         [[buffer(0)]],
                              device const bf16 *cosb      [[buffer(1)]],
                              device const bf16 *sinb      [[buffer(2)]],
                              device const int  *positions [[buffer(3)]],
                              device bf16       *o         [[buffer(4)]],
                              constant uint &M             [[buffer(5)]],
                              constant uint &N             [[buffer(6)]],
                              constant uint &heads         [[buffer(7)]],
                              constant uint &rotary_dim    [[buffer(8)]],
                              constant uint &pos_bstride   [[buffer(9)]],
                              constant uint &interleaved   [[buffer(10)]],
                              uint tid [[thread_position_in_grid]]) {
    constexpr uint QPR = D / 8; // four logical pairs per thread
    if (tid >= M * QPR) return;
    const uint row = tid / QPR;
    const uint p4 = (tid % QPR) * 4;
    const uint token = row % N;
    const uint batch = row / (heads * N);
    const int pos = positions[batch * pos_bstride + token];
    const uint rp = rotary_dim / 2;
    const long xb = (long)row * D;
    const long csb = (long)pos * rp;

    for (uint j = 0; j < 4; ++j) {
        const uint p = p4 + j;
        if (p < rp) {
            const uint i0 = interleaved != 0 ? 2 * p : p;
            const uint i1 = interleaved != 0 ? 2 * p + 1 : rp + p;
            const float a = float(x[xb + i0]);
            const float b = float(x[xb + i1]);
            const float c = float(cosb[csb + p]);
            const float s = float(sinb[csb + p]);
            o[xb + i0] = bf16(a * c - b * s);
            o[xb + i1] = bf16(a * s + b * c);
        } else {
            const uint i0 = rotary_dim + 2 * (p - rp);
            o[xb + i0] = x[xb + i0];
            o[xb + i0 + 1] = x[xb + i0 + 1];
        }
    }
}

#define instantiate_rotary_positioned(DVAL)                                    \
  template [[host_name("rotary_positioned_" #DVAL)]] [[kernel]] void           \
  rotary_positioned<DVAL>(device const bf16 *x [[buffer(0)]],                   \
                          device const bf16 *cosb [[buffer(1)]],                \
                          device const bf16 *sinb [[buffer(2)]],                \
                          device const int *positions [[buffer(3)]],            \
                          device bf16 *o [[buffer(4)]],                         \
                          constant uint &M [[buffer(5)]],                       \
                          constant uint &N [[buffer(6)]],                       \
                          constant uint &heads [[buffer(7)]],                   \
                          constant uint &rotary_dim [[buffer(8)]],              \
                          constant uint &pos_bstride [[buffer(9)]],             \
                          constant uint &interleaved [[buffer(10)]],            \
                          uint tid [[thread_position_in_grid]]);

instantiate_rotary_positioned(64);
instantiate_rotary_positioned(128);
instantiate_rotary_positioned(256);
instantiate_rotary_positioned(512);

// ---------------------------------------------------------------------------
// Three-axis multimodal RoPE (M-RoPE), always using split-half/NeoX pairing.
// positions is either (3,N), shared by the batch, or (B,3,N).  sections count
// rotary pairs and sum to rotary_dim/2.  section_interleaved selects either
// contiguous [T...H...W...] sections or the Qwen3-VL THWTHW... axis map.
// Frequency index p never resets at an axis transition, matching the public
// BaseRT metadata contract and llama.cpp's rope_multi definition.
// ---------------------------------------------------------------------------
template <int D>
kernel void mrope_positioned(device const bf16 *x         [[buffer(0)]],
                             device const bf16 *cosb      [[buffer(1)]],
                             device const bf16 *sinb      [[buffer(2)]],
                             device const int  *positions [[buffer(3)]],
                             device bf16       *o         [[buffer(4)]],
                             constant uint &M             [[buffer(5)]],
                             constant uint &N             [[buffer(6)]],
                             constant uint &heads         [[buffer(7)]],
                             constant uint &rotary_dim    [[buffer(8)]],
                             constant uint &pos_bstride   [[buffer(9)]],
                             constant uint &section_t     [[buffer(10)]],
                             constant uint &section_h     [[buffer(11)]],
                             constant uint &section_w     [[buffer(12)]],
                             constant uint &section_interleaved [[buffer(13)]],
                             uint tid [[thread_position_in_grid]]) {
    constexpr uint QPR = D / 8;
    if (tid >= M * QPR) return;
    const uint row = tid / QPR;
    const uint p4 = (tid % QPR) * 4;
    const uint token = row % N;
    const uint batch = row / (heads * N);
    const uint rp = rotary_dim / 2;
    const long xb = (long)row * D;
    const long pb = (long)batch * pos_bstride;

    for (uint j = 0; j < 4; ++j) {
        const uint p = p4 + j;
        if (p < rp) {
            uint axis;
            if (section_interleaved != 0) {
                axis = p % 3;
            } else if (p < section_t) {
                axis = 0;
            } else if (p < section_t + section_h) {
                axis = 1;
            } else {
                axis = 2;
            }
            const int pos = positions[pb + (long)axis * N + token];
            const long cs = (long)pos * rp + p;
            const float a = float(x[xb + p]);
            const float b = float(x[xb + rp + p]);
            const float c = float(cosb[cs]);
            const float s = float(sinb[cs]);
            o[xb + p] = bf16(a * c - b * s);
            o[xb + rp + p] = bf16(a * s + b * c);
        } else {
            const uint i0 = rotary_dim + 2 * (p - rp);
            o[xb + i0] = x[xb + i0];
            o[xb + i0 + 1] = x[xb + i0 + 1];
        }
    }
    (void)section_w;
}

#define instantiate_mrope_positioned(DVAL)                                     \
  template [[host_name("mrope_positioned_" #DVAL)]] [[kernel]] void            \
  mrope_positioned<DVAL>(device const bf16 *x [[buffer(0)]],                    \
                         device const bf16 *cosb [[buffer(1)]],                 \
                         device const bf16 *sinb [[buffer(2)]],                 \
                         device const int *positions [[buffer(3)]],             \
                         device bf16 *o [[buffer(4)]],                          \
                         constant uint &M [[buffer(5)]],                        \
                         constant uint &N [[buffer(6)]],                        \
                         constant uint &heads [[buffer(7)]],                    \
                         constant uint &rotary_dim [[buffer(8)]],               \
                         constant uint &pos_bstride [[buffer(9)]],              \
                         constant uint &section_t [[buffer(10)]],               \
                         constant uint &section_h [[buffer(11)]],               \
                         constant uint &section_w [[buffer(12)]],               \
                         constant uint &section_interleaved [[buffer(13)]],     \
                         uint tid [[thread_position_in_grid]]);

instantiate_mrope_positioned(64);
instantiate_mrope_positioned(128);
instantiate_mrope_positioned(256);
instantiate_mrope_positioned(512);

// Two-axis vision RoPE. Mode 0 is Gemma: two independent split-half rotations
// over D/2 x/y channel blocks. Mode 1 is Qwen: global split-half pairing, with
// x/y frequency sections repeated across the two global halves.
template <int D>
kernel void vision_rope_2d_kernel(
    device const bf16 *x [[buffer(0)]], device const bf16 *cosb [[buffer(1)]],
    device const bf16 *sinb [[buffer(2)]], device const int *positions [[buffer(3)]],
    device bf16 *out [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], constant uint &heads [[buffer(7)]],
    constant uint &max_position [[buffer(8)]],
    constant uint &global_split [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  constexpr uint PAIRS = D / 4;
  constexpr uint QUADS = PAIRS / 4;
  if (tid >= rows * QUADS) return;
  const uint row = tid / QUADS;
  const uint p4 = (tid % QUADS) * 4;
  const uint token = row % tokens;
  const uint batch = row / (heads * tokens);
  const long xb = (long)row * D;
  for (uint j = 0; j < 4; ++j) {
    const uint p = p4 + j;
    const int px = metal::clamp(positions[((long)batch * tokens + token) * 2],
                                0, int(max_position) - 1);
    const int py = metal::clamp(positions[((long)batch * tokens + token) * 2 + 1],
                                0, int(max_position) - 1);
    const float cx = float(cosb[(long)px * PAIRS + p]);
    const float sx = float(sinb[(long)px * PAIRS + p]);
    const float cy = float(cosb[(long)py * PAIRS + p]);
    const float sy = float(sinb[(long)py * PAIRS + p]);
    if (global_split == 0) {
      const float x0 = float(x[xb + p]);
      const float x1 = float(x[xb + PAIRS + p]);
      const float y0 = float(x[xb + 2 * PAIRS + p]);
      const float y1 = float(x[xb + 3 * PAIRS + p]);
      out[xb + p] = bf16(x0 * cx - x1 * sx);
      out[xb + PAIRS + p] = bf16(x0 * sx + x1 * cx);
      out[xb + 2 * PAIRS + p] = bf16(y0 * cy - y1 * sy);
      out[xb + 3 * PAIRS + p] = bf16(y0 * sy + y1 * cy);
    } else {
      const float x0 = float(x[xb + p]);
      const float y0 = float(x[xb + PAIRS + p]);
      const float x1 = float(x[xb + 2 * PAIRS + p]);
      const float y1 = float(x[xb + 3 * PAIRS + p]);
      out[xb + p] = bf16(x0 * cx - x1 * sx);
      out[xb + PAIRS + p] = bf16(y0 * cy - y1 * sy);
      out[xb + 2 * PAIRS + p] = bf16(x0 * sx + x1 * cx);
      out[xb + 3 * PAIRS + p] = bf16(y0 * sy + y1 * cy);
    }
  }
}

#define instantiate_vision_rope_2d(DVAL)                                      \
  template [[host_name("vision_rope_2d_D" #DVAL)]] [[kernel]] void           \
  vision_rope_2d_kernel<DVAL>(device const bf16 *x [[buffer(0)]],              \
    device const bf16 *cosb [[buffer(1)]], device const bf16 *sinb [[buffer(2)]],\
    device const int *positions [[buffer(3)]], device bf16 *out [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], constant uint &tokens [[buffer(6)]],    \
    constant uint &heads [[buffer(7)]], constant uint &max_position [[buffer(8)]],\
    constant uint &global_split [[buffer(9)]],                                \
    uint tid [[thread_position_in_grid]]);

instantiate_vision_rope_2d(64)
instantiate_vision_rope_2d(128)
instantiate_vision_rope_2d(256)
instantiate_vision_rope_2d(512)

}
