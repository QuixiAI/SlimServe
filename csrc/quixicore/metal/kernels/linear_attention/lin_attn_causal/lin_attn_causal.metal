#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Causal linear attention (identity feature map), bf16, D=64.
//   out_i = sum_{j<=i} (q_i . k_j) v_j
// Chunked running-state scan, one simdgroup per (batch, head). For each chunk
// (8 queries/keys):
//   1. inter-chunk: out_c  = q_c @ KV_state         (KV from earlier chunks)
//   2. intra-chunk: A = q_c @ k_c^T, causal-mask (lower-tri), out_c += A @ v_c
//   3. update:      KV_state += k_c^T @ v_c
// This is the causal analogue of linear_attn (the chunked scan that makes linear
// attention O(N) and causal).
template <int D>
kernel void lin_attn_causal(device   bf16     *q [[buffer(0)]],
                            device   bf16     *k [[buffer(1)]],
                            device   bf16     *v [[buffer(2)]],
                            device   bf16     *o [[buffer(3)]],
                            constant unsigned &N [[buffer(4)]],
                            constant unsigned &H [[buffer(5)]],
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "lin_attn_causal currently supports D=64");
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    gl_t gq(q, nullptr, H, N, nullptr);
    gl_t gk(k, nullptr, H, N, nullptr);
    gl_t gv(v, nullptr, H, N, nullptr);
    gl_t go(o, nullptr, H, N, nullptr);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int chunks = N / 8;

    rt_fl<D, D> kv;     // running KV state (sum over earlier chunks of k^T v)
    zero(kv);

    for (int c = 0; c < chunks; c++) {
        rt_bf<8, D> q_reg;
        rt_bf<8, D, ducks::rt_layout::col> k_reg;  // col layout: serves both A=Q@K^T and KV+=K^T@V
        rt_bf<8, D> v_reg;
        load(q_reg, gq, {batch, head, c, 0}, laneId);
        load(k_reg, gk, {batch, head, c, 0}, laneId);
        load(v_reg, gv, {batch, head, c, 0}, laneId);

        rt_fl<8, D> o_reg;
        zero(o_reg);

        // 1. inter-chunk: out_c = q_c @ KV_state(prev)
        rt_bf<D, D> kv_bf;
        copy(kv_bf, kv);
        mma_AB(o_reg, q_reg, kv_bf, o_reg);

        // 2. intra-chunk causal: A = q_c @ k_c^T, lower-triangular mask, out_c += A @ v_c
        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);
        float zero_fill = 0.0f;
        make_causal(att, att, laneId, zero_fill);   // strictly-upper -> 0 (no future)
        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(o_reg, att_bf, v_reg, o_reg);

        store(go, o_reg, {batch, head, c, 0}, laneId);

        // 3. update state: KV_state += k_c^T @ v_c
        mma_AtB(kv, k_reg, v_reg, kv);
    }
}

#define instantiate_lin_attn_causal(D)                                  \
  template [[host_name("lin_attn_causal_" #D)]] [[kernel]] void         \
  lin_attn_causal<D>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]], \
    device bf16 *v [[buffer(2)]], device bf16 *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_lin_attn_causal(64);

// ---------------------------------------------------------------------------
// Chunked-parallel causal linear attention (3 kernels). The serial kernel above
// runs ONE simdgroup per (batch, head) — B*H simdgroups total — which starves
// the GPU at long N. The chunked form exposes N/L-way parallelism:
//   K1 lin_chunk_kv:   per chunk c, KV_c = sum_{j in chunk} k_j^T v_j   (fp32)
//   K2 lin_chunk_scan: exclusive prefix over chunks, S_c = sum_{c' < c} KV_c'
//   K3 lin_chunk_out:  per chunk, the serial scan body seeded with S_c
// Work is O(N·D²) like the serial scan (K1 duplicates K3's in-register state
// updates — the price of the parallel decomposition), but grid (C, H, B).
// ---------------------------------------------------------------------------
constant constexpr const int LIN_CHUNK_L = 64;   // rows per chunk (8 subtiles)

template <int D>
kernel void lin_chunk_kv(device   bf16     *k [[buffer(0)]],
                         device   bf16     *v [[buffer(1)]],
                         device   float    *S [[buffer(2)]],   // (B,H,C,D,D)
                         constant unsigned &N [[buffer(3)]],
                         constant unsigned &H [[buffer(4)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "lin_chunk_kv currently supports D=64");
    constexpr int TPC = LIN_CHUNK_L / 8;
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    gl_t gk(k, nullptr, H, N, nullptr);
    gl_t gv(v, nullptr, H, N, nullptr);
    const int c = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int C = (int)N / LIN_CHUNK_L;

    rt_fl<D, D> kv;
    zero(kv);
    for (int t = 0; t < TPC; ++t) {
        rt_bf<8, D, ducks::rt_layout::col> k_reg;
        rt_bf<8, D> v_reg;
        load(k_reg, gk, {batch, head, c * TPC + t, 0}, laneId);
        load(v_reg, gv, {batch, head, c * TPC + t, 0}, laneId);
        mma_AtB(kv, k_reg, v_reg, kv);
    }
    gl<float, 1, -1, D, D> gs(S, nullptr, (int)H * C, nullptr, nullptr);
    store(gs, kv, {batch, head * C + c, 0, 0}, laneId);   // flat (b*H*C + h*C + c) slice
}

// Exclusive prefix-sum of the (D,D) chunk states along the chunk axis, one
// threadgroup per (batch*head); each thread owns D*D/256 elements and marches
// the chunk axis (parallel over elements, sequential over C).
template <int D>
kernel void lin_chunk_scan(device const float *Sin [[buffer(0)]],
                           device float       *Sex [[buffer(1)]],
                           constant unsigned  &C   [[buffer(2)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  tid      [[thread_index_in_threadgroup]]) {
    const long base = (long)blockIdx.x * (long)C * D * D;
    for (int e = (int)tid; e < D * D; e += 256) {
        float run = 0.0f;
        long idx = base + e;
        for (int c = 0; c < (int)C; ++c, idx += D * D) {
            const float t = Sin[idx];
            Sex[idx] = run;
            run += t;
        }
    }
}

// Per-chunk output: the serial scan body over the chunk's TPC 8-row tiles,
// seeded with the scanned inter-chunk state S_c.
template <int D>
kernel void lin_chunk_out(device   bf16       *q   [[buffer(0)]],
                          device   bf16       *k   [[buffer(1)]],
                          device   bf16       *v   [[buffer(2)]],
                          device   const float *Sex [[buffer(3)]],
                          device   bf16       *o   [[buffer(4)]],
                          constant unsigned   &N   [[buffer(5)]],
                          constant unsigned   &H   [[buffer(6)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "lin_chunk_out currently supports D=64");
    constexpr int TPC = LIN_CHUNK_L / 8;
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    gl_t gq(q, nullptr, H, N, nullptr);
    gl_t gk(k, nullptr, H, N, nullptr);
    gl_t gv(v, nullptr, H, N, nullptr);
    gl_t go(o, nullptr, H, N, nullptr);
    const int c = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int C = (int)N / LIN_CHUNK_L;

    rt_fl<D, D> kv;   // running state, seeded with the inter-chunk prefix
    gl<float, 1, -1, D, D> gs(const_cast<device float*>(Sex), nullptr,
                              (int)H * C, nullptr, nullptr);
    load(kv, gs, {batch, head * C + c, 0, 0}, laneId);

    for (int t = 0; t < TPC; ++t) {
        const int tile = c * TPC + t;
        rt_bf<8, D> q_reg;
        rt_bf<8, D, ducks::rt_layout::col> k_reg;
        rt_bf<8, D> v_reg;
        load(q_reg, gq, {batch, head, tile, 0}, laneId);
        load(k_reg, gk, {batch, head, tile, 0}, laneId);
        load(v_reg, gv, {batch, head, tile, 0}, laneId);

        rt_fl<8, D> o_reg;
        zero(o_reg);
        rt_bf<D, D> kv_bf;
        copy(kv_bf, kv);
        mma_AB(o_reg, q_reg, kv_bf, o_reg);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);
        float zero_fill = 0.0f;
        make_causal(att, att, laneId, zero_fill);
        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(o_reg, att_bf, v_reg, o_reg);

        store(go, o_reg, {batch, head, tile, 0}, laneId);
        mma_AtB(kv, k_reg, v_reg, kv);
    }
}

#define instantiate_lin_chunk(D)                                                     \
  template [[host_name("lin_chunk_kv_" #D)]] [[kernel]] void                         \
  lin_chunk_kv<D>(device bf16 *k [[buffer(0)]], device bf16 *v [[buffer(1)]],        \
    device float *S [[buffer(2)]], constant unsigned &N [[buffer(3)]],               \
    constant unsigned &H [[buffer(4)]],                                              \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint laneId [[thread_index_in_simdgroup]]);                                      \
  template [[host_name("lin_chunk_scan_" #D)]] [[kernel]] void                       \
  lin_chunk_scan<D>(device const float *Sin [[buffer(0)]],                           \
    device float *Sex [[buffer(1)]], constant unsigned &C [[buffer(2)]],             \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]]);                                    \
  template [[host_name("lin_chunk_out_" #D)]] [[kernel]] void                        \
  lin_chunk_out<D>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]],       \
    device bf16 *v [[buffer(2)]], device const float *Sex [[buffer(3)]],             \
    device bf16 *o [[buffer(4)]], constant unsigned &N [[buffer(5)]],                \
    constant unsigned &H [[buffer(6)]],                                              \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_lin_chunk(64);

}
