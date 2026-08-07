#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Mamba-2 / SSD (selective state space) forward, materialized chunked form, bf16, D=64.
//   Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j
// where cumlog = cumsum(log a) is the running log-decay (precomputed on host, fp32).
// This is the SSD attention-equivalent: M = (C @ B^T) (.) L, L the decay-causal matrix,
// Y = M @ X. One simdgroup per (batch, head, query-chunk); loops over key-chunks <= query.
//
// The decay matrix L[i,j] = exp(cumlog_i[i] - cumlog_j[j]) is built from broadcasts:
// add_row(colvec cumlog_i) then sub_col(rowvec cumlog_j), then exp.
template <int D>
kernel void mamba2(device   bf16     *C  [[buffer(0)]],
                   device   bf16     *Bm [[buffer(1)]],
                   device   bf16     *X  [[buffer(2)]],
                   device   float    *cl [[buffer(3)]],   // cumlog (B,H,N), fp32
                   device   bf16     *Y  [[buffer(4)]],
                   constant unsigned &N  [[buffer(5)]],
                   constant unsigned &H  [[buffer(6)]],
                   uint3 blockIdx [[threadgroup_position_in_grid]],
                   uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "mamba2 supports D in {64, 128}");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;           // C, B, X, Y : (B,H,N,D)
    using gl_cl = gl<float, 1, -1, 1, -1>;            // cumlog (B,H,N) viewed sequence-along-cols
    gl_t  gC(C, nullptr, H, N, nullptr);
    gl_t  gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr);
    gl_t  gY(Y, nullptr, H, N, nullptr);
    // gl.get<VEC> offsets by idx.c * VEC::length, so index the sequence chunk via idx.c for
    // BOTH the col_vec and row_vec loads (a vec read pulls VEC::length contiguous values).
    gl_cl gcl(cl, nullptr, H, nullptr, N);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int qi = blockIdx.x;                         // this query chunk

    rt_bf<8, D> c_reg;
    load(c_reg, gC, {batch, head, qi, 0}, laneId);
    typename rt_fl<8, 8>::col_vec cumlog_i;             // per query row
    load(cumlog_i, gcl, {batch, head, 0, qi}, laneId);

    rt_fl<8, D> y_reg;
    zero(y_reg);

    for (int kj = 0; kj <= qi; kj++) {
        rt_bf<8, D, ducks::rt_layout::col> b_reg;       // col layout for C @ B^T
        rt_bf<8, D> x_reg;
        load(b_reg, gB, {batch, head, kj, 0}, laneId);
        load(x_reg, gX, {batch, head, kj, 0}, laneId);
        typename rt_fl<8, 8>::row_vec cumlog_j;          // per key col
        load(cumlog_j, gcl, {batch, head, 0, kj}, laneId);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, c_reg, b_reg, att);                 // C @ B^T

        // decay_log[i,j] = cumlog_i[i] - cumlog_j[j]; decay = exp(decay_log)
        rt_fl<8, 8> decay;
        zero(decay);
        add_row(decay, decay, cumlog_i);
        sub_col(decay, decay, cumlog_j);
        exp(decay, decay);
        mul(att, att, decay);

        if (kj == qi) {
            float zero_fill = 0.0f;
            make_causal(att, att, laneId, zero_fill);    // future positions -> 0
        }

        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(y_reg, att_bf, x_reg, y_reg);
    }
    store(gY, y_reg, {batch, head, qi, 0}, laneId);
}

#define instantiate_mamba2(D)                                  \
  template [[host_name("mamba2_" #D)]] [[kernel]] void         \
  mamba2<D>(device bf16 *C [[buffer(0)]], device bf16 *Bm [[buffer(1)]], \
    device bf16 *X [[buffer(2)]], device float *cl [[buffer(3)]], \
    device bf16 *Y [[buffer(4)]], \
    constant unsigned &N [[buffer(5)]], constant unsigned &H [[buffer(6)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_mamba2(64);
instantiate_mamba2(128);

// ---------------------------------------------------------------------------
// Chunked linear-time SSD (3 kernels), shared by mamba2 AND lin_attn_decay
// (identical math; lin_attn_decay feeds cl = -slope*position). The kernel above
// is the QUADRATIC materialized form: every 8-row query tile rescans all
// earlier key tiles — O(N²·D) work. The chunked form is O(N·(L+D)·D):
//   K1 ssd_chunk_kv:   KV_c = sum_{j in chunk} exp(cl[r_c]-cl[j]) b_j^T x_j,
//                      referenced at r_c = last row of chunk c            (fp32)
//   K2 ssd_chunk_scan: S_c = sum_{c'<c} (prod decays) KV_c', re-referenced via
//                      the per-chunk factor L_c = exp(cl[r_c]-cl[r_{c-1}])
//   K3 ssd_chunk_out:  intra-chunk decay tiles (<= L keys, the loop above,
//                      chunk-bounded) + inter y += diag(exp(cl_i-cl[r_{c-1}]))
//                      * (C @ S_c)
// All exponents are <= 0 for a decreasing cumlog (a < 1), so everything stays
// bounded. Chunk L = 64 rows.
// ---------------------------------------------------------------------------
constant constexpr const int SSD_CHUNK_L = 64;

// The DxD chunk state is register-feasible per simdgroup only at 64x64, so the state is tiled
// into QB x QB quadrants of 64x64 (QB = D/64): the kv grid gains a quadrant axis
// (grid.x = C * QB * QB), each simdgroup owning one quadrant built from 64-wide column slices
// of B and X. At D=64 (QB=1) this reduces exactly to the untiled kernel.
template <int D>
kernel void ssd_chunk_kv(device   bf16     *Bm [[buffer(0)]],
                         device   bf16     *X  [[buffer(1)]],
                         device   float    *cl [[buffer(2)]],
                         device   float    *S  [[buffer(3)]],   // (B,H,C,D,D)
                         constant unsigned &N  [[buffer(4)]],
                         constant unsigned &H  [[buffer(5)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "ssd_chunk_kv supports D in {64,128}");
    constexpr int TPC = SSD_CHUNK_L / 8;
    constexpr int QB = D / 64;                        // 64-wide quadrant blocks per side
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);
    const int cq = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int c = cq / (QB * QB);                     // chunk
    const int kb = (cq % (QB * QB)) / QB;             // state row block (B-dim)
    const int xb = cq % QB;                           // state col block (X-dim)
    const int C = (int)N / SSD_CHUNK_L;
    const float cl_rc = cl[((long)batch * (int)H + head) * (long)N
                           + (long)(c + 1) * SSD_CHUNK_L - 1];   // reference r_c

    rt_fl<64, 64> kv;
    zero(kv);
    for (int t = 0; t < TPC; ++t) {
        const int tile = c * TPC + t;
        rt_bf<8, 64, ducks::rt_layout::col> b_reg;
        rt_bf<8, 64> x_reg;
        load(b_reg, gB, {batch, head, tile, kb}, laneId);
        load(x_reg, gX, {batch, head, tile, xb}, laneId);
        // per-row weight w_j = exp(cl[r_c] - cl[j])  (<= 1)
        typename rt_fl<8, 8>::col_vec lj, w;
        load(lj, gcl, {batch, head, 0, tile}, laneId);
        sub(w, lj, cl_rc);
        mul(w, w, -1.0f);
        exp(w, w);
        rt_fl<8, 64> x_fl;
        copy(x_fl, x_reg);
        mul_row(x_fl, x_fl, w);
        rt_bf<8, 64> x_w;
        copy(x_w, x_fl);
        mma_AtB(kv, b_reg, x_w, kv);
    }
    gl<float, 1, -1, D, D> gs(S, nullptr, (int)H * C, nullptr, nullptr);
    store(gs, kv, {batch, head * C + c, kb, xb}, laneId);
}

// Exclusive decayed prefix over chunks: run = L_c * run + KV_c, S_ex[c] = run-before.
template <int D>
kernel void ssd_chunk_scan(device const float *Sin [[buffer(0)]],
                           device const float *cl  [[buffer(1)]],
                           device bf16        *Sex [[buffer(2)]],
                           constant unsigned  &C   [[buffer(3)]],
                           constant unsigned  &N   [[buffer(4)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  tid      [[thread_index_in_threadgroup]]) {
    const long bh = blockIdx.x;
    const long base = bh * (long)C * D * D;
    device const float* clbh = cl + bh * (long)N;
    for (int e = (int)tid; e < D * D; e += 256) {
        float run = 0.0f;
        long idx = base + e;
        for (int c = 0; c < (int)C; ++c, idx += D * D) {
            const float t = Sin[idx];
            Sex[idx] = bf16(run);
            // re-reference S from r_{c-1} to r_c before adding KV_c (run is 0 at c=0)
            const float lam = (c > 0)
                ? metal::exp(clbh[(long)(c + 1) * SSD_CHUNK_L - 1]
                             - clbh[(long)c * SSD_CHUNK_L - 1])
                : 0.0f;
            run = lam * run + t;
        }
    }
}

// Per-query-tile output: intra-chunk decay tiles (bounded by the chunk) plus the
// inter-chunk state term. COOPERATIVE: grid (C, H, B) with TPC warps per threadgroup — one
// threadgroup per chunk, each warp owning one query tile. Every S_c quadrant is staged into
// threadgroup memory ONCE per chunk (cooperative load) and consumed by all TPC warps —
// amortizing the dominant state-read traffic by TPC (the D=128 mid-range bottleneck).
// Output accumulates per 64-wide half (QB halves, QB = D/64); the inter-chunk term loops the
// staged S quadrants: y[:, xq] += C[:, kq] @ S[kq, xq].
template <int D>
kernel void ssd_chunk_out(device   bf16       *Cq [[buffer(0)]],
                          device   bf16       *Bm [[buffer(1)]],
                          device   bf16       *X  [[buffer(2)]],
                          device   float      *cl [[buffer(3)]],
                          device   const bf16 *Sex [[buffer(4)]],
                          device   bf16       *Y  [[buffer(5)]],
                          constant unsigned   &N  [[buffer(6)]],
                          constant unsigned   &H  [[buffer(7)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  tid  [[thread_index_in_threadgroup]],
                          uint  warp [[simdgroup_index_in_threadgroup]],
                          uint  laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "ssd_chunk_out supports D in {64,128}");
    constexpr int TPC = SSD_CHUNK_L / 8;
    constexpr int QB = D / 64;
    using G = group<TPC>;
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gC(Cq, nullptr, H, N, nullptr);
    gl_t  gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr);
    gl_t  gY(Y, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);
    const int c = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int qi = c * TPC + (int)warp;           // this warp's query tile
    const int C = (int)N / SSD_CHUNK_L;

    rt_bf<8, D> c_reg;
    load(c_reg, gC, {batch, head, qi, 0}, laneId);
    typename rt_fl<8, 8>::col_vec cumlog_i;
    load(cumlog_i, gcl, {batch, head, 0, qi}, laneId);

    rt_fl<8, 64> y_blk[QB];
    #pragma clang loop unroll(full)
    for (int xq = 0; xq < QB; ++xq)
        zero(y_blk[xq]);

    // inter-chunk: y = diag(exp(cl_i - cl[r_{c-1}])) * (C @ S_c)   (skip for chunk 0)
    threadgroup st_bf<64, 64> sS;                 // one staged state quadrant (8KB)
    if (c > 0) {
        gl<bfloat, 1, -1, D, D> gs(const_cast<device bf16*>(Sex), nullptr,
                                   (int)H * C, nullptr, nullptr);
        for (int kq = 0; kq < QB; ++kq) {
            rt_bf<8, 64> c_kq;
            load(c_kq, gC, {batch, head, qi, kq}, laneId);
            for (int xq = 0; xq < QB; ++xq) {
                G::load(sS, gs, {batch, head * C + c, kq, xq}, tid);   // once per chunk
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                rt_bf<64, 64> s_bf;
                load(s_bf, sS, laneId);
                mma_AB(y_blk[xq], c_kq, s_bf, y_blk[xq]);
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }
        }
        const float cl_ref = cl[((long)batch * (int)H + head) * (long)N
                                + (long)c * SSD_CHUNK_L - 1];     // r_{c-1}
        typename rt_fl<8, 8>::col_vec w;
        sub(w, cumlog_i, cl_ref);
        exp(w, w);
        #pragma clang loop unroll(full)
        for (int xq = 0; xq < QB; ++xq)
            mul_row(y_blk[xq], y_blk[xq], w);
    }

    // intra-chunk: the quadratic loop, bounded to this chunk
    for (int kj = c * TPC; kj <= qi; kj++) {
        rt_bf<8, D, ducks::rt_layout::col> b_reg;
        load(b_reg, gB, {batch, head, kj, 0}, laneId);
        typename rt_fl<8, 8>::row_vec cumlog_j;
        load(cumlog_j, gcl, {batch, head, 0, kj}, laneId);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, c_reg, b_reg, att);

        rt_fl<8, 8> decay;
        zero(decay);
        add_row(decay, decay, cumlog_i);
        sub_col(decay, decay, cumlog_j);
        exp(decay, decay);
        mul(att, att, decay);

        if (kj == qi) {
            float zero_fill = 0.0f;
            make_causal(att, att, laneId, zero_fill);
        }

        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        #pragma clang loop unroll(full)
        for (int xq = 0; xq < QB; ++xq) {
            rt_bf<8, 64> x_xq;
            load(x_xq, gX, {batch, head, kj, xq}, laneId);
            mma_AB(y_blk[xq], att_bf, x_xq, y_blk[xq]);
        }
    }
    #pragma clang loop unroll(full)
    for (int xq = 0; xq < QB; ++xq)
        store(gY, y_blk[xq], {batch, head, qi, xq}, laneId);
}

#define instantiate_ssd_chunk(D)                                                     \
  template [[host_name("ssd_chunk_kv_" #D)]] [[kernel]] void                         \
  ssd_chunk_kv<D>(device bf16 *Bm [[buffer(0)]], device bf16 *X [[buffer(1)]],       \
    device float *cl [[buffer(2)]], device float *S [[buffer(3)]],                   \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint laneId [[thread_index_in_simdgroup]]);                                      \
  template [[host_name("ssd_chunk_scan_" #D)]] [[kernel]] void                       \
  ssd_chunk_scan<D>(device const float *Sin [[buffer(0)]],                           \
    device const float *cl [[buffer(1)]], device bf16 *Sex [[buffer(2)]],            \
    constant unsigned &C [[buffer(3)]], constant unsigned &N [[buffer(4)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]]);                                       \
  template [[host_name("ssd_chunk_out_" #D)]] [[kernel]] void                        \
  ssd_chunk_out<D>(device bf16 *Cq [[buffer(0)]], device bf16 *Bm [[buffer(1)]],     \
    device bf16 *X [[buffer(2)]], device float *cl [[buffer(3)]],                    \
    device const bf16 *Sex [[buffer(4)]], device bf16 *Y [[buffer(5)]],              \
    constant unsigned &N [[buffer(6)]], constant unsigned &H [[buffer(7)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                        \
    uint warp [[simdgroup_index_in_threadgroup]],                                    \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_ssd_chunk(64);
instantiate_ssd_chunk(128);

// ---------------------------------------------------------------------------
// Chunked linear-time SSD BACKWARD — O(N·(L+D)·D) like the forward, vs the quadratic
// mamba2_bwd_row/col below. Derived from the forward's chunk decomposition:
//   G_c   = sum_{t in c} exp(cl_t - cl[r_{c-1}]) C_t^T dY_t          (= dSex_c; K4 gstate)
//   dKV_c = G_{c+1} + lam_{c+1} dKV_{c+1},  dKV_{C-1} = 0,           (K5 rscan — the reverse
//           lam_k = exp(cl[r_k] - cl[r_{k-1}])                        decayed suffix scan)
//   dC_t  = intra(chunk-bounded) + w_t (dY_t Sex_c^T),  w_t = exp(cl_t - cl[r_{c-1}])   (K6)
//   dX_j  = intra + w'_j (B_j dKV_c),   dB_j = intra + w'_j (X_j dKV_c^T),              (K7)
//           w'_j = exp(cl[r_c] - cl_j)
// dcl = rowsum(M) - colsum(M) (M = dSt∘S) is split IN-KERNEL: the inter-chunk halves collapse
// to row dots against already-computed outputs — rowsum(M) = r_intra + <dC_inter, C_i> and
// colsum(M) = cc_intra + <dX_inter, X_j> — so the host just combines (r+ri)-(cc+ci); no Y
// recompute. Same 64x64 quadrant tiling as the forward; Sex/dKV are read bf16.
// ---------------------------------------------------------------------------

// K4: G_c quadrants (fp32). Same skeleton as ssd_chunk_kv with (B,X) -> (C,dY) and the weight
// referenced at r_{c-1} WITHOUT the sign flip. G_0 is never consumed by the reverse scan; it is
// computed with ref = r_0 only to keep its (unused) exponents bounded.
template <int D>
kernel void ssd_chunk_gstate(device   bf16     *Cq [[buffer(0)]],
                             device   bf16     *dY [[buffer(1)]],
                             device   float    *cl [[buffer(2)]],
                             device   float    *G  [[buffer(3)]],   // (B,H,C,D,D)
                             constant unsigned &N  [[buffer(4)]],
                             constant unsigned &H  [[buffer(5)]],
                             uint3 blockIdx [[threadgroup_position_in_grid]],
                             uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "ssd_chunk_gstate supports D in {64,128}");
    constexpr int TPC = SSD_CHUNK_L / 8;
    constexpr int QB = D / 64;
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gC(Cq, nullptr, H, N, nullptr);
    gl_t  gdY(dY, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);
    const int cq = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int c = cq / (QB * QB);
    const int kb = (cq % (QB * QB)) / QB;             // state row block (C-dim)
    const int xb = cq % QB;                           // state col block (dY-dim)
    const int C = (int)N / SSD_CHUNK_L;
    const long clbase = ((long)batch * (int)H + head) * (long)N;
    const float ref = cl[clbase + (long)(c > 0 ? c : 1) * SSD_CHUNK_L - 1];   // r_{c-1} (r_0 at c=0)

    rt_fl<64, 64> g;
    zero(g);
    for (int t = 0; t < TPC; ++t) {
        const int tile = c * TPC + t;
        rt_bf<8, 64, ducks::rt_layout::col> c_reg;
        rt_bf<8, 64> dy_reg;
        load(c_reg, gC, {batch, head, tile, kb}, laneId);
        load(dy_reg, gdY, {batch, head, tile, xb}, laneId);
        // per-row weight w_t = exp(cl_t - cl[r_{c-1}])  (<= 1 for c > 0)
        typename rt_fl<8, 8>::col_vec lt, w;
        load(lt, gcl, {batch, head, 0, tile}, laneId);
        sub(w, lt, ref);
        exp(w, w);
        rt_fl<8, 64> dy_fl;
        copy(dy_fl, dy_reg);
        mul_row(dy_fl, dy_fl, w);
        rt_bf<8, 64> dy_w;
        copy(dy_w, dy_fl);
        mma_AtB(g, c_reg, dy_w, g);
    }
    gl<float, 1, -1, D, D> gg(G, nullptr, (int)H * C, nullptr, nullptr);
    store(gg, g, {batch, head * C + c, kb, xb}, laneId);
}

// K5: reverse exclusive decayed suffix over chunks (fp32 accumulate, bf16 out for the mma).
template <int D>
kernel void ssd_chunk_rscan(device const float *Gin [[buffer(0)]],
                            device const float *cl  [[buffer(1)]],
                            device bf16        *dKV [[buffer(2)]],
                            constant unsigned  &C   [[buffer(3)]],
                            constant unsigned  &N   [[buffer(4)]],
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint  tid      [[thread_index_in_threadgroup]]) {
    const long bh = blockIdx.x;
    const long base = bh * (long)C * D * D;
    device const float* clbh = cl + bh * (long)N;
    for (int e = (int)tid; e < D * D; e += 256) {
        float run = 0.0f;                                    // = dKV_{C-1}
        for (int c = (int)C - 1; c >= 0; --c) {
            dKV[base + (long)c * D * D + e] = bf16(run);
            if (c > 0) {                                     // dKV_{c-1} = G_c + lam_c dKV_c
                const float lam = metal::exp(clbh[(long)(c + 1) * SSD_CHUNK_L - 1]
                                             - clbh[(long)c * SSD_CHUNK_L - 1]);
                run = Gin[base + (long)c * D * D + e] + lam * run;
            }
        }
    }
}

// K6: dC — chunk-bounded intra tiles + the inter term w_t (dY_t Sex_c^T). COOPERATIVE:
// grid (C, H, B) x TPC warps; Sex quadrants staged into threadgroup memory once per chunk.
template <int D>
kernel void ssd_chunk_bwd_row(device   bf16     *Cq [[buffer(0)]],
                              device   bf16     *Bm [[buffer(1)]],
                              device   bf16     *X  [[buffer(2)]],
                              device   float    *cl [[buffer(3)]],
                              device   bf16     *dY [[buffer(4)]],
                              device   const bf16 *Sex [[buffer(5)]],
                              device   bf16     *dC [[buffer(6)]],
                              device   float    *r  [[buffer(7)]],   // rowsum(M) intra, (B,H,N)
                              device   float    *ri [[buffer(8)]],   // <dC_inter, C_i>, (B,H,N)
                              constant unsigned &N  [[buffer(9)]],
                              constant unsigned &H  [[buffer(10)]],
                              uint3 blockIdx [[threadgroup_position_in_grid]],
                              uint  tid  [[thread_index_in_threadgroup]],
                              uint  warp [[simdgroup_index_in_threadgroup]],
                              uint  laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "ssd_chunk_bwd_row supports D in {64,128}");
    constexpr int TPC = SSD_CHUNK_L / 8;
    constexpr int QB = D / 64;
    using G = group<TPC>;
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gC(Cq, nullptr, H, N, nullptr), gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr), gdY(dY, nullptr, H, N, nullptr);
    gl_t  gdC(dC, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);
    gl_cl gr(r, nullptr, H, nullptr, N), gri(ri, nullptr, H, nullptr, N);
    const int c = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int i = c * TPC + (int)warp;            // this warp's row tile
    const int C = (int)N / SSD_CHUNK_L;

    rt_bf<8, D> c_i, dy_i;
    load(c_i, gC, {batch, head, i, 0}, laneId);
    load(dy_i, gdY, {batch, head, i, 0}, laneId);
    typename rt_fl<8, 8>::col_vec cumlog_i;
    load(cumlog_i, gcl, {batch, head, 0, i}, laneId);

    rt_fl<8, 64> dc_blk[QB];
    #pragma clang loop unroll(full)
    for (int kq = 0; kq < QB; ++kq)
        zero(dc_blk[kq]);
    typename rt_fl<8, 8>::col_vec r_acc, ri_acc;   // in-kernel dcl split: dcl = (r+ri)-(cc+ci)
    zero(r_acc);
    zero(ri_acc);

    // inter: dC_i[:, kq] += w_i * sum_xq dY_i[:, xq] @ Sex_c[kq, xq]^T   (skip chunk 0)
    threadgroup st_bf<64, 64> sS;                 // one staged state quadrant (8KB)
    if (c > 0) {
        gl<bfloat, 1, -1, D, D> gs(const_cast<device bf16*>(Sex), nullptr,
                                   (int)H * C, nullptr, nullptr);
        for (int kq = 0; kq < QB; ++kq) {
            for (int xq = 0; xq < QB; ++xq) {
                G::load(sS, gs, {batch, head * C + c, kq, xq}, tid);   // once per chunk
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                rt_bf<64, 64, ducks::rt_layout::col> s_col;
                load(s_col, sS, laneId);
                rt_bf<8, 64> dy_xq;
                load(dy_xq, gdY, {batch, head, i, xq}, laneId);
                mma_ABt(dc_blk[kq], dy_xq, s_col, dc_blk[kq]);
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }
        }
        const float cl_ref = cl[((long)batch * (int)H + head) * (long)N
                                + (long)c * SSD_CHUNK_L - 1];   // r_{c-1}
        typename rt_fl<8, 8>::col_vec w;
        sub(w, cumlog_i, cl_ref);
        exp(w, w);
        #pragma clang loop unroll(full)
        for (int kq = 0; kq < QB; ++kq) {
            mul_row(dc_blk[kq], dc_blk[kq], w);
            // r_inter = <dC_inter, C_i> rowwise — the inter part of rowsum(M) with no Y needed
            rt_bf<8, 64> c_kq2;
            load(c_kq2, gC, {batch, head, i, kq}, laneId);
            rt_fl<8, 64> cxi;
            copy(cxi, c_kq2);
            mul(cxi, cxi, dc_blk[kq]);
            row_sum(ri_acc, cxi, ri_acc, laneId);
        }
    }

    // intra: the quadratic bwd_row loop, bounded to this chunk
    for (int j = c * TPC; j <= i; ++j) {
        rt_bf<8, D, ducks::rt_layout::col> b_col, x_col;
        load(b_col, gB, {batch, head, j, 0}, laneId);
        load(x_col, gX, {batch, head, j, 0}, laneId);
        typename rt_fl<8, 8>::row_vec cumlog_j;
        load(cumlog_j, gcl, {batch, head, 0, j}, laneId);

        rt_fl<8, 8> dSt;
        zero(dSt);
        mma_ABt(dSt, dy_i, x_col, dSt);            // dY_i·X^T_j
        rt_fl<8, 8> Ld;
        zero(Ld);
        add_row(Ld, Ld, cumlog_i);
        sub_col(Ld, Ld, cumlog_j);
        exp(Ld, Ld);
        rt_fl<8, 8> dG;
        mul(dG, dSt, Ld);                          // dG = dSt ∘ L
        rt_fl<8, 8> G;
        zero(G);
        mma_ABt(G, c_i, b_col, G);                 // C_i·B^T_j (for S = G∘L -> M)
        rt_fl<8, 8> S;
        mul(S, G, Ld);
        if (j == i) {
            float zf = 0.0f;
            make_causal(dG, dG, laneId, zf);
            make_causal(S, S, laneId, zf);
        }
        rt_fl<8, 8> M;
        mul(M, dSt, S);                            // intra rowsum of dSt∘S
        row_sum(r_acc, M, r_acc, laneId);
        rt_bf<8, 8> dG_bf;
        copy(dG_bf, dG);
        #pragma clang loop unroll(full)
        for (int kq = 0; kq < QB; ++kq) {
            rt_bf<8, 64> b_kq;
            load(b_kq, gB, {batch, head, j, kq}, laneId);
            mma_AB(dc_blk[kq], dG_bf, b_kq, dc_blk[kq]);   // dC_i += dG·B_j
        }
    }
    #pragma clang loop unroll(full)
    for (int kq = 0; kq < QB; ++kq)
        store(gdC, dc_blk[kq], {batch, head, i, kq}, laneId);
    store(gr, r_acc, {batch, head, 0, i}, laneId);
    store(gri, ri_acc, {batch, head, 0, i}, laneId);
}

// K7: dB, dX — chunk-bounded intra tiles + the inter terms through dKV_c. COOPERATIVE:
// grid (C, H, B) x TPC warps; each dKV quadrant staged into threadgroup memory once per chunk
// and consumed twice (row layout for dX's mma_AB, col layout for dB's mma_ABt).
template <int D>
kernel void ssd_chunk_bwd_col(device   bf16     *Cq [[buffer(0)]],
                              device   bf16     *Bm [[buffer(1)]],
                              device   bf16     *X  [[buffer(2)]],
                              device   float    *cl [[buffer(3)]],
                              device   bf16     *dY [[buffer(4)]],
                              device   const bf16 *dKV [[buffer(5)]],
                              device   bf16     *dB [[buffer(6)]],
                              device   bf16     *dX [[buffer(7)]],
                              device   float    *cc [[buffer(8)]],   // colsum(M) intra, (B,H,N)
                              device   float    *ci [[buffer(9)]],   // <dX_inter, X_j>, (B,H,N)
                              constant unsigned &N  [[buffer(10)]],
                              constant unsigned &H  [[buffer(11)]],
                              uint3 blockIdx [[threadgroup_position_in_grid]],
                              uint  tid  [[thread_index_in_threadgroup]],
                              uint  warp [[simdgroup_index_in_threadgroup]],
                              uint  laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "ssd_chunk_bwd_col supports D in {64,128}");
    constexpr int TPC = SSD_CHUNK_L / 8;
    constexpr int QB = D / 64;
    using G = group<TPC>;
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gC(Cq, nullptr, H, N, nullptr), gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr), gdY(dY, nullptr, H, N, nullptr);
    gl_t  gdB(dB, nullptr, H, N, nullptr), gdX(dX, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);
    gl_cl gcc(cc, nullptr, H, nullptr, N), gci(ci, nullptr, H, nullptr, N);
    const int c = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    const int j = c * TPC + (int)warp;            // this warp's col tile
    const int C = (int)N / SSD_CHUNK_L;

    rt_bf<8, D, ducks::rt_layout::col> b_col, x_col;
    load(b_col, gB, {batch, head, j, 0}, laneId);
    load(x_col, gX, {batch, head, j, 0}, laneId);
    typename rt_fl<8, 8>::row_vec cumlog_j;
    load(cumlog_j, gcl, {batch, head, 0, j}, laneId);

    rt_fl<8, 64> db_blk[QB], dx_blk[QB];
    #pragma clang loop unroll(full)
    for (int q = 0; q < QB; ++q) {
        zero(db_blk[q]);
        zero(dx_blk[q]);
    }
    typename rt_fl<8, 8>::row_vec c_acc;           // intra colsum(M)
    typename rt_fl<8, 8>::col_vec ci_acc;          // <dX_inter, X_j> rowwise (rows = positions j)
    zero(c_acc);
    zero(ci_acc);

    // inter: dX_j += w'_j B_j@dKV_c ; dB_j += w'_j X_j@dKV_c^T   (dKV of the last chunk is 0)
    threadgroup st_bf<64, 64> sK;                 // one staged dKV quadrant (8KB)
    if (c < C - 1) {
        gl<bfloat, 1, -1, D, D> gk(const_cast<device bf16*>(dKV), nullptr,
                                   (int)H * C, nullptr, nullptr);
        for (int kq = 0; kq < QB; ++kq) {
            rt_bf<8, 64> b_kq;
            load(b_kq, gB, {batch, head, j, kq}, laneId);
            for (int xq = 0; xq < QB; ++xq) {
                G::load(sK, gk, {batch, head * C + c, kq, xq}, tid);   // once per chunk
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                rt_bf<64, 64> k_row;
                load(k_row, sK, laneId);
                mma_AB(dx_blk[xq], b_kq, k_row, dx_blk[xq]);
                rt_bf<64, 64, ducks::rt_layout::col> k_col;
                load(k_col, sK, laneId);
                rt_bf<8, 64> x_xq;
                load(x_xq, gX, {batch, head, j, xq}, laneId);
                mma_ABt(db_blk[kq], x_xq, k_col, db_blk[kq]);
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }
        }
        // w'_j = exp(cl[r_c] - cl_j)
        const float cl_rc = cl[((long)batch * (int)H + head) * (long)N
                               + (long)(c + 1) * SSD_CHUNK_L - 1];
        typename rt_fl<8, 8>::col_vec lj, w;
        load(lj, gcl, {batch, head, 0, j}, laneId);
        sub(w, lj, cl_rc);
        mul(w, w, -1.0f);
        exp(w, w);
        #pragma clang loop unroll(full)
        for (int q = 0; q < QB; ++q) {
            mul_row(dx_blk[q], dx_blk[q], w);
            mul_row(db_blk[q], db_blk[q], w);
            // c_inter = <dX_inter, X_j> rowwise — the inter part of colsum(M) with no Y needed
            rt_bf<8, 64> x_q2;
            load(x_q2, gX, {batch, head, j, q}, laneId);
            rt_fl<8, 64> xdx;
            copy(xdx, x_q2);
            mul(xdx, xdx, dx_blk[q]);
            row_sum(ci_acc, xdx, ci_acc, laneId);
        }
    }

    // intra: the quadratic bwd_col loop, bounded to this chunk
    const int i_end = (c + 1) * TPC;
    for (int i = j; i < i_end; ++i) {
        rt_bf<8, D> c_i, dy_i;
        load(c_i, gC, {batch, head, i, 0}, laneId);
        load(dy_i, gdY, {batch, head, i, 0}, laneId);
        typename rt_fl<8, 8>::col_vec cumlog_i;
        load(cumlog_i, gcl, {batch, head, 0, i}, laneId);

        rt_fl<8, 8> G;
        zero(G);
        mma_ABt(G, c_i, b_col, G);
        rt_fl<8, 8> Ld;
        zero(Ld);
        add_row(Ld, Ld, cumlog_i);
        sub_col(Ld, Ld, cumlog_j);
        exp(Ld, Ld);
        rt_fl<8, 8> S;
        mul(S, G, Ld);
        rt_fl<8, 8> dSt;
        zero(dSt);
        mma_ABt(dSt, dy_i, x_col, dSt);
        rt_fl<8, 8> dG;
        mul(dG, dSt, Ld);
        if (i == j) {
            float zf = 0.0f;
            make_causal(S, S, laneId, zf);
            make_causal(dG, dG, laneId, zf);
        }
        rt_fl<8, 8> M;
        mul(M, dSt, S);                            // intra colsum of dSt∘S
        col_sum(c_acc, M, c_acc, laneId);
        rt_bf<8, 8> S_bf, dG_bf;
        copy(S_bf, S);
        copy(dG_bf, dG);
        rt_bf<8, 8, ducks::rt_layout::col> S_col, dG_col;
        swap_layout(S_col, S_bf, laneId);
        swap_layout(dG_col, dG_bf, laneId);
        #pragma clang loop unroll(full)
        for (int q = 0; q < QB; ++q) {
            rt_bf<8, 64> dy_q, c_q;
            load(dy_q, gdY, {batch, head, i, q}, laneId);
            load(c_q, gC, {batch, head, i, q}, laneId);
            mma_AtB(dx_blk[q], S_col, dy_q, dx_blk[q]);    // dX_j += S^T·dY_i
            mma_AtB(db_blk[q], dG_col, c_q, db_blk[q]);    // dB_j += dG^T·C_i
        }
    }
    #pragma clang loop unroll(full)
    for (int q = 0; q < QB; ++q) {
        store(gdB, db_blk[q], {batch, head, j, q}, laneId);
        store(gdX, dx_blk[q], {batch, head, j, q}, laneId);
    }
    store(gcc, c_acc, {batch, head, 0, j}, laneId);
    store(gci, ci_acc, {batch, head, 0, j}, laneId);
}

#define instantiate_ssd_chunk_bwd(D)                                                 \
  template [[host_name("ssd_chunk_gstate_" #D)]] [[kernel]] void                     \
  ssd_chunk_gstate<D>(device bf16 *Cq [[buffer(0)]], device bf16 *dY [[buffer(1)]],  \
    device float *cl [[buffer(2)]], device float *G [[buffer(3)]],                   \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint laneId [[thread_index_in_simdgroup]]);                                      \
  template [[host_name("ssd_chunk_rscan_" #D)]] [[kernel]] void                      \
  ssd_chunk_rscan<D>(device const float *Gin [[buffer(0)]],                          \
    device const float *cl [[buffer(1)]], device bf16 *dKV [[buffer(2)]],            \
    constant unsigned &C [[buffer(3)]], constant unsigned &N [[buffer(4)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]]);                                       \
  template [[host_name("ssd_chunk_bwd_row_" #D)]] [[kernel]] void                    \
  ssd_chunk_bwd_row<D>(device bf16 *Cq [[buffer(0)]], device bf16 *Bm [[buffer(1)]], \
    device bf16 *X [[buffer(2)]], device float *cl [[buffer(3)]],                    \
    device bf16 *dY [[buffer(4)]], device const bf16 *Sex [[buffer(5)]],             \
    device bf16 *dC [[buffer(6)]], device float *r [[buffer(7)]],                    \
    device float *ri [[buffer(8)]], constant unsigned &N [[buffer(9)]],              \
    constant unsigned &H [[buffer(10)]],                                             \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                        \
    uint warp [[simdgroup_index_in_threadgroup]],                                    \
    uint laneId [[thread_index_in_simdgroup]]);                                      \
  template [[host_name("ssd_chunk_bwd_col_" #D)]] [[kernel]] void                    \
  ssd_chunk_bwd_col<D>(device bf16 *Cq [[buffer(0)]], device bf16 *Bm [[buffer(1)]], \
    device bf16 *X [[buffer(2)]], device float *cl [[buffer(3)]],                    \
    device bf16 *dY [[buffer(4)]], device const bf16 *dKV [[buffer(5)]],             \
    device bf16 *dB [[buffer(6)]], device bf16 *dX [[buffer(7)]],                    \
    device float *cc [[buffer(8)]], device float *ci [[buffer(9)]],                  \
    constant unsigned &N [[buffer(10)]], constant unsigned &H [[buffer(11)]],        \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]],                                        \
    uint warp [[simdgroup_index_in_threadgroup]],                                    \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_ssd_chunk_bwd(64);
instantiate_ssd_chunk_bwd(128);

// ---------------------------------------------------------------------------
// Mamba-2 / SSD BACKWARD (quadratic, no atomics), D in {64,128}. The forward is the
// attention-equivalent G = C·Bᵀ, S = tril(G∘L), Y = S·X (C↔Q, B↔K, X↔V, S↔P; no softmax).
// Both kernels recompute S and dSt = dY·Xᵀ from C,B,X,cl (like attn_bwd recomputes P), per-tile:
//   dSt[i,j] = dY_i·X_j ;  dG = tril(dSt∘L) ;  M = tril(dSt∘S)
//   mamba2_bwd_row (fix row tile i, loop j<=i): dC_i = Σ_j dG[i,j]·B_j ,  r_i = rowsum_i(M)
//   mamba2_bwd_col (fix col tile j, loop i>=j): dB_j = Σ_i dG[i,j]·C_i , dX_j = Σ_i S[i,j]·dY_i ,
//                                               c_j = colsum_j(M)
// Then (host) dcl = r - c, dloga = reverse_cumsum(dcl), da = dloga / a.
// ---------------------------------------------------------------------------
template <int D>
kernel void mamba2_bwd_row(device   bf16     *C  [[buffer(0)]],
                           device   bf16     *Bm [[buffer(1)]],
                           device   bf16     *X  [[buffer(2)]],
                           device   float    *cl [[buffer(3)]],
                           device   bf16     *dY [[buffer(4)]],
                           device   bf16     *dC [[buffer(5)]],
                           device   float    *r  [[buffer(6)]],   // rowsum(M), (B,H,N)
                           constant unsigned &N  [[buffer(7)]],
                           constant unsigned &H  [[buffer(8)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "mamba2_bwd_row supports D in {64,128}");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gC(C, nullptr, H, N, nullptr), gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr), gdY(dY, nullptr, H, N, nullptr), gdC(dC, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N), gr(r, nullptr, H, nullptr, N);

    const int i = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    rt_bf<8, D> c_i, dy_i;
    load(c_i, gC, {batch, head, i, 0}, laneId);
    load(dy_i, gdY, {batch, head, i, 0}, laneId);
    typename rt_fl<8, 8>::col_vec cumlog_i;
    load(cumlog_i, gcl, {batch, head, 0, i}, laneId);

    rt_fl<8, D> dc_acc;
    zero(dc_acc);
    typename rt_fl<8, 8>::col_vec r_acc;
    zero(r_acc);

    for (int j = 0; j <= i; ++j) {
        rt_bf<8, D> b_row;
        rt_bf<8, D, ducks::rt_layout::col> b_col, x_col;
        load(b_row, gB, {batch, head, j, 0}, laneId);
        swap_layout(b_col, b_row, laneId);
        load(x_col, gX, {batch, head, j, 0}, laneId);
        typename rt_fl<8, 8>::row_vec cumlog_j;
        load(cumlog_j, gcl, {batch, head, 0, j}, laneId);

        rt_fl<8, 8> G;
        zero(G);
        mma_ABt(G, c_i, b_col, G);                 // C_i·Bᵀ_j
        rt_fl<8, 8> Ld;
        zero(Ld);
        add_row(Ld, Ld, cumlog_i);
        sub_col(Ld, Ld, cumlog_j);
        exp(Ld, Ld);                               // L[i,j] = exp(cl_i - cl_j)
        rt_fl<8, 8> S;
        mul(S, G, Ld);                             // S = G ∘ L
        rt_fl<8, 8> dSt;
        zero(dSt);
        mma_ABt(dSt, dy_i, x_col, dSt);            // dY_i·Xᵀ_j
        rt_fl<8, 8> dG;
        mul(dG, dSt, Ld);                          // dG = dSt ∘ L
        if (j == i) {
            float zf = 0.0f;
            make_causal(S, S, laneId, zf);
            make_causal(dG, dG, laneId, zf);
        }
        rt_fl<8, 8> M;
        mul(M, dSt, S);                            // M = dSt ∘ S (S already tril-masked)
        row_sum(r_acc, M, r_acc, laneId);
        rt_bf<8, 8> dG_bf;
        copy(dG_bf, dG);
        mma_AB(dc_acc, dG_bf, b_row, dc_acc);      // dC_i += dG·B_j
    }
    store(gdC, dc_acc, {batch, head, i, 0}, laneId);
    store(gr, r_acc, {batch, head, 0, i}, laneId);
}

template <int D>
kernel void mamba2_bwd_col(device   bf16     *C  [[buffer(0)]],
                           device   bf16     *Bm [[buffer(1)]],
                           device   bf16     *X  [[buffer(2)]],
                           device   float    *cl [[buffer(3)]],
                           device   bf16     *dY [[buffer(4)]],
                           device   bf16     *dB [[buffer(5)]],
                           device   bf16     *dX [[buffer(6)]],
                           device   float    *cc [[buffer(7)]],   // colsum(M), (B,H,N)
                           constant unsigned &N  [[buffer(8)]],
                           constant unsigned &H  [[buffer(9)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "mamba2_bwd_col supports D in {64,128}");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;
    using gl_cl = gl<float, 1, -1, 1, -1>;
    gl_t  gC(C, nullptr, H, N, nullptr), gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr), gdY(dY, nullptr, H, N, nullptr);
    gl_t  gdB(dB, nullptr, H, N, nullptr), gdX(dX, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N), gcc(cc, nullptr, H, nullptr, N);

    const int j = blockIdx.x, head = blockIdx.y, batch = blockIdx.z;
    rt_bf<8, D, ducks::rt_layout::col> b_col, x_col;
    load(b_col, gB, {batch, head, j, 0}, laneId);
    load(x_col, gX, {batch, head, j, 0}, laneId);
    typename rt_fl<8, 8>::row_vec cumlog_j;
    load(cumlog_j, gcl, {batch, head, 0, j}, laneId);

    rt_fl<8, D> db_acc, dx_acc;
    zero(db_acc); zero(dx_acc);
    typename rt_fl<8, 8>::row_vec c_acc;
    zero(c_acc);

    const int q_blocks = (int)N / 8;
    for (int i = j; i < q_blocks; ++i) {
        rt_bf<8, D> c_i, dy_i;
        load(c_i, gC, {batch, head, i, 0}, laneId);
        load(dy_i, gdY, {batch, head, i, 0}, laneId);
        typename rt_fl<8, 8>::col_vec cumlog_i;
        load(cumlog_i, gcl, {batch, head, 0, i}, laneId);

        rt_fl<8, 8> G;
        zero(G);
        mma_ABt(G, c_i, b_col, G);
        rt_fl<8, 8> Ld;
        zero(Ld);
        add_row(Ld, Ld, cumlog_i);
        sub_col(Ld, Ld, cumlog_j);
        exp(Ld, Ld);
        rt_fl<8, 8> S;
        mul(S, G, Ld);
        rt_fl<8, 8> dSt;
        zero(dSt);
        mma_ABt(dSt, dy_i, x_col, dSt);
        rt_fl<8, 8> dG;
        mul(dG, dSt, Ld);
        if (i == j) {
            float zf = 0.0f;
            make_causal(S, S, laneId, zf);
            make_causal(dG, dG, laneId, zf);
        }
        rt_fl<8, 8> M;
        mul(M, dSt, S);
        col_sum(c_acc, M, c_acc, laneId);
        // dX_j += Sᵀ·dY_i  (contract over i -> swap to col layout, mma_AtB)
        rt_bf<8, 8> S_bf;
        copy(S_bf, S);
        rt_bf<8, 8, ducks::rt_layout::col> S_col;
        swap_layout(S_col, S_bf, laneId);
        mma_AtB(dx_acc, S_col, dy_i, dx_acc);
        // dB_j += dGᵀ·C_i
        rt_bf<8, 8> dG_bf;
        copy(dG_bf, dG);
        rt_bf<8, 8, ducks::rt_layout::col> dG_col;
        swap_layout(dG_col, dG_bf, laneId);
        mma_AtB(db_acc, dG_col, c_i, db_acc);
    }
    store(gdB, db_acc, {batch, head, j, 0}, laneId);
    store(gdX, dx_acc, {batch, head, j, 0}, laneId);
    store(gcc, c_acc, {batch, head, 0, j}, laneId);
}

#define instantiate_mamba2_bwd(D)                                                    \
  template [[host_name("mamba2_bwd_row_" #D)]] [[kernel]] void mamba2_bwd_row<D>(     \
    device bf16 *C [[buffer(0)]], device bf16 *Bm [[buffer(1)]], device bf16 *X [[buffer(2)]], \
    device float *cl [[buffer(3)]], device bf16 *dY [[buffer(4)]], device bf16 *dC [[buffer(5)]], \
    device float *r [[buffer(6)]], constant unsigned &N [[buffer(7)]],               \
    constant unsigned &H [[buffer(8)]], uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);                                      \
  template [[host_name("mamba2_bwd_col_" #D)]] [[kernel]] void mamba2_bwd_col<D>(     \
    device bf16 *C [[buffer(0)]], device bf16 *Bm [[buffer(1)]], device bf16 *X [[buffer(2)]], \
    device float *cl [[buffer(3)]], device bf16 *dY [[buffer(4)]], device bf16 *dB [[buffer(5)]], \
    device bf16 *dX [[buffer(6)]], device float *cc [[buffer(7)]],                   \
    constant unsigned &N [[buffer(8)]], constant unsigned &H [[buffer(9)]],          \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);

instantiate_mamba2_bwd(64);
instantiate_mamba2_bwd(128);

// ---------------------------------------------------------------------------
// ssd_decode: single-token SSD decode step against a persistent (D x D) state.
// Per (batch, head), with scalar decay alpha_t (= a_t = exp(cl_t - cl_{t-1})):
//
//     S[p,n] <- alpha * S[p,n] + x[p] * k[n]      (decay + rank-1 write)
//     y[p]    = sum_n S[p,n] * q[n]               (readout AFTER the write)
//
// This is exactly one row of the quadratic form above: y_t = sum_{j<=t} (q_t.k_j)
// exp(cl_t - cl_j) x_j, carried as a running state instead of a rescan — the O(D^2)
// generation step for mamba2 / lin_attn_decay (q=C_t, k=B_t, x=X_t; alpha=1 for
// undecayed linear attention).
//
// One threadgroup per (batch, head); one thread per output row p (D threads). Each
// thread owns the full row S[p,:], so the decay, the rank-1 update and the matvec
// readout are all independent across rows — no cross-thread reduction. q/k (shared
// over rows) are staged into threadgroup memory once. Sin/Sout may ALIAS (in-place
// update, the torch path) or not (functional update, the MLX path): thread p reads
// and writes only its own row element-by-element, so aliasing is race-free. State
// is fp32 (a recurrence read/written every step).
template <int D>
kernel void ssd_decode(device const float *Sin   [[buffer(0)]],   // (B,H,D,D)  S[p,n]
                       device const float *alpha [[buffer(1)]],   // (B,H)      per-token decay
                       device const float *x     [[buffer(2)]],   // (B,H,D)    write vector, per p
                       device const float *k     [[buffer(3)]],   // (B,H,D)    key, per n
                       device const float *q     [[buffer(4)]],   // (B,H,D)    query, per n
                       device       float *Sout  [[buffer(5)]],   // (B,H,D,D)  may alias Sin
                       device       float *y     [[buffer(6)]],   // (B,H,D)    per p
                       constant unsigned  &H     [[buffer(7)]],
                       uint3 blockIdx [[threadgroup_position_in_grid]],
                       uint  tid      [[thread_index_in_threadgroup]]) {
    static_assert(D == 64 || D == 128, "ssd_decode supports D in {64,128}");
    const uint h  = blockIdx.y;
    const uint b  = blockIdx.z;
    const uint bh = b * H + h;

    threadgroup float kk[D];        // k, shared across all rows p
    threadgroup float qq[D];        // q
    kk[tid] = k[bh * D + tid];
    qq[tid] = q[bh * D + tid];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    const uint  p  = tid;                              // this thread owns output row p in [0, D)
    const float a  = alpha[bh];
    const float xp = x[bh * D + p];
    device const float* Sr = Sin  + (bh * D + p) * (uint)D;   // S[p, :]
    device       float* Sw = Sout + (bh * D + p) * (uint)D;
    float acc = 0.0f;
    for (uint n = 0; n < D; ++n) {
        float s = a * Sr[n] + xp * kk[n];              // decayed state + rank-1 write
        Sw[n] = s;                                     // persist the new state
        acc += s * qq[n];                              // readout against q
    }
    y[bh * D + p] = acc;
}

#define instantiate_ssd_decode(D)                                                    \
  template [[host_name("ssd_decode_" #D)]] [[kernel]] void ssd_decode<D>(            \
    device const float *Sin [[buffer(0)]], device const float *alpha [[buffer(1)]],  \
    device const float *x [[buffer(2)]], device const float *k [[buffer(3)]],        \
    device const float *q [[buffer(4)]], device float *Sout [[buffer(5)]],           \
    device float *y [[buffer(6)]], constant unsigned &H [[buffer(7)]],               \
    uint3 blockIdx [[threadgroup_position_in_grid]],                                 \
    uint tid [[thread_index_in_threadgroup]]);

instantiate_ssd_decode(64);
instantiate_ssd_decode(128);

}
