#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// q38_gdn_core: everything between the qkvz projection and the output
// projection of a Gated DeltaNet layer, in ONE dispatch:
//
//   b/a gate heads (dense BF16 [2*Hv, hidden] dotted with the normed hidden)
//   causal conv (kernel 4) + SiLU over the [q|k|v] channels, conv state update
//   q/k l2 norm + scale, decay = exp(-exp(A_log) * softplus(a + dt_bias)),
//   beta = sigmoid(b)
//   delta-rule recurrence over the m rows with per-row state checkpoints
//   gated RMSNorm (weight * silu(z)) of the recurrence output
//
// It replaces muse_dense_gemv + gdn_fused_prepare + gdn_recur_spec +
// gdn_gated_rmsnorm_f32 and their five fp32 scratch tensors. On the M1 Ultra
// each of those costs ~20-45 us of launch latency regardless of size, and
// there are 48 GDN layers.
//
// One threadgroup per (VALUE head, sequence group): grid (Hv, groups). Each stages its key head's q
// and k, its own v and its two gates for all rows in threadgroup memory,
// walks its 128 x 128 state (4 state rows per simdgroup per batch, 4 key
// elements per lane), then normalizes. The q/k conv is recomputed by the
// vpk value heads that share a key head (it is tiny) so that no data crosses
// threadgroups; that needs the conv state double-buffered (read `conv_rd`,
// write `conv_wr`, the host swaps), since siblings read the channels the
// first sibling writes.
//
// State conventions are the engine's (same as the QuixiCore spec-mode
// kernels): the conv state row of a channel holds [h1, h2, x0 .. x(m-1)]
// after a step and the history window is read at column `accepted - 1`;
// the recurrence loads slot_table[accepted - 1] and stores the state after
// row t to slot_table[t].
//
// Fixed geometry: Dk = Dv = 128, conv kernel 4, m <= 8.
// ---------------------------------------------------------------------------

struct q38_gdn_params {
  int Hk;            // key heads
  int vpk;           // value heads per key head
  int hidden;        // normed hidden width (ba input)
  int qkvz_stride;   // elements per qkvz row
  int z_off;         // column of z inside a qkvz row
  int conv_cols;     // columns per conv state channel
  float prep_eps;
  float q_scale;
  float k_scale;
  float norm_eps;
  int fold;          // 1: `h` is residual * (1 + w); finish the RMSNorm here (rsqrt of the mean square)
  int npart;         // sum-of-squares partials per row (fold)
  float fold_eps;
  int out_half;      // 1: write `out` as half * 2^-4 (input of an XH GEMV) instead of BF16
  long log_stride;   // elements per log entry: k [Hk][128] | delta [Hv][128] | decay [Hv]
};

// One run of rows of one sequence. The grid's y is the group: every
// sequence's conv state, base state and update log of a layer live in one
// buffer (`state`), and the offsets below say where this sequence's are.
// (One dispatch per sequence measured 6 ms slower per step at 8 sequences.)
struct q38_gdn_group {
  int row0;          // first row of the run in qkvz / h / out
  int m;             // rows in the run
  int accepted;      // rows accepted from the sequence's previous step (>= 1)
  int n_replay;      // committed log entries to apply to the base state first
  int flush;         // 1: store the replayed state back as the new base
  int conv_rd;       // element offset of the conv state to read
  int conv_wr;       // element offset of the conv state to write
  int base_off;      // element offset of the base state [Hv][Dv][Dk]
  int log_rd;        // element offset of the log entries to replay
  int log_wr;        // element offset where this step's first entry goes
};

#define Q38_D 128
#define Q38_MAXM 8
#define Q38_HIST 3
#define Q38_GDN_SG 16  // simdgroups per threadgroup (512 threads)

struct q38_vals {
  float v[Q38_MAXM][4];
};

// conv + SiLU for 4 channels of one lane across all rows. Reads the history
// window from `rd`, and when `store` writes the next state row to `wr`.
static inline q38_vals q38_conv_rows(device const bfloat* qkvz, device const float* conv_w,
                                     device const float* rd, device float* wr, bool store,
                                     int m, int qkvz_stride, int conv_cols, int accepted,
                                     int c0) {
  q38_vals out;
  float hist[4][Q38_HIST];
  const int ro = accepted - 1;
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i) {
    device const float* row = rd + (long)(c0 + i) * conv_cols;
#pragma clang loop unroll(full)
    for (int j = 0; j < Q38_HIST; ++j) hist[i][j] = row[ro + j];
    if (store) {
#pragma clang loop unroll(full)
      for (int j = 1; j < Q38_HIST; ++j) wr[(long)(c0 + i) * conv_cols + j - 1] = hist[i][j];
    }
  }
  for (int t = 0; t < m; ++t) {
    device const bfloat* x = qkvz + (long)t * qkvz_stride + c0;
    const float4 cur = float4(*(device const bfloat4*)x);
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      device const float* w = conv_w + (long)(c0 + i) * 4;
      float v = cur[i] * w[3] + hist[i][0] * w[0] + hist[i][1] * w[1] + hist[i][2] * w[2];
      v *= 1.0f / (1.0f + exp(-v));
      out.v[t][i] = v;
      hist[i][0] = hist[i][1];
      hist[i][1] = hist[i][2];
      hist[i][2] = cur[i];
      if (store) wr[(long)(c0 + i) * conv_cols + (Q38_HIST - 1) + t] = cur[i];
    }
  }
  return out;
}

[[host_name("q38_gdn_core")]] kernel void q38_gdn_core(
    device const bfloat* qkvz_all [[buffer(0)]], // [rows][qkvz_stride]: q | k | v | z
    device const bfloat* h_all [[buffer(1)]],    // [rows][hidden] normed hidden
    device const bfloat* ba_w [[buffer(2)]],     // [2*Hv][hidden]: b rows then a rows
    device const float* conv_w [[buffer(3)]],    // [channels][4]
    device float* state [[buffer(4)]],           // every sequence's conv state, base state and log
    device const float* A_log [[buffer(5)]],     // [Hv]
    device const float* dt_bias [[buffer(6)]],   // [Hv]
    device const q38_gdn_group* groups [[buffer(7)]],
    device const bfloat* norm_w [[buffer(9)]],   // [Dv]
    device bfloat* out_all [[buffer(10)]],       // [rows][Hv*Dv]
    constant q38_gdn_params& P [[buffer(11)]],
    device const bfloat* in_norm_w [[buffer(12)]],  // [hidden] (fold)
    device const atomic_uint* partials [[buffer(13)]],  // [m][2] sum-of-squares accumulators (fold; one group)
    uint3 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float tq[Q38_MAXM][Q38_D];
  threadgroup float tk[Q38_MAXM][Q38_D];
  threadgroup float tv[Q38_MAXM][Q38_D];
  threadgroup float ty[Q38_MAXM][Q38_D];
  threadgroup float tdecay[Q38_MAXM];
  threadgroup float tbeta[Q38_MAXM];
  threadgroup float tpart[2 * Q38_MAXM][8];

  const int hv = int(tg.x);
  const int hk = hv / P.vpk;
  const bool first = (hv % P.vpk) == 0;  // owns the q/k conv state write
  const int Hk = P.Hk;
  const int Hv = Hk * P.vpk;
  const q38_gdn_group G = groups[tg.y];
  const int m = G.m;
  device const bfloat* qkvz = qkvz_all + (long)G.row0 * P.qkvz_stride;
  device const bfloat* h = h_all + (long)G.row0 * P.hidden;
  device bfloat* out = out_all + (long)G.row0 * Hv * Q38_D;
  device float* ssm = state;
  device float* glog = state;
  device const float* rd = state + G.conv_rd;
  device float* wr = state + G.conv_wr;
  const int l4 = int(lane) * 4;

  // ---- phase A: stage q, k, v and the gates for every row ----
  if (sg < 2) {
    const bool is_k = sg == 1;
    const int c0 = (is_k ? Hk * Q38_D : 0) + hk * Q38_D + l4;
    const q38_vals cv = q38_conv_rows(qkvz, conv_w, rd, wr, first, m, P.qkvz_stride,
                                      P.conv_cols, G.accepted, c0);
    for (int t = 0; t < m; ++t) {
      float ss = 0.f;
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) ss += cv.v[t][i] * cv.v[t][i];
      ss = simd_sum(ss);
      const float sc = (is_k ? P.k_scale : P.q_scale) * rsqrt(ss / float(Q38_D) + P.prep_eps);
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) {
        if (is_k) {
          tk[t][l4 + i] = cv.v[t][i] * sc;
          if (first) glog[G.log_wr + (long)t * P.log_stride + hk * Q38_D + l4 + i] = cv.v[t][i] * sc;
        } else
          tq[t][l4 + i] = cv.v[t][i] * sc;
      }
    }
  } else if (sg == 2) {
    const int c0 = 2 * Hk * Q38_D + hv * Q38_D + l4;
    const q38_vals cv = q38_conv_rows(qkvz, conv_w, rd, wr, true, m, P.qkvz_stride,
                                      P.conv_cols, G.accepted, c0);
    for (int t = 0; t < m; ++t)
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) tv[t][l4 + i] = cv.v[t][i];
  } else {
    // sg 3..15: the b and a gate heads, one dense row dot per (row, head).
    // The 2m dots are cut into `nc` column chunks each and dealt over the 13
    // simdgroups, so the longest chain a lane walks is 40 / nc loads instead
    // of 40 per row in sequence (which was half of this kernel's time).
    const int ntask = 2 * m;
    const int nc = max(1, 13 / ntask);
    for (int u = int(sg) - 3; u < ntask * nc; u += 13) {
      const int task = u / nc, ch = u % nc;
      const int t = task >> 1;
      const bool is_a = (task & 1) != 0;
      device const bfloat* w = ba_w + (long)((is_a ? Hv : 0) + hv) * P.hidden;
      device const bfloat* x = h + (long)t * P.hidden;
      const int nit = P.hidden / 128;
      const int c1 = l4 + 128 * ((ch + 1) * nit / nc);
      float acc = 0.f;
      for (int c = l4 + 128 * (ch * nit / nc); c < c1; c += 128) {
        const float4 wv = float4(*(device const bfloat4*)(w + c));
        const float4 xv = float4(*(device const bfloat4*)(x + c));
        acc += dot(wv, xv);
      }
      acc = simd_sum(acc);
      if (lane == 0) tpart[task][ch] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && int(lane) < 2 * m) {
    const int task = int(lane);
    const int t = task >> 1;
    const int nc = max(1, 13 / (2 * m));
    float acc = 0.f;
    for (int c = 0; c < nc; ++c) acc += tpart[task][c];
    if (P.fold != 0) {
      // Gemma RMSNorm of the raw residual row, folded into the dot
      const uint lo = atomic_load_explicit(&partials[2 * t], memory_order_relaxed);
      const uint hi = atomic_load_explicit(&partials[2 * t + 1], memory_order_relaxed);
      const float ss = (float(hi) * 4294967296.0f + float(lo)) * (1.0f / 4194304.0f);
      acc *= rsqrt(ss / float(P.hidden) + P.fold_eps);
    }
    acc = float(bfloat(acc));  // the projection output is BF16 in the reference
    if (task & 1) {
      const float alpha = acc + dt_bias[hv];
      const float sp = alpha > 20.0f ? alpha : log(1.0f + exp(alpha));
      tdecay[t] = exp(-exp(A_log[hv]) * sp);
      glog[G.log_wr + (long)t * P.log_stride + (long)(Hk + Hv) * Q38_D + hv] = tdecay[t];
    } else {
      tbeta[t] = 1.0f / (1.0f + exp(-acc));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- phase B: recurrence; 16 simdgroups cover 128 state rows in 2 batches ----
  // The state lives in registers for the whole step. Memory holds one base
  // state per layer plus a log of the rank-1 updates (decay, k, delta) made
  // since: S <- decay * S + delta (x) k. A step replays the committed entries,
  // runs its own rows and appends their updates; nothing but the log is
  // written unless the host asks for a flush (log full). Writing one state
  // copy per row for speculative rollback, as the first version did, cost
  // 1.15 ms per row per step in memory traffic (3.1 MB per layer per row).
  const long head_off = G.base_off + (long)hv * Q38_D * Q38_D;
  device const float* lr = glog + G.log_rd;
  device float* lw = glog + G.log_wr;
  const long lk = (long)hk * Q38_D + l4;
  const long ld = (long)(Hk + hv) * Q38_D;
  const long lg = (long)(Hk + Hv) * Q38_D + hv;
  for (int batch = 0; batch < 32 / Q38_GDN_SG; ++batch) {
    const int dv0 = (batch * Q38_GDN_SG + int(sg)) * 4;
    float st[4][4];
#pragma clang loop unroll(full)
    for (int r = 0; r < 4; ++r) {
      const float4 s4 = *(device const float4*)(ssm + head_off + (long)(dv0 + r) * Q38_D + l4);
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) st[r][i] = s4[i];
    }
    for (int e = 0; e < G.n_replay; ++e) {
      device const float* en = lr + (long)e * P.log_stride;
      const float g = en[lg];
      const float4 k4 = *(device const float4*)(en + lk);
      const float4 d4 = *(device const float4*)(en + ld + dv0);
#pragma clang loop unroll(full)
      for (int r = 0; r < 4; ++r)
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) st[r][i] = st[r][i] * g + k4[i] * d4[r];
    }
    if (G.flush != 0) {
#pragma clang loop unroll(full)
      for (int r = 0; r < 4; ++r)
        *(device float4*)(ssm + head_off + (long)(dv0 + r) * Q38_D + l4) =
            float4(st[r][0], st[r][1], st[r][2], st[r][3]);
    }
    for (int t = 0; t < m; ++t) {
      const float g = tdecay[t];
      const float beta = tbeta[t];
      float kk[4], qq[4];
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) {
        kk[i] = tk[t][l4 + i];
        qq[i] = tq[t][l4 + i];
      }
#pragma clang loop unroll(full)
      for (int r = 0; r < 4; ++r) {
        const int dv = dv0 + r;
        float kv = 0.f;
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) kv += st[r][i] * g * kk[i];
        kv = simd_sum(kv);
        const float delta = (tv[t][dv] - kv) * beta;
        float o = 0.f;
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
          st[r][i] = st[r][i] * g + kk[i] * delta;  // same form as the replay
          o += st[r][i] * qq[i];
        }
        o = simd_sum(o);
        if (lane == 0) {
          ty[t][dv] = float(bfloat(o));
          lw[(long)t * P.log_stride + ld + dv] = delta;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- phase C: gated RMSNorm; simdgroup t handles row t ----
  if (int(sg) < m) {
    const int t = int(sg);
    float yv[4];
    float ss = 0.f;
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      yv[i] = ty[t][l4 + i];
      ss += yv[i] * yv[i];
    }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / float(Q38_D) + P.norm_eps);
    device const bfloat* z = qkvz + (long)t * P.qkvz_stride + P.z_off + hv * Q38_D + l4;
    device bfloat* o = out + ((long)t * Hv + hv) * Q38_D + l4;
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      const float gate = float(z[i]);
      const float sgate = gate / (1.0f + exp(-gate));
      const float ov = yv[i] * inv * float(norm_w[l4 + i]) * sgate;
      if (P.out_half != 0)
        ((device half*)o)[i] = half(ov * 0.0625f);
      else
        o[i] = bfloat(ov);
    }
  }
}

// ---------------------------------------------------------------------------
// q38_gdn_core_big: the same layer for a prefill chunk of ONE sequence with
// any number of rows (G.m), in one dispatch. The rows are walked in runs of
// 8: phase A/B/C as above per run, with the conv history read from the rows before
// each run and the recurrent state stored back to the base after
// every run (so there is no update log: the committed entries are replayed
// once, before the first run, and the host starts an empty log afterwards).
// At the end the conv state row of every channel holds its last three inputs
// in columns 0..2, which is what a following step reads with accepted = 1.
//
// As 8-row dispatches a 256-row chunk cost 32 launches per layer, each
// replaying up to 16 log entries before its own 8 rows.
//
// 384 threads (12 simdgroups) per threadgroup: this kernel's pipeline allows
// no more (maxTotalThreadsPerThreadgroup; with 512 it ran garbage without an
// error). Simdgroups 0..2 do the convs, 3..11 the gate dots, and the 128
// state rows take three batches of 12 simdgroups x 4 rows.
// ---------------------------------------------------------------------------

// conv + SiLU for 4 channels of one lane over the rows of one run; `hist`
// (oldest first) is carried between runs.
static inline q38_vals q38_conv_run(device const bfloat* qkvz, device const float* conv_w, thread float* hist,
                                    int m, int qkvz_stride, int c0) {
  q38_vals out;
  for (int t = 0; t < m; ++t) {
    device const bfloat* x = qkvz + (long)t * qkvz_stride + c0;
    const float4 cur = float4(*(device const bfloat4*)x);
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      device const float* w = conv_w + (long)(c0 + i) * 4;
      float v = cur[i] * w[3] + hist[3 * i] * w[0] + hist[3 * i + 1] * w[1] + hist[3 * i + 2] * w[2];
      v *= 1.0f / (1.0f + exp(-v));
      out.v[t][i] = v;
      hist[3 * i] = hist[3 * i + 1];
      hist[3 * i + 1] = hist[3 * i + 2];
      hist[3 * i + 2] = cur[i];
    }
  }
  return out;
}

#define Q38_BIG_SG 12       // simdgroups per threadgroup (384 threads)
#define Q38_BIG_GATE_SG 9   // of which the gate dots use 3..11

[[host_name("q38_gdn_core_big")]] kernel void q38_gdn_core_big(
    device const bfloat* qkvz_all [[buffer(0)]], device const bfloat* h_all [[buffer(1)]],
    device const bfloat* ba_w [[buffer(2)]], device const float* conv_w [[buffer(3)]],
    device float* state [[buffer(4)]], device const float* A_log [[buffer(5)]],
    device const float* dt_bias [[buffer(6)]], device const q38_gdn_group* groups [[buffer(7)]],
    device const bfloat* norm_w [[buffer(9)]], device bfloat* out_all [[buffer(10)]],
    constant q38_gdn_params& P [[buffer(11)]], uint3 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float tq[Q38_MAXM][Q38_D];
  threadgroup float tk[Q38_MAXM][Q38_D];
  threadgroup float tv[Q38_MAXM][Q38_D];
  threadgroup float ty[Q38_MAXM][Q38_D];
  threadgroup float tdecay[Q38_MAXM];
  threadgroup float tbeta[Q38_MAXM];
  threadgroup float tpart[2 * Q38_MAXM][8];

  const int hv = int(tg.x);
  const int hk = hv / P.vpk;
  const bool first = (hv % P.vpk) == 0;  // owns the q/k conv state write
  const int Hk = P.Hk;
  const int Hv = Hk * P.vpk;
  const q38_gdn_group G = groups[0];
  const int M = G.m;
  device float* ssm = state;
  device const float* rd = state + G.conv_rd;
  device float* wr = state + G.conv_wr;
  const int l4 = int(lane) * 4;

  // conv channels of this lane (simdgroups 0, 1: q and k of the key head; 2: v) and their history
  const int cc0 = sg == 0 ? hk * Q38_D + l4
                : sg == 1 ? Hk * Q38_D + hk * Q38_D + l4 : 2 * Hk * Q38_D + hv * Q38_D + l4;

  const long head_off = G.base_off + (long)hv * Q38_D * Q38_D;
  device const float* lr = state + G.log_rd;
  const long lk = (long)hk * Q38_D + l4;
  const long lg = (long)(Hk + Hv) * Q38_D + hv;

  for (int row0 = 0; row0 < M; row0 += Q38_MAXM) {
    const int m = min(Q38_MAXM, M - row0);
    device const bfloat* qkvz = qkvz_all + (long)(G.row0 + row0) * P.qkvz_stride;
    device const bfloat* h = h_all + (long)(G.row0 + row0) * P.hidden;
    device bfloat* out = out_all + (long)(G.row0 + row0) * Hv * Q38_D;
    // conv history of this lane's channels before the run (oldest first): earlier rows of the
    // chunk, or the sequence's conv state for the rows before the chunk
    float hist[12];
    if (sg < 3) {
#pragma clang loop unroll(full)
      for (int j = 0; j < Q38_HIST; ++j) {
        const int g = row0 - Q38_HIST + j;
        if (g >= 0) {
          const float4 x4 = float4(*(device const bfloat4*)(qkvz_all + (long)(G.row0 + g) * P.qkvz_stride + cc0));
#pragma clang loop unroll(full)
          for (int i = 0; i < 4; ++i) hist[3 * i + j] = x4[i];
        } else {
#pragma clang loop unroll(full)
          for (int i = 0; i < 4; ++i)
            hist[3 * i + j] = rd[(long)(cc0 + i) * P.conv_cols + (G.accepted - 1) + (g + Q38_HIST)];
        }
      }
    }

    // ---- phase A ----
    if (sg < 2) {
      const bool is_k = sg == 1;
      const q38_vals cv = q38_conv_run(qkvz, conv_w, hist, m, P.qkvz_stride, cc0);
      for (int t = 0; t < m; ++t) {
        float ss = 0.f;
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) ss += cv.v[t][i] * cv.v[t][i];
        ss = simd_sum(ss);
        const float sc = (is_k ? P.k_scale : P.q_scale) * rsqrt(ss / float(Q38_D) + P.prep_eps);
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
          if (is_k)
            tk[t][l4 + i] = cv.v[t][i] * sc;
          else
            tq[t][l4 + i] = cv.v[t][i] * sc;
        }
      }
    } else if (sg == 2) {
      const q38_vals cv = q38_conv_run(qkvz, conv_w, hist, m, P.qkvz_stride, cc0);
      for (int t = 0; t < m; ++t)
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) tv[t][l4 + i] = cv.v[t][i];
    } else {
      const int ntask = 2 * m;
      const int nc = max(1, Q38_BIG_GATE_SG / ntask);
      for (int u = int(sg) - 3; u < ntask * nc; u += Q38_BIG_GATE_SG) {
        const int task = u / nc, ch = u % nc;
        const int t = task >> 1;
        const bool is_a = (task & 1) != 0;
        device const bfloat* w = ba_w + (long)((is_a ? Hv : 0) + hv) * P.hidden;
        device const bfloat* x = h + (long)t * P.hidden;
        const int nit = P.hidden / 128;
        const int c1 = l4 + 128 * ((ch + 1) * nit / nc);
        float acc = 0.f;
        for (int c = l4 + 128 * (ch * nit / nc); c < c1; c += 128) {
          const float4 wv = float4(*(device const bfloat4*)(w + c));
          const float4 xv = float4(*(device const bfloat4*)(x + c));
          acc += dot(wv, xv);
        }
        acc = simd_sum(acc);
        if (lane == 0) tpart[task][ch] = acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && int(lane) < 2 * m) {
      const int task = int(lane);
      const int t = task >> 1;
      const int nc = max(1, Q38_BIG_GATE_SG / (2 * m));
      float acc = 0.f;
      for (int c = 0; c < nc; ++c) acc += tpart[task][c];
      acc = float(bfloat(acc));  // the projection output is BF16 in the reference
      if (task & 1) {
        const float alpha = acc + dt_bias[hv];
        const float sp = alpha > 20.0f ? alpha : log(1.0f + exp(alpha));
        tdecay[t] = exp(-exp(A_log[hv]) * sp);
      } else {
        tbeta[t] = 1.0f / (1.0f + exp(-acc));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase B: the state comes from the base and goes back to it ----
    for (int batch = 0; batch < (32 + Q38_BIG_SG - 1) / Q38_BIG_SG; ++batch) {
      const int dv0 = (batch * Q38_BIG_SG + int(sg)) * 4;
      if (dv0 >= Q38_D) continue;  // the last batch is not full (whole simdgroups skip)
      float st[4][4];
#pragma clang loop unroll(full)
      for (int r = 0; r < 4; ++r) {
        const float4 s4 = *(device const float4*)(ssm + head_off + (long)(dv0 + r) * Q38_D + l4);
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) st[r][i] = s4[i];
      }
      if (row0 == 0) {
        const long ld = (long)(Hk + hv) * Q38_D;
        for (int e = 0; e < G.n_replay; ++e) {
          device const float* en = lr + (long)e * P.log_stride;
          const float g = en[lg];
          const float4 k4 = *(device const float4*)(en + lk);
          const float4 d4 = *(device const float4*)(en + ld + dv0);
#pragma clang loop unroll(full)
          for (int r = 0; r < 4; ++r)
#pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) st[r][i] = st[r][i] * g + k4[i] * d4[r];
        }
      }
      for (int t = 0; t < m; ++t) {
        const float g = tdecay[t];
        const float beta = tbeta[t];
        float kk[4], qq[4];
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
          kk[i] = tk[t][l4 + i];
          qq[i] = tq[t][l4 + i];
        }
#pragma clang loop unroll(full)
        for (int r = 0; r < 4; ++r) {
          const int dv = dv0 + r;
          float kv = 0.f;
#pragma clang loop unroll(full)
          for (int i = 0; i < 4; ++i) kv += st[r][i] * g * kk[i];
          kv = simd_sum(kv);
          const float delta = (tv[t][dv] - kv) * beta;
          float o = 0.f;
#pragma clang loop unroll(full)
          for (int i = 0; i < 4; ++i) {
            st[r][i] = st[r][i] * g + kk[i] * delta;
            o += st[r][i] * qq[i];
          }
          o = simd_sum(o);
          if (lane == 0) ty[t][dv] = float(bfloat(o));
        }
      }
#pragma clang loop unroll(full)
      for (int r = 0; r < 4; ++r)
        *(device float4*)(ssm + head_off + (long)(dv0 + r) * Q38_D + l4) =
            float4(st[r][0], st[r][1], st[r][2], st[r][3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase C ----
    if (int(sg) < m) {
      const int t = int(sg);
      float yv[4];
      float ss = 0.f;
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) {
        yv[i] = ty[t][l4 + i];
        ss += yv[i] * yv[i];
      }
      ss = simd_sum(ss);
      const float inv = rsqrt(ss / float(Q38_D) + P.norm_eps);
      device const bfloat* z = qkvz + (long)t * P.qkvz_stride + P.z_off + hv * Q38_D + l4;
      device bfloat* o = out + ((long)t * Hv + hv) * Q38_D + l4;
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) {
        const float gate = float(z[i]);
        const float sgate = gate / (1.0f + exp(-gate));
        o[i] = bfloat(yv[i] * inv * float(norm_w[l4 + i]) * sgate);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the next run overwrites the staged rows
  }

  // conv state for the next step: the last three inputs in columns 0..2 (read with accepted = 1).
  // A chunk has more than 8 rows, so they are all rows of the chunk.
  if (sg == 2 || (sg < 2 && first)) {
#pragma clang loop unroll(full)
    for (int j = 0; j < Q38_HIST; ++j) {
      const float4 x4 = float4(*(device const bfloat4*)(qkvz_all + (long)(G.row0 + M - Q38_HIST + j) * P.qkvz_stride + cc0));
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) wr[(long)(cc0 + i) * P.conv_cols + j] = x4[i];
    }
  }
}
