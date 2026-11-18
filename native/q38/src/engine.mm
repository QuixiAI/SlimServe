// q38 engine: Qwen3.8-27B decode on Metal with no framework in the loop.
//
// One command buffer per step covers embedding rows (host memcpy into a
// shared buffer), all 64 layers, the final norm, the LM head and argmax.
// Weights are memory-mapped from a q38 pack and wrapped as no-copy buffers.
// Kernels and their host ABI come from the QuixiCore Metal library
// (csrc/quixicore/metal) through the framework-agnostic launchers in
// tk_launch.h; the layer emit sequence follows muse_q38_run in
// csrc/quixicore/tm_metal/qc_metal_serving.mm.
//
// Sequence model (one stream): a step runs m tokens causally at positions
// [pos, pos+m) and leaves per-position checkpoints of the recurrent state;
// commit(n) keeps the first n. A 1-token decode, an 8-token prefill chunk and
// a speculative verify are the same code path with different m and n.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <Accelerate/Accelerate.h>

#include <fcntl.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "q38/engine.h"
#include "tk_launch.h"

namespace q38 {

constexpr size_t kPage = 16384;

struct Buf {
  id<MTLBuffer> b = nil;
  size_t off = 0;
  size_t len = 0;  // bytes this view covers (for hazard tracking)
  Buf at(size_t byte_off) const { return Buf{b, off + byte_off, len - byte_off}; }
  void* ptr() const { return (uint8_t*)[b contents] + off; }
};

struct Range {
  uintptr_t lo, hi;
};
static inline bool overlap(const Range& a, const Range& b) {
  return a.lo < b.hi && b.lo < a.hi;
}

struct Engine;
struct Fold;
struct Res;

// Encoder adapter for tk_launch.h. With concurrent dispatch a memory barrier
// is inserted only when a dispatch touches a range written since the last
// barrier (or writes a range read since then), so independent kernels overlap.
struct Enc {
  using in_t = const Buf&;
  using out_t = const Buf&;
  id<MTLComputeCommandEncoder> enc = nil;
  Engine* eng = nullptr;
  bool concurrent = true;
  std::vector<Range> pend_src, pend_dst, cur_src, cur_dst;
  int dispatches = 0, barriers = 0;
  bool skip_cur = false;  // profiling: Q38_SKIP drops matching kernels
  unsigned long cur_max = 0;  // thread limit of the current pipeline
  std::string cur_name;

  void pipeline(const std::string& name);
  void in(const Buf& t, int i) {
    [enc setBuffer:t.b offset:t.off atIndex:i];
    cur_src.push_back(range(t));
  }
  void out(const Buf& t, int i) {
    [enc setBuffer:t.b offset:t.off atIndex:i];
    cur_dst.push_back(range(t));
  }
  template <class T>
  void bytes(const T& v, int i) {
    [enc setBytes:&v length:sizeof(T) atIndex:i];
  }
  void dispatch(int gx, int gy, int gz, int tx, int ty, int tz) {
    if (skip_cur) {
      cur_src.clear();
      cur_dst.clear();
      return;
    }
    if ((unsigned long)tx * ty * tz > cur_max) {
      // more threads per threadgroup than the pipeline allows runs garbage without any error
      fprintf(stderr, "q38: kernel %s dispatched with %d threads per threadgroup, its limit is %lu\n",
              cur_name.c_str(), tx * ty * tz, cur_max);
      abort();
    }
    if (concurrent) fence();
    cur_src.clear();
    cur_dst.clear();
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
        threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
    ++dispatches;
  }

 private:
  static Range range(const Buf& t) {
    const uintptr_t lo = (uintptr_t)[t.b contents] + t.off;
    return Range{lo, lo + (t.len ? t.len : 1)};
  }
  void fence() {
    bool conflict = false;
    for (const auto& r : cur_src)
      for (const auto& d : pend_dst)
        if (overlap(r, d)) conflict = true;
    for (const auto& r : cur_dst) {
      for (const auto& d : pend_dst)
        if (overlap(r, d)) conflict = true;
      for (const auto& s : pend_src)
        if (overlap(r, s)) conflict = true;
    }
    if (conflict) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      pend_src.clear();
      pend_dst.clear();
      ++barriers;
    }
    pend_src.insert(pend_src.end(), cur_src.begin(), cur_src.end());
    pend_dst.insert(pend_dst.end(), cur_dst.begin(), cur_dst.end());
  }
};

struct QProj {
  int fmt = 0;  // 0 fp8ch, 1 nvfp4 planar, 2 dense bf16, 3 u4, 4 u5 (uniform planar, kernels/q38_gemu.metal)
  int N = 0, K = 0;
  Buf w, scale, gscale;
  Buf bits;  // u5: the low-bit plane
  bool uniform() const { return fmt == 3 || fmt == 4; }
  // formats with a half-input (XH) q38 kernel
  bool xh_ok() const { return fmt == 1 || uniform(); }
};

struct Layer {
  bool is_gdn = false;
  Buf in_norm, post_norm;
  QProj gate_up, down;
  // gdn
  QProj qkvz, ba, gdn_out;
  Buf conv_w, A_log, dt_bias, gdn_norm;
  // attention
  QProj qkv, o;
  Buf q_norm, k_norm;
};

// Per-sequence state of one layer: GDN conv history, recurrent base state and
// update log, or the attention K/V rows.
struct SeqLayer {
  Buf conv_state, ssm_state, gdn_log, kc, vc;
};

// One independent token stream. A step carries rows of one or more sequences
// (a causal run of rows per sequence); everything that is not row-local lives
// here.
struct Seq {
  bool live = false;
  int64_t pos = 0;
  int64_t rope_off = 0, rope_off_next = 0;  // text rotary position = sequence index + rope_off
  int last_m = 0;    // rows of the uncommitted step (0 = none pending)
  int accepted = 1;  // rows kept from the previous step
  int conv_cur = 1;  // conv state copy holding the current history
  // Fused GDN state log (kernels/q38_gdn.metal phase B): committed entries in
  // the current half, the rows of the previous step right behind them.
  int gdn_log_n = 0, gdn_prev_m = 0, gdn_log_cur = 0;
  int64_t kv_cap = 0;  // tokens the attention K/V buffers hold (grown on demand)
  std::vector<SeqLayer> L;
  // MTP draft head: own K/V; pair (h_i, x_{i+1}) sits at MTP position i
  Buf mkc, mvc;
  int64_t mtp_cap = 0, mtp_pos = 0;
  int mtp_chain = 0;  // tentative draft rows past mtp_pos
  int mtp_row = 0;    // row of m_out / m_logits holding this sequence's last MTP output
  // Outputs of its last step: rows [out_row0, out_row0 + out_m) of x / logits,
  // or host copies after a snapshot load.
  int out_row0 = 0, out_m = 0;
  bool restored = false;
  bool logits_last_only = false;  // big step: only the last row's logits exist (logits row 0)
  int big_last = 0;               // big step pending: rows of its last GDN sub-step
  int big_m = 0;                  // rows of the last step if it was a big one (their hidden rows are in x)
  std::vector<uint16_t> r_tail, r_logits;
  // DSpine drafter: context K/V rings per drafter layer ([kDRing][kv heads][128]
  // BF16, slot = position & (kDRing - 1)). Context rows [d_base, d_n) exist;
  // d_n trails `pos` by the committed rows whose taps are not appended yet
  // (d_pend rows: d_taps rows d_row0.., or the host copy d_stash).
  std::vector<Buf> dk, dv;
  int64_t d_cap = 0, d_n = 0, d_base = 0;
  int d_row0 = 0, d_pend = 0, d_stash_n = 0;
  float d_oov = 0;  // running share of its tokens outside the draft vocabulary
  bool d_taps_ok = false;  // the last step recorded taps
  std::vector<uint16_t> d_stash;
};

struct Group {
  int seq, m, row0;
};
// rows [row0, row0 + n) of a row-major buffer, as a view with an exact length
static inline Buf rows(const Buf& b, size_t row0, size_t n, size_t row_bytes) {
  return Buf{b.b, b.off + row0 * row_bytes, n * row_bytes};
}

struct Engine {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  std::vector<id<MTLLibrary>> libs;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipes;
  std::string error;

  // pack
  int fd = -1;
  uint8_t* map = nullptr;
  size_t map_len = 0, data_base = 0;
  NSDictionary* tensors = nil;
  NSDictionary* meta = nil;
  std::vector<id<MTLBuffer>> keep;

  // geometry
  int num_layers = 0, hidden = 0, inter = 0, vocab = 0;
  int heads = 0, kv_heads = 0, head_dim = 0, rot_dim = 0;
  int gHk = 0, gHv = 0, gDk = 0, gDv = 0, conv_kernel = 0;
  int q_sz = 0, kv_sz = 0, qkv_n = 0, gdn_qkv_ch = 0, gdn_w = 0;
  float eps = 1e-6f, attn_scale = 0.f, rope_theta = 0.f;
  float gdn_prep_eps = 0.f, gdn_q_scale = 0.f, gdn_k_scale = 0.f;
  int max_ctx = 32768, max_rows = 8, block_size = 16, max_blocks = 0;
  int conv_cols = 0, conv_stride = 0, ssm_stride = 0;
  bool serial = false;
  double read_gb = 0;

  std::vector<Layer> layers;
  Buf embed, final_norm;
  QProj lm_head;
  Buf cos_sin;
  // Explicit (image) positions: a step may carry per-row (t, h, w) rotary
  // positions; its rows are built on the host in cs_step and addressed by row
  // index. Text after an image continues at sequence index + rope_off.
  Buf cs_step, row_idx;
  const Buf* cs_cur = nullptr;  // rotary rows the attention kernels read this pass
  std::vector<double> rope_inv;
  // A step of ONE sequence may carry up to kBigRows rows (prompt prefill): the
  // projections then go through the GEMM kernels and the GDN core runs in
  // 8-row sub-steps. Such a step is committed whole.
  static constexpr int kBigRows = 512;
  static constexpr int kMaxSeqs = 8;  // max_rows: every sequence in a step has at least one row
  std::vector<Seq> seqs;  // seqs[0] always exists: the sequence of the single-stream API

  // scratch (all sized for max_rows)
  Buf x, h, hh, res, res2, t_hidden, qkvz, ba_s, gnormed;
  Buf attn_qkv, q, k, v, gate, attn_out, gu, mid;
  Buf gq, gk, gv, gdecay, gbeta, gy;
  Buf pa_tmp, pa_ml, pa_es;
  Buf attn_pa, attn_pm;  // q38_attn partition partials
  Buf attn_sel;          // row -> sequence group of the pass (q38_attn_part_ms)
  static constexpr int kAttnMaxParts = 512;
  std::vector<Buf> acc;  // norm-folding sum-of-squares accumulators, one [m][2] u32 per seam
  Buf acc_all;
  Buf logits, argmax;
  // per-step metadata
  Buf positions, slots, exp_seq, exp_bt, spec_cu, conv_slots, slot_table, num_accepted;

  // MTP draft head (optional sidecar pack)
  bool has_mtp = false;
  bool has_uniform = false;  // pack has u4/u5 projections: half-activation path at every M
  Layer mtp;
  QProj mtp_fc;
  Buf mtp_pre_e, mtp_pre_h, mtp_norm;
  Buf m_emb, m_hid, fc_in, mx, m_out, m_logits, m_argmax, m_positions, m_slots, m_exp_seq;
  double mtp_read_gb = 0;

  // DSpine block drafter (optional sidecar pack, tools/pack_dspine.py; kernels/q38_dspine.metal)
  static constexpr int kDLayers = 5, kDRing = 16384, kDTopK = 16, kDCode = 1024, kDRank = 512, kDHd = 128;
  struct DLayer {
    Buf in_norm, post_norm, q_norm, k_norm, read, write;
    QProj qkv, kv, o, gate_up, down;
    Buf kvp;  // context k|v projection rows
  };
  bool has_dspine = false;
  DLayer dl[kDLayers];
  QProj d_fc;
  Buf d_hidden_norm, d_norm, d_gate_b, d_packed_w;
  Buf d_tsv, d_tg, d_write_t;  // sampling-tail tables (pack_dspine.py)
  // Draft vocabulary: the LM head rows of the most frequent tokens, copied in
  // rank order, so a draft pass reads those rows only. A sequence whose recent
  // tokens fall outside the list too often (d_oov) drafts with the full head.
  QProj d_head;
  std::vector<int32_t> d_vocab;
  std::vector<uint8_t> d_in_vocab;
  int d_heads = 0, d_kvh = 0, d_mask_tok = 0, d_window = 0;
  std::vector<int> d_tap_of;  // target layer -> tap column, or -1
  float d_eps = 1e-6f, d_beps = 1e-6f;
  std::vector<double> d_inv;  // rotary inverse frequencies
  Buf d_taps, d_taps2, d_fcout, d_ctx, d_cs, d_slot, d_seqlen, d_x, d_in, d_inh, d_res, d_res2, d_h,
      d_qkv, d_q, d_ao, d_t, d_t2, d_raw, d_rms, d_packed, d_proj, d_msg, d_delta, d_hh, d_pa, d_pm, d_logits, d_scrap, d_c, d_hin, d_hinh;
  std::vector<float> d_gb, d_G, d_nw;  // host copies for the sampling tail
  double d_read_gb = 0, d_gpu_ms = 0, d_tail_ms = 0, d_wall_ms = 0;
  double d_tpool = 0;
  int64_t d_calls = 0;
  double d_tp[3] = {0, 0, 0};  // tail split: top-k scan, bridge products, head rows
  std::vector<int32_t> d_force;  // debug: tokens the next draft call takes instead of drawing

  static constexpr int kGdnLogCap = 16;
  size_t gdn_log_stride = 0;
  // Fused GDN state: per GDN layer ONE buffer holding, for each of kMaxSeqs
  // sequences, [conv state x2 | null + base recurrent state | update log x2]
  // (floats; the per-sequence SeqLayer buffers are views into it), so the
  // kernel serves every sequence of a step in one dispatch.
  std::vector<Buf> gdn_arena;
  size_t gdn_conv_fl = 0, gdn_ssm_fl = 0, gdn_log_fl = 0, gdn_region_fl = 0;
  Buf gdn_groups;  // q38_gdn_group table of the step being encoded (rewritten per layer: same content)
  int layer_cap = 0;
  bool want_argmax = true;  // GPU argmax of each logits row (off when the host samples)

  bool fail(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    error = buf;
    return false;
  }

  id<MTLComputePipelineState> pipe(const std::string& name) {
    auto it = pipes.find(name);
    if (it != pipes.end()) return it->second;
    NSString* n = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> fn = nil;
    for (id<MTLLibrary> lib : libs) {
      fn = [lib newFunctionWithName:n];
      if (fn) break;
    }
    if (!fn) {
      fprintf(stderr, "q38: kernel %s not found in any metallib\n", name.c_str());
      abort();
    }
    NSError* err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) {
      fprintf(stderr, "q38: pipeline %s: %s\n", name.c_str(),
              err.localizedDescription.UTF8String);
      abort();
    }
    // A kernel that needs many registers gets a smaller thread limit; dispatching more
    // threads per threadgroup than this silently runs garbage (found 2026-11-10 with a
    // 512-thread GDN variant limited to fewer). Q38_KLOG prints the limits.
    if (getenv("Q38_KLOG"))
      fprintf(stderr, "pipeline %s: max %lu threads per threadgroup, simd width %lu\n", name.c_str(),
              (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)p.threadExecutionWidth);
    pipes[name] = p;
    return p;
  }

  // `owned`: the returned view is the only reference (per-sequence state that
  // goes away with its sequence); otherwise the engine keeps the buffer.
  Buf alloc(size_t bytes, bool owned = false) {
    bytes = (bytes + kPage - 1) / kPage * kPage;
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) {
      fprintf(stderr, "q38: allocation of %zu bytes failed\n", bytes);
      abort();
    }
    if (!owned) keep.push_back(b);
    return Buf{b, 0, bytes};
  }

  bool tensor(const std::string& name, Buf* out, const char* want_dtype = nullptr) {
    NSDictionary* e = tensors[[NSString stringWithUTF8String:name.c_str()]];
    if (!e) return fail("pack has no tensor %s", name.c_str());
    if (want_dtype && strcmp([e[@"dtype"] UTF8String], want_dtype) != 0)
      return fail("tensor %s is %s, wanted %s", name.c_str(), [e[@"dtype"] UTF8String], want_dtype);
    const size_t off = [e[@"offset"] unsignedLongLongValue];
    const size_t nbytes = [e[@"nbytes"] unsignedLongLongValue];
    const size_t len = (nbytes + kPage - 1) / kPage * kPage;
    if (data_base + off + len > map_len) return fail("tensor %s runs past the file", name.c_str());
    id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:map + data_base + off
                                             length:len
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
    if (!b) return fail("Metal could not wrap tensor %s (%zu bytes) without a copy", name.c_str(), len);
    keep.push_back(b);
    *out = Buf{b, 0, nbytes};
    return true;
  }

  bool proj(const std::string& name, QProj* p) {
    NSDictionary* d = meta[@"projections"][[NSString stringWithUTF8String:name.c_str()]];
    if (!d) return fail("pack has no projection %s", name.c_str());
    NSString* fmt = d[@"fmt"];
    p->N = [d[@"N"] intValue];
    p->K = [d[@"K"] intValue];
    if ([fmt isEqualToString:@"fp8ch"]) {
      p->fmt = 0;
      return tensor(name + ".w", &p->w, "u8") && tensor(name + ".scale", &p->scale, "f32");
    }
    if ([fmt isEqualToString:@"nvfp4"]) {
      p->fmt = 1;
      return tensor(name + ".w", &p->w, "u8") && tensor(name + ".scale", &p->scale, "u8") &&
             tensor(name + ".gscale", &p->gscale, "f32");
    }
    if ([fmt isEqualToString:@"u4"] || [fmt isEqualToString:@"u5"]) {
      p->fmt = [fmt isEqualToString:@"u5"] ? 4 : 3;
      has_uniform = true;
      if (!tensor(name + ".w", &p->w, "u8") || !tensor(name + ".scale", &p->scale, "u8") ||
          !tensor(name + ".gscale", &p->gscale, "f32"))
        return false;
      if (p->fmt == 4) return tensor(name + ".b", &p->bits, "u8");
      p->bits = p->w;  // bound, never read
      return true;
    }
    if (![fmt isEqualToString:@"bf16"]) return fail("projection %s has unknown format", name.c_str());
    p->fmt = 2;
    return tensor(name + ".w", &p->w, "bf16");
  }

  bool open_pack(const char* path);
  bool build();
  int seq_new();
  void seq_free(int id);
  void seq_reset(Seq& S);
  bool seq_ok(int id) const { return id >= 0 && id < (int)seqs.size() && seqs[id].live; }
  void ensure_kv(Seq& S, int64_t tokens);
  void ensure_mtp_kv(Seq& S, int64_t tokens);
  const uint16_t* hid_row(const Seq& S, int row) const {
    return S.restored ? S.r_tail.data() + (size_t)row * hidden
                      : (const uint16_t*)x.ptr() + (size_t)(S.out_row0 + row) * hidden;
  }
  const uint16_t* logits_row(const Seq& S, int row) const {
    if (S.restored) return S.r_logits.data();  // only the last kept row survives a snapshot
    if (S.logits_last_only) return (const uint16_t*)logits.ptr();
    return (const uint16_t*)logits.ptr() + (size_t)(S.out_row0 + row) * vocab;
  }
  int step(const Group* g, int ng, const int32_t* tokens, const uint16_t* embeds,
           const int32_t* pos3, q38_step_stats* stats);
  void commit_rows(Seq& S, int n);
  struct AGroup {
    int row0, m;
    int64_t width;
    const Buf* kc;
    const Buf* vc;
  };
  void emit_attn(Enc& e, const Layer& L, int m, const AGroup* ag, int nag, const Buf& pos_buf,
                 const Buf& slot_buf, const Buf& seq_buf, const Buf& xin, const Buf& out,
                 const struct Fold* fold, const struct Res* rs, const Buf* xin_half = nullptr);
  struct q38_snapshot* snapshot_save(Seq& S);
  int snapshot_load(Seq& S, const struct q38_snapshot* s, bool global);
  bool load_mtp(const char* path);
  int mtp_run(const Group* g, int ng, const int32_t* tokens, const uint16_t* embeds,
              const uint16_t* const* hid, bool commit, bool head, q38_step_stats* stats);
  bool load_dspine(const char* path);
  void d_ensure(Seq& S, int64_t tokens);
  struct DInline {
    Seq* S;
    Buf taps;
    int n, off;
  };
  void d_emit_ctx(Enc& e, Seq& S, const Buf& taps, int n, int off);
  int d_take(Seq& S, Buf* taps, int off);
  int d_flush(Seq& S);
  int d_preserve(std::vector<DInline>* inl);
  int64_t d_win() const {  // context positions a block attends to
    return d_window > 0 ? std::min<int64_t>(d_window, kDRing - kBigRows - 16) : kDRing - kBigRows - 16;
  }
  struct DReq {
    int seq;
    int32_t anchor;
    int k;
    const double* u;
    q38_draft_row* out;
    int D;  // out: rows written
  };
  int dspine_draft(int seq, int32_t anchor, int k, const q38_sampling* sp, float stop_below,
                   const double* u, q38_draft_row* out, q38_step_stats* stats);
  int dspine_draft_groups(DReq* rq, int ng, const q38_sampling* sp, float stop_below, q38_step_stats* stats);
};

void Enc::pipeline(const std::string& name) {
  // Profiling aids. Q38_SKIP=a,b drops every dispatch whose kernel name
  // contains a or b (outputs become garbage; the step-time delta is that
  // kernel class's cost). Q38_KLOG=1 prints each kernel name once.
  static const char* skip = getenv("Q38_SKIP");
  static const bool klog = getenv("Q38_KLOG") != nullptr;
  static std::map<std::string, int> seen;
  skip_cur = false;
  if (skip) {
    std::string list(skip);
    size_t a = 0;
    while (a <= list.size()) {
      size_t b = list.find(',', a);
      if (b == std::string::npos) b = list.size();
      if (b > a && name.find(list.substr(a, b - a)) != std::string::npos) skip_cur = true;
      a = b + 1;
    }
  }
  if (klog && seen[name]++ == 0) fprintf(stderr, "kernel %s\n", name.c_str());
  id<MTLComputePipelineState> ps = eng->pipe(name);
  cur_max = (unsigned long)ps.maxTotalThreadsPerThreadgroup;
  cur_name = name;
  [enc setComputePipelineState:ps];
}

static double now_ms() {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  // round to nearest even on the dropped 16 bits
  const uint32_t lsb = (u >> 16) & 1;
  u += 0x7fff + lsb;
  return (uint16_t)(u >> 16);
}

bool Engine::open_pack(const char* path) {
  fd = open(path, O_RDONLY);
  if (fd < 0) return fail("cannot open %s", path);
  struct stat st;
  fstat(fd, &st);
  map_len = (size_t)st.st_size;
  map = (uint8_t*)mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) return fail("mmap of %s failed", path);
  if (memcmp(map, "Q38PACK1", 8) != 0) return fail("%s is not a q38 pack", path);
  uint64_t hlen;
  memcpy(&hlen, map + 8, 8);
  data_base = 16 + hlen;
  if (data_base % kPage) return fail("pack data region is not page aligned");
  NSData* hd = [NSData dataWithBytesNoCopy:map + 16 length:hlen freeWhenDone:NO];
  NSError* err = nil;
  NSDictionary* root = [NSJSONSerialization JSONObjectWithData:hd options:0 error:&err];
  if (!root) return fail("pack header: %s", err.localizedDescription.UTF8String);
  meta = root[@"meta"];
  tensors = root[@"tensors"];
  return true;
}

bool Engine::build() {
  num_layers = [meta[@"num_layers"] intValue];
  hidden = [meta[@"hidden"] intValue];
  inter = [meta[@"inter"] intValue];
  vocab = [meta[@"vocab"] intValue];
  eps = [meta[@"eps"] floatValue];
  read_gb = [meta[@"decode_read_bytes"] doubleValue] / 1e9;
  NSDictionary* a = meta[@"attn"];
  heads = [a[@"heads"] intValue];
  kv_heads = [a[@"kv_heads"] intValue];
  head_dim = [a[@"head_dim"] intValue];
  rot_dim = [a[@"rot_dim"] intValue];
  rope_theta = [a[@"rope_theta"] floatValue];
  attn_scale = 1.0f / sqrtf((float)head_dim);
  NSDictionary* g = meta[@"gdn"];
  gHk = [g[@"k_heads"] intValue];
  gHv = [g[@"v_heads"] intValue];
  gDk = [g[@"k_dim"] intValue];
  gDv = [g[@"v_dim"] intValue];
  conv_kernel = [g[@"conv_kernel"] intValue];
  // The kernel's rms-form norm recovers l2norm with these constants.
  gdn_prep_eps = 1e-6f / (float)gDk;
  gdn_q_scale = 1.0f / (float)gDk;
  gdn_k_scale = 1.0f / sqrtf((float)gDk);

  q_sz = heads * head_dim;
  kv_sz = kv_heads * head_dim;
  qkv_n = 2 * q_sz + 2 * kv_sz;
  gdn_qkv_ch = 2 * gHk * gDk + gHv * gDv;
  gdn_w = gdn_qkv_ch + gHv * gDv;
  max_blocks = (max_ctx + block_size - 1) / block_size;
  conv_cols = conv_kernel - 1 + max_rows;
  conv_stride = gdn_qkv_ch * conv_cols;
  ssm_stride = gHv * gDv * gDk;

  if (!tensor("embed", &embed, "bf16") || !tensor("final_norm", &final_norm, "bf16") ||
      !proj("lm_head", &lm_head))
    return false;

  NSArray* kinds = meta[@"layer_types"];
  layers.resize(num_layers);
  gdn_log_stride = (size_t)(gHk + gHv) * gDk + gHv;
  for (int l = 0; l < num_layers; ++l) {
    Layer& L = layers[l];
    const std::string o = "L" + std::to_string(l) + ".";
    L.is_gdn = [kinds[l] isEqualToString:@"gdn"];
    if (!tensor(o + "in_norm", &L.in_norm, "bf16") || !tensor(o + "post_norm", &L.post_norm, "bf16") ||
        !proj(o + "gate_up", &L.gate_up) || !proj(o + "down", &L.down))
      return false;
    if (L.is_gdn) {
      if (!proj(o + "qkvz", &L.qkvz) || !proj(o + "ba", &L.ba) || !proj(o + "gdn_out", &L.gdn_out) ||
          !tensor(o + "conv_w", &L.conv_w, "f32") || !tensor(o + "A_log", &L.A_log, "f32") ||
          !tensor(o + "dt_bias", &L.dt_bias, "f32") || !tensor(o + "gdn_norm", &L.gdn_norm, "bf16"))
        return false;
    } else {
      if (!proj(o + "qkv", &L.qkv) || !proj(o + "o", &L.o) ||
          !tensor(o + "q_norm", &L.q_norm, "bf16") || !tensor(o + "k_norm", &L.k_norm, "bf16"))
        return false;
    }
  }

  const size_t m = max_rows;
  const size_t mb = std::max<size_t>(max_rows, kBigRows);  // rows of a big (prefill) step
  x = alloc(mb * hidden * 2);
  h = alloc(mb * hidden * 2);
  hh = alloc(mb * hidden * 2);  // half * 2^-4 copy of h for the XH GEMVs
  res = alloc(mb * hidden * 2);
  res2 = alloc(mb * hidden * 2);
  t_hidden = alloc(mb * hidden * 2);
  qkvz = alloc(mb * gdn_w * 2);
  ba_s = alloc(mb * 2 * gHv * 2);
  gnormed = alloc(mb * gHv * gDv * 2);
  attn_qkv = alloc(mb * qkv_n * 2);
  q = alloc(mb * q_sz * 2);
  k = alloc(mb * kv_sz * 2);
  v = alloc(mb * kv_sz * 2);
  gate = alloc(mb * q_sz * 2);
  attn_out = alloc(mb * q_sz * 2);
  gu = alloc(mb * 2 * inter * 2);
  mid = alloc(mb * inter * 2);
  gq = alloc(m * gHk * gDk * 4);
  gk = alloc(m * gHk * gDk * 4);
  gv = alloc(m * gHv * gDv * 4);
  gdecay = alloc(m * gHv * 4);
  gbeta = alloc(m * gHv * 4);
  gy = alloc(m * gHv * gDv * 4);
  attn_pa = alloc(m * heads * kAttnMaxParts * head_dim * 4);
  attn_pm = alloc(m * heads * kAttnMaxParts * 2 * 4);
  attn_sel = alloc(mb * 4);
  pa_tmp = alloc(m * heads * 64 * head_dim * 4);
  pa_ml = alloc(m * heads * 64 * 4);
  pa_es = alloc(m * heads * 64 * 4);
  acc_all = alloc((size_t)(2 * num_layers + 2) * m * 8);
  for (int i = 0; i < 2 * num_layers + 2; ++i) {
    Buf a = acc_all.at((size_t)i * m * 8);
    a.len = m * 8;
    acc.push_back(a);
  }
  logits = alloc(m * (size_t)vocab * 2);
  argmax = alloc(m * 8);
  positions = alloc(mb * 4);
  slots = alloc(mb * 8);
  exp_seq = alloc(mb * 4);
  exp_bt = alloc(m * (size_t)max_blocks * 4);
  spec_cu = alloc(2 * 4);
  conv_slots = alloc(4);
  slot_table = alloc(m * 4);
  num_accepted = alloc(4);

  // One request: every row shares the identity block table.
  int32_t* bt = (int32_t*)exp_bt.ptr();
  for (size_t r = 0; r < m; ++r)
    for (int b = 0; b < max_blocks; ++b) bt[r * max_blocks + b] = b;
  ((int32_t*)conv_slots.ptr())[0] = 1;
  int32_t* st = (int32_t*)slot_table.ptr();
  for (size_t r = 0; r < m; ++r) st[r] = (int32_t)(r + 1);

  // Rotary cache rows [cos(R/2) || sin(R/2)], bf16, one per position.
  cos_sin = alloc((size_t)max_ctx * rot_dim * 2);
  uint16_t* cs = (uint16_t*)cos_sin.ptr();
  const int R2 = rot_dim / 2;
  std::vector<double> inv(R2);
  for (int j = 0; j < R2; ++j) inv[j] = 1.0 / pow((double)rope_theta, (double)(2 * j) / rot_dim);
  for (int p = 0; p < max_ctx; ++p) {
    for (int j = 0; j < R2; ++j) {
      const float ang = (float)p * (float)inv[j];
      cs[(size_t)p * rot_dim + j] = f32_to_bf16(cosf(ang));
      cs[(size_t)p * rot_dim + R2 + j] = f32_to_bf16(sinf(ang));
    }
  }
  rope_inv = inv;
  cs_step = alloc(mb * rot_dim * 2);
  row_idx = alloc(mb * 4);
  for (size_t i = 0; i < mb; ++i) ((int32_t*)row_idx.ptr())[i] = (int32_t)i;
  cs_cur = &cos_sin;
  seqs.reserve(kMaxSeqs);
  seq_new();  // sequence 0
  return true;
}

int Engine::seq_new() {
  int id = -1;
  for (int i = 0; i < (int)seqs.size(); ++i)
    if (!seqs[i].live) {
      id = i;
      break;
    }
  if (id < 0) {
    if ((int)seqs.size() >= kMaxSeqs) return -1;
    seqs.emplace_back();
    id = (int)seqs.size() - 1;
  }
  Seq& S = seqs[id];
  S = Seq{};
  S.live = true;
  S.L.resize(num_layers);
  const bool fused = getenv("Q38_GDN_OFF") == nullptr;
  if (fused && gdn_arena.empty()) {
    auto r4 = [](size_t n) { return (n + 3) / 4 * 4; };
    gdn_conv_fl = r4((size_t)2 * conv_stride);
    gdn_ssm_fl = r4((size_t)2 * ssm_stride);
    gdn_log_fl = r4((size_t)2 * kGdnLogCap * gdn_log_stride);
    gdn_region_fl = gdn_conv_fl + gdn_ssm_fl + gdn_log_fl;
    gdn_arena.resize(num_layers);
    for (int l = 0; l < num_layers; ++l)
      if (layers[l].is_gdn) gdn_arena[l] = alloc((size_t)kMaxSeqs * gdn_region_fl * 4);
    gdn_groups = alloc((kMaxSeqs + kBigRows / 8) * 64);  // + one entry per sub-step of a big step
  }
  for (int l = 0; l < num_layers; ++l) {
    if (!layers[l].is_gdn) continue;  // K/V come from ensure_kv
    if (fused) {
      const Buf& A = gdn_arena[l];
      const size_t base = (size_t)id * gdn_region_fl * 4;
      S.L[l].conv_state = Buf{A.b, A.off + base, gdn_conv_fl * 4};
      S.L[l].ssm_state = Buf{A.b, A.off + base + gdn_conv_fl * 4, gdn_ssm_fl * 4};
      S.L[l].gdn_log = Buf{A.b, A.off + base + (gdn_conv_fl + gdn_ssm_fl) * 4, gdn_log_fl * 4};
      // the region may have held another sequence
      memset(S.L[l].conv_state.ptr(), 0, (size_t)2 * conv_stride * 4);
      memset(S.L[l].ssm_state.ptr(), 0, (size_t)2 * ssm_stride * 4);
      continue;
    }
    // Unfused diagnostic path: slot 0 is the kernels' null block; the stream
    // lives in slot 1 (conv) and slots 1..max_rows (one recurrent checkpoint
    // per step row).
    S.L[l].conv_state = alloc((size_t)2 * conv_stride * 4, true);
    S.L[l].ssm_state = alloc((size_t)(1 + max_rows) * ssm_stride * 4, true);
    S.L[l].gdn_log = alloc((size_t)2 * kGdnLogCap * gdn_log_stride * 4, true);
  }
  return id;
}

void Engine::seq_free(int id) {
  if (id <= 0 || !seq_ok(id)) return;  // sequence 0 lives as long as the engine
  seqs[id] = Seq{};
}

// K/V rows are allocated for the context a sequence actually reaches and
// doubled as it grows, so many short sequences do not each reserve the full
// context window.
void Engine::ensure_kv(Seq& S, int64_t tokens) {
  if (tokens <= S.kv_cap) return;
  int64_t cap = std::max<int64_t>({tokens, 2 * S.kv_cap, 4096});
  cap = std::min<int64_t>((cap + block_size - 1) / block_size * block_size,
                          (int64_t)max_blocks * block_size);
  const size_t row = (size_t)kv_heads * head_dim * 2;
  const size_t have = (size_t)(S.pos + S.last_m) * row;
  for (int l = 0; l < num_layers; ++l) {
    if (layers[l].is_gdn) continue;
    Buf kc = alloc((size_t)cap * row, true), vc = alloc((size_t)cap * row, true);
    if (have) {
      memcpy(kc.ptr(), S.L[l].kc.ptr(), have);
      memcpy(vc.ptr(), S.L[l].vc.ptr(), have);
    }
    S.L[l].kc = kc;
    S.L[l].vc = vc;
  }
  S.kv_cap = cap;
}

void Engine::ensure_mtp_kv(Seq& S, int64_t tokens) {
  if (tokens <= S.mtp_cap) return;
  int64_t cap = std::max<int64_t>({tokens, 2 * S.mtp_cap, 4096});
  cap = std::min<int64_t>((cap + block_size - 1) / block_size * block_size,
                          (int64_t)max_blocks * block_size);
  const size_t row = (size_t)kv_heads * head_dim * 2;
  const size_t have = (size_t)(S.mtp_pos + S.mtp_chain) * row;
  Buf kc = alloc((size_t)cap * row, true), vc = alloc((size_t)cap * row, true);
  if (have) {
    memcpy(kc.ptr(), S.mkc.ptr(), have);
    memcpy(vc.ptr(), S.mvc.ptr(), have);
  }
  S.mkc = kc;
  S.mvc = vc;
  S.mtp_cap = cap;
}

void Engine::commit_rows(Seq& S, int n) {
  if (n == S.last_m) S.rope_off = S.rope_off_next;  // explicit-position steps are kept whole
  S.rope_off_next = S.rope_off;
  S.pos += n;
  S.accepted = S.big_last ? S.big_last : n;  // GDN history: rows of the last (sub-)step
  S.big_last = 0;
  S.last_m = 0;
  if (has_dspine) S.d_pend = S.d_taps_ok ? n : 0;
}

void Engine::seq_reset(Seq& S) {
  S.pos = 0;
  S.rope_off = S.rope_off_next = 0;
  S.last_m = 0;
  S.accepted = 1;
  S.conv_cur = 1;
  S.gdn_log_n = S.gdn_prev_m = S.gdn_log_cur = 0;
  S.mtp_pos = 0;
  S.mtp_chain = 0;
  S.out_m = 0;
  S.restored = false;
  S.d_n = S.d_base = 0;
  S.d_pend = S.d_stash_n = 0;
  S.d_taps_ok = false;
  S.d_oov = 0;
  // K/V rows go back to the system; they are allocated again as the sequence grows
  S.kv_cap = S.mtp_cap = 0;
  S.mkc = S.mvc = Buf{};
  for (int l = 0; l < num_layers; ++l) {
    if (!layers[l].is_gdn) {
      S.L[l].kc = S.L[l].vc = Buf{};
      continue;
    }
    memset(S.L[l].conv_state.ptr(), 0, (size_t)2 * conv_stride * 4);
    memset(S.L[l].ssm_state.ptr(), 0, (size_t)2 * ssm_stride * 4);  // null + slot 1
  }
}

// Norm folding (kernels/q38_gemv.metal): a consumer with a Fold reads the raw
// residual row and applies the RMSNorm itself; a producer with a Res adds the
// residual in its epilogue and leaves sum-of-squares partials for that.
struct Fold {
  const Buf* nw;    // norm weight [K]
  const Buf* part;  // partials [m][npart]
  int npart;
  float eps;
};
struct Res {
  const Buf* rsd;   // residual rows to add
  const Buf* part;  // sum-of-squares accumulator to fill
  const Buf* nw;    // the NEXT norm's weight
  const Buf* hs;    // out: residual * (1 + w) rows for the consumers
};

// Columns per pass of the q38 GEMV kernels for m rows: at most 4 (more blows
// the register budget), and split evenly so no pass computes pad columns
// (6 rows = 2 x 3, not 4 + 2 padded to 4).
static inline int q38_cols(int m) {
  // Up to 8 rows share one pass (Q38_COLS_MAX=4 restores two passes for 5..8).
  static const int cmax = getenv("Q38_COLS_MAX") ? atoi(getenv("Q38_COLS_MAX")) : 8;
  const int passes = (m + cmax - 1) / cmax;
  return (m + passes - 1) / passes;
}
// The BF16-input fused kernels (norm fold, residual, gate/up) have no wide
// variants: at most 4 columns a pass.
static inline int q38_cols4(int m) {
  const int passes = (m + 3) / 4;
  return (m + passes - 1) / passes;
}
// Kernel name suffix for r1 columns: c<r1> up to 4 columns; 5 columns still
// fit with a whole 16-weight group per lane iteration (w16c5); 6..8 need the
// half-group variant (w8c<r1>), or the register state spills (8 rows: 111 ms
// against 68). Measured on the mix1 pack, ms per step for 4..8 rows: 40.2,
// 50.0, 55.3, 61.4, 67.6 (two passes: 58.2, 58.7, 67.6, 69.0). The half-group
// kernel is slower where the whole-group one fits (4 rows: 44.8 against 40.2).
// Q38_GW=8|16 forces one for r1 >= 5.
static inline std::string q38_csfx(int r1) {
  static const int gw = getenv("Q38_GW") ? atoi(getenv("Q38_GW")) : 0;
  if (r1 <= 4) return std::string("c") + char('0' + r1);
  return "w" + std::to_string(gw ? gw : r1 == 5 ? 16 : 8) + "c" + char('0' + r1);
}
static inline int q38_cthreads(int) { return 64; }

// `xh`: xin holds half * 2^-4 rows (NVFP4 projections only; callers check).
// Many-row projection through the 8x8 matrix primitive (kernels/q38_gemm.metal).
static bool gemm_ok(const QProj& p) {
  return (p.fmt == 1 || p.uniform()) && p.N % 64 == 0 && p.K % 32 == 0;
}
static void emit_gemm(Enc& e, const QProj& p, const Buf& xin, const Buf& out, int m, bool xh) {
  e.pipeline(std::string(p.fmt == 4 ? "q38_gemm_u5" : p.fmt == 3 ? "q38_gemm_u4" : "q38_gemm_nvfp4") +
             (xh ? "_xh" : ""));
  e.out(out, 0);
  e.in(p.w, 1);
  e.in(xin, 2);
  e.in(p.scale, 3);
  e.in(p.gscale, 4);
  e.bytes(p.N, 5);
  e.bytes(p.K, 6);
  e.bytes(m, 7);
  e.in(p.uniform() ? p.bits : p.w, 8);
  e.dispatch((m + 31) / 32, p.N / 64, 1, 128, 1, 1);
}

static void emit_qproj(Enc& e, const QProj& p, const Buf& xin, const Buf& out, int m,
                       const Fold* fold = nullptr, const Res* res = nullptr, bool xh = false) {
  static const bool gemm_all = getenv("Q38_GEMM") != nullptr;  // diagnostic: GEMM kernels at any row count
  static const char* gemm_skip = getenv("Q38_GEMM_SKIP");  // diagnostic: N values (comma list) kept off the GEMM route
  bool skip = false;
  if (gemm_skip) { char key[32]; snprintf(key, sizeof key, ",%d,", p.N); skip = strstr(gemm_skip, key) != nullptr; snprintf(key, sizeof key, ",%dx%d,", p.N, p.K); skip = skip || strstr(gemm_skip, key) != nullptr; }
  if ((gemm_all || m > 8) && !skip && !fold && !res && gemm_ok(p)) {
    emit_gemm(e, p, xin, out, m, xh);
    return;
  }
  if (p.uniform() && !xh) {
    // callers route u4/u5 through the half path; reaching here means a
    // diagnostic switch turned it off
    fprintf(stderr, "q38: u4/u5 projection without the half-activation path (Q38_XH/Q38_NORM_OFF/"
                    "Q38_GDN_OFF/Q38_ATTN_OFF are not usable with this pack)\n");
    abort();
  }
  if (xh) {
    const int r1 = q38_cols(m);
    e.pipeline(std::string(p.fmt == 4 ? "q38_u5_mvx_" : p.fmt == 3 ? "q38_u4_mvx_" : "q38_nvfp4_mvx_") +
               q38_csfx(r1));
    e.out(out, 0);
    e.in(p.w, 1);
    e.in(xin, 2);
    e.in(p.scale, 3);
    e.in(p.gscale, 4);
    e.bytes(p.N, 5);
    e.bytes(p.K, 6);
    e.bytes(m, 7);
    if (p.uniform()) e.in(p.bits, 8);
    e.dispatch(p.N / 8, (m + r1 - 1) / r1, 1, q38_cthreads(r1), 1, 1);
    return;
  }
  if (fold || res) {
    // only NVFP4 has these variants; callers check before asking
    const int r1 = q38_cols4(m);
    e.pipeline(std::string(fold ? "q38_nvfp4_mvn_c" : "q38_nvfp4_mvr_c") + char('0' + r1));
    e.out(out, 0);
    e.in(p.w, 1);
    e.in(xin, 2);
    e.in(p.scale, 3);
    e.in(p.gscale, 4);
    e.bytes(p.N, 5);
    e.bytes(p.K, 6);
    e.bytes(m, 7);
    if (fold) {
      e.in(*fold->part, 9);
      e.bytes(fold->eps, 11);
    } else {
      e.in(*res->rsd, 12);
      e.out(*res->part, 13);
      e.in(*res->nw, 14);
      e.out(*res->hs, 15);
    }
    e.dispatch(p.N / 8, (m + r1 - 1) / r1, 1, 64, 1, 1);
    return;
  }
  // Q38_MV4R_MIN: smallest row count routed to the mv4r kernels (default 3).
  static const int mv4r_min = getenv("Q38_MV4R_MIN") ? atoi(getenv("Q38_MV4R_MIN")) : 3;
  static const char* nv_env = getenv("Q38_NV");
  static const bool nv_off = nv_env && !strcmp(nv_env, "off");
  static const char* nv_kernel = (nv_env && *nv_env && !nv_off) ? nv_env : nullptr;
  static const bool nv_m1 = getenv("Q38_NV_M1") != nullptr;  // q38 kernel at m = 1 too
  static const int nv_nr = nv_kernel ? nv_kernel[3] - '0' : 4;
  static const int nv_r1 = nv_kernel ? nv_kernel[5] - '0' : 4;
  if (p.fmt == 2) {
    e.pipeline("mittens::muse_dense_gemv");
    e.out(out, 0);
    e.in(p.w, 1);
    e.in(xin, 2);
    e.bytes(p.N, 3);
    e.bytes(p.K, 4);
    e.dispatch(p.N, m, 1, 32, 1, 1);
    return;
  }
  if (p.fmt == 0) {
    if (m == 1)
      tk::launch_qgemv_fp8ch(e, out, p.w, xin, p.scale, p.N, p.K, "bfloat16");
    else if (m >= mv4r_min && m <= 8)
      tk::launch_qgemv_fp8ch_mv4r(e, out, p.w, xin, p.scale, p.N, p.K, m, "bfloat16");
    else
      tk::launch_qgemv_fp8ch_mb(e, out, p.w, xin, p.scale, p.N, p.K, m, "bfloat16");
  } else {
    if (m == 1 && !nv_m1)
      tk::launch_qgemv_nvfp4_planar(e, out, p.w, xin, p.scale, p.gscale, p.N, p.K, "bfloat16");
    else if (m >= 1 && m <= 8 && !nv_off) {
      // q38 single-pass half kernels (kernels/q38_gemv.metal). Columns per
      // pass follow m so no pad column is computed; 5..8 rows take two
      // 4-column passes. Q38_NV=<h|f|s>_r<NR>c<R1> forces one variant,
      // Q38_NV=off restores the QuixiCore route.
      const int nr = nv_kernel ? nv_nr : 4;
      const int r1 = nv_kernel ? nv_r1 : q38_cols(m);
      e.pipeline(nv_kernel ? std::string("q38_nvfp4_mv_") + nv_kernel
                           : r1 > 4 ? "q38_nvfp4_mv_h_" + q38_csfx(r1)
                           : std::string("q38_nvfp4_mv_h_r4c") + char('0' + r1));
      e.out(out, 0);
      e.in(p.w, 1);
      e.in(xin, 2);
      e.in(p.scale, 3);
      e.in(p.gscale, 4);
      e.bytes(p.N, 5);
      e.bytes(p.K, 6);
      e.bytes(m, 7);
      e.dispatch(p.N / (nr * 2), (m + r1 - 1) / r1, 1, q38_cthreads(r1), 1, 1);
    } else if (m >= mv4r_min && m <= 8)
      tk::launch_qgemv_nvfp4_mv4r(e, out, p.w, xin, p.scale, p.gscale, p.N, p.K, m, "bfloat16");
    else
      tk::launch_qgemv_nvfp4_planar_mb(e, out, p.w, xin, p.scale, p.gscale, p.N, p.K, m, "bfloat16");
  }
}


// MLP front half: gate_up projection + SwiGLU into `mid`. NVFP4 weights take
// the fused kernel (one dispatch); other formats use GEMV + qc_swiglu.
// `xh`: xin is half * 2^-4; with `half_out` `mid` is written the same way (for
// an XH down projection), otherwise as BF16.
static void emit_gate_up(Enc& e, const QProj& p, const Buf& xin, const Buf& gu, const Buf& mid,
                         int inter, int m, const Fold* fold = nullptr, bool xh = false,
                         bool half_out = false) {
  static const bool fused = getenv("Q38_GU_OFF") == nullptr;
  if (fused && p.uniform() && xh && m <= 8) {
    const int r1 = q38_cols(m);
    e.pipeline(std::string(p.fmt == 4 ? "q38_u5_" : "q38_u4_") + (half_out ? "gux_" : "guxb_") +
               q38_csfx(r1));
    e.out(mid, 0);
    e.in(p.w, 1);
    e.in(xin, 2);
    e.in(p.scale, 3);
    e.in(p.gscale, 4);
    e.bytes(inter, 5);
    e.bytes(p.K, 6);
    e.bytes(m, 7);
    e.in(p.bits, 8);
    e.dispatch(inter / 4, (m + r1 - 1) / r1, 1, q38_cthreads(r1), 1, 1);
    return;
  }
  if (fused && p.fmt == 1 && m <= 8) {
    const int r1 = xh ? q38_cols(m) : q38_cols4(m);  // wide kernels: half-input variants only
    e.pipeline(xh ? std::string(half_out ? "q38_nvfp4_gux_" : "q38_nvfp4_guxb_") + q38_csfx(r1)
                  : std::string(fold ? "q38_nvfp4_gun_c" : "q38_nvfp4_gu_h_c") + char('0' + r1));
    e.out(mid, 0);
    e.in(p.w, 1);
    e.in(xin, 2);
    e.in(p.scale, 3);
    e.in(p.gscale, 4);
    e.bytes(inter, 5);
    e.bytes(p.K, 6);
    e.bytes(m, 7);
    if (fold) {
      e.in(*fold->part, 9);
      e.bytes(fold->eps, 11);
    }
    e.dispatch(inter / 4, (m + r1 - 1) / r1, 1, q38_cthreads(r1), 1, 1);
    return;
  }
  emit_qproj(e, p, xin, gu, m);
  tk::launch_qc_swiglu(e, gu, mid, inter, /*has_clamp=*/0, /*limit=*/0.0f, /*oai_form=*/0,
                       /*alpha=*/1.0f, /*beta=*/0.0f, m * inter, "bfloat16");
}

// Gated full attention for `m` rows whose context ends at `width` tokens:
// qkv projection, q/k norm + RoPE + gate split, KV scatter, split-K paged
// attention, sigmoid gate, output projection into t_hidden. Input is `h`.
void Engine::emit_attn(Enc& e, const Layer& L, int m, const AGroup* ag, int nag,
                       const Buf& pos_buf, const Buf& slot_buf, const Buf& seq_buf,
                       const Buf& xin, const Buf& out, const Fold* fold, const Res* rs,
                       const Buf* xin_half) {
  const uint64_t kv_block_stride = (uint64_t)block_size * kv_heads * head_dim;
  const int n4_attn = m * q_sz / 4;
  auto elem = [&](int n4) { e.dispatch((n4 + 255) / 256, 1, 1, 256, 1, 1); };
  if (xin_half && L.qkv.xh_ok())
    emit_qproj(e, L.qkv, *xin_half, attn_qkv, m, nullptr, nullptr, true);
  else
    emit_qproj(e, L.qkv, xin, attn_qkv, m, fold);
  tk::launch_qk_norm_rope_gate(e, attn_qkv, L.q_norm, L.k_norm, *cs_cur, pos_buf, q, gate, k,
                               heads, kv_heads, head_dim, rot_dim, eps, qkv_n, m, "bfloat16",
                               "i32");
  {
    const int width4 = kv_sz / 4;
    e.pipeline("mittens::muse_copy_rows");
    e.out(v, 0);
    e.in(attn_qkv.at((size_t)(2 * q_sz + kv_sz) * 2), 1);
    e.bytes(width4, 2);
    e.bytes(qkv_n, 3);
    e.bytes(m * width4, 4);
    elem(m * width4);
  }
  static const bool q38_attn = getenv("Q38_ATTN_OFF") == nullptr;
  const bool grouped = q38_attn && heads == 6 * kv_heads && head_dim == 256;
  // half output for an XH output projection (never with a folded residual)
  const bool o_half = grouped && xin_half && L.o.xh_ok() && !rs;
  // The K/V rows are per sequence: one scatter per group's run of rows.
  int max_width = 1;
  for (int gi = 0; gi < nag; ++gi) {
    const AGroup& G = ag[gi];
    const size_t r0 = (size_t)G.row0, gm = (size_t)G.m;
    max_width = std::max(max_width, (int)G.width);
    tk::launch_kv_cache_scatter(e, rows(k, r0, gm, (size_t)kv_sz * 2),
                                rows(v, r0, gm, (size_t)kv_sz * 2), rows(slot_buf, r0, gm, 8),
                                *G.kc, *G.vc, G.m, kv_heads, head_dim, block_size,
                                kv_block_stride, "bfloat16");
  }
  if (grouped) {
    // Grouped-query attention, K/V read once per KV head (kernels/q38_attn.metal).
    // All rows go in one dispatch; with several sequences each row selects
    // its own K/V buffers (at most 8 groups: max_rows).
    // the partials buffers hold max_rows * kAttnMaxParts (row, partition) units
    const int max_parts = std::min(kAttnMaxParts, max_rows * kAttnMaxParts / std::max(m, 1));
    const int ps = std::max(32, (max_width + max_parts - 1) / max_parts);
    struct {
      int m, heads, kv_heads, part_size, n_parts, out_half;
      float scale;
    } ap = {m, heads, kv_heads, ps, (max_width + ps - 1) / ps, o_half ? 1 : 0, attn_scale};
    if (nag == 1) {
      e.pipeline("q38_attn_part");
      e.in(q, 0);
      e.in(*ag[0].kc, 1);
      e.in(*ag[0].vc, 2);
      e.in(seq_buf, 3);
      e.out(attn_pa, 4);
      e.out(attn_pm, 5);
      e.bytes(ap, 6);
    } else {
      int32_t* sel = (int32_t*)attn_sel.ptr();
      for (int gi = 0; gi < nag; ++gi)
        for (int i = 0; i < ag[gi].m; ++i) sel[ag[gi].row0 + i] = gi;
      e.pipeline("q38_attn_part_ms");
      e.in(q, 0);
      for (int gi = 0; gi < 8; ++gi) {
        e.in(*ag[gi < nag ? gi : 0].kc, 1 + gi);
        e.in(*ag[gi < nag ? gi : 0].vc, 9 + gi);
      }
      e.in(seq_buf, 17);
      e.out(attn_pa, 18);
      e.out(attn_pm, 19);
      e.bytes(ap, 20);
      e.in(attn_sel, 21);
    }
    e.dispatch(ap.n_parts, kv_heads, m, 32, 1, 1);
    e.pipeline("q38_attn_reduce");
    e.in(attn_pa, 0);
    e.in(attn_pm, 1);
    e.in(gate, 2);
    e.out(attn_out, 3);
    e.bytes(ap, 4);
    e.dispatch(heads, m, 1, 32, 1, 1);
  } else {
    // Reference path (one sequence only; the caller enforces it).
    const AGroup& G = ag[0];
    const int width = (int)G.width;
    const int units = m * heads;
    int np = std::max(1, std::min((1536 + units - 1) / units, 64));
    np = std::min(np, width);
    const int part_size = (width + np - 1) / np;
    const int n_parts = (width + part_size - 1) / part_size;
    tk::launch_paged_attention_partition(
        e, q, *G.kc, *G.vc, exp_bt, seq_buf, pa_tmp, pa_ml, pa_es, m, heads, kv_heads,
        head_dim, block_size, /*block_table_stride=*/max_blocks, attn_scale, n_parts,
        part_size, /*window=*/0, /*softcap=*/0.0f, kv_block_stride, "bfloat16");
    tk::launch_paged_attention_reduce(e, pa_tmp, pa_ml, pa_es, attn_out, m, heads, head_dim,
                                      n_parts, /*sinks=*/q, /*has_sink=*/0, "bfloat16");
    e.pipeline("mittens::muse_sigmoid_mul_exact");
    e.out(attn_out, 0);
    e.in(gate, 1);
    e.bytes(n4_attn, 2);
    elem(n4_attn);
  }
  emit_qproj(e, L.o, attn_out, out, m, nullptr, rs, o_half);
}

int Engine::step(const Group* grp_in, int ng, const int32_t* tokens, const uint16_t* embeds,
                 const int32_t* pos3, q38_step_stats* stats) {
  // validate first: a rejected call must leave the pending rows uncommitted
  static const bool gdn_fused = getenv("Q38_GDN_OFF") == nullptr;  // see gdn_fused_on()
  // big steps: the GDN core walks all rows in one dispatch (Q38_GDN_BIG_OFF: 8-row sub-step dispatches)
  static const bool gdn_big_loop = getenv("Q38_GDN_BIG_OFF") == nullptr;
  static const bool norm_fused = getenv("Q38_NORM_OFF") == nullptr;
  if (ng < 1 || ng > max_rows) return -1;
  Group grp[16];
  int m = 0;
  for (int gi = 0; gi < ng; ++gi) {
    const Group& G = grp_in[gi];
    if (!seq_ok(G.seq) || G.m < 1) return -1;
    for (int j = 0; j < gi; ++j)
      if (grp_in[j].seq == G.seq) return -1;  // one run of rows per sequence
    const Seq& S = seqs[G.seq];
    if (S.pos + S.last_m + G.m > max_ctx) return -2;
    grp[gi] = {G.seq, G.m, m};
    m += G.m;
  }
  const bool big = m > max_rows;
  if (big && (ng != 1 || m > kBigRows)) return -1;
  // the unfused diagnostic paths know one sequence and at most max_rows rows
  if ((ng > 1 || big) && (!gdn_fused || !norm_fused || getenv("Q38_ATTN_OFF"))) return -6;
  for (int i = 0; i < m; ++i)
    if (tokens[i] >= vocab || (tokens[i] < 0 && !embeds)) return -3;
  if (pos3)
    for (int i = 0; i < m; ++i) {
      if (pos3[3 * i] < 0) continue;  // implicit: the sequence's own next text position
      for (int j = 0; j < 3; ++j)
        if (pos3[3 * i + j] < 0 || pos3[3 * i + j] >= max_ctx) return -2;
    }
  if (!d_in_vocab.empty())
    for (int gi = 0; gi < ng; ++gi) {  // how far each sequence strays from the draft vocabulary
      Seq& S = seqs[grp[gi].seq];
      for (int i = 0; i < grp[gi].m; ++i) {
        const int32_t t = tokens[grp[gi].row0 + i];
        if (t >= 0) S.d_oov += (1.0f / 128.0f) * ((d_in_vocab[t] ? 0.f : 1.f) - S.d_oov);
      }
    }
  for (int gi = 0; gi < ng; ++gi) {
    Seq& S = seqs[grp[gi].seq];
    if (S.last_m) commit_rows(S, S.last_m);  // implicit commit of everything pending
    ensure_kv(S, S.pos + grp[gi].m);
  }
  std::vector<DInline> d_inl;
  if (has_dspine && d_preserve(&d_inl)) return -4;  // this step overwrites the tap rows

  const double t0 = now_ms();
  // Per-step host inputs: embedding rows and metadata.
  const uint16_t* emb = (const uint16_t*)embed.ptr();
  uint16_t* xr = (uint16_t*)x.ptr();
  for (int i = 0; i < m; ++i)
    memcpy(xr + (size_t)i * hidden,
           tokens[i] < 0 ? embeds + (size_t)i * hidden : emb + (size_t)tokens[i] * hidden,
           (size_t)hidden * 2);
  int32_t* pp = (int32_t*)positions.ptr();
  int64_t* sl = (int64_t*)slots.ptr();
  int32_t* es = (int32_t*)exp_seq.ptr();
  cs_cur = &cos_sin;
  if (pos3) {
    // Interleaved M-RoPE: frequency j rotates by the height position when
    // j % 3 == 1 (first 11), the width position when j % 3 == 2 (first 10),
    // and the temporal position otherwise. Text rows have all three equal.
    const int R2 = rot_dim / 2;
    const uint16_t* tab = (const uint16_t*)cos_sin.ptr();
    uint16_t* cs = (uint16_t*)cs_step.ptr();
    for (int gi = 0; gi < ng; ++gi)
      for (int i = 0; i < grp[gi].m; ++i) {  // implicit rows: resolved here, then treated as text
        const Seq& S = seqs[grp[gi].seq];
        const int r = grp[gi].row0 + i;
        if (pos3[3 * r] < 0)
          memcpy(cs + (size_t)r * rot_dim, tab + (size_t)(S.pos + i + S.rope_off) * rot_dim,
                 (size_t)rot_dim * 2);
      }
    for (int i = 0; i < m; ++i) {
      const int32_t* p = pos3 + 3 * i;
      if (p[0] < 0) continue;
      if (p[0] == p[1] && p[1] == p[2]) {
        memcpy(cs + (size_t)i * rot_dim, tab + (size_t)p[0] * rot_dim, (size_t)rot_dim * 2);
        continue;
      }
      for (int j = 0; j < R2; ++j) {
        const int axis = (j % 3 == 1 && j < 33) ? 1 : (j % 3 == 2 && j < 30) ? 2 : 0;
        const float ang = (float)p[axis] * (float)rope_inv[j];
        cs[(size_t)i * rot_dim + j] = f32_to_bf16(cosf(ang));
        cs[(size_t)i * rot_dim + R2 + j] = f32_to_bf16(sinf(ang));
      }
    }
    cs_cur = &cs_step;
  }
  AGroup ag[16];
  for (int gi = 0; gi < ng; ++gi) {
    Seq& S = seqs[grp[gi].seq];
    const int r0 = grp[gi].row0, gm = grp[gi].m;
    S.rope_off_next = S.rope_off;
    if (pos3) {
      const int32_t* l = pos3 + 3 * (r0 + gm - 1);
      if (l[0] >= 0)
        S.rope_off_next = (int64_t)std::max(l[0], std::max(l[1], l[2])) + 1 - (S.pos + gm);
    }
    for (int i = 0; i < gm; ++i) {
      pp[r0 + i] = pos3 ? r0 + i : (int32_t)(S.pos + i + S.rope_off);
      sl[r0 + i] = S.pos + i;
      es[r0 + i] = (int32_t)(S.pos + i + 1);  // a row attends to itself and everything before
    }
    ag[gi] = {r0, gm, S.pos + gm, nullptr, nullptr};
  }
  {  // unfused diagnostic paths (single sequence)
    const Seq& S0 = seqs[grp[0].seq];
    ((int32_t*)spec_cu.ptr())[0] = 0;
    ((int32_t*)spec_cu.ptr())[1] = m;
    ((int32_t*)num_accepted.ptr())[0] = S0.accepted;
  }

  const int cap = layer_cap > 0 && layer_cap < num_layers ? layer_cap : num_layers;
  const bool full = cap == num_layers;

  const int n4_hidden = m * hidden / 4;
  // Folding the residual RMSNorm into the GEMVs measured -0.8 ms at m = 1 and
  // +0.7 to +2 ms at m >= 3 (the RES/NORM variants are heavier per column), so
  // it is on for single-row steps only. Q38_FOLD=all|off overrides.
  static const char* fold_env = getenv("Q38_FOLD");
  const bool norm_fold = norm_fused && !has_uniform && ng == 1 &&
                         (fold_env ? !strcmp(fold_env, "all") : m == 1);
  // Half activations (XH GEMVs): the producers write half * 2^-4 rows so the
  // weight kernels skip the per-group BF16 conversion. Multi-row steps are
  // ALU-bound and gain 5 ms at m = 3; Q38_XH=all|off overrides.
  // fused GDN, per sequence: log entries to replay (committed + accepted rows
  // of its last step); flush the base when this step's rows would not fit
  // behind them
  int gdn_nrep[16];
  bool gdn_flush[16];
  for (int gi = 0; gi < ng; ++gi) {
    const Seq& S = seqs[grp[gi].seq];
    gdn_nrep[gi] = S.gdn_log_n + (S.gdn_prev_m ? S.accepted : 0);
    gdn_flush[gi] = gdn_nrep[gi] + grp[gi].m > kGdnLogCap;
  }
  static const char* xh_env = getenv("Q38_XH");
  const bool xh = norm_fused && gdn_fused && !norm_fold &&
                  (xh_env ? !strcmp(xh_env, "all") : (m >= 2 || has_uniform)) && m <= 8;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
    Enc e;
    e.eng = this;
    e.concurrent = !serial;
    e.enc = serial ? [cb computeCommandEncoder]
                   : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

    for (DInline& it : d_inl) d_emit_ctx(e, *it.S, it.taps, it.n, it.off);  // reads the old taps first
    const Buf* rcur = nullptr;
    int rflip = 0;
    // `tap` >= 0: also copy the summed residual into that DSpine tap column
    auto norm_add = [&](const Buf& xin, const Buf* resid, const Buf& w, const Buf& o,
                        const Buf& rout, int tap = -1) {
      if (!norm_fused) {
        tk::launch_gemma_rms_norm_add_dyn(e, xin, *resid, w, o, rout, (uint32_t)m, hidden, eps);
        return;
      }
      e.pipeline("q38_norm_add");
      e.in(xin, 0);
      e.in(resid ? *resid : xin, 1);
      e.in(w, 2);
      e.out(o, 3);
      e.out(rout, 4);
      e.bytes(hidden, 5);
      e.bytes(eps, 6);
      e.bytes((int)(resid != nullptr), 7);
      e.bytes(hidden, 8);
      e.out(hh, 9);
      e.bytes((int)xh, 10);
      if (tap >= 0)
        e.out(d_taps.at((size_t)tap * hidden * 2), 11);
      else
        e.in(w, 11);  // bound, never written
      e.bytes(tap >= 0 ? kDLayers * hidden : 0, 12);
      e.dispatch(m, 1, 1, 512, 1, 1);
    };
    auto seam = [&](const Buf& xin, const Buf& w, int tap = -1) {
      const Buf& rout = rflip ? res2 : res;
      rflip ^= 1;
      norm_add(xin, rcur, w, h, rout, tap);
      rcur = &rout;
    };
    auto elem = [&](int n4) { e.dispatch((n4 + 255) / 256, 1, 1, 256, 1, 1); };

    // Norm folding state: when `pend` is set, the residual row *rcur is the
    // un-normalized input of the next consumers and `fold` describes its norm.
    const int npart = hidden / 4;
    Fold fold{nullptr, nullptr, npart, eps};
    int acc_next = 0;
    memset(acc_all.ptr(), 0, (size_t)(2 * num_layers + 2) * max_rows * 8);
    bool pend = false;
    auto first_proj = [&](const Layer& L) -> const QProj& { return L.is_gdn ? L.qkvz : L.qkv; };
    // producer: write t + residual into the other residual buffer, with partials
    auto res_out = [&](Res* r) -> const Buf& {
      const Buf& rout = rflip ? res2 : res;
      r->rsd = rcur;
      r->part = &acc[acc_next++];
      r->hs = &h;
      rflip ^= 1;
      return rout;
    };

    for (int l = 0; l < cap; ++l) {
      const Layer& L = layers[l];
      if (l == 0 && norm_fused) {
        norm_add(x, nullptr, L.in_norm, h, res);
        rcur = &res;
        rflip = 1;
      } else if (l == 0) {
        tk::launch_gemma_rms_norm_dyn(e, x, L.in_norm, h, (uint32_t)m, hidden, eps);
        e.pipeline("mittens::muse_copy_rows");
        e.out(res, 0);
        e.in(x, 1);
        e.bytes(hidden / 4, 2);
        e.bytes(hidden, 3);
        e.bytes(n4_hidden, 4);
        elem(n4_hidden);
        rcur = &res;
        rflip = 1;
      } else if (pend) {
        // the previous down projection already wrote *rcur and h = residual * (1 + in_norm)
      } else {
        // this seam forms the residual after layer l - 1: the drafter's tap when it is one
        seam(t_hidden, L.in_norm, has_dspine ? d_tap_of[l - 1] : -1);
      }
      const Fold* f_in = pend ? &fold : nullptr;
      const Buf& mix_in = h;
      pend = false;

      // mixer output folds into the MLP's gate_up when both are NVFP4
      const QProj& out_proj = L.is_gdn ? L.gdn_out : L.o;
      const bool fold_post = norm_fold && out_proj.fmt == 1 && L.gate_up.fmt == 1;
      Res rs{nullptr, nullptr, &L.post_norm, nullptr};
      const Fold f_copy = fold;  // the mixer reads the input fold while we set up the next
      const Fold* f_mix = f_in ? &f_copy : nullptr;
      const Buf* mix_out = &t_hidden;
      if (fold_post) {
        mix_out = &res_out(&rs);
      }

      if (L.is_gdn) {
        const bool gout_h = xh && L.gdn_out.xh_ok();
        if (xh && L.qkvz.xh_ok())
          emit_qproj(e, L.qkvz, hh, qkvz, m, nullptr, nullptr, true);
        else
          emit_qproj(e, L.qkvz, mix_in, qkvz, m, f_mix);
        if (gdn_fused) {
          // One dispatch for gates + conv/prepare + recurrence + gated norm
          // (kernels/q38_gdn.metal): grid (value heads, sequence groups).
          struct GG {
            int row0, m, accepted, n_replay, flush;
            int conv_rd, conv_wr, base_off, log_rd, log_wr;  // element offsets in the layer's arena
          };
          GG* gg = (GG*)gdn_groups.ptr();
          if (big) {
            // One sequence, more rows than the kernel chains: consecutive
            // 8-row sub-steps, each exactly what a committed step would be
            // (same content for every layer; the host state is advanced once
            // after the layer loop).
            const Seq& S = seqs[grp[0].seq];
            const long reg = (long)grp[0].seq * (long)gdn_region_fl;
            const long lg0 = reg + (long)(gdn_conv_fl + gdn_ssm_fl);
            int conv_cur = S.conv_cur, log_n = S.gdn_log_n, prev_m = S.gdn_prev_m, log_cur = S.gdn_log_cur,
                acc_rows = S.accepted;
            if (gdn_big_loop) {
              // q38_gdn_core_big: all rows in one dispatch, the committed log replayed once,
              // the state stored back to the base, no log entries written
              gg[kMaxSeqs] = {0, m, acc_rows, log_n + (prev_m ? acc_rows : 0), 1,
                              (int)(reg + (long)conv_cur * conv_stride),
                              (int)(reg + (long)(1 - conv_cur) * conv_stride),
                              (int)(reg + (long)gdn_conv_fl + (long)ssm_stride),
                              (int)(lg0 + (long)log_cur * kGdnLogCap * (long)gdn_log_stride), 0};
            }
            for (int r0 = 0, si = 0; r0 < m && !gdn_big_loop; r0 += 8, ++si) {
              const int sm = std::min(8, m - r0);
              const int nrep = log_n + (prev_m ? acc_rows : 0);
              const bool flush = nrep + sm > kGdnLogCap;
              gg[kMaxSeqs + si] = {r0, sm, acc_rows, nrep, flush ? 1 : 0,
                                   (int)(reg + (long)conv_cur * conv_stride),
                                   (int)(reg + (long)(1 - conv_cur) * conv_stride),
                                   (int)(reg + (long)gdn_conv_fl + (long)ssm_stride),
                                   (int)(lg0 + (long)log_cur * kGdnLogCap * (long)gdn_log_stride),
                                   (int)(lg0 + (long)((flush ? 1 - log_cur : log_cur) * kGdnLogCap +
                                                      (flush ? 0 : nrep)) * (long)gdn_log_stride)};
              conv_cur = 1 - conv_cur;
              log_n = flush ? 0 : nrep;
              if (flush) log_cur = 1 - log_cur;
              prev_m = sm;
              acc_rows = sm;
            }
          }
          for (int gi = 0; gi < ng && !big; ++gi) {
            const Seq& S = seqs[grp[gi].seq];
            const long reg = (long)grp[gi].seq * (long)gdn_region_fl;
            const long lg0 = reg + (long)(gdn_conv_fl + gdn_ssm_fl);
            const int nrep = gdn_nrep[gi];
            const bool flush = gdn_flush[gi];
            gg[gi] = {grp[gi].row0, grp[gi].m, S.accepted, nrep, flush ? 1 : 0,
                      (int)(reg + (long)S.conv_cur * conv_stride),
                      (int)(reg + (long)(1 - S.conv_cur) * conv_stride),
                      (int)(reg + (long)gdn_conv_fl + (long)ssm_stride),
                      (int)(lg0 + (long)S.gdn_log_cur * kGdnLogCap * (long)gdn_log_stride),
                      (int)(lg0 + (long)((flush ? 1 - S.gdn_log_cur : S.gdn_log_cur) * kGdnLogCap +
                                         (flush ? 0 : nrep)) * (long)gdn_log_stride)};
          }
          struct {
            int Hk, vpk, hidden, qkvz_stride, z_off, conv_cols;
            float prep_eps, q_scale, k_scale, norm_eps;
            int fold, npart;
            float fold_eps;
            int out_half;
            long log_stride;
          } gp = {gHk, gHv / gHk, hidden, gdn_w, gdn_qkv_ch, conv_cols, gdn_prep_eps, gdn_q_scale,
                  gdn_k_scale, eps, f_mix ? 1 : 0, npart, eps, gout_h ? 1 : 0, (long)gdn_log_stride};
          e.pipeline(big && gdn_big_loop ? "q38_gdn_core_big" : "q38_gdn_core");
          e.in(qkvz, 0);
          e.in(mix_in, 1);
          e.in(L.ba.w, 2);
          e.in(L.conv_w, 3);
          e.out(gdn_arena[l], 4);
          e.in(L.A_log, 5);
          e.in(L.dt_bias, 6);
          e.in(L.gdn_norm, 9);
          e.out(gnormed, 10);
          e.bytes(gp, 11);
          e.in(L.in_norm, 12);
          e.in(f_mix ? *f_mix->part : acc[0], 13);
          if (big && gdn_big_loop) {
            e.in(rows(gdn_groups, (size_t)kMaxSeqs, 1, sizeof(GG)), 7);
            e.dispatch(gHv, 1, 1, 384, 1, 1);  // 12 simdgroups: this kernel's thread limit
          } else if (big) {
            for (int si = 0; si < (m + 7) / 8; ++si) {  // the arena is written each time: these run in order
              // every range again: the encoder forgets a dispatch's ranges once it is issued
              e.in(qkvz, 0);
              e.in(mix_in, 1);
              e.out(gdn_arena[l], 4);
              e.out(gnormed, 10);
              e.in(rows(gdn_groups, (size_t)(kMaxSeqs + si), 1, sizeof(GG)), 7);
              e.dispatch(gHv, 1, 1, 512, 1, 1);
            }
          } else {
            e.in(gdn_groups, 7);
            e.dispatch(gHv, ng, 1, 512, 1, 1);
          }
        } else {
          emit_qproj(e, L.ba, h, ba_s, m);
          const SeqLayer& SL = seqs[grp[0].seq].L[l];
          tk::launch_gdn_fused_prepare(
              e, qkvz, ba_s, L.conv_w, SL.conv_state, spec_cu, conv_slots, L.A_log, L.dt_bias,
              gq, gk, gv, gdecay, gbeta, /*R=*/1, gHk, gHv, gDk, gDv, conv_kernel,
              /*load_initial=*/1, /*qkvz_stride=*/gdn_w, /*ba_stride=*/2 * gHv, conv_stride,
              gdn_prep_eps, gdn_q_scale, gdn_k_scale, conv_cols, num_accepted,
              /*spec_mode=*/1, "bfloat16");
          tk::launch_gdn_recur_spec(e, gq, gk, gv, gdecay, gbeta, SL.ssm_state, spec_cu,
                                    slot_table, num_accepted, gy, /*R=*/1, gHk, gHv, gDv, gDk,
                                    ssm_stride, /*table_stride=*/max_rows,
                                    /*table_width=*/max_rows, "float32");
          tk::launch_gdn_gated_rmsnorm_f32(e, gy, qkvz.at((size_t)gdn_qkv_ch * 2), L.gdn_norm,
                                           gnormed, m * gHv, gHv, gDv, /*z_stride=*/gdn_w, eps,
                                           "bfloat16");
        }
        emit_qproj(e, L.gdn_out, gnormed, *mix_out, m, nullptr, fold_post ? &rs : nullptr,
                   gout_h && gdn_fused);
      } else {
        for (int gi = 0; gi < ng; ++gi) {
          ag[gi].kc = &seqs[grp[gi].seq].L[l].kc;
          ag[gi].vc = &seqs[grp[gi].seq].L[l].vc;
        }
        emit_attn(e, L, m, ag, ng, positions, slots, exp_seq, mix_in, *mix_out, f_mix,
                  fold_post ? &rs : nullptr, xh ? &hh : nullptr);
      }

      // gate_up reads the half copy when it is NVFP4; `mid` is half only if
      // the down projection is NVFP4 as well (mixed packs keep some at FP8)
      static const bool gu_fused = getenv("Q38_GU_OFF") == nullptr;
      const bool gu_h = xh && gu_fused && L.gate_up.xh_ok();
      const bool mlp_h = gu_h && L.down.xh_ok();
      Fold f_post{&L.post_norm, rs.part, npart, eps};
      if (fold_post) {
        rcur = mix_out;
        emit_gate_up(e, L.gate_up, h, gu, mid, inter, m, &f_post);
      } else {
        seam(t_hidden, L.post_norm);
        if (gu_h)
          emit_gate_up(e, L.gate_up, hh, gu, mid, inter, m, nullptr, true, mlp_h);
        else
          emit_gate_up(e, L.gate_up, h, gu, mid, inter, m);
      }

      // down folds into the next layer's first projection (never the last
      // emitted layer: the tail wants t_hidden and the residual separately)
      const bool fold_next = norm_fold && l + 1 < cap && L.down.fmt == 1 &&
                             first_proj(layers[l + 1]).fmt == 1 &&
                             (!layers[l + 1].is_gdn || gdn_fused);
      if (fold_next) {
        Res rd{nullptr, nullptr, &layers[l + 1].in_norm, nullptr};
        const Buf& rout = res_out(&rd);
        emit_qproj(e, L.down, mid, rout, m, nullptr, &rd);
        rcur = &rout;
        fold.part = rd.part;
        pend = true;
      } else {
        emit_qproj(e, L.down, mid, t_hidden, m, nullptr, nullptr, mlp_h);
      }
      // DSpine tap of this layer: normally written by the next layer's seam; its
      // own dispatch only where no such seam runs
      if (has_dspine && d_tap_of[l] >= 0 && (fold_next || !norm_fused || l + 1 >= cap)) {
        e.pipeline("q38_d_tap");
        e.in(fold_next ? *rcur : t_hidden, 0);
        e.in(*rcur, 1);
        e.out(d_taps.at((size_t)d_tap_of[l] * hidden * 2), 2);
        e.bytes(hidden, 3);
        e.bytes((int)!fold_next, 4);
        e.bytes(kDLayers * hidden, 5);
        e.dispatch(m, 1, 1, 512, 1, 1);
      }
    }

    if (full) {
      const Buf& rout = rflip ? res2 : res;
      norm_add(t_hidden, rcur, final_norm, x, rout);
      if (big) {
        // only the last row's logits are wanted after a prefill chunk
        const Buf last = rows(x, (size_t)(m - 1), 1, (size_t)hidden * 2);
        if (gemm_ok(lm_head))
          emit_gemm(e, lm_head, last, logits, 1, false);
        else
          emit_qproj(e, lm_head, last, logits, 1);
      } else if (xh && lm_head.xh_ok())
        emit_qproj(e, lm_head, hh, logits, m, nullptr, nullptr, true);
      else
        emit_qproj(e, lm_head, x, logits, m);
      if (want_argmax && !big) {
      e.pipeline("mittens::muse_argmax");
      e.out(argmax, 0);
      e.in(logits, 1);
      e.bytes(vocab, 2);
      e.dispatch(m, 1, 1, 256, 1, 1);
      }
    } else {
      // Residual stream after layer `cap`, for parity against the reference.
      e.pipeline("mittens::muse_add_out");
      e.out(x, 0);
      e.in(t_hidden, 1);
      e.in(*rcur, 2);
      e.bytes(n4_hidden, 3);
      elem(n4_hidden);
    }

    if (gdn_fused && big && gdn_big_loop) {
      Seq& S = seqs[grp[0].seq];
      S.conv_cur = 1 - S.conv_cur;
      S.gdn_log_n = 0;  // the base holds the state after the chunk
      S.gdn_prev_m = 0;
      S.big_last = 1;   // conv history is read at column 0 next
    } else if (gdn_fused && big) {
      Seq& S = seqs[grp[0].seq];  // the same walk as the sub-steps above
      int acc_rows = S.accepted;
      for (int r0 = 0; r0 < m; r0 += 8) {
        const int sm = std::min(8, m - r0);
        const int nrep = S.gdn_log_n + (S.gdn_prev_m ? acc_rows : 0);
        const bool flush = nrep + sm > kGdnLogCap;
        S.conv_cur = 1 - S.conv_cur;
        S.gdn_log_n = flush ? 0 : nrep;
        if (flush) S.gdn_log_cur = 1 - S.gdn_log_cur;
        S.gdn_prev_m = sm;
        acc_rows = sm;
      }
      S.big_last = acc_rows;
    } else if (gdn_fused)
      for (int gi = 0; gi < ng; ++gi) {
        Seq& S = seqs[grp[gi].seq];
        S.conv_cur = 1 - S.conv_cur;
        S.gdn_log_n = gdn_flush[gi] ? 0 : gdn_nrep[gi];
        if (gdn_flush[gi]) S.gdn_log_cur = 1 - S.gdn_log_cur;
        S.gdn_prev_m = grp[gi].m;
      }
    [e.enc endEncoding];
    const double t1 = now_ms();
    const double u1 = (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9;
    [cb commit];
    [cb waitUntilCompleted];
    const double u2 = (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9;
    const double t2 = now_ms();
    if (getenv("Q38_TLOG"))
      fprintf(stderr, "tlog step: encode %.3f  commit->start %.3f  gpu %.3f  end->wake %.3f ms\n", t1 - t0,
              (cb.GPUStartTime - u1) * 1e3, (cb.GPUEndTime - cb.GPUStartTime) * 1e3, (u2 - cb.GPUEndTime) * 1e3);
    if (cb.status != MTLCommandBufferStatusCompleted) {
      error = cb.error ? cb.error.localizedDescription.UTF8String : "command buffer failed";
      return -4;
    }
    if (stats) {
      stats->encode_ms = t1 - t0;
      stats->gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
      stats->wall_ms = t2 - t0;
      stats->dispatches = e.dispatches;
      stats->barriers = e.barriers;
    }
    d_tpool = now_ms();
  }
  if (getenv("Q38_TLOG")) fprintf(stderr, "tlog step pool drain %.3f ms\n", now_ms() - d_tpool);
  for (int gi = 0; gi < ng; ++gi) {
    Seq& S = seqs[grp[gi].seq];
    S.last_m = grp[gi].m;
    S.out_row0 = grp[gi].row0;
    S.out_m = grp[gi].m;
    S.restored = false;
    S.logits_last_only = big;
    S.big_m = big ? m : 0;
    S.d_row0 = grp[gi].row0;
    S.d_taps_ok = has_dspine && full;
    S.d_stash_n = 0;
    if (big) {  // what is kept addressable: the last rows
      S.out_m = gdn_big_loop ? std::min(8, m) : S.big_last;
      S.out_row0 = m - S.out_m;
    }
  }
  return 0;
}

bool Engine::load_mtp(const char* path) {
  // Borrow the pack-reading members for the sidecar, then restore them.
  auto s_map = map; auto s_len = map_len; auto s_base = data_base; auto s_fd = fd;
  NSDictionary* s_t = tensors; NSDictionary* s_m = meta;
  bool ok = open_pack(path);
  if (ok) {
    const std::string o = "MTP.";
    ok = tensor(o + "pre_e", &mtp_pre_e, "bf16") && tensor(o + "pre_h", &mtp_pre_h, "bf16") &&
         tensor(o + "in_norm", &mtp.in_norm, "bf16") && tensor(o + "post_norm", &mtp.post_norm, "bf16") &&
         tensor(o + "q_norm", &mtp.q_norm, "bf16") && tensor(o + "k_norm", &mtp.k_norm, "bf16") &&
         tensor(o + "norm", &mtp_norm, "bf16") && proj(o + "fc", &mtp_fc) && proj(o + "qkv", &mtp.qkv) &&
         proj(o + "o", &mtp.o) && proj(o + "gate_up", &mtp.gate_up) && proj(o + "down", &mtp.down);
    mtp_read_gb = [meta[@"decode_read_bytes"] doubleValue] / 1e9;
  }
  map = s_map; map_len = s_len; data_base = s_base; fd = s_fd; tensors = s_t; meta = s_m;
  if (!ok) return false;
  mtp.is_gdn = false;
  const size_t m = max_rows;
  m_emb = alloc(m * hidden * 2);
  m_hid = alloc(m * hidden * 2);
  fc_in = alloc(m * 2 * hidden * 2);
  mx = alloc(m * hidden * 2);
  m_out = alloc(m * hidden * 2);
  m_logits = alloc(m * (size_t)vocab * 2);
  m_argmax = alloc(8);
  m_positions = alloc(m * 4);
  m_slots = alloc(m * 8);
  m_exp_seq = alloc(m * 4);
  has_mtp = true;
  return true;
}

// One pass of the MTP head over (hidden, next-token) pairs, a run of rows per
// sequence at that sequence's MTP positions mtp_pos + mtp_chain ...:
// fc([norm(embed) | norm(hidden)]) -> gated attention layer (own KV per
// sequence) -> mtp.norm, and with `head` the shared LM head on the last row
// of every run. `commit` makes the rows part of the committed history;
// otherwise they are tentative draft rows.
int Engine::mtp_run(const Group* grp_in, int ng, const int32_t* tokens, const uint16_t* embeds,
                    const uint16_t* const* hid, bool commit, bool head, q38_step_stats* stats) {
  if (!has_mtp) return -10;
  if (ng < 1 || ng > max_rows) return -1;
  Group grp[16];
  AGroup ag[16];
  int m = 0;
  for (int gi = 0; gi < ng; ++gi) {
    const Group& G = grp_in[gi];
    if (!seq_ok(G.seq) || G.m < 1) return -1;
    for (int j = 0; j < gi; ++j)
      if (grp_in[j].seq == G.seq) return -1;
    const Seq& S = seqs[G.seq];
    if (S.mtp_pos + (commit ? 0 : S.mtp_chain) + G.m > max_ctx) return -2;
    grp[gi] = {G.seq, G.m, m};
    m += G.m;
  }
  if (m > max_rows) return -1;
  for (int i = 0; i < m; ++i)
    if (tokens[i] >= vocab || (tokens[i] < 0 && !embeds)) return -3;
  cs_cur = &cos_sin;  // the draft head keeps plain sequence positions
  const double t0 = now_ms();
  const uint16_t* emb = (const uint16_t*)embed.ptr();
  // hidden rows first: `hid` may alias m_out (chained drafts)
  uint16_t* hr = (uint16_t*)m_hid.ptr();
  for (int i = 0; i < m; ++i) memmove(hr + (size_t)i * hidden, hid[i], (size_t)hidden * 2);
  uint16_t* er = (uint16_t*)m_emb.ptr();
  for (int i = 0; i < m; ++i)
    memcpy(er + (size_t)i * hidden,
           tokens[i] < 0 ? embeds + (size_t)i * hidden : emb + (size_t)tokens[i] * hidden,
           (size_t)hidden * 2);
  int32_t* pp = (int32_t*)m_positions.ptr();
  int64_t* sl = (int64_t*)m_slots.ptr();
  int32_t* es = (int32_t*)m_exp_seq.ptr();
  for (int gi = 0; gi < ng; ++gi) {
    Seq& S = seqs[grp[gi].seq];
    if (commit) S.mtp_chain = 0;
    const int64_t p0 = S.mtp_pos + S.mtp_chain;
    ensure_mtp_kv(S, p0 + grp[gi].m);
    for (int i = 0; i < grp[gi].m; ++i) {
      const int r = grp[gi].row0 + i;
      pp[r] = (int32_t)(p0 + i);
      sl[r] = p0 + i;
      es[r] = (int32_t)(p0 + i + 1);
    }
    ag[gi] = {grp[gi].row0, grp[gi].m, p0 + grp[gi].m, &S.mkc, &S.mvc};
  }
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
    Enc e;
    e.eng = this;
    e.concurrent = !serial;
    e.enc = serial ? [cb computeCommandEncoder]
                   : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    auto norm = [&](const Buf& xin, const Buf* resid, const Buf& w, const Buf& o, int o_stride,
                    const Buf& rout, bool half = false) {
      e.pipeline("q38_norm_add");
      e.in(xin, 0);
      e.in(resid ? *resid : xin, 1);
      e.in(w, 2);
      e.out(o, 3);
      e.out(rout, 4);
      e.bytes(hidden, 5);
      e.bytes(eps, 6);
      e.bytes((int)(resid != nullptr), 7);
      e.bytes(o_stride, 8);
      e.out(half ? hh : o, 9);  // half copy only for a uniform LM head
      e.bytes((int)half, 10);
      e.in(w, 11);
      e.bytes((int)0, 12);
      e.dispatch(m, 1, 1, 512, 1, 1);
    };
    // fc input rows are [norm_e(embedding) | norm_h(hidden)]
    norm(m_emb, nullptr, mtp_pre_e, fc_in, 2 * hidden, res);
    norm(m_hid, nullptr, mtp_pre_h, fc_in.at((size_t)hidden * 2), 2 * hidden, res2);
    emit_qproj(e, mtp_fc, fc_in, mx, m);
    norm(mx, nullptr, mtp.in_norm, h, hidden, res);  // res = layer input
    emit_attn(e, mtp, m, ag, ng, m_positions, m_slots, m_exp_seq, h, t_hidden, nullptr, nullptr);
    norm(t_hidden, &res, mtp.post_norm, h, hidden, res2);
    emit_gate_up(e, mtp.gate_up, h, gu, mid, inter, m);
    emit_qproj(e, mtp.down, mid, t_hidden, m);
    // a u4/u5 LM head only has half-input kernels: give it a half copy of m_out
    const bool head_half = head && lm_head.uniform();
    norm(t_hidden, &res2, mtp_norm, m_out, hidden, res, head_half);
    if (head) {
      // one run: the LM head on its last row only; several: on every row (the
      // weights are read once either way), each run using its last row
      const int hr0 = ng == 1 ? m - 1 : 0, hn = ng == 1 ? 1 : m;
      const Buf& hin = head_half ? hh : m_out;
      emit_qproj(e, lm_head, hin.at((size_t)hr0 * hidden * 2), m_logits.at((size_t)hr0 * vocab * 2),
                 hn, nullptr, nullptr, head_half);
      if (want_argmax && ng == 1) {
        e.pipeline("mittens::muse_argmax");
        e.out(m_argmax, 0);
        e.in(m_logits.at((size_t)hr0 * vocab * 2), 1);
        e.bytes(vocab, 2);
        e.dispatch(1, 1, 1, 256, 1, 1);
      }
    }
    [e.enc endEncoding];
    const double t1 = now_ms();
    [cb commit];
    [cb waitUntilCompleted];
    const double t2 = now_ms();
    if (cb.status != MTLCommandBufferStatusCompleted) {
      error = cb.error ? cb.error.localizedDescription.UTF8String : "command buffer failed";
      return -4;
    }
    if (stats) {
      stats->encode_ms = t1 - t0;
      stats->gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
      stats->wall_ms = t2 - t0;
      stats->dispatches = e.dispatches;
      stats->barriers = e.barriers;
    }
  }
  for (int gi = 0; gi < ng; ++gi) {
    Seq& S = seqs[grp[gi].seq];
    if (commit)
      S.mtp_pos += grp[gi].m;
    else
      S.mtp_chain += grp[gi].m;
    S.mtp_row = grp[gi].row0 + grp[gi].m - 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// DSpine block drafter
// ---------------------------------------------------------------------------
static inline float bf16_f32(uint16_t v) {
  const uint32_t b = (uint32_t)v << 16;
  float x;
  memcpy(&x, &b, 4);
  return x;
}

// Largest `k` BF16 values of lg[0..n): keys/indices in descending order.
// Returns how many were found (inf/nan are skipped).
static int topk_bf16(const uint16_t* lg, int vocab, int k, int32_t* hk, int32_t* hi) {
  auto key = [](uint16_t v) -> int32_t { return (v & 0x8000) ? -(int32_t)(v & 0x7FFF) : (int32_t)v; };
  int cnt = 0;
  int32_t thr = INT32_MIN;
  auto consider = [&](int i) {
    const int32_t kv = key(lg[i]);
    if (cnt == k && kv <= thr) return;
    if ((lg[i] & 0x7F80) == 0x7F80) return;  // inf / nan
    int pos = cnt < k ? cnt++ : k - 1;
    while (pos > 0 && hk[pos - 1] < kv) {
      hk[pos] = hk[pos - 1];
      hi[pos] = hi[pos - 1];
      --pos;
    }
    hk[pos] = kv;
    hi[pos] = i;
    if (cnt == k) thr = hk[k - 1];
  };
  int i = 0;
  for (; i < vocab && cnt < k; ++i) consider(i);
  // Once the heap is full almost nothing beats the threshold: test 16 logits
  // at a time and fall to the scalar path only for a block with a candidate.
  for (; i + 16 <= vocab; i += 16) {
    const int16x8_t t16 = vdupq_n_s16((int16_t)std::max(thr, (int32_t)INT16_MIN));
    uint16_t any = 0;
    for (int h8 = 0; h8 < 2; ++h8) {
      const int16x8_t v = vreinterpretq_s16_u16(vld1q_u16(lg + i + 8 * h8));
      const int16x8_t neg = vshrq_n_s16(v, 15);
      const int16x8_t mag = vandq_s16(v, vdupq_n_s16(0x7FFF));
      const int16x8_t kv = vsubq_s16(veorq_s16(mag, neg), neg);
      any |= vmaxvq_u16(vcgtq_s16(kv, t16));
    }
    if (any)
      for (int j = i; j < i + 16; ++j) consider(j);
  }
  for (; i < vocab; ++i) consider(i);
  return cnt;
}

bool Engine::load_dspine(const char* path) {
  if (lm_head.fmt != 1) return fail("the DSpine sampling tail reads NVFP4 LM head rows; this pack's head is another format");
  auto s_map = map; auto s_len = map_len; auto s_base = data_base; auto s_fd = fd;
  NSDictionary* s_t = tensors; NSDictionary* s_m = meta;
  bool ok = open_pack(path);
  std::vector<int> taps;
  std::vector<float> d_G_load;
  if (ok) {
    ok = [meta[@"kind"] isEqualToString:@"dspine"] && [meta[@"tables"] intValue] == 1 && [meta[@"layers"] intValue] == kDLayers &&
         [meta[@"head_dim"] intValue] == kDHd && [meta[@"code_size"] intValue] == kDCode &&
         [meta[@"rank"] intValue] == kDRank && [meta[@"head_topk"] intValue] == kDTopK;
    if (!ok) fail("%s is not a DSpine pack this engine knows (5 layers, head dim 128, code 1024, rank 512, top 16, with tables)", path);
  }
  if (ok) {
    d_heads = [meta[@"heads"] intValue];
    d_kvh = [meta[@"kv_heads"] intValue];
    d_mask_tok = [meta[@"mask_token"] intValue];
    d_window = [meta[@"window"] intValue];
    d_eps = [meta[@"eps"] floatValue];
    d_beps = [meta[@"bridge_eps"] floatValue];
    const double theta = [meta[@"rope_theta"] doubleValue];
    d_inv.resize(kDHd / 2);
    for (int j = 0; j < kDHd / 2; ++j) d_inv[j] = pow(theta, -(double)j / (kDHd / 2));
    for (NSNumber* t in meta[@"taps"]) taps.push_back([t intValue]);
    d_read_gb = [meta[@"decode_read_bytes"] doubleValue] / 1e9;
    ok = (int)taps.size() == kDLayers && d_heads == 4 * d_kvh && d_kvh <= 8 &&
         tensor("D.hidden_norm", &d_hidden_norm, "bf16") && tensor("D.norm", &d_norm, "bf16") &&
         tensor("D.tsv", &d_tsv, "bf16") && tensor("D.tg", &d_tg, "bf16") &&
         tensor("D.write_t", &d_write_t, "bf16") && tensor("D.gate_bias", &d_gate_b, "bf16") &&
         tensor("D.packed", &d_packed_w, "bf16") && proj("D.fc", &d_fc);
    for (int l = 0; ok && l < kDLayers; ++l) {
      const std::string o = "D" + std::to_string(l) + ".";
      DLayer& L = dl[l];
      ok = tensor(o + "in_norm", &L.in_norm, "bf16") && tensor(o + "post_norm", &L.post_norm, "bf16") &&
           tensor(o + "q_norm", &L.q_norm, "bf16") && tensor(o + "k_norm", &L.k_norm, "bf16") &&
           tensor(o + "read", &L.read, "bf16") && tensor(o + "write", &L.write, "bf16") &&
           proj(o + "qkv", &L.qkv) && proj(o + "o", &L.o) && proj(o + "gate_up", &L.gate_up) &&
           proj(o + "down", &L.down);
      if (ok && (L.qkv.fmt != 1 || L.qkv.N != (d_heads + 2 * d_kvh) * kDHd || L.qkv.K != hidden))
        ok = fail("DSpine projections must be NVFP4 (run tools/quantize.py on the drafter pack)");
      if (ok) {
        // k | v rows of the fused projection: what a context row needs
        const size_t r0 = (size_t)d_heads * kDHd;
        L.kv = L.qkv;
        L.kv.N = 2 * d_kvh * kDHd;
        L.kv.w = L.qkv.w.at(r0 * (hidden / 2));
        L.kv.scale = L.qkv.scale.at(r0 * (hidden / 16));
      }
    }
    if (ok && (d_fc.K != kDLayers * hidden || d_fc.N != hidden)) ok = fail("DSpine fc has the wrong shape");
    const int nv = [meta[@"draft_vocab"] intValue];
    Buf vbuf;
    if (ok && nv > 0 && (ok = tensor("D.vocab", &vbuf, "u8")))
      d_vocab.assign((const int32_t*)vbuf.ptr(), (const int32_t*)vbuf.ptr() + nv);
    Buf gbuf;
    if (ok && (ok = tensor("D.G", &gbuf, "f32")))
      d_G_load.assign((const float*)gbuf.ptr(), (const float*)gbuf.ptr() + (size_t)kDRank * kDRank);
  }
  map = s_map; map_len = s_len; data_base = s_base; fd = s_fd; tensors = s_t; meta = s_m;
  if (!ok) return false;
  d_tap_of.assign(num_layers, -1);
  for (int t = 0; t < kDLayers; ++t) {
    if (taps[t] < 0 || taps[t] >= num_layers) return fail("DSpine tap layer %d is out of range", taps[t]);
    d_tap_of[taps[t]] = t;
  }
  const size_t H = hidden, mb = kBigRows, m = 8;
  d_taps = alloc(mb * kDLayers * H * 2);
  d_taps2 = alloc(mb * kDLayers * H * 2);
  d_fcout = alloc(mb * H * 2);
  d_ctx = alloc(mb * H * 2);
  d_scrap = alloc(mb * H * 2);
  for (int l = 0; l < kDLayers; ++l) dl[l].kvp = alloc(mb * 2 * d_kvh * kDHd * 2);
  d_cs = alloc((mb + m) * kDHd * 4);
  d_slot = alloc((mb + m) * 4);
  d_seqlen = alloc(m * 4);
  d_x = alloc(m * H * 2);
  d_in = alloc(m * H * 2);
  d_inh = alloc(m * H * 2);
  d_res = alloc(m * H * 2);
  d_res2 = alloc(m * H * 2);
  d_h = alloc(m * H * 2);
  d_qkv = alloc(m * (size_t)(d_heads + 2 * d_kvh) * kDHd * 2);
  d_q = alloc(m * (size_t)d_heads * kDHd * 2);
  d_ao = alloc(m * (size_t)d_heads * kDHd * 2);
  d_t = alloc(m * H * 2);
  d_t2 = alloc(m * H * 2);
  d_raw = alloc(m * H * 2);
  d_rms = alloc(m * 4);
  d_packed = alloc(m * kDRank * 2);  // the bridge's q rows of the last layer
  d_proj = alloc(m * kDCode * 2);
  d_msg = alloc(m * kDRank * 2);
  d_delta = alloc(m * H * 2);
  d_hh = alloc(m * H * 2);
  d_pa = alloc(m * (size_t)d_heads * 64 * kDHd * 4);
  d_pm = alloc(m * (size_t)d_heads * 64 * 2 * 4);
  d_logits = alloc(m * (size_t)vocab * 2);
  d_c = alloc(m * kDRank * 4);
  d_hin = alloc((m + 1) * H * 2);
  d_hinh = alloc((m + 1) * H * 2);
  if (!d_vocab.empty()) {
    const size_t nv = d_vocab.size(), wb = (size_t)hidden / 2, sb = (size_t)hidden / 16;
    d_head = lm_head;
    d_head.N = (int)nv;
    d_head.w = alloc(nv * wb);
    d_head.scale = alloc(nv * sb);
    d_in_vocab.assign(vocab, 0);
    for (size_t i = 0; i < nv; ++i) {
      const size_t t = (size_t)d_vocab[i];
      if (t >= (size_t)vocab) return fail("DSpine draft vocabulary has a token id out of range");
      memcpy((uint8_t*)d_head.w.ptr() + i * wb, (const uint8_t*)lm_head.w.ptr() + t * wb, wb);
      memcpy((uint8_t*)d_head.scale.ptr() + i * sb, (const uint8_t*)lm_head.scale.ptr() + t * sb, sb);
      d_in_vocab[t] = 1;
    }
  }
  // host copies for the sampling tail
  d_gb.resize(kDRank);
  for (int i = 0; i < kDRank; ++i) d_gb[i] = bf16_f32(((const uint16_t*)d_gate_b.ptr())[i]);
  d_G = d_G_load;
  d_nw.resize(H);
  for (size_t i = 0; i < H; ++i) d_nw[i] = bf16_f32(((const uint16_t*)d_norm.ptr())[i]) + 1.0f;
  // compile every pipeline a draft call can use now, not inside the first calls
  for (const char* k : {"q38_d_tap", "q38_d_attn_part", "q38_d_attn_reduce", "q38_d_qkv", "q38_d_attn1", "q38_d_dense",
                        "q38_d_dense_f", "q38_gemm_nvfp4", "q38_d_bridge_msg", "q38_d_bridge_write"})
    pipe(k);
  for (int r = 2; r <= 8; ++r) {
    const int c4 = q38_cols4(r), c = q38_cols(r);
    pipe(std::string("q38_nvfp4_gu_h_c") + char('0' + c4));
    pipe(std::string("q38_nvfp4_mv_h_r4c") + char('0' + c4));
    if (c > 4) pipe("q38_nvfp4_mv_h_" + q38_csfx(c));
    pipe("q38_nvfp4_mvx_" + q38_csfx(c));
    pipe("q38_nvfp4_gux_" + q38_csfx(c));
  }
  has_dspine = true;
  if (seqs[0].pos == 0 && !seqs[0].last_m) {
    // One throwaway call (a zero context row, a full block) so the first real
    // one does not pay for wiring the drafter's weights: 180 ms measured.
    Seq& S = seqs[0];
    memset(d_taps.ptr(), 0, (size_t)kDLayers * H * 2);
    S.pos = 1;
    S.d_pend = 1;
    S.d_row0 = 0;
    const q38_sampling sp = {1.0f, 20, 0.95f};
    const double u[7] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    q38_draft_row rows_out[8];
    dspine_draft(0, 0, 7, &sp, 0.f, u, rows_out, nullptr);
    S.pos = 0;
    S.d_n = S.d_base = 0;
    S.d_pend = 0;
    d_gpu_ms = d_tail_ms = d_wall_ms = 0;
    d_tp[0] = d_tp[1] = d_tp[2] = 0;
    d_calls = 0;
  }
  return true;
}

void Engine::d_ensure(Seq& S, int64_t tokens) {
  if (tokens <= S.d_cap || S.d_cap == kDRing) return;
  int64_t cap = std::max<int64_t>(2048, S.d_cap);
  while (cap < tokens && cap < kDRing) cap *= 2;
  const size_t row = (size_t)d_kvh * kDHd * 2;
  const size_t have = (size_t)std::min<int64_t>(S.d_n, S.d_cap) * row;
  if (S.dk.empty()) {
    S.dk.resize(kDLayers);
    S.dv.resize(kDLayers);
  }
  for (int l = 0; l < kDLayers; ++l) {
    Buf kc = alloc((size_t)cap * row, true), vc = alloc((size_t)cap * row, true);
    if (have) {
      memcpy(kc.ptr(), S.dk[l].ptr(), have);
      memcpy(vc.ptr(), S.dv[l].ptr(), have);
    }
    S.dk[l] = kc;
    S.dv[l] = vc;
  }
  S.d_cap = cap;
}

namespace {
struct DQkv {
  int src_stride, nq, nk, cs_row0;
  float eps;
};
struct DAttn {
  int m, heads, kv_heads, part_size, n_parts, lo, ring_mask;
  float scale;
  int out_half;
};
}  // namespace

// Append `n` context rows (taps rows -> fc -> hidden_norm -> per layer k|v) at
// positions [d_n, d_n + n). Host inputs use d_cs / d_slot rows [0, n).
void Engine::d_emit_ctx(Enc& e, Seq& S, const Buf& taps, int n, int off) {
  d_ensure(S, S.d_n + n + 8);
  float* cs = (float*)d_cs.ptr() + (size_t)off * kDHd;
  int32_t* sl = (int32_t*)d_slot.ptr() + off;
  const size_t hb = (size_t)hidden * 2;
  const Buf fco = rows(d_fcout, (size_t)off, (size_t)n, hb), ctx = rows(d_ctx, (size_t)off, (size_t)n, hb);
  for (int i = 0; i < n; ++i) {
    const double p = (double)(S.d_n + i);
    for (int j = 0; j < kDHd / 2; ++j) {
      const double a = p * d_inv[j];
      cs[(size_t)i * kDHd + j] = (float)cos(a);
      cs[(size_t)i * kDHd + kDHd / 2 + j] = (float)sin(a);
    }
    sl[i] = (int32_t)((S.d_n + i) & (kDRing - 1));
  }
  emit_qproj(e, d_fc, taps, fco, n);
  e.pipeline("q38_norm_add");
  e.in(fco, 0);
  e.in(fco, 1);
  e.in(d_hidden_norm, 2);
  e.out(ctx, 3);
  e.out(rows(d_scrap, (size_t)off, (size_t)n, hb), 4);
  e.bytes(hidden, 5);
  e.bytes(d_eps, 6);
  e.bytes((int)0, 7);
  e.bytes(hidden, 8);
  e.out(rows(d_scrap, (size_t)off, (size_t)n, hb), 9);
  e.bytes((int)0, 10);
  e.in(d_hidden_norm, 11);
  e.bytes((int)0, 12);
  e.dispatch(n, 1, 1, 512, 1, 1);
  const int kvw = d_kvh * kDHd;
  for (int l = 0; l < kDLayers; ++l) {
    DLayer& L = dl[l];
    const Buf kvp = rows(L.kvp, (size_t)off, (size_t)n, (size_t)2 * kvw * 2);
    emit_qproj(e, L.kv, ctx, kvp, n);
    const DQkv qp = {2 * kvw, 0, d_kvh, off, d_eps};
    e.pipeline("q38_d_qkv");
    e.in(kvp, 0);
    e.in(L.k_norm, 1);
    e.in(L.k_norm, 2);
    e.in(d_cs, 3);
    e.in(d_slot.at((size_t)off * 4), 4);
    e.out(rows(d_scrap, (size_t)off, (size_t)n, hb), 5);  // no q heads: bound, never written
    e.out(S.dk[l], 6);
    e.out(S.dv[l], 7);
    e.bytes(qp, 8);
    e.dispatch(2 * d_kvh, n, 1, 32, 1, 1);
  }
  S.d_n += n;
}

// The committed rows of S's last step that its context still lacks: returns
// how many and where their tap rows are (a host stash is copied to d_taps2 rows
// [off, ..)); the caller appends them.
int Engine::d_take(Seq& S, Buf* taps, int off) {
  const int n = S.d_pend;
  if (!n) return 0;
  S.d_pend = 0;
  if (S.d_n != S.pos - n) S.d_n = S.d_base = S.pos - n;  // context lost track: start over here
  const size_t rb = (size_t)kDLayers * hidden * 2;
  *taps = rows(d_taps, (size_t)S.d_row0, (size_t)n, rb);
  if (S.d_stash_n) {
    memcpy((uint8_t*)d_taps2.ptr() + (size_t)off * rb, S.d_stash.data(), (size_t)n * rb);
    *taps = rows(d_taps2, (size_t)off, (size_t)n, rb);
    S.d_stash_n = 0;
  }
  return n;
}

// Append the committed rows of S's last step to its drafter context (own
// command buffer). Used when no draft call does it first.
int Engine::d_flush(Seq& S) {
  Buf taps;
  const int n = d_take(S, &taps, 0);
  if (!n) return 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
    Enc e;
    e.eng = this;
    e.concurrent = !serial;
    e.enc = serial ? [cb computeCommandEncoder]
                   : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    d_emit_ctx(e, S, taps, n, 0);
    [e.enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      error = cb.error ? cb.error.localizedDescription.UTF8String : "command buffer failed";
      return -4;
    }
  }
  return 0;
}

// A step is about to overwrite the tap rows. Rows other sequences have not
// committed yet get a host copy. Committed rows that are not in a drafter
// context yet are appended inside the step's own command buffer (`inl`, ahead
// of the layers, which then overwrite the taps); what does not fit its scratch
// rows goes through a command buffer of its own first.
int Engine::d_preserve(std::vector<DInline>* inl) {
  int off = 0;
  for (Seq& S : seqs) {
    if (!S.live) continue;
    if (S.d_pend) {
      if (off + S.d_pend > kBigRows) {
        if (d_flush(S)) return -4;
      } else {
        DInline it;
        it.S = &S;
        it.off = off;
        it.n = d_take(S, &it.taps, off);
        off += it.n;
        inl->push_back(it);
      }
    }
    if (S.last_m && S.d_taps_ok && !S.d_stash_n) {
      const size_t rb = (size_t)kDLayers * hidden * 2;
      S.d_stash.resize((size_t)S.last_m * rb / 2);
      memcpy(S.d_stash.data(), (const uint8_t*)d_taps.ptr() + (size_t)S.d_row0 * rb, (size_t)S.last_m * rb);
      S.d_stash_n = S.last_m;
    }
  }
  return 0;
}

int Engine::dspine_draft(int seq, int32_t anchor, int k, const q38_sampling* sp, float stop_below,
                         const double* u, q38_draft_row* out, q38_step_stats* stats) {
  DReq rq = {seq, anchor, k, u, out, 0};
  const int rc = dspine_draft_groups(&rq, 1, sp, stop_below, stats);
  return rc < 0 ? rc : rq.D;
}

// One drafter pass for several sequences: their blocks are laid out group
// after group (at most 8 rows in all) and share every pass over the drafter's
// weights and the LM head; attention and the bridge run per group (own
// context ring, own anchor).
int Engine::dspine_draft_groups(DReq* rq, int ng, const q38_sampling* sp, float stop_below,
                                q38_step_stats* stats) {
  if (!has_dspine) return -10;
  if (ng < 1 || ng > 4) return -1;
  int row0[4], m = 0;
  for (int gi = 0; gi < ng; ++gi) {
    const DReq& q = rq[gi];
    if (!seq_ok(q.seq) || q.k < 1 || q.k > 7 || q.anchor < 0 || q.anchor >= vocab) return -1;
    for (int j = 0; j < gi; ++j)
      if (rq[j].seq == q.seq) return -1;
    if (seqs[q.seq].last_m) return -7;  // commit first: the drafter continues the committed sequence
    row0[gi] = m;
    m += q.k + 1;
  }
  if (m > 8) return -1;
  // one-dispatch attention (Q38_D_ATTN1_OFF: the three-dispatch reference path, one sequence only)
  static const bool attn1 = getenv("Q38_D_ATTN1_OFF") == nullptr;
  if (!attn1 && ng > 1) return -6;
  const double t0 = now_ms();
  const int H = hidden;
  // context rows still to append ride in the same command buffer
  Buf taps[4];
  int n_ctx[4], ctx_off[4], coff = 0;
  for (int gi = 0; gi < ng; ++gi) {
    Seq& S = seqs[rq[gi].seq];
    ctx_off[gi] = coff;
    n_ctx[gi] = d_take(S, &taps[gi], coff);
    coff += n_ctx[gi];
    if (S.d_n + n_ctx[gi] != S.pos) S.d_n = S.d_base = S.pos - n_ctx[gi];  // out of step: no context before these rows
    d_ensure(S, S.pos + rq[gi].k + 9);
  }
  // block inputs: embeddings of [anchor, mask, ...], rotary rows after the context rows
  const uint16_t* emb = (const uint16_t*)embed.ptr();
  uint16_t* xr = (uint16_t*)d_x.ptr();
  const int cs0 = kBigRows;  // block rows of d_cs / d_slot sit after the context rows
  float* cs = (float*)d_cs.ptr() + (size_t)cs0 * kDHd;
  int32_t* sl = (int32_t*)d_slot.ptr() + cs0;
  int32_t* sq = (int32_t*)d_seqlen.ptr();
  const int64_t win = d_win();
  int lo_g[4];
  bool sub_all = !d_vocab.empty();
  for (int gi = 0; gi < ng; ++gi) {
    const Seq& S = seqs[rq[gi].seq];
    const int64_t P0 = S.pos;
    const int gm = rq[gi].k + 1;
    for (int i = 0; i < gm; ++i) {
      const int r = row0[gi] + i;
      memcpy(xr + (size_t)r * H, emb + (size_t)(i ? d_mask_tok : rq[gi].anchor) * H, (size_t)H * 2);
      const double p = (double)(P0 + i);
      for (int j = 0; j < kDHd / 2; ++j) {
        const double a = p * d_inv[j];
        cs[(size_t)r * kDHd + j] = (float)cos(a);
        cs[(size_t)r * kDHd + kDHd / 2 + j] = (float)sin(a);
      }
      sl[r] = (int32_t)((P0 + i) & (kDRing - 1));
      sq[r] = (int32_t)(P0 + i + 1);
    }
    lo_g[gi] = (int)std::max<int64_t>(S.d_base, P0 + gm - win);
    sub_all = sub_all && S.d_oov < 0.02f;
  }
  Seq& S = seqs[rq[0].seq];  // the reference attention path (one sequence)
  const int64_t P0 = S.pos;
  const int lo = lo_g[0];
  const int width = (int)(P0 + m) - lo;
  const int ps = std::max(32, (width + 63) / 64);
  // half * 2^-4 activations into the weight kernels, as the target does at 2+ rows
  static const bool xh_env = getenv("Q38_D_XH_OFF") == nullptr;
  const bool xh = xh_env && m >= 2;
  // reduced LM head unless a sequence has been leaving the draft vocabulary (Q38_D_FULLHEAD: never)
  static const bool full_head = getenv("Q38_D_FULLHEAD") != nullptr;
  const bool sub = sub_all && !full_head;
  const int hv = sub ? (int)d_vocab.size() : vocab;
  const DAttn ap = {m, d_heads, d_kvh, ps, (width + ps - 1) / ps, lo, kDRing - 1,
                    1.0f / sqrtf((float)kDHd), xh ? 1 : 0};
  const int qw = d_heads * kDHd, kvw = d_kvh * kDHd, qkvw = qw + 2 * kvw;
  // LM head rows: the draft rows only, packed. The last layer writes group g's
  // rows into d_hin starting one row before its first draft row (dr0[g]), last
  // group first, so each anchor row is overwritten by the previous group's
  // last draft row (the first anchor lands on spare row 0).
  int dr0[4], hn = 0;
  for (int gi = 0; gi < ng; ++gi) {
    dr0[gi] = hn;
    hn += rq[gi].k;
  }
  const bool head_h = hn >= 2;  // the LM head's wide kernels read half rows
  const size_t hb = (size_t)H * 2;
  double gpu = 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
    Enc e;
    e.eng = this;
    e.concurrent = !serial;
    e.enc = serial ? [cb computeCommandEncoder]
                   : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    for (int gi = 0; gi < ng; ++gi)
      if (n_ctx[gi]) d_emit_ctx(e, seqs[rq[gi].seq], taps[gi], n_ctx[gi], ctx_off[gi]);
    // `oh`: half * 2^-4 copy of the normed rows (input of the XH weight kernels)
    auto norm = [&](const Buf& xin, const Buf* resid, const Buf& w, const Buf& o, const Buf& rout,
                    const Buf& oh) {
      e.pipeline("q38_norm_add");
      e.in(xin, 0);
      e.in(resid ? *resid : xin, 1);
      e.in(w, 2);
      e.out(o, 3);
      e.out(rout, 4);
      e.bytes(H, 5);
      e.bytes(d_eps, 6);
      e.bytes((int)(resid != nullptr), 7);
      e.bytes(H, 8);
      e.out(oh, 9);
      e.bytes((int)xh, 10);
      e.in(w, 11);
      e.bytes((int)0, 12);
      e.dispatch(m, 1, 1, 512, 1, 1);
    };
    auto dense = [&](const Buf& o, const Buf& w, const Buf& xin, int N, int K, const Buf* r) {
      e.pipeline("q38_d_dense");
      e.out(o, 0);
      e.in(w, 1);
      e.in(xin, 2);
      e.bytes(N, 3);
      e.bytes(K, 4);
      e.in(r ? *r : xin, 5);
      e.bytes((int)(r != nullptr), 6);
      e.dispatch(N / 8, m, 1, 32, 1, 1);
    };
    norm(d_x, nullptr, dl[0].in_norm, d_in, d_res, d_inh);  // residual = x
    for (int l = 0; l < kDLayers; ++l) {
      DLayer& L = dl[l];
      const bool last = l == kDLayers - 1;
      emit_qproj(e, L.qkv, xh ? d_inh : d_in, d_qkv, m, nullptr, nullptr, xh);
      if (attn1) {
        for (int gi = 0; gi < ng; ++gi) {
          const Seq& G = seqs[rq[gi].seq];
          const size_t r0 = (size_t)row0[gi], gm = (size_t)rq[gi].k + 1;
          const struct {
            int heads, kv_heads, lo, ctx_hi, ring_mask, qkv_stride, cs_row0, out_half;
            float scale, eps;
          } a1 = {d_heads, d_kvh, lo_g[gi], (int)G.pos, kDRing - 1, qkvw, cs0 + (int)r0, xh ? 1 : 0, ap.scale, d_eps};
          e.pipeline("q38_d_attn1");
          e.in(rows(d_qkv, r0, gm, (size_t)qkvw * 2), 0);
          e.in(L.q_norm, 1);
          e.in(L.k_norm, 2);
          e.in(d_cs, 3);
          e.in(G.dk[l], 4);
          e.in(G.dv[l], 5);
          e.out(rows(d_ao, r0, gm, (size_t)qw * 2), 6);
          e.bytes(a1, 7);
          e.dispatch(d_kvh, (int)gm, 1, 256, 1, 1);
        }
      } else {
        const DQkv qp = {qkvw, d_heads, d_kvh, cs0, d_eps};
        e.pipeline("q38_d_qkv");
        e.in(d_qkv, 0);
        e.in(L.q_norm, 1);
        e.in(L.k_norm, 2);
        e.in(d_cs, 3);
        e.in(d_slot.at((size_t)cs0 * 4), 4);
        e.out(d_q, 5);
        e.out(S.dk[l], 6);
        e.out(S.dv[l], 7);
        e.bytes(qp, 8);
        e.dispatch(d_heads + 2 * d_kvh, m, 1, 32, 1, 1);
        e.pipeline("q38_d_attn_part");
        e.in(d_q, 0);
        e.in(S.dk[l], 1);
        e.in(S.dv[l], 2);
        e.in(d_seqlen, 3);
        e.out(d_pa, 4);
        e.out(d_pm, 5);
        e.bytes(ap, 6);
        e.dispatch(ap.n_parts, d_kvh, m, 32, 1, 1);
        e.pipeline("q38_d_attn_reduce");
        e.in(d_pa, 0);
        e.in(d_pm, 1);
        e.out(d_ao, 2);
        e.bytes(ap, 3);
        e.dispatch(d_heads, m, 1, 32, 1, 1);
      }
      emit_qproj(e, L.o, d_ao, d_t, m, nullptr, nullptr, xh);
      norm(d_t, &d_res, L.post_norm, d_h, d_res2, d_hh);  // d_res2 = residual + attention
      emit_gate_up(e, L.gate_up, xh ? d_hh : d_h, gu, mid, inter, m, nullptr, xh, xh);
      emit_qproj(e, L.down, mid, d_t2, m, nullptr, nullptr, xh);
      // code bridge + next norm
      dense(d_proj, L.read, d_t2, kDCode, H, &d_res2);
      for (int gi = 0; gi < ng; ++gi) {  // row 0 of a group is its anchor
        const size_t r0 = (size_t)row0[gi], gm = (size_t)rq[gi].k + 1;
        e.pipeline("q38_d_bridge_msg");
        e.in(rows(d_proj, r0, gm, (size_t)kDCode * 2), 0);
        e.in(d_tsv, 1);
        e.in(d_packed_w, 2);
        e.in(d_gate_b, 3);
        e.out(rows(d_packed, r0, gm, (size_t)kDRank * 2), 4);
        e.out(rows(d_msg, r0, gm, (size_t)kDRank * 2), 5);
        e.bytes(rq[gi].anchor, 6);
        e.bytes(d_beps, 7);
        e.dispatch(kDRank, (int)gm, 1, 32, 1, 1);
      }
      dense(d_delta, L.write, d_msg, H, kDRank, nullptr);
      for (int gi = ng - 1; gi >= 0; --gi) {
        const size_t r0 = (size_t)row0[gi], gm = (size_t)rq[gi].k + 1;
        e.pipeline("q38_d_bridge_write");
        e.in(rows(d_t2, r0, gm, hb), 0);
        e.in(rows(d_delta, r0, gm, hb), 1);
        e.in(rows(d_res2, r0, gm, hb), 2);
        e.in(last ? d_norm : dl[l + 1].in_norm, 3);
        e.out(rows(d_res, r0, gm, hb), 4);
        e.out(last ? rows(d_hin, (size_t)dr0[gi], gm, hb) : rows(d_in, r0, gm, hb), 5);
        e.bytes(H, 6);
        e.bytes(d_eps, 7);
        e.out(last ? rows(d_hinh, (size_t)dr0[gi], gm, hb) : rows(d_inh, r0, gm, hb), 8);
        e.bytes((int)(last ? head_h : xh), 9);
        e.out(rows(d_raw, r0, gm, hb), 10);
        e.out(rows(d_rms, r0, gm, 4), 11);
        e.dispatch((int)gm, 1, 1, 512, 1, 1);
      }
    }
    emit_qproj(e, sub ? d_head : lm_head, (head_h ? d_hinh : d_hin).at(hb), d_logits, hn, nullptr, nullptr,
               head_h);
    {  // C = raw @ W_last for the tail's norm, fp32 [m][512]
      e.pipeline("q38_d_dense_f");
      e.out(d_c, 0);
      e.in(d_write_t, 1);
      e.in(d_raw, 2);
      e.bytes((int)kDRank, 3);
      e.bytes(H, 4);
      e.dispatch(kDRank, m, 1, 32, 1, 1);
    }
    [e.enc endEncoding];
    const double te = now_ms();
    const double u1 = (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9;
    [cb commit];
    [cb waitUntilCompleted];
    const double u2 = (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9;
    if (getenv("Q38_TLOG"))
      fprintf(stderr, "tlog draft: encode %.3f  commit->start %.3f  gpu %.3f  end->wake %.3f ms\n", te - t0,
              (cb.GPUStartTime - u1) * 1e3, (cb.GPUEndTime - cb.GPUStartTime) * 1e3, (u2 - cb.GPUEndTime) * 1e3);
    if (getenv("Q38_DLOG"))
      fprintf(stderr, "dspine: ctx %d rows %d  encode %.2f ms  wait %.2f ms  gpu %.2f ms  kernel %.2f ms  disp %d barriers %d\n",
              coff, m, te - t0, now_ms() - te, (cb.GPUEndTime - cb.GPUStartTime) * 1e3,
              (cb.kernelEndTime - cb.kernelStartTime) * 1e3, e.dispatches, e.barriers);
    if (cb.status != MTLCommandBufferStatusCompleted) {
      error = cb.error ? cb.error.localizedDescription.UTF8String : "command buffer failed";
      return -4;
    }
    gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    if (stats) {
      stats->gpu_ms = gpu;
      stats->dispatches = e.dispatches;
      stats->barriers = e.barriers;
    }
    d_tpool = now_ms();
  }
  const double t1 = now_ms();
  if (getenv("Q38_TLOG")) fprintf(stderr, "tlog draft pool drain %.3f ms\n", t1 - d_tpool);

  // Sampling tail (propose.py sampled_table + serial_sample_chain), one row at
  // a time for the token actually drawn before it. With tsv = [G_p e | A e] of
  // the sender (e = 32 codebook[sender]) and g = sigmoid(q_row + G_p e + b) * A e:
  //   refined = raw + rms * (W g);  t_c = head_c . (refined / rms(refined) * w_norm)
  //           = (head_c . (raw * w_norm) + rms * (tg_c . g)) / rms(refined)
  //   |refined|^2 = H rms^2 + 2 rms (raw W) . g + rms^2 g' (W'W) g
  // so no product over the hidden width is left on the host but 16 head rows.
  static const float e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                 -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  static float e4m3[256];
  static bool e4m3_ok = false;
  if (!e4m3_ok) {
    for (int c = 0; c < 256; ++c) {
      const int ex = (c >> 3) & 15, mn = c & 7;
      const float v = ex ? ldexpf(1.0f + mn / 8.0f, ex - 7) : ldexpf(mn / 8.0f, -6);
      e4m3[c] = (c & 0x80) ? -v : v;
    }
    e4m3_ok = true;
  }
  const float gs = ((const float*)lm_head.gscale.ptr())[0];
  const uint8_t* hw = (const uint8_t*)lm_head.w.ptr();
  const uint8_t* hs = (const uint8_t*)lm_head.scale.ptr();
  const uint16_t* tsvb = (const uint16_t*)d_tsv.ptr();
  const uint16_t* tgb = (const uint16_t*)d_tg.ptr();
  const float* cmat = (const float*)d_c.ptr();
  const uint16_t* rawb = (const uint16_t*)d_raw.ptr();
  const uint16_t* pkb = (const uint16_t*)d_packed.ptr();
  const float* rms = (const float*)d_rms.ptr();
  const float inv_temp = 1.0f / (sp->temperature > 0 ? sp->temperature : 1.0f);
  const int top_k = std::max(1, std::min(sp->top_k > 0 ? sp->top_k : kDTopK, kDTopK));
  const float top_p = (sp->top_p > 0 && sp->top_p < 1) ? sp->top_p : 1.0f;
  std::vector<float> g(kDRank), Gg(kDRank), rw(H);
  // How many drafts to hand to the verifier. stop_below >= 0: stop once the
  // product of the drawn tokens' probabilities falls below it. stop_below < 0:
  // marginal rule at a rate of -stop_below tokens per ms: draft j is kept
  // while its estimated chance of being reached and accepted pays for the
  // verify row it adds (measured row costs: step time by rows plus 0.35 ms).
  // The acceptance estimate
  // 0.25 + 0.72 q is a fit to logged chains (perf notebook, DSpine entry).
  static const float row_ms[7] = {2.15f, 3.65f, 5.95f, 5.45f, 5.85f, 5.65f, 6.25f};  // 2026-11-17 kernels
  const bool marginal = stop_below < 0 && d_force.empty();
  const float lam = -stop_below;
  for (int gi = 0; gi < ng; ++gi) {
  const int32_t anchor = rq[gi].anchor;
  const int k = rq[gi].k;
  const double* u = rq[gi].u;
  q38_draft_row* out = rq[gi].out;
  int32_t sender = anchor;
  double conf = 1.0, reach = 1.0;
  int D = 0, keep = -1;
  for (int i = 0; i < k; ++i) {
    const int r = row0[gi] + i + 1;
    int32_t hk[kDTopK], cand[kDTopK];
    const double tp0 = now_ms();
    const int nc = topk_bf16((const uint16_t*)d_logits.ptr() + (size_t)(dr0[gi] + i) * hv, hv, kDTopK, hk, cand);
    if (nc < 1) break;
    if (sub)
      for (int c = 0; c < nc; ++c) cand[c] = d_vocab[cand[c]];
    std::sort(cand, cand + nc);  // ascending id, as the reference lays its lanes out
    const double tp1 = now_ms();
    const uint16_t* ts = tsvb + (size_t)sender * 2 * kDRank;
    float cross = 0;
    for (int j = 0; j < kDRank; ++j) {
      const float a = bf16_f32(pkb[(size_t)r * kDRank + j]) + bf16_f32(ts[j]) + d_gb[j];
      g[j] = bf16_f32(ts[kDRank + j]) / (1.0f + expf(-a));
      cross += cmat[(size_t)r * kDRank + j] * g[j];
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, kDRank, kDRank, 1.0f, d_G.data(), kDRank, g.data(), 1, 0.0f,
                Gg.data(), 1);
    const float quad = cblas_sdot(kDRank, g.data(), 1, Gg.data(), 1);
    const float ss = (float)H * rms[r] * rms[r] + 2.0f * rms[r] * cross + rms[r] * rms[r] * quad;
    for (int j = 0; j < H; ++j) rw[j] = bf16_f32(rawb[(size_t)r * H + j]) * d_nw[j];
    const float inv = 1.0f / sqrtf(std::max(ss, 0.0f) / H + d_eps);
    const double tp2 = now_ms();
    float t[kDTopK], tmax = -INFINITY;
    for (int c = 0; c < nc; ++c) {
      const uint8_t* wr = hw + (size_t)cand[c] * (H / 2);
      const uint8_t* sr = hs + (size_t)cand[c] * (H / 16);
      float acc = 0;
      for (int gI = 0; gI < H / 16; ++gI) {
        float a = 0;
        const float* xx = rw.data() + gI * 16;
        for (int b = 0; b < 8; ++b) {
          const uint8_t by = wr[gI * 8 + b];
          a += e2m1[by & 15] * xx[2 * b] + e2m1[by >> 4] * xx[2 * b + 1];
        }
        acc += a * e4m3[sr[gI]];
      }
      const uint16_t* tgr = tgb + (size_t)cand[c] * kDRank;
      float resp = 0;
      for (int j = 0; j < kDRank; ++j) resp += bf16_f32(tgr[j]) * g[j];
      t[c] = (acc * gs + rms[r] * resp) * inv * inv_temp;
      tmax = std::max(tmax, t[c]);
    }
    d_tp[0] += tp1 - tp0;
    d_tp[1] += tp2 - tp1;
    d_tp[2] += now_ms() - tp2;
    double p[kDTopK], psum = 0;
    for (int c = 0; c < nc; ++c) {
      p[c] = exp((double)(t[c] - tmax));
      psum += p[c];
    }
    // joint top-k / top-p (propose.py filter_probs): keep a lane when fewer than
    // top_k lanes are ahead of it and the mass ahead is at most top_p
    double w[kDTopK], wsum = 0;
    for (int c = 0; c < nc; ++c) {
      p[c] /= psum;
    }
    for (int c = 0; c < nc; ++c) {
      int rank = 0;
      double before = 0;
      for (int d = 0; d < nc; ++d)
        if (p[d] > p[c] || (p[d] == p[c] && d < c)) {
          ++rank;
          before += p[d];
        }
      w[c] = (rank < top_k && before <= top_p) ? p[c] : 0.0;
      wsum += w[c];
    }
    q38_draft_row& R = out[D];
    R.n = 0;
    double cdf = 0;
    int pick = -1, lastk = 0;
    const double uu = u[i] * 1.0;
    for (int c = 0; c < nc; ++c) {
      if (w[c] <= 0) continue;
      const double qc = w[c] / wsum;
      R.ids[R.n] = cand[c];
      R.probs[R.n] = (float)qc;
      cdf += qc;
      if (pick < 0 && uu < cdf) pick = R.n;
      lastk = R.n;
      ++R.n;
    }
    if (pick < 0) pick = lastk;
    R.tok = R.ids[pick];
    if ((int)d_force.size() > i) {  // forced chain: report q as is, continue from the given token
      R.tok = d_force[i];
      sender = R.tok;
      ++D;
      continue;
    }
    sender = R.tok;
    conf *= R.probs[pick];
    ++D;
    if (marginal) {
      reach *= 0.25 + 0.72 * R.probs[pick];
      if (reach < lam * row_ms[i]) {
        keep = std::max(i, 1);
        break;
      }
      continue;
    }
    if (conf < stop_below) break;
  }
  if (keep >= 0) D = std::min(D, keep);
  rq[gi].D = D;
  }
  d_force.clear();
  const double t2 = now_ms();
  d_gpu_ms += gpu;
  d_tail_ms += t2 - t1;
  d_wall_ms += t2 - t0;
  ++d_calls;
  if (stats) {
    stats->encode_ms = t2 - t1;  // host sampling tail
    stats->wall_ms = t2 - t0;
  }
  return 0;
}

}  // namespace q38

using q38::Engine;
struct q38_engine {
  Engine e;
};

q38_engine* q38_create(const char* pack_path, const char* metallibs, const q38_config* cfg,
                       char* err, int err_len) {
  @autoreleasepool {
    q38_engine* h = new q38_engine();
    Engine& e = h->e;
    auto bail = [&](const std::string& why) -> q38_engine* {
      if (err && err_len > 0) snprintf(err, err_len, "%s", why.c_str());
      delete h;
      return nullptr;
    };
    if (cfg) {
      if (cfg->max_ctx > 0) e.max_ctx = cfg->max_ctx;
      if (cfg->max_rows > 0) e.max_rows = cfg->max_rows;
      if (cfg->block_size > 0) e.block_size = cfg->block_size;
      e.serial = cfg->serial != 0;
    }
    if (e.max_rows > 8) return bail("max_rows above 8 needs the GEMM path (not built yet)");
    e.dev = MTLCreateSystemDefaultDevice();
    if (!e.dev) return bail("no Metal device");
    e.queue = [e.dev newCommandQueue];
    std::string libs = metallibs ? metallibs : "";
    size_t start = 0;
    while (start <= libs.size()) {
      size_t end = libs.find(':', start);
      if (end == std::string::npos) end = libs.size();
      const std::string p = libs.substr(start, end - start);
      if (!p.empty()) {
        NSError* nerr = nil;
        id<MTLLibrary> lib =
            [e.dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]]
                               error:&nerr];
        if (!lib) return bail("cannot load metallib " + p + ": " + nerr.localizedDescription.UTF8String);
        e.libs.push_back(lib);
      }
      start = end + 1;
    }
    if (e.libs.empty()) return bail("no metallib given");
    if (!e.open_pack(pack_path) || !e.build()) return bail(e.error);
    return h;
  }
}

void q38_destroy(q38_engine* h) {
  if (!h) return;
  // every no-copy MTLBuffer over the mapping must be gone before the unmap
  void* map = h->e.map;
  const size_t map_len = h->e.map_len;
  const int fd = h->e.fd;
  @autoreleasepool {
    delete h;
  }
  if (map && map != MAP_FAILED) munmap(map, map_len);
  if (fd >= 0) close(fd);
}

struct q38_snapshot {
  int64_t pos = 0, mtp_pos = 0, rope_off = 0;
  std::vector<std::vector<float>> ssm;    // per GDN layer: [Hv][Dv][Dk]
  std::vector<std::vector<float>> conv;   // per GDN layer: [channels][kernel - 1]
  std::vector<std::vector<uint8_t>> kv;   // per attention layer: K rows then V rows
  std::vector<uint8_t> mtp_kv;
  // DSpine drafter context: the last d_w positions of every layer's K then V rows
  std::vector<uint8_t> d_kv;
  int64_t d_w = 0;
  float d_oov = 0;
  std::vector<uint16_t> last_logits;  // logits of the last kept row (what follows the snapshot)
  std::vector<uint16_t> tail;  // final hidden rows of the last step's kept rows (MTP pairs not fed yet)
  int tail_m = 0;
  int64_t bytes = 0;
};

static bool gdn_fused_on() {
  static const bool on = getenv("Q38_GDN_OFF") == nullptr;
  return on;
}

q38_snapshot* Engine::snapshot_save(Seq& S_) {
  if (has_dspine && S_.d_pend && !S_.last_m && d_flush(S_)) return nullptr;
  const Seq& Q = S_;
  if (Q.last_m || !gdn_fused_on()) return nullptr;
  auto* sn = new q38_snapshot;
  sn->pos = Q.pos;
  sn->mtp_pos = Q.mtp_pos;
  sn->d_oov = Q.d_oov;
  sn->rope_off = Q.rope_off;
  const int hist = conv_kernel - 1;
  const int ro = Q.accepted - 1;
  const int nrep = Q.gdn_log_n + (Q.gdn_prev_m ? Q.accepted : 0);
  const size_t kv_row = (size_t)kv_heads * head_dim * 2;
  for (int l = 0; l < num_layers; ++l) {
    const SeqLayer& L = Q.L[l];
    if (!layers[l].is_gdn) {
      const size_t pos = (size_t)Q.pos;
      std::vector<uint8_t> b(2 * kv_row * pos);
      if (pos) {
        memcpy(b.data(), L.kc.ptr(), kv_row * pos);
        memcpy(b.data() + kv_row * pos, L.vc.ptr(), kv_row * pos);
      }
      sn->bytes += (int64_t)b.size();
      sn->kv.push_back(std::move(b));
      continue;
    }
    // base state with the committed log entries applied (kernel phase B replay)
    std::vector<float> S((size_t)ssm_stride);
    memcpy(S.data(), (const float*)L.ssm_state.ptr() + ssm_stride, (size_t)ssm_stride * 4);
    const float* lg = (const float*)L.gdn_log.ptr() + (size_t)Q.gdn_log_cur * kGdnLogCap * gdn_log_stride;
    const int vpk = gHv / gHk;
    for (int en = 0; en < nrep; ++en) {
      const float* E = lg + (size_t)en * gdn_log_stride;
      const float* kall = E;
      const float* dall = E + (size_t)gHk * gDk;
      const float* gall = E + (size_t)(gHk + gHv) * gDk;
      for (int hv = 0; hv < gHv; ++hv) {
        const float g = gall[hv];
        const float* kk = kall + (size_t)(hv / vpk) * gDk;
        const float* dd = dall + (size_t)hv * gDv;
        float* sh = S.data() + (size_t)hv * gDv * gDk;
        for (int dv = 0; dv < gDv; ++dv) {
          const float d = dd[dv];
          float* row = sh + (size_t)dv * gDk;
          // same rounding as the GPU replay (measured: this FMA form is bit-exact)
          for (int i = 0; i < gDk; ++i) row[i] = fmaf(kk[i], d, row[i] * g);
        }
      }
    }
    std::vector<float> C((size_t)gdn_qkv_ch * hist);
    const float* rd = (const float*)L.conv_state.ptr() + (size_t)Q.conv_cur * conv_stride;
    for (int c = 0; c < gdn_qkv_ch; ++c)
      for (int j = 0; j < hist; ++j) C[(size_t)c * hist + j] = rd[(size_t)c * conv_cols + ro + j];
    sn->bytes += (int64_t)(S.size() + C.size()) * 4;
    sn->ssm.push_back(std::move(S));
    sn->conv.push_back(std::move(C));
  }
  sn->tail_m = Q.gdn_prev_m ? Q.accepted : 0;
  if (sn->tail_m) {
    const uint16_t* tr = hid_row(Q, 0);
    sn->tail.assign(tr, tr + (size_t)sn->tail_m * hidden);
    const uint16_t* lr = logits_row(Q, sn->tail_m - 1);
    sn->last_logits.assign(lr, lr + vocab);
  }
  sn->bytes += (int64_t)(sn->tail.size() + sn->last_logits.size()) * 2;
  if (has_mtp && Q.mtp_pos) {
    const size_t mp = (size_t)Q.mtp_pos;
    sn->mtp_kv.resize(2 * kv_row * mp);
    memcpy(sn->mtp_kv.data(), Q.mkc.ptr(), kv_row * mp);
    memcpy(sn->mtp_kv.data() + kv_row * mp, Q.mvc.ptr(), kv_row * mp);
    sn->bytes += (int64_t)sn->mtp_kv.size();
  }
  if (has_dspine && Q.d_n == Q.pos && Q.d_n > Q.d_base) {
    const size_t drow = (size_t)d_kvh * kDHd * 2;
    const int64_t W = std::min(Q.d_n - Q.d_base, d_win());
    sn->d_w = W;
    sn->d_kv.resize((size_t)kDLayers * 2 * W * drow);
    uint8_t* o = sn->d_kv.data();
    for (int l = 0; l < kDLayers; ++l)
      for (int which = 0; which < 2; ++which) {
        const uint8_t* src = (const uint8_t*)(which ? Q.dv[l] : Q.dk[l]).ptr();
        for (int64_t i = 0; i < W; ++i, o += drow)
          memcpy(o, src + (size_t)((Q.d_n - W + i) & (kDRing - 1)) * drow, drow);
      }
    sn->bytes += (int64_t)sn->d_kv.size();
  }
  return sn;
}

int Engine::snapshot_load(Seq& Q, const q38_snapshot* sn, bool global) {
  if (!sn || !gdn_fused_on() || sn->pos > max_ctx) return -1;
  const int hist = conv_kernel - 1;
  size_t gi = 0, ai = 0;
  Q.pos = 0;  // nothing to carry over when the K/V buffers grow
  Q.last_m = 0;
  ensure_kv(Q, std::max<int64_t>(sn->pos, 1));
  for (int l = 0; l < num_layers; ++l) {
    SeqLayer& L = Q.L[l];
    if (!layers[l].is_gdn) {
      if (ai >= sn->kv.size()) return -2;
      const std::vector<uint8_t>& b = sn->kv[ai++];
      memcpy(L.kc.ptr(), b.data(), b.size() / 2);
      memcpy(L.vc.ptr(), b.data() + b.size() / 2, b.size() / 2);
      continue;
    }
    if (gi >= sn->ssm.size()) return -2;
    memcpy((float*)L.ssm_state.ptr() + ssm_stride, sn->ssm[gi].data(), (size_t)ssm_stride * 4);
    float* wr = (float*)L.conv_state.ptr() + (size_t)Q.conv_cur * conv_stride;
    const std::vector<float>& C = sn->conv[gi++];
    for (int c = 0; c < gdn_qkv_ch; ++c)
      for (int j = 0; j < hist; ++j) wr[(size_t)c * conv_cols + j] = C[(size_t)c * hist + j];
  }
  Q.mtp_pos = 0;
  Q.mtp_chain = 0;
  if (has_mtp && !sn->mtp_kv.empty()) {
    ensure_mtp_kv(Q, sn->mtp_pos);
    memcpy(Q.mkc.ptr(), sn->mtp_kv.data(), sn->mtp_kv.size() / 2);
    memcpy(Q.mvc.ptr(), sn->mtp_kv.data() + sn->mtp_kv.size() / 2, sn->mtp_kv.size() / 2);
  }
  if (global) {
    // single-stream API: the tail rows go back where a step would have left them
    memcpy(x.ptr(), sn->tail.data(), sn->tail.size() * 2);
    if (sn->tail_m)
      memcpy((uint16_t*)logits.ptr() + (size_t)(sn->tail_m - 1) * vocab, sn->last_logits.data(),
             (size_t)vocab * 2);
    Q.restored = false;
    Q.out_row0 = 0;
  } else {
    Q.r_tail = sn->tail;
    Q.r_logits = sn->last_logits;
    Q.restored = true;
  }
  Q.out_m = sn->tail_m;
  Q.pos = sn->pos;
  Q.rope_off = Q.rope_off_next = sn->rope_off;
  Q.last_m = 0;
  Q.accepted = 1;  // conv history sits at columns 0.., as after a reset
  Q.gdn_log_n = Q.gdn_prev_m = 0;
  Q.mtp_pos = has_mtp ? sn->mtp_pos : 0;
  Q.mtp_chain = 0;
  // the drafter's context: the window the snapshot carries, or none
  Q.d_n = Q.d_base = Q.pos;
  Q.d_pend = Q.d_stash_n = 0;
  Q.d_taps_ok = false;
  Q.d_oov = sn->d_oov;
  if (has_dspine && sn->d_w > 0 && sn->d_kv.size() == (size_t)kDLayers * 2 * sn->d_w * d_kvh * kDHd * 2) {
    const size_t drow = (size_t)d_kvh * kDHd * 2;
    const int64_t W = sn->d_w;
    Q.d_n = 0;  // nothing to carry over when the rings grow
    d_ensure(Q, Q.pos + 16);
    const uint8_t* o = sn->d_kv.data();
    for (int l = 0; l < kDLayers; ++l)
      for (int which = 0; which < 2; ++which) {
        uint8_t* dst = (uint8_t*)(which ? Q.dv[l] : Q.dk[l]).ptr();
        for (int64_t i = 0; i < W; ++i, o += drow)
          memcpy(dst + (size_t)((Q.pos - W + i) & (kDRing - 1)) * drow, o, drow);
      }
    Q.d_n = Q.pos;
    Q.d_base = Q.pos - W;
  }
  return 0;
}

void q38_reset(q38_engine* h) { h->e.seq_reset(h->e.seqs[0]); }
q38_snapshot* q38_snapshot_save(q38_engine* h) { return h->e.snapshot_save(h->e.seqs[0]); }
int q38_snapshot_load(q38_engine* h, const q38_snapshot* s) {
  return h->e.snapshot_load(h->e.seqs[0], s, true);
}

// ---- several sequences ----
int q38_seq_new(q38_engine* h) { return h->e.seq_new(); }
void q38_seq_free(q38_engine* h, int seq) { h->e.seq_free(seq); }
int q38_seq_reset(q38_engine* h, int seq) {
  if (!h->e.seq_ok(seq)) return -1;
  h->e.seq_reset(h->e.seqs[seq]);
  return 0;
}
int q38_seq_out_rows(const q38_engine* h, int seq) {
  return h->e.seq_ok(seq) ? h->e.seqs[seq].out_m : -1;
}
int q38_big_rows(const q38_engine*) { return Engine::kBigRows; }
int64_t q38_seq_position(const q38_engine* h, int seq) {
  return h->e.seq_ok(seq) ? h->e.seqs[seq].pos : -1;
}
int q38_step_groups(q38_engine* h, const q38_group* g, int n, const int32_t* tokens,
                    const uint16_t* embeds, const int32_t* pos3, q38_step_stats* stats) {
  if (n < 1 || n > 16) return -1;
  q38::Group grp[16];
  for (int i = 0; i < n; ++i) grp[i] = {g[i].seq, g[i].m, 0};
  return h->e.step(grp, n, tokens, embeds, pos3, stats);
}
int q38_seq_commit(q38_engine* h, int seq, int n) {
  Engine& e = h->e;
  if (!e.seq_ok(seq)) return -1;
  q38::Seq& S = e.seqs[seq];
  if (!S.last_m || n < 1 || n > S.last_m) return -1;
  if (S.big_last && n != S.last_m) return -1;  // a big step is kept whole
  e.commit_rows(S, n);
  return 0;
}
q38_snapshot* q38_seq_snapshot_save(q38_engine* h, int seq) {
  return h->e.seq_ok(seq) ? h->e.snapshot_save(h->e.seqs[seq]) : nullptr;
}
int q38_seq_snapshot_load(q38_engine* h, int seq, const q38_snapshot* s) {
  return h->e.seq_ok(seq) ? h->e.snapshot_load(h->e.seqs[seq], s, false) : -1;
}
int64_t q38_snapshot_position(const q38_snapshot* s) { return s->pos; }
int64_t q38_snapshot_bytes(const q38_snapshot* s) { return s->bytes; }
void q38_snapshot_free(q38_snapshot* s) { delete s; }
int q38_snapshot_tail_rows(const q38_snapshot* s) { return s->tail_m; }
int q38_has_mtp(const q38_engine* h) { return h->e.has_mtp ? 1 : 0; }

int q38_step(q38_engine* h, const int32_t* tokens, int m, q38_step_stats* stats) {
  const q38::Group g{0, m, 0};
  return h->e.step(&g, 1, tokens, nullptr, nullptr, stats);
}

int q38_step_mm(q38_engine* h, const int32_t* tokens, const uint16_t* embeds, const int32_t* pos3,
                int m, q38_step_stats* stats) {
  const q38::Group g{0, m, 0};
  return h->e.step(&g, 1, tokens, embeds, pos3, stats);
}

int q38_commit(q38_engine* h, int n) { return q38_seq_commit(h, 0, n); }

int64_t q38_position(const q38_engine* h) { return h->e.seqs[0].pos; }
int q38_vocab(const q38_engine* h) { return h->e.vocab; }
int q38_hidden(const q38_engine* h) { return h->e.hidden; }
const int64_t* q38_argmax(const q38_engine* h) { return (const int64_t*)h->e.argmax.ptr(); }
const uint16_t* q38_logits_bf16(const q38_engine* h) { return (const uint16_t*)h->e.logits.ptr(); }
const uint16_t* q38_final_bf16(const q38_engine* h) { return (const uint16_t*)h->e.x.ptr(); }
static int32_t sample_row(const uint16_t* lg, int vocab, const q38_sampling* sp, double u,
                          int32_t* ids, float* probs, int* n_out) {
  const int k = std::max(1, std::min(sp->top_k > 0 ? sp->top_k : 64, 64));
  // top-k by a threshold scan: BF16 bit patterns order like floats once the
  // sign is folded, so compare as int32 keys and only touch the heap on a hit.
  auto key = [](uint16_t v) -> int32_t { return (v & 0x8000) ? -(int32_t)(v & 0x7FFF) : (int32_t)v; };
  int32_t hk[64];
  int32_t hi[64];
  int cnt = 0;
  int32_t thr = INT32_MIN;
  auto consider = [&](int i) {
    const int32_t kv = key(lg[i]);
    if (cnt == k && kv <= thr) return;
    if ((lg[i] & 0x7F80) == 0x7F80) return;  // inf / nan
    int pos = cnt < k ? cnt++ : k - 1;
    while (pos > 0 && hk[pos - 1] < kv) {
      hk[pos] = hk[pos - 1];
      hi[pos] = hi[pos - 1];
      --pos;
    }
    hk[pos] = kv;
    hi[pos] = i;
    if (cnt == k) thr = hk[k - 1];
  };
  int i = 0;
  for (; i < vocab && cnt < k; ++i) consider(i);
  // Once the heap is full almost nothing beats the threshold: test 16 logits
  // at a time and fall to the scalar path only for a block with a candidate.
  for (; i + 16 <= vocab; i += 16) {
    const int16x8_t t16 = vdupq_n_s16((int16_t)std::max(thr, (int32_t)INT16_MIN));
    uint16_t any = 0;
    for (int h8 = 0; h8 < 2; ++h8) {
      const int16x8_t v = vreinterpretq_s16_u16(vld1q_u16(lg + i + 8 * h8));
      const int16x8_t neg = vshrq_n_s16(v, 15);
      const int16x8_t mag = vandq_s16(v, vdupq_n_s16(0x7FFF));
      const int16x8_t kv = vsubq_s16(veorq_s16(mag, neg), neg);
      any |= vmaxvq_u16(vcgtq_s16(kv, t16));
    }
    if (any)
      for (int j = i; j < i + 16; ++j) consider(j);
  }
  for (; i < vocab; ++i) consider(i);
  auto f = [](uint16_t v) { uint32_t b = (uint32_t)v << 16; float x; memcpy(&x, &b, 4); return x; };
  const float temp = sp->temperature > 0 ? sp->temperature : 1.0f;
  double p[64], sum = 0;
  const float top = f(lg[hi[0]]);
  for (int i = 0; i < cnt; ++i) {
    p[i] = exp((double)(f(lg[hi[i]]) - top) / temp);
    sum += p[i];
  }
  // top-p over the top-k renormalized distribution (keep the token that crosses)
  int keep = cnt;
  if (sp->top_p > 0 && sp->top_p < 1) {
    double acc = 0;
    for (int i = 0; i < cnt; ++i) {
      acc += p[i] / sum;
      if (acc >= sp->top_p) { keep = i + 1; break; }
    }
  }
  double ksum = 0;
  for (int i = 0; i < keep; ++i) ksum += p[i];
  double acc = 0;
  int pick = keep - 1;
  for (int i = 0; i < keep; ++i) {
    const double pi = p[i] / ksum;
    if (ids) ids[i] = hi[i];
    if (probs) probs[i] = (float)pi;
    acc += pi;
    if (pick == keep - 1 && u < acc) pick = i;
  }
  if (n_out) *n_out = keep;
  return hi[pick];
}

int32_t q38_sample(const q38_engine* h, int row, const q38_sampling* sp, double u,
                   int32_t* ids, float* probs, int* n_out) {
  return q38_seq_sample(h, 0, row, sp, u, ids, probs, n_out);
}

int32_t q38_seq_sample(const q38_engine* h, int seq, int row, const q38_sampling* sp, double u,
                       int32_t* ids, float* probs, int* n_out) {
  const Engine& e = h->e;
  return sample_row(e.logits_row(e.seqs[seq], row), e.vocab, sp, u, ids, probs, n_out);
}

int32_t q38_mtp_sample(const q38_engine* h, const q38_sampling* sp, double u, int32_t* ids,
                       float* probs, int* n_out) {
  return q38_seq_mtp_sample(h, 0, sp, u, ids, probs, n_out);
}

int32_t q38_seq_mtp_sample(const q38_engine* h, int seq, const q38_sampling* sp, double u,
                           int32_t* ids, float* probs, int* n_out) {
  const Engine& e = h->e;
  return sample_row((const uint16_t*)e.m_logits.ptr() + (size_t)e.seqs[seq].mtp_row * e.vocab,
                    e.vocab, sp, u, ids, probs, n_out);
}

int q38_load_dspine(q38_engine* h, const char* path, char* err, int err_len) {
  if (h->e.load_dspine(path)) return 0;
  if (err && err_len > 0) snprintf(err, err_len, "%s", h->e.error.c_str());
  return -1;
}
int q38_has_dspine(const q38_engine* h) { return h->e.has_dspine ? 1 : 0; }
int q38_seq_dspine_draft(q38_engine* h, int seq, int32_t anchor, int k, const q38_sampling* s,
                         float stop_below, const double* u, q38_draft_row* out,
                         q38_step_stats* stats) {
  return h->e.dspine_draft(seq, anchor, k, s, stop_below, u, out, stats);
}
int q38_dspine_draft_groups(q38_engine* h, const int* seqs, const int32_t* anchors, const int* k, int n,
                            const q38_sampling* s, float stop_below, const double* u, q38_draft_row* out,
                            int* n_out, q38_step_stats* stats) {
  if (n < 1 || n > 4) return -1;
  Engine::DReq rq[4];
  for (int i = 0; i < n; ++i) rq[i] = {seqs[i], anchors[i], k[i], u + 8 * i, out + 8 * i, 0};
  const int rc = h->e.dspine_draft_groups(rq, n, s, stop_below, stats);
  if (rc < 0) return rc;
  for (int i = 0; i < n; ++i) n_out[i] = rq[i].D;
  return 0;
}
void q38_dspine_times(const q38_engine* h, double* gpu_ms, double* tail_ms, double* wall_ms,
                      int64_t* calls) {
  if (gpu_ms) *gpu_ms = h->e.d_gpu_ms;
  if (tail_ms) *tail_ms = h->e.d_tail_ms;
  if (wall_ms) *wall_ms = h->e.d_wall_ms;
  if (calls) *calls = h->e.d_calls;
  if (getenv("Q38_DLOG"))
    fprintf(stderr, "dspine tail per call: top-k %.3f ms, bridge products %.3f ms, head rows %.3f ms\n",
            h->e.d_tp[0] / h->e.d_calls, h->e.d_tp[1] / h->e.d_calls, h->e.d_tp[2] / h->e.d_calls);
}
double q38_dspine_read_gb(const q38_engine* h) { return h->e.d_read_gb; }
void q38_dspine_force(q38_engine* h, const int32_t* tokens, int n) { h->e.d_force.assign(tokens, tokens + n); }

int q38_load_mtp(q38_engine* h, const char* path, char* err, int err_len) {
  if (h->e.load_mtp(path)) return 0;
  if (err && err_len > 0) snprintf(err, err_len, "%s", h->e.error.c_str());
  return -1;
}

int q38_mtp_feed(q38_engine* h, const int32_t* tokens, const int* target_rows, int m, int head,
                 q38_step_stats* stats) {
  return q38_mtp_feed_mm(h, tokens, nullptr, target_rows, m, head, stats);
}

int q38_mtp_feed_mm(q38_engine* h, const int32_t* tokens, const uint16_t* embeds,
                    const int* target_rows, int m, int head, q38_step_stats* stats) {
  const q38_group g{0, m};
  return q38_mtp_feed_groups(h, &g, 1, tokens, embeds, target_rows, head, stats);
}

int q38_mtp_feed_groups(q38_engine* h, const q38_group* g, int n, const int32_t* tokens,
                        const uint16_t* embeds, const int* target_rows, int head,
                        q38_step_stats* stats) {
  Engine& e = h->e;
  if (n < 1 || n > 16) return -1;
  const uint16_t* hid[64];
  q38::Group grp[16];
  int m = 0;
  for (int i = 0; i < n; ++i) {
    if (!e.seq_ok(g[i].seq) || g[i].m < 1 || m + g[i].m > e.max_rows) return -1;
    const q38::Seq& S = e.seqs[g[i].seq];
    for (int j = 0; j < g[i].m; ++j) {
      const int r = target_rows[m + j];
      if (r < 0 || r >= S.out_m) return -5;
      hid[m + j] = e.hid_row(S, r);
    }
    grp[i] = {g[i].seq, g[i].m, 0};
    m += g[i].m;
  }
  return e.mtp_run(grp, n, tokens, embeds, hid, true, head != 0, stats);
}

int q38_mtp_feed_big(q38_engine* h, int seq, int first_row, int n, const int32_t* tokens,
                     const uint16_t* embeds, q38_step_stats* stats) {
  Engine& e = h->e;
  if (!e.seq_ok(seq) || n < 1 || n > e.max_rows) return -1;
  const q38::Seq& S = e.seqs[seq];
  if (S.restored || first_row < 0 || first_row + n > S.big_m) return -5;
  const uint16_t* hid[16];
  for (int j = 0; j < n; ++j) hid[j] = (const uint16_t*)e.x.ptr() + (size_t)(first_row + j) * e.hidden;
  q38::Group grp = {seq, n, 0};
  return e.mtp_run(&grp, 1, tokens, embeds, hid, true, false, stats);
}

int q38_mtp_chain(q38_engine* h, int32_t token, q38_step_stats* stats) {
  const int seq = 0;
  return q38_mtp_chain_groups(h, &seq, &token, 1, stats);
}

int q38_mtp_chain_groups(q38_engine* h, const int* seqs, const int32_t* tokens, int n,
                         q38_step_stats* stats) {
  Engine& e = h->e;
  if (n < 1 || n > e.max_rows) return -1;
  const uint16_t* hid[16];
  q38::Group grp[16];
  for (int i = 0; i < n; ++i) {
    if (!e.seq_ok(seqs[i])) return -1;
    // the sequence's previous MTP output row (post mtp.norm) is the next step's hidden
    hid[i] = (const uint16_t*)e.m_out.ptr() + (size_t)e.seqs[seqs[i]].mtp_row * e.hidden;
    grp[i] = {seqs[i], 1, 0};
  }
  return e.mtp_run(grp, n, tokens, nullptr, hid, false, true, stats);
}

double q38_mtp_read_gb(const q38_engine* h) { return h->e.mtp_read_gb; }

void q38_set_argmax(q38_engine* h, int on) { h->e.want_argmax = on != 0; }

void q38_set_layer_cap(q38_engine* h, int cap) { h->e.layer_cap = cap; }
double q38_decode_read_gb(const q38_engine* h) { return h->e.read_gb; }
