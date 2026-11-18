// q38 generator (include/q38/generate.h). Plain C++ over the engine C ABI.
//
// Speculative decode is lossless against the target's filtered distribution:
// each draft d ~ q is kept with probability min(1, p(d) / q(d)); on the first
// rejection the token is redrawn from the residual (p - q)+, and when every
// draft is kept one more token comes from the last verify row.
//
// Invariants:
//   hist      tokens the engine has committed (engine position == hist.size())
//   pending   the next token of the sequence, already drawn but neither fed
//             to the engine nor returned to the caller
//   unfed     rows of the last target step whose (hidden, following token)
//             pair the MTP head has not seen; the last row's follower is the
//             next token to be fed, known only when that step is built
#include "q38/generate.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Snap {
  std::vector<int32_t> toks;
  q38_snapshot* s = nullptr;
  int unfed_n = 0;
  int32_t unfed_tok[8] = {0};
  std::vector<uint16_t> unfed_emb;
  uint64_t stamp = 0;
};

}  // namespace

struct q38_gen {
  q38_engine* e = nullptr;
  q38_gen_config c{};
  uint64_t rs = 0;
  bool mtp = false;
  bool dsp = false;  // DSpine block drafter instead of the MTP head
  float dsp_ema = 2.f;  // drafts handed to the verifier lately: sizes the next block
  std::vector<int32_t> hist;
  int32_t pending = -1;
  bool done = true;
  int unfed_n = 0;
  int32_t unfed_tok[8] = {0};  // follower of row i of the last step (last one filled at flush)
  // Embedding of follower i when unfed_tok[i] < 0 (an image row), [8][hidden].
  std::vector<uint16_t> unfed_emb;
  int hidden = 0;
  void set_emb(int row, const uint16_t* e) {
    if (unfed_emb.empty()) unfed_emb.resize((size_t)8 * hidden);
    memcpy(unfed_emb.data() + (size_t)row * hidden, e, (size_t)hidden * 2);
  }
  std::vector<Snap> snaps;
  uint64_t clock = 0;
  q38_gen_stats st{};

  double uni() {  // splitmix64
    rs += 0x9E3779B97F4A7C15ull;
    uint64_t z = rs;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0;
  }
  bool is_eos(int32_t t) const {
    for (int i = 0; i < c.n_eos; ++i)
      if (c.eos[i] == t) return true;
    return false;
  }
  // Feed the pairs of the last step to the MTP head; `next` follows its last row.
  int flush(int32_t next, bool head, const uint16_t* next_emb = nullptr) {
    if (!mtp || unfed_n == 0) return 0;
    int rows[8];
    for (int i = 0; i < unfed_n; ++i) rows[i] = i;
    unfed_tok[unfed_n - 1] = next;
    if (next < 0) set_emb(unfed_n - 1, next_emb);
    q38_step_stats ms;
    const int rc = q38_mtp_feed_mm(e, unfed_tok, unfed_emb.empty() ? nullptr : unfed_emb.data(), rows,
                                   unfed_n, head ? 1 : 0, &ms);
    st.draft_ms += ms.wall_ms;
    unfed_n = 0;
    return rc;
  }
  // The engine committed rows `t[0..m)`; their followers are t[1..m) and one unknown.
  // `emb` ([m][hidden]) holds the embeddings of the rows with a negative token.
  void committed(const int32_t* t, int m, const uint16_t* emb = nullptr) {
    hist.insert(hist.end(), t, t + m);
    unfed_n = m;
    for (int i = 0; i + 1 < m; ++i) {
      unfed_tok[i] = t[i + 1];
      if (t[i + 1] < 0) set_emb(i, emb + (size_t)(i + 1) * hidden);
    }
  }
  int64_t snap_bytes() const {
    int64_t b = 0;
    for (const Snap& s : snaps)
      b += q38_snapshot_bytes(s.s) + (int64_t)s.toks.size() * 4 + (int64_t)s.unfed_emb.size() * 2;
    return b;
  }
  void store_snapshot() {
    for (Snap& s : snaps)
      if (s.toks == hist) {  // already have this prefix
        s.stamp = ++clock;
        return;
      }
    q38_snapshot* sn = q38_snapshot_save(e);
    if (!sn) return;
    Snap s;
    s.toks = hist;
    s.s = sn;
    s.unfed_n = unfed_n;
    memcpy(s.unfed_tok, unfed_tok, sizeof unfed_tok);
    for (int i = 0; i + 1 < unfed_n; ++i)
      if (unfed_tok[i] < 0) s.unfed_emb = unfed_emb;
    s.stamp = ++clock;
    snaps.push_back(std::move(s));
    while (snaps.size() > 1 && snap_bytes() > c.max_snapshot_bytes) {
      size_t old = 0;
      for (size_t i = 1; i < snaps.size(); ++i)
        if (snaps[i].stamp < snaps[old].stamp) old = i;
      q38_snapshot_free(snaps[old].s);
      snaps.erase(snaps.begin() + (long)old);
    }
  }
};

void q38_gen_config_default(q38_gen_config* c) {
  memset(c, 0, sizeof *c);
  c->sampling = {1.0f, 20, 0.95f};
  c->draft = {1.0f, 20, 0.95f};
  c->depth = 2;
  c->adaptive = 0.5f;
  c->chunk = 8;
  c->seed = 42;
  c->eos[0] = 248046;
  c->eos[1] = 248044;
  c->n_eos = 2;
  c->snapshot_prompt_end = 1;
  c->max_snapshot_bytes = 16ll << 30;
}

q38_gen* q38_gen_create(q38_engine* e, const q38_gen_config* c) {
  if (!e || !c || c->chunk < 1 || c->chunk > 8 || c->depth < 0 || c->depth > 7 || c->n_eos < 0 ||
      c->n_eos > 4)
    return nullptr;
  if (c->depth > 0 && !q38_has_mtp(e) && !q38_has_dspine(e)) return nullptr;
  q38_gen* g = new q38_gen;
  g->e = e;
  g->c = *c;
  g->rs = c->seed;
  g->dsp = c->depth > 0 && q38_has_dspine(e);  // preferred when both are loaded
  g->mtp = c->depth > 0 && !g->dsp;
  g->hidden = q38_hidden(e);
  q38_set_argmax(e, 0);
  q38_reset(e);
  return g;
}

void q38_gen_free(q38_gen* g) {
  if (!g) return;
  for (Snap& s : g->snaps) q38_snapshot_free(s.s);
  delete g;
}

void q38_gen_reseed(q38_gen* g, uint64_t seed) { g->rs = seed; }

void q38_gen_set_sampling(q38_gen* g, const q38_sampling* s) {
  q38_sampling v = *s;
  if (v.temperature <= 0.f) {  // "greedy": one candidate
    v.temperature = 1.f;
    v.top_k = 1;
  }
  v.top_k = std::min(64, std::max(1, v.top_k));
  if (!(v.top_p > 0.f) || v.top_p > 1.f) v.top_p = 1.f;
  g->c.sampling = v;
}

void q38_gen_set_eos(q38_gen* g, const int32_t* ids, int n) {
  g->c.n_eos = std::min(4, std::max(0, n));
  for (int i = 0; i < g->c.n_eos; ++i) g->c.eos[i] = ids[i];
}

void q38_gen_get_stats(const q38_gen* g, q38_gen_stats* s) {
  *s = g->st;
  s->snapshots = (int)g->snaps.size();
  s->snapshot_bytes = g->snap_bytes();
}

int q38_gen_prompt(q38_gen* g, const int32_t* tokens, int n, const int* snap_at, int n_snap) {
  return q38_gen_prompt_mm(g, tokens, n, nullptr, 0, nullptr, snap_at, n_snap);
}

int q38_gen_prompt_mm(q38_gen* g, const int32_t* tokens, int n, const uint16_t* embeds, int n_emb,
                      const int32_t* pos3, const int* snap_at, int n_snap) {
  if (!g || !tokens || n < 1) return -1;
  g->dsp_ema = 2.f;  // block sizing starts over with each prompt (replays must match)
  // erow[i]: row of `embeds` for prompt index i (the k-th negative token takes row k)
  std::vector<int> erow;
  {
    int k = 0;
    for (int i = 0; i < n; ++i)
      if (tokens[i] < 0) {
        if (erow.empty()) erow.assign((size_t)n, -1);
        erow[(size_t)i] = k++;
      }
    if (k != n_emb || (k && !embeds) || tokens[n - 1] < 0) return -1;
  }
  const size_t H = (size_t)g->hidden;
  auto emb_of = [&](int i) { return embeds + (size_t)erow[(size_t)i] * H; };
  std::vector<uint16_t> semb;
  if (!erow.empty()) semb.resize(8 * H);
  const double t0 = now_ms();
  g->done = true;
  g->pending = -1;
  // What can be reused. The live sequence must be shorter than the prompt
  // (its last logits may be gone); a snapshot carries the logits of its last
  // row, so one that covers the whole prompt is usable as is.
  const size_t live = g->hist.size();
  bool extend = live > 0 && live < (size_t)n && (int64_t)live == q38_position(g->e) &&
                std::equal(g->hist.begin(), g->hist.end(), tokens);
  Snap* best = nullptr;
  for (Snap& s : g->snaps)
    if (s.toks.size() <= (size_t)n && (!best || s.toks.size() > best->toks.size()) &&
        std::equal(s.toks.begin(), s.toks.end(), tokens))
      best = &s;
  if (extend && best && best->toks.size() > live) extend = false;
  if (!extend) {
    if (best) {
      if (q38_snapshot_load(g->e, best->s)) return -2;
      g->hist = best->toks;
      g->unfed_n = best->unfed_n;
      memcpy(g->unfed_tok, best->unfed_tok, sizeof g->unfed_tok);
      if (!best->unfed_emb.empty()) g->unfed_emb = best->unfed_emb;
      best->stamp = ++g->clock;
    } else {
      q38_reset(g->e);
      g->hist.clear();
      g->unfed_n = 0;
    }
  }
  const int reused = (int)g->hist.size();
  int si = 0, last_rows = (!extend && best) ? q38_snapshot_tail_rows(best->s) : 0;
  q38_step_stats ss;
  for (int i = reused; i < n;) {
    while (si < n_snap && snap_at[si] <= i) ++si;
    int m = std::min(g->c.chunk, n - i);
    if (si < n_snap && snap_at[si] < i + m) m = snap_at[si] - i;  // end the step on the boundary
    if (g->flush(tokens[i], false, tokens[i] < 0 ? emb_of(i) : nullptr)) return -3;
    bool any = false;
    for (int j = 0; j < m; ++j)
      if (tokens[i + j] < 0) {
        memcpy(semb.data() + (size_t)j * H, emb_of(i + j), H * 2);
        any = true;
      }
    if (q38_step_mm(g->e, tokens + i, any ? semb.data() : nullptr, pos3 ? pos3 + 3 * (size_t)i : nullptr,
                    m, &ss))
      return -4;
    if (q38_commit(g->e, m)) return -4;
    g->committed(tokens + i, m, any ? semb.data() : nullptr);
    last_rows = m;
    i += m;
    const bool at_mark = si < n_snap && snap_at[si] == i;
    if (at_mark || (i == n && g->c.snapshot_prompt_end)) g->store_snapshot();
  }
  g->pending = q38_sample(g->e, last_rows - 1, &g->c.sampling, g->uni(), nullptr, nullptr, nullptr);
  g->done = false;
  g->st.prompt_tokens = n;
  g->st.reused_tokens = reused;
  g->st.prefill_ms = now_ms() - t0;
  g->st.cycles = g->st.emitted = g->st.drafted = g->st.kept = 0;
  g->st.target_ms = g->st.draft_ms = 0;
  return 0;
}

int q38_gen_next(q38_gen* g, int32_t* out, int cap) {
  if (!g || !out || cap < 1) return -1;
  if (g->done) return 0;
  const q38_sampling& sp = g->c.sampling;
  const int32_t x = g->pending;
  if (g->is_eos(x)) {  // returned, never fed
    out[0] = x;
    g->done = true;
    ++g->st.emitted;
    return 1;
  }
  // Draft. Depth is bounded by what the caller can take this cycle.
  int32_t d[8];
  int32_t qid[8][64];
  float qp[8][64];
  int qn[8];
  int D = 0;
  const int depth = (g->mtp || g->dsp) ? std::min(g->c.depth, cap - 1) : 0;
  q38_step_stats ms;
  if (g->dsp && depth > 0) {
    // one drafter pass proposes every position; `adaptive` bounds the chain's joint probability
    // The block is sized two past what the verifier has been getting lately
    // (a block row costs 0.7 ms): short on prose, long where chains run.
    static const bool fixed_k = getenv("Q38_D_FIXED_K") != nullptr;
    static const float kpad = getenv("Q38_D_KPAD") ? (float)atof(getenv("Q38_D_KPAD")) : 2.0f;
    const int kk = fixed_k ? depth : std::min(depth, std::max(2, (int)(g->dsp_ema + kpad)));
    double uu[8];
    for (int i = 0; i < kk; ++i) uu[i] = g->uni();
    q38_draft_row dr[8];
    D = q38_seq_dspine_draft(g->e, 0, x, kk, &g->c.draft, g->c.adaptive, uu, dr, &ms);
    if (D < 0) return -3;
    g->dsp_ema = 0.75f * g->dsp_ema + 0.25f * (float)D;
    g->st.draft_ms += ms.wall_ms;
    for (int j = 0; j < D; ++j) {
      d[j] = dr[j].tok;
      qn[j] = dr[j].n;
      memcpy(qid[j], dr[j].ids, sizeof dr[j].ids);
      memcpy(qp[j], dr[j].probs, sizeof dr[j].probs);
    }
  }
  if (g->flush(x, depth > 0)) return -3;
  while (g->mtp && D < depth) {
    d[D] = q38_mtp_sample(g->e, &g->c.draft, g->uni(), qid[D], qp[D], &qn[D]);
    float qd = 0;
    for (int i = 0; i < qn[D]; ++i)
      if (qid[D][i] == d[D]) qd = qp[D][i];
    ++D;
    // a deeper draft only pays when the ones before it are likely to be kept
    if (D == depth || qd < g->c.adaptive) break;
    if (q38_mtp_chain(g->e, d[D - 1], &ms)) return -3;
    g->st.draft_ms += ms.wall_ms;
  }
  int32_t rows[8];
  rows[0] = x;
  for (int j = 0; j < D; ++j) rows[j + 1] = d[j];
  q38_step_stats ts;
  if (q38_step(g->e, rows, D + 1, &ts)) return -4;
  g->st.target_ms += ts.wall_ms;
  ++g->st.cycles;

  int acc = 0;
  int32_t next = -1;
  bool eos = false;
  for (int j = 0; j < D; ++j) {
    int32_t pid[64];
    float pp[64];
    int pn;
    q38_sample(g->e, j, &sp, 0.0, pid, pp, &pn);
    float pd = 0, qd = 0;
    for (int i = 0; i < pn; ++i)
      if (pid[i] == d[j]) pd = pp[i];
    for (int i = 0; i < qn[j]; ++i)
      if (qid[j][i] == d[j]) qd = qp[j][i];
    ++g->st.drafted;
    if (g->uni() * qd < pd) {  // keep with probability min(1, p / q)
      ++g->st.kept;
      if (g->is_eos(d[j])) {  // an accepted end of sequence: return it, do not commit its row
        eos = true;
        break;
      }
      ++acc;
      continue;
    }
    double r[64], rsum = 0;  // residual (p - q)+ over p's support
    for (int i = 0; i < pn; ++i) {
      float qi = 0;
      for (int t = 0; t < qn[j]; ++t)
        if (qid[j][t] == pid[i]) qi = qp[j][t];
      r[i] = std::max(0.0, (double)pp[i] - qi);
      rsum += r[i];
    }
    const double u = g->uni() * rsum;
    double c = 0;
    next = pid[pn - 1];
    for (int i = 0; i < pn; ++i) {
      c += r[i];
      if (u < c) {
        next = pid[i];
        break;
      }
    }
    break;
  }
  int n = 0;
  out[n++] = x;
  for (int j = 0; j < acc; ++j) out[n++] = d[j];
  if (eos) {
    out[n++] = d[acc];
    g->done = true;
  } else if (next < 0) {
    next = q38_sample(g->e, D, &sp, g->uni(), nullptr, nullptr, nullptr);  // every draft kept: bonus row
  }
  // Q38_GLOG=file: one line per cycle (drafts, accepted, then the drafter's
  // probability of each drafted token), for tuning the draft policy
  static FILE* glog = getenv("Q38_GLOG") ? fopen(getenv("Q38_GLOG"), "a") : nullptr;
  if (glog && D > 0) {
    fprintf(glog, "%d %d", D, acc);
    for (int j = 0; j < D; ++j) {
      float qd = 0;
      for (int i = 0; i < qn[j]; ++i)
        if (qid[j][i] == d[j]) qd = qp[j][i];
      fprintf(glog, " %.5f", qd);
    }
    fprintf(glog, "\n");
  }
  if (q38_commit(g->e, 1 + acc)) return -4;
  g->committed(rows, 1 + acc);
  g->pending = next;
  g->st.emitted += n;
  return n;
}

void q38_abi_sizes(int out[4]) {
  out[0] = (int)sizeof(q38_config);
  out[1] = (int)sizeof(q38_sampling);
  out[2] = (int)sizeof(q38_gen_config);
  out[3] = (int)sizeof(q38_gen_stats);
}
