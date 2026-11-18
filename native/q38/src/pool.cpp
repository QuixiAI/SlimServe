// q38 stream pool (include/q38/pool.h). Plain C++ over the engine C ABI.
//
// A slot is one engine sequence with the tokens it has committed. A stream is
// one generation bound to a slot; when it ends the slot stays behind, idle,
// so the next turn of that conversation extends it in place.
//
// One cycle:
//   1. streams whose pending token ends the sequence, or that ran out of
//      budget, finish without a row
//   2. the MTP head sees the (hidden, following token) pairs the last pass
//      left behind, for every slot at once, and produces draft logits
//   3. drafts are drawn (one more chained MTP pass for a second draft)
//   4. ONE target pass: [pending, drafts...] per decoding stream, a chunk of
//      prompt rows per prefilling stream
//   5. each stream verifies its drafts (lossless rejection sampling, as in
//      generate.cpp), commits, and reports its tokens
//
// Rows per pass are capped at 8. Decode rows come first; while a prompt is
// being prefilled it gets the rest, and never less than 4 rows, so a long
// prompt cannot be starved by a full house of decoders (which then take
// turns). Drafting is worth its rows only while few streams decode: depth
// `depth` for one stream, 1 for two or three, none from four up or while
// anything prefills.
#include "q38/pool.h"

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

const int kRows = 8;

struct Snap {
  std::vector<int32_t> toks;
  q38_snapshot* s = nullptr;
  int unfed_n = 0;
  int32_t unfed_tok[8] = {0};
  std::vector<uint16_t> unfed_emb;
  uint64_t stamp = 0;
};

struct Slot {
  int seq = -1;
  std::vector<int32_t> hist;  // tokens the engine sequence has committed
  // rows of this sequence's last pass whose (hidden, follower) pair the MTP
  // head has not seen; the last row's follower is filled in at flush
  int unfed_n = 0;
  int32_t unfed_tok[8] = {0};
  std::vector<uint16_t> unfed_emb;  // [8][hidden], rows whose follower is an image row
  int stream = -1;
  uint64_t stamp = 0;
};

struct Stream {
  bool used = false;
  int slot = -1;
  std::vector<int32_t> prompt;
  std::vector<uint16_t> embeds;
  std::vector<int32_t> pos3;
  std::vector<int> erow;  // prompt index -> row of embeds, -1 for text
  std::vector<int> marks;
  size_t mark_i = 0;
  size_t fed = 0;  // prompt tokens committed
  int phase = 0;   // -1 waiting for a shared prefix, 0 prefill, 1 decode, 2 done
  int32_t pending = -1;
  q38_sampling sp{};
  uint64_t rs = 0;
  int32_t eos[4] = {0};
  int n_eos = 0;
  int64_t left = 0;
  int64_t reused = 0;
  bool announced = false;
  // this cycle
  int D = 0, want = 0, m = 0;
  int32_t d[8];
  int32_t qid[8][64];
  float qp[8][64];
  int qn[8];
  float dsp_ema = 2.f;  // DSpine: drafts handed to the verifier lately (sizes the next block)

  double uni() {  // splitmix64
    rs += 0x9E3779B97F4A7C15ull;
    uint64_t z = rs;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0;
  }
  bool is_eos(int32_t t) const {
    for (int i = 0; i < n_eos; ++i)
      if (eos[i] == t) return true;
    return false;
  }
};

}  // namespace

struct q38_pool {
  q38_engine* e = nullptr;
  q38_pool_config c{};
  bool mtp = false;
  bool dsp = false;  // DSpine block drafter (preferred when loaded): the engine keeps its context itself
  int hidden = 0;
  std::vector<Slot> slots;
  std::vector<Stream> streams;
  std::vector<Snap> snaps;
  uint64_t clock = 0;
  int rr = 0;  // round-robin start for decoders that have to take turns
  int big_rows = 0;   // rows of the engine's big (prefill) step
  int since_big = 0;  // ordinary passes since the last big one while streams decode
  int rr_big = 0;
  q38_pool_stats st{};

  void set_emb(Slot& s, int row, const uint16_t* emb) {
    if (s.unfed_emb.empty()) s.unfed_emb.resize((size_t)8 * hidden);
    memcpy(s.unfed_emb.data() + (size_t)row * hidden, emb, (size_t)hidden * 2);
  }
  // host bytes an idle sequence keeps: K/V rows of 16 layers and the MTP head, plus GDN state
  int64_t slot_bytes(const Slot& s) const {
    return s.hist.empty() ? 0 : (int64_t)s.hist.size() * 70000 + 320000000;
  }
  int64_t cache_bytes() const {
    int64_t b = 0;
    for (const Snap& s : snaps)
      b += q38_snapshot_bytes(s.s) + (int64_t)s.toks.size() * 4 + (int64_t)s.unfed_emb.size() * 2;
    for (const Slot& s : slots)
      if (s.stream < 0) b += slot_bytes(s);
    return b;
  }
  void drop_slot(Slot& s) {
    q38_seq_reset(e, s.seq);
    s.hist.clear();
    s.unfed_n = 0;
  }
  // Evict the least recently used snapshot or idle sequence until the cache fits.
  void trim() {
    while (cache_bytes() > c.max_cache_bytes) {
      int bs = -1, bl = -1;
      for (size_t i = 0; i < snaps.size(); ++i)
        if (bs < 0 || snaps[i].stamp < snaps[(size_t)bs].stamp) bs = (int)i;
      for (size_t i = 0; i < slots.size(); ++i)
        if (slots[i].stream < 0 && !slots[i].hist.empty() &&
            (bl < 0 || slots[i].stamp < slots[(size_t)bl].stamp))
          bl = (int)i;
      if (bs < 0 && bl < 0) return;
      const bool take_snap = bs >= 0 && (bl < 0 || snaps[(size_t)bs].stamp < slots[(size_t)bl].stamp);
      if (take_snap) {
        if (snaps.size() == 1 && bl < 0) return;  // always keep the newest snapshot
        q38_snapshot_free(snaps[(size_t)bs].s);
        snaps.erase(snaps.begin() + bs);
      } else {
        drop_slot(slots[(size_t)bl]);
      }
    }
  }
  void store_snapshot(Slot& sl) {
    for (Snap& s : snaps)
      if (s.toks == sl.hist) {
        s.stamp = ++clock;
        return;
      }
    q38_snapshot* sn = q38_seq_snapshot_save(e, sl.seq);
    if (!sn) return;
    Snap s;
    s.toks = sl.hist;
    s.s = sn;
    s.unfed_n = sl.unfed_n;
    memcpy(s.unfed_tok, sl.unfed_tok, sizeof s.unfed_tok);
    for (int i = 0; i + 1 < sl.unfed_n; ++i)
      if (sl.unfed_tok[i] < 0) s.unfed_emb = sl.unfed_emb;
    s.stamp = ++clock;
    snaps.push_back(std::move(s));
    trim();
  }
  // The engine committed rows t[0..m) of this slot; emb ([m][hidden]) has the image rows.
  void committed(Slot& sl, const int32_t* t, int m, const uint16_t* emb) {
    sl.hist.insert(sl.hist.end(), t, t + m);
    if (m > kRows) {  // a big prefill chunk: its rows are not shown to the MTP head
      sl.unfed_n = 0;
      return;
    }
    sl.unfed_n = m;
    for (int i = 0; i + 1 < m; ++i) {
      sl.unfed_tok[i] = t[i + 1];
      if (t[i + 1] < 0) set_emb(sl, i, emb + (size_t)(i + 1) * hidden);
    }
  }
  void finish(Stream& s) {
    s.phase = 2;
    if (s.slot >= 0) slots[(size_t)s.slot].unfed_n = 0;  // its hidden rows will not survive the next pass
  }
};

void q38_pool_config_default(q38_pool_config* c) {
  memset(c, 0, sizeof *c);
  c->draft = {1.0f, 20, 0.95f};
  c->depth = 2;
  c->adaptive = 0.5f;
  c->max_streams = 8;
  c->max_cache_bytes = 16ll << 30;
  c->mtp_prompt_tail = 256;  // measured on a 2k prompt: acceptance unchanged from 32 up, prefill 3% faster
  c->big_every = 2;
  c->big_rows = 256;
}

q38_pool* q38_pool_create(q38_engine* e, const q38_pool_config* c) {
  if (!e || !c || c->max_streams < 1 || c->max_streams > 8 || c->depth < 0 || c->depth > 7)
    return nullptr;
  if (c->depth > 0 && !q38_has_mtp(e) && !q38_has_dspine(e)) return nullptr;
  q38_pool* p = new q38_pool;
  p->e = e;
  p->c = *c;
  p->dsp = c->depth > 0 && q38_has_dspine(e);
  p->mtp = c->depth > 0 && !p->dsp;
  p->hidden = q38_hidden(e);
  p->big_rows = q38_big_rows(e);
  p->streams.resize((size_t)c->max_streams);
  q38_set_argmax(e, 0);
  q38_reset(e);
  Slot s0;
  s0.seq = 0;
  p->slots.push_back(s0);
  return p;
}

void q38_pool_free(q38_pool* p) {
  if (!p) return;
  for (Snap& s : p->snaps) q38_snapshot_free(s.s);
  for (Slot& s : p->slots)
    if (s.seq > 0) q38_seq_free(p->e, s.seq);
  q38_reset(p->e);
  delete p;
}

int q38_pool_active(const q38_pool* p) {
  int n = 0;
  for (const Stream& s : p->streams) n += s.used && s.phase != 2;
  return n;
}

void q38_pool_get_stats(const q38_pool* p, q38_pool_stats* s) {
  *s = p->st;
  s->snapshots = (int)p->snaps.size();
  s->idle_sequences = 0;
  for (const Slot& sl : p->slots) s->idle_sequences += sl.stream < 0 && !sl.hist.empty();
  s->cache_bytes = p->cache_bytes();
}

// Bind a stream to a sequence and to whatever prefix can be reused. Returns 0
// when it is running, 1 when it should wait (another stream is prefilling a
// longer prefix this prompt shares and will leave a snapshot there), negative
// on error.
static int start_stream(q38_pool* p, int si) {
  Stream& s = p->streams[(size_t)si];
  const int n = (int)s.prompt.size();
  const int32_t* tokens = s.prompt.data();
  // What can be reused: an idle sequence that is a proper prefix of the
  // prompt (its last logits are gone, so it must be shorter), or a snapshot
  // (which carries the logits of its last row, so it may cover the whole prompt).
  int live = -1;
  Snap* best = nullptr;
  for (int pass = 0; pass < 2; ++pass) {
    live = -1;
    best = nullptr;
    for (size_t i = 0; i < p->slots.size(); ++i) {
      const Slot& sl = p->slots[i];
      if (sl.stream >= 0 || sl.hist.empty() || sl.hist.size() >= (size_t)n) continue;
      if ((int64_t)sl.hist.size() != q38_seq_position(p->e, sl.seq)) continue;
      if (!std::equal(sl.hist.begin(), sl.hist.end(), tokens)) continue;
      if (live < 0 || sl.hist.size() > p->slots[(size_t)live].hist.size()) live = (int)i;
    }
    for (Snap& sn : p->snaps)
      if (sn.toks.size() <= (size_t)n && (!best || sn.toks.size() > best->toks.size()) &&
          std::equal(sn.toks.begin(), sn.toks.end(), tokens))
        best = &sn;
    if (pass) break;
    // A stream still prefilling a prefix this prompt shares: let it get to the
    // end of the shared part and snapshot there, instead of prefilling the
    // same tokens twice in the same passes. (Requests that fan out from one
    // long context arrive together; without this each pays the whole prompt.)
    const size_t have = std::max(live >= 0 ? p->slots[(size_t)live].hist.size() : 0,
                                 best ? best->toks.size() : 0);
    bool wait = false, stored = false;
    for (size_t i = 0; i < p->streams.size(); ++i) {
      Stream& t = p->streams[i];
      if ((int)i == si || !t.used || t.phase != 0 || t.slot < 0) continue;
      size_t c = 0;
      const size_t lim = std::min(t.prompt.size(), (size_t)n);
      while (c < lim && t.prompt[c] == tokens[c]) ++c;
      if (c < have + 64 || t.fed > c) continue;  // not worth a wait, or already past it
      if (t.fed == c) {
        p->store_snapshot(p->slots[(size_t)t.slot]);
        stored = true;
      } else {
        auto it = std::lower_bound(t.marks.begin() + (long)t.mark_i, t.marks.end(), (int)c);
        if (it == t.marks.end() || *it != (int)c) t.marks.insert(it, (int)c);
        wait = true;
      }
    }
    if (wait) return 1;
    if (!stored) break;
  }
  if (live >= 0 && best && best->toks.size() > p->slots[(size_t)live].hist.size()) live = -1;
  int slot = live;
  bool have_logits = false;
  if (slot < 0) {
    // a sequence to start in: an empty idle one, a new one, or the least recently used idle one
    for (size_t i = 0; i < p->slots.size() && slot < 0; ++i)
      if (p->slots[i].stream < 0 && p->slots[i].hist.empty()) slot = (int)i;
    if (slot < 0 && (int)p->slots.size() < p->c.max_streams) {
      Slot ns;
      ns.seq = q38_seq_new(p->e);
      if (ns.seq < 0) return -8;
      p->slots.push_back(ns);
      slot = (int)p->slots.size() - 1;
    }
    if (slot < 0) {
      for (size_t i = 0; i < p->slots.size(); ++i)
        if (p->slots[i].stream < 0 &&
            (slot < 0 || p->slots[i].stamp < p->slots[(size_t)slot].stamp))
          slot = (int)i;
      if (slot < 0) return -7;
    }
    Slot& sl = p->slots[(size_t)slot];
    if (best) {
      if (q38_seq_snapshot_load(p->e, sl.seq, best->s)) return -2;
      sl.hist = best->toks;
      sl.unfed_n = best->unfed_n;
      memcpy(sl.unfed_tok, best->unfed_tok, sizeof sl.unfed_tok);
      if (!best->unfed_emb.empty()) sl.unfed_emb = best->unfed_emb;
      best->stamp = ++p->clock;
      have_logits = q38_snapshot_tail_rows(best->s) > 0;
    } else {
      p->drop_slot(sl);
    }
  }
  Slot& sl = p->slots[(size_t)slot];
  sl.stream = si;
  sl.stamp = ++p->clock;
  s.slot = slot;
  s.fed = sl.hist.size();
  s.reused = (int64_t)s.fed;
  while (s.mark_i < s.marks.size() && (size_t)s.marks[s.mark_i] <= s.fed) ++s.mark_i;
  s.phase = 0;
  if (s.fed == (size_t)n) {
    if (!have_logits) return -2;  // cannot happen: only a snapshot covers the whole prompt
    const int tail = q38_snapshot_tail_rows(best->s);
    s.pending = q38_seq_sample(p->e, sl.seq, tail - 1, &s.sp, s.uni(), nullptr, nullptr, nullptr);
    s.phase = 1;
  }
  return 0;
}


int q38_pool_open(q38_pool* p, const q38_pool_request* r) {
  if (!p || !r || !r->tokens || r->n < 1 || r->n_eos < 0 || r->n_eos > 4 || r->max_tokens < 1)
    return -1;
  const int n = r->n;
  int si = -1;
  for (size_t i = 0; i < p->streams.size(); ++i)
    if (!p->streams[i].used) {
      si = (int)i;
      break;
    }
  if (si < 0) return -7;
  Stream s;
  {
    int k = 0;
    for (int i = 0; i < n; ++i)
      if (r->tokens[i] < 0) {
        if (s.erow.empty()) s.erow.assign((size_t)n, -1);
        s.erow[(size_t)i] = k++;
      }
    if (k != r->n_emb || (k && !r->embeds) || r->tokens[n - 1] < 0) return -1;
    if (k) s.embeds.assign(r->embeds, r->embeds + (size_t)k * p->hidden);
  }
  s.used = true;
  s.prompt.assign(r->tokens, r->tokens + n);
  if (r->pos3) s.pos3.assign(r->pos3, r->pos3 + (size_t)3 * n);
  for (int i = 0; i < r->n_snap; ++i)
    if (r->snap_at[i] > 0 && r->snap_at[i] <= n) s.marks.push_back(r->snap_at[i]);
  std::sort(s.marks.begin(), s.marks.end());
  s.sp = r->sampling;
  if (s.sp.temperature <= 0.f) {  // "greedy": one candidate
    s.sp.temperature = 1.f;
    s.sp.top_k = 1;
  }
  s.sp.top_k = std::min(64, std::max(1, s.sp.top_k));
  if (!(s.sp.top_p > 0.f) || s.sp.top_p > 1.f) s.sp.top_p = 1.f;
  s.rs = r->seed;
  s.n_eos = r->n_eos;
  for (int i = 0; i < r->n_eos; ++i) s.eos[i] = r->eos[i];
  s.left = r->max_tokens;

  s.phase = -1;
  p->streams[(size_t)si] = std::move(s);
  const int rc = start_stream(p, si);
  if (rc < 0) {
    p->streams[(size_t)si] = Stream{};
    return rc;
  }
  return si;
}

void q38_pool_close(q38_pool* p, int stream) {
  if (!p || stream < 0 || stream >= (int)p->streams.size() || !p->streams[(size_t)stream].used) return;
  Stream& s = p->streams[(size_t)stream];
  if (s.slot >= 0) {
    Slot& sl = p->slots[(size_t)s.slot];
    // a pending pass that was never committed leaves nothing behind
    sl.unfed_n = 0;
    sl.stream = -1;
    sl.stamp = ++p->clock;
    if ((int64_t)sl.hist.size() != q38_seq_position(p->e, sl.seq)) p->drop_slot(sl);
  }
  s = Stream{};
  p->trim();
}

int q38_pool_step(q38_pool* p, q38_pool_event* ev, int cap) {
  if (!p || !ev || cap < (int)p->streams.size()) return -1;
  int nev = 0;
  auto event = [&](int si) -> q38_pool_event& {
    q38_pool_event& x = ev[nev++];
    memset(&x, 0, sizeof x);
    x.stream = si;
    Stream& s = p->streams[(size_t)si];
    x.reused = s.reused;
    if (!s.announced && s.phase >= 1) {
      x.prompt_done = 1;
      s.announced = true;
    }
    return x;
  };

  // streams that waited for a shared prefix: start them once it is there (or its owner is gone)
  for (size_t i = 0; i < p->streams.size(); ++i)
    if (p->streams[i].used && p->streams[i].phase == -1) {
      const int rc = start_stream(p, (int)i);
      if (rc < 0) return rc;
    }

  // 1. streams that end without a row
  std::vector<int> dec, pre;
  for (size_t i = 0; i < p->streams.size(); ++i) {
    Stream& s = p->streams[i];
    if (!s.used || s.phase == 2 || s.phase == -1) continue;
    if (s.phase == 0) {
      pre.push_back((int)i);
      continue;
    }
    if (s.is_eos(s.pending)) {  // returned, never fed
      q38_pool_event& x = event((int)i);
      x.tokens[x.n++] = s.pending;
      x.done = 1;
      ++p->st.emitted;
      p->finish(s);
    } else if (s.left <= 0) {
      event((int)i).done = 2;
      p->finish(s);
    } else {
      dec.push_back((int)i);
    }
  }
  if (dec.empty() && pre.empty()) return nev;

  // A big pass: up to 64 prompt rows of ONE prefilling stream through the
  // engine's many-row path (twice the rows per second of 8-row passes). Nothing
  // else rides in it, so while streams decode it only gets every
  // (big_every + 1)-th pass. The rows just before a snapshot mark or the end
  // of the prompt are left to ordinary passes: the snapshot and the first
  // sample want their outputs, the MTP head wants the last pairs.
  int big_m = 0;
  if (!pre.empty() && p->c.big_every >= 0 && p->big_rows > kRows &&
      (dec.empty() || (p->c.big_every > 0 && p->since_big >= p->c.big_every))) {
    const int np = (int)pre.size();
    for (int k = 0; k < np && !big_m; ++k) {
      const int si = pre[(size_t)((p->rr_big + k) % np)];
      const Stream& s = p->streams[(size_t)si];
      const long end = (long)s.prompt.size();
      const long bound = s.mark_i < s.marks.size() ? std::min<long>(s.marks[s.mark_i], end) : end;
      // The rows before a boundary go through ordinary passes (the snapshot and the first
      // sample want their outputs). The MTP head gets its pairs from the big pass itself.
      const long keep = 8;
      // nothing else moves during a big pass: 64 rows (0.3 s) while streams decode, big_rows
      // (256: 1 s) while none does
      const long cap = dec.empty() ? std::min(p->big_rows, std::max(p->c.big_rows, 16)) : std::min(p->big_rows, 64);
      const long bm = std::min<long>(cap, bound - (long)s.fed - keep);
      if (bm < 16) continue;
      big_m = (int)bm;
      p->rr_big = (p->rr_big + k + 1) % np;
      pre.assign(1, si);
      dec.clear();  // everyone else sits this pass out (their MTP pairs are still fed below)
    }
  }
  if (big_m)
    p->since_big = 0;
  else if (!dec.empty())
    ++p->since_big;

  // rows: decoders first, a prefill never under 4 rows
  const int n_dec = (int)dec.size(), n_pre = (int)pre.size();
  const int pre_rows = n_pre ? std::max(kRows - n_dec, 4) : 0;
  const int dec_now = std::min(n_dec, kRows - pre_rows);
  if (dec_now < n_dec) {  // take turns
    std::rotate(dec.begin(), dec.begin() + (p->rr++ % n_dec), dec.end());
    dec.resize((size_t)dec_now);
  }
  int depth = 0;
  if (p->mtp && !n_pre) depth = n_dec == 1 ? p->c.depth : (n_dec <= 3 ? std::min(1, p->c.depth) : 0);
  // DSpine: a lone stream drafts blocks of up to 7; two streams take two drafts
  // each and three or four one each, through one shared drafter pass (measured
  // 2026-11-17, essay, tok/s aggregate: 2 streams 79.3 with one draft each,
  // 82.4 with two, 80.7 with three; 3 streams 82.2 without drafts, 90.2 with;
  // 4 streams 94.7 without, 99.4 with). Five or more leave no rows for drafts.
  if (p->dsp && !n_pre) {
    static const char* dtab = getenv("Q38_POOL_DEPTHS");  // diagnostic: draft depth for 2, 3, 4 streams, e.g. "1,1,1"
    int d2 = 2, d3 = 1, d4 = 1;
    if (dtab) sscanf(dtab, "%d,%d,%d", &d2, &d3, &d4);
    depth = n_dec == 1 ? 7 : n_dec == 2 ? d2 : n_dec == 3 ? d3 : n_dec == 4 ? d4 : 0;
  }
  // rows left after one per decoder bound the drafts
  if (dec_now) depth = std::min(depth, (kRows - dec_now) / dec_now);
  // keep the MTP history complete while drafting is possible; a full house does not pay for it
  const bool feed = p->mtp && (n_pre > 0 || n_dec <= 3);
  for (int si : dec) {
    Stream& s = p->streams[(size_t)si];
    s.want = (int)std::min<int64_t>(depth, s.left - 1);
    s.D = 0;
  }

  // prefill chunks
  {
    int left = pre_rows, who = n_pre;
    for (int si : pre) {
      Stream& s = p->streams[(size_t)si];
      int m = std::min<int>({left / who, (int)(s.prompt.size() - s.fed), kRows});
      if (s.mark_i < s.marks.size() && (size_t)s.marks[s.mark_i] < s.fed + (size_t)m)
        m = (int)((size_t)s.marks[s.mark_i] - s.fed);  // end the pass on the boundary
      s.m = std::max(m, 1);
      left -= s.m;
      --who;
    }
    if (big_m) p->streams[(size_t)pre[0]].m = big_m;
  }

  q38_step_stats ms;
  // 2. MTP: pairs the last pass left behind, follower of each slot's last row known now
  if (p->mtp) {
    q38_group g[8];
    int32_t tok[8];
    int rows[8];
    std::vector<uint16_t> emb;
    int ng = 0, m = 0;
    bool head = false;
    auto send = [&]() -> int {
      if (!ng) return 0;
      const int rc = q38_mtp_feed_groups(p->e, g, ng, tok, emb.empty() ? nullptr : emb.data(), rows,
                                         head ? 1 : 0, &ms);
      p->st.draft_ms += ms.wall_ms;
      ng = m = 0;
      head = false;
      emb.clear();
      return rc;
    };
    auto add = [&](Stream& s, int32_t next, const uint16_t* next_emb, bool want_head) -> int {
      Slot& sl = p->slots[(size_t)s.slot];
      if (!sl.unfed_n) return 0;
      if (!feed) {
        sl.unfed_n = 0;
        return 0;
      }
      if (m + sl.unfed_n > kRows && send()) return -3;
      sl.unfed_tok[sl.unfed_n - 1] = next;
      if (next < 0) p->set_emb(sl, sl.unfed_n - 1, next_emb);
      g[ng++] = {sl.seq, sl.unfed_n};
      for (int i = 0; i < sl.unfed_n; ++i) {
        tok[m + i] = sl.unfed_tok[i];
        rows[i + m] = i;
        if (sl.unfed_tok[i] < 0) {
          if (emb.empty()) emb.resize((size_t)kRows * p->hidden);
          memcpy(emb.data() + (size_t)(m + i) * p->hidden, sl.unfed_emb.data() + (size_t)i * p->hidden,
                 (size_t)p->hidden * 2);
        }
      }
      m += sl.unfed_n;
      head = head || want_head;
      sl.unfed_n = 0;
      return 0;
    };
    for (int si : pre) {
      Stream& s = p->streams[(size_t)si];
      if (p->c.mtp_prompt_tail > 0 && s.prompt.size() - s.fed > (size_t)p->c.mtp_prompt_tail) {
        p->slots[(size_t)s.slot].unfed_n = 0;  // far from the end of the prompt: not shown to the head
        continue;
      }
      const int32_t nx = s.prompt[s.fed];
      if (add(s, nx, nx < 0 ? s.embeds.data() + (size_t)s.erow[s.fed] * p->hidden : nullptr, false)) return -3;
    }
    for (int si : dec) {
      Stream& s = p->streams[(size_t)si];
      const bool had = p->slots[(size_t)s.slot].unfed_n > 0;
      if (add(s, s.pending, nullptr, s.want > 0)) return -3;
      if (!had || !feed) s.want = 0;  // no fresh pair, no draft logits
    }
    // decoders that are sitting this pass out keep their pairs only until the
    // next pass overwrites the hidden rows: feed them now too
    for (size_t i = 0; i < p->streams.size(); ++i) {
      Stream& s = p->streams[i];
      if (!s.used || s.phase != 1 || std::find(dec.begin(), dec.end(), (int)i) != dec.end()) continue;
      if (add(s, s.pending, nullptr, false)) return -3;
    }
    if (send()) return -3;

    // 3. drafts
    for (int round = 0;; ++round) {
      int cs[8];
      int32_t ct[8];
      int nc = 0;
      for (int si : dec) {
        Stream& s = p->streams[(size_t)si];
        if (s.D != round || s.D >= s.want) continue;
        const int j = s.D;
        s.d[j] = q38_seq_mtp_sample(p->e, p->slots[(size_t)s.slot].seq, &p->c.draft, s.uni(), s.qid[j],
                                    s.qp[j], &s.qn[j]);
        float qd = 0;
        for (int i = 0; i < s.qn[j]; ++i)
          if (s.qid[j][i] == s.d[j]) qd = s.qp[j][i];
        ++s.D;
        // a deeper draft only pays when the ones before it are likely to be kept
        if (s.D < s.want && qd >= p->c.adaptive) {
          cs[nc] = p->slots[(size_t)s.slot].seq;
          ct[nc++] = s.d[j];
        } else {
          s.want = s.D;
        }
      }
      if (!nc) break;
      if (q38_mtp_chain_groups(p->e, cs, ct, nc, &ms)) return -3;
      p->st.draft_ms += ms.wall_ms;
    }
  }

  if (p->dsp) {
    // adaptive < 0 names the marginal rule's rate; the MTP-style threshold has no meaning here
    const float rate = p->c.adaptive < 0 ? p->c.adaptive : -0.06f;
    // one drafter pass for every drafting stream
    int sq[4], kk[4], nout[4], who[4], n = 0;
    int32_t an[4];
    double uu[32];
    q38_draft_row dr[32];
    for (int si : dec) {
      Stream& s = p->streams[(size_t)si];
      if (s.want <= 0 || n == 4) continue;
      kk[n] = n_dec == 1 ? std::min(s.want, std::max(2, (int)(s.dsp_ema + 2.0f))) : s.want;
      for (int i = 0; i < kk[n]; ++i) uu[8 * n + i] = s.uni();
      sq[n] = p->slots[(size_t)s.slot].seq;
      an[n] = s.pending;
      who[n++] = si;
    }
    if (n) {
      if (q38_dspine_draft_groups(p->e, sq, an, kk, n, &p->c.draft, rate, uu, dr, nout, &ms)) return -3;
      p->st.draft_ms += ms.wall_ms;
      for (int a = 0; a < n; ++a) {
        Stream& s = p->streams[(size_t)who[a]];
        for (int j = 0; j < nout[a]; ++j) {
          s.d[j] = dr[8 * a + j].tok;
          s.qn[j] = dr[8 * a + j].n;
          memcpy(s.qid[j], dr[8 * a + j].ids, sizeof dr[0].ids);
          memcpy(s.qp[j], dr[8 * a + j].probs, sizeof dr[0].probs);
        }
        s.D = nout[a];
        if (n_dec == 1) s.dsp_ema = 0.75f * s.dsp_ema + 0.25f * (float)nout[a];
      }
    }
  }

  // 4. the target pass
  const int row_cap = std::max(kRows, big_m);
  q38_group g[8];
  std::vector<int32_t> rowv((size_t)row_cap);
  int32_t* const row = rowv.data();
  int ng = 0, m = 0;
  bool any_emb = false, any_pos = false;
  for (int si : pre) any_pos = any_pos || !p->streams[(size_t)si].pos3.empty();
  std::vector<uint16_t> semb;
  std::vector<int32_t> spos;
  if (any_pos) spos.assign((size_t)3 * row_cap, -1);
  for (int si : dec) {
    Stream& s = p->streams[(size_t)si];
    g[ng++] = {p->slots[(size_t)s.slot].seq, 1 + s.D};
    row[m++] = s.pending;
    for (int j = 0; j < s.D; ++j) row[m++] = s.d[j];
  }
  for (int si : pre) {
    Stream& s = p->streams[(size_t)si];
    g[ng++] = {p->slots[(size_t)s.slot].seq, s.m};
    for (int j = 0; j < s.m; ++j) {
      const size_t pi = s.fed + (size_t)j;
      row[m] = s.prompt[pi];
      if (row[m] < 0) {
        if (semb.empty()) semb.resize((size_t)row_cap * p->hidden);
        memcpy(semb.data() + (size_t)m * p->hidden, s.embeds.data() + (size_t)s.erow[pi] * p->hidden,
               (size_t)p->hidden * 2);
        any_emb = true;
      }
      if (!s.pos3.empty()) memcpy(spos.data() + (size_t)3 * m, s.pos3.data() + 3 * pi, 12);
      ++m;
    }
  }
  q38_step_stats ts;
  if (q38_step_groups(p->e, g, ng, row, any_emb ? semb.data() : nullptr, any_pos ? spos.data() : nullptr, &ts))
    return -4;
  p->st.target_ms += ts.wall_ms;
  if (getenv("Q38_POOL_LOG"))
    fprintf(stderr, "pass %lld: %d rows in %d groups (%d decode, %d prefill) wall %.1f gpu %.1f encode %.2f ms, %d dispatches %d barriers\n",
            (long long)p->st.cycles, m, ng, (int)dec.size(), (int)pre.size(), ts.wall_ms, ts.gpu_ms, ts.encode_ms,
            ts.dispatches, ts.barriers);
  ++p->st.cycles;
  p->st.rows += m;

  // 5. verify, commit, report
  int r0 = 0;
  for (int si : dec) {
    Stream& s = p->streams[(size_t)si];
    Slot& sl = p->slots[(size_t)s.slot];
    const int D = s.D;
    int acc = 0;
    int32_t next = -1;
    bool eos = false;
    for (int j = 0; j < D; ++j) {
      int32_t pid[64];
      float pp[64];
      int pn;
      q38_seq_sample(p->e, sl.seq, j, &s.sp, 0.0, pid, pp, &pn);
      float pd = 0, qd = 0;
      for (int i = 0; i < pn; ++i)
        if (pid[i] == s.d[j]) pd = pp[i];
      for (int i = 0; i < s.qn[j]; ++i)
        if (s.qid[j][i] == s.d[j]) qd = s.qp[j][i];
      ++p->st.drafted;
      if (s.uni() * qd < pd) {  // keep with probability min(1, p / q)
        ++p->st.kept;
        if (s.is_eos(s.d[j])) {  // an accepted end of sequence: return it, do not commit its row
          eos = true;
          break;
        }
        ++acc;
        continue;
      }
      double r[64], rsum = 0;  // residual (p - q)+ over p's support
      for (int i = 0; i < pn; ++i) {
        float qi = 0;
        for (int t = 0; t < s.qn[j]; ++t)
          if (s.qid[j][t] == pid[i]) qi = s.qp[j][t];
        r[i] = std::max(0.0, (double)pp[i] - qi);
        rsum += r[i];
      }
      const double u = s.uni() * rsum;
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
    q38_pool_event& x = event(si);
    x.tokens[x.n++] = s.pending;
    for (int j = 0; j < acc; ++j) x.tokens[x.n++] = s.d[j];
    if (eos) {
      x.tokens[x.n++] = s.d[acc];
    } else if (next < 0) {
      next = q38_seq_sample(p->e, sl.seq, D, &s.sp, s.uni(), nullptr, nullptr, nullptr);  // every draft kept: bonus row
    }
    if (q38_seq_commit(p->e, sl.seq, 1 + acc)) return -4;
    p->committed(sl, row + r0, 1 + acc, nullptr);
    s.pending = next;
    s.left -= x.n;
    p->st.emitted += x.n;
    if (eos) {
      x.done = 1;
      p->finish(s);
    } else if (s.left <= 0) {
      x.done = 2;
      p->finish(s);
    }
    r0 += 1 + D;
  }
  for (int si : pre) {
    Stream& s = p->streams[(size_t)si];
    Slot& sl = p->slots[(size_t)s.slot];
    if (q38_seq_commit(p->e, sl.seq, s.m)) return -4;
    p->committed(sl, row + r0, s.m, any_emb ? semb.data() + (size_t)r0 * p->hidden : nullptr);
    if (big_m && feed) {
      // MTP pairs of this chunk: (row j, the prompt token after it) for the rows inside the
      // window the head is shown; the followers are all known (rows remain after a big pass)
      const size_t n_all = s.prompt.size();
      const size_t from = p->c.mtp_prompt_tail > 0 && n_all > (size_t)p->c.mtp_prompt_tail
                              ? n_all - (size_t)p->c.mtp_prompt_tail : 0;
      std::vector<uint16_t> femb;
      for (int j = (int)std::max<long>(0, (long)from - (long)s.fed); j < s.m; j += kRows) {
        const int n = std::min(kRows, s.m - j);
        int32_t ft[8];
        bool fe = false;
        for (int i = 0; i < n; ++i) {
          const size_t pi = s.fed + (size_t)(j + i) + 1;
          ft[i] = s.prompt[pi];
          if (ft[i] < 0) {
            if (femb.empty()) femb.resize((size_t)kRows * p->hidden);
            memcpy(femb.data() + (size_t)i * p->hidden, s.embeds.data() + (size_t)s.erow[pi] * p->hidden,
                   (size_t)p->hidden * 2);
            fe = true;
          }
        }
        if (q38_mtp_feed_big(p->e, sl.seq, j, n, ft, fe ? femb.data() : nullptr, &ms)) return -3;
        p->st.draft_ms += ms.wall_ms;
      }
    }
    s.fed += (size_t)s.m;
    const bool at_mark = s.mark_i < s.marks.size() && (size_t)s.marks[s.mark_i] == s.fed;
    while (s.mark_i < s.marks.size() && (size_t)s.marks[s.mark_i] <= s.fed) ++s.mark_i;
    const bool end = s.fed == s.prompt.size();
    if (at_mark || end) p->store_snapshot(sl);
    if (end) {
      s.pending = q38_seq_sample(p->e, sl.seq, s.m - 1, &s.sp, s.uni(), nullptr, nullptr, nullptr);
      s.phase = 1;
      event(si);
    }
    r0 += s.m;
  }
  return nev;
}

void q38_pool_abi_sizes(int out[4]) {
  out[0] = (int)sizeof(q38_pool_config);
  out[1] = (int)sizeof(q38_pool_request);
  out[2] = (int)sizeof(q38_pool_event);
  out[3] = (int)sizeof(q38_pool_stats);
}
