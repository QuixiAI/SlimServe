#include <cmath>
#include <chrono>
// q38_lab: drive the engine from the command line.
//
//   q38_lab --pack P --metallib A[:B] --tokens ids.txt [--chunk 8] [--gen N]
//           [--cap L] [--dump out.bin] [--serial] [--ctx N] [--bench N]
//
// Prefills the token file in chunks of --chunk rows, then generates --gen
// tokens greedily one row at a time (greedy here is a parity and timing
// instrument, not a quality benchmark). --dump writes the last step's rows as
// float32: logits [m, vocab], or with --cap the residual stream [m, hidden].
// --bench N re-runs N single-row decode steps and reports the distribution.

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "q38/engine.h"
#include "q38/generate.h"
#include "q38/pool.h"

static float bf16(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int main(int argc, char** argv) {
  std::string pack, metallib, tokens_path, dump;
  int snap_test = 0, gen_api = 0, multi_test = 0, pool_bench = 0, stagger = 0, big_test = 0, big_head = 0;
  int chunk = 8, gen = 0, cap = 0, serial = 0, ctx = 0, bench = 0, sample = 0;
  uint64_t seed = 42;
  std::string out_tokens, dump_all, mtp_pack, dspine_pack;
  int gen_once = 0, plen = 0;
  std::string dsp_accept;
  int depth = 3;
  double adaptive = 0.0;  // keep drafting only while q(draft) >= this
  double draft_temp = 1.0;  // sampler temperature for drafts (any value stays lossless)
  int draft_topk = 20;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--pack") pack = next();
    else if (a == "--metallib") metallib = next();
    else if (a == "--tokens") tokens_path = next();
    else if (a == "--dump") dump = next();
    else if (a == "--chunk") chunk = atoi(next());
    else if (a == "--gen") gen = atoi(next());
    else if (a == "--cap") cap = atoi(next());
    else if (a == "--ctx") ctx = atoi(next());
    else if (a == "--bench") bench = atoi(next());
    else if (a == "--serial") serial = 1;
    else if (a == "--sample") sample = 1;
    else if (a == "--seed") seed = strtoull(next(), nullptr, 10);
    else if (a == "--out-tokens") out_tokens = next();
    else if (a == "--dump-all") dump_all = next();
    else if (a == "--snap-test") snap_test = 1;
    else if (a == "--gen-api") gen_api = 1;
    else if (a == "--multi-test") multi_test = atoi(next());
    else if (a == "--pool-bench") pool_bench = atoi(next());
    else if (a == "--big-test") big_test = atoi(next());
    else if (a == "--big-head") big_head = atoi(next());
    else if (a == "--stagger") stagger = atoi(next());
    else if (a == "--mtp") mtp_pack = next();
    else if (a == "--dspine") { dspine_pack = next(); gen_api = 1; }  // DSpine block drafter (generator API)
    else if (a == "--gen-once") gen_once = 1;
    else if (a == "--dspine-accept") { dsp_accept = next(); }  // teacher-forced q(true token) per anchor -> file
    else if (a == "--plen") plen = atoi(next());  // generator API: one decode, no snapshot replays
    else if (a == "--depth") depth = atoi(next());
    else if (a == "--adaptive") adaptive = atof(next());
    else if (a == "--draft-temp") draft_temp = atof(next());
    else if (a == "--draft-topk") draft_topk = atoi(next());
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (pack.empty() || metallib.empty() || tokens_path.empty()) {
    fprintf(stderr, "usage: q38_lab --pack P --metallib A[:B] --tokens ids.txt [options]\n");
    return 2;
  }
  std::vector<int32_t> toks;
  {
    std::ifstream f(tokens_path);
    int v;
    while (f >> v) toks.push_back(v);
  }
  if (toks.empty()) { fprintf(stderr, "no tokens in %s\n", tokens_path.c_str()); return 2; }

  q38_config cfg = {ctx, 8, 16, serial};
  char err[512] = {0};
  const double t_load = [NSDate timeIntervalSinceReferenceDate];
  q38_engine* e = q38_create(pack.c_str(), metallib.c_str(), &cfg, err, sizeof err);
  if (!e) { fprintf(stderr, "q38_create: %s\n", err); return 1; }
  printf("loaded in %.2fs; %.2f GB of weights read per token\n",
         [NSDate timeIntervalSinceReferenceDate] - t_load, q38_decode_read_gb(e));
  q38_set_layer_cap(e, cap);
  if (sample || !mtp_pack.empty()) q38_set_argmax(e, 0);
  const bool spec = !mtp_pack.empty() || !dspine_pack.empty();
  if (!dspine_pack.empty()) {
    if (q38_load_dspine(e, dspine_pack.c_str(), err, sizeof err)) { fprintf(stderr, "dspine: %s\n", err); return 1; }
    printf("DSpine drafter loaded: %.2f GB per block + LM head\n", q38_dspine_read_gb(e));
    if (depth < 1 || depth > 7) { fprintf(stderr, "--depth 1..7\n"); return 2; }
  } else if (spec) {
    if (q38_load_mtp(e, mtp_pack.c_str(), err, sizeof err)) { fprintf(stderr, "mtp: %s\n", err); return 1; }
    printf("MTP head loaded: %.2f GB per draft step + LM head\n", q38_mtp_read_gb(e));
    if (depth < 1 || depth > 7) { fprintf(stderr, "--depth 1..7\n"); return 2; }
  }
  const int vocab = q38_vocab(e), hidden = q38_hidden(e);

  if (snap_test) {
    // Prefix-cache check: run the first half, snapshot, run the second half;
    // then wreck the sequence (reset + the prompt reversed), load the snapshot
    // and run the second half again. Logits of the final step must agree.
    if (chunk < 1) return 2;
    q38_step_stats s2;
    const size_t half = toks.size() / 2;
    auto run = [&](const int32_t* t, size_t n) {
      for (size_t i = 0; i < n; i += chunk) {
        const int m = (int)std::min<size_t>(chunk, n - i);
        if (q38_step(e, t + i, m, &s2)) { fprintf(stderr, "step failed\n"); exit(1); }
        if (q38_commit(e, m)) { fprintf(stderr, "commit failed\n"); exit(1); }
      }
      return (int)std::min<size_t>(chunk, n - (n - 1) / chunk * chunk);
    };
    run(toks.data(), half);
    q38_snapshot* sn = q38_snapshot_save(e);
    if (!sn) { fprintf(stderr, "snapshot_save failed\n"); return 1; }
    printf("snapshot at %lld: %.1f MB\n", (long long)q38_snapshot_position(sn), q38_snapshot_bytes(sn) / 1e6);
    const int lm = run(toks.data() + half, toks.size() - half);
    std::vector<uint16_t> ref(q38_logits_bf16(e), q38_logits_bf16(e) + (size_t)lm * vocab);
    std::vector<int64_t> ra(q38_argmax(e), q38_argmax(e) + lm);
    q38_reset(e);
    std::vector<int32_t> rev(toks.rbegin(), toks.rend());
    run(rev.data(), rev.size());
    if (q38_snapshot_load(e, sn)) { fprintf(stderr, "snapshot_load failed\n"); return 1; }
    run(toks.data() + half, toks.size() - half);
    const uint16_t* got = q38_logits_bf16(e);
    auto bf = [](uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; };
    double maxd = 0; size_t ndiff = 0; int amis = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (ref[i] != got[i]) ++ndiff;
      maxd = std::max(maxd, (double)fabsf(bf(ref[i]) - bf(got[i])));
    }
    for (int i = 0; i < lm; ++i) amis += q38_argmax(e)[i] != ra[i];
    printf("snap-test: %d rows, %zu of %zu logits differ, max |diff| %.4f, argmax mismatches %d\n",
           lm, ndiff, ref.size(), maxd, amis);
    // Two loads of the same snapshot take the same arithmetic path and must
    // agree bit for bit, and so must the first comparison: the CPU replay of
    // the state log uses the same FMA form as the GPU kernel.
    std::vector<uint16_t> first(got, got + ref.size());
    q38_reset(e);
    run(rev.data(), rev.size() / 2);
    if (q38_snapshot_load(e, sn)) { fprintf(stderr, "snapshot_load failed\n"); return 1; }
    run(toks.data() + half, toks.size() - half);
    size_t rediff = 0;
    for (size_t i = 0; i < ref.size(); ++i) rediff += first[i] != q38_logits_bf16(e)[i];
    printf("snap-test: second load differs from the first in %zu logits\n", rediff);
    q38_snapshot_free(sn);
    q38_destroy(e);
    return (amis == 0 && ndiff == 0 && rediff == 0) ? 0 : 1;
  }

  if (big_test > 0) {
    // Prefill in chunks of `big_test` rows (the many-row path) against 8-row
    // chunks: K greedy decode steps follow each, the second fed the first's
    // tokens, and every decode row's logits are compared.
    const int K = 16, C = big_test;
    const bool hid = getenv("Q38_BT_HID") != nullptr;  // compare the final hidden rows (works under --cap)
    const int W = hid ? q38_hidden(e) : vocab;
    auto bf = [](uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; };
    auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    q38_step_stats st;
    std::vector<int32_t> tok;
    std::vector<uint16_t> lg;
    double ms[2] = {0, 0}, gpu[2] = {0, 0};
    double min_cos = 1, maxd = 0;
    int amis = 0;
    const bool skip_ref = getenv("Q38_BT_SKIP_REF") != nullptr;  // timing only: no 8-row pass, no comparison
    for (int pass = skip_ref ? 1 : 0; pass < 2; ++pass) {
      const int c = pass ? C : 8;
      q38_reset(e);
      // warm the pipelines so the timing is of steady passes
      const double t0 = now();
      int lm = 0;
      for (size_t p = 0; p < toks.size(); p += lm) {
        lm = (int)std::min<size_t>(c, toks.size() - p);
        q38_group g = {0, lm};
        if (q38_step_groups(e, &g, 1, toks.data() + p, nullptr, nullptr, &st) || q38_seq_commit(e, 0, lm)) {
          fprintf(stderr, "prefill failed (chunk %d at %zu)\n", lm, p);
          return 1;
        }
        gpu[pass] += st.gpu_ms;
        if (hid && getenv("Q38_BT_ALLROWS")) {  // every prefill row's final hidden, both passes
          static std::vector<uint16_t> all[2];
          const int H = q38_hidden(e);
          const uint16_t* f = q38_final_bf16(e);
          all[pass].insert(all[pass].end(), f, f + (size_t)lm * H);
          if (pass == 1 && p + lm >= toks.size()) {
            printf("big-test: all rows cos:");
            for (size_t r = 0; r < toks.size(); ++r) {
              const uint16_t* a = all[0].data() + r * H;
              const uint16_t* b = all[1].data() + r * H;
              double dot = 0, na = 0, nb = 0;
              for (int v = 0; v < H; ++v) { const double x = bf(a[v]), y = bf(b[v]); dot += x * y; na += x * x; nb += y * y; }
              printf(" %.3f", dot / std::sqrt(na * nb));
            }
            printf("\n");
          }
        }
      }
      ms[pass] = now() - t0;
      if (hid) {  // the prefill rows themselves: the tail of the last chunk
        static std::vector<uint16_t> ref_rows;
        static int ref_m = 0;
        const int H = q38_hidden(e);
        const uint16_t* f = q38_final_bf16(e);
        if (pass == 0) { ref_rows.assign(f, f + (size_t)lm * H); ref_m = lm; }
        else {
          const int n = std::min(ref_m, lm);
          printf("big-test: prefill tail rows cos:");
          for (int r = 0; r < n; ++r) {
            const uint16_t* a = ref_rows.data() + (size_t)(ref_m - n + r) * H;
            const uint16_t* b = f + (size_t)(lm - n + r) * H;
            double dot = 0, na = 0, nb = 0;
            for (int v = 0; v < H; ++v) { const double x = bf(a[v]), y = bf(b[v]); dot += x * y; na += x * x; nb += y * y; }
            printf(" %.5f", dot / std::sqrt(na * nb));
          }
          printf("\n");
        }
      }
      q38_sampling greedy = {0};
      greedy.temperature = 0;
      int32_t ids[4]; float pr[4]; int nn = 0;
      for (int k = 0; k < K; ++k) {
        int32_t t;
        if (pass == 0) {
          const int row = k == 0 ? q38_seq_out_rows(e, 0) - 1 : 0;
          const uint16_t* l = k == 0 ? q38_logits_bf16(e) + (size_t)(lm - 1) * vocab : q38_logits_bf16(e);
          (void)row; (void)ids; (void)pr; (void)nn;
          int best = 0;
          for (int v = 1; v < vocab; ++v) if (bf(l[v]) > bf(l[best])) best = v;
          t = hid ? toks[k] : best;
          tok.push_back(t);
        } else t = skip_ref ? toks[k] : tok[k];
        if (q38_step(e, &t, 1, &st) || q38_commit(e, 1)) { fprintf(stderr, "decode failed\n"); return 1; }
        const uint16_t* l = hid ? q38_final_bf16(e) : q38_logits_bf16(e);
        if (pass == 0) lg.insert(lg.end(), l, l + W);
        else if (!skip_ref) {
          const uint16_t* a = lg.data() + (size_t)k * W;
          double dot = 0, na = 0, nb = 0; int ia = 0, ib = 0;
          for (int v = 0; v < W; ++v) {
            const double x = bf(a[v]), y = bf(l[v]);
            dot += x * y; na += x * x; nb += y * y;
            maxd = std::max(maxd, std::fabs(x - y));
            if (x > bf(a[ia])) ia = v;
            if (y > bf(l[ib])) ib = v;
          }
          min_cos = std::min(min_cos, dot / std::sqrt(na * nb));
          amis += ia != ib;
        }
      }
    }
    printf("big-test: %zu prompt tokens; 8-row prefill %.0f ms (%.1f tok/s, gpu %.0f), %d-row prefill %.0f ms (%.1f tok/s, gpu %.0f)\n",
           toks.size(), ms[0], toks.size() * 1e3 / ms[0], gpu[0], C, ms[1], toks.size() * 1e3 / ms[1], gpu[1]);
    printf("big-test: %d decode rows, min cos %.6f, max |dlogit| %.3f, argmax mismatches %d\n", K, min_cos, maxd, amis);
    return 0;
  }
  if (multi_test > 0) {
    // Several sequences in shared steps against each one alone. Sequence i
    // gets the prompt minus its last 5 * i tokens. Alone: chunked prefill,
    // then K greedy steps. Together: the prefills interleaved (so runs of
    // different lengths share steps and sequences finish at different times),
    // then K steps of one row per sequence, fed the tokens the solo run chose.
    // The logits of every decode row are compared with the solo run's.
    const int N = multi_test, K = 24;
    if (N < 1 || N > 8 || (size_t)(5 * N) >= toks.size()) return 2;
    auto bf = [](uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; };
    std::vector<std::vector<int32_t>> prompt(N), solo_tok(N);
    std::vector<std::vector<uint16_t>> solo_lg(N);
    q38_step_stats st;
    double solo_ms = 0;
    for (int i = 0; i < N; ++i) {
      prompt[i].assign(toks.begin(), toks.end() - 5 * i);
      q38_reset(e);
      int lm = 0;
      for (size_t p = 0; p < prompt[i].size(); p += lm) {
        lm = (int)std::min<size_t>(8, prompt[i].size() - p);
        if (q38_step(e, prompt[i].data() + p, lm, &st) || q38_commit(e, lm)) { fprintf(stderr, "solo prefill failed\n"); return 1; }
      }
      int32_t t = (int32_t)q38_argmax(e)[lm - 1];
      for (int k = 0; k < K; ++k) {
        solo_tok[i].push_back(t);
        if (q38_step(e, &t, 1, &st) || q38_commit(e, 1)) { fprintf(stderr, "solo step failed\n"); return 1; }
        solo_ms += st.gpu_ms;
        solo_lg[i].insert(solo_lg[i].end(), q38_logits_bf16(e), q38_logits_bf16(e) + vocab);
        t = (int32_t)q38_argmax(e)[0];
      }
    }
    std::vector<int> id(N);
    id[0] = 0;
    q38_reset(e);
    for (int i = 1; i < N; ++i) {
      id[i] = q38_seq_new(e);
      if (id[i] < 0) { fprintf(stderr, "seq_new failed\n"); return 1; }
    }
    std::vector<size_t> at(N, 0);
    const int per = std::max(1, 8 / N);
    for (;;) {
      q38_group g[8];
      int32_t row[8];
      int ng = 0, m = 0;
      for (int i = 0; i < N && m < 8; ++i) {
        const int take = (int)std::min<size_t>({(size_t)per, prompt[i].size() - at[i], (size_t)(8 - m)});
        if (take <= 0) continue;
        g[ng++] = {id[i], take};
        for (int j = 0; j < take; ++j) row[m++] = prompt[i][at[i] + j];
        at[i] += take;
      }
      if (!ng) break;
      if (q38_step_groups(e, g, ng, row, nullptr, nullptr, &st)) { fprintf(stderr, "group prefill failed\n"); return 1; }
      for (int j = 0; j < ng; ++j)
        if (q38_seq_commit(e, g[j].seq, g[j].m)) { fprintf(stderr, "group commit failed\n"); return 1; }
    }
    double maxd = 0, min_cos = 1, multi_ms = 0;
    int amis = 0;
    for (int k = 0; k < K; ++k) {
      q38_group g[8];
      int32_t row[8];
      for (int i = 0; i < N; ++i) {
        g[i] = {id[i], 1};
        row[i] = solo_tok[i][k];
      }
      if (q38_step_groups(e, g, N, row, nullptr, nullptr, &st)) { fprintf(stderr, "group step failed\n"); return 1; }
      multi_ms += st.gpu_ms;
      for (int i = 0; i < N; ++i) {
        if (q38_seq_commit(e, id[i], 1)) { fprintf(stderr, "group commit failed\n"); return 1; }
        const uint16_t* a = solo_lg[i].data() + (size_t)k * vocab;
        const uint16_t* b = q38_logits_bf16(e) + (size_t)i * vocab;
        double dot = 0, na = 0, nb = 0;
        int ia = 0, ib = 0;
        for (int v = 0; v < vocab; ++v) {
          const double x = bf(a[v]), y = bf(b[v]);
          dot += x * y; na += x * x; nb += y * y;
          maxd = std::max(maxd, fabs(x - y));
          if (bf(a[v]) > bf(a[ia])) ia = v;
          if (bf(b[v]) > bf(b[ib])) ib = v;
        }
        min_cos = std::min(min_cos, dot / sqrt(na * nb));
        amis += ia != ib;
        if (getenv("Q38_MT_VERBOSE"))
          printf("  seq %d step %2d cos %.6f argmax %d/%d top logit %.2f/%.2f\n", i, k, dot / sqrt(na * nb), ia, ib,
                 bf(a[ia]), bf(b[ib]));
      }
    }
    {
      // Isolation: the same joint run with every other sequence replaced by
      // unrelated content must leave sequence 0's logits bit-identical.
      std::vector<uint16_t> first, second;
      for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < N; ++i) q38_seq_reset(e, id[i]);
        std::vector<std::vector<int32_t>> pr = prompt;
        if (pass)
          for (int i = 1; i < N; ++i) std::reverse(pr[i].begin(), pr[i].end());
        std::vector<size_t> at2(N, 0);
        for (;;) {
          q38_group g[8];
          int32_t row[8];
          int ng = 0, m = 0;
          for (int i = 0; i < N && m < 8; ++i) {
            const int take = (int)std::min<size_t>({(size_t)per, pr[i].size() - at2[i], (size_t)(8 - m)});
            if (take <= 0) continue;
            g[ng++] = {id[i], take};
            for (int j = 0; j < take; ++j) row[m++] = pr[i][at2[i] + j];
            at2[i] += take;
          }
          if (!ng) break;
          if (q38_step_groups(e, g, ng, row, nullptr, nullptr, &st)) { fprintf(stderr, "iso prefill failed\n"); return 1; }
          for (int j = 0; j < ng; ++j) q38_seq_commit(e, g[j].seq, g[j].m);
        }
        std::vector<uint16_t>& out = pass ? second : first;
        for (int k = 0; k < K; ++k) {
          q38_group g[8];
          int32_t row[8];
          for (int i = 0; i < N; ++i) {
            g[i] = {id[i], 1};
            row[i] = (pass && i) ? 1000 + 37 * i + k : solo_tok[i][k];
          }
          if (q38_step_groups(e, g, N, row, nullptr, nullptr, &st)) { fprintf(stderr, "iso step failed\n"); return 1; }
          for (int i = 0; i < N; ++i) q38_seq_commit(e, id[i], 1);
          out.insert(out.end(), q38_logits_bf16(e), q38_logits_bf16(e) + vocab);
        }
      }
      size_t nd = 0;
      for (size_t i = 0; i < first.size(); ++i) nd += first[i] != second[i];
      printf("multi-test: isolation, sequence 0 with other sequences replaced: %zu of %zu logits differ\n", nd, first.size());
      if (nd) amis = N * K;
    }
    for (int i = 0; i < N; ++i)
      if (q38_seq_position(e, id[i]) != (int64_t)prompt[i].size() + K) { fprintf(stderr, "position mismatch\n"); return 1; }
    printf("multi-test: %d sequences x %d decode rows: min cosine %.6f, max |diff| %.3f, argmax mismatches %d of %d\n",
           N, K, min_cos, maxd, amis, N * K);
    printf("multi-test: gpu per step %.2f ms for %d rows (%.1f tok/s aggregate) vs %.2f ms alone (%.1f tok/s)\n",
           multi_ms / K, N, N * K * 1e3 / multi_ms, solo_ms / (N * K), N * K * 1e3 / solo_ms);
    for (int i = 1; i < N; ++i) q38_seq_free(e, id[i]);
    q38_destroy(e);
    return amis <= N * K / 20 ? 0 : 1;  // isolation failures set amis = N * K
  }

  if (pool_bench > 0) {
    // N streams of the same prompt (different seeds) through the stream pool
    // (q38/pool.h); stream i opens `stagger` cycles after stream i - 1, so
    // with --stagger prompts are prefilled while others decode. With one
    // stream the tokens must equal the single-stream generator's.
    const int N = pool_bench;
    if (N < 1 || N > 8) return 2;
    std::vector<int32_t> ref;
    if (N == 1) {
      q38_gen_config gc;
      q38_gen_config_default(&gc);
      gc.depth = spec ? depth : 0;
      gc.adaptive = (float)adaptive;
      gc.draft = {(float)draft_temp, draft_topk, 0.95f};
      gc.seed = seed;
      gc.snapshot_prompt_end = 0;
      q38_gen* g = q38_gen_create(e, &gc);
      if (!g || q38_gen_prompt(g, toks.data(), (int)toks.size(), nullptr, 0)) { fprintf(stderr, "generator failed\n"); return 1; }
      int32_t buf[8];
      while ((int)ref.size() < gen) {
        const int n = q38_gen_next(g, buf, std::min(8, gen - (int)ref.size()));
        if (n <= 0) break;
        ref.insert(ref.end(), buf, buf + n);
      }
      q38_gen_free(g);
    }
    q38_pool_config pc;
    q38_pool_config_default(&pc);
    pc.depth = spec ? depth : 0;
    pc.adaptive = (float)adaptive;
    pc.draft = {(float)draft_temp, draft_topk, 0.95f};
    pc.max_streams = N;
    if (getenv("Q38_MTP_TAIL")) pc.mtp_prompt_tail = atoi(getenv("Q38_MTP_TAIL"));
    if (getenv("Q38_BIG_EVERY")) pc.big_every = atoi(getenv("Q38_BIG_EVERY"));
    if (getenv("Q38_BIG_ROWS")) pc.big_rows = atoi(getenv("Q38_BIG_ROWS"));
    q38_pool* pool = q38_pool_create(e, &pc);
    if (!pool) { fprintf(stderr, "q38_pool_create failed\n"); return 1; }
    std::vector<std::vector<int32_t>> out(N);
    std::vector<double> t_open(N, 0), t_first(N, 0), t_done(N, 0);
    std::vector<int> sid(N, -1);
    int opened = 0, finished = 0;
    long cycle = 0;
    auto now = []() { return [NSDate timeIntervalSinceReferenceDate]; };
    const double t0 = now();
    double t_all = 0;  // when the last stream finished its prompt
    long tok_all = 0;
    while (finished < N) {
      while (opened < N && cycle >= (long)opened * stagger) {
        q38_pool_request r;
        memset(&r, 0, sizeof r);
        r.tokens = toks.data();
        r.n = (int)toks.size();
        r.sampling = {1.0f, 20, 0.95f};
        r.seed = seed + (uint64_t)opened;
        r.eos[0] = 248046; r.eos[1] = 248044; r.n_eos = 2;
        r.max_tokens = gen;
        sid[opened] = q38_pool_open(pool, &r);
        if (sid[opened] < 0) { fprintf(stderr, "q38_pool_open failed: %d\n", sid[opened]); return 1; }
        t_open[opened] = now();
        ++opened;
      }
      q38_pool_event ev[8];
      const int n = q38_pool_step(pool, ev, 8);
      if (n < 0) { fprintf(stderr, "q38_pool_step failed: %d\n", n); return 1; }
      ++cycle;
      const double t = now();
      for (int k = 0; k < n; ++k) {
        int i = 0;
        while (i < N && sid[i] != ev[k].stream) ++i;
        if (ev[k].n && out[i].empty()) t_first[i] = t;
        out[i].insert(out[i].end(), ev[k].tokens, ev[k].tokens + ev[k].n);
        if (t_all > 0) tok_all += ev[k].n;
        if (ev[k].done) {
          t_done[i] = t;
          ++finished;
        }
      }
      if (t_all == 0 && opened == N) {
        bool all = true;
        for (int i = 0; i < N; ++i) all = all && !out[i].empty();
        if (all) t_all = t;
      }
    }
    const double t1 = now();
    q38_pool_stats ps;
    q38_pool_get_stats(pool, &ps);
    long total = 0;
    for (int i = 0; i < N; ++i) {
      total += (long)out[i].size();
      printf("  stream %d: %zu tokens, first token after %.2f s, %.2f tok/s after that\n", i, out[i].size(),
             t_first[i] - t_open[i], (out[i].size() - 1) / std::max(1e-9, t_done[i] - t_first[i]));
    }
    double t_last = 0;
    for (int i = 0; i < N; ++i) t_last = std::max(t_last, t_done[i]);
    printf("pool-bench: %d streams, %ld tokens in %.2f s wall (%.2f tok/s including prefill)\n", N, total, t1 - t0,
           total / (t1 - t0));
    if (t_last > t_all && tok_all > 0)
      printf("pool-bench: all streams decoding: %.2f tok/s aggregate\n", tok_all / (t_last - t_all));
    printf("pool-bench: %lld passes, %.2f rows/pass, target %.2f ms/pass, draft %.2f ms/pass, acceptance %.1f%% of %lld drafts\n",
           (long long)ps.cycles, (double)ps.rows / ps.cycles, ps.target_ms / ps.cycles, ps.draft_ms / ps.cycles,
           ps.drafted ? 100.0 * ps.kept / ps.drafted : 0.0, (long long)ps.drafted);
    int rc = 0;
    if (N == 1) {
      const size_t cmpn = std::min(ref.size(), out[0].size()) > 8 ? std::min(ref.size(), out[0].size()) - 8 : 0;
      const bool same = cmpn > 0 && std::equal(ref.begin(), ref.begin() + (long)cmpn, out[0].begin());
      printf("pool-bench: one stream vs the generator API (%zu tokens): %s\n", cmpn, same ? "IDENTICAL" : "DIFFERENT");
      rc = same ? 0 : 1;
    }
    if (!out_tokens.empty()) {
      FILE* f = fopen(out_tokens.c_str(), "w");
      for (int i = 0; i < N; ++i) {
        for (int32_t t : out[i]) fprintf(f, "%d ", t);
        fprintf(f, "\n");
      }
      fclose(f);
    }
    for (int i = 0; i < N; ++i) q38_pool_close(pool, sid[i]);
    q38_pool_free(pool);
    q38_destroy(e);
    return rc;
  }

  if (!dsp_accept.empty()) {
    // Teacher-forced drafter check: the token file is prompt (first --plen
    // tokens) + continuation. For every anchor after the prompt, draft 7
    // tokens with the chain forced along the file and write, per draft
    // position, q(true token) and whether it was a candidate.
    const int N = (int)toks.size();
    q38_step_stats ss;
    for (int i = 0; i < plen;) {
      const int m = std::min(8, plen - i);
      if (q38_step(e, toks.data() + i, m, &ss) || q38_commit(e, m)) { fprintf(stderr, "prefill failed\n"); return 1; }
      i += m;
    }
    FILE* f = fopen(dsp_accept.c_str(), "w");
    const q38_sampling dsp = {(float)draft_temp, draft_topk, 0.95f};
    for (int a = plen; a + 1 < N; ++a) {
      const int k = std::min(7, N - a - 1);
      q38_draft_row dr[8];
      double uu[8] = {0};
      q38_dspine_force(e, toks.data() + a + 1, k);
      const int D = q38_seq_dspine_draft(e, 0, toks[a], k, &dsp, 0.f, uu, dr, &ss);
      if (D < 0) { fprintf(stderr, "draft failed: %d\n", D); return 1; }
      for (int i = 0; i < 7; ++i) {
        float q = -1;
        if (i < D) {
          q = 0;
          for (int j = 0; j < dr[i].n; ++j)
            if (dr[i].ids[j] == toks[a + 1 + i]) q = dr[i].probs[j];
        }
        fprintf(f, "%s%.6f", i ? " " : "", q);
      }
      fprintf(f, "\n");
      if (q38_step(e, toks.data() + a, 1, &ss) || q38_commit(e, 1)) { fprintf(stderr, "step failed\n"); return 1; }
    }
    fclose(f);
    printf("wrote %s: %d anchors\n", dsp_accept.c_str(), N - 1 - plen);
    q38_destroy(e);
    return 0;
  }

  if (gen_api) {
    // Decode through the library generator (q38/generate.h): same draws as
    // the in-file loop below for the same seed. Then the prompt again, which
    // must come back from the prompt-end snapshot with identical tokens, and
    // once more after an unrelated prompt has replaced the sequence.
    q38_gen_config gc;
    q38_gen_config_default(&gc);
    gc.depth = spec ? depth : 0;
    gc.adaptive = (float)adaptive;
    gc.draft = {(float)draft_temp, draft_topk, 0.95f};
    gc.chunk = chunk;
    gc.seed = seed;
    q38_gen* g = q38_gen_create(e, &gc);
    if (!g) { fprintf(stderr, "q38_gen_create failed\n"); return 1; }
    auto decode = [&](std::vector<int32_t>& o, double* secs) {
      const double t = [NSDate timeIntervalSinceReferenceDate];
      int32_t buf[8];
      while ((int)o.size() < gen) {
        const int n = q38_gen_next(g, buf, std::min(8, gen - (int)o.size()));
        if (n < 0) { fprintf(stderr, "q38_gen_next failed: %d\n", n); exit(1); }
        if (n == 0) break;
        o.insert(o.end(), buf, buf + n);
      }
      *secs = [NSDate timeIntervalSinceReferenceDate] - t;
    };
    const int marks[1] = {(int)toks.size() / 2};
    if (q38_gen_prompt(g, toks.data(), (int)toks.size(), marks, 1)) { fprintf(stderr, "prompt failed\n"); return 1; }
    q38_gen_stats gs;
    q38_gen_get_stats(g, &gs);
    printf("prefill: %lld tokens (%lld reused) in %.1f ms\n", (long long)gs.prompt_tokens, (long long)gs.reused_tokens, gs.prefill_ms);
    std::vector<int32_t> o1, o2, o3;
    double s1, s2, s3;
    decode(o1, &s1);
    q38_gen_get_stats(g, &gs);
    printf("%sdecode (generator API, depth %d, sampled): %zu tokens in %.3f s = %.2f tok/s end to end\n",
           spec ? "spec " : "", gc.depth, o1.size(), s1, o1.size() / s1);
    printf("  cycles %lld  tokens/cycle %.3f  acceptance %.1f%%  target %.2f ms/cycle  draft %.2f ms/cycle  "
           "snapshots %d (%.0f MB)\n", (long long)gs.cycles, (double)gs.emitted / gs.cycles,
           gs.drafted ? 100.0 * gs.kept / gs.drafted : 0.0, gs.target_ms / gs.cycles, gs.draft_ms / gs.cycles,
           gs.snapshots, gs.snapshot_bytes / 1e6);
    if (q38_has_dspine(e)) {
      double dg, dt, dw;
      int64_t dc;
      q38_dspine_times(e, &dg, &dt, &dw, &dc);
      printf("  drafter per call: gpu %.2f ms  host tail %.2f ms  wall %.2f ms  (%lld calls)\n",
             dg / dc, dt / dc, dw / dc, (long long)dc);
    }
    if (gen_once) {
      if (!out_tokens.empty()) {
        FILE* f = fopen(out_tokens.c_str(), "w");
        for (int32_t t : o1) fprintf(f, "%d\n", t);
        fclose(f);
      }
      q38_gen_free(g);
      q38_destroy(e);
      return 0;
    }
    q38_gen_reseed(g, seed);
    if (q38_gen_prompt(g, toks.data(), (int)toks.size(), nullptr, 0)) return 1;
    q38_gen_get_stats(g, &gs);
    printf("same prompt again: %lld of %lld tokens reused, %.1f ms\n", (long long)gs.reused_tokens,
           (long long)gs.prompt_tokens, gs.prefill_ms);
    decode(o2, &s2);
    std::vector<int32_t> rev(toks.rbegin(), toks.rend());
    if (q38_gen_prompt(g, rev.data(), (int)rev.size(), nullptr, 0)) return 1;
    std::vector<int32_t> junk;
    { int keep = gen; gen = std::min(gen, 12); double sj; decode(junk, &sj); gen = keep; }
    // a prompt that shares only the first half: must resume from the mid snapshot
    std::vector<int32_t> half(toks.begin(), toks.begin() + marks[0]);
    half.insert(half.end(), toks.begin() + marks[0], toks.end());
    q38_gen_reseed(g, seed);
    if (q38_gen_prompt(g, half.data(), (int)half.size(), nullptr, 0)) return 1;
    q38_gen_get_stats(g, &gs);
    printf("after an unrelated prompt: %lld of %lld tokens reused, %.1f ms\n", (long long)gs.reused_tokens,
           (long long)gs.prompt_tokens, gs.prefill_ms);
    decode(o3, &s3);
    printf("gen-api: repeat %s, repeat after unrelated prompt %s\n", o1 == o2 ? "IDENTICAL" : "DIFFERENT",
           o1 == o3 ? "IDENTICAL" : "DIFFERENT");
    if (!out_tokens.empty()) {
      FILE* f = fopen(out_tokens.c_str(), "w");
      for (int32_t t : o1) fprintf(f, "%d ", t);
      fclose(f);
    }
    q38_gen_free(g);
    q38_destroy(e);
    return (o1 == o2 && o1 == o3) ? 0 : 1;
  }

  q38_step_stats st;
  int last_m = 0;
  // --dump-all: raw BF16 logits of every prefill row, [tokens, vocab]
  FILE* fall = dump_all.empty() ? nullptr : fopen(dump_all.c_str(), "wb");
  double pre_wall = 0, pre_gpu = 0;
  if (chunk < 1) { fprintf(stderr, "--chunk must be >= 1\n"); return 2; }
  // --big-head N: the first N tokens go in as 64-row prefill chunks (no logits:
  // --dump-all gets zero rows for them), the rest as usual. Scores the state a
  // big prefill leaves behind (tools/kld.py engine --big-head).
  size_t head = std::min<size_t>(big_head > 0 ? big_head : 0, toks.size());
  for (size_t i = 0; i < head;) {
    static const int head_chunk = getenv("Q38_BIG_HEAD_CHUNK") ? atoi(getenv("Q38_BIG_HEAD_CHUNK")) : 64;
    const int m = (int)std::min<size_t>((size_t)head_chunk, head - i);
    q38_group g = {0, m};
    if (q38_step_groups(e, &g, 1, toks.data() + i, nullptr, nullptr, &st) || q38_seq_commit(e, 0, m)) {
      fprintf(stderr, "big head step failed\n");
      return 1;
    }
    pre_wall += st.wall_ms;
    pre_gpu += st.gpu_ms;
    if (fall) {
      std::vector<uint16_t> z((size_t)m * vocab, 0);
      fwrite(z.data(), 2, z.size(), fall);
    }
    i += m;
  }
  for (size_t i = head; i < toks.size(); i += chunk) {
    const int m = (int)std::min<size_t>(chunk, toks.size() - i);
    const int rc = q38_step(e, toks.data() + i, m, &st);
    if (rc) { fprintf(stderr, "step failed: %d\n", rc); return 1; }
    pre_wall += st.wall_ms;
    pre_gpu += st.gpu_ms;
    last_m = m;
    if (fall) fwrite(q38_logits_bf16(e), 2, (size_t)m * vocab, fall);
    if (spec && i + m < toks.size()) {
      // pair (hidden of row r, the token after it), all known inside the prompt
      int rows[8];
      for (int r = 0; r < m; ++r) rows[r] = r;
      q38_step_stats ms;
      if (q38_mtp_feed(e, toks.data() + i + 1, rows, m, 0, &ms)) { fprintf(stderr, "mtp feed failed\n"); return 1; }
      pre_wall += ms.wall_ms;
      pre_gpu += ms.gpu_ms;
    }
    if (!fall) printf("prefill rows %zu..%zu  m=%d  wall %.1f ms  gpu %.1f ms  encode %.2f ms  "
           "dispatches %d  barriers %d\n",
           i, i + m - 1, m, st.wall_ms, st.gpu_ms, st.encode_ms, st.dispatches, st.barriers);
  }
  if (fall) fclose(fall);
  printf("prefill: %zu tokens, %.1f ms wall (%.1f tok/s), %.1f ms gpu\n", toks.size(), pre_wall,
         toks.size() * 1e3 / pre_wall, pre_gpu);

  if (!dump.empty()) {
    const int width = cap ? hidden : vocab;
    const uint16_t* src = cap ? q38_final_bf16(e) : q38_logits_bf16(e);
    std::vector<float> out((size_t)last_m * width);
    for (size_t j = 0; j < out.size(); ++j) out[j] = bf16(src[j]);
    FILE* f = fopen(dump.c_str(), "wb");
    fwrite(out.data(), 4, out.size(), f);
    fclose(f);
    printf("dumped %d x %d float32 to %s\n", last_m, width, dump.c_str());
  }
  if (cap) { q38_destroy(e); return 0; }

  if (!sample && mtp_pack.empty()) {
    printf("argmax after prefill:");
    for (int i = 0; i < last_m; ++i) printf(" %lld", q38_argmax(e)[i]);
    printf("\n");
  }

  if (gen > 0 && spec) {
    // Speculative decode with the MTP head, lossless against the target's
    // filtered distribution (temperature 1.0, top-k 20, top-p 0.95): each
    // draft d ~ q is kept with probability min(1, p(d)/q(d)); on rejection
    // the token is redrawn from the residual (p - q)+, and when every draft
    // is kept one more token comes from the last verify row.
    const q38_sampling sp = {1.0f, 20, 0.95f};
    uint64_t rs = seed;
    auto uni = [&]() {
      rs += 0x9E3779B97F4A7C15ull;
      uint64_t z = rs;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      z ^= z >> 31;
      return (double)(z >> 11) / 9007199254740992.0;
    };
    const q38_sampling dsp = {(float)draft_temp, draft_topk, 0.95f};
    long depth_hist[9] = {0};
    std::vector<int32_t> outv;
    int32_t x_new = q38_sample(e, last_m - 1, &sp, uni(), nullptr, nullptr, nullptr);
    // pending committed-history feed: rows of the last target step + tokens after them
    int feed_rows[8], feed_n = last_m;
    int32_t feed_tok[8];
    const size_t tail = toks.size() - last_m;
    for (int r = 0; r < last_m; ++r) {
      feed_rows[r] = r;
      feed_tok[r] = r + 1 < last_m ? toks[tail + r + 1] : x_new;
    }
    long cycles = 0, drafted[8] = {0}, kept[8] = {0};
    double t_target = 0, t_mtp = 0, g_target = 0, g_mtp = 0;
    bool done = false;
    const double t_gen = [NSDate timeIntervalSinceReferenceDate];
    while (!done && (int)outv.size() < gen) {
      outv.push_back(x_new);
      if (x_new == 248046 || x_new == 248044) break;
      int32_t d[8];
      int32_t qid[8][64];
      float qp[8][64];
      int qn[8];
      q38_step_stats ms;
      if (q38_mtp_feed(e, feed_tok, feed_rows, feed_n, 1, &ms)) { fprintf(stderr, "mtp feed failed\n"); return 1; }
      t_mtp += ms.wall_ms; g_mtp += ms.gpu_ms;
      // Draft up to `depth` tokens; with --adaptive stop at the first draft
      // the head itself is unsure of (a deeper draft only pays when the ones
      // before it are likely to be kept).
      int D = 0;
      while (D < depth) {
        d[D] = q38_mtp_sample(e, &dsp, uni(), qid[D], qp[D], &qn[D]);
        float qd = 0;
        for (int i = 0; i < qn[D]; ++i) if (qid[D][i] == d[D]) qd = qp[D][i];
        ++D;
        if (D == depth || qd < adaptive) break;
        if (q38_mtp_chain(e, d[D - 1], &ms)) { fprintf(stderr, "mtp chain failed\n"); return 1; }
        t_mtp += ms.wall_ms; g_mtp += ms.gpu_ms;
      }
      ++depth_hist[D];
      int32_t rows_tok[8];
      rows_tok[0] = x_new;
      for (int j = 0; j < D; ++j) rows_tok[j + 1] = d[j];
      if (q38_step(e, rows_tok, D + 1, &st)) { fprintf(stderr, "verify step failed\n"); return 1; }
      t_target += st.wall_ms; g_target += st.gpu_ms;
      ++cycles;
      int acc = 0;
      int32_t next = -1;
      for (int j = 0; j < D; ++j) {
        int32_t pid[64];
        float pp[64];
        int pn;
        q38_sample(e, j, &sp, 0.0, pid, pp, &pn);
        float pd = 0, qd = 0;
        for (int i = 0; i < pn; ++i) if (pid[i] == d[j]) pd = pp[i];
        for (int i = 0; i < qn[j]; ++i) if (qid[j][i] == d[j]) qd = qp[j][i];
        ++drafted[j];
        if (uni() * qd < pd) {  // accept with probability min(1, p/q)
          ++kept[j];
          ++acc;
          outv.push_back(d[j]);
          if (d[j] == 248046 || d[j] == 248044 || (int)outv.size() >= gen) { done = true; break; }
          continue;
        }
        // residual (p - q)+ over p's support
        double r[64], rsum = 0;
        for (int i = 0; i < pn; ++i) {
          float qi = 0;
          for (int t = 0; t < qn[j]; ++t) if (qid[j][t] == pid[i]) qi = qp[j][t];
          r[i] = std::max(0.0, (double)pp[i] - qi);
          rsum += r[i];
        }
        double u = uni() * rsum, c = 0;
        next = pid[pn - 1];
        for (int i = 0; i < pn; ++i) { c += r[i]; if (u < c) { next = pid[i]; break; } }
        break;
      }
      if (done) break;
      if (next < 0) next = q38_sample(e, D, &sp, uni(), nullptr, nullptr, nullptr);  // bonus token
      q38_commit(e, 1 + acc);
      // committed pairs: (row 0, d1) ... (row acc-1, d_acc), (row acc, next)
      feed_n = acc + 1;
      for (int r = 0; r < feed_n; ++r) {
        feed_rows[r] = r;
        feed_tok[r] = r < acc ? d[r] : next;
      }
      x_new = next;
    }
    const double dt = [NSDate timeIntervalSinceReferenceDate] - t_gen;
    if (!out_tokens.empty()) {
      FILE* f = fopen(out_tokens.c_str(), "w");
      for (int32_t t : outv) fprintf(f, "%d ", t);
      fclose(f);
    }
    printf("spec decode (MTP depth %d, adaptive %.2f, sampled): %zu tokens in %.3f s = %.2f tok/s "
           "end to end\n", depth, adaptive, outv.size(), dt, outv.size() / dt);
    printf("  drafts per cycle:");
    for (int j = 1; j <= depth; ++j) printf(" %d:%ld", j, depth_hist[j]);
    printf("\n");
    printf("  cycles %ld  tokens/cycle %.3f  acceptance by draft position:", cycles,
           (double)outv.size() / std::max(1L, cycles));
    for (int j = 0; j < depth; ++j) printf(" %.1f%%", drafted[j] ? 100.0 * kept[j] / drafted[j] : 0.0);
    printf("\n  per cycle: verify %.2f ms wall (%.2f gpu), MTP drafting %.2f ms wall (%.2f gpu), "
           "other %.2f ms\n", t_target / cycles, g_target / cycles, t_mtp / cycles, g_mtp / cycles,
           (dt * 1e3 - t_target - t_mtp) / cycles);
  } else
  if (gen > 0) {
    // --sample: the model's default sampling (temperature 1.0, top-k 20,
    // top-p 0.95), seeded; stops at EOS. Without it: greedy, a parity and
    // timing instrument only.
    const q38_sampling sp = {1.0f, 20, 0.95f};
    uint64_t rs = seed;
    auto uni = [&]() {
      rs += 0x9E3779B97F4A7C15ull;
      uint64_t z = rs;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      z ^= z >> 31;
      return (double)(z >> 11) / 9007199254740992.0;
    };
    auto pick = [&](int row) -> int32_t {
      return sample ? q38_sample(e, row, &sp, uni(), nullptr, nullptr, nullptr)
                    : (int32_t)q38_argmax(e)[row];
    };
    int32_t next = pick(last_m - 1);
    std::vector<double> wall, gpu, enc;
    std::vector<int32_t> outv;
    const double t_gen = [NSDate timeIntervalSinceReferenceDate];
    for (int i = 0; i < gen; ++i) {
      outv.push_back(next);
      if (sample && (next == 248046 || next == 248044)) break;
      if (q38_step(e, &next, 1, &st)) { fprintf(stderr, "\nstep failed\n"); return 1; }
      wall.push_back(st.wall_ms);
      gpu.push_back(st.gpu_ms);
      enc.push_back(st.encode_ms);
      next = pick(0);
    }
    const double dt = [NSDate timeIntervalSinceReferenceDate] - t_gen;
    if (out_tokens.empty()) {
      printf("generated:");
      for (int32_t t : outv) printf(" %d", t);
      printf("\n");
    } else {
      FILE* f = fopen(out_tokens.c_str(), "w");
      for (int32_t t : outv) fprintf(f, "%d ", t);
      fclose(f);
    }
    auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    printf("decode%s: %zu tokens in %.3f s = %.2f tok/s end to end  step wall median %.2f ms  "
           "gpu median %.2f ms  encode median %.3f ms  (%d dispatches, %d barriers per step)\n",
           sample ? " (sampled)" : " (greedy)", wall.size(), dt, wall.size() / dt, med(wall),
           med(gpu), med(enc), st.dispatches, st.barriers);
    printf("effective bandwidth at the gpu median: %.0f GB/s\n",
           q38_decode_read_gb(e) / (med(gpu) / 1e3));
  }
  (void)bench;
  q38_destroy(e);
  return 0;
}
