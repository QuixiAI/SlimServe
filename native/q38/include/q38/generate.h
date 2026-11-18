// q38 generator: prompt handling with prefix snapshots and the decode loop
// (plain, or speculative with the MTP head) on top of the engine C ABI.
// One generator drives one engine's single sequence.
#pragma once
#include <stdint.h>

#include "q38/engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38_gen q38_gen;

typedef struct {
  q38_sampling sampling;      // target distribution (model default 1.0 / 20 / 0.95)
  q38_sampling draft;         // MTP draft distribution
  int depth;                  // MTP drafts per cycle, 0 = plain decode (needs q38_load_mtp when > 0)
  float adaptive;             // stop drafting once a draft's own probability is below this
  int chunk;                  // prefill rows per step (1..8)
  uint64_t seed;
  int32_t eos[4];             // end-of-sequence token ids
  int n_eos;
  int snapshot_prompt_end;    // keep a snapshot at the end of every prompt
  int64_t max_snapshot_bytes; // host memory budget for snapshots (least recently used dropped)
} q38_gen_config;

typedef struct {
  int64_t prompt_tokens;      // length of the last prompt
  int64_t reused_tokens;      // of which taken from a snapshot or the live sequence
  double prefill_ms;
  int64_t cycles, emitted;    // decode cycles and tokens since the last prompt
  int64_t drafted, kept;      // MTP drafts verified / accepted
  double target_ms, draft_ms; // wall time in target steps and in the MTP head
  int snapshots;
  int64_t snapshot_bytes;
} q38_gen_stats;

void q38_gen_config_default(q38_gen_config* c);
q38_gen* q38_gen_create(q38_engine* e, const q38_gen_config* c);
void q38_gen_free(q38_gen* g);

// Make `tokens[0..n)` the sequence and get ready to decode after it. The
// longest usable prefix is reused: the live sequence if the prompt extends
// it, otherwise the longest stored snapshot whose tokens are a prefix of the
// prompt; the rest is prefilled. `snap_at` lists prompt lengths (ascending,
// 1..n) at which to store a snapshot on the way, e.g. message boundaries.
// Returns 0 on success.
int q38_gen_prompt(q38_gen* g, const int32_t* tokens, int n, const int* snap_at, int n_snap);

// One decode cycle. Writes 1..cap tokens (cap >= 1, at most 8 are produced)
// and returns the count; 0 once an end-of-sequence token has been returned
// (it is returned as the last token of its cycle); negative on error.
int q38_gen_next(q38_gen* g, int32_t* out, int cap);

void q38_gen_reseed(q38_gen* g, uint64_t seed);
// Target sampling for the following prompts (per-request settings). A
// temperature of 0 selects the most likely token; top_k is clamped to 1..64.
void q38_gen_set_sampling(q38_gen* g, const q38_sampling* s);
// Prompt with image rows. A negative token marks a row whose input embedding
// comes from `embeds` ([n_emb][hidden] BF16; the k-th negative token takes row
// k). The negative value itself is the row's identity for prefix reuse, so the
// caller derives it from the image content (two different images must not
// share values). `pos3` ([n][3], may be NULL for text-only prompts) gives the
// (temporal, height, width) rotary position of every prompt token. The last
// token must be a text token.
int q38_gen_prompt_mm(q38_gen* g, const int32_t* tokens, int n, const uint16_t* embeds, int n_emb,
                      const int32_t* pos3, const int* snap_at, int n_snap);

// Replace the end-of-sequence set (at most 4 ids).
void q38_gen_set_eos(q38_gen* g, const int32_t* ids, int n);
void q38_gen_get_stats(const q38_gen* g, q38_gen_stats* s);

// sizeof of the structs that cross this ABI, for foreign-language bindings to
// check their own layouts against: q38_config, q38_sampling, q38_gen_config,
// q38_gen_stats.
void q38_abi_sizes(int out[4]);

#ifdef __cplusplus
}
#endif
