// q38 stream pool: several generations sharing one engine.
//
// Every cycle (q38_pool_step) is ONE pass over the weights carrying rows of
// every active stream: one decode row per decoding stream (plus MTP draft rows
// while few streams decode), and chunks of prompt rows for streams that are
// still prefilling. Nobody waits for another stream to finish; a prompt that
// arrives while others decode is prefilled in the same passes.
//
// Sampling is lossless speculative decoding exactly as in q38/generate.h, per
// stream, with a per-stream seed. Prefix reuse is shared by all streams: a
// prompt resumes from the longest of (a) a finished stream's sequence that is
// still alive and is a prefix of it, (b) a stored snapshot that is a prefix of
// it. Snapshots are taken at the marks the caller names and at the prompt end.
//
// Not thread-safe: one thread calls everything.
#pragma once

#include <stdint.h>

#include "q38/engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38_pool q38_pool;

typedef struct {
  q38_sampling draft;   // how drafts are drawn from the MTP head
  int depth;            // MTP drafts per cycle for a lone decoding stream (0 = no drafting)
  float adaptive;       // stop drafting once a draft's own probability is below this
  // With the DSpine drafter loaded (q38_load_dspine) the pool drafts with it
  // instead of the MTP head: `depth` > 0 only switches drafting on (blocks of
  // up to 7 for a lone stream, one draft each for two streams), and a negative
  // `adaptive` is the marginal rule's rate (see q38_seq_dspine_draft; any
  // other value means the default 0.06).
  int max_streams;      // 1..8
  int64_t max_cache_bytes;  // host memory for snapshots plus idle sequences
  // The MTP head is shown only the last `mtp_prompt_tail` (hidden, token)
  // pairs of a prompt (0 = all of it). The head's own attention history is
  // what drafts are conditioned on; feeding a long prompt to it costs a pass
  // per chunk.
  int mtp_prompt_tail;
  // Big prefill passes (many prompt rows of one stream through the many-row
  // kernels, 2 to 2.5 times the prompt rows per second; nothing else rides in
  // such a pass): -1 never, 0 only while no stream decodes, N > 0 also one
  // after every N ordinary passes while streams decode.
  int big_every;
  // Rows of a big pass while no stream decodes (up to q38_big_rows(); while
  // streams decode a big pass carries 64). Measured on a 2,048-token prompt:
  // 64 rows 213 prompt tokens/s, 256 rows 260, 512 rows 264; a 256-row pass
  // takes about a second, during which a new request waits.
  int big_rows;
} q38_pool_config;

typedef struct {
  const int32_t* tokens;   // prompt; a negative id is an image row (see generate.h)
  int n;
  const uint16_t* embeds;  // [n_emb][hidden] BF16 rows for the negative ids, in order (copied)
  int n_emb;
  const int32_t* pos3;     // [n][3] rotary positions, or NULL for text (copied)
  const int* snap_at;      // prompt lengths to snapshot at (ascending), may be NULL
  int n_snap;
  q38_sampling sampling;
  uint64_t seed;
  int32_t eos[4];
  int n_eos;
  int64_t max_tokens;      // generation stops after this many tokens (reason 2)
} q38_pool_request;

typedef struct {
  int stream;
  int n;              // tokens produced this cycle (0 while prefilling)
  int32_t tokens[8];
  int done;           // 0 running, 1 end of sequence (the token is included), 2 max_tokens
  int prompt_done;    // 1 on the first event after the prompt is in
  int64_t reused;     // prompt tokens taken from a live sequence or snapshot
} q38_pool_event;

typedef struct {
  int64_t cycles, rows;          // engine passes and rows they carried
  int64_t emitted, drafted, kept;
  double target_ms, draft_ms;
  int snapshots, idle_sequences;
  int64_t cache_bytes;
} q38_pool_stats;

void q38_pool_config_default(q38_pool_config* c);
q38_pool* q38_pool_create(q38_engine* e, const q38_pool_config* c);
void q38_pool_free(q38_pool* p);

// Start a stream. Returns its id (>= 0), -7 when max_streams are running
// (open again after one finishes), or another negative value for a bad
// request.
int q38_pool_open(q38_pool* p, const q38_pool_request* r);
// Stop a stream (finished or not). Its sequence stays alive for prefix reuse.
void q38_pool_close(q38_pool* p, int stream);
// One cycle. Fills up to `cap` events (one per stream that made progress or
// finished; cap >= max_streams) and returns how many; 0 when no stream is
// running; negative on an engine error.
int q38_pool_step(q38_pool* p, q38_pool_event* ev, int cap);
int q38_pool_active(const q38_pool* p);
void q38_pool_get_stats(const q38_pool* p, q38_pool_stats* s);
// sizeof(q38_pool_config), q38_pool_request, q38_pool_event, q38_pool_stats
void q38_pool_abi_sizes(int out[4]);

#ifdef __cplusplus
}
#endif
