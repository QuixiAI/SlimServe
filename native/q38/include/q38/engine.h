// q38: purpose-built Metal engine for Qwen3.8-27B on Apple Silicon.
// Plain C ABI so a Rust server (or anything else) can drive it.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38_engine q38_engine;

typedef struct {
  int max_ctx;     // tokens of full-attention KV per stream (default 32768)
  int max_rows;    // rows per step: 1 decode, up to 8 verify/prefill (default 8)
  int block_size;  // KV page size in tokens (default 16)
  int serial;      // 1 = serial dispatch encoder (debug); 0 = concurrent + barriers
} q38_config;

typedef struct {
  double encode_ms;  // host time to encode the command buffer
  double gpu_ms;     // GPUEndTime - GPUStartTime
  double wall_ms;    // encode start to completion
  int dispatches;
  int barriers;
} q38_step_stats;

// metallibs: ':'-separated list, searched in order for each kernel name.
q38_engine* q38_create(const char* pack_path, const char* metallibs,
                       const q38_config* cfg, char* err, int err_len);
void q38_destroy(q38_engine* e);

// Forget the sequence: position 0, zero recurrent state.
void q38_reset(q38_engine* e);

// Run `m` tokens at positions [pos, pos + m) causally. Their state is
// tentative until q38_commit. Returns 0 on success.
int q38_step(q38_engine* e, const int32_t* tokens, int m, q38_step_stats* stats);

// Multimodal step. Row i takes its input embedding from embeds[i * hidden ..]
// (BF16, already projected to the model width, e.g. a vision tower output)
// when tokens[i] < 0, and from the token table otherwise; `embeds` may be
// NULL when no token is negative. `pos3`, when not NULL, gives every row its
// (temporal, height, width) rotary position ([m][3]); text rows carry the
// same value three times. After a step with explicit positions, steps
// without them continue at 1 + the largest position of its last row, so such
// a step must be committed whole.
int q38_step_mm(q38_engine* e, const int32_t* tokens, const uint16_t* embeds, const int32_t* pos3,
                int m, q38_step_stats* stats);

// Keep the first `n` rows (1..m) of the last step; the rest are discarded.
// Called implicitly with n = m if the next q38_step arrives first.
int q38_commit(q38_engine* e, int n);

int64_t q38_position(const q38_engine* e);

// Sequence snapshots (prefix caching). A snapshot is a self-contained host
// copy of the committed sequence state at the current position: recurrent
// GDN state, conv history, attention KV rows [0, pos) and the MTP head's KV.
// Save requires no pending (uncommitted) step. Loading replaces the engine's
// sequence with the snapshot's; the snapshot stays valid and reusable.
typedef struct q38_snapshot q38_snapshot;
q38_snapshot* q38_snapshot_save(q38_engine* e);
int q38_snapshot_load(q38_engine* e, const q38_snapshot* s);
int64_t q38_snapshot_position(const q38_snapshot* s);
int64_t q38_snapshot_bytes(const q38_snapshot* s);
void q38_snapshot_free(q38_snapshot* s);
// Rows of the step that ended at the snapshot position whose final hiddens the
// snapshot carries (restored as rows 0.. of q38_final_bf16, so MTP pairs that
// were not fed yet can still be fed after a load). The logits of the last of
// those rows are restored too: q38_sample(row = tail_rows - 1) works after a
// load without stepping.
int q38_snapshot_tail_rows(const q38_snapshot* s);
int q38_vocab(const q38_engine* e);
int q38_hidden(const q38_engine* e);

// Outputs of the last step, valid until the next step.
const int64_t* q38_argmax(const q38_engine* e);        // [m]
const uint16_t* q38_logits_bf16(const q38_engine* e);  // [m, vocab]
const uint16_t* q38_final_bf16(const q38_engine* e);   // [m, hidden] (see layer cap)

// Sampling over a row of the last step's logits (host side: the logits live
// in unified memory, so this is a scan, not a copy). The filtered, normalized
// distribution after temperature -> top-k -> top-p is returned through
// ids/probs (capacity >= top_k, sorted by descending probability) so a
// speculative verifier can reuse it; the return value is the token drawn
// with the uniform `u` in [0, 1).
typedef struct {
  float temperature;  // model default 1.0
  int top_k;          // model default 20 (max 64)
  float top_p;        // model default 0.95
} q38_sampling;
int32_t q38_sample(const q38_engine* e, int row, const q38_sampling* s, double u,
                   int32_t* ids, float* probs, int* n);

// ---- MTP draft head (the checkpoint's multi-token-prediction layer) ----
// Sidecar pack written by tools/pack_mtp.py.
int q38_load_mtp(q38_engine* e, const char* mtp_pack, char* err, int err_len);
int q38_has_mtp(const q38_engine* e);
// Committed history: feed `m` pairs (final hidden of row target_rows[i] of the
// last q38_step, tokens[i] = the token that FOLLOWS that row). Drops any
// tentative draft rows. With head != 0 the draft logits for the token after
// tokens[m-1] are produced.
int q38_mtp_feed(q38_engine* e, const int32_t* tokens, const int* target_rows, int m, int head,
                 q38_step_stats* stats);
// Same, with embeds[i * hidden ..] as the embedding of tokens[i] when it is negative.
int q38_mtp_feed_mm(q38_engine* e, const int32_t* tokens, const uint16_t* embeds,
                    const int* target_rows, int m, int head, q38_step_stats* stats);
// Tentative draft: one more step from the previous MTP output and `token`
// (the token just drafted). Always produces draft logits.
int q38_mtp_chain(q38_engine* e, int32_t token, q38_step_stats* stats);
// Sample from the last MTP draft logits (same contract as q38_sample).
int32_t q38_mtp_sample(const q38_engine* e, const q38_sampling* s, double u, int32_t* ids,
                       float* probs, int* n);
double q38_mtp_read_gb(const q38_engine* e);

// ---- Several sequences in one engine ----
// The functions above drive sequence 0. More sequences share the weights and
// the scratch; each has its own position, recurrent state, K/V rows (grown on
// demand) and MTP state. A grouped step runs a causal run of rows for each of
// several sequences in ONE pass over the weights: rows are laid out group
// after group, at most max_rows in total, each sequence at most once. Row
// indices in the per-sequence calls (sample, MTP target rows, commit) count
// from the start of that sequence's run. The outputs of a sequence's last
// step are valid until the next step of ANY sequence; its pending rows can
// still be committed afterwards.
typedef struct {
  int seq, m;
} q38_group;
int q38_seq_new(q38_engine* e);  // id >= 1, or -1 when all 16 are in use
void q38_seq_free(q38_engine* e, int seq);
int q38_seq_reset(q38_engine* e, int seq);
int64_t q38_seq_position(const q38_engine* e, int seq);
// A pos3 row whose first value is negative means "this sequence's own next
// text position", so text and image runs can share a step.
int q38_step_groups(q38_engine* e, const q38_group* g, int n, const int32_t* tokens,
                    const uint16_t* embeds, const int32_t* pos3, q38_step_stats* stats);
int q38_seq_commit(q38_engine* e, int seq, int n);
// Prefill chunks: a step with ONE group may carry up to q38_big_rows() rows
// (512). It runs through the many-row matrix kernels (2,048-token prompt: 106
// prompt tokens/s in 8-row steps, 213 in 64-row steps, 260 in 256-row steps),
// must be committed whole, and afterwards only the last q38_seq_out_rows()
// rows (at most 8) are addressable by the per-sequence calls: row 0 is the
// first of those, and only the LAST of them has logits.
int q38_big_rows(const q38_engine* e);
int q38_seq_out_rows(const q38_engine* e, int seq);
int32_t q38_seq_sample(const q38_engine* e, int seq, int row, const q38_sampling* s, double u,
                       int32_t* ids, float* probs, int* n);
// Snapshots are interchangeable between sequences. After a load, the
// sequence's tail rows and last logits are its own copies (other sequences'
// step outputs are untouched).
q38_snapshot* q38_seq_snapshot_save(q38_engine* e, int seq);
int q38_seq_snapshot_load(q38_engine* e, int seq, const q38_snapshot* s);
// MTP head over several sequences: pairs laid out group after group;
// target_rows[i] is a row of that sequence's last step. With head != 0 every
// group gets draft logits after its last pair.
int q38_mtp_feed_groups(q38_engine* e, const q38_group* g, int n, const int32_t* tokens,
                        const uint16_t* embeds, const int* target_rows, int head,
                        q38_step_stats* stats);
// Pairs from the rows of a big step (see q38_big_rows): (hidden of row
// first_row + i of that step, tokens[i]) for i < n <= 8. Valid after the step
// is committed and until the next step of ANY sequence. No draft logits.
int q38_mtp_feed_big(q38_engine* e, int seq, int first_row, int n, const int32_t* tokens,
                     const uint16_t* embeds, q38_step_stats* stats);
// One more tentative draft step for each listed sequence.
int q38_mtp_chain_groups(q38_engine* e, const int* seqs, const int32_t* tokens, int n,
                         q38_step_stats* stats);
int32_t q38_seq_mtp_sample(const q38_engine* e, int seq, const q38_sampling* s, double u,
                           int32_t* ids, float* probs, int* n);

// ---- DSpine block drafter (tencent/Qwen3.8-27B-DSpine-b16) ----
// Sidecar pack written by tools/pack_dspine.py (+ tools/quantize.py). Once
// loaded, every target step also records the five residual taps the drafter
// reads, and the engine keeps the drafter's context in step with what each
// sequence commits; there is nothing to feed.
int q38_load_dspine(q38_engine* e, const char* dspine_pack, char* err, int err_len);
int q38_has_dspine(const q38_engine* e);
// One drafted token: `tok` was drawn from the distribution (ids, probs) of n
// entries (the drafter's candidates after temperature / top-k / top-p).
typedef struct {
  int32_t tok;
  int n;
  int32_t ids[16];
  float probs[16];
} q38_draft_row;
// Draft up to `k` (1..7) tokens that follow `anchor`, the next token of the
// sequence (not stepped yet; the sequence must have no uncommitted step). One
// drafter pass proposes all k positions; the tokens are then drawn in order,
// each conditioned on the one before, with the uniforms u[0..k). Drafting
// stops early once the product of the drawn tokens' probabilities falls below
// `stop_below`; a negative `stop_below` selects the marginal rule instead, at
// a rate of -stop_below tokens per ms (a draft is kept while its estimated
// chance of being accepted pays for the verify row it adds; about the decode
// rate, 0.06, is the tuned value). Returns the number of rows written to
// `out`, negative on error.
int q38_seq_dspine_draft(q38_engine* e, int seq, int32_t anchor, int k, const q38_sampling* s,
                         float stop_below, const double* u, q38_draft_row* out,
                         q38_step_stats* stats);
// The same for up to 4 sequences in one drafter pass (their blocks share
// every pass over the weights): sum of (k[i] + 1) at most 8. Sequence i uses
// the uniforms u[8 i ..] and writes n_out[i] rows at out[8 i ..].
int q38_dspine_draft_groups(q38_engine* e, const int* seqs, const int32_t* anchors, const int* k, int n,
                            const q38_sampling* s, float stop_below, const double* u, q38_draft_row* out,
                            int* n_out, q38_step_stats* stats);
// Accumulated drafter time since load: GPU, host sampling tail, wall (ms), and calls.
void q38_dspine_times(const q38_engine* e, double* gpu_ms, double* tail_ms, double* wall_ms,
                      int64_t* calls);
double q38_dspine_read_gb(const q38_engine* e);
// Debug: the next draft call takes tokens[i] as its i-th token instead of
// drawing it (the rows still report the distribution it would have drawn
// from), and does not stop early.
void q38_dspine_force(q38_engine* e, const int32_t* tokens, int n);

// GPU argmax of every logits row (default on). Turn off when sampling on the
// host: it costs about 0.4 ms per step.
void q38_set_argmax(q38_engine* e, int on);

// Debug: stop after `cap` layers (0 = all). With a cap, q38_final_bf16 holds
// the residual stream after that layer and logits/argmax are not produced.
void q38_set_layer_cap(q38_engine* e, int cap);

// Bytes of projection weights read per token (from the pack header).
double q38_decode_read_gb(const q38_engine* e);

#ifdef __cplusplus
}
#endif
