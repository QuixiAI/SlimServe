# DSV4 NaN/degeneration incident tooling (2026-08-10/11)

Diagnostic harnesses from the silent-degeneration investigation. Full
narrative: the incident entries in `perf/optimization_status.md`.

Runtime tripwires (env-gated, in-tree):

- `VLLM_NAN_WATCH=1` - async per-step NaN-row check on target logits
  (both model runners); logs row indices. Permanently on in production.
- `VLLM_NAN_WATCH_LAYERS=1` - sticky per-layer NaN counters on DSV4
  decoder layers, plus intra-layer slots for layers 0-2
  (100+4L+p; p: 0=attn-in, 1=attn-out, 2=moe-in, 3=moe-out). Dumped by
  the logits tripwire on first detection; min slot = birth site.
- `VLLM_DSV4_TOPK_VALIDATE=1` - decode top-k indices vs per-row
  seq_lens at the indexer.
- `VLLM_DSV4_FILTERED_TOPK_MIN_ROWS` - top-k dispatch override
  (0=always FilteredTopK, huge=always persistent, default 32).

Multi-boot campaigns (single-boot results are worthless for this bug -
expression is a per-boot lottery with hours-scale phase drift):

- `campaign_full.sh` / `campaign_piecewise.sh` - 6-boot storm-rate arms
  (FULL capture-64 vs PIECEWISE capture-32).
- `campaign_vfix.sh` - 8-boot validation of the bt_per_token
  persistence fix (result: 0/8 storms; pre-fix 2/6).
- `campaign_env2.sh` - interleaved full-length env arms
  (baseline vs VLLM_DSV4_ALIGNED_Q8=0) for rare-seed attribution.
  Run this WHEN PRODUCTION NAN_WATCH SHOWS EVENTS (an active phase);
  in a dormant phase every arm reads zero and nothing is learned.
- `reproducer.py` - decode+prefill mixed-batch load (1-4 spec streams +
  24K chunked prefills); negative on the one boot tried.
- `mhc_ab_test.py` / `topk_ab_test.py` - hostile-memory kernel A/B
  units (all passing: the kernels are clean in isolation).

Paths in the scripts point at the 2026-08 session scratchpad for
corpora/regions; regenerate region files from any large text corpus
(disjoint 1000-token windows) before reuse.
