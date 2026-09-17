## Phase: bringup

Take the profile from registered to correct. Slow is fine; earlier
campaigns started at 0.1-2 tok/s. Follow the bring-up steps of the plan in
HANDOFF.md and `references/platform-ops.md` for the boot protocol.

1. Adapter / loader for the artifact's layout; platform routing for every
   blocker on the list (device-neutral allocation, backend selectors,
   torch-native fallbacks where a kernel is missing, `language_model_only`
   where vision is not yet qualified); pool / cache layout for the model's
   attention and state groups; router, activation clamps, expert counts.
2. First tokens through `slimserve {{ID}} --serve` (detached; the record
   is gated so `SLIMSERVE_SERVE_IN_PROGRESS=1`), primer, ramp.
3. Correctness, in order of strength (`references/method.md`): teacher-
   forced comparison against the bar engine's logprobs or per-layer cosine
   against a reference dump; coherent greedy text; a needle at 1.5k-4k; the
   quant author's quality fixture if one exists; canaries (text, tool call,
   image if the model has vision and the platform path exists).
4. **N0 pins**: the exact-token gate set for this platform (8tok /
   off1-2000 / 2500x64 or the equivalents named in the plan), greedy and
   sha-pinned, plus the seeded arm. A `gate.sh` under
   `perf/results/{{DATE}}/{{ID}}-baseline/` that the later phases reuse.
   Record in `perf/baseline_status.md` (N0) and the notebook.
5. Restate per-token bytes from the load census and the ceiling ladder now
   that the real footprint is known; update the plan's wave order if the
   census changed the byte ranking.

Fix, do not work around: a wrong hidden state at layer k is isolated to the
first failing op and fixed in the responsible layer of this repo.

Marker evidence keys: `boots` (true), `pins` (list of {leg, sha, tok_s,
wall_s}), `correctness` (teacher-forced / cosine numbers, needle tiers),
`footprint` (resident GiB, KV pool, per-seq state), `first_decode_tok_s`.
