## Phase: decode

Make decode as fast as this hardware allows: single-stream first, then
speculation, then multi-stream. Every wave runs the regimen in the skill;
every wave gets a notebook entry, a gate, and a commit. Read the wave plan
in HANDOFF.md and re-rank it from a fresh step attribution before the first
kernel.

**Waves, byte-ordered.** Start with the class that owns the most bytes or
the most step time (the attribution decides): expert / MoE GEMVs, dense
projections, the attention or recurrence core, glue (norms, router,
sampler, embedding, lm_head), launch fusion, host syncs and metadata,
async scheduling. For each: the precedent (bar engine kernel, another
platform's kernel in this repo, an upstream PR), the design note, the
kernel with its kill switch (opt-in through `{{ID}}`'s env block), the
parity test at the serving shapes, the microbench GB/s against the box's
ceiling, the gate. Bit-exact where possible; ULP with the reversion
sentinel otherwise; re-pin with the teacher-forced check when numerics
change on purpose.

**Speculation.** Whichever drafter is fastest on this box ships: the
checkpoint's own MTP block, a published DFlash / DFlash2 / DSpark / EAGLE
head, or none. Same gate for each candidate, acceptance recorded per
position, k swept on paired step ms (small k for recurrent mixers), a
batch-size schedule so speculation turns off where it loses. Judge on
step rate and tok/s on non-degenerate text, not on a looping prompt.

**Multi-stream.** Concurrent exact-token gates (`--concurrency 4 / 8 / 16`
or the record's `max_num_seqs`, output long enough that prefill stops
dominating), per-request decode alongside aggregate. Size `max_num_seqs`,
the KV pool and the batch cap for the record from measured per-sequence
bytes. Batching must earn its keep: the standing bar is at least 2x the
c=1 aggregate at c=4 and 3x at c=16 where memory allows; below it, find
what fails to amortize (per-row expert dispatch, batch-kernel row caps,
scheduler budget reserved by the speculator, preemption) and fix it, or
prove the ceiling from distinct-expert / bandwidth arithmetic in the
notebook. Speculation that loses at batch is scheduled off, not removed.

**Exit.** Single-stream decode beats the bar (or the notebook proves the
ceiling with a roofline and attributes the residual), the concurrency
curve is pinned, every wave is journaled and committed, the plan's status
log carries the final numbers, and the next phase's first item is named.

Marker evidence keys: `single_stream` ({ours, bar, workload, drafter,
acceptance}), `no_drafter` ({ours, bar}), `speculation` ({drafter, k,
schedule, acceptance, verdict}), `concurrency` (list of {c, aggregate,
per_request, exact}), `scaling` ({c4_over_c1, cmax_over_c1, cmax}),
`step_ms`, `bytes_per_token`, `ceiling_defended` (text if below the bar),
`waves` ({retained, rejected, parked}), `pins` (current shas).
