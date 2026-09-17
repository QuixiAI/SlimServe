---
name: profile-campaign
description: Run a SlimServe profile campaign - bring up one model x quant x platform on this machine, beat the baseline engine on single-stream decode, multi-stream decode and prefill, clean up, open the PR and drive it through CodeRabbit and Astra review until it is ready for a human. Use when asked to start, resume or continue a profile campaign, or when tools/campaign/run_campaign.py launches a phase.
---

# SlimServe profile campaign

A campaign produces one profile record: **this exact quant of this exact model
on this exact hardware**, faster than the best public way to run it on this
box, with the evidence to prove it. The driver
(`tools/campaign/run_campaign.py`) runs the phases below one at a time, each
in its own session; you are inside one phase and the driver relaunches you
until that phase's marker says done. Everything that must outlive a session
lives in the repo (HANDOFF.md, `perf/`, `slimserve/profiles.json`, tests) or
under the campaign state dir, never in your context alone.

Read next, in this order, before doing anything in a phase:
`AGENTS.md`, `CLAUDE.md`, `perf/perf.md`, the campaign section at the top of
`HANDOFF.md` (yours, once discover has written it), the newest entries of
`perf/optimization_status.md`, `perf/baseline_status.md`, the profile in
`slimserve/profiles.json`, then the reference for the phase:
`references/method.md` (gates, notebook, rubric, PR description),
`references/platform-ops.md` (how this box boots, builds and profiles),
`references/research.md` (where to find precedent),
`references/review-loop.md` (CodeRabbit and Astra mechanics).

## Phases and their definition of done

| phase | done when |
| --- | --- |
| discover | artifact verified; source + profile registered (`status: in-progress`); memory budget and per-token bytes derived; research digest written; the bar engine measured on this box and recorded in `perf/baseline_status.md`; the campaign plan is the top section of HANDOFF.md; registry tests pass |
| bringup | the profile boots to health through `slimserve <id> --serve`, produces coherent text, passes per-layer or teacher-forced parity against a reference and a needle at moderate context; the first exact-token pins (N0) are recorded |
| decode | byte-ordered kernel waves done, speculation decided (whichever drafter is fastest here), single-stream decode beats the bar, and **multi-stream decode scales** (concurrent exact-token gates pinned; batching earns its keep or the notebook proves the ceiling); every wave journaled |
| prefill | prefill throughput and TTFT beat the bar at short, mid and deep context; concurrent prefill measured; every change journaled |
| cleanup | failed experiments deleted, kill switches audited, tests for every shape gate, ruff clean, other profiles re-gated bit-exact, profile flipped to `supported` (or its gate reason stated), domain-sliced commits, `tools/campaign/guard.sh` passes, PR description drafted |
| pr | `origin/main` merged (semantically, never one side wholesale), pins re-gated on the merged build, branch pushed, draft PR open with the bar table first, CodeRabbit triggered |
| review | CodeRabbit and Astra rounds addressed until a round yields nothing new, every thread replied to and resolved, PR body carries the final numbers, PR marked ready |
| port | (optional) retained kernels ported to the QuixiCore library per its sync contract |

The definitions are the exit criteria. The driver validates the marker
mechanically (profile present, tests pass, guard passes, PR state) and sends
you back in if it disagrees, so write the marker only when the table row is
true.

## The regimen inside a phase

Every optimization item goes through this loop once. Do not skip steps, do
not reorder them, and do not stop the loop at a commit.

1. **Research.** Name the precedent: a PR in vLLM / SGLang / llama.cpp / the
   quant author's engine, a kernel in a reference tree, or a SlimServe
   notebook entry from another platform. State what transfers to this
   hardware and what does not. Verify any "no X exists" premise online
   before building on it.
2. **Rank** by expected end-to-end gain from the current step attribution
   (ms per step per class, launches per step), not from isolated kernel
   speedups. Bytes first: the largest byte class is the first wave.
3. **Design note** in the notebook entry's Hypothesis: what changes, what the
   profile should show afterwards, the expected ms/step, and the correctness
   contract (bit-exact, ULP, or a re-pin with its teacher-forced check).
4. **Implement** in `csrc/quixicore/` and the model/layer files. One kill
   switch per change (`VLLM_QC_*` / platform env), opt-in through this
   profile's env block so no other profile can execute it.
5. **Correctness**: parity test in `tests/kernels/` at the real serving
   shapes against an fp32/float64 reference; microbench at the real shapes
   with GB/s against the box's measured ceiling.
6. **End to end through the profile only** (`slimserve <id> --serve`, the
   exact-token harness): the gate set for the platform, paired like-state
   boots, never a single-offset number.
7. **Notebook entry** (`perf/optimization_status.md`, format in
   `references/method.md`) with raw artifacts under
   `perf/results/YYYY-MM-DD/<run-id>/`, pins to `perf/baseline_status.md`,
   then a commit. Rejections are one paragraph and the code is removed.
8. **Next item**, in the same turn.

Stop rules: park an item when the profiler pair shows under 1% and the
exact-token boot is neutral; re-rank after every retained wave from a fresh
attribution; declare a ceiling only with a roofline (measured bytes at the
box's measured bandwidth) and the residual attributed.

## Hard rules

Each of these was learned the expensive way in earlier campaigns. They are
not negotiable inside a campaign.

- **Scope.** Work only on the assigned profile. Any change that reaches
  another profile is scoped off behind this profile's opt-in or the other
  profile's pins are re-gated bit-exact and recorded. Answer "does this reach
  another profile" by reading the code path, not by booting.
- **Authorship and privacy.** Commit with the identity already configured
  on the machine (`git config user.email` must be a `users.noreply.github.com`
  address). Never set `GIT_AUTHOR_*` / `GIT_COMMITTER_*`, never `--author`,
  never a `Co-Authored-By` or assistance trailer, never a mention of
  automated help in a commit message, PR body or PR comment. Never put a
  real email address, a session scratchpad path, a host name or deployment
  configuration in anything committed or posted. `tools/campaign/guard.sh`
  checks all of this before every push; run it yourself first.
- **Measurement honesty.** "The process started" is not "the system works".
  A path is validated when the real profile reaches health, serves the
  harness, passes correctness and produces recorded tok/s. vLLM interval
  logs are diagnostics, not results. State exactly what was run (smoke,
  parity, single-stream, concurrent). Numbers are comparable only between
  like-state boots on the same prompt shape; the harness aggregate includes
  prefill, so never compare it against a decode probe.
- **Journal as you go.** Every retained or rejected change has baseline,
  hypothesis, correctness, throughput, decision and raw artifact path in
  the notebook before its commit. Pins in `perf/baseline_status.md`. Raw
  logs under `perf/results/`; never cite `/tmp` or a scratchpad.
- **Durable paths.** Working state that must survive a crash goes under
  `~/.local/scratch/` or `perf/results/`, never `/tmp` or `/private/tmp`.
  Long-lived harness inputs are copied to `perf/results/harness_assets/`.
- **Servers are detached.** Boot the server with `start_new_session=True`
  (see `references/platform-ops.md`); the agent harness kills background
  task trees once a model pins tens of GiB. Never rebuild while a server
  runs. The operator has authorized killing any local server whenever the
  campaign needs the memory; report what was stopped.
- **Never stop at a milestone.** A finished wave, a written handoff or a
  passed gate is not a stopping point; begin the next ranked item in the
  same turn. End the turn only with a phase marker written, or blocked on
  something only a human can provide, and say which in the marker.
- **No questions to the operator.** Nobody is watching. Decide from the
  repo's precedent and record the decision and its reasoning in HANDOFF.md.
  If a decision is genuinely the operator's (a licensing choice, deleting
  data, spending on a download over a terabyte), write a `blocked` marker
  that states the question and the answer you would give.
- **Wait in the foreground.** Boots and gates under ten minutes run in
  foreground Bash with a long timeout; only runs past the Bash ceiling go to
  the background, and then batch the follow-on work before ending the turn.
- **Speculation is a measurement, not a preference.** Whichever drafter is
  fastest on this hardware ships (native MTP, DFlash, DSpark, whatever is
  published for the model); acceptance is recorded in every gate; a
  batch-adaptive schedule turns it off where it loses.
- **Prefer the repo's own precedent.** Study the optimized path on another
  platform before writing a new one; preserve its layouts, fusion
  boundaries and lessons unless this hardware gives a measured reason.
- **Concise, factual writing.** HANDOFF.md, notebook entries, PR text and
  review replies state facts and numbers with their sources. No narrative
  about the process, no hedging, no praise.

## Delegation

You write every line on the serving path yourself: kernels, bindings,
model and layer code, the adapter, the profile record, the gates, and every
decision to retain or reject. Subagents (the `Agent` tool, a cheaper model
such as `opus` or `sonnet`) are for work that is read-only or mechanically
verifiable and that you specify in one paragraph with the exact output you
need: research sweeps over PRs, kernels and notebooks that come back as a
digest with sources; log, trace and profile parsing; drafting review-thread
replies from a fix you already made; the byte-identical kernel port with its
drift check; polling a reviewer. Read every delegated result before using
it, and never let a subagent touch a file under `csrc/`, `vllm/`,
`slimserve/` or `tests/`. The driver's `--delegate` flag narrows this
further (`none` forbids subagents outright); the campaign prompt states the
policy in force.

## Signalling the driver

The driver reads `$SLIMSERVE_CAMPAIGN_DIR/phase-<phase>.json` after every
session. Write it with the Python below (never by hand-editing a template)
when the phase's definition of done is true, or when you are blocked:

```python
import json, os, datetime
marker = {
    "phase": os.environ["SLIMSERVE_CAMPAIGN_PHASE"],
    "status": "done",            # or "blocked"
    "summary": "<three sentences: what was achieved, the headline numbers, what is next>",
    "evidence": {},              # the phase prompt lists the required keys
    "next": "<the first item of the next phase, or the human's question if blocked>",
    "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path = os.path.join(os.environ["SLIMSERVE_CAMPAIGN_DIR"], f"phase-{marker['phase']}.json")
json.dump(marker, open(path, "w"), indent=1)
```

A `blocked` marker stops the whole campaign and pages the operator, so use
it only for a decision that is truly theirs. A session that ends without a
marker is relaunched with a resume prompt; before that happens, make sure
HANDOFF.md's status log says exactly where you are.

## Handoffs

The campaign plan is the top section of HANDOFF.md and it is also the
status log: append a dated entry at every milestone (bring-up gates, each
retained wave, the speculation verdict, the concurrency verdict, prefill
exit, cleanup exit, PR open, each review round). An entry holds: the pins,
the tree state (branch, HEAD, uncommitted files), the ranked open items, the
verification recipe (copy-paste commands), and the traps that cost time.
The next session, or the next attempt of this one, starts from that entry
and does not re-research.
