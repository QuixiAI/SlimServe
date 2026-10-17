# Review loop: CodeRabbit, Astra, self-review

The campaign is finished when the PR has been reviewed every automated way
available and every finding has been addressed, so that the human's review
starts from a clean slate. `tools/campaign/review.py` wraps the mechanics;
this file is the etiquette.

## Order of a round

1. Push the branch (after `tools/campaign/guard.sh` passes).
2. CodeRabbit: automatic on a non-draft PR; a draft PR needs
   `review.py cr-trigger --pr N` (`@coderabbitai review`). It is
   incremental (reviews only new commits) and sometimes rate limited or
   skipped because "base or head changed"; `review.py cr-wait` reports
   which and when to retry. A round takes ten to twenty minutes.
3. Astra (Codex, GPT-6 Astra): `review.py astra-review --base origin/main
   --out perf/results/<date>/astra_round<N>.log`. The model is always named
   explicitly (`-c model="gpt-6-astra"`); the default model varies per
   machine. `codex review --base` takes no custom prompt; use
   `review.py astra-exec` with a prompt file for a targeted follow-up
   ("re-check finding 2 after commit X"). If the preflight says the CLI is
   not authenticated, record `astra: unavailable (codex login needed)` in
   the marker and the final report and continue with CodeRabbit alone.
4. Self-review: run the `code-review` skill on the branch diff and the
   `simplify` pass on the changed files before the first push, so the
   external reviewers see a clean diff.

## Addressing findings

- Fetch every unresolved thread (`review.py cr-threads --pr N
  --unresolved`) and every Astra finding. Reproduce each one before fixing
  it: a reviewer reasoning over the code finds real bugs on paths the gates
  never exercised (multi-request, multi-chunk, NaN rows, unbuilt
  extensions). "Found while verifying" defects get their own notebook entry.
- Fix, or decline with reasoning. A decline states the measured or
  structural reason (a widen-to-float suggestion that breaks bit parity is
  declined with the parity evidence). Never dismiss a comment, never
  resolve a thread without a reply.
- After the fixes: rebuild, re-gate the pins (bit-exact expected for
  hardening fixes; a roll is recorded with its check), run the tests,
  commit (one commit per round, message states the round and the pins),
  `guard.sh`, push.
- Reply on each thread (`review.py cr-reply --pr N --comment-id <id>
  --body-file f`) with what changed and the commit, then resolve it
  (`review.py cr-resolve --thread-id <id>`); post one summary comment per
  round on the PR listing fixed / declined and the re-gate numbers.
- Re-trigger CodeRabbit and re-run Astra on the new head.

## Termination

A round is clean when the newest CodeRabbit review on the current head adds
no thread that is not already replied-and-resolved or declined-with-
reasoning, and the Astra review on the same head reports no P1 / P2
finding. Two consecutive clean rounds are not required; one clean round
after the last fix is. Then:

- update the PR title and body with the final numbers (the bar table is
  first), remove any "bring-up caveat" that no longer holds;
- `gh pr ready N`;
- write the review-round history into HANDOFF.md's status log and the
  campaign marker.

## Text rules for anything posted

Facts, numbers, commits, file paths. No process narrative, no thanks, no
mention of how the fix was produced. No email addresses, host names or
scratchpad paths. The commit identity is the machine's noreply identity and
nothing else appears in author or committer fields.
