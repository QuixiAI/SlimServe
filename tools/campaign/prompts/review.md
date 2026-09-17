## Phase: review

Drive PR {{PR}} through every automated review until a round is clean,
then mark it ready. Mechanics and etiquette: `references/review-loop.md`
and `tools/campaign/review.py --help`. Astra: {{ASTRA_STATUS}}.

Each round:

1. Wait for CodeRabbit on the current head (`review.py cr-wait --pr {{PR}}
   --since <last push>`); on rate limit or skip, wait the stated time and
   re-trigger (`review.py cr-trigger`). Fetch the unresolved threads
   (`review.py cr-threads --pr {{PR}} --unresolved`).
2. Run Astra on the same head when available: `review.py astra-review
   --base {{BASE}} --out perf/results/<date>/astra_round<N>.log` (model
   `{{ASTRA_MODEL}}` is passed explicitly). Read the findings file.
3. Reproduce every finding before fixing it - reviewers reading code find
   what single-request gates never exercise (multi-request steps, multi-
   chunk prefill, NaN rows, unbuilt extensions, shape fallbacks). Fix, or
   decline with the measured / structural reason. A defect found while
   verifying gets its own notebook entry and fix.
4. Rebuild, tests, re-gate the pins (bit-exact expected; a roll is recorded
   with its check), notebook entry for the round, one commit, `guard.sh`,
   push.
5. Reply on every thread with what changed and the commit (`review.py
   cr-reply`), resolve the fixed ones (`review.py cr-resolve`), post one
   summary comment for the round (fixed / declined, pins, tests, probe),
   re-trigger CodeRabbit, and re-run Astra on the new head.

Stop when the newest CodeRabbit review on the current head adds nothing
that is not already replied-and-resolved or declined-with-reasoning, and
Astra reports no P1 / P2 on that head (or is unavailable and recorded as
such). Then update the PR title and body with the final numbers (bar table
first; drop any bring-up caveat that no longer holds), `gh pr ready {{PR}}`,
write the round history into the status log, and write the marker.

Marker evidence keys: `pr`, `rounds` (list of {n, head, coderabbit:
{new, fixed, declined}, astra: {p1, p2, p3, fixed, declined}, pins}),
`unresolved` (0), `astra` (`clean` | `unavailable (reason)` |
`declined-with-reasoning: ...`), `ready` (true), `human_should_test`
(three bullets).
