## Phase: cleanup

Turn the campaign tree into a reviewable, domain-sliced branch without
changing a single serving number. Cleanup is a zero-numeric-change class:
the full gate set must pass bit-exact at the end.

1. **Ledger.** Walk `git diff {{BASE}}` and the untracked files, file by
   file: keep / fix / delete, written into the status log before touching
   anything.
2. **Dead code.** Delete rejected and parked experiments, their kernels,
   their env hooks and their benches. One-line notebook pointers replace
   documented negatives in code comments.
3. **Switch audit.** Classify every `VLLM_QC_*` / platform env switch the
   branch added: production config (keep, document once), ULP reversion
   sentinel (keep the minimal set), bit-exact-proven kill switch (delete
   the switch and the old branch), census / geometry / experiment switch
   (delete with its code). Every survivor is opt-in through `{{ID}}`'s env
   block and reaches no other profile (answer by reading the code path).
4. **Hygiene.** Comments state constraints the code cannot show, never
   history or wave numbers. Tests for every shape gate the branch added
   (which M / N / K / dtype / batch takes each path) and the kernel parity
   tests, all passing on this build. `uvx ruff check` and `ruff format
   --check` clean on changed files. No scratchpad paths, no host names, no
   emails, no `/tmp` citations (`tools/campaign/guard.sh` checks).
5. **Other profiles.** Rebuild the native artifacts from the cleaned
   sources and re-gate every other profile whose kernels or Python paths
   this branch touched, bit-exact against their pins; record the
   consecutive count.
6. **The record.** Flip `{{ID}}` to `supported` (remove `status`,
   `status_reason`, `status_detail`) and rewrite its notes so every setting
   states what it is and why, with the measured numbers, the drafter
   verdict, the concurrency numbers, and any decision of record. If it
   must stay gated, the reason is one sentence of measurement, not intent.
7. **Commits.** Domain-sliced, in this order, each with the numbers that
   justify it: kernels + bindings; serving path; adapter / loader; profile
   + CLI + tests; perf notebook + baselines + HANDOFF.md. Identity and
   message rules per the skill. `tools/campaign/guard.sh --base {{BASE}}`
   passes.
8. **PR description** drafted to `perf/results/{{DATE}}/PR_DESCRIPTION.md`
   per the template in `references/method.md`: the bar table first, then
   what was built, the retained / rejected ledger, decisions of record,
   anchors, open items, tests, kernel port.
9. Self-review before anyone else sees it: run the `code-review` skill on
   the branch diff and the `simplify` pass on changed files; fix what they
   find; re-gate if anything numeric moved (it must not).

Marker evidence keys: `commits` (list of {sha, subject}), `tests` ({files,
passed, build}), `anchors` (list of {profile, result, consecutive}),
`status` (`supported` or the gate reason), `status_reason` (if gated),
`pr_description` (path), `guard` (`pass`).
