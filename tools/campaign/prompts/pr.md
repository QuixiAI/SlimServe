## Phase: pr

Land the branch on a PR against `{{REMOTE}}` main, reviewed-ready.

1. `git fetch {{REMOTE}}`; if `{{BASE}}` moved, merge it into the branch
   **semantically** (both sides survive; never resolve a conflict by taking
   one side wholesale; a merge that resurrects deleted paths is wrong). If
   the remote history was rewritten, do not merge: reset to the new base
   and re-apply this branch's own commits.
2. Rebuild the native artifacts on the merged tree; re-run the tests; re-
   gate the pins and the other profiles' anchors on the merged build; fix
   what the merge broke (the notebook records "merge regressions" as their
   own entry). Finish the PR description with the merged-build numbers.
3. `tools/campaign/guard.sh --base {{BASE}}` must pass. Push the branch to
   `{{REMOTE}}`.
4. Open the PR as a **draft** with `gh pr create --draft --base main
   --head {{BRANCH}} --title "<the template's H1>" --body-file
   perf/results/<date>/PR_DESCRIPTION.md`. The title carries the ratio
   against the bar. Record the number.
5. Trigger CodeRabbit (`{{PYTHON}} tools/campaign/review.py cr-trigger
   --pr N`), run the Astra preflight, and append the PR facts to the status
   log (number, head, pins, what a human should test first).

Marker evidence keys: `pr` (number), `url`, `head` (sha), `merged_base`
(sha), `pins_on_merged_build` (list), `coderabbit_triggered_at`.
