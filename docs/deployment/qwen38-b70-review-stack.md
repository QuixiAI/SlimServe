# B70 review branch and commit stack

Integration branch: `feat/b70-qwen38-serving`.
Tested source base: `c6d98594035689600885c858e2f112208668c199` (`review/b70-base`).

The current remote `main` has substantially different history and source from
this tested checkout (806 changed files at the time of inspection). This stack
preserves the serving implementation; it is not rebased onto current `main`.
Do not open the integration branch directly against `main` and mistake the
historical diff for these changes. Use the stack bases below for bounded reviews,
or forward-port selected commits to `main` with their listed dependencies.

```bash
git fetch origin feat/b70-qwen38-serving
git worktree add ../SlimServe-b70-review origin/feat/b70-qwen38-serving
```

## Commit boundaries

| Commit | Change | Review head | Review base |
| --- | --- | --- | --- |
| `346c3bfb0d` | Load Qwen3.5 GGUF checkpoints with embedded MTP heads | `review/b70-01-gguf-mtp` | `review/b70-base` |
| `044e0a586a` | Reuse split Q8 weights in XPU decode kernels | `review/b70-02-q8-kernels` | `review/b70-01-gguf-mtp` |
| `c139ad7d96` | Coordinate XPU worker teardown before releasing device memory | `review/b70-03-worker-lifecycle` | `review/b70-02-q8-kernels` |
| `8f98bd2870` | Carry bounded semantic checkpoint hints through recurrent caching | `review/b70-04-semantic-cache` | `review/b70-03-worker-lifecycle` |
| `c6df785f15` | Count Qwen reasoning spans opened by the chat template | `review/b70-05-reasoning-usage` | `review/b70-04-semantic-cache` |
| `3c76699d20` | Constrain tool JSON and report incomplete responses accurately | `review/b70-06-tool-protocol` | `review/b70-05-reasoning-usage` |
| `e26bf185f3` | Mask ragged tensor-descriptor tails in Triton attention | `review/b70-07-attention` | `review/b70-06-tool-protocol` |
| `a76a5b0746` | Register reproducible Qwen3.8 B70 profiles and tool template | `review/b70-08-profiles` | `review/b70-07-attention` |

The initial documentation/benchmark commit follows the profile commit. Each
numbered review branch contains the preceding stack. Opening a PR against its
immediate predecessor shows only that logical change. Merging these directly
into a different codebase still requires a forward port; no conflict-free claim
is made for current `main`.

## Dependencies and verification

- GGUF/MTP and Q8 commits use the XPU foundation already present in the saved base (`ff9dfeba3`, `7af0a50f3`, `c6d985940`). The native Q8 Python tests mock the SYCL entrypoints; historical serving evidence is documented separately.
- Lifecycle cleanup is independent of the tool grammar; its 120-second timeout is now selected by B70 profiles instead of changing every platform default.
- Semantic checkpoints require the complete renderer-to-IPC-to-scheduler chain. The review version bounds copying before allocation and skips multimodal histories; this latter cleanup has CPU tests but was not reloaded into the live service.
- Tool grammar, required-call reasoning suppression, truthful Responses terminal events, and request-schema conflicts are one protocol unit. Thinking budgets are profile defaults, with explicit request overrides preserved.
- The profile/template commit depends on model loading, protocol default support, attention selection, and template metadata forwarding. It carries FP8/Q8/BF16 recipes; BF16 remains in-progress after failed qualification.
- The current active service continues using `/home/alex/SlimServe`; preparing these commits did not switch its checkout or index. The isolated branch includes the bounded-copy and profile-scope refinements beyond the active source.

## Known stability limit

Live traffic subsequently stopped making progress with eight active requests,
then failed on a `sample_tokens` RPC timeout. A clean process restart stalled
rank3 on creation of a scalar XPU tensor; its native stack waited inside Intel
Level Zero. A driver GT0 reset of PCI `0000:aa:00.0` restored startup and health
at 16:33 EDT on 2026-09-19. This recovery is not a root-cause fix or soak
qualification. Do not quote the local 2.1x throughput result as stable throughput
for every workload. Device-reset operations are intentionally not automated by
the provided systemd example.

See [the deployment guide](qwen38-b70.md) for recipe, dependencies, measurements,
validation limits, and portable serving commands. No checkpoints, credentials,
traffic captures, API keys, or Evalhub automation keys are included.

## Final CPU validation

262 CPU tests passed across three non-overlapping commands in the isolated
checkout: 28 hardware/model contract tests, 61 profile/fetch/template tests,
and 173 combined protocol/budget/cache tests. The combined runs emitted the
known duplicate XPU operator registration warning. Existing eight GPU
ragged-tail reference tests and the local serving measurements were collected
before packaging; no GPU benchmark was run against competing live traffic.
Native SYCL code was not rebuilt or numerically requalified during this pass.

The source base's working tree and staged index were preserved. The unrelated
local edit removing commit-authorship guidance from CLAUDE.md is intentionally
not part of this stack.

## Follow-up correctness commits

All of these are on the published integration branch; each is independently
cherry-pickable with the existing protocol/profile dependencies above.

| Commit | Change | Review head | Review base |
| --- | --- | --- | --- |
| `aae9efe33` | Empty/multipart assistant history and refusal handling | `review/b70-10-assistant-history` | `review/b70-09-deployment-evidence` |
| `12856ae2a` | Typed generation-error SSE and background failure storage | `review/b70-11-stream-errors` | `review/b70-10-assistant-history` |
| `88ebacda1` | Stable Qwen argument JSON emitted at tool-call closure | `review/b70-12-qwen-arguments` | `review/b70-11-stream-errors` |
| `811839379` | Canonical streamed items retained in terminal Responses | `review/b70-13-response-parity` | `review/b70-12-qwen-arguments` |

The fourth review diff also contains an intervening handoff documentation commit.
Cherry-pick the listed source commit alone when preparing a source-only PR.
Further protocol receipts and validation are documented in
[qwen38-b70-protocol-followup.md](qwen38-b70-protocol-followup.md).

### Chat-template precedent

The Qwen template/profile/package integration follows merged
[PR #33](https://github.com/QuixiAI/SlimServe/pull/33), commit `6fb5ac23a`,
which followed GLM compatibility [PR #32](https://github.com/QuixiAI/SlimServe/pull/32).
The Qwen-specific asset is checked in, included in package data, selected through
`chat_template_asset`, and receives authoritative request-level tool choice.
It renders required/named/none/auto policy, explicit strictness guidance, reasoning
effort, and thinking enable/disable using Qwen's native ChatML/XML tokens.
It does not copy GLM's model-specific default of disabling thinking for tool calls.
Six existing Qwen template policy/history tests passed during the follow-up audit;
startup logs confirm the live profile loads this asset.

### Pull and prepare a PR elsewhere

```bash
git fetch origin feat/b70-qwen38-serving 'refs/heads/review/b70-*:refs/remotes/origin/review/b70-*'
git switch -c b70-review origin/feat/b70-qwen38-serving
git log --oneline origin/review/b70-base..HEAD
```

Use the saved base to review only this stack. To target current `main`, create a
new branch there and forward-port selected commits with their prerequisites;
do not merge the entire historical branch and its unrelated base history.
The live source remains in its original dirty checkout; the isolated review
checkout is the source of commits. The original unrelated CLAUDE.md edit stays
outside this stack. Generated binaries, model weights, credentials, local service
secrets, and raw traffic are excluded.

### Diagnostic tooling and latest validation

`review/b70-14-serving-diagnostics` packages the optional proxy and compact audit
scripts under `tools/serving_diagnostics/`; see its README for portable commands.
No captures or credentials are included. Its 19 CPU fixture tests passed.
The current integration head also includes this guide and profile-note updates.

The 2026-09-19 audit compared every modified/untracked live source file with this
review checkout: 61 matched byte-for-byte and none was absent. Remaining differences
were the documented profile-scoped budgets/shutdown/replay settings, bounded
semantic-prefix refinement and tests, newer Responses test fixtures, added notebook
notes, and the deliberately excluded unrelated CLAUDE.md edit.

Post-fix capture snapshot: 34 completed Responses, 63 paired tool calls, zero item-ID,
call-ID, argument, or status mismatches and zero HTTP errors. Four other streams
lacked terminal events and one failed; the full coding evaluation remains ongoing.
This is evidence for the protocol fixes, not full correctness/soak qualification.


### Engine-stall follow-up, 2026-09-20

Additional independently reviewable commits on the integration branch:

| Commit | Change | Live state |
| --- | --- | --- |
| `af784bf19` | Preserve grammar masks and natural speculative reasoning closure at the cap | Deployed |
| `4c41dad47` | Keep trailing empty assistant replay items attached to their tool turn | Deployed |
| `c94d343d2` | Proxy stream lifecycle instrumentation | Deployed |
| `230ec4817` | Token-progress watchdog with per-worker native stacks | Running as a separate read-only service |
| `a42ba2257` | Correct post-completion proxy disconnect classification | Tested; pending proxy reload |

Coding evaluation attempt 5 still hit a whole-engine sample_tokens timeout after
the budget fix, so engine stability is not qualified. Attempt 6 repeats the same
serving configuration with stack capture armed. The watchdog does not alter model
execution or reset devices. Native stack attachment can briefly pause a target
only when the measured no-progress threshold is reached.
