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

The final documentation/benchmark commit follows the profile commit. Each
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
