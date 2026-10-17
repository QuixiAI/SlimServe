## Phase: discover

Produce the campaign plan and the bar. Read `references/research.md` and
`references/method.md` first. Study the previous campaigns' HANDOFF.md
sections and the last two campaigns' notebook entries to see how a bring-up
was structured before; copy that structure.

1. **Artifact.** Register the source and the profile in
   `slimserve/profiles.json` (a `sources.<key>` entry with repo / revision /
   files / sha256 / `min_memory_bytes` or `min_gpus` per platform, a
   `profiles.<id>` entry with one `{{PLATFORM}}` variant, `status:
   "in-progress"` with `status_reason`, engine args and env copied from the
   closest existing record and every serving-policy field stated
   explicitly). Then `slimserve {{ID}} -y` to fetch or resume the download
   in the background while you continue. Dump the header / config: tensor
   census by quant type, sizes, architecture facts, tokenizer, chat template,
   vision sidecars, MTP / drafter blocks. Registry tests must pass:
   `PYTHONPATH={{WORKTREE}} {{PYTHON}} -m pytest tests/slimserve -q`.
2. **Box.** Memory budget (weights resident, KV pool, per-sequence state,
   working-set ceiling), per-token bytes by tensor class, and the bandwidth
   ladder (best single kernel measured here, sustained, current) so the
   ceiling is a number before the first wave. Environment audit per
   `references/platform-ops.md`.
3. **Research digest.** Online and in-repo, per `references/research.md`:
   the quant author's engine and its kernels for this layout, upstream
   PRs / issues for the model on vLLM / SGLang / llama.cpp / MLX, published
   speeds with their protocol, every drafter available for the model
   (verify online), and every technique from the other platforms'
   notebooks that transfers, ranked by expected step time recovered. Write
   it as copy-this / avoid-that with sources.
4. **The bar.** {{BAR}}. Install or build it under `~/.local/scratch/`,
   clean memory, its best config (its own speculation on and off, its
   recommended power / clock settings), and measure on this box: decode c1
   at 1000 in / 2000 out, decode without its drafter, multi-stream if it
   batches, prefill at 1000 and 2048 (and 2500 through a server if it has
   one). Record the exact commands, versions and numbers in
   `perf/baseline_status.md` as the bar, with raw output under
   `perf/results/{{DATE}}/{{ID}}-bar/`. If it cannot run here, record why
   and take the best published number with its protocol as the bar.
5. **Blocker list** with file:line for every platform-gated branch, missing
   op and layout mismatch between the artifact and the model code on
   `main`; the reference-dump recipe for correctness (which reference, which
   layers, which prompt).
6. **The plan** as a new top section of HANDOFF.md: mission (beat the bar
   by as much as possible on single-stream decode, multi-stream decode and
   prefill), target table, memory budget, per-token bytes and ceiling
   ladder, what exists on main / what is missing, research digest, roadmap
   (bring-up steps; decode waves byte-ordered; speculation candidates;
   concurrency sizing; prefill waves; cleanup; PR), risks with mitigations,
   the gate set and boot protocol for this box, and an empty status log.
   No time estimates.

Marker evidence keys: `bar` (engine, version, config, numbers per
workload), `profile_registered` (true), `handoff_section` (heading text),
`blockers` (count and top three), `ceiling` (bytes per token, GB/s ladder,
tok/s ceiling), `drafters` (list with source and verdict-so-far).
