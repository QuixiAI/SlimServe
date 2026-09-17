# Profile campaign automation

`tools/campaign/` turns a SlimServe profile campaign into one command. A
campaign is what the earlier bring-ups did by hand: pick a model x quant x
platform, clock the best public engine for it on this machine, bring the
profile up, optimize decode (single-stream, then speculation, then
multi-stream), optimize prefill, clean the tree, open the PR, and drive it
through CodeRabbit and Astra until only a human's sign-off is left.

The driver runs headless Claude Code sessions, one phase at a time, in a
dedicated worktree, and validates each phase's exit mechanically before
moving on. The doctrine the agent follows is
`.claude/skills/profile-campaign/SKILL.md` (also usable interactively as
`/profile-campaign`); the per-phase instructions are `prompts/*.md`.

## Launch

```bash
# from the main SlimServe checkout (the one with .venv)
cd ~/Code/slimserve/SlimServe
cat > ~/.local/scratch/brief-glm53f-q2-1.md <<'B'
Beat antirez's ds4 on his GLM-5.3-Flash Q2 GGUF on this Mac Studio the way DSV4 was beaten.
Study ds4's kernels for this layout, the vLLM/SGLang/llama.cpp PRs for glm5-next, and the
perf notes from the DSV4 and Qwen3.8 Metal campaigns. Whichever drafter is fastest ships.
B
nohup python3 tools/campaign/run_campaign.py start \
    --id glm53f-q2-1 --model-ref antirez/glm-5.3-flash-gguf --quant Q2 \
    --bar "antirez ds4 upstream (~/.local/scratch/ds4-upstream), --mtp on and off" \
    --brief ~/.local/scratch/brief-glm53f-q2-1.md \
    > ~/.local/scratch/campaigns/glm53f-q2-1.driver.log 2>&1 &
```

`start` creates `<repo>-<id>` as a worktree on branch `<id>-campaign` from
`origin/main`, detects the platform with `slimserve.hardware`, checks the
git identity, `gh` auth and the Codex CLI, writes the brief, and enters the
phase loop. Run it under `nohup`, `tmux` or `screen`: it lives for days.

Other entry points:

```bash
python3 tools/campaign/run_campaign.py status --id glm53f-q2-1
python3 tools/campaign/run_campaign.py resume --id glm53f-q2-1          # after a stop, crash or reboot
python3 tools/campaign/run_campaign.py render --id glm53f-q2-1 --phase decode   # see a phase prompt
python3 tools/campaign/run_campaign.py stop   --id glm53f-q2-1

# finish an existing branch/PR that was taken part-way by hand:
python3 tools/campaign/run_campaign.py start --id glm53f-q2-1 --model-ref ... --quant Q2 \
    --worktree ~/Code/slimserve/SlimServe-glm53f --branch glm53f-metal-campaign \
    --from-phase review --pr 30 --brief brief.md
```

State lives under `~/.local/scratch/campaigns/<id>/` (override with
`SLIMSERVE_CAMPAIGN_ROOT`): `state.json`, `brief.md`, `campaign.log`,
`runs/<ts>-<phase>-a<N>.{prompt.md,jsonl}` (every session's prompt and
full stream), `phase-<phase>.json` markers, `BLOCKED.md` when a human is
needed, `FINAL_REPORT.md` at the end.

## Phases

| phase | exit (validated by the driver) |
| --- | --- |
| discover | profile registered, registry tests pass, bar recorded in `perf/baseline_status.md`, plan is the top of HANDOFF.md |
| bringup | profile boots, N0 pins recorded, notebook entry exists |
| decode | single-stream vs bar, drafter verdict, concurrency curve and scaling ratios in the marker |
| prefill | prefill table vs bar at >= 2 lengths |
| cleanup | tracked tree committed, registry tests + `guard.sh` pass, record `supported` (or reason), PR description drafted |
| pr | PR exists on the branch, guard passes |
| review | PR not draft, 0 unresolved threads, CodeRabbit reviewed the current head, Astra clean or recorded unavailable |
| port | optional (`--port-repo`): kernels ported to the QuixiCore library |

The agent signals with a JSON marker (schema in the skill). The driver
rejects a marker whose mechanical checks fail and sends the reasons back
into the next session; a session that ends without a marker is resumed
(`--resume` on the same session, or a fresh session from HANDOFF.md with
`--fresh-sessions`); three sessions with no change to the tree, notebook,
`perf/results` or marker stop the campaign with `BLOCKED.md`. A `blocked`
marker (a decision only the operator can make) does the same immediately.

## Guard rails

- `guard.sh` (run by the driver before `pr` / `review` accept, and by the
  agent before every push): noreply commit identity on every commit since
  base, no attribution or tooling lines in messages, no private emails or
  scratchpad paths in the diff, no `deploy/`, `.env` or credential-shaped
  literals (the CI guard's rules), no Metal library or compiled binaries,
  ruff clean on changed Python, registry tests pass.
- `hooks/block-ai-attribution.sh` is installed as a PreToolUse hook on every
  campaign session (through `--settings`, independent of the user's own
  settings.json): blocks `GIT_AUTHOR_*`, `--author`, non-noreply
  `user.email`, attribution trailers, tooling mentions in commit / PR text,
  private addresses in any git or gh command, and staging the metallib.
- Private addresses come from `~/.local/scratch/campaigns/private_emails.txt`
  (one per line, never in the repo) plus the machine's global git email when
  it is not a noreply address.
- Sessions run with `--permission-mode bypassPermissions` on the operator's
  own machine, `--model fable --effort xhigh` by default
  (`SLIMSERVE_CAMPAIGN_MODEL` / `SLIMSERVE_CAMPAIGN_EFFORT` or flags), and
  the reference trees that exist (`~/llama.cpp`, `~/ds4`,
  `~/.local/scratch/ds4-upstream`, `~/QuixiCore`, `~/Code/QuixiCore-*`,
  `~/models`) added as readable directories.

## Review loop mechanics

`review.py` wraps what the agent needs: `astra-preflight` (is the Codex
CLI authenticated and is `gpt-6-astra` available), `astra-review --base
origin/main --out log` (the model is always named explicitly; `codex
review --base` takes no custom prompt, use `astra-exec` for targeted
follow-ups), `cr-trigger` (`@coderabbitai review`, needed on draft PRs),
`cr-wait` (polls for a review on the current head; exits 3 on rate limit,
4 on skip), `cr-threads --unresolved`, `cr-reply --comment-id`,
`cr-resolve --thread-id`, `cr-status`. Every command prints one JSON line.

## Delegation (`--delegate`)

The orchestrating model writes every line on the serving path itself
(kernels, bindings, model code, adapter, profile, gates, retain / reject
decisions). `--delegate research` (default) lets it hand read-only or
mechanically verifiable work to cheaper subagents: research sweeps that
return a digest with sources, log / trace parsing, drafting review replies
from fixes it already made, the byte-identical kernel port, reviewer
polling. `--delegate none` forbids subagents; `--delegate verified` also
allows subagents to write tests and benches against an oracle the
orchestrator specifies, never serving-path code. Session records carry the
cost per model so the split is visible in `status` and `FINAL_REPORT.md`.

## What the operator does

Before: nothing beyond launching. `start` runs a preflight and refuses to
begin until every requirement holds, printing the exact command for each
one that does not: `claude` logged in, `gh` logged in with push rights to
the target repo, the Codex CLI logged in with the Astra model available
(`--no-astra` to run CodeRabbit-only on purpose), the repo venv importing
`slimserve` and `vllm`, `uvx` present for ruff, and the worktree's git
identity a `users.noreply.github.com` address (set the global default to
that address too, so a fresh clone on any machine cannot leak a real
email). The machine must be free to run inference: the campaign kills any
local server when it needs the memory.

During: nothing. `status` shows the phase and attempts; `campaign.log` and
`runs/*.jsonl` show what the agent is doing; HANDOFF.md's top section is
the agent's own status log.

After: read `FINAL_REPORT.md`, test the profile, review the PR. If
`BLOCKED.md` appears, answer its question in the brief (or by hand in
HANDOFF.md) and `resume`.
