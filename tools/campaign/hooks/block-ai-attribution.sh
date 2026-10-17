#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher: Bash). Enforces CLAUDE.md "Commit
# authorship": AI is never an author, human contributors keep theirs. Blocks
# commands that set an AI commit identity, add a Co-authored-by (or other
# attribution trailer) naming an AI, mention automated assistance in a commit
# or PR text (inline or in a file the command reads), or post a private
# address. Preserving a human contributor's authorship (--author,
# GIT_AUTHOR_*, a human Co-authored-by) is allowed.
# Exit 2 = block, with the reason on stderr for the model. Reads the hook
# JSON on stdin.
set -u
input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0
# The text to scan: the command plus every file it hands git / gh as message
# text (-F/--file, --body-file, --template, $(cat FILE), < FILE, ...), so a
# message written to a file first is scanned like an inline -m.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
scan=$(printf '%s' "$input" | python3 "$here/command_guard.py" expand 2>/dev/null) || scan=$cmd
[ -z "$scan" ] && scan=$cmd
block() { printf 'BLOCKED by block-ai-attribution: %s\n' "$*" >&2; exit 2; }

# Names and addresses that identify an AI tool or model, never a person.
AI_ID='claude|anthropic|codex|openai|chatgpt|copilot|gemini|\bgpt\b|gpt-[0-9]|\b(ai|llm) (agent|assistant)\b'
# AI commit identity: an author/committer set to an AI name or address.
# Setting a human identity (preserving a contributor's authorship) passes.
ident=$(printf '%s' "$cmd" | grep -oiE "GIT_(AUTHOR|COMMITTER)_(NAME|EMAIL)=(\"[^\"]*\"|'[^']*'|[^ ]*)|--author[= ](\"[^\"]*\"|'[^']*'|[^ ]*)|user\.(name|email)[= ](\"[^\"]*\"|'[^']*'|[^ ]*)" || true)
if [ -n "$ident" ] && printf '%s' "$ident" | grep -qiE "$AI_ID"; then
  block "AI is never a commit author or committer (CLAUDE.md Commit authorship); commit with the human operator's or contributor's identity"
fi
# attribution / tooling mentions in commit or PR text
if printf '%s' "$cmd" | grep -qE 'git (commit|merge|tag)|gh pr (create|edit|comment|review)|gh api [^|;&]*(comments|reviews|pulls)'; then
  # trailers naming an AI (a human Co-authored-by is legitimate)
  TRAILER="(co-authored-by|signed-off-by|assisted-by|generated-by|helped-by):.*($AI_ID)"
  ATTRIB="generated (with|by) \\[?($AI_ID|an? (ai|llm|model|agent)\b)|(assisted|written|authored|produced|drafted|implemented) (by|with) ($AI_ID|an? (ai|llm|model|agent)\b)|with (the )?(help|assistance) of ($AI_ID|an? (ai|llm|model|agent))|ai[- ]assisted|automated assistance"
  if printf '%s' "$scan" | grep -qiE "$TRAILER|$ATTRIB"; then
    block "commit messages, PR bodies and review replies must not credit an AI as author, co-author or assistant (CLAUDE.md Commit authorship)"
  fi
fi
# private addresses anywhere in a git/gh command
IFS=':' read -r -a privs <<< "${CAMPAIGN_PRIVATE_EMAILS:-}"
if printf '%s' "$cmd" | grep -qE '\bgit\b|\bgh\b'; then
  for p in ${privs[@]+"${privs[@]}"}; do
    [ -n "$p" ] && printf '%s' "$scan" | grep -qF -- "$p" && block "a private email address must never reach git or GitHub"
  done
fi
# never stage the Metal library or a deploy dir
if printf '%s' "$cmd" | grep -qE 'git add[^|;&]*(quixicore_metal\.metallib|deploy/)'; then
  block "vllm/quixicore_metal.metallib and deploy/ are never staged on a campaign branch"
fi
exit 0
