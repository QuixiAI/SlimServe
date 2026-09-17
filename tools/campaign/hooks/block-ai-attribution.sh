#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher: Bash). Blocks commands that would
# forge authorship, add attribution trailers, mention the tooling in a commit
# or PR text, or post a private address. Exit 2 = block, with the reason on
# stderr for the model. Reads the hook JSON on stdin.
set -u
input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0
block() { printf 'BLOCKED by block-ai-attribution: %s\n' "$*" >&2; exit 2; }

# identity forgery
if printf '%s' "$cmd" | grep -qE 'GIT_(AUTHOR|COMMITTER)_(NAME|EMAIL)|git commit[^|;&]* --author|git config[^|;&]* user\.(name|email)'; then
  if printf '%s' "$cmd" | grep -qE 'user\.email[^|;&]*users\.noreply\.github\.com'; then :; else
    block "never set GIT_AUTHOR_*/GIT_COMMITTER_*, --author, or a non-noreply user.email; commit with the machine's configured identity"
  fi
fi
# attribution / tooling mentions in commit or PR text
if printf '%s' "$cmd" | grep -qE 'git commit|gh pr (create|edit|comment|review)|gh api [^|;&]*(comments|reviews|pulls)'; then
  ATTRIB='co-authored-by|generated (with|by)|signed-off-by:.*(claude|codex|anthropic|openai|gpt)|(assisted|written|authored|produced|drafted|implemented) (by|with) (claude|codex|anthropic|openai|gpt|an? (ai|llm|model|agent)\b)|with (the )?(help|assistance) of (claude|codex|an? (ai|llm|model|agent))|ai[- ]assisted|automated assistance'
  if printf '%s' "$cmd" | grep -qiE "$ATTRIB"; then
    block "commit messages, PR bodies and review replies must not credit a tool or model as author or assistant"
  fi
fi
# private addresses anywhere in a git/gh command
IFS=':' read -r -a privs <<< "${CAMPAIGN_PRIVATE_EMAILS:-}"
if printf '%s' "$cmd" | grep -qE '\bgit\b|\bgh\b'; then
  for p in ${privs[@]+"${privs[@]}"}; do
    [ -n "$p" ] && printf '%s' "$cmd" | grep -qF -- "$p" && block "a private email address must never reach git or GitHub"
  done
fi
# never stage the Metal library or a deploy dir
if printf '%s' "$cmd" | grep -qE 'git add[^|;&]*(quixicore_metal\.metallib|deploy/)'; then
  block "vllm/quixicore_metal.metallib and deploy/ are never staged on a campaign branch"
fi
exit 0
