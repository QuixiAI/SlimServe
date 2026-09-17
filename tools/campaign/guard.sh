#!/usr/bin/env bash
# Pre-push guard for a SlimServe campaign branch. Run from the worktree.
#
#   tools/campaign/guard.sh [--base origin/main] [--allow-metallib]
#
# Fails (exit 1) on anything that has burned a campaign before: a commit
# identity that is not a GitHub noreply address, attribution trailers, a
# private email or a session scratchpad path in the diff, deployment
# configuration, a Metal library binary, lint errors, or failing registry
# tests. Env: CAMPAIGN_PRIVATE_EMAILS (colon-separated, never committed),
# CAMPAIGN_PYTHON (interpreter for pytest; default .venv/bin/python or python3).
set -u
BASE="origin/main"; ALLOW_METALLIB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --allow-metallib) ALLOW_METALLIB=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
fail=0
say() { printf '%s\n' "$*"; }
bad() { say "FAIL: $*"; fail=1; }
ok()  { say "ok:   $*"; }

top=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not a git worktree" >&2; exit 2; }
cd "$top"
git rev-parse --verify --quiet "$BASE" >/dev/null || { echo "base $BASE not found (git fetch?)" >&2; exit 2; }
merge_base=$(git merge-base "$BASE" HEAD)

# 1. identity
email=$(git config user.email || true)
name=$(git config user.name || true)
case "$email" in
  *@users.noreply.github.com) ok "commit identity $name <$email>";;
  *) bad "git user.email is '$email'; must be a users.noreply.github.com address (git config user.email <id>+<login>@users.noreply.github.com)";;
esac

# 2. commits since base: identities and messages
while IFS=$'\t' read -r sha ae ce; do
  case "$ae" in *@users.noreply.github.com) ;; *) bad "commit $sha author email '$ae' is not a noreply address";; esac
  case "$ce" in *@users.noreply.github.com) ;; *) bad "commit $sha committer email '$ce' is not a noreply address";; esac
done < <(git log --format='%h%x09%ae%x09%ce' "$merge_base..HEAD")
msgs=$(git log --format='%h %B' "$merge_base..HEAD")
# Attribution phrasing, not bare product names: a commit may describe tooling
# that drives an agent or a model whose name contains "gpt"; it may not credit one.
ATTRIB='co-authored-by|generated (with|by)|signed-off-by:.*(claude|codex|anthropic|openai|gpt)|(assisted|written|authored|produced|drafted|implemented) (by|with) (claude|codex|anthropic|openai|gpt|an? (ai|llm|model|agent)\b)|with (the )?(help|assistance) of (claude|codex|an? (ai|llm|model|agent))|ai[- ]assisted|automated assistance'
if printf '%s' "$msgs" | grep -niE "$ATTRIB" >/dev/null; then
  bad "a commit message since $BASE carries an attribution line:"; printf '%s' "$msgs" | grep -niE "$ATTRIB" | head -5
else
  ok "commit messages carry no attribution lines ($(git rev-list --count "$merge_base..HEAD") commits)"
fi

# 3. the diff: private emails, scratchpad paths, host paths in code
# The guard and the hook carry these patterns themselves; exclude them from the scan.
SELF=(':!tools/campaign/guard.sh' ':!tools/campaign/hooks/*')
added=$(git diff "$merge_base" --unified=0 -- . ':!*.lock' ':!*.metallib' "${SELF[@]}" | grep -E '^\+' | grep -vE '^\+\+\+' || true)
IFS=':' read -r -a privs <<< "${CAMPAIGN_PRIVATE_EMAILS:-}"
for p in ${privs[@]+"${privs[@]}"}; do
  [ -z "$p" ] && continue
  if printf '%s' "$added" | grep -F -q -- "$p"; then bad "private address present in the diff: ${p%%@*}@..."; fi
done
other_emails=$(printf '%s' "$added" | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' | grep -vE 'users\.noreply\.github\.com|example\.com|vllm\.ai|@(users|noreply)' | sort -u || true)
if [ -n "$other_emails" ]; then bad "email addresses added by this branch (allowed: noreply only):"; printf '%s\n' "$other_emails" | head -5; else ok "no email addresses added"; fi
if printf '%s' "$added" | grep -nE '/private/tmp/|/tmp/claude|/scratchpad/|claude-[0-9]+/' >/dev/null; then
  bad "session scratchpad / tmp paths in the diff:"; printf '%s' "$added" | grep -nE '/private/tmp/|/tmp/claude|/scratchpad/|claude-[0-9]+/' | head -5
else ok "no scratchpad or tmp paths"; fi
code_hosts=$(git diff "$merge_base" --unified=0 -- '*.py' '*.cu' '*.cuh' '*.metal' '*.mm' '*.h' '*.hpp' '*.cpp' '*.json' '*.toml' '*.yaml' '*.yml' '*.sh' "${SELF[@]}" | grep -E '^\+' | grep -vE '^\+\+\+' | grep -nE '/Users/[a-z]|/home/[a-z]|/raid/' || true)
if [ -n "$code_hosts" ]; then say "warn: host-specific absolute paths in code/config (documentation is fine, code is not):"; printf '%s\n' "$code_hosts" | head -5; fi

# 4. deployment configuration (mirrors .github/workflows/no-deploy-config.yml)
if git ls-files --error-unmatch deploy/ >/dev/null 2>&1; then bad "deploy/ is tracked"; else ok "deploy/ not tracked"; fi
envfiles=$(git ls-files | grep -E '(^|/)\.env$|(^|/)[^/]*\.env$|(^|/)env$' || true)
[ -n "$envfiles" ] && bad "environment files tracked: $envfiles"
PAT='(API_KEY|APIKEY|_SECRET|_PASSWORD|ACCESS_TOKEN|AUTH_TOKEN|BEARER_TOKEN|PRIVATE_KEY)(=[A-Za-z0-9_./+-]{6,}|[[:space:]]*=[[:space:]]*["'"'"'][A-Za-z0-9_./+-]{6,}["'"'"'])'
EXC='(CHANGE_ME|REPLACE_ME|PLACEHOLDER|YOUR_|EXAMPLE|DUMMY|TEST_|FAKE_)'
if git grep -nIE "$PAT" -- . ':!*.lock' ':!.github/workflows/*' ':!tools/campaign/guard.sh' | grep -viE "$EXC" >/dev/null; then bad "credential-shaped literal assignment in the tree"; else ok "no credential-shaped literals"; fi

# 5. platform binaries
if [ "$ALLOW_METALLIB" = 0 ] && git diff --name-only "$merge_base" | grep -q 'quixicore_metal.metallib'; then
  bad "vllm/quixicore_metal.metallib changed on this branch; the maintainer regenerates it (or pass --allow-metallib when the box builds the tracked Metal standard)"
fi
if git diff --name-only "$merge_base" | grep -E '\.(so|dylib|a|o)$' >/dev/null; then bad "compiled binaries in the diff: $(git diff --name-only "$merge_base" | grep -E '\.(so|dylib|a|o)$' | tr '\n' ' ')"; fi

# 6. lint (changed python files only)
pyfiles=$(git diff --name-only "$merge_base" -- '*.py' | while read -r f; do [ -f "$f" ] && echo "$f"; done)
if [ -n "$pyfiles" ]; then
  if command -v uvx >/dev/null 2>&1; then
    if uvx ruff check $pyfiles >/tmp/guard_ruff.$$ 2>&1 && uvx ruff format --check $pyfiles >>/tmp/guard_ruff.$$ 2>&1; then ok "ruff clean on $(printf '%s\n' "$pyfiles" | wc -l | tr -d ' ') files"; else bad "ruff:"; tail -20 /tmp/guard_ruff.$$; fi
    rm -f /tmp/guard_ruff.$$
  else say "warn: uvx not found; ruff skipped"; fi
fi

# 7. registry tests
PY="${CAMPAIGN_PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }
if PYTHONPATH="$top" "$PY" -m pytest tests/slimserve -q -x -p no:cacheprovider >/tmp/guard_pytest.$$ 2>&1; then ok "registry tests pass ($(tail -1 /tmp/guard_pytest.$$))"
elif [ -z "$(git diff --name-only "$merge_base" -- slimserve/ tests/slimserve/)" ]; then
  say "warn: registry tests fail on this tree but the branch touches neither slimserve/ nor tests/slimserve/ (pre-existing on $BASE):"; grep -E '^(FAILED|ERROR)' /tmp/guard_pytest.$$ | head -3
else bad "registry tests:"; tail -15 /tmp/guard_pytest.$$; fi
rm -f /tmp/guard_pytest.$$

# 8. tree state
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then say "warn: tracked files have uncommitted changes"; fi

if [ "$fail" = 0 ]; then say "GUARD PASS ($(git rev-parse --short HEAD) vs $BASE)"; exit 0; else say "GUARD FAIL"; exit 1; fi
