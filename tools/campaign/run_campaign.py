#!/usr/bin/env python3
"""Drive a SlimServe profile campaign end to end with headless Claude Code sessions.

One campaign = one profile record (model x quant x platform). The driver owns
the phase sequence, the worktree, the campaign state directory and the
mechanical checks; the agent owns the work inside a phase and reports back
through a marker file (see .claude/skills/profile-campaign/SKILL.md).

    run_campaign.py start  --id glm53f-q2-1 --model-ref antirez/glm-5.3-flash-gguf \
                           --quant Q2 --bar "antirez ds4" --brief brief.md
    run_campaign.py resume --id glm53f-q2-1
    run_campaign.py status --id glm53f-q2-1
    run_campaign.py render --id glm53f-q2-1 --phase decode   # print the prompt
    run_campaign.py stop   --id glm53f-q2-1

State lives under $SLIMSERVE_CAMPAIGN_ROOT (default ~/.local/scratch/campaigns)/<id>/.
Python 3.9+, standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PHASES = [
    "discover",
    "bringup",
    "decode",
    "prefill",
    "cleanup",
    "pr",
    "review",
    "port",
    "done",
]
TOOLS_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = TOOLS_DIR / "prompts"
SKILL_REL = Path(".claude/skills/profile-campaign/SKILL.md")
STATE_ROOT = Path(
    os.environ.get(
        "SLIMSERVE_CAMPAIGN_ROOT", str(Path.home() / ".local/scratch/campaigns")
    )
)
DEFAULT_REF_TREES = [
    "~/llama.cpp",
    "~/ds4",
    "~/.local/scratch/ds4-upstream",
    "~/QuixiCore",
    "~/Code/QuixiCore-Metal",
    "~/Code/QuixiCore-CUDA",
    "~/Code/QuixiCore-ROCm",
    "~/models",
    "~/.local/scratch",
]
NOREPLY = "users.noreply.github.com"
DELEGATION_TEXT = {
    "none": (
        "`--delegate none`: do not spawn subagents. You do every task yourself, "
        "including research."
    ),
    "research": (
        "`--delegate research`: you write every line on the serving path yourself "
        "(kernels, bindings, model/layer code, adapter, profile, gates, retain/"
        "reject decisions). You may spawn subagents (Agent tool; prefer model "
        "`opus`, `sonnet` for pure reading) only for read-only or mechanically "
        "verifiable work with a one-paragraph spec and a named output: research "
        "sweeps that return a digest with sources, log/trace/profile parsing, "
        "drafting review-thread replies from a fix you already made, the byte-"
        "identical kernel port with its drift check, polling a reviewer. Read "
        "every result before using it. A subagent never edits csrc/, vllm/, "
        "slimserve/ or tests/."
    ),
    "verified": (
        "`--delegate verified`: as `research`, and subagents may also write "
        "tests and microbenches against an oracle and shapes you specify, which "
        "you run and read before trusting. Serving-path code (csrc/, vllm/, "
        "slimserve/) stays yours."
    ),
}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    if input_text is None:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            timeout=timeout,
            check=check,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=timeout,
        check=check,
        capture_output=True,
        text=True,
        input=input_text,
    )


def which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for cand in (Path.home() / ".local/bin" / name, Path("/opt/homebrew/bin") / name):
        if cand.exists():
            return str(cand)
    return None


class Campaign:
    def __init__(self, cid: str):
        self.id = cid
        self.dir = STATE_ROOT / cid
        self.state_path = self.dir / "state.json"
        self.state: dict[str, Any] = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        tmp.replace(self.state_path)

    def log(self, msg: str) -> None:
        line = f"[{utc()}] {msg}"
        print(line, flush=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "campaign.log", "a") as fh:
            fh.write(line + "\n")

    # -- paths ---------------------------------------------------------------
    @property
    def worktree(self) -> Path:
        return Path(self.state["worktree"])

    @property
    def repo(self) -> Path:
        return Path(self.state["repo"])

    @property
    def python(self) -> str:
        return self.state["python"]

    def marker_path(self, phase: str) -> Path:
        return self.dir / f"phase-{phase}.json"

    def read_marker(self, phase: str) -> dict[str, Any] | None:
        p = self.marker_path(phase)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            self.log(f"marker {p} is not valid JSON ({exc}); ignoring")
            return None


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def detect_platform(repo: Path, python: str) -> dict[str, Any]:
    fields = ["platform", "device_name", "count", "memory_bytes", "host_ram_bytes"]
    code = (
        "import json, slimserve.hardware as h; m = h.detect(); "
        f"print(json.dumps({{k: getattr(m, k, None) for k in {fields!r}}}))"
    )
    env = dict(os.environ, PYTHONPATH=str(repo))
    out = run([python, "-c", code], cwd=repo, env=env, timeout=60)
    if out.returncode != 0:
        raise SystemExit(f"hardware detection failed:\n{out.stderr}")
    raw = json.loads(out.stdout.strip().splitlines()[-1])
    return {
        "platform": raw.get("platform"),
        "device": raw.get("device_name") or "",
        "count": raw.get("count") or 0,
        "memory_bytes": raw.get("memory_bytes") or 0,
        "host_ram_bytes": raw.get("host_ram_bytes") or 0,
    }


def git(worktree: Path, *args: str, check: bool = True) -> str:
    out = run(["git", *args], cwd=worktree)
    if check and out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {worktree}:\n{out.stderr}")
    return out.stdout.strip()


def _repo_slug(worktree: Path, remote: str) -> str | None:
    url = git(worktree, "remote", "get-url", remote, check=False)
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def preflight(c: Campaign, args: argparse.Namespace) -> None:
    """Refuse to start until every requirement holds; print the fix for each."""
    rows: list[tuple[str, str, str]] = []  # (status, check, detail-or-fix)

    def req(ok: bool, check: str, fix: str, warn_only: bool = False) -> None:
        rows.append(("ok" if ok else ("warn" if warn_only else "FAIL"), check, fix))

    for tool in ("claude", "gh", "git"):
        req(bool(which(tool)), f"{tool} on PATH", f"install {tool}")
    auth = run(["claude", "auth", "status"], timeout=60)
    try:
        logged = bool(json.loads(auth.stdout).get("loggedIn"))
    except ValueError:
        logged = False
    req(logged, "claude logged in", "run: claude auth login")
    gh_auth = run(["gh", "auth", "status"], timeout=60)
    req(gh_auth.returncode == 0, "gh logged in", "run: gh auth login")
    slug = _repo_slug(c.worktree, c.state["remote"])
    if slug:
        perm = run(
            ["gh", "api", f"repos/{slug}", "--jq", ".permissions.push"], timeout=60
        )
        req(
            perm.stdout.strip() == "true",
            f"push rights on {slug}",
            f"the gh account needs push access to {slug} "
            "(or fork it and pass --remote)",
        )
        c.state["repo_slug"] = slug
    else:
        req(
            False,
            "remote is a GitHub repo",
            f"remote {c.state['remote']} is not github.com",
        )
    email = git(c.worktree, "config", "user.email", check=False)
    req(
        email.endswith(NOREPLY) or (bool(args.git_email) and email == args.git_email),
        f"worktree git identity {email!r}",
        f"run: git -C {c.worktree} config user.email <id>+<login>@{NOREPLY}",
    )
    glob_email = run(["git", "config", "--global", "user.email"]).stdout.strip()
    req(
        glob_email.endswith(NOREPLY),
        f"global git identity {glob_email!r}",
        f"run: git config --global user.email <id>+<login>@{NOREPLY} "
        "(a fresh clone would leak the real address)",
        warn_only=True,
    )
    imp = run(
        [c.python, "-c", "import slimserve, vllm"],
        cwd=c.worktree,
        env=dict(os.environ, PYTHONPATH=str(c.worktree)),
        timeout=300,
    )
    err = imp.stderr.strip().splitlines()[-1][:160] if imp.returncode else ""
    req(
        imp.returncode == 0,
        f"{c.python} imports slimserve + vllm",
        f"pass --python <venv interpreter with vLLM/SlimServe installed>: {err}",
    )
    req(
        bool(which("uvx")),
        "uvx on PATH (ruff for the guard)",
        "install uv",
        warn_only=True,
    )
    astra: dict[str, Any] = {"available": False, "detail": "codex CLI not on PATH"}
    if which("codex"):
        probe = run(
            [
                sys.executable,
                str(TOOLS_DIR / "review.py"),
                "astra-preflight",
                "--model",
                args.astra_model,
            ],
            timeout=300,
        )
        try:
            astra = json.loads(probe.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            astra = {"available": False, "detail": (probe.stdout + probe.stderr)[-300:]}
    c.state["astra"] = dict(
        astra, model=args.astra_model, checked_at=utc(), required=not args.no_astra
    )
    req(
        bool(astra.get("available")) or args.no_astra,
        f"Astra review ({args.astra_model} via codex)",
        "run: codex logout && codex login  (then re-run), or pass --no-astra to review "
        f"with CodeRabbit only. Detail: {astra.get('detail', '')[:160]}",
        warn_only=args.no_astra,
    )
    trees = [
        str(Path(p).expanduser()) for p in DEFAULT_REF_TREES + (args.ref_tree or [])
    ]
    c.state["ref_trees"] = [p for p in trees if Path(p).exists()]
    req(
        bool(c.state["ref_trees"]),
        f"reference trees ({len(c.state['ref_trees'])} of {len(trees)})",
        "none of the usual reference trees exist; pass --ref-tree",
        warn_only=True,
    )

    width = max(len(r[1]) for r in rows)
    c.log("PREFLIGHT")
    for status, check, fix in rows:
        line = f"  {status:4s} {check.ljust(width)}"
        if status != "ok" and fix:
            line += f"  -> {fix}"
        c.log(line)
    failed = [r for r in rows if r[0] == "FAIL"]
    if failed:
        c.log(
            f"preflight failed ({len(failed)}); fix the items above, "
            "then `start --force`"
        )
        c.save()
        raise SystemExit(2)


def ensure_worktree(c: Campaign, args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"{repo} is not a git checkout")
    branch = args.branch or f"{c.id}-campaign"
    worktree = (
        Path(args.worktree).expanduser().resolve()
        if args.worktree
        else repo.parent / f"{repo.name}-{c.id}"
    )
    if worktree.exists():
        cur = git(worktree, "rev-parse", "--abbrev-ref", "HEAD", check=False)
        c.log(f"reusing worktree {worktree} on {cur}")
    else:
        run(["git", "fetch", args.remote], cwd=repo, timeout=600)
        exists = (
            run(
                ["git", "rev-parse", "--verify", "--quiet", branch], cwd=repo
            ).returncode
            == 0
        )
        if exists:
            out = run(["git", "worktree", "add", str(worktree), branch], cwd=repo)
        else:
            out = run(
                ["git", "worktree", "add", "-b", branch, str(worktree), args.base],
                cwd=repo,
            )
        if out.returncode != 0:
            raise SystemExit(f"git worktree add failed:\n{out.stderr}")
        c.log(f"created worktree {worktree} on {branch} from {args.base}")
    c.state.update(
        repo=str(repo),
        worktree=str(worktree),
        branch=branch,
        base=args.base,
        remote=args.remote,
    )
    python = args.python or (
        str(repo / ".venv/bin/python")
        if (repo / ".venv/bin/python").exists()
        else sys.executable
    )
    c.state["python"] = python


def write_brief(c: Campaign, args: argparse.Namespace) -> None:
    brief_path = c.dir / "brief.md"
    text = Path(args.brief).expanduser().read_text() if args.brief else ""
    bar = args.bar or (
        "choose in discover: the fastest public way to run this quant on this box"
    )
    header = [
        f"Profile id: {c.id}",
        f"Model / artifact: {args.model_ref}",
        f"Quant: {args.quant}",
        (
            f"Platform: {c.state['platform']['platform']} "
            f"({c.state['platform']['device']} x{c.state['platform']['count']})"
        ),
        f"Bar (baseline engine): {bar}",
        (
            "Goal: beat the bar on single-stream decode, multi-stream decode and "
            "prefill by as much as possible; "
            "a supported profile record with pinned gates; a PR reviewed by CodeRabbit "
            "and Astra and ready for a human."
        ),
    ]
    brief_path.write_text("\n".join(header) + "\n\n" + text.strip() + "\n")
    c.state["brief_path"] = str(brief_path)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def render_prompt(c: Campaign, phase: str, feedback: str = "") -> str:
    attempt = c.state.get("attempts", {}).get(phase, 0) + 1
    ctx = {
        "ID": c.id,
        "PHASE": phase,
        "ATTEMPT": str(attempt),
        "MODEL_REF": c.state.get("model_ref", ""),
        "QUANT": c.state.get("quant", ""),
        "PLATFORM": c.state["platform"]["platform"] or "unknown",
        "DEVICE": f"{c.state['platform']['device']} x{c.state['platform']['count']}",
        "BAR": c.state.get("bar") or "not fixed by the operator: choose it in discover",
        "BRIEF": Path(c.state["brief_path"]).read_text().strip(),
        "REPO": c.state["repo"],
        "WORKTREE": c.state["worktree"],
        "BRANCH": c.state["branch"],
        "BASE": c.state["base"],
        "REMOTE": c.state["remote"],
        "PYTHON": c.state["python"],
        "STATE_DIR": str(c.dir),
        "MARKER_PATH": str(c.marker_path(phase)),
        "PR": str(c.state.get("pr") or "none yet"),
        "ASTRA_STATUS": (
            "available"
            if c.state.get("astra", {}).get("available")
            else f"unavailable ({c.state.get('astra', {}).get('detail', '')[:120]})"
        ),
        "ASTRA_MODEL": c.state.get("astra", {}).get("model", "gpt-6-astra"),
        "REF_TREES": "\n".join(f"- {p}" for p in c.state.get("ref_trees", []))
        or "- none found",
        "PHASES_DONE": ", ".join(
            p for p in PHASES if c.state.get("phase_done", {}).get(p)
        )
        or "none",
        "PORT_REPO": c.state.get("port_repo") or "not configured",
        "FEEDBACK": feedback.strip(),
        "DATE": dt.date.today().isoformat(),
        "DELEGATION": DELEGATION_TEXT[c.state.get("delegate", "research")],
    }
    common = (PROMPTS_DIR / "_common.md").read_text()
    body = (PROMPTS_DIR / f"{phase}.md").read_text()
    text = common + "\n\n" + body
    if attempt > 1 or feedback:
        text = (PROMPTS_DIR / "resume.md").read_text() + "\n\n" + text
    for key, val in ctx.items():
        text = text.replace("{{" + key + "}}", val)
    leftover = sorted(set(re.findall(r"{{([A-Z_]+)}}", text)))
    if leftover:
        raise SystemExit(f"unrendered placeholders in {phase} prompt: {leftover}")
    return text


# ---------------------------------------------------------------------------
# running the agent
# ---------------------------------------------------------------------------


def claude_command(
    c: Campaign, phase: str, args_model: str, args_effort: str, resume_sid: str | None
) -> list[str]:
    skill_text = (c.worktree / SKILL_REL).read_text()
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": str(
                                TOOLS_DIR / "hooks" / "block-ai-attribution.sh"
                            ),
                            "timeout": 10,
                        }
                    ],
                }
            ]
        },
    }
    cmd = [
        which("claude") or "claude",
        "-p",
        "--model",
        args_model,
        "--effort",
        args_effort,
        "--permission-mode",
        "bypassPermissions",
        "--permission-prompts",
        "none",
        "--output-format",
        "stream-json",
        "--verbose",
        "--autocompact",
        "auto",
        "--append-system-prompt",
        skill_text,
        "--settings",
        json.dumps(settings),
        "--name",
        f"campaign-{c.id}-{phase}",
    ]
    for tree in c.state.get("ref_trees", []):
        cmd += ["--add-dir", tree]
    cmd += ["--add-dir", str(c.dir), "--add-dir", str(c.repo)]
    if resume_sid:
        cmd += ["--resume", resume_sid]
    return cmd


def parse_stream(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"tool_uses": 0, "session_id": None, "result": None}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("type")
                if t == "system" and rec.get("subtype") == "init":
                    summary["session_id"] = rec.get("session_id")
                elif t == "assistant":
                    for blk in rec.get("message", {}).get("content", []):
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            summary["tool_uses"] += 1
                elif t == "result":
                    summary["result"] = {
                        k: rec.get(k)
                        for k in (
                            "subtype",
                            "is_error",
                            "num_turns",
                            "total_cost_usd",
                            "duration_ms",
                            "stop_reason",
                            "session_id",
                        )
                    }
                    summary["result_text"] = (rec.get("result") or "")[:2000]
                    summary["cost_by_model"] = {
                        m: round(float(u.get("costUSD") or 0), 4)
                        for m, u in (rec.get("modelUsage") or {}).items()
                    }
                    summary["session_id"] = (
                        rec.get("session_id") or summary["session_id"]
                    )
    except FileNotFoundError:
        pass
    return summary


def fingerprint(c: Campaign, phase: str) -> str:
    h = hashlib.sha256()
    h.update(git(c.worktree, "rev-parse", "HEAD", check=False).encode())
    h.update(git(c.worktree, "status", "--porcelain", check=False).encode())
    for rel in (
        "perf/optimization_status.md",
        "perf/baseline_status.md",
        "HANDOFF.md",
        "slimserve/profiles.json",
    ):
        p = c.worktree / rel
        if p.exists():
            h.update(f"{rel}:{p.stat().st_size}:{int(p.stat().st_mtime)}".encode())
    results = c.worktree / "perf/results"
    if results.exists():
        newest = 0.0
        count = 0
        for root, _dirs, files in os.walk(results):
            for f in files:
                count += 1
                with contextlib.suppress(OSError):
                    newest = max(newest, os.stat(os.path.join(root, f)).st_mtime)
        h.update(f"results:{count}:{int(newest)}".encode())
    m = c.marker_path(phase)
    if m.exists():
        h.update(m.read_bytes())
    return h.hexdigest()


def run_attempt(
    c: Campaign, phase: str, args: argparse.Namespace, feedback: str
) -> dict[str, Any]:
    attempts = c.state.setdefault("attempts", {})
    attempts[phase] = attempts.get(phase, 0) + 1
    n = attempts[phase]
    prompt = render_prompt(c, phase, feedback)
    runs = c.dir / "runs"
    runs.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = runs / f"{stamp}-{phase}-a{n}"
    base.with_suffix(".prompt.md").write_text(prompt)
    log_path = base.with_suffix(".jsonl")
    resume_sid = None
    if n > 1 and not args.fresh_sessions:
        resume_sid = c.state.get("last_session", {}).get(phase)
    cmd = claude_command(c, phase, args.model, args.effort, resume_sid)
    env = dict(os.environ)
    env.update(
        {
            "SLIMSERVE_CAMPAIGN_ID": c.id,
            "SLIMSERVE_CAMPAIGN_DIR": str(c.dir),
            "SLIMSERVE_CAMPAIGN_PHASE": phase,
            "SLIMSERVE_CAMPAIGN_WORKTREE": str(c.worktree),
            "SLIMSERVE_CAMPAIGN_PYTHON": c.python,
            "SLIMSERVE_CAMPAIGN_TOOLS": str(TOOLS_DIR),
            "CAMPAIGN_PRIVATE_EMAILS": private_emails(),
        }
    )
    c.log(
        f"phase {phase} attempt {n}: launching claude "
        f"({'resume ' + resume_sid if resume_sid else 'fresh session'}); "
        f"log {log_path.name}"
    )
    c.state["running"] = {
        "phase": phase,
        "attempt": n,
        "log": str(log_path),
        "started": utc(),
    }
    c.save()
    t0 = time.time()
    with open(log_path, "w") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(c.worktree),
            env=env,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
        )
        c.state["running"]["pid"] = proc.pid
        c.save()
        try:
            proc.communicate(
                prompt,
                timeout=args.attempt_timeout_hours * 3600
                if args.attempt_timeout_hours
                else None,
            )
        except subprocess.TimeoutExpired:
            c.log(
                f"attempt {n} exceeded {args.attempt_timeout_hours} h; terminating the "
                "session"
            )
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
    elapsed = time.time() - t0
    summary = parse_stream(log_path)
    summary.update(exit_code=proc.returncode, elapsed_s=int(elapsed), attempt=n)
    if summary.get("session_id"):
        c.state.setdefault("last_session", {})[phase] = summary["session_id"]
        c.state.setdefault("sessions", []).append(
            {
                "phase": phase,
                "attempt": n,
                "session_id": summary["session_id"],
                "log": str(log_path),
                "elapsed_s": int(elapsed),
                "cost_usd": (summary.get("result") or {}).get("total_cost_usd"),
                "cost_by_model": summary.get("cost_by_model") or {},
                "delegate": c.state.get("delegate"),
            }
        )
    c.state["running"] = None
    res = summary.get("result") or {}
    c.log(
        f"phase {phase} attempt {n} ended: exit {proc.returncode}, "
        f"{summary['tool_uses']} tool calls, "
        f"{int(elapsed / 60)} min, cost ${res.get('total_cost_usd') or 0:.2f}, stop "
        f"{res.get('stop_reason')}"
    )
    c.save()
    return summary


def private_emails() -> str:
    """Addresses that must never appear in anything pushed. Kept outside the repo."""
    found: list[str] = []
    listing = STATE_ROOT / "private_emails.txt"
    if listing.exists():
        found += [
            line.strip()
            for line in listing.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    glob_email = run(["git", "config", "--global", "user.email"]).stdout.strip()
    if glob_email and not glob_email.endswith(NOREPLY):
        found.append(glob_email)
    return ":".join(dict.fromkeys(found))


# ---------------------------------------------------------------------------
# validation of phase markers
# ---------------------------------------------------------------------------


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def profiles_json(c: Campaign) -> dict[str, Any]:
    return json.loads((c.worktree / "slimserve/profiles.json").read_text())


def registry_tests_pass(c: Campaign) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONPATH=str(c.worktree))
    out = run(
        [
            c.python,
            "-m",
            "pytest",
            "tests/slimserve",
            "-q",
            "-x",
            "-p",
            "no:cacheprovider",
        ],
        cwd=c.worktree,
        env=env,
        timeout=900,
    )
    tail = (out.stdout + out.stderr).strip().splitlines()[-3:]
    return out.returncode == 0, " | ".join(tail)


def guard_passes(c: Campaign) -> tuple[bool, str]:
    env = dict(
        os.environ, CAMPAIGN_PRIVATE_EMAILS=private_emails(), CAMPAIGN_PYTHON=c.python
    )
    out = run(
        [
            "bash",
            str(TOOLS_DIR / "guard.sh"),
            "--base",
            f"{c.state['remote']}/{c.state['base'].split('/')[-1]}",
        ],
        cwd=c.worktree,
        env=env,
        timeout=1800,
    )
    return out.returncode == 0, (out.stdout + out.stderr).strip()[-1500:]


def pr_view(c: Campaign, fields: str) -> dict[str, Any] | None:
    pr = c.state.get("pr")
    if not pr:
        return None
    out = run(
        ["gh", "pr", "view", str(pr), "--json", fields], cwd=c.worktree, timeout=120
    )
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def validate_marker(
    c: Campaign, phase: str, marker: dict[str, Any]
) -> tuple[bool, list[str]]:
    why: list[str] = []
    ev = marker.get("evidence") or {}
    if not isinstance(ev, dict):
        return False, ["marker.evidence must be an object"]
    if len(str(marker.get("summary", ""))) < 40:
        why.append("marker.summary is too short to be a handoff")
    wt = c.worktree

    def file_mentions(rel: str, needle: str, head_lines: int | None = None) -> bool:
        p = wt / rel
        if not p.exists():
            return False
        text = p.read_text(errors="replace")
        if head_lines:
            text = "\n".join(text.splitlines()[:head_lines])
        return needle.lower() in text.lower()

    if phase == "discover":
        if c.id not in profiles_json(c).get("profiles", {}):
            why.append(f"profile {c.id} is not registered in slimserve/profiles.json")
        if not file_mentions("HANDOFF.md", c.id, head_lines=120):
            why.append(
                "the campaign plan is not the top section of HANDOFF.md (profile id "
                "absent from its first 120 lines)"
            )
        if not file_mentions("perf/baseline_status.md", c.id):
            why.append(
                "perf/baseline_status.md has no entry naming the profile (the bar must "
                "be recorded there)"
            )
        bar = ev.get("bar")
        if not (isinstance(bar, dict) and any(_num(v) for v in _flatten(bar))):
            why.append("evidence.bar must hold the measured baseline-engine numbers")
        ok, tail = registry_tests_pass(c)
        if not ok:
            why.append(f"registry tests fail: {tail}")
    elif phase == "bringup":
        pins = ev.get("pins")
        if not (
            isinstance(pins, list)
            and pins
            and all(isinstance(p, dict) and ("sha" in p or "value" in p) for p in pins)
        ):
            why.append(
                "evidence.pins must list the N0 exact-token pins (each with a sha or "
                "value)"
            )
        if not file_mentions("perf/optimization_status.md", c.id):
            why.append("no notebook entry names the profile yet")
        if ev.get("boots") is not True:
            why.append(
                "evidence.boots must be true (the profile reached /health through "
                "slimserve --serve)"
            )
    elif phase == "decode":
        ss = ev.get("single_stream") or {}
        if not (_num(ss.get("ours")) and _num(ss.get("bar"))):
            why.append(
                "evidence.single_stream needs numeric ours and bar (tok/s, same "
                "workload)"
            )
        conc = ev.get("concurrency")
        if not (
            isinstance(conc, list)
            and len(conc) >= 2
            and all(
                _num(r.get("c")) and _num(r.get("aggregate"))
                for r in conc
                if isinstance(r, dict)
            )
        ):
            why.append(
                "evidence.concurrency must list >= 2 rows {c, aggregate, per_request} "
                "from exact-token gates"
            )
        if not isinstance(ev.get("speculation"), dict):
            why.append(
                "evidence.speculation must state the drafter verdict (drafter, k, "
                "acceptance, or none + reason)"
            )
        sc = ev.get("scaling") or {}
        if not (_num(sc.get("c4_over_c1")) or _num(sc.get("cmax_over_c1"))):
            why.append("evidence.scaling needs c4_over_c1 and/or cmax_over_c1")
        if (
            _num(ss.get("ours"))
            and _num(ss.get("bar"))
            and ss["ours"] < ss["bar"]
            and len(str(ev.get("ceiling_defended", ""))) < 60
        ):
            why.append(
                "single-stream is below the bar and evidence.ceiling_defended does not "
                "explain why (roofline + residual)"
            )
    elif phase == "prefill":
        pf = ev.get("prefill")
        if not (
            isinstance(pf, list)
            and len(pf) >= 2
            and all(isinstance(r, dict) and _num(r.get("ours")) for r in pf)
        ):
            why.append("evidence.prefill must list >= 2 lengths {tokens, ours, bar}")
    elif phase == "cleanup":
        dirty = git(wt, "status", "--porcelain", "--untracked-files=no", check=False)
        if dirty:
            why.append(
                "tracked files are uncommitted (cleanup ends with domain-sliced "
                "commits):\n" + dirty[:800]
            )
        ok, tail = registry_tests_pass(c)
        if not ok:
            why.append(f"registry tests fail: {tail}")
        ok, tail = guard_passes(c)
        if not ok:
            why.append("tools/campaign/guard.sh fails:\n" + tail)
        prof = profiles_json(c).get("profiles", {}).get(c.id, {})
        statuses = [prof.get("status")] + [
            v.get("status") for v in (prof.get("variants") or {}).values()
        ]
        gated = any(s not in (None, "supported") for s in statuses)
        if gated and len(str(ev.get("status_reason", ""))) < 40:
            why.append(
                "the profile is still gated (status != supported) and "
                "evidence.status_reason does not say why"
            )
        desc = ev.get("pr_description")
        if not (desc and (wt / desc).exists()):
            why.append(
                "evidence.pr_description must point at the drafted PR description file"
            )
    elif phase == "pr":
        pr = ev.get("pr")
        if not isinstance(pr, int):
            why.append("evidence.pr must be the PR number")
        else:
            c.state["pr"] = pr
            view = pr_view(c, "number,url,isDraft,headRefName")
            if not view:
                why.append(f"gh pr view {pr} failed")
            elif view.get("headRefName") != c.state["branch"]:
                why.append(
                    f"PR {pr} head is {view.get('headRefName')}, not "
                    f"{c.state['branch']}"
                )
            else:
                c.state["pr_url"] = view.get("url")
        ok, tail = guard_passes(c)
        if not ok:
            why.append("tools/campaign/guard.sh fails:\n" + tail)
    elif phase == "review":
        view = pr_view(c, "isDraft,url,state")
        if not view:
            why.append("no PR recorded / gh pr view failed")
        else:
            if view.get("isDraft"):
                why.append(
                    "the PR is still a draft (gh pr ready <n> when the last round is "
                    "clean)"
                )
            if view.get("state") not in ("OPEN", "MERGED"):
                why.append(f"PR state is {view.get('state')}")
        status = run(
            [
                sys.executable,
                str(TOOLS_DIR / "review.py"),
                "cr-status",
                "--pr",
                str(c.state.get("pr") or 0),
            ],
            cwd=wt,
            timeout=300,
        )
        try:
            st = json.loads(status.stdout.strip().splitlines()[-1])
            if st.get("unresolved", 1) != 0:
                why.append(f"{st.get('unresolved')} review threads are unresolved")
            if st.get("head_reviewed") is False:
                why.append(
                    "CodeRabbit has not reviewed the current head (trigger it and wait "
                    "for the round)"
                )
        except (ValueError, IndexError):
            why.append(
                "review.py cr-status did not return JSON: "
                + (status.stdout + status.stderr)[-300:]
            )
        astra = str(ev.get("astra", ""))
        if not astra.startswith(("clean", "unavailable", "declined")):
            why.append(
                "evidence.astra must be 'clean', 'unavailable (...)' or "
                "'declined-with-reasoning: ...'"
            )
        if c.state.get("astra", {}).get("available") and astra.startswith(
            "unavailable"
        ):
            why.append(
                "Astra was available at preflight; run the review (review.py "
                "astra-review) instead of marking it unavailable"
            )
    elif phase == "port":
        if not (ev.get("port_branch") or ev.get("port_pr")):
            why.append("evidence.port_branch or evidence.port_pr is required")
    return (not why), why


def _flatten(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _flatten(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _flatten(v)]
    return [obj]


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def advance(c: Campaign, phase: str) -> None:
    c.state.setdefault("phase_done", {})[phase] = utc()
    idx = PHASES.index(phase)
    nxt = PHASES[idx + 1]
    if nxt == "port" and not c.state.get("port_repo"):
        c.state.setdefault("phase_done", {})["port"] = "skipped (no --port-repo)"
        nxt = "done"
    c.state["phase"] = nxt
    c.log(f"phase {phase} DONE -> {nxt}")
    c.save()


def main_loop(c: Campaign, args: argparse.Namespace) -> int:
    stall: dict[str, int] = c.state.setdefault("stall", {})
    last_fp: dict[str, str] = c.state.setdefault("last_fingerprint", {})
    feedback = ""
    started = time.time()
    while c.state["phase"] != "done":
        phase = c.state["phase"]
        if args.max_hours and (time.time() - started) > args.max_hours * 3600:
            c.log(
                f"campaign wall-clock budget of {args.max_hours} h reached; stopping "
                "(resume with `resume`)"
            )
            return 3
        marker = c.read_marker(phase)
        if marker and marker.get("status") == "blocked":
            block = c.dir / "BLOCKED.md"
            block.write_text(
                f"# Campaign {c.id} blocked in phase "
                f"{phase}\n\n{marker.get('summary', '')}\n\n"
                f"Question for the operator:\n{marker.get('next', '')}\n\n"
                f"Evidence:\n{json.dumps(marker.get('evidence', {}), indent=1)}\n"
            )
            c.log(f"BLOCKED in {phase}: {marker.get('next', '')[:300]} (see {block})")
            c.save()
            return 2
        if marker and marker.get("status") == "done":
            ok, why = validate_marker(c, phase, marker)
            if ok:
                (c.dir / f"phase-{phase}.accepted.json").write_text(
                    json.dumps(marker, indent=1)
                )
                advance(c, phase)
                feedback = ""
                continue
            feedback = (
                "The driver rejected your done marker for phase "
                + phase
                + " because:\n- "
                + "\n- ".join(why)
                + "\nFix the underlying gaps (not the marker text), then rewrite the "
                "marker."
            )
            c.log(f"marker for {phase} rejected: {why}")
            c.marker_path(phase).rename(
                c.dir / f"phase-{phase}.rejected-{int(time.time())}.json"
            )
        if c.state.get("attempts", {}).get(phase, 0) >= args.max_attempts:
            c.log(
                f"phase {phase} hit the attempt limit ({args.max_attempts}); stopping "
                "for a human"
            )
            (c.dir / "BLOCKED.md").write_text(
                f"# Campaign {c.id}: phase {phase} exceeded {args.max_attempts} "
                "attempts\n\n"
                f"Last feedback:\n{feedback}\n"
            )
            return 2
        before = fingerprint(c, phase)
        summary = run_attempt(c, phase, args, feedback)
        after = fingerprint(c, phase)
        if after == before:
            stall[phase] = stall.get(phase, 0) + 1
            c.log(f"no progress detected in {phase} ({stall[phase]} consecutive)")
            feedback = (feedback + "\n" if feedback else "") + (
                "Your previous session changed nothing in the tree, the notebook, "
                "perf/results or the marker. "
                "Do real work this time: read HANDOFF.md's status log, pick the next "
                "ranked item, and execute it."
            )
            if stall[phase] >= args.stall_limit:
                c.log(
                    f"phase {phase} stalled {stall[phase]} times; stopping for a human"
                )
                (c.dir / "BLOCKED.md").write_text(
                    f"# Campaign {c.id}: phase {phase} stalled\n\nNo progress across "
                    f"{stall[phase]} sessions. Last session "
                    f"output:\n\n{summary.get('result_text', '')}\n"
                )
                return 2
        else:
            stall[phase] = 0
            if not c.read_marker(phase):
                feedback = (
                    "Your previous session ended without a phase marker. Continue from "
                    "HANDOFF.md's status "
                    "log; write the marker only when the definition of done is true."
                )
        last_fp[phase] = after
        c.save()
        time.sleep(args.pause_s)
    final_report(c)
    return 0


def astra_line(c: Campaign) -> str:
    astra = c.state.get("astra", {})
    if astra.get("available"):
        return "reviewed"
    return "unavailable - " + str(astra.get("detail", ""))[:120]


def final_report(c: Campaign) -> None:
    lines = [f"# Campaign {c.id}: complete", ""]
    lines.append(f"PR: {c.state.get('pr_url') or c.state.get('pr')}")
    lines.append(f"Branch: {c.state.get('branch')} in {c.state.get('worktree')}")
    lines.append("Astra: " + astra_line(c))
    lines.append("")
    lines.append("## Phase summaries")
    for ph in PHASES:
        acc = c.dir / f"phase-{ph}.accepted.json"
        if acc.exists():
            m = json.loads(acc.read_text())
            lines.append(f"### {ph}\n{m.get('summary', '')}\n")
            lines.append(
                "```json\n" + json.dumps(m.get("evidence", {}), indent=1) + "\n```\n"
            )
    cost = sum((s.get("cost_usd") or 0) for s in c.state.get("sessions", []))
    hours = sum((s.get("elapsed_s") or 0) for s in c.state.get("sessions", [])) / 3600
    lines.append(
        f"## Sessions\n{len(c.state.get('sessions', []))} sessions, "
        f"{hours:.1f} agent-hours, ${cost:.0f} API list price.\n"
    )
    lines.append(
        "Next: a human reviews and tests the PR. Everything automated has been done."
    )
    by_model = cost_by_model(c)
    if by_model:
        lines.append(
            "By model: " + ", ".join(f"{m} ${v:.0f}" for m, v in by_model.items())
        )
    (c.dir / "FINAL_REPORT.md").write_text("\n".join(lines))
    c.state["finished"] = utc()
    c.save()
    c.log(f"campaign complete; report at {c.dir / 'FINAL_REPORT.md'}")
    if c.state.get("notify_cmd"):
        subprocess.run(c.state["notify_cmd"], shell=True)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> int:
    c = Campaign(args.id)
    if c.state and not args.force:
        raise SystemExit(
            f"campaign {args.id} already exists at {c.dir}; use `resume`, or `start "
            "--force` to re-plan"
        )
    c.dir.mkdir(parents=True, exist_ok=True)
    c.state.update(
        id=args.id,
        model_ref=args.model_ref,
        quant=args.quant,
        bar=args.bar,
        created=utc(),
        phase=args.from_phase,
        port_repo=args.port_repo,
        notify_cmd=args.notify_cmd,
        attempts={},
        sessions=[],
        phase_done={},
    )
    if args.pr:
        c.state["pr"] = args.pr
    c.state["delegate"] = args.delegate or "research"
    ensure_worktree(c, args)
    c.state["platform"] = detect_platform(c.repo, c.python)
    c.log(f"platform: {c.state['platform']}")
    preflight(c, args)
    write_brief(c, args)
    for ph in PHASES[: PHASES.index(args.from_phase)]:
        c.state["phase_done"][ph] = "skipped by --from-phase"
    c.save()
    c.log(
        f"campaign {args.id} initialised at {c.dir}; starting at phase "
        f"{args.from_phase}"
    )
    if args.plan_only:
        print(render_prompt(c, args.from_phase))
        return 0
    return main_loop(c, args)


def cmd_resume(args: argparse.Namespace) -> int:
    c = Campaign(args.id)
    if not c.state:
        raise SystemExit(f"no campaign {args.id} under {STATE_ROOT}")
    if (
        c.state.get("running")
        and c.state["running"].get("pid")
        and _alive(c.state["running"]["pid"])
    ):
        raise SystemExit(
            f"campaign {args.id} is already running (pid {c.state['running']['pid']}); "
            "`stop` it first"
        )
    c.state["running"] = None
    if args.from_phase:
        c.state["phase"] = args.from_phase
    if args.delegate:
        c.state["delegate"] = args.delegate
    c.log("resuming")
    return main_loop(c, args)


def cmd_status(args: argparse.Namespace) -> int:
    c = Campaign(args.id)
    if not c.state:
        raise SystemExit(f"no campaign {args.id} under {STATE_ROOT}")
    s = c.state
    print(
        f"campaign {c.id}: phase {s.get('phase')} "
        f"({'running' if s.get('running') else 'idle'})"
    )
    print(
        f"  worktree {s.get('worktree')} branch {s.get('branch')} "
        f"pr {s.get('pr_url') or s.get('pr') or '-'}"
    )
    print(
        f"  platform {s.get('platform', {}).get('platform')} "
        f"astra {'ok' if s.get('astra', {}).get('available') else 'unavailable'}"
    )
    for ph in PHASES:
        done = s.get("phase_done", {}).get(ph)
        att = s.get("attempts", {}).get(ph, 0)
        mark = c.read_marker(ph)
        flag = (
            "done"
            if done
            else (
                "blocked"
                if mark and mark.get("status") == "blocked"
                else ("marker pending" if mark else "")
            )
        )
        print(
            f"  {ph:9s} attempts {att:2d} {flag} "
            f"{done if isinstance(done, str) and 'skipped' in done else ''}"
        )
    if (c.dir / "BLOCKED.md").exists():
        print("\n" + (c.dir / "BLOCKED.md").read_text()[:1500])
    by_model = cost_by_model(c)
    if by_model:
        print(
            "  cost by model: "
            + ", ".join(f"{m} ${v:.0f}" for m, v in by_model.items())
        )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    c = Campaign(args.id)
    if not c.state:
        raise SystemExit(f"no campaign {args.id}")
    print(render_prompt(c, args.phase, args.feedback or ""))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    c = Campaign(args.id)
    pid = (c.state.get("running") or {}).get("pid")
    if not pid or not _alive(pid):
        print("nothing running")
        return 0
    os.kill(pid, signal.SIGINT)
    for _ in range(30):
        if not _alive(pid):
            break
        time.sleep(1)
    else:
        os.kill(pid, signal.SIGTERM)
    c.state["running"] = None
    c.save()
    c.log(f"stopped session pid {pid}")
    return 0


def cost_by_model(c: Campaign) -> dict[str, float]:
    totals: dict[str, float] = {}
    for sess in c.state.get("sessions", []):
        for model, usd in (sess.get("cost_by_model") or {}).items():
            totals[model] = round(totals.get(model, 0.0) + usd, 2)
    return totals


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def loop_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--model", default=os.environ.get("SLIMSERVE_CAMPAIGN_MODEL", "fable")
        )
        sp.add_argument(
            "--effort", default=os.environ.get("SLIMSERVE_CAMPAIGN_EFFORT", "xhigh")
        )
        sp.add_argument("--max-attempts", type=int, default=40, help="per phase")
        sp.add_argument(
            "--stall-limit",
            type=int,
            default=3,
            help="consecutive no-progress sessions before stopping",
        )
        sp.add_argument(
            "--attempt-timeout-hours",
            type=float,
            default=0,
            help="0 = no per-session limit",
        )
        sp.add_argument(
            "--max-hours",
            type=float,
            default=0,
            help="0 = no campaign wall-clock limit",
        )
        sp.add_argument(
            "--pause-s", type=int, default=20, help="pause between sessions"
        )
        sp.add_argument(
            "--fresh-sessions",
            action="store_true",
            help="never --resume; every attempt starts from HANDOFF.md",
        )
        sp.add_argument("--from-phase", choices=PHASES[:-1], default=None)
        sp.add_argument(
            "--delegate",
            choices=sorted(DELEGATION_TEXT),
            default=None,
            help="what the orchestrator may hand to subagents (default research)",
        )

    s = sub.add_parser("start", help="plan and run a new campaign")
    s.add_argument(
        "--id",
        required=True,
        help="profile id, e.g. glm53f-q2-1 (<model>-<quant>-<gpus>)",
    )
    s.add_argument(
        "--model-ref",
        required=True,
        help="HF repo / GGUF / checkpoint the profile serves",
    )
    s.add_argument(
        "--quant", required=True, help="quant tag as registered in profiles.json"
    )
    s.add_argument(
        "--bar",
        default=None,
        help="baseline engine to beat (e.g. 'antirez ds4 upstream', 'vanilla vLLM'); "
        "chosen in discover if omitted",
    )
    s.add_argument(
        "--brief",
        default=None,
        help="free-text operator brief (PR links, constraints, must-use techniques)",
    )
    s.add_argument(
        "--repo",
        default=os.environ.get("SLIMSERVE_REPO", str(Path.cwd())),
        help="main SlimServe checkout",
    )
    s.add_argument(
        "--worktree",
        default=None,
        help="worktree path (default <repo>-<id> beside the repo)",
    )
    s.add_argument("--branch", default=None, help="branch name (default <id>-campaign)")
    s.add_argument("--base", default="origin/main")
    s.add_argument("--remote", default="origin")
    s.add_argument(
        "--python",
        default=None,
        help="interpreter with vLLM/SlimServe installed (default "
        "<repo>/.venv/bin/python)",
    )
    s.add_argument(
        "--git-email",
        default="",
        help="an additional allowed commit email (must still be a noreply address to "
        "pass the guard)",
    )
    s.add_argument("--astra-model", default="gpt-6-astra")
    s.add_argument(
        "--no-astra",
        action="store_true",
        help="run without the Astra (codex) review; otherwise codex must be logged in",
    )
    s.add_argument(
        "--ref-tree",
        action="append",
        help="extra reference tree to expose (repeatable)",
    )
    s.add_argument(
        "--port-repo",
        default=None,
        help="QuixiCore checkout to port kernels into (enables the port phase)",
    )
    s.add_argument(
        "--pr",
        type=int,
        default=None,
        help="existing PR number when starting from pr/review",
    )
    s.add_argument(
        "--notify-cmd",
        default=None,
        help="shell command run when the campaign finishes",
    )
    s.add_argument(
        "--plan-only",
        action="store_true",
        help="initialise state and print the first prompt, do not run",
    )
    s.add_argument("--force", action="store_true")
    loop_args(s)
    s.set_defaults(func=cmd_start)
    s.set_defaults(from_phase="discover")

    r = sub.add_parser("resume", help="continue an existing campaign")
    r.add_argument("--id", required=True)
    loop_args(r)
    r.set_defaults(func=cmd_resume)

    st = sub.add_parser("status")
    st.add_argument("--id", required=True)
    st.set_defaults(func=cmd_status)

    rd = sub.add_parser("render", help="print the prompt a phase would receive")
    rd.add_argument("--id", required=True)
    rd.add_argument("--phase", required=True, choices=PHASES[:-1])
    rd.add_argument("--feedback", default="")
    rd.set_defaults(func=cmd_render)

    sp = sub.add_parser("stop")
    sp.add_argument("--id", required=True)
    sp.set_defaults(func=cmd_stop)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "start" and args.from_phase is None:
        args.from_phase = "discover"
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
