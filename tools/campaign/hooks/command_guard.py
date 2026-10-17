#!/usr/bin/env python3
"""Shell-command analysis behind the campaign's Claude Code PreToolUse hooks.

Campaign sessions run with ``--permission-mode bypassPermissions``, so these
hooks are the only enforcement of the server policy and the authorship rule.

    command_guard.py kill        hook (Bash, Write|Edit): block stopping a
                                 process the campaign did not boot
    command_guard.py git-bypass  hook (Bash): block skipping git hooks
    command_guard.py expand      print the command plus the text of every
                                 file it feeds to git / gh as a message, for
                                 block-ai-attribution.sh to scan

Hooks read the hook JSON on stdin and exit 2 (reason on stderr) to block.

Server policy. Stopping processes (``kill``, ``pkill``, ``killall``,
``fuser -k``, ``systemctl stop|restart|kill``, ``docker stop|kill``,
``tmux kill-*``, ``os.kill`` in ``python -c`` ..., also behind ``sudo``,
``bash -c``, ``xargs``, ``$(...)``) is allowed only when the operator launched
the run with ``--allow-kill-servers`` (the driver then exports
SLIMSERVE_CAMPAIGN_ALLOW_KILL=1), or when every target PID is a server the
campaign booted: listed in SLIMSERVE_CAMPAIGN_PID_FILE (written by
tools/campaign/boot_server.py), a member of such a server's process group, or
a descendant of one. ``kill -0`` (a liveness probe) is always allowed. The
PID file itself may only be written by boot_server.py.

This is a guard against an agent overreaching, not a sandbox: a script file
the agent writes and then runs is not inspected.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ALLOW_ENV = "SLIMSERVE_CAMPAIGN_ALLOW_KILL"
PID_FILE_ENV = "SLIMSERVE_CAMPAIGN_PID_FILE"
HOOKS_DIR_ENV = "SLIMSERVE_CAMPAIGN_GIT_HOOKS"
MAX_READ = 1 << 20

_SUB = "\x01{}\x01"
_SUB_RE = re.compile("\x01(\\d+)\x01")
_SEPARATORS = {";", ";;", "&", "&&", "|", "||", "|&", "(", ")", "\n"}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "fish"}
_PY = re.compile(r"^python(\d+(\.\d+)?)?$")


class Unparseable(Exception):
    pass


# ---------------------------------------------------------------------------
# lexing
# ---------------------------------------------------------------------------


def _match_close(text: str, i: int) -> int:
    """Index of the ')' closing a '$(' whose body starts at i."""
    depth, sq, dq, n = 1, False, False, len(text)
    while i < n:
        c = text[i]
        if sq:
            sq = c != "'"
        elif c == "\\":
            i += 1
        elif c == "'" and not dq:
            sq = True
        elif c == '"':
            dq = not dq
        elif not dq or c in "()":
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return n


def lift_substitutions(text: str) -> tuple[str, list[str]]:
    """Replace each $(...) and `...` with a placeholder; return the inner
    commands. Arithmetic $((...)) is left in place."""
    out: list[str] = []
    inners: list[str] = []
    i, n, sq, dq = 0, len(text), False, False
    while i < n:
        c = text[i]
        if sq:
            out.append(c)
            sq = c != "'"
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        if c == "'" and not dq:
            sq = True
        elif c == '"':
            dq = not dq
        elif c == "$" and text.startswith("$((", i):
            j = _match_close(text, i + 2)
            out.append(text[i : j + 1])
            i = j + 1
            continue
        elif c == "$" and text.startswith("$(", i):
            j = _match_close(text, i + 2)
            inners.append(text[i + 2 : j])
            out.append(_SUB.format(len(inners) - 1))
            i = j + 1
            continue
        elif c == "`":
            j = text.find("`", i + 1)
            j = n if j < 0 else j
            inners.append(text[i + 1 : j])
            out.append(_SUB.format(len(inners) - 1))
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out), inners


_HEREDOC = re.compile(r"(?<!<)<<(?!<)(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_SHELL_PREFIX = re.compile(r"(^|[\s;&|(/])(bash|sh|zsh|dash|ksh)\b")
_PY_PREFIX = re.compile(r"(^|[\s;&|(/])python(\d+(\.\d+)?)?\b")


def strip_heredocs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove here-document bodies (message text, not commands). Returns the
    remaining text and (line-before-<<, body) per document."""
    lines = text.split("\n")
    out: list[str] = []
    docs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in _HEREDOC.finditer(line):
            delim, tabs = m.group(3), m.group(1) == "-"
            body: list[str] = []
            while i < len(lines):
                ln = lines[i]
                i += 1
                if (ln.lstrip("\t") if tabs else ln) == delim:
                    break
                body.append(ln)
            docs.append((line[: m.start()], "\n".join(body)))
    return "\n".join(out), docs


def simple_commands(text: str) -> list[tuple[list[str], list[str]]]:
    """Split a (substitution-free) command line into simple commands.
    Returns (words, input-redirection targets) per command."""
    lex = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""
    try:
        toks = list(lex)
    except ValueError as exc:
        raise Unparseable(str(exc)) from exc
    cmds: list[tuple[list[str], list[str]]] = []
    words: list[str] = []
    inputs: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _SEPARATORS:
            if words:
                cmds.append((words, inputs))
            words, inputs = [], []
        elif t and set(t) <= set("<>&|"):
            # redirection: drop an fd number before it and the target after it
            if words and words[-1].isdigit() and i > 0 and toks[i - 1] == words[-1]:
                words.pop()
            if i + 1 < len(toks) and toks[i + 1] not in _SEPARATORS:
                if t in ("<", "<<<"):
                    inputs.append(toks[i + 1])
                i += 1
        else:
            words.append(t)
        i += 1
    if words:
        cmds.append((words, inputs))
    return cmds


def _skip_opts(words: list[str], with_arg: set[str] = frozenset()) -> list[str]:
    while words and words[0].startswith("-") and words[0] != "-":
        opt = words.pop(0)
        if opt == "--":
            break
        if opt in with_arg and words:
            words.pop(0)
    return words


def unwrap(words: list[str]) -> tuple[list[str], bool]:
    """Strip assignments and command wrappers (sudo, env, nohup, timeout,
    xargs ...). Returns the remaining words and whether xargs fed them."""
    words = list(words)
    via_xargs = False
    while words:
        b = os.path.basename(words[0])
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            words.pop(0)
        elif b in ("sudo", "doas"):
            words.pop(0)
            _skip_opts(words, {"-u", "-g", "-C", "-h", "-p", "-U", "-r", "-t", "-D"})
        elif b == "env":
            words.pop(0)
            while words and (
                words[0].startswith("-")
                or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])
            ):
                if words.pop(0) in ("-u", "-C", "-S") and words:
                    words.pop(0)
        elif b in (
            "nohup",
            "setsid",
            "exec",
            "command",
            "builtin",
            "time",
            "caffeinate",
            "unbuffer",
        ):
            words.pop(0)
            _skip_opts(words)
        elif b == "nice":
            words.pop(0)
            _skip_opts(words, {"-n"})
        elif b == "timeout":
            words.pop(0)
            _skip_opts(words, {"-s", "-k", "--signal", "--kill-after"})
            if words:
                words.pop(0)
        elif b in ("ionice", "stdbuf", "chrt", "taskset"):
            words.pop(0)
            _skip_opts(words, {"-c", "-n", "-p"})
            while words and re.fullmatch(r"[0-9a-fx,-]+", words[0]):
                words.pop(0)
        elif b == "xargs":
            words.pop(0)
            _skip_opts(words, {"-I", "-n", "-P", "-L", "-d", "-E", "-s", "-a"})
            via_xargs = True
        else:
            break
    return words, via_xargs


# ---------------------------------------------------------------------------
# file resolution (shared by the kill guard and the attribution scan)
# ---------------------------------------------------------------------------


def _path(p: str, cwd: str | None) -> Path:
    p = os.path.expanduser(os.path.expandvars(p))
    q = Path(p)
    if not q.is_absolute() and cwd:
        q = Path(cwd) / q
    return q


def read_file(p: str, cwd: str | None) -> str | None:
    if "$" in os.path.expandvars(p) or _SUB_RE.search(p):
        return None
    q = _path(p, cwd)
    try:
        if not q.is_file():
            return None
        with open(q, errors="replace") as fh:
            return fh.read(MAX_READ)
    except OSError:
        return None


def cat_target(inner: str) -> str | None:
    """The literal file of `cat FILE` / `< FILE`, else None."""
    s = inner.strip()
    try:
        if s.startswith("<"):
            toks = shlex.split(s[1:])
            return toks[0] if len(toks) == 1 else None
        toks = shlex.split(s)
    except ValueError:
        return None
    if len(toks) == 2 and toks[0] == "cat" and not toks[1].startswith("-"):
        return toks[1]
    return None


# ---------------------------------------------------------------------------
# process-stopping commands
# ---------------------------------------------------------------------------

DYNAMIC = None  # target set that cannot be resolved to literal PIDs


def _targets(tok: str, inners: list[str], cwd: str | None) -> list[int] | None:
    if re.fullmatch(r"-?\d+", tok):
        return [int(tok)]
    m = _SUB_RE.fullmatch(tok)
    if m:
        path = cat_target(inners[int(m.group(1))])
        body = read_file(path, cwd) if path else None
        if body is not None:
            nums = [ln.split()[0] for ln in body.splitlines() if ln.strip()]
            if nums and all(re.fullmatch(r"-?\d+", x) for x in nums):
                return [int(x) for x in nums]
    return DYNAMIC


def _kill_action(args: list[str], inners, cwd) -> tuple[str, list[int] | None] | None:
    targets: list[int] = []
    dynamic = False
    seen_dd = False
    signal = None
    i = 0
    while i < len(args):
        a = args[i]
        if not seen_dd and not targets and not dynamic:
            if a in ("-l", "-L", "--list", "--table"):
                return None
            if a == "--":
                seen_dd = True
                i += 1
                continue
            if a in ("-s", "-n", "--signal"):
                signal = args[i + 1] if i + 1 < len(args) else None
                i += 2
                continue
            if a.startswith("-"):
                signal = a[1:]
                i += 1
                continue
        t = _targets(a, inners, cwd)
        if t is None:
            dynamic = True
        else:
            targets += t
        i += 1
    if signal in ("0", "SIG0"):
        return None  # liveness probe
    if not targets and not dynamic:
        return None
    return "kill", (None if dynamic else targets)


def _first_arg(words: list[str]) -> str | None:
    for w in words:
        if not w.startswith("-"):
            return w
    return None


def kill_actions(
    text: str, cwd: str | None, depth: int = 0
) -> list[tuple[str, list[int] | None]]:
    """Every process-stopping action in a shell command line."""
    if depth > 8:
        return [("nested command", DYNAMIC)]
    text, docs = strip_heredocs(text)
    acts: list[tuple[str, list[int] | None]] = []
    for prefix, body in docs:
        # a here-document fed to a shell or python is a script, not text
        if _SHELL_PREFIX.search(prefix):
            acts += kill_actions(body, cwd, depth + 1)
        elif _PY_PREFIX.search(prefix):
            acts += kill_actions("python3 -c " + shlex.quote(body), cwd, depth + 1)
    flat, inners = lift_substitutions(text)
    for inner in inners:
        acts += kill_actions(inner, cwd, depth + 1)
    for words, _inputs in simple_commands(flat):
        words, via_xargs = unwrap(words)
        if not words:
            continue
        b = os.path.basename(words[0])
        rest = words[1:]
        if b in _SHELLS:
            for k, w in enumerate(rest):
                if w.startswith("-") and not w.startswith("--") and "c" in w[1:]:
                    if k + 1 < len(rest):
                        script = rest[k + 1]
                        script = _SUB_RE.sub(
                            lambda m: "$(" + inners[int(m.group(1))] + ")", script
                        )
                        acts += kill_actions(script, cwd, depth + 1)
                    break
        elif b == "eval":
            acts += kill_actions(" ".join(rest), cwd, depth + 1)
        elif _PY.match(b) and "-c" in rest:
            code = (
                rest[rest.index("-c") + 1] if rest.index("-c") + 1 < len(rest) else ""
            )
            calls = re.findall(r"\bos\.kill(?:pg)?\s*\(\s*([^,)]*)", code)
            if calls:
                if all(re.fullmatch(r"-?\d+", c.strip()) for c in calls):
                    acts.append(("os.kill", [int(c) for c in calls]))
                else:
                    acts.append(("os.kill", DYNAMIC))
            if (
                re.search(
                    r"\.(send_signal|terminate|kill)\s*\(|psutil|subprocess.*\bkill\b",
                    code,
                )
                and not calls
            ):
                acts.append(("python process signal", DYNAMIC))
        elif b == "kill":
            act = _kill_action(rest, inners, cwd)
            if via_xargs and not any(w in ("-l", "-L", "--list") for w in rest):
                acts.append(("xargs kill", DYNAMIC))  # PIDs arrive on stdin
            elif act:
                acts.append(act)
        elif b in ("pkill", "killall", "skill", "slay"):
            acts.append((b, DYNAMIC))
        elif b == "fuser":
            if any(
                w == "--kill" or (re.fullmatch(r"-[a-zA-Z]+", w) and "k" in w)
                for w in rest
            ):
                acts.append(("fuser -k", DYNAMIC))
        elif b == "systemctl":
            verbs = {
                "stop",
                "restart",
                "kill",
                "try-restart",
                "reload-or-restart",
                "try-reload-or-restart",
                "condrestart",
                "isolate",
                "mask",
            }
            if any(w in verbs for w in rest):
                acts.append(("systemctl", DYNAMIC))
        elif b == "service":
            if any(w in ("stop", "restart", "force-reload") for w in rest):
                acts.append(("service", DYNAMIC))
        elif b == "launchctl":
            if any(w in ("stop", "kill", "bootout", "unload", "remove") for w in rest):
                acts.append(("launchctl", DYNAMIC))
        elif b in ("docker", "podman", "nerdctl"):
            sub = _first_arg(rest)
            if sub in ("container", "compose") and sub in rest:
                sub = _first_arg(rest[rest.index(sub) + 1 :])
            if sub in ("stop", "kill", "restart", "rm", "down", "pause"):
                acts.append((f"{b} {sub}", DYNAMIC))
        elif b == "tmux" and any(w.startswith("kill-") for w in rest):
            acts.append(("tmux kill", DYNAMIC))
        elif (
            b == "screen" and "-X" in rest and any(w in ("quit", "kill") for w in rest)
        ):
            acts.append(("screen quit", DYNAMIC))
    return acts


_KILL_WORD = re.compile(
    r"(^|[\s;&|(`'\"/])(kill|pkill|killall|fuser|systemctl|launchctl)\b"
    r"|tmux\s+.*kill-|docker\s+(stop|kill)|os\.kill"
)


def safe_kill_actions(text: str, cwd: str | None) -> list[tuple[str, list[int] | None]]:
    try:
        return kill_actions(text, cwd)
    except Unparseable:
        return [("unparseable command", DYNAMIC)] if _KILL_WORD.search(text) else []


# ---------------------------------------------------------------------------
# ownership
# ---------------------------------------------------------------------------


def _ps(pid: int, field: str) -> str | None:
    try:
        out = subprocess.run(
            ["ps", "-o", f"{field}=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            env=dict(os.environ, LC_ALL="C"),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    s = out.stdout.strip()
    return s or None


def owned_pids(pid_file: str | None) -> set[int]:
    """PIDs boot_server.py recorded that still name the same process."""
    owned: set[int] = set()
    if not pid_file:
        return owned
    try:
        lines = Path(pid_file).read_text().splitlines()
    except OSError:
        return owned
    for ln in lines:
        parts = ln.split("\t")
        if not parts or not parts[0].strip().isdigit():
            continue
        pid = int(parts[0])
        started = parts[1].strip() if len(parts) > 1 else ""
        now = _ps(pid, "lstart")
        if now is None:
            continue
        if started and " ".join(now.split()) != " ".join(started.split()):
            continue  # PID reused by another process
        owned.add(pid)
    return owned


def is_owned(pid: int, owned: set[int]) -> bool:
    if pid < 0:
        return -pid in owned  # process group of a booted server
    if pid in owned:
        return True
    pgid = _ps(pid, "pgid")
    if pgid and pgid.isdigit() and int(pgid) in owned:
        return True
    cur = pid
    for _ in range(64):
        ppid = _ps(cur, "ppid")
        if not ppid or not ppid.isdigit() or int(ppid) <= 1:
            return False
        cur = int(ppid)
        if cur in owned:
            return True
    return False


def protected_paths() -> list[Path]:
    out = []
    for var in (PID_FILE_ENV, HOOKS_DIR_ENV):
        v = os.environ.get(var)
        if v:
            out.append(Path(v).expanduser().resolve())
    return out


_WRITEY = re.compile(
    r">|\btee\b|\bsed\s+-i|\brm\b|\btruncate\b|\bmv\b|\bcp\b|\bdd\b|\bln\b|"
    r"\bchmod\b|\binstall\b|\bpython|\bperl\b|\bruby\b|\bnode\b|\bwrite"
)


def check_kill(data: dict) -> str | None:
    """Reason to block, or None."""
    tool = data.get("tool_name", "")
    ti = data.get("tool_input") or {}
    cwd = data.get("cwd") or os.getcwd()
    prot = protected_paths()
    if tool != "Bash":
        fp = ti.get("file_path") or ti.get("notebook_path")
        if fp and prot:
            p = _path(fp, cwd).resolve()
            if any(p == q or q in p.parents for q in prot):
                return (
                    f"{fp} is campaign-owned (server PID file / git hooks); only "
                    "tools/campaign/boot_server.py and the driver write it"
                )
        return None
    cmd = ti.get("command") or ""
    if not cmd:
        return None
    if prot:
        names = [str(q) for q in prot] + [PID_FILE_ENV, HOOKS_DIR_ENV]
        names += [q.name for q in prot if q.name not in ("", "githooks")]
        if (
            any(n in cmd for n in names)
            and _WRITEY.search(cmd)
            and "boot_server.py" not in cmd
        ):
            return (
                "the campaign PID file and git hooks are written only by "
                "tools/campaign/boot_server.py and the driver"
            )
    acts = safe_kill_actions(cmd, cwd)
    if not acts or os.environ.get(ALLOW_ENV) == "1":
        return None
    owned = owned_pids(os.environ.get(PID_FILE_ENV))
    for what, targets in acts:
        if targets is None:
            return (
                f"`{what}` without literal PIDs of servers this campaign booted. "
                "Server policy: stop only servers started with "
                "tools/campaign/boot_server.py, by PID (e.g. "
                f'`kill $(cat "${PID_FILE_ENV}")` or `kill <pid>`). Stopping '
                "anything else needs the operator to relaunch with "
                "--allow-kill-servers; otherwise write a `blocked` marker."
            )
        bad = [t for t in targets if not is_owned(t, owned)]
        if bad:
            return (
                f"`{what}` targets PID(s) {bad} that this campaign did not boot "
                "(not in the campaign PID file, its process groups or their "
                "children). Write a `blocked` marker naming the process, or "
                "the operator relaunches with --allow-kill-servers."
            )
    return None


# ---------------------------------------------------------------------------
# git hook bypasses
# ---------------------------------------------------------------------------

_COMMIT_ARG_OPTS = {
    "-m",
    "-F",
    "-c",
    "-C",
    "-t",
    "--message",
    "--file",
    "--author",
    "--date",
    "--template",
    "--trailer",
    "--reuse-message",
    "--reedit-message",
    "--cleanup",
    "--fixup",
    "--squash",
    "--pathspec-from-file",
}


def _git_sub(words: list[str]) -> tuple[str | None, list[str]]:
    i = 1
    while i < len(words):
        w = words[i]
        if w in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            i += 2
            continue
        if w.startswith("-"):
            i += 1
            continue
        return w, words[i + 1 :]
    return None, []


def check_git_bypass(data: dict) -> str | None:
    if data.get("tool_name", "Bash") != "Bash":
        return None
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\bgit\b", cmd):
        return None
    if re.search(r"hookspath", cmd, re.I):
        return "core.hooksPath is set by the campaign driver; do not override it"
    try:
        text, docs = strip_heredocs(cmd)
        flat, inners = lift_substitutions(text)
        texts = [flat] + inners
        texts += [body for prefix, body in docs if _SHELL_PREFIX.search(prefix)]
        cmds = [c for t in texts for c in simple_commands(t)]
    except Unparseable:
        if re.search(r"git\s+(commit|push|merge)\b[^;&|]*--no-verify", cmd):
            return "--no-verify skips the campaign's git hooks"
        return None
    stack = list(cmds)
    while stack:
        words, _ = stack.pop()
        words, _x = unwrap(words)
        if not words:
            continue
        b = os.path.basename(words[0])
        if (
            b in _SHELLS
            and len(words) > 2
            and words[1].startswith("-")
            and "c" in words[1]
        ):
            with contextlib.suppress(Unparseable):
                stack += simple_commands(words[2])
            continue
        if b != "git":
            continue
        sub, args = _git_sub(words)
        if sub not in (
            "commit",
            "push",
            "merge",
            "am",
            "cherry-pick",
            "revert",
            "rebase",
        ):
            continue
        skip = False
        for a in args:
            if skip:
                skip = False
                continue
            if a == "--no-verify":
                return f"`git {sub} --no-verify` skips the campaign's git hooks"
            if sub != "commit":
                continue
            if a in _COMMIT_ARG_OPTS:
                skip = True
                continue
            if re.fullmatch(r"-[A-Za-z]+", a):
                for k, ch in enumerate(a[1:]):
                    if ch == "n":
                        return (
                            "`git commit -n` (--no-verify) skips the campaign's "
                            "git hooks"
                        )
                    if ch in "mFcCt":
                        skip = k == len(a) - 2  # value is the next word
                        break
    return None


# ---------------------------------------------------------------------------
# attribution scan text
# ---------------------------------------------------------------------------

_FILE_OPTS = {"-F", "--file", "--body-file", "--template", "-t"}


def expand(cmd: str, cwd: str | None) -> str:
    """The command plus every file it hands git/gh as message text: -F/--file,
    --body-file, --template, gh api @file fields, $(cat FILE), < FILE, and
    `cat FILE` piped into the command."""
    parts = [cmd]
    try:
        text, _docs = strip_heredocs(cmd)
        flat, inners = lift_substitutions(text)
    except Exception:
        return cmd
    files: list[str] = []
    for inner in inners:
        t = cat_target(inner)
        if t:
            files.append(t)
    texts = [flat] + inners
    for t in texts:
        try:
            cmds = simple_commands(t)
        except Unparseable:
            continue
        for words, inputs in cmds:
            files += inputs
            words, _x = unwrap(words)
            if not words:
                continue
            b = os.path.basename(words[0])
            if b in _SHELLS and len(words) > 2 and "c" in words[1]:
                parts.append(expand(words[2], cwd))
                continue
            if b == "cat":
                files += [w for w in words[1:] if not w.startswith("-")]
                continue
            if b not in ("git", "gh"):
                continue
            for k, w in enumerate(words[1:], 1):
                nxt = words[k + 1] if k + 1 < len(words) else None
                if nxt and (
                    (w in _FILE_OPTS and nxt != "-") or (w == "--input" and b == "gh")
                ):
                    files.append(nxt)
                elif "=" in w and w.split("=", 1)[0] in _FILE_OPTS:
                    files.append(w.split("=", 1)[1])
                elif re.fullmatch(r"-F[^-=].*", w) and b == "git":
                    files.append(w[2:])
                if b == "gh" and "=@" in w:
                    files.append(w.split("=@", 1)[1])
    for f in files:
        if _SUB_RE.search(f):
            continue
        body = read_file(f, cwd)
        if body:
            parts.append(body)
    return "\n".join(parts)


# ---------------------------------------------------------------------------


def _hook(fn) -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    reason = fn(data)
    if reason:
        print(f"BLOCKED by campaign command_guard: {reason}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode == "kill":
        return _hook(check_kill)
    if mode == "git-bypass":
        return _hook(check_git_bypass)
    if mode == "expand":
        data = json.load(sys.stdin)
        ti = data.get("tool_input") or {}
        sys.stdout.write(expand(ti.get("command") or "", data.get("cwd")))
        return 0
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
