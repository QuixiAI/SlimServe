# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The campaign's enforcement layer: campaign sessions run Claude Code with
bypassPermissions, so the PreToolUse hooks, the session settings and the
pre-push guard are the only things that hold the server policy and the
authorship rule. Each test drives the real hook scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "campaign"
GUARD_PY = TOOLS / "hooks" / "command_guard.py"
ATTRIB_SH = TOOLS / "hooks" / "block-ai-attribution.sh"
GUARD_SH = TOOLS / "guard.sh"
BOOT = TOOLS / "boot_server.py"
ALLOW = "SLIMSERVE_CAMPAIGN_ALLOW_KILL"
PID_FILE = "SLIMSERVE_CAMPAIGN_PID_FILE"
HOOKS = "SLIMSERVE_CAMPAIGN_GIT_HOOKS"


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "run_campaign_under_test", TOOLS / "run_campaign.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bash(cmd: str, cwd: Path) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(cwd)}


def _hook(argv: list[str], payload: dict, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _stop(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                return
        time.sleep(0.1)


@pytest.fixture
def camp(tmp_path):
    pid_file = tmp_path / "servers.pids"
    pid_file.touch()
    env = {k: v for k, v in os.environ.items() if k != ALLOW}
    env[PID_FILE] = str(pid_file)
    env[HOOKS] = str(tmp_path / "githooks")
    env.pop("CAMPAIGN_PRIVATE_EMAILS", None)
    procs: list[int] = []
    ns = SimpleNamespace(dir=tmp_path, env=env, pid_file=pid_file, procs=procs)

    def kill(cmd: str, allow: bool = False) -> int:
        e = dict(env, **({ALLOW: "1"} if allow else {}))
        r = _hook([sys.executable, str(GUARD_PY), "kill"], _bash(cmd, tmp_path), e)
        if r.returncode == 2 and "BLOCKED" not in r.stderr:
            return -1  # the hook crashed; that is not a block
        return r.returncode

    def foreign() -> int:
        # a process this campaign did not boot
        p = subprocess.Popen(["sleep", "300"], start_new_session=True)
        procs.append(p.pid)
        return p.pid

    def boot(code: str) -> int:
        out = subprocess.run(
            [
                sys.executable,
                str(BOOT),
                "--log",
                str(tmp_path / "srv.log"),
                "--",
                sys.executable,
                "-c",
                code,
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        pid = int(out.stdout.strip())
        procs.append(pid)
        return pid

    ns.kill, ns.foreign, ns.boot = kill, foreign, boot
    yield ns
    for pid in procs:
        _stop(pid)


# ---------------------------------------------------------------------------
# server policy
# ---------------------------------------------------------------------------

KILL_FORMS = [
    "pkill -f vllm",
    "killall python3",
    "kill {pid}",
    "kill -9 {pid}",
    "sudo kill -TERM {pid}",
    "kill -TERM -- -{pid}",
    "bash -c 'kill {pid}'",
    'sh -c "pkill -f vllm"',
    "echo {pid} | xargs kill",
    "kill $(pgrep -f vllm)",
    "ls; kill `pgrep vllm`",
    "nohup pkill vllm &",
    "env FOO=1 timeout 5 kill {pid}",
    "fuser -k 8000/tcp",
    "fuser -km /dev/nvidia0",
    "systemctl stop vllm",
    "sudo systemctl restart ollama",
    "docker stop srv",
    "docker kill srv",
    "tmux kill-server",
    "tmux kill-session -t serve",
    "python3 -c 'import os; os.kill({pid}, 15)'",
    "bash <<'EOF'\npkill -f vllm\nEOF",
]


@pytest.mark.parametrize("form", KILL_FORMS)
def test_unowned_kill_is_blocked_by_default(camp, form):
    assert camp.kill(form.format(pid=camp.foreign())) == 2


@pytest.mark.parametrize("form", KILL_FORMS)
def test_allow_kill_servers_lifts_the_block(camp, form):
    assert camp.kill(form.format(pid=camp.foreign()), allow=True) == 0


def test_killing_a_campaign_booted_server_is_allowed(camp):
    pid = camp.boot("import time; time.sleep(300)")
    assert camp.kill(f"kill {pid}") == 0
    assert camp.kill(f"kill -TERM -- -{pid}") == 0  # its process group
    assert camp.kill(f'kill $(cat "${PID_FILE}")') == 0
    assert camp.kill(f"sudo kill -9 {pid}") == 0
    # an owned PID does not cover a foreign one in the same command
    assert camp.kill(f"kill {pid} {camp.foreign()}") == 2
    assert camp.kill("pkill -f sleep") == 2  # patterns are never owned


def test_children_of_a_booted_server_are_owned(camp, tmp_path):
    marker = tmp_path / "child.pid"
    camp.boot(
        "import subprocess, time, pathlib; "
        "p = subprocess.Popen(['sleep', '300']); "
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(300)"
    )
    for _ in range(100):
        if marker.exists() and marker.read_text():
            break
        time.sleep(0.05)
    child = int(marker.read_text())
    camp.procs.append(child)
    assert camp.kill(f"kill {child}") == 0


@pytest.mark.parametrize(
    "cmd",
    [
        "git commit -m 'kill the stale server path'",
        "grep -n kill vllm/worker.py",
        "kill -l",
        "echo pkill",
        "git commit -F - <<'EOF'\nkill 1 stale branch\npkill note\nEOF",
    ],
)
def test_text_that_mentions_kill_passes(camp, cmd):
    assert camp.kill(cmd) == 0


def test_liveness_probe_passes(camp):
    assert camp.kill(f"kill -0 {camp.foreign()}") == 0


def test_pid_file_is_driver_owned(camp, tmp_path):
    pid = camp.foreign()
    assert camp.kill(f'echo {pid} >> "${PID_FILE}"') == 2
    assert camp.kill(f"echo {pid} >> {camp.pid_file}") == 2
    env = dict(camp.env)
    for path, rc in ((camp.pid_file, 2), (tmp_path / "notes.md", 0)):
        payload = {
            "tool_name": "Write",
            "tool_input": {"file_path": str(path), "content": "1\n"},
            "cwd": str(tmp_path),
        }
        assert (
            _hook([sys.executable, str(GUARD_PY), "kill"], payload, env).returncode
            == rc
        )


# ---------------------------------------------------------------------------
# authorship: files the command reads are scanned like an inline -m
# ---------------------------------------------------------------------------

AI_MSG = "Fix router\n\nCo-Authored-By: Claude Opus <bot@example.com>\n"
AI_BODY = "Summary\n\nGenerated with [Claude Code](https://claude.com/claude-code)\n"
HUMAN_MSG = "Fix router\n\nCo-authored-by: Jane Doe <jane@users.noreply.github.com>\n"


def _attrib(cmd: str, cwd: Path, env: dict) -> int:
    return _hook(["bash", str(ATTRIB_SH)], _bash(cmd, cwd), env).returncode


@pytest.mark.parametrize(
    "cmd",
    [
        "git commit -F msg.txt",
        "git commit --file=msg.txt",
        "git commit --file msg.txt",
        'git commit -m "$(cat msg.txt)"',
        "git commit -m `cat msg.txt`",
        "git commit -F - < msg.txt",
        "cat msg.txt | git commit -F -",
        "git commit --template msg.txt",
        "gh pr create --title t --body-file body.md",
        "gh pr create --title t -F body.md",
        "gh pr edit 7 --body-file=body.md",
    ],
)
def test_ai_attribution_in_a_file_is_blocked(camp, cmd):
    (camp.dir / "msg.txt").write_text(AI_MSG)
    (camp.dir / "body.md").write_text(AI_BODY)
    assert _attrib(cmd, camp.dir, camp.env) == 2


@pytest.mark.parametrize(
    "cmd",
    [
        "git commit -F msg.txt",
        'git commit -m "$(cat msg.txt)"',
        "gh pr create --title t --body-file msg.txt",
        "git commit --author='Jane Doe <jane@users.noreply.github.com>' -m fix",
    ],
)
def test_human_authorship_still_passes(camp, cmd):
    (camp.dir / "msg.txt").write_text(HUMAN_MSG)
    assert _attrib(cmd, camp.dir, camp.env) == 0


@pytest.mark.parametrize(
    "cmd,rc",
    [
        ("git push --no-verify origin HEAD:br", 2),
        ("git commit --no-verify -m fix", 2),
        ("git commit -n -m fix", 2),
        ("git commit -anm fix", 2),
        ("bash -c 'git push --no-verify'", 2),
        ("git -c core.hooksPath=/dev/null push origin HEAD", 2),
        ("git config core.hooksPath /dev/null", 2),
        ("git push origin HEAD:br", 0),
        ("git commit -am 'no-verify is not passed here'", 0),
        ("git commit -m -n", 0),  # -n is the message
        ("git push -n origin HEAD", 0),  # push -n is --dry-run; hooks still run
    ],
)
def test_git_hook_bypass(camp, cmd, rc):
    r = _hook(
        [sys.executable, str(GUARD_PY), "git-bypass"], _bash(cmd, camp.dir), camp.env
    )
    assert r.returncode == rc, r.stderr


# ---------------------------------------------------------------------------
# guard.sh, pre-push installation, session settings and environment
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, env=env, timeout=120
    )


def _repo(tmp_path: Path) -> tuple[Path, dict]:
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Jane Doe",
        GIT_AUTHOR_EMAIL="jane@users.noreply.github.com",
        GIT_COMMITTER_NAME="Jane Doe",
        GIT_COMMITTER_EMAIL="jane@users.noreply.github.com",
    )
    env.pop("CAMPAIGN_PRIVATE_EMAILS", None)
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True
    )
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "config", "user.name", "Jane Doe")
    _git(repo, "config", "user.email", "jane@users.noreply.github.com")
    (repo / "README").write_text("x\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-qm", "base", env=env)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")
    return repo, env


def test_guard_checks_merge_commit_authors(tmp_path):
    repo, env = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "work")
    _git(repo, "checkout", "-qb", "side")
    (repo / "a").write_text("a\n")
    _git(repo, "add", "a")
    _git(repo, "commit", "-qm", "side", env=env)
    _git(repo, "checkout", "-q", "work")
    ai = dict(env, GIT_AUTHOR_NAME="Claude", GIT_AUTHOR_EMAIL="bot@example.com")
    r = _git(repo, "merge", "--no-ff", "-qm", "merge side", "side", env=ai)
    assert r.returncode == 0, r.stderr
    out = subprocess.run(
        ["bash", str(GUARD_SH), "--base", "origin/main"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert out.returncode == 1
    assert "AI author/committer" in out.stdout


@pytest.mark.parametrize(
    "line,flagged",
    # the address is assembled so this file's own diff carries none
    [("@pytest.fixture", False), ('owner = "jane' + "@" + 'corp.com"', True)],
)
def test_guard_email_scan_ignores_decorators(tmp_path, line, flagged):
    repo, env = _repo(tmp_path)
    (repo / "notes.txt").write_text(line + "\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-qm", "notes", env=env)
    out = subprocess.run(
        ["bash", str(GUARD_SH), "--base", "origin/main"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert ("FAIL: email addresses added" in out.stdout) is flagged, out.stdout


def test_pre_push_guard_installed_in_the_campaign_worktree(tmp_path, monkeypatch):
    repo, env = _repo(tmp_path)
    wt = tmp_path / "wt"
    assert (
        _git(repo, "worktree", "add", "-qb", "camp", str(wt), "origin/main").returncode
        == 0
    )
    drv = _load_driver()
    monkeypatch.setattr(drv, "STATE_ROOT", tmp_path / "campaigns")
    c = drv.Campaign("t1")
    c.state.update(
        repo=str(repo),
        worktree=str(wt),
        remote="origin",
        base="origin/main",
        python=sys.executable,
    )
    drv.install_git_hooks(c)
    hooks = c.dir / "githooks"
    assert _git(wt, "config", "core.hooksPath").stdout.strip() == str(hooks)
    # only the campaign worktree: the operator's checkout keeps its hooks
    assert _git(repo, "config", "core.hooksPath").stdout.strip() == ""
    assert os.access(hooks / "pre-push", os.X_OK)
    # an AI-authored commit is refused at push time, a human one goes through
    ai = dict(env, GIT_AUTHOR_NAME="Claude", GIT_AUTHOR_EMAIL="bot@example.com")
    (wt / "f").write_text("1\n")
    _git(wt, "add", "f")
    _git(wt, "commit", "-qm", "change", env=ai)
    assert _git(wt, "push", "origin", "HEAD:camp", env=env).returncode != 0
    _git(wt, "reset", "-q", "--hard", "origin/main")
    (wt / "g").write_text("1\n")
    _git(wt, "add", "g")
    _git(wt, "commit", "-qm", "change", env=env)
    r = _git(wt, "push", "origin", "HEAD:camp", env=env)
    assert r.returncode == 0, r.stdout + r.stderr


def test_session_settings_disable_cli_attribution_and_install_hooks():
    s = _load_driver().session_settings()
    assert s["includeCoAuthoredBy"] is False
    assert s["attribution"] == {"commit": "", "pr": ""}
    pre = s["hooks"]["PreToolUse"]
    bash = next(m for m in pre if m["matcher"] == "Bash")["hooks"]
    cmds = " ".join(h["command"] for h in bash)
    assert "block-ai-attribution.sh" in cmds
    assert "command_guard.py' kill" in cmds or "command_guard.py kill" in cmds
    assert "git-bypass" in cmds
    assert any("Write" in m["matcher"] for m in pre)


@pytest.mark.parametrize("flag", [False, True])
def test_allow_kill_env_follows_the_launch_flag(tmp_path, monkeypatch, flag):
    drv = _load_driver()
    monkeypatch.setattr(drv, "STATE_ROOT", tmp_path)
    monkeypatch.setenv(ALLOW, "1")  # never inherited from the operator's shell
    c = drv.Campaign("t2")
    c.state.update(
        worktree=str(tmp_path), python=sys.executable, allow_kill_servers=flag
    )
    env = drv.session_env(c, "decode")
    assert (env.get(ALLOW) == "1") is flag
    assert Path(env[PID_FILE]).is_file()


def test_quixicore_cuda_is_a_default_reference_tree(monkeypatch):
    drv = _load_driver()
    monkeypatch.delenv("SLIMSERVE_CAMPAIGN_REF_TREES", raising=False)
    assert "~/QuixiCore/QuixiCore-CUDA" in drv.default_ref_trees()
    monkeypatch.setenv("SLIMSERVE_CAMPAIGN_REF_TREES", os.pathsep.join(["/a", "/b"]))
    assert drv.default_ref_trees() == ["/a", "/b"]
