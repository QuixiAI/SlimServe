#!/usr/bin/env python3
"""Boot a server for a campaign, detached, and record it as campaign-owned.

    tools/campaign/boot_server.py --log LOG [--cwd DIR] -- CMD [ARG ...]

Starts CMD in its own session (``start_new_session=True``, so the agent
harness cannot take it down with the session's process tree), appends
``pid<TAB>start-time<TAB>command`` to $SLIMSERVE_CAMPAIGN_PID_FILE and prints
the PID. The campaign's PreToolUse hook lets the agent stop exactly these
servers (their process group and children) unless the operator passed
--allow-kill-servers. Stop one with ``kill <pid>`` or ``kill -- -<pid>``
(the whole process group).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PID_FILE_ENV = "SLIMSERVE_CAMPAIGN_PID_FILE"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True, help="server stdout/stderr")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given (use -- CMD ...)")
    pid_file = os.environ.get(PID_FILE_ENV)
    if not pid_file:
        print(f"{PID_FILE_ENV} is not set (run inside a campaign)", file=sys.stderr)
        return 2
    log = Path(args.log).expanduser()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    started = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(proc.pid)],
        capture_output=True,
        text=True,
        env=dict(os.environ, LC_ALL="C"),
    ).stdout.strip()
    line = "\t".join([str(proc.pid), " ".join(started.split()), " ".join(cmd)])
    with open(pid_file, "a") as fh:
        fh.write(line.replace("\n", " ") + "\n")
    print(proc.pid)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
