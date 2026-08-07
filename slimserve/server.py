# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the OpenAI-compatible server for a plan.

Both modes go through this. `--serve` hands the process over to it; the chat
prompt starts one on a private port and talks to it over HTTP. That is what
makes the interactive answers and the API answers come from one engine with one
configuration, and it is what gives the prompt token-by-token streaming, which
the offline `LLM` path cannot do.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from typing import IO

from slimserve import term
from slimserve.engine import apply_env, serve_argv
from slimserve.registry import Plan

_MODULE = "vllm.entrypoints.openai.api_server"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def exec_server(plan: Plan, host: str, port: int) -> int:
    """Replace this process with the API server."""
    apply_env(plan)
    argv = [sys.executable, "-m", _MODULE, *serve_argv(plan, host, port)]
    term.ok(
        f"serving {plan.engine.get('served_model_name', plan.title)} on "
        f"http://{host}:{port}"
    )
    os.execv(sys.executable, argv)
    return 0  # unreachable; execv does not return


class Server:
    """A child API server, owned for the lifetime of a chat session."""

    def __init__(self, plan: Plan, port: int | None = None) -> None:
        self.plan = plan
        self.host = "127.0.0.1"
        self.port = port or free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, log_path: str | None = None) -> None:
        env = dict(os.environ)
        for key, value in self.plan.env.items():
            env.setdefault(key, value)
        argv = [
            sys.executable,
            "-m",
            _MODULE,
            *serve_argv(self.plan, self.host, self.port),
        ]
        # The engine's own logs would shred the prompt, so they go to a file
        # and the path is printed once, up front, for when a load goes wrong.
        self._log = open(log_path or os.devnull, "wb")  # noqa: SIM115 -- closed in stop()
        if log_path:
            term.info(f"engine log: {log_path}")
        self.process = subprocess.Popen(
            argv,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    def wait_until_ready(self, timeout: float = 3600.0) -> None:
        """Block until /health answers, or the child dies trying."""
        import requests

        assert self.process is not None
        started = time.monotonic()
        spinner = "|/-\\"
        tick = 0
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"engine exited with code {self.process.returncode} "
                    "before it became ready"
                )
            with contextlib.suppress(Exception):
                if (
                    requests.get(f"{self.base_url}/health", timeout=2).status_code
                    == 200
                ):
                    elapsed = time.monotonic() - started
                    term.progress(f"loaded in {elapsed:.0f}s", done=True)
                    return
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                raise RuntimeError(f"engine not ready after {timeout:.0f}s")
            term.progress(f"{spinner[tick % 4]} loading model… {elapsed:.0f}s")
            tick += 1
            time.sleep(1.0)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=30)
            if self.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self._log is not None:
            self._log.close()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
