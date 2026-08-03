# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Terminal output vocabulary.

One colour per kind of message, applied only when the destination is a TTY, so
a piped run is plain text. Generated tokens go to stdout; everything else goes
to stderr, which is what lets `slimserve k3-6 -p ... > answer.txt` work.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
GREY = "\x1b[90m"
ORANGE = "\x1b[38;5;208m"
RED = "\x1b[31m"
BOLD = "\x1b[1m"
DIM_WHITE = "\x1b[38;5;250m"
RESET = "\x1b[0m"


def is_tty(fp: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return fp.isatty()
    except (AttributeError, ValueError):
        return False


def paint(text: str, colour: str, fp: TextIO) -> str:
    return f"{colour}{text}{RESET}" if is_tty(fp) else text


def _emit(colour: str, message: str, fp: TextIO) -> None:
    print(paint(message, colour, fp), file=fp, flush=True)


def info(message: str) -> None:
    _emit("", message, sys.stderr)


def step(message: str) -> None:
    _emit(CYAN, message, sys.stderr)


def ok(message: str) -> None:
    _emit(GREEN, message, sys.stderr)


def note(message: str) -> None:
    _emit(YELLOW, message, sys.stderr)


def warn(message: str) -> None:
    _emit(ORANGE, f"slimserve: {message}", sys.stderr)


def fail(message: str) -> None:
    _emit(RED, f"slimserve: {message}", sys.stderr)


def die(message: str, code: int = 2) -> None:
    fail(message)
    raise SystemExit(code)


def progress(message: str, done: bool = False) -> None:
    """One line, rewritten in place on a TTY and appended otherwise."""
    if is_tty(sys.stderr):
        end = "\n" if done else ""
        print(f"\r{CYAN}{message}{RESET}\x1b[K", end=end, file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def human_bytes(n: int) -> str:
    for limit, unit in ((1 << 40, "TiB"), (1 << 30, "GiB"), (1 << 20, "MiB")):
        if n >= limit:
            scaled = n / limit
            return f"{scaled:.0f} {unit}" if scaled >= 100 else f"{scaled:.1f} {unit}"
    return f"{n / 1024:.1f} KiB"
