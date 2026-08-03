# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The interactive prompt.

Modelled on ds4's REPL: the model's voice goes to stdout, the tool's voice to
stderr, Ctrl-C stops generation without leaving the prompt, and every turn ends
with one line of rates. Both models served here are vision models, so a bare
image path or URL on the input line attaches that image to the turn.
"""

from __future__ import annotations

import contextlib
import mimetypes
import os
import readline  # noqa: F401  -- importing it is what gives input() line editing
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pybase64 as base64
import regex as re

from slimserve import term
from slimserve.registry import Plan
from slimserve.stream import FrameFilter, chat_completion

HISTORY = Path.home() / ".slimserve_history"
HISTORY_LIMIT = 512

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)

COMMANDS = """Commands:
  /help          Show this help.
  /image PATH    Attach an image (or URL) to the next message.
  /clear         Forget the conversation so far.
  /system TEXT   Set the system prompt. Bare /system shows it.
  /tokens N      Cap generated tokens per reply. Bare /tokens shows it.
  /temp F        Set sampling temperature. Bare /temp shows it.
  /stats         Show the last turn's rates.
  /quit, /exit   Leave the prompt.
  Ctrl+C         Stop generation and return to the prompt.
  Ctrl+D         Leave the prompt.
A bare image path or URL on its own line is attached to your next message."""


class _Interrupt:
    """Ctrl-C ends the current generation, never the process."""

    def __init__(self) -> None:
        self.raised = False
        self._previous = None

    def __enter__(self) -> _Interrupt:
        self.raised = False
        self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def _handle(self, *_: object) -> None:
        self.raised = True


@dataclass
class Session:
    plan: Plan
    base_url: str
    model: str
    system: str = ""
    max_tokens: int = 2048
    temperature: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_images: list[str] = field(default_factory=list)
    last_stats: str = ""

    def payload(self, text: str) -> list[dict[str, Any]]:
        """The full message list for this turn, images folded into the user turn."""
        content: list[dict[str, Any]] = []
        for reference in self.pending_images:
            content.append(_image_part(reference))
        content.append({"type": "text", "text": text})
        turn = {"role": "user", "content": content}
        history = list(self.messages)
        if self.system:
            history.insert(0, {"role": "system", "content": self.system})
        return history + [turn]


def _image_part(reference: str) -> dict[str, Any]:
    """An OpenAI content part. Local files become data: URIs."""
    if _URL.fullmatch(reference):
        return {"type": "image_url", "image_url": {"url": reference}}
    path = Path(reference).expanduser()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def _looks_like_image(line: str) -> bool:
    if _URL.fullmatch(line):
        return line.lower().endswith(_IMAGE_SUFFIXES)
    if any(line.lower().endswith(suffix) for suffix in _IMAGE_SUFFIXES):
        return Path(line).expanduser().is_file()
    return False


def _load_history() -> None:
    readline.set_history_length(HISTORY_LIMIT)
    if HISTORY.is_file():
        with contextlib.suppress(OSError):
            readline.read_history_file(str(HISTORY))


def _save_history() -> None:
    with contextlib.suppress(OSError):
        readline.write_history_file(str(HISTORY))


def run(
    plan: Plan, base_url: str, prompt: str | None = None, max_tokens: int = 2048
) -> int:
    """One-shot when `prompt` is given, otherwise the interactive loop."""
    session = Session(
        plan=plan,
        base_url=base_url,
        model=plan.engine.get("served_model_name", "model"),
        max_tokens=max_tokens,
    )
    if prompt is not None:
        _turn(session, prompt)
        return 0

    _load_history()
    print(COMMANDS)
    while True:
        try:
            line = input("slimserve> ")
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            break

        line = line.strip()
        if not line:
            continue
        _save_history()

        if line.startswith("/"):
            if _command(session, line) is False:
                break
            continue
        if _looks_like_image(line):
            session.pending_images.append(line)
            term.ok(f"attached {line}")
            continue
        _turn(session, line)
    return 0


def _arg(line: str, name: str) -> str | None:
    """The argument after `/name`, or None when the command was bare.

    Matching requires a word boundary so `/tokensfoo` is not `/tokens foo`.
    """
    if line == name:
        return ""
    if line.startswith(name) and line[len(name)].isspace():
        return line[len(name) :].strip()
    return None


def _command(session: Session, line: str) -> bool | None:
    """Returns False to leave the prompt."""
    if line in ("/quit", "/exit"):
        return False
    if line == "/help":
        print(COMMANDS)
        return None
    if line == "/stats":
        print(session.last_stats or "no turns yet")
        return None
    if line == "/clear":
        session.messages.clear()
        session.pending_images.clear()
        term.ok("conversation cleared")
        return None

    for name, apply in (
        ("/image", _set_image),
        ("/system", _set_system),
        ("/tokens", _set_tokens),
        ("/temp", _set_temp),
    ):
        argument = _arg(line, name)
        if argument is not None:
            apply(session, argument)
            return None

    term.fail(f"unknown command: {line}")
    term.fail("type /help for commands")
    return None


def _set_image(session: Session, argument: str) -> None:
    if not argument:
        if session.pending_images:
            print("\n".join(session.pending_images))
        else:
            print("no images attached")
        return
    path = Path(argument).expanduser()
    if not _URL.fullmatch(argument) and not path.is_file():
        term.fail(f"no such file: {argument}")
        return
    session.pending_images.append(argument if _URL.fullmatch(argument) else str(path))
    term.ok(f"attached {argument}")


def _set_system(session: Session, argument: str) -> None:
    if not argument:
        print(session.system or "(no system prompt)")
        return
    session.system = argument
    session.messages.clear()
    term.ok("system prompt set; conversation cleared")


def _set_tokens(session: Session, argument: str) -> None:
    if not argument:
        print(session.max_tokens)
        return
    try:
        value = int(argument)
    except ValueError:
        term.fail(f"/tokens wants an integer, got {argument!r}")
        return
    if value < 1:
        term.fail("/tokens must be at least 1")
        return
    session.max_tokens = value
    print(value)


def _set_temp(session: Session, argument: str) -> None:
    if not argument:
        print(session.temperature)
        return
    try:
        value = float(argument)
    except ValueError:
        term.fail(f"/temp wants a number, got {argument!r}")
        return
    if not 0.0 <= value <= 2.0:
        term.fail("/temp must be between 0 and 2")
        return
    session.temperature = value
    print(value)


def _turn(session: Session, text: str) -> None:
    """Stream one reply, printing each delta as it lands."""
    messages = session.payload(text)
    frame = FrameFilter()
    pieces: list[str] = []
    first_token_at: float | None = None
    started = time.monotonic()

    with _Interrupt() as interrupt:
        try:
            for delta in chat_completion(
                session.base_url,
                session.model,
                messages,
                max_tokens=session.max_tokens,
                temperature=session.temperature,
                chat_template_kwargs=session.plan.chat_template_kwargs or None,
            ):
                if interrupt.raised:
                    break
                if first_token_at is None:
                    first_token_at = time.monotonic()
                visible = frame.feed(delta)
                if visible:
                    pieces.append(visible)
                    print(visible, end="", flush=True)
        except KeyboardInterrupt:
            interrupt.raised = True
        except Exception as error:  # a bad turn must not end the session
            print()
            term.fail(str(error))
            session.pending_images.clear()
            return

    tail = frame.flush()
    if tail:
        pieces.append(tail)
        print(tail, end="", flush=True)
    print(flush=True)

    answer = "".join(pieces).strip()
    elapsed = time.monotonic() - started
    if interrupt.raised:
        term.note("interrupted")
    if not answer:
        session.pending_images.clear()
        return

    # History carries the cleaned answer: the frame is re-applied by the chat
    # template on the next turn, and keeping it would nest one inside another.
    session.messages.append({"role": "user", "content": _plain(session, text)})
    session.messages.append({"role": "assistant", "content": answer})
    session.pending_images.clear()

    generated = len(answer.split())
    ttft = (first_token_at - started) if first_token_at else elapsed
    session.last_stats = (
        f"slimserve: {ttft:.1f}s to first token, "
        f"{generated / elapsed if elapsed else 0.0:.1f} words/s, {elapsed:.1f}s"
    )
    term.step(session.last_stats)


def _plain(session: Session, text: str) -> Any:
    """Keep images in history so follow-up questions can refer back to them."""
    if not session.pending_images:
        return text
    content: list[dict[str, Any]] = [
        _image_part(reference) for reference in session.pending_images
    ]
    content.append({"type": "text", "text": text})
    return content


def banner(plan: Plan) -> None:
    term.info("")
    term.ok(f"{plan.title} — {plan.quant.title} on {plan.gpus} GPUs")
    if os.environ.get("SLIMSERVE_CACHE"):
        term.info(f"models: {plan.model_dir}")
