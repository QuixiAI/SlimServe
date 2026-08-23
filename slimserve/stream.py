# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-by-token output from the OpenAI-compatible endpoint."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import regex as re

# K3 renders its native XTML frame into the completion text, so a finished
# answer trails control tokens: "Paris. <|close|> response <|sep|> ...".
# The answer is everything before the first one.
_XTML = re.compile(r"<\|(?:close|sep|end_of_msg|message|response|think)\|>")
_LONGEST_TOKEN = len("<|end_of_msg|>")


def visible_text(raw: str) -> str:
    """The part of a completion a person asked for, without the chat frame."""
    match = _XTML.search(raw)
    return (raw[: match.start()] if match else raw).strip()


class FrameFilter:
    """Strips the chat frame from a stream without ever printing half a token.

    A control token can arrive split across two deltas, so anything that could
    still become one is held back until the next delta proves otherwise.
    """

    def __init__(self) -> None:
        self._held = ""
        self.stopped = False

    def feed(self, delta: str) -> str:
        if self.stopped:
            return ""
        buffer = self._held + delta
        match = _XTML.search(buffer)
        if match:
            self.stopped = True
            self._held = ""
            return buffer[: match.start()]
        # Hold back a trailing fragment that could still open a control token.
        keep = 0
        for size in range(min(len(buffer), _LONGEST_TOKEN - 1), 0, -1):
            tail = buffer[-size:]
            if tail.startswith("<") and "<|".startswith(tail[:2]):
                keep = size
                break
        self._held = buffer[len(buffer) - keep :] if keep else ""
        return buffer[: len(buffer) - keep] if keep else buffer

    def flush(self) -> str:
        if self.stopped:
            return ""
        remainder, self._held = self._held, ""
        return remainder


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float | None = None,
    seed: int | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> Iterator[str]:
    """Yield content deltas from a streaming chat completion.

    Sampling parameters are omitted from the request unless given, so the
    server's (i.e. the model's shipped) defaults apply. Greedy decoding is
    never used in this stack; pass `seed` when a run must be repeatable.
    """
    import requests

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs

    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        stream=True,
        timeout=timeout,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(f"{response.status_code}: {response.text[:400]}")
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                return
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                # Reasoning models split the reply across fields, and which
                # name carries it depends on the parser. Missing one of these
                # shows an empty answer for a model that in fact replied.
                for key in ("reasoning_content", "reasoning", "content"):
                    piece = delta.get(key)
                    if piece:
                        yield piece
