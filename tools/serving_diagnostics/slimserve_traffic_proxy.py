#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming observability proxy for an OpenAI-compatible SlimServe server."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

import anyio
import httpx
from capture_paths import DEFAULT_LOG_DIR
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("PROXY_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
LOG_DIR = DEFAULT_LOG_DIR
CAPTURE_LIMIT = int(os.environ.get("PROXY_CAPTURE_LIMIT_BYTES", str(32 * 1024 * 1024)))
LOG_MAX_BYTES = int(os.environ.get("PROXY_LOG_MAX_BYTES", str(256 * 1024 * 1024)))
LOG_BACKUPS = int(os.environ.get("PROXY_LOG_BACKUPS", "8"))
CAPTURE_PATHS = frozenset(("/v1/responses", "/v1/chat/completions"))

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "host",
    }
)
SECRET_HEADER_RE = re.compile(r"authorization|api[-_]key|token|cookie", re.I)
SECRET_FIELD_RE = re.compile(
    r"^(authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|bearer|cookie)$",
    re.I,
)


def _logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("slimserve-traffic")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_DIR / "traffic.jsonl",
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = logging.getLogger("slimserve-traffic")


def _redact(value):
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if SECRET_FIELD_RE.match(str(key)) else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _body_for_log(data: bytes, content_type: str) -> object:
    text = data.decode("utf-8", errors="replace")
    if "json" in content_type.lower():
        try:
            return _redact(json.loads(text))
        except json.JSONDecodeError:
            return {"malformed_json": True, "raw": text}
    # SSE contains JSON payloads but preserving the exact wire text is useful when
    # diagnosing malformed incremental tool arguments.
    return text


def _safe_headers(headers) -> dict[str, str]:
    return {
        key: ("[REDACTED]" if SECRET_HEADER_RE.search(key) else value)
        for key, value in headers.items()
        if key.lower() not in {"content-length", "host"}
    }


def _upstream_headers(headers) -> dict[str, str]:
    result = {
        key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP
    }
    # Keep captures readable and preserve exact SSE text instead of forwarding a
    # compressed upstream representation.
    result["accept-encoding"] = "identity"
    return result


def _downstream_headers(headers) -> dict[str, str]:
    return {
        key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP
    }


CLIENT: httpx.AsyncClient | None = None


async def startup() -> None:
    global CLIENT
    _logger()
    CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        limits=httpx.Limits(
            max_connections=512, max_keepalive_connections=128, keepalive_expiry=30.0
        ),
    )


async def shutdown() -> None:
    if CLIENT is not None:
        await CLIENT.aclose()


@asynccontextmanager
async def lifespan(_app):
    await startup()
    try:
        yield
    finally:
        await shutdown()


async def proxy(request: Request) -> Response:
    assert CLIENT is not None
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    started = time.time()
    request_body = await request.body()
    capture = request.url.path in CAPTURE_PATHS
    url = UPSTREAM + request.url.path
    if request.url.query:
        url += "?" + request.url.query

    if capture:
        LOGGER.info(
            json.dumps(
                {
                    "event": "request_start",
                    "timestamp": started,
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query,
                    "request_headers": _safe_headers(request.headers),
                    "request_body": _body_for_log(
                        request_body, request.headers.get("content-type", "")
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    upstream_request = CLIENT.build_request(
        request.method,
        url,
        headers=_upstream_headers(request.headers),
        content=request_body,
    )
    try:
        upstream_response = await CLIENT.send(upstream_request, stream=True)
    except Exception as exc:
        elapsed_ms = round((time.time() - started) * 1000, 3)
        LOGGER.info(
            json.dumps(
                {
                    "timestamp": started,
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": 502,
                    "elapsed_ms": elapsed_ms,
                    "proxy_error": repr(exc),
                    "request_body": _body_for_log(
                        request_body, request.headers.get("content-type", "")
                    )
                    if capture
                    else None,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return Response("upstream unavailable", status_code=502)

    captured = bytearray()
    truncated = False

    async def body_stream() -> AsyncIterator[bytes]:
        nonlocal truncated
        try:
            async for chunk in upstream_response.aiter_raw():
                if capture and len(captured) < CAPTURE_LIMIT:
                    room = CAPTURE_LIMIT - len(captured)
                    captured.extend(chunk[:room])
                    truncated = truncated or len(chunk) > room
                elif capture:
                    truncated = True
                yield chunk
        finally:
            # Starlette background tasks do not run after every stream error.
            # Shield cleanup from downstream cancellation as well.
            with anyio.CancelScope(shield=True):
                await finish()

    finished = False

    async def finish() -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        await upstream_response.aclose()
        elapsed_ms = round((time.time() - started) * 1000, 3)
        event = {
            "timestamp": started,
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query,
            "status": upstream_response.status_code,
            "elapsed_ms": elapsed_ms,
            "request_headers": _safe_headers(request.headers),
            "response_headers": _safe_headers(upstream_response.headers),
        }
        if capture:
            event["request_body"] = _body_for_log(
                request_body, request.headers.get("content-type", "")
            )
            event["response_body"] = _body_for_log(
                bytes(captured), upstream_response.headers.get("content-type", "")
            )
            event["response_capture_truncated"] = truncated
        LOGGER.info(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

    return StreamingResponse(
        body_stream(),
        status_code=upstream_response.status_code,
        headers=_downstream_headers(upstream_response.headers),
        background=BackgroundTask(finish),
    )


app = Starlette(
    routes=[
        Route(
            "/{path:path}",
            proxy,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        )
    ],
    lifespan=lifespan,
)
