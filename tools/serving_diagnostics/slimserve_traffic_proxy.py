#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming observability proxy for an OpenAI-compatible SlimServe server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from logging.handlers import RotatingFileHandler

import anyio
import httpx
from capture_paths import DEFAULT_LOG_DIR
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

UPSTREAM = os.environ.get("PROXY_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
LOG_DIR = DEFAULT_LOG_DIR
CAPTURE_LIMIT = int(os.environ.get("PROXY_CAPTURE_LIMIT_BYTES", str(32 * 1024 * 1024)))
LOG_MAX_BYTES = int(os.environ.get("PROXY_LOG_MAX_BYTES", str(256 * 1024 * 1024)))
LOG_BACKUPS = int(os.environ.get("PROXY_LOG_BACKUPS", "8"))
CLOSE_TIMEOUT_SECONDS = 10.0
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


LOGGER = _logger()


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


def _downstream_headers(headers) -> list[tuple[bytes, bytes]]:
    return [
        (key, value)
        for key, value in headers.raw
        if key.decode("ascii").lower() not in HOP_BY_HOP
    ]


CLIENT: httpx.AsyncClient | None = None


async def startup() -> None:
    global CLIENT
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


@contextmanager
def _collapse_excgroups():
    # Keep the original stream/send error instead of an AnyIO task-group wrapper.
    try:
        yield
    except BaseException as exc:
        while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        raise exc


class CapturedStreamingResponse(StreamingResponse):
    """Observe both ASGI directions; finalize even when sending headers fails."""

    def __init__(self, *args, lifecycle, finish, started_monotonic, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifecycle = lifecycle
        self.finish = finish
        self.started_monotonic = started_monotonic

    async def __call__(self, scope, receive, send):
        state = self.lifecycle
        primary_error = None

        async def observed_send(message):
            try:
                await send(message)
            except anyio.get_cancelled_exc_class():
                raise
            except Exception as exc:
                state["downstream_send_error"] = repr(exc)[:2048]
                state["downstream_send_error_ms"] = round(
                    (time.monotonic() - self.started_monotonic) * 1000, 3
                )
                raise
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    state["downstream_chunks_sent"] += 1
                    state["downstream_bytes_sent"] += len(body)
                if not message.get("more_body", False):
                    state["downstream_complete"] = True

        try:
            # Observe disconnect on ASGI 2.4+ even while upstream stalls.
            with _collapse_excgroups():
                async with anyio.create_task_group() as group:

                    async def disconnect():
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                state["downstream_disconnect"] = True
                                state["downstream_disconnect_ms"] = round(
                                    (time.monotonic() - self.started_monotonic) * 1000,
                                    3,
                                )
                                group.cancel_scope.cancel()
                                return

                    group.start_soon(disconnect)
                    await self.stream_response(observed_send)
                    group.cancel_scope.cancel()
        except anyio.get_cancelled_exc_class() as exc:
            primary_error = exc
            state["cancelled"] = True
            raise
        except BaseException as exc:
            primary_error = exc
            state["asgi_error"] = repr(exc)[:2048]
            raise
        finally:
            # AnyIO shielding handles level cancellation; a separate asyncio
            # task survives repeated uvicorn Task.cancel() during cleanup.
            with anyio.CancelScope(shield=True):
                cleanup = asyncio.create_task(self.finish())
                cancelled = None
                while True:
                    try:
                        await asyncio.shield(cleanup)
                        break
                    except asyncio.CancelledError as exc:
                        state["cancelled"] = True
                        cancelled = exc
                        if cleanup.done():
                            break
                if cancelled is not None and primary_error is None:
                    raise cancelled


async def proxy(request: Request) -> Response:
    assert CLIENT is not None
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    started = time.time()
    started_monotonic = time.monotonic()
    capture = request.url.path in CAPTURE_PATHS
    try:
        request_body = await request.body()
    except ClientDisconnect:
        LOGGER.info(
            json.dumps(
                {
                    "event": "request_abandoned",
                    "timestamp": started,
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": 499,
                    "elapsed_ms": round(
                        (time.monotonic() - started_monotonic) * 1000, 3
                    ),
                    "termination_reason": "client_abandoned_input",
                    "downstream_disconnect": True,
                    "request_body_complete": False,
                },
                separators=(",", ":"),
            )
        )
        return Response(status_code=499)
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
    except anyio.get_cancelled_exc_class():
        LOGGER.info(
            json.dumps(
                {
                    "timestamp": started,
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": None,
                    "elapsed_ms": round(
                        (time.monotonic() - started_monotonic) * 1000, 3
                    ),
                    "termination_reason": "cancelled",
                    "upstream_state": "awaiting_headers",
                    "cancelled": True,
                },
                separators=(",", ":"),
            )
        )
        raise
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - started_monotonic) * 1000, 3)
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
    finalized = False
    lifecycle = {
        "upstream_state": "not_started",
        "upstream_headers_ms": round((time.monotonic() - started_monotonic) * 1000, 3),
        "upstream_chunks_received": 0,
        "upstream_bytes_received": 0,
        "first_chunk_ms": None,
        "last_chunk_ms": None,
        "downstream_chunks_sent": 0,
        "downstream_bytes_sent": 0,
        "downstream_complete": False,
        "downstream_disconnect": False,
        "downstream_disconnect_ms": None,
        "downstream_send_error_ms": None,
        "cancelled": False,
    }

    async def body_stream() -> AsyncIterator[bytes]:
        nonlocal truncated
        lifecycle["upstream_state"] = "streaming"
        try:
            async for chunk in upstream_response.aiter_raw():
                chunk_ms = round((time.monotonic() - started_monotonic) * 1000, 3)
                if lifecycle["first_chunk_ms"] is None:
                    lifecycle["first_chunk_ms"] = chunk_ms
                lifecycle["last_chunk_ms"] = chunk_ms
                lifecycle["upstream_chunks_received"] += 1
                lifecycle["upstream_bytes_received"] += len(chunk)
                if capture:
                    room = max(0, CAPTURE_LIMIT - len(captured))
                    captured.extend(chunk[:room])
                    truncated = truncated or len(chunk) > room
                yield chunk
            lifecycle["upstream_state"] = "eof"
        except anyio.get_cancelled_exc_class():
            lifecycle["upstream_state"] = "cancelled"
            raise
        except Exception as exc:
            lifecycle["upstream_state"] = "error"
            lifecycle["upstream_error"] = repr(exc)[:2048]
            raise

    body_iterator = body_stream()

    async def finish() -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True
        try:
            # HTTPX already closes on EOF. Interrupted streams must release
            # their connection too, with a bounded cleanup budget.
            with anyio.move_on_after(CLOSE_TIMEOUT_SECONDS, shield=True) as close_scope:
                try:
                    if not upstream_response.is_closed:
                        await upstream_response.aclose()
                finally:
                    await body_iterator.aclose()
            if close_scope.cancel_called:
                lifecycle["upstream_close_error"] = "close_timeout"
        except Exception as exc:
            lifecycle["upstream_close_error"] = repr(exc)[:2048]
        finally:
            if lifecycle.get("downstream_send_error"):
                reason = "downstream_send_error"
            elif lifecycle["downstream_disconnect"]:
                reason = "downstream_disconnect"
            elif lifecycle.get("upstream_error"):
                reason = "upstream_error"
            elif lifecycle["cancelled"]:
                reason = "cancelled"
            elif lifecycle.get("asgi_error"):
                reason = "asgi_error"
            elif lifecycle["downstream_complete"]:
                reason = "complete"
            else:
                reason = "interrupted"
            event = {
                "timestamp": started,
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "status": upstream_response.status_code,
                "elapsed_ms": round((time.monotonic() - started_monotonic) * 1000, 3),
                "request_headers": _safe_headers(request.headers),
                "response_headers": _safe_headers(upstream_response.headers),
                "termination_reason": reason,
                **lifecycle,
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

    response = CapturedStreamingResponse(
        body_iterator,
        status_code=upstream_response.status_code,
        lifecycle=lifecycle,
        finish=finish,
        started_monotonic=started_monotonic,
    )
    response.raw_headers = _downstream_headers(upstream_response.headers)
    return response


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
