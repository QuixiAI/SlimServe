# SPDX-License-Identifier: Apache-2.0
"""Opt-in 4xx request/response body diagnostics."""

import asyncio
import json
import os
import stat

from vllm.entrypoints.serve.utils.server_utils import (
    BadRequestBodyCaptureMiddleware,
)


def _scope(*, authorization: str = "Bearer never-log-this"):
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "root_path": "",
        "client": ("127.0.0.1", 12345),
        "headers": [
            (b"authorization", authorization.encode()),
            (b"content-type", b"application/json"),
            (b"x-request-id", b"req-diagnostic"),
        ],
    }


def _receive_chunks(*chunks: bytes):
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    return receive


async def _sink(_message):
    return None


def test_captures_only_4xx_and_never_headers(tmp_path):
    async def bad_request(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"x-request-id", b"response-id")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"error":"Extra data"}',
            }
        )

    async def run():
        log_path = tmp_path / "bad-requests.jsonl"
        middleware = BadRequestBodyCaptureMiddleware(
            bad_request,
            path=str(log_path),
            max_body_bytes=1024,
        )
        await middleware(
            _scope(),
            _receive_chunks(b'{"input":', b'"hello"}'),
            _sink,
        )

        record = json.loads(log_path.read_text())
        assert record["status"] == 400
        assert record["request_id"] == "req-diagnostic"
        assert record["request_body"] == '{"input":"hello"}'
        assert record["response_body"] == '{"error":"Extra data"}'
        assert "never-log-this" not in log_path.read_text()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600

    asyncio.run(run())


def test_truncates_bodies_but_hashes_complete_stream(tmp_path):
    async def bad_request(_scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 422, "headers": []})
        await send({"type": "http.response.body", "body": b"response-too-long"})

    async def run():
        log_path = tmp_path / "bad-requests.jsonl"
        middleware = BadRequestBodyCaptureMiddleware(
            bad_request,
            path=str(log_path),
            max_body_bytes=8,
        )
        await middleware(
            _scope(),
            _receive_chunks(b"request-too-long"),
            _sink,
        )

        record = json.loads(log_path.read_text())
        assert record["request_body"] == "request-"
        assert record["request_body_bytes"] == len(b"request-too-long")
        assert record["request_body_truncated"] is True
        assert record["response_body"] == "response"
        assert record["response_body_bytes"] == len(b"response-too-long")
        assert record["response_body_truncated"] is True
        assert len(record["request_sha256"]) == 64
        assert len(record["response_sha256"]) == 64

    asyncio.run(run())


def test_success_is_not_recorded(tmp_path):
    async def ok(_scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def run():
        log_path = tmp_path / "bad-requests.jsonl"
        middleware = BadRequestBodyCaptureMiddleware(
            ok,
            path=str(log_path),
            max_body_bytes=1024,
        )
        await middleware(_scope(), _receive_chunks(b"healthy"), _sink)
        assert not log_path.exists()

    asyncio.run(run())


def test_unauthenticated_requests_are_not_recorded(tmp_path):
    """The capture wraps AuthenticationMiddleware in build_app; a client
    without a valid key must not be able to make the server write to disk."""
    from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware

    async def bad_request(_scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"bad"})

    async def run():
        log_path = tmp_path / "bad-requests.jsonl"
        # Same nesting as build_app: capture outside authentication.
        middleware = BadRequestBodyCaptureMiddleware(
            AuthenticationMiddleware(bad_request, tokens=["right-key"]),
            path=str(log_path),
            max_body_bytes=1024,
        )
        scope = _scope(authorization="Bearer wrong-key")
        scope["path"] = "/v1/responses"
        await middleware(scope, _receive_chunks(b"{}"), _sink)
        assert not log_path.exists()

        await middleware(
            _scope(authorization="Bearer right-key"),
            _receive_chunks(b"{}"),
            _sink,
        )
        assert json.loads(log_path.read_text())["status"] == 400

    asyncio.run(run())


def test_forbidden_status_is_not_recorded(tmp_path):
    async def forbidden(_scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 403, "headers": []})
        await send({"type": "http.response.body", "body": b"no"})

    async def run():
        log_path = tmp_path / "bad-requests.jsonl"
        middleware = BadRequestBodyCaptureMiddleware(
            forbidden, path=str(log_path), max_body_bytes=1024
        )
        await middleware(_scope(), _receive_chunks(b"{}"), _sink)
        assert not log_path.exists()

    asyncio.run(run())


def test_file_is_capped_and_rotated_once(tmp_path):
    async def bad_request(_scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"x" * 100})

    async def run():
        log_path = tmp_path / "bad-requests.jsonl"
        rotated = tmp_path / "bad-requests.jsonl.1"
        middleware = BadRequestBodyCaptureMiddleware(
            bad_request,
            path=str(log_path),
            max_body_bytes=1024,
            max_file_bytes=2048,
        )
        for _ in range(40):
            await middleware(_scope(), _receive_chunks(b"y" * 100), _sink)
            assert log_path.stat().st_size <= 2048
            if rotated.exists():
                assert rotated.stat().st_size <= 2048
        assert rotated.exists()
        assert stat.S_IMODE(rotated.stat().st_mode) == 0o600
        assert not (tmp_path / "bad-requests.jsonl.2").exists()
        # Every surviving line is a complete record.
        for path in (log_path, rotated):
            for line in path.read_text().splitlines():
                assert json.loads(line)["status"] == 400

    asyncio.run(run())


async def _not_found(_scope, receive, send):
    await receive()
    await send({"type": "http.response.start", "status": 404, "headers": []})
    await send({"type": "http.response.body", "body": b"not found"})


def test_unauthenticated_non_api_path_is_not_recorded(tmp_path):
    """AuthenticationMiddleware exempts paths outside GUARDED_PREFIX, so a
    keyless ``POST /not-v1`` reaches the app and 404s. It must not be
    written to disk."""
    from vllm.entrypoints.serve.utils.server_utils import AuthenticationMiddleware

    async def run():
        log_path = tmp_path / "bad-requests.jsonl"
        middleware = BadRequestBodyCaptureMiddleware(
            AuthenticationMiddleware(_not_found, tokens=["right-key"]),
            path=str(log_path),
            max_body_bytes=1024,
            tokens=["right-key"],
        )
        scope = _scope(authorization="")
        scope["headers"] = [h for h in scope["headers"] if h[0] != b"authorization"]
        scope["path"] = "/not-v1"
        await middleware(scope, _receive_chunks(b"{}"), _sink)

        options = _scope(authorization="Bearer wrong-key")
        options["method"] = "OPTIONS"
        await middleware(options, _receive_chunks(b"{}"), _sink)
        assert not log_path.exists()

        # Even a valid key does not make non-API paths eligible.
        scope = _scope(authorization="Bearer right-key")
        scope["path"] = "/not-v1"
        await middleware(scope, _receive_chunks(b"{}"), _sink)
        assert not log_path.exists()

        # A keyed API-path 404 is still captured.
        scope = _scope(authorization="Bearer right-key")
        scope["path"] = "/v1/nope"
        await middleware(scope, _receive_chunks(b"{}"), _sink)
        assert json.loads(log_path.read_text())["status"] == 404

    asyncio.run(run())


def test_concurrent_rotation_keeps_previous_window(tmp_path, monkeypatch):
    """Two API-server processes (two middleware instances) that both see a
    full file must not both rotate: the second rotation would replace
    ``.1`` with the first one's fresh file and destroy the previous window."""
    import threading

    from vllm.entrypoints.serve.utils import server_utils

    log_path = tmp_path / "bad-requests.jsonl"
    rotated = tmp_path / "bad-requests.jsonl.1"
    first = BadRequestBodyCaptureMiddleware(
        _not_found, path=str(log_path), max_body_bytes=64, max_file_bytes=300
    )
    second = BadRequestBodyCaptureMiddleware(
        _not_found, path=str(log_path), max_body_bytes=64, max_file_bytes=300
    )
    old_window = [{"window": "old", "n": n, "pad": "p" * 40} for n in range(4)]
    for record in old_window:
        first._write_record(record)
    assert not rotated.exists()

    real_replace = os.replace
    other = {}

    def interleaving_replace(src, dst):
        # The first process has decided to rotate; the second process writes
        # at this very moment. Give it a chance to run before continuing.
        if "thread" not in other:
            other["thread"] = threading.Thread(
                target=second._write_record,
                args=({"window": "second", "pad": "q" * 40},),
            )
            other["thread"].start()
            other["thread"].join(timeout=0.5)
        return real_replace(src, dst)

    monkeypatch.setattr(server_utils.os, "replace", interleaving_replace)
    first._write_record({"window": "first", "pad": "r" * 40})
    other["thread"].join(timeout=5)
    assert not other["thread"].is_alive()
    monkeypatch.setattr(server_utils.os, "replace", real_replace)

    def windows(path):
        return [json.loads(line)["window"] for line in path.read_text().splitlines()]

    rotated_windows = windows(rotated)
    live_windows = windows(log_path)
    assert rotated_windows == ["old"] * 4
    assert sorted(live_windows) == ["first", "second"]


def test_write_failure_logs_once_and_disables_capture(tmp_path, monkeypatch):
    from unittest import mock

    from vllm.entrypoints.serve.utils import server_utils

    fake_logger = mock.Mock()
    monkeypatch.setattr(server_utils, "logger", fake_logger)

    async def bad_request(_scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"bad"})

    async def run():
        # Parent directory does not exist: every write would fail.
        middleware = BadRequestBodyCaptureMiddleware(
            bad_request,
            path=str(tmp_path / "missing" / "bad-requests.jsonl"),
            max_body_bytes=1024,
        )
        writes = []
        real_write = middleware._write_record

        def counting_write(record):
            writes.append(record)
            return real_write(record)

        middleware._write_record = counting_write
        sent = []

        async def collect(message):
            sent.append(message)

        for _ in range(5):
            await middleware(_scope(), _receive_chunks(b"{}"), collect)
        # Requests keep being served normally.
        assert sum(m["type"] == "http.response.start" for m in sent) == 5
        assert len(writes) == 1
        assert middleware.disabled
        assert fake_logger.error.call_count == 1
        assert fake_logger.exception.call_count == 0

    asyncio.run(run())
