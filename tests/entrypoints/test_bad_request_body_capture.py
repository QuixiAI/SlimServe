# SPDX-License-Identifier: Apache-2.0
"""Opt-in 4xx request/response body diagnostics."""

import asyncio
import json
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
