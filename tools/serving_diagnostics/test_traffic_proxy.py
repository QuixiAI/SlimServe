# SPDX-License-Identifier: Apache-2.0
import json
import unittest
from unittest.mock import Mock, patch

import httpx
import slimserve_traffic_proxy as proxy
from starlette.requests import Request


class Stream(httpx.AsyncByteStream):
    def __init__(self, fail=False):
        self.closed = False
        self.fail = fail

    async def __aiter__(self):
        yield b"data: first\n\n"
        if self.fail:
            raise httpx.ReadError("interrupted")
        yield b"data: last\n\n"

    async def aclose(self):
        self.closed = True


def request():
    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "query_string": b"",
            "scheme": "http",
            "server": ("localhost", 8000),
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, stream):
        logger = Mock()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=stream
                )
            )
        ) as client:
            with (
                patch.object(proxy, "CLIENT", client),
                patch.object(proxy, "LOGGER", logger),
                patch.object(proxy, "CAPTURE_LIMIT", 4),
            ):
                response = await proxy.proxy(request())
                chunks = []
                try:
                    async for chunk in response.body_iterator:
                        chunks.append(chunk)
                except httpx.ReadError:
                    self.assertTrue(stream.fail)
                await response.background()
        records = [json.loads(c.args[0]) for c in logger.info.call_args_list]
        self.assertEqual(len(records), 2)
        self.assertTrue(stream.closed)
        self.assertEqual(records[-1]["response_body"], "data")
        self.assertTrue(records[-1]["response_capture_truncated"])
        return b"".join(chunks)

    async def test_capture_limit_does_not_truncate_forwarded_stream(self):
        self.assertEqual(
            await self.exercise(Stream()), b"data: first\n\ndata: last\n\n"
        )

    async def test_upstream_error_closes_connection_and_records_partial_capture(self):
        self.assertEqual(await self.exercise(Stream(fail=True)), b"data: first\n\n")

    def test_redacts_selected_secrets_without_rewriting_payload_text(self):
        self.assertEqual(
            proxy._safe_headers({"Authorization": "secret"}),
            {"Authorization": "[REDACTED]"},
        )
        self.assertEqual(
            proxy._body_for_log(
                b'{"api_key":"secret","text":"hello"}', "application/json"
            ),
            {"api_key": "[REDACTED]", "text": "hello"},
        )
