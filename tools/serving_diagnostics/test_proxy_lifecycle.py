# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only ASGI fixtures; no live service or model output is used."""

import asyncio
import importlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import anyio
import httpx
from starlette.requests import Request

# Importing the proxy creates its rotating logger; isolate even that side effect.
with TemporaryDirectory() as log_dir:
    with patch.dict(os.environ, {"PROXY_LOG_DIR": log_dir}):
        import capture_paths

        with patch.object(capture_paths, "DEFAULT_LOG_DIR", Path(log_dir)):
            proxy = importlib.import_module("slimserve_traffic_proxy")
    for handler in list(proxy.LOGGER.handlers):
        handler.close()
        proxy.LOGGER.removeHandler(handler)


class FixtureStream(httpx.AsyncByteStream):
    def __init__(self, chunks=(b"alpha", b"beta"), error=None, stall=False):
        self.chunks = chunks
        self.error = error
        self.stall = stall
        self.waiting = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = None
        self.close_error = None
        self.close_calls = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error
        if self.stall:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.close_calls += 1
        self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = Mock()
        self.log_patch = patch.object(proxy, "LOGGER", self.logger)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.queue = asyncio.Queue()
        self.queue.put_nowait(
            {"type": "http.request", "body": b"{}", "more_body": False}
        )
        self.sent = []
        self.scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "POST",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "server": ("fixture", 80),
            "client": ("fixture", 1234),
        }

    async def response(self, stream):
        response = httpx.Response(
            207,
            headers=[
                ("content-type", "text/event-stream"),
                ("x-fixture", "yes"),
                ("set-cookie", "a=1"),
                ("set-cookie", "b=2"),
            ],
            stream=stream,
        )
        client = Mock()
        client.build_request.return_value = httpx.Request(
            "POST", "http://fixture/v1/responses"
        )

        async def send_request(*args, **kwargs):
            return response

        client.send = send_request
        with patch.object(proxy, "CLIENT", client):
            return await proxy.proxy(Request(self.scope, self.queue.get))

    async def send(self, message):
        self.sent.append(message)

    def record(self, stream):
        records = [json.loads(call.args[0]) for call in self.logger.info.call_args_list]
        finals = [
            record for record in records if record.get("event") != "request_start"
        ]
        self.assertEqual(len(finals), 1)
        self.assertEqual(stream.close_calls, 1)
        return finals[0]

    async def test_complete_stream_preserves_wire_status_and_headers(self):
        stream = FixtureStream()
        response = await self.response(stream)
        await response(self.scope, self.queue.get, self.send)
        record = self.record(stream)
        self.assertEqual(record["termination_reason"], "complete")
        self.assertEqual(record["upstream_state"], "eof")
        self.assertEqual(record["upstream_chunks_received"], 2)
        self.assertEqual(record["upstream_bytes_received"], 9)
        self.assertEqual(record["downstream_bytes_sent"], 9)
        self.assertLessEqual(record["first_chunk_ms"], record["last_chunk_ms"])
        self.assertEqual(record["response_body"], "alphabeta")
        self.assertEqual(self.sent[0]["status"], 207)
        self.assertIn((b"x-fixture", b"yes"), self.sent[0]["headers"])
        self.assertEqual(
            [value for key, value in self.sent[0]["headers"] if key == b"set-cookie"],
            [b"a=1", b"b=2"],
        )
        self.assertEqual(b"".join(m.get("body", b"") for m in self.sent), b"alphabeta")
        self.assertFalse(self.sent[-1]["more_body"])
        await response.finish()
        self.record(stream)

    async def test_final_send_disconnect_does_not_cancel_accepted_completion(self):
        stream = FixtureStream()
        response = await self.response(stream)

        async def send(message):
            self.sent.append(message)
            if message["type"] == "http.response.body" and not message["more_body"]:
                self.queue.put_nowait({"type": "http.disconnect"})
                await asyncio.sleep(0)
                await asyncio.sleep(0)

        await response(self.scope, self.queue.get, send)
        record = self.record(stream)
        self.assertEqual(record["termination_reason"], "complete")
        self.assertTrue(record["downstream_complete"])
        self.assertTrue(record["downstream_disconnect"])
        self.assertTrue(record["downstream_disconnect_during_final_send"])

    async def test_failed_final_send_still_reports_original_error_after_disconnect(
        self,
    ):
        stream = FixtureStream()
        response = await self.response(stream)
        error = OSError("fixture final send failure")

        async def send(message):
            if message["type"] == "http.response.body" and not message["more_body"]:
                self.queue.put_nowait({"type": "http.disconnect"})
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                raise error

        with self.assertRaises(OSError) as caught:
            await response(self.scope, self.queue.get, send)
        self.assertIs(caught.exception, error)
        record = self.record(stream)
        self.assertEqual(record["termination_reason"], "downstream_send_error")
        self.assertFalse(record["downstream_complete"])
        self.assertTrue(record["downstream_disconnect"])

    async def test_capture_limit_does_not_truncate_wire(self):
        stream = FixtureStream()
        with patch.object(proxy, "CAPTURE_LIMIT", 3):
            response = await self.response(stream)
            await response(self.scope, self.queue.get, self.send)
        record = self.record(stream)
        self.assertEqual(record["response_body"], "alp")
        self.assertTrue(record["response_capture_truncated"])
        self.assertEqual(record["downstream_bytes_sent"], 9)

    async def test_upstream_error_keeps_partial_capture_and_original_error(self):
        error = OSError("fixture upstream failure")
        stream = FixtureStream(chunks=(b"alpha",), error=error)
        response = await self.response(stream)
        with self.assertRaises(OSError) as caught:
            await response(self.scope, self.queue.get, self.send)
        self.assertIs(caught.exception, error)
        record = self.record(stream)
        self.assertEqual(record["termination_reason"], "upstream_error")
        self.assertEqual(record["response_body"], "alpha")
        self.assertFalse(record["downstream_complete"])
        self.assertTrue(all(m.get("more_body", True) for m in self.sent))

    async def test_observed_disconnect_interrupts_stalled_upstream(self):
        stream = FixtureStream(chunks=(b"alpha",), stall=True)
        response = await self.response(stream)
        task = asyncio.create_task(response(self.scope, self.queue.get, self.send))
        await stream.waiting.wait()
        self.queue.put_nowait({"type": "http.disconnect"})
        await asyncio.wait_for(task, 1)
        record = self.record(stream)
        self.assertEqual(record["termination_reason"], "downstream_disconnect")
        self.assertTrue(record["downstream_disconnect"])
        self.assertLessEqual(
            record["last_chunk_ms"], record["downstream_disconnect_ms"]
        )
        self.assertLessEqual(record["downstream_disconnect_ms"], record["elapsed_ms"])
        self.assertEqual(record["upstream_state"], "cancelled")
        self.assertFalse(record["cancelled"])

    async def test_downstream_send_failure_on_headers_and_body(self):
        for failed_type in ("http.response.start", "http.response.body"):
            with self.subTest(failed_type=failed_type):
                self.logger.reset_mock()
                self.queue.put_nowait(
                    {"type": "http.request", "body": b"{}", "more_body": False}
                )
                stream = FixtureStream()
                response = await self.response(stream)
                error = OSError("fixture downstream failure")

                async def fail_send(message, failed_type=failed_type, error=error):
                    if message["type"] == failed_type:
                        raise error

                with self.assertRaises(OSError) as caught:
                    await response(self.scope, self.queue.get, fail_send)
                self.assertIs(caught.exception, error)
                record = self.record(stream)
                self.assertEqual(record["termination_reason"], "downstream_send_error")
                self.assertEqual(record["downstream_bytes_sent"], 0)
                self.assertLessEqual(
                    record["upstream_headers_ms"], record["downstream_send_error_ms"]
                )
                self.assertLessEqual(
                    record["downstream_send_error_ms"], record["elapsed_ms"]
                )
                self.assertFalse(record["downstream_disconnect"])

    async def test_repeated_task_cancellation_during_close(self):
        stream = FixtureStream(chunks=(b"alpha",), stall=True)
        stream.close_release = asyncio.Event()
        response = await self.response(stream)
        task = asyncio.create_task(response(self.scope, self.queue.get, self.send))
        await stream.waiting.wait()
        task.cancel()
        await stream.close_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        stream.close_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        record = self.record(stream)
        self.assertEqual(record["termination_reason"], "cancelled")
        self.assertTrue(record["cancelled"])

    async def test_anyio_cancellation_shields_cleanup(self):
        stream = FixtureStream(chunks=(b"alpha",), stall=True)
        response = await self.response(stream)
        async with anyio.create_task_group() as group:
            group.start_soon(response, self.scope, self.queue.get, self.send)
            await stream.waiting.wait()
            group.cancel_scope.cancel()
        self.assertEqual(self.record(stream)["termination_reason"], "cancelled")

    async def test_cleanup_failure_does_not_replace_upstream_error(self):
        error = RuntimeError("fixture source error")
        stream = FixtureStream(error=error)
        stream.close_error = RuntimeError("fixture close error")
        response = await self.response(stream)
        with self.assertRaises(RuntimeError) as caught:
            await response(self.scope, self.queue.get, self.send)
        self.assertIs(caught.exception, error)
        self.assertIn("upstream_close_error", self.record(stream))

    async def test_close_timeout_still_finalizes_capture(self):
        stream = FixtureStream(error=RuntimeError("fixture upstream error"))
        stream.close_release = asyncio.Event()
        response = await self.response(stream)
        with (
            patch.object(proxy, "CLOSE_TIMEOUT_SECONDS", 0.01),
            self.assertRaises(RuntimeError),
        ):
            await response(self.scope, self.queue.get, self.send)
        self.assertEqual(self.record(stream)["upstream_close_error"], "close_timeout")

    async def test_cancellation_while_awaiting_headers_is_recorded(self):
        waiting = asyncio.Event()
        client = Mock()

        async def send_request(*args, **kwargs):
            waiting.set()
            await asyncio.Event().wait()

        client.send = send_request
        with patch.object(proxy, "CLIENT", client):
            task = asyncio.create_task(proxy.proxy(Request(self.scope, self.queue.get)))
            await waiting.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        records = [json.loads(call.args[0]) for call in self.logger.info.call_args_list]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["upstream_state"], "awaiting_headers")
        self.assertEqual(records[-1]["termination_reason"], "cancelled")

    async def test_request_body_disconnect_is_abandoned_without_upstream(self):
        await self.queue.get()
        self.queue.put_nowait(
            {"type": "http.request", "body": b"partial", "more_body": True}
        )
        self.queue.put_nowait({"type": "http.disconnect"})
        client = Mock()
        with patch.object(proxy, "CLIENT", client):
            response = await proxy.proxy(Request(self.scope, self.queue.get))
        self.assertEqual(response.status_code, 499)
        client.build_request.assert_not_called()
        records = [json.loads(call.args[0]) for call in self.logger.info.call_args_list]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["termination_reason"], "client_abandoned_input")
        self.assertNotIn("request_body", records[0])


if __name__ == "__main__":
    unittest.main()
