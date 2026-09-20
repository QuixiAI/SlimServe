# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm.entrypoints.serve.utils.api_utils import sse_with_keepalive


def test_sse_keepalive_preserves_pending_stream_item() -> None:
    async def run() -> None:
        release = asyncio.Event()

        async def source():
            yield "data: first\n\n"
            await release.wait()
            yield "data: second\n\n"

        stream = sse_with_keepalive(source(), interval_seconds=0.01)
        assert await anext(stream) == "data: first\n\n"
        assert await anext(stream) == ": keepalive\n\n"
        assert await anext(stream) == ": keepalive\n\n"
        release.set()
        assert await anext(stream) == "data: second\n\n"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(run())


def test_sse_keepalive_closes_silent_upstream() -> None:
    async def run() -> None:
        closed = asyncio.Event()

        async def source():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        stream = sse_with_keepalive(source(), interval_seconds=0.01)
        assert await anext(stream) == ": keepalive\n\n"
        await stream.aclose()
        await asyncio.wait_for(closed.wait(), timeout=1)

    asyncio.run(run())


def test_sse_keepalive_rejects_nonpositive_interval() -> None:
    async def run() -> None:
        async def source():
            yield "data: event\n\n"

        stream = sse_with_keepalive(source(), interval_seconds=0)
        with pytest.raises(ValueError, match="interval must be positive"):
            await anext(stream)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("spec", "mode"),
    [
        ("2.3", "pending_disconnect"),
        ("2.3", "heartbeat_disconnect"),
        ("2.3", "heartbeat_send_failure"),
        ("2.4", "heartbeat_send_failure"),
        ("2.3", "heartbeat_send_failure_cancel"),
        ("2.4", "heartbeat_send_failure_cancel"),
        ("2.3", "data_send_failure"),
        ("2.4", "data_send_failure"),
        ("2.3", "repeated_task_cancel"),
        ("2.4", "repeated_task_cancel"),
    ],
)
def test_sse_response_finishes_abort_cleanup(monkeypatch, spec, mode) -> None:
    from types import SimpleNamespace

    from starlette.requests import ClientDisconnect
    from starlette.responses import StreamingResponse

    from vllm.entrypoints.serve.utils import api_utils

    original_keepalive = api_utils.sse_with_keepalive
    monkeypatch.setattr(
        api_utils,
        "sse_with_keepalive",
        lambda content: original_keepalive(content, interval_seconds=0.005),
    )

    async def run() -> None:
        source_started = asyncio.Event()
        abort_started = asyncio.Event()
        abort_finished = asyncio.Event()
        body_sent = asyncio.Event()
        source_closed = asyncio.Event()
        sent = []
        abort_count = 0
        send_error = OSError("synthetic downstream send failure")

        async def source():
            nonlocal abort_count
            try:
                source_started.set()
                if mode == "data_send_failure":
                    yield "data: original\n\n"
                await asyncio.Event().wait()
                yield "unreachable"
            except (asyncio.CancelledError, GeneratorExit):
                abort_count += 1
                abort_started.set()
                # Engine abort can suspend while sending to EngineCore. It
                # must survive the response task's cancelled AnyIO scope.
                await asyncio.sleep(0.02)
                abort_finished.set()
                raise
            finally:
                source_closed.set()

        @api_utils.load_aware_call
        async def route(request, raw_request):
            response = StreamingResponse(
                source(), status_code=201, media_type="text/event-stream"
            )
            response.raw_headers.extend([(b"x-test", b"a"), (b"x-test", b"b")])
            return response

        response = await route(
            None,
            SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
        )

        async def receive():
            if mode == "pending_disconnect":
                await source_started.wait()
            elif mode == "heartbeat_disconnect":
                await body_sent.wait()
            else:
                await asyncio.Event().wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body":
                body_sent.set()
                if "send_failure" in mode:
                    raise send_error
                await asyncio.Event().wait()

        task = asyncio.create_task(
            response({"type": "http", "asgi": {"spec_version": spec}}, receive, send)
        )
        if mode == "repeated_task_cancel":
            await source_started.wait()
            task.cancel()
            await asyncio.wait_for(abort_started.wait(), 1)
            task.cancel()
        if mode == "heartbeat_send_failure_cancel":
            await asyncio.wait_for(abort_started.wait(), 1)
            task.cancel()
        if "send_failure" in mode:
            error_type = ClientDisconnect if spec == "2.4" else OSError
            with pytest.raises(error_type) as caught:
                await asyncio.wait_for(task, 1)
            if spec == "2.3":
                assert caught.value is send_error
        elif mode == "repeated_task_cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            await asyncio.wait_for(task, 1)

        assert abort_count == 1
        assert abort_finished.is_set()
        assert source_closed.is_set()
        assert sent[0]["status"] == 201
        assert [h for h in sent[0]["headers"] if h[0] == b"x-test"] == [
            (b"x-test", b"a"),
            (b"x-test", b"b"),
        ]
        bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
        if mode == "data_send_failure":
            assert bodies == [b"data: original\n\n"]
        elif mode.startswith("heartbeat"):
            assert bodies == [b": keepalive\n\n"]
        else:
            assert bodies == []
        # Every cleanup task must be joined before ASGI response completion.
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(run())


def test_sse_cleanup_failure_preserves_send_exception() -> None:
    from vllm.entrypoints.serve.utils import api_utils

    async def run() -> None:
        class Source:
            def __aiter__(self):
                return self

            async def __anext__(self):
                return "data: original\n\n"

            async def aclose(self):
                raise ValueError("synthetic cleanup failure")

        response = api_utils._SSEKeepaliveResponse(
            api_utils.sse_with_keepalive(Source()), media_type="text/event-stream"
        )
        original_error = OSError("synthetic send failure")

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                raise original_error

        with pytest.raises(OSError) as caught:
            await response(
                {"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send
            )
        assert caught.value is original_error

    asyncio.run(run())


def test_sse_cleanup_timeout_is_bounded(monkeypatch) -> None:
    from vllm.entrypoints.serve.utils import api_utils

    monkeypatch.setattr(api_utils, "SSE_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def run() -> None:
        finished = asyncio.Event()

        class SlowClose:
            async def aclose(self):
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()

        await asyncio.wait_for(api_utils._close_sse_stream(SlowClose()), 0.2)
        await asyncio.wait_for(finished.wait(), 0.2)

    asyncio.run(run())


def test_sse_slow_abort_timeout_drains_pending_task(monkeypatch) -> None:
    from vllm.entrypoints.serve.utils import api_utils

    monkeypatch.setattr(api_utils, "SSE_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def run() -> None:
        closed = asyncio.Event()
        abort_started = asyncio.Event()

        async def source():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            except asyncio.CancelledError:
                abort_started.set()
                # Simulate an abort transport that does not finish within the
                # graceful cleanup deadline, but cooperates with cancellation.
                await asyncio.Event().wait()
                raise
            finally:
                closed.set()

        stream = api_utils.sse_with_keepalive(source(), interval_seconds=0.001)
        assert await anext(stream) == ": keepalive\n\n"
        await asyncio.wait_for(stream.aclose(), 0.2)
        assert abort_started.is_set()
        assert closed.is_set()
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(run())
