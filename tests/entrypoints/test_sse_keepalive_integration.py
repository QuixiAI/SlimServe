# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keepalive wiring must work independently of optional server load tracking."""

import asyncio
from functools import partial
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.serve.utils import api_utils


@pytest.mark.parametrize("load_tracking", [False, True])
def test_load_aware_sse_preserves_events_status_and_headers(monkeypatch, load_tracking):
    monkeypatch.setattr(
        api_utils,
        "sse_with_keepalive",
        partial(api_utils.sse_with_keepalive, interval_seconds=0.001),
    )

    async def run():
        state = SimpleNamespace(enable_server_load_tracking=load_tracking)
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        release = asyncio.Event()
        closed = asyncio.Event()
        events = ['event: fixture\ndata: {"value":1}\n\n', "data: fixture-done\n\n"]

        async def source():
            try:
                yield events[0]
                await release.wait()
                yield events[1]
            finally:
                closed.set()

        original = StreamingResponse(
            source(),
            status_code=207,
            headers={"x-fixture": "preserved"},
            media_type="text/event-stream",
        )

        @api_utils.load_aware_call
        async def route(raw_request):
            return original

        response = await route(raw_request=request)
        assert response is original
        assert response.status_code == 207
        assert response.headers["x-fixture"] == "preserved"
        assert await anext(response.body_iterator) == events[0]
        assert (
            await asyncio.wait_for(anext(response.body_iterator), timeout=0.5)
            == ": keepalive\n\n"
        )
        release.set()
        assert await anext(response.body_iterator) == events[1]
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
        assert closed.is_set()
        if load_tracking:
            assert state.server_load_metrics == 1
            await response.background()
            assert state.server_load_metrics == 0
        else:
            assert not hasattr(state, "server_load_metrics")
            assert response.background is None

    asyncio.run(run())


def test_load_aware_keeps_json_response_unchanged_when_tracking_disabled():
    async def run():
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(enable_server_load_tracking=False)
            )
        )
        original = JSONResponse({"fixture": True}, status_code=201)

        @api_utils.load_aware_call
        async def route(raw_request):
            return original

        response = await route(raw_request=request)
        assert response is original
        assert response.status_code == 201
        assert response.body == b'{"fixture":true}'

    asyncio.run(run())


def test_load_aware_does_not_add_comments_to_non_sse_stream():
    async def run():
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(enable_server_load_tracking=False)
            )
        )

        async def source():
            yield b"fixture"

        iterator = source()
        original = StreamingResponse(iterator, media_type="text/plain")

        @api_utils.load_aware_call
        async def route(raw_request):
            return original

        response = await route(raw_request=request)
        assert response is original
        assert response.body_iterator is iterator
        assert [item async for item in response.body_iterator] == [b"fixture"]

    asyncio.run(run())
