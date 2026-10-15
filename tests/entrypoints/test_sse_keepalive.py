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


def _sse_endpoint():
    from types import SimpleNamespace

    from fastapi.responses import StreamingResponse

    from vllm.entrypoints.serve.utils.api_utils import load_aware_call

    async def body():
        yield "data: event\n\n"

    @load_aware_call
    async def endpoint(request, raw_request):
        return StreamingResponse(body(), media_type="text/event-stream")

    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    return endpoint, raw_request


@pytest.mark.parametrize("interval, wrapped", [("30", True), ("0", False)])
def test_sse_keepalive_interval_from_env(monkeypatch, interval, wrapped) -> None:
    monkeypatch.setenv("VLLM_SSE_KEEPALIVE_INTERVAL_S", interval)
    endpoint, raw_request = _sse_endpoint()
    response = asyncio.run(endpoint(None, raw_request=raw_request))
    is_wrapped = getattr(response.body_iterator, "__name__", "") == (
        "sse_with_keepalive"
    )
    assert is_wrapped is wrapped
