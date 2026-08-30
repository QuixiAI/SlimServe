# SPDX-License-Identifier: Apache-2.0
"""AdmissionControlMiddleware: cap, release, and scope behavior."""

import asyncio

import pytest

from vllm.entrypoints.serve.utils.server_utils import AdmissionControlMiddleware


def _scope(path: str, method: str = "POST"):
    return {"type": "http", "method": method, "path": path, "root_path": ""}


class _App:
    """ASGI app that parks in-flight requests until released."""

    def __init__(self):
        self.release = asyncio.Event()
        self.started = 0

    async def __call__(self, scope, receive, send):
        self.started += 1
        await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def _null_receive():
    return {"type": "http.request"}


class _Sink:
    def __init__(self):
        self.status = None

    async def __call__(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]


def test_over_limit_gets_429_and_slots_free_on_completion():
    async def run():
        app = _App()
        mw = AdmissionControlMiddleware(app, max_concurrent=2)

        sinks = [_Sink() for _ in range(3)]
        tasks = [
            asyncio.create_task(
                mw(_scope("/v1/chat/completions"), _null_receive, sink)
            )
            for sink in sinks
        ]
        await asyncio.sleep(0.01)
        # Two admitted and parked; the third bounced immediately with 429.
        assert app.started == 2
        assert sorted(s.status for s in sinks if s.status is not None) == [429]

        app.release.set()
        await asyncio.gather(*tasks)
        assert [s.status for s in sinks].count(200) == 2

        # Slots were released: a new request is admitted.
        sink = _Sink()
        await mw(_scope("/v1/completions"), _null_receive, sink)
        assert sink.status == 200
        assert mw.in_flight == 0

    asyncio.run(run())


def test_slot_released_when_app_raises():
    class _Boom:
        async def __call__(self, scope, receive, send):
            raise RuntimeError("engine died")

    async def run():
        mw = AdmissionControlMiddleware(_Boom(), max_concurrent=1)
        with pytest.raises(RuntimeError):
            await mw(_scope("/v1/chat/completions"), _null_receive, _Sink())
        assert mw.in_flight == 0

    asyncio.run(run())


def test_non_generation_routes_are_never_capped():
    async def run():
        app = _App()
        app.release.set()
        mw = AdmissionControlMiddleware(app, max_concurrent=1)
        mw.in_flight = 1  # saturate

        for scope in (
            _scope("/v1/models", method="GET"),
            _scope("/health", method="GET"),
            _scope("/v1/chat/completions", method="OPTIONS"),
            {"type": "lifespan"},
        ):
            await mw(scope, _null_receive, _Sink())
        # Lifespan needs a functioning downstream app too, so it counts.
        assert app.started == 4

    asyncio.run(run())
