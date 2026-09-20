# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disconnect cleanup must reach EngineCore without relying on generator GC."""

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import ClientDisconnect

from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.responses.api_router import create_responses
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.utils import api_utils
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("mode", ["heartbeat_send_failure", "data_send_failure"])
@pytest.mark.parametrize("background", [False, True])
def test_responses_disconnect_engine_abort_ownership(
    monkeypatch, spec, mode, background
):
    monkeypatch.setattr(
        api_utils,
        "sse_with_keepalive",
        partial(api_utils.sse_with_keepalive, interval_seconds=0.005),
    )

    async def run():
        waiting = asyncio.Event()
        aborted = asyncio.Event()
        retained_generators = []
        request = ResponsesRequest(
            model="test",
            input="test",
            stream=True,
            store=background,
            background=background,
        )
        params = SamplingParams(max_tokens=4)
        context = SimpleContext()
        request_id = request.request_id + "_0"

        async def get():
            waiting.set()
            await asyncio.Event().wait()

        output = RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=[1],
            prompt_logprobs=None,
            finished=False,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="fixture",
                    token_ids=[2],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=None,
                )
            ],
        )
        initial = [output] if mode == "data_send_failure" else []
        collector = SimpleNamespace(
            request_id=request_id,
            get_nowait=lambda: initial.pop() if initial else None,
            get=get,
            close=Mock(),
        )

        async def abort_send(ids):
            assert ids == [request_id]
            await asyncio.sleep(0.02)
            aborted.set()

        engine = SimpleNamespace(
            add_request=AsyncMock(return_value=collector),
            log_requests=False,
            output_processor=SimpleNamespace(
                abort_requests=Mock(return_value=[request_id]),
            ),
            engine_core=SimpleNamespace(
                abort_requests_async=AsyncMock(side_effect=abort_send)
            ),
        )
        engine.abort = partial(AsyncLLM.abort, engine)

        def generate(*args, **kwargs):
            generator = AsyncLLM.generate(engine, *args, **kwargs)
            retained_generators.append(generator)
            return generator

        engine.generate = generate
        serving = object.__new__(OpenAIServingResponses)
        serving.model_config = SimpleNamespace(max_model_len=128)
        serving.engine_client = engine
        serving._log_inputs = Mock()
        serving.use_harmony = False
        serving.parser = None
        serving.event_store = {}
        serving.response_store = {}
        serving.response_store_lock = asyncio.Lock()
        result = serving._generate_with_builtin_tools(
            request.request_id,
            {"prompt_token_ids": [1]},
            params,
            context,
        )
        args = (
            request,
            params,
            result,
            context,
            "test",
            None,
            RequestResponseMetadata(request_id=request.request_id),
        )
        producer = None
        if background:
            producer = asyncio.create_task(
                serving._run_background_request_stream(*args)
            )
            await asyncio.wait_for(waiting.wait(), 1)
            events = serving.responses_background_stream_generator(request.request_id)
        else:
            events = serving.responses_stream_generator(*args)
        retained_generators.extend([result, events])
        serving.create_responses = AsyncMock(return_value=events)

        async def receive():
            await asyncio.Event().wait()

        raw_request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(openai_serving_responses=serving)
            ),
            receive=receive,
        )
        response = await create_responses(request, raw_request)
        error = OSError("synthetic downstream send failure")

        async def send(message):
            if message["type"] != "http.response.body":
                return
            body = message["body"]
            if (mode == "heartbeat_send_failure" and body == b": keepalive\n\n") or (
                mode == "data_send_failure"
                and b"event: response.output_item.added\n" in body
            ):
                raise error

        with pytest.raises(ClientDisconnect if spec == "2.4" else OSError):
            await asyncio.wait_for(
                response(
                    {"type": "http", "asgi": {"spec_version": spec}}, receive, send
                ),
                1,
            )
        if background:
            # Closing a replay subscriber must not cancel the independently
            # owned generation. Its producer must remain active until cancelled.
            assert not aborted.is_set()
            assert not producer.done()
            producer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await producer
        assert aborted.is_set()
        engine.output_processor.abort_requests.assert_called_once_with(
            (request_id,),
            True,
        )
        engine.engine_core.abort_requests_async.assert_awaited_once_with([request_id])
        collector.close.assert_called_once()
        assert all(g.ag_frame is None for g in retained_generators)
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(run())
