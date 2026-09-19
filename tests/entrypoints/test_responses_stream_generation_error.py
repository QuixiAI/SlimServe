# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation failures must survive Responses SSE serialization unchanged."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import TypeAdapter

from vllm.entrypoints.openai.engine.protocol import (
    GenerationError,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.api_router import _convert_stream_to_sse_events
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    StreamingResponsesResponse,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    "failure", ["finish_reason", "generator", "finalize", "envelope"]
)
@pytest.mark.parametrize("store", [False, True])
def test_generation_failure_emits_valid_terminal_sse(failure, store):
    async def run():
        serving = object.__new__(OpenAIServingResponses)
        serving.use_harmony = False
        serving.parser = None
        serving.response_store = {}
        serving.response_store_lock = asyncio.Lock()
        serving.event_store = {}
        request = ResponsesRequest(model="test", input="test", stream=True, store=store)
        params = SamplingParams(max_tokens=4)
        context = SimpleContext()
        message = "Synthetic generation failure"

        async def results():
            if failure == "generator":
                raise GenerationError(message)
            if failure == "finish_reason":
                context.append_output(
                    RequestOutput(
                        request_id=request.request_id,
                        prompt="test",
                        prompt_token_ids=[1],
                        prompt_logprobs=None,
                        finished=True,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="",
                                token_ids=[],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="error",
                            )
                        ],
                    )
                )
                yield context

        if failure == "finish_reason":
            message = "Internal server error"
        elif failure == "finalize":
            serving.responses_full_generator = AsyncMock(
                side_effect=GenerationError(message)
            )
        elif failure == "envelope":
            serving.responses_full_generator = AsyncMock(
                return_value=serving.create_error_response(message)
            )

        args = (
            request,
            params,
            results(),
            context,
            "test",
            None,
            RequestResponseMetadata(request_id=request.request_id),
        )
        if store:
            # Exercise background replay as well: it must stop at the failure.
            await serving._run_background_request_stream(*args)
            stream = serving.responses_background_stream_generator(request.request_id)
        else:
            stream = serving.responses_stream_generator(*args)

        async def collect():
            return [event async for event in _convert_stream_to_sse_events(stream)]

        serialized = await asyncio.wait_for(collect(), timeout=2)
        expected = ["response.created", "response.in_progress", "response.failed"]
        events = []
        for event, event_type in zip(serialized, expected, strict=True):
            assert event.startswith(f"event: {event_type}\ndata: ")
            payload = event.split("\ndata: ", 1)[1].strip()
            TypeAdapter(StreamingResponsesResponse).validate_json(payload)
            events.append(json.loads(payload))
        assert [event["sequence_number"] for event in events] == [0, 1, 2]
        assert [event["response"]["status"] for event in events] == [
            "in_progress",
            "in_progress",
            "failed",
        ]
        assert all(event["response"]["id"] == request.request_id for event in events)
        assert events[-1]["response"]["error"] == {
            "code": "server_error",
            "message": message,
        }
        assert events[-1]["response"]["output"] == []
        if store:
            saved = serving.response_store[request.request_id]
            assert isinstance(saved, ResponsesResponse)
            assert saved.status == "failed"
            assert saved.error.message == message
        else:
            assert serving.response_store == {}

    asyncio.run(run())
