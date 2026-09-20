# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interrupted function arguments must never be advertised as executable calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
)

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingState,
    emit_simple_tool_call_delta,
    emit_simple_tool_call_done,
    emit_simple_tool_call_open,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    "arguments", ["", " ", '{"query":"unterminated', "{", "NaN", '{"x":Infinity}']
)
def test_invalid_arguments_are_never_completed(arguments):
    state = SimpleStreamingState()
    emit_simple_tool_call_open(state, "lookup", 0)
    if arguments:
        emit_simple_tool_call_delta(state, arguments)
    events = emit_simple_tool_call_done(state)
    assert [event.type for event in events] == ["response.output_item.done"]
    assert events[0].item.status == "incomplete"
    assert events[0].item.arguments == arguments


@pytest.mark.parametrize("arguments", ["{}", "[]", '{"query":"hello"}', '{"extra":42}'])
def test_valid_json_does_not_impose_schema_constraints(arguments):
    state = SimpleStreamingState()
    emit_simple_tool_call_open(state, "lookup", 0)
    emit_simple_tool_call_delta(state, arguments)
    events = emit_simple_tool_call_done(state)
    assert events[0].type == "response.function_call_arguments.done"
    assert events[0].arguments == arguments
    assert events[-1].item.status == "completed"


def test_forced_truncation_does_not_complete_active_call():
    state = SimpleStreamingState()
    emit_simple_tool_call_open(state, "lookup", 0)
    emit_simple_tool_call_delta(state, "{}")
    events = emit_simple_tool_call_done(state, incomplete=True)
    assert [event.type for event in events] == ["response.output_item.done"]
    assert events[-1].item.status == "incomplete"


def make_context(finish_reason):
    context = SimpleContext()
    context.append_output(
        RequestOutput(
            request_id="test",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            finished=True,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="tool output",
                    token_ids=[2],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
        )
    )
    return context


async def empty_generator():
    if False:
        yield


def make_serving(output):
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving._initialize_tool_sessions = AsyncMock()
    serving._make_response_output_items = Mock(return_value=output)
    return serving


def request_args(context):
    request = ResponsesRequest(
        model="test", input="test", store=False, max_output_tokens=4
    )
    return (
        request,
        SamplingParams(max_tokens=4),
        empty_generator(),
        context,
        "test",
        None,
        RequestResponseMetadata(request_id=request.request_id),
    )


@pytest.mark.parametrize(
    ("arguments", "finish_reason", "status"),
    [
        ('{"x":', "length", "incomplete"),
        ('{"x":', "stop", "failed"),
        ("", "stop", "failed"),
        ("{}", "stop", "completed"),
        ("{}", "length", "incomplete"),
    ],
)
def test_final_response_matches_stream_terminal_event(arguments, finish_reason, status):
    async def run():
        item = ResponseFunctionToolCall(
            type="function_call",
            name="lookup",
            call_id="call_test",
            arguments=arguments,
            status="completed",
        )
        serving = make_serving([item])
        context = make_context(finish_reason)
        response = await serving.responses_full_generator(*request_args(context))
        assert response.status == status
        if arguments in ("", '{"x":') or status == "incomplete":
            assert response.output[0].status == "incomplete"
        if status == "failed":
            assert response.error.code == "server_error"
        if status == "incomplete":
            assert response.incomplete_details.reason == "max_output_tokens"

        async def no_events(*args):
            if False:
                yield

        serving._process_simple_streaming_events = no_events
        events = [
            event
            async for event in serving.responses_stream_generator(
                *request_args(context)
            )
        ]
        assert events[-1].type == f"response.{status}"
        assert events[-1].response.status == status
        assert [e.sequence_number for e in events] == list(range(len(events)))

    asyncio.run(run())


def test_custom_tool_freeform_input_is_not_validated_as_json():
    async def run():
        item = ResponseCustomToolCall(
            type="custom_tool_call",
            name="python",
            call_id="call_test",
            input="print('hello')",
        )
        serving = make_serving([item])
        response = await serving.responses_full_generator(
            *request_args(make_context("stop"))
        )
        assert response.status == "completed"
        assert response.output[0].input == "print('hello')"

    asyncio.run(run())


@pytest.mark.parametrize(
    "terminal", ["response.incomplete", "response.failed", "response.completed"]
)
def test_background_stream_stops_on_each_terminal_event(terminal):
    async def run():
        serving = make_serving([])
        serving.event_store = {
            "test": ([SimpleNamespace(type=terminal)], asyncio.Event())
        }

        async def collect():
            return [
                event
                async for event in serving.responses_background_stream_generator("test")
            ]

        events = await asyncio.wait_for(collect(), timeout=1)
        assert len(events) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("arguments", "finish_reason", "completed"),
    [
        ('{"query":"unfinished', "length", False),
        ("{}", "length", False),
        ("{}", "stop", True),
    ],
)
def test_stream_processor_flush_respects_finish_reason(
    arguments, finish_reason, completed
):
    async def run():
        context = make_context(finish_reason)
        context.response_parser = Mock()
        context.response_parser.parse_delta.return_value = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(name="lookup", arguments=arguments),
                )
            ]
        )
        serving = make_serving([])
        serving.parser = None

        async def results():
            yield context

        args = list(request_args(context))
        args[2] = results()
        events = [
            event
            async for event in serving._process_simple_streaming_events(
                *args, 1, lambda event: event
            )
        ]
        done = [
            event
            for event in events
            if event.type == "response.function_call_arguments.done"
        ]
        assert bool(done) == completed
        assert events[-1].item.status == ("completed" if completed else "incomplete")

    asyncio.run(run())
