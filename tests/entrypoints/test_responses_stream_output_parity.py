# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Terminal Responses must preserve the calls already emitted over SSE."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.responses.api_router import _convert_stream_to_sse_events
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.qwen3 import Qwen3Parser, qwen3_config
from vllm.sampling_params import SamplingParams, StructuredOutputsParams


async def _stream(text, finish_reason="stop", *, legacy=False, no_events=False):
    request = ResponsesRequest(
        model="test",
        input="test",
        stream=True,
        store=True,
        tool_choice="required",
        tools=[
            {
                "type": "function",
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "strict": False,
            }
        ],
    )
    params = SamplingParams(
        max_tokens=64,
        structured_outputs=StructuredOutputsParams(
            json_object=True, _required_tool_call=True
        ),
    )
    tokenizer = make_mock_tokenizer(
        {"<think>": 200, "</think>": 201, "<tool_call>": 202, "</tool_call>": 203}
    )
    parser = Qwen3Parser(
        tokenizer,
        tools=request.tools,
        parser_engine_config=replace(qwen3_config(), stream_arg_deltas=legacy),
    )
    parser.parse = Mock(wraps=parser.parse)
    context = SimpleContext()
    context.response_parser = parser
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving.parser = None
    serving.enable_auto_tools = True
    serving.enable_log_outputs = False
    serving._initialize_tool_sessions = AsyncMock()
    serving.response_store = {}
    serving.response_store_lock = asyncio.Lock()

    if finish_reason == "abort":
        # Cancellation may already be recorded by the cancel endpoint.
        cancelled = ResponsesResponse.from_request(
            request, params, "test", 1, [], "cancelled"
        )
        serving.response_store[request.request_id] = cancelled
    else:
        cancelled = None

    async def results():
        for start in range(0, len(text), 7):
            last = start + 7 >= len(text)
            context.append_output(
                RequestOutput(
                    request_id=request.request_id,
                    prompt="test",
                    prompt_token_ids=[1],
                    prompt_logprobs=None,
                    finished=last,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=text[start : start + 7],
                            token_ids=[],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason=finish_reason if last else None,
                        )
                    ],
                )
            )
            yield context

    if no_events:

        async def consume_without_events(*args):
            async for _ in args[2]:
                pass
            if False:
                yield

        serving._process_simple_streaming_events = consume_without_events

    serialized = [
        json.loads(event.split("\ndata: ", 1)[1])
        async for event in _convert_stream_to_sse_events(
            serving.responses_stream_generator(
                request,
                params,
                results(),
                context,
                "test",
                tokenizer,
                RequestResponseMetadata(request_id=request.request_id),
            )
        )
    ]
    parser.parse.assert_not_called()
    saved = serving.response_store[request.request_id]
    if cancelled is not None:
        assert saved is cancelled
    return serialized, saved.model_dump(mode="json", by_alias=True)


def _assert_terminal_parity(events, stored, status):
    assert events[-1]["type"] == f"response.{status}"
    terminal = events[-1]["response"]
    done = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    ]
    assert terminal["output"] == done
    assert stored == terminal
    for item in done:
        if item["type"] != "function_call":
            continue
        fragments = [
            event["delta"]
            for event in events
            if event["type"] == "response.function_call_arguments.delta"
            and event["item_id"] == item["id"]
        ]
        assert "".join(fragments) == item["arguments"]
        added = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["id"] == item["id"]
        )
        assert added["call_id"] == item["call_id"]
    return terminal


@pytest.mark.parametrize("legacy", [False, True])
def test_duplicate_parameter_never_changes_arguments_or_identity_at_terminal(legacy):
    text = (
        "<tool_call>\n<function=f>\n"
        "<parameter=value>first</parameter>\n"
        "<parameter=value>second</parameter>\n"
        "</function>\n</tool_call>"
    )
    events, stored = asyncio.run(_stream(text, legacy=legacy))
    status = "failed" if legacy else "completed"
    terminal = _assert_terminal_parity(events, stored, status)
    item = terminal["output"][0]
    if legacy:
        assert item["arguments"] == '{"value": "first'
        assert item["status"] == "incomplete"
        assert terminal["error"]["code"] == "server_error"
        assert not any(
            e["type"] == "response.function_call_arguments.done" for e in events
        )
    else:
        assert json.loads(item["arguments"]) == {"value": "second"}
        assert item["status"] == "completed"


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "suffix",
    ["partial", "partial</parameter>", "partial</parameter></function></tool_call>"],
)
def test_length_truncation_is_incomplete_even_with_parseable_arguments(legacy, suffix):
    text = "<tool_call><function=f><parameter=value>" + suffix
    events, stored = asyncio.run(_stream(text, "length", legacy=legacy))
    terminal = _assert_terminal_parity(events, stored, "incomplete")
    assert terminal["incomplete_details"]["reason"] == "max_output_tokens"
    assert terminal["output"][-1]["status"] == "incomplete"
    assert not any(e["type"] == "response.function_call_arguments.done" for e in events)
    if not legacy and suffix == "partial":
        # EOF closes parser state, but drops an unclosed XML parameter. The
        # resulting {} must retain the source generation's incomplete status.
        assert terminal["output"][-1]["arguments"] == "{}"


def test_multiple_calls_and_other_items_keep_stream_order_and_identity():
    text = (
        "Planning. </think>Before tools. "
        "<tool_call><function=f><parameter=value>one</parameter></function></tool_call>"
        "Between tools. "
        "<tool_call><function=f><parameter=value>two</parameter></function></tool_call>"
        "After tools."
    )
    events, stored = asyncio.run(_stream(text))
    terminal = _assert_terminal_parity(events, stored, "completed")
    assert [item["type"] for item in terminal["output"]] == [
        "reasoning",
        "message",
        "function_call",
        "message",
        "function_call",
        "message",
    ]
    calls = [item for item in terminal["output"] if item["type"] == "function_call"]
    assert [json.loads(call["arguments"]) for call in calls] == [
        {"value": "one"},
        {"value": "two"},
    ]
    assert calls[0]["call_id"] != calls[1]["call_id"]


def test_no_streamed_items_cannot_reconstruct_successful_required_call():
    text = (
        "<tool_call><function=f><parameter=value>one</parameter></function></tool_call>"
    )
    events, stored = asyncio.run(_stream(text, no_events=True))
    terminal = _assert_terminal_parity(events, stored, "failed")
    assert terminal["output"] == []
    assert "required tool call" in terminal["error"]["message"]


def test_abort_keeps_cancelled_storage_and_does_not_emit_completion():
    events, stored = asyncio.run(
        _stream("<tool_call><function=f><parameter=value>partial", "abort")
    )
    assert stored["status"] == "cancelled"
    assert not any(
        event["type"]
        in ("response.completed", "response.failed", "response.incomplete")
        for event in events
    )


@pytest.mark.parametrize(
    ("finish_reason", "item_status", "response_status"),
    [
        ("stop", "completed", "completed"),
        ("stop", "incomplete", "failed"),
        ("length", "completed", "incomplete"),
        ("length", "incomplete", "incomplete"),
        ("abort", "completed", "cancelled"),
    ],
)
def test_response_status_does_not_rewrite_emitted_item_status(
    finish_reason, item_status, response_status
):
    from openai.types.responses import ResponseFunctionToolCall

    from tests.entrypoints.test_responses_function_completion import (
        make_context,
        make_serving,
        request_args,
    )

    async def run():
        item = ResponseFunctionToolCall(
            type="function_call",
            id="fc_streamed",
            call_id="call_streamed",
            name="f",
            arguments="{}",
            status=item_status,
        )
        emitted = item.model_dump(mode="json")
        serving = make_serving([])
        response = await serving.responses_full_generator(
            *request_args(make_context(finish_reason)), streamed_output=[item]
        )
        assert response.status == response_status
        assert response.output[0].model_dump(mode="json") == emitted
        serving._make_response_output_items.assert_not_called()
        if response_status == "failed":
            assert response.error.code == "server_error"

    asyncio.run(run())
