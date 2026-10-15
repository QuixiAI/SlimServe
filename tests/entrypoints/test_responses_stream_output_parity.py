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
from vllm.entrypoints.openai.responses.serving import (
    RESPONSE_CANCELLED_CODE,
    OpenAIServingResponses,
)
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
    # Responses has no response.cancelled event: the stream still ends, with
    # a typed error event that names the cancellation.
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == RESPONSE_CANCELLED_CODE
    assert [event["sequence_number"] for event in events] == list(range(len(events)))


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


def _logprob_outputs():
    from vllm.logprobs import Logprob

    # Two deltas of two tokens; top-2 alternatives per position. "é" checks
    # that token bytes are the UTF-8 encoding, as in the non-streamed path.
    tokens = [(11, "Hé"), (12, "llo"), (13, ", "), (14, "world")]
    outputs = []
    for start in (0, 2):
        chunk = tokens[start : start + 2]
        outputs.append(
            CompletionOutput(
                index=0,
                text="".join(text for _, text in chunk),
                token_ids=[token_id for token_id, _ in chunk],
                cumulative_logprob=None,
                logprobs=[
                    {
                        token_id: Logprob(-0.25, rank=1, decoded_token=text),
                        99: Logprob(-2.5, rank=2, decoded_token="alt"),
                    }
                    for token_id, text in chunk
                ],
                finish_reason="stop" if start else None,
            )
        )
    return outputs


def _logprob_serving():
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving.parser = None
    serving.enable_auto_tools = False
    serving.enable_log_outputs = False
    serving.return_tokens_as_token_ids = False
    serving._initialize_tool_sessions = AsyncMock()
    serving.response_store = {}
    serving.response_store_lock = asyncio.Lock()
    return serving


def _logprob_request():
    return ResponsesRequest(
        model="test",
        input="test",
        store=False,
        include=["message.output_text.logprobs"],
        top_logprobs=2,
    )


async def _logprob_context(context, request):
    for output in _logprob_outputs():
        context.append_output(
            RequestOutput(
                request_id=request.request_id,
                prompt="test",
                prompt_token_ids=[1],
                prompt_logprobs=None,
                finished=output.finish_reason is not None,
                outputs=[output],
            )
        )
        yield context


def test_streamed_terminal_output_keeps_requested_logprobs():
    async def run():
        request = _logprob_request()
        params = SamplingParams(max_tokens=16, logprobs=2)
        context = SimpleContext()
        events = [
            event
            async for event in _logprob_serving().responses_stream_generator(
                request,
                params,
                _logprob_context(context, request),
                context,
                "test",
                None,
                RequestResponseMetadata(request_id=request.request_id),
            )
        ]
        streamed = [
            logprob.model_dump()
            for event in events
            if event.type == "response.output_text.delta"
            for logprob in event.logprobs
        ]
        assert [logprob["token"] for logprob in streamed] == [
            "Hé",
            "llo",
            ", ",
            "world",
        ]

        # The same request, not streamed, is the reference.
        context = SimpleContext()
        async for _ in _logprob_context(context, request):
            pass
        reference = await _logprob_serving().responses_full_generator(
            request,
            params,
            _logprob_context(SimpleContext(), request),
            context,
            "test",
            None,
            RequestResponseMetadata(request_id=request.request_id),
        )
        expected = reference.output[0].content[0].logprobs
        assert expected is not None and len(expected) == 4
        assert expected[0].bytes == list("Hé".encode())

        terminal = events[-1]
        assert terminal.type == "response.completed"
        part = terminal.response.output[0].content[0]
        assert part.text == "Héllo, world"
        assert part.logprobs == expected
        text_done = next(e for e in events if e.type == "response.output_text.done")
        assert [logprob.model_dump() for logprob in text_done.logprobs] == streamed

    asyncio.run(run())


def test_streamed_terminal_output_omits_unrequested_logprobs():
    async def run():
        request = ResponsesRequest(model="test", input="test", store=False)
        context = SimpleContext()
        events = [
            event
            async for event in _logprob_serving().responses_stream_generator(
                request,
                SamplingParams(max_tokens=16),
                _logprob_context(context, request),
                context,
                "test",
                None,
                RequestResponseMetadata(request_id=request.request_id),
            )
        ]
        assert events[-1].response.output[0].content[0].logprobs is None

    asyncio.run(run())


_SPANNING_CHUNKS = [
    # The second chunk carries the end marker with the first message tokens.
    [(200, "<think>"), (21, "plan")],
    [(201, "</think>"), (22, "Hi"), (23, " there")],
]
_SCAFFOLD_CHUNKS = [
    # The template scaffold arrives alone and produces no text delta.
    [(200, "<think>"), (21, "plan")],
    [(201, "</think>")],
    [(24, "\n\n")],
    [(22, "Hi"), (23, " there")],
]


def _reasoning_logprob_outputs(chunks):
    from vllm.logprobs import Logprob

    outputs = []
    for i, chunk in enumerate(chunks):
        last = i == len(chunks) - 1
        outputs.append(
            CompletionOutput(
                index=0,
                text="".join(text for _, text in chunk),
                token_ids=[token_id for token_id, _ in chunk],
                cumulative_logprob=None,
                logprobs=[
                    {token_id: Logprob(-0.5, rank=1, decoded_token=text)}
                    for token_id, text in chunk
                ],
                finish_reason="stop" if last else None,
            )
        )
    return outputs


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (_SPANNING_CHUNKS, ["Hi", " there"]),
        (_SCAFFOLD_CHUNKS, ["\n\n", "Hi", " there"]),
    ],
)
def test_output_text_logprobs_cover_only_message_tokens_with_reasoning_parser(
    chunks, expected
):
    """Streamed and non-streamed output_text logprobs exclude reasoning and
    its markers, even when one chunk spans the reasoning boundary."""

    tokenizer = make_mock_tokenizer({"<think>": 200, "</think>": 201})

    def parsed_context():
        context = SimpleContext()
        context.response_parser = Qwen3Parser(tokenizer, tools=[])
        return context

    async def feed(context, request):
        for output in _reasoning_logprob_outputs(chunks):
            context.append_output(
                RequestOutput(
                    request_id=request.request_id,
                    prompt="test",
                    prompt_token_ids=[1],
                    prompt_logprobs=None,
                    finished=output.finish_reason is not None,
                    outputs=[output],
                )
            )
            yield context

    async def run():
        request = ResponsesRequest(
            model="test",
            input="test",
            store=False,
            include=["message.output_text.logprobs"],
        )
        params = SamplingParams(max_tokens=16, logprobs=1)

        context = parsed_context()
        events = [
            event
            async for event in _logprob_serving().responses_stream_generator(
                request,
                params,
                feed(context, request),
                context,
                "test",
                tokenizer,
                RequestResponseMetadata(request_id=request.request_id),
            )
        ]
        streamed = [
            logprob.token
            for event in events
            if event.type == "response.output_text.delta"
            for logprob in event.logprobs
        ]
        terminal = events[-1]
        assert terminal.type == "response.completed"
        message = next(i for i in terminal.response.output if i.type == "message")
        streamed_part = message.content[0]

        context = parsed_context()
        async for _ in feed(context, request):
            pass
        reference = await _logprob_serving().responses_full_generator(
            request,
            params,
            feed(SimpleContext(), request),
            context,
            "test",
            tokenizer,
            RequestResponseMetadata(request_id=request.request_id),
        )
        reference_message = next(i for i in reference.output if i.type == "message")
        reference_part = reference_message.content[0]

        assert streamed_part.text == reference_part.text == "Hi there"
        assert streamed == expected
        assert [lp.token for lp in reference_part.logprobs] == expected
        assert streamed_part.logprobs == reference_part.logprobs

    asyncio.run(run())


@pytest.mark.parametrize(
    ("output", "prompt", "expected"),
    [
        ([200, 21], [1], 0),  # still reasoning
        ([200, 21, 201], [1], 0),  # end marker is not content
        ([200, 21, 201, 22, 23], [1], 2),
        ([22, 23], [200, 5, 201], 2),  # prompt already closed reasoning
    ],
)
def test_delegating_parser_content_token_count(output, prompt, expected):
    from vllm.parser.abstract_parser import DelegatingParser
    from vllm.reasoning.deepseek_r1_reasoning_parser import (
        DeepSeekR1ReasoningParser,
    )

    class _Parser(DelegatingParser):
        reasoning_parser_cls = DeepSeekR1ReasoningParser

    parser = _Parser(make_mock_tokenizer({"<think>": 200, "</think>": 201}))
    assert parser.content_token_count(output, prompt) == expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [([200, 21], 0), ([200, 21, 201], 0), ([21, 201, 22, 201, 23], 3)],
)
def test_engine_parser_content_token_count(output, expected):
    tokenizer = make_mock_tokenizer({"<think>": 200, "</think>": 201})
    assert Qwen3Parser(tokenizer, tools=[]).content_token_count(output) == expected
