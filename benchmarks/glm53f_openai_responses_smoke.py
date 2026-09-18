#!/usr/bin/env python3
"""Extended live OpenAI Responses/structured-output smoke tests for GLM-5.3-Flash."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import glm53f_tool_matrix as base


def response_function_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }


WEATHER = response_function_tool("get_weather", "Return weather for a city.")
TIME = response_function_tool("get_time", "Return local time for a city.")
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "integer"},
        "label": {"type": "string"},
    },
    "required": ["code", "label"],
    "additionalProperties": False,
}


def response_payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "temperature": 0,
        "max_output_tokens": 128,
    }


def function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in response.get("output", []) if item.get("type") == "function_call"
    ]


def one_call(response: dict[str, Any], name: str, city: str) -> bool:
    calls = function_calls(response)
    if len(calls) != 1 or calls[0].get("name") != name:
        return False
    try:
        arguments = json.loads(calls[0].get("arguments", ""))
    except (TypeError, json.JSONDecodeError):
        return False
    return response.get("status") == "completed" and arguments == {"city": city}


def two_calls(response: dict[str, Any]) -> bool:
    calls = function_calls(response)
    if len(calls) != 2 or response.get("status") != "completed":
        return False
    parsed = []
    try:
        for call in calls:
            parsed.append((call.get("name"), json.loads(call.get("arguments", ""))))
    except (TypeError, json.JSONDecodeError):
        return False
    return sorted(parsed) == sorted(
        [
            ("get_weather", {"city": "Reykjavik"}),
            ("get_time", {"city": "Tokyo"}),
        ]
    )


def exact_text(response: dict[str, Any], expected: str) -> bool:
    return (
        response.get("status") == "completed"
        and not function_calls(response)
        and base.response_text(response).strip() == expected
    )


def exact_text_without_any_tool(response: dict[str, Any], expected: str) -> bool:
    return (
        exact_text(response, expected)
        and not base.response_custom_calls(response)
    )


def has_reasoning(response: dict[str, Any]) -> bool:
    return any(item.get("type") == "reasoning" for item in response.get("output", []))


def structured_text(response: dict[str, Any]) -> bool:
    try:
        value = json.loads(base.response_text(response))
    except (TypeError, json.JSONDecodeError):
        return False
    return response.get("status") == "completed" and value == {
        "code": 7,
        "label": "ok",
    }


def structured_chat(response: dict[str, Any]) -> bool:
    try:
        value = json.loads(base.chat_text(response))
    except (TypeError, json.JSONDecodeError):
        return False
    return response.get("choices", [{}])[0].get("finish_reason") == "stop" and value == {
        "code": 7,
        "label": "ok",
    }


def nonstream_cases(model: str):
    call_weather = "Call get_weather exactly once with city Reykjavik, then stop."
    function_required = response_payload(model, call_weather)
    function_required.update(
        {
            "tools": [WEATHER],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
        }
    )

    # Match OpenAI Responses clients such as Evalhub: required tool choice,
    # with parallel_tool_calls and max_tool_calls left at protocol defaults.
    function_required_defaults = response_payload(model, call_weather)
    function_required_defaults.update(
        {
            "tools": [WEATHER],
            "tool_choice": "required",
        }
    )

    function_named_thinking = copy.deepcopy(function_required)
    function_named_thinking.update(
        {
            "tool_choice": {"type": "function", "name": "get_weather"},
            "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
        }
    )

    function_required_thinking = copy.deepcopy(function_required)
    function_required_thinking["chat_template_kwargs"] = {
        "thinking": True,
        "enable_thinking": True,
    }

    function_auto_call = copy.deepcopy(function_required)
    function_auto_call["tool_choice"] = "auto"

    function_auto_text = response_payload(model, "Do not call a tool. Reply exactly OPTIONAL_TEXT_OK")
    function_auto_text.update(
        {
            "tools": [WEATHER],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
        }
    )

    function_none = response_payload(model, "Do not call a tool. Reply exactly RESPONSES_NONE_OK")
    function_none.update({"tools": [WEATHER], "tool_choice": "none"})

    allowed_required = response_payload(
        model, "Call get_time exactly once with city Tokyo, then stop."
    )
    allowed_required.update(
        {
            "tools": [WEATHER, TIME],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "get_time"}],
            },
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
        }
    )

    allowed_auto_call = copy.deepcopy(allowed_required)
    allowed_auto_call["tool_choice"]["mode"] = "auto"

    allowed_auto_text = response_payload(
        model, "Do not call a tool. Reply exactly ALLOWED_AUTO_TEXT_OK"
    )
    allowed_auto_text.update(
        {
            "tools": [WEATHER, TIME],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "get_time"}],
            },
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
        }
    )

    function_parallel = response_payload(
        model,
        "Call get_weather for Reykjavik and get_time for Tokyo, exactly once each, then stop.",
    )
    function_parallel.update(
        {
            "tools": [WEATHER, TIME],
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "max_tool_calls": 2,
            "max_output_tokens": 256,
        }
    )

    # Standard Responses shape for function tools: parallel calls are allowed
    # and max_tool_calls is omitted. OpenAI documents max_tool_calls for
    # built-in tools, so native function calls must terminate without it.
    function_parallel_defaults = copy.deepcopy(function_parallel)
    function_parallel_defaults.pop("max_tool_calls")

    responses_schema = response_payload(
        model, 'Return JSON with code 7 and label "ok". Do not add other fields.'
    )
    responses_schema.update(
        {
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "smoke_result",
                    "schema": RESULT_SCHEMA,
                    "strict": True,
                }
            },
        }
    )

    chat_schema = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": 'Return JSON with code 7 and label "ok". Do not add other fields.',
            }
        ],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "smoke_result",
                "schema": RESULT_SCHEMA,
                "strict": True,
            },
        },
    }

    patch_prompt = (
        "greeting.txt contains exactly `Hello, world!\\n`. Call apply_patch "
        "exactly once to change it to exactly `Hello, SlimServe!\\n`, then stop."
    )
    custom_required = base.custom_payload(model, patch_prompt)
    custom_required.update(
        {"tool_choice": "required", "parallel_tool_calls": False, "max_tool_calls": 1}
    )

    custom_named_thinking = copy.deepcopy(custom_required)
    custom_named_thinking.update(
        {
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
        }
    )

    custom_auto_call = copy.deepcopy(custom_required)
    custom_auto_call["tool_choice"] = "auto"

    custom_auto_text = base.custom_payload(
        model, "Do not call a tool. Reply exactly CUSTOM_AUTO_TEXT_OK"
    )
    custom_auto_text.update(
        {"tool_choice": "auto", "parallel_tool_calls": False, "max_tool_calls": 1}
    )

    custom_none = base.custom_payload(
        model, "Do not call a tool. Reply exactly CUSTOM_NONE_OK"
    )
    custom_none.update({"tool_choice": "none", "max_output_tokens": 64})

    allowed_custom_required = copy.deepcopy(custom_required)
    allowed_custom_required["tool_choice"] = {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "custom", "name": "apply_patch"}],
    }

    allowed_custom_auto_call = copy.deepcopy(allowed_custom_required)
    allowed_custom_auto_call["tool_choice"]["mode"] = "auto"

    allowed_custom_auto_text = copy.deepcopy(custom_auto_text)
    allowed_custom_auto_text["input"] = (
        "Do not call a tool. Reply exactly ALLOWED_CUSTOM_AUTO_TEXT_OK"
    )
    allowed_custom_auto_text["tool_choice"] = {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "custom", "name": "apply_patch"}],
    }

    return [
        ("responses_function_required", "/v1/responses", function_required, lambda r: one_call(r, "get_weather", "Reykjavik")),
        ("responses_function_required_defaults", "/v1/responses", function_required_defaults, lambda r: one_call(r, "get_weather", "Reykjavik")),
        ("responses_function_named_thinking", "/v1/responses", function_named_thinking, lambda r: one_call(r, "get_weather", "Reykjavik")),
        ("responses_function_required_thinking", "/v1/responses", function_required_thinking, lambda r: one_call(r, "get_weather", "Reykjavik") and has_reasoning(r)),
        ("responses_function_auto_call", "/v1/responses", function_auto_call, lambda r: one_call(r, "get_weather", "Reykjavik")),
        ("responses_function_auto_text", "/v1/responses", function_auto_text, lambda r: exact_text(r, "OPTIONAL_TEXT_OK")),
        ("responses_function_none", "/v1/responses", function_none, lambda r: exact_text(r, "RESPONSES_NONE_OK")),
        ("responses_allowed_tools_required", "/v1/responses", allowed_required, lambda r: one_call(r, "get_time", "Tokyo")),
        ("responses_allowed_tools_auto_call", "/v1/responses", allowed_auto_call, lambda r: one_call(r, "get_time", "Tokyo")),
        ("responses_allowed_tools_auto_text", "/v1/responses", allowed_auto_text, lambda r: exact_text(r, "ALLOWED_AUTO_TEXT_OK")),
        ("responses_function_parallel_required", "/v1/responses", function_parallel, two_calls),
        ("responses_function_parallel_required_defaults", "/v1/responses", function_parallel_defaults, two_calls),
        ("responses_custom_lark_required", "/v1/responses", custom_required, base.one_patch_call),
        ("responses_custom_lark_named_thinking_policy", "/v1/responses", custom_named_thinking, base.one_patch_call),
        ("responses_custom_lark_auto_call", "/v1/responses", custom_auto_call, base.one_patch_call),
        ("responses_custom_lark_auto_text", "/v1/responses", custom_auto_text, lambda r: exact_text_without_any_tool(r, "CUSTOM_AUTO_TEXT_OK")),
        ("responses_custom_lark_none", "/v1/responses", custom_none, lambda r: exact_text_without_any_tool(r, "CUSTOM_NONE_OK")),
        ("responses_allowed_custom_lark_required", "/v1/responses", allowed_custom_required, base.one_patch_call),
        ("responses_allowed_custom_lark_auto_call", "/v1/responses", allowed_custom_auto_call, base.one_patch_call),
        ("responses_allowed_custom_lark_auto_text", "/v1/responses", allowed_custom_auto_text, lambda r: exact_text_without_any_tool(r, "ALLOWED_CUSTOM_AUTO_TEXT_OK")),
        ("responses_json_schema", "/v1/responses", responses_schema, structured_text),
        ("chat_json_schema", "/v1/chat/completions", chat_schema, structured_chat),
    ]


def stream_events(
    base_url: str,
    payload: dict[str, Any],
    api_key: str | None,
) -> list[dict[str, Any]]:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/responses",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    events = []
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                events.append(json.loads(data))
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    return events


def completed_response(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [event for event in events if event.get("type") == "response.completed"]
    return completed[-1].get("response") if completed else None


def streaming_cases(model: str):
    function_payload = response_payload(
        model, "Call get_weather exactly once with city Reykjavik, then stop."
    )
    function_payload.update(
        {
            "tools": [WEATHER],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
            "stream": True,
        }
    )

    custom_payload = base.custom_payload(
        model,
        "greeting.txt contains exactly `Hello, world!\\n`. Call apply_patch exactly once "
        "to change it to exactly `Hello, SlimServe!\\n`, then stop.",
    )
    custom_payload.update(
        {
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
            "stream": True,
        }
    )

    allowed_function_payload = copy.deepcopy(function_payload)
    allowed_function_payload["tool_choice"] = {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "get_weather"}],
    }

    allowed_custom_payload = copy.deepcopy(custom_payload)
    allowed_custom_payload["tool_choice"] = {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "custom", "name": "apply_patch"}],
    }

    schema_payload = response_payload(
        model, 'Return JSON with code 7 and label "ok". Do not add other fields.'
    )
    schema_payload.update(
        {
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "smoke_result",
                    "schema": RESULT_SCHEMA,
                    "strict": True,
                }
            },
            "stream": True,
        }
    )
    return [
        (
            "responses_function_stream",
            function_payload,
            "response.function_call_arguments.delta",
            lambda response: response is not None and one_call(response, "get_weather", "Reykjavik"),
        ),
        (
            "responses_custom_lark_stream",
            custom_payload,
            "response.custom_tool_call_input.delta",
            lambda response: response is not None and base.one_patch_call(response),
        ),
        (
            "responses_allowed_function_stream",
            allowed_function_payload,
            "response.function_call_arguments.delta",
            lambda response: response is not None and one_call(response, "get_weather", "Reykjavik"),
        ),
        (
            "responses_allowed_custom_lark_stream",
            allowed_custom_payload,
            "response.custom_tool_call_input.delta",
            lambda response: response is not None and base.one_patch_call(response),
        ),
        (
            "responses_json_schema_stream",
            schema_payload,
            "response.output_text.delta",
            lambda response: response is not None and structured_text(response),
        ),
    ]


def continuation_case(model: str, base_url: str, api_key: str | None) -> bool:
    prompt = (
        "Call get_weather once for Reykjavik. After the tool result, reply exactly "
        "FUNCTION_CONTINUATION_OK."
    )
    first = response_payload(model, prompt)
    first.update(
        {
            "tools": [WEATHER],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
        }
    )
    first_response = base.post(base_url, "/v1/responses", first, api_key)
    calls = function_calls(first_response)
    if len(calls) != 1:
        return False
    call = calls[0]
    second = response_payload(model, "")
    second["input"] = [
        {"role": "user", "content": prompt},
        call,
        {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": "Sunny and 12 C.",
        },
    ]
    second.update({"tools": [WEATHER], "tool_choice": "none"})
    response = base.post(base_url, "/v1/responses", second, api_key)
    return exact_text(response, "FUNCTION_CONTINUATION_OK")


def custom_continuation_case(
    model: str, base_url: str, api_key: str | None
) -> bool:
    prompt = (
        "greeting.txt contains exactly `Hello, world!\\n`. Call apply_patch once "
        "to change it to exactly `Hello, SlimServe!\\n`. After the tool result, "
        "reply exactly CUSTOM_CONTINUATION_OK."
    )
    first = base.custom_payload(model, prompt)
    first.update(
        {"tool_choice": "required", "parallel_tool_calls": False, "max_tool_calls": 1}
    )
    first_response = base.post(base_url, "/v1/responses", first, api_key)
    calls = base.response_custom_calls(first_response)
    if len(calls) != 1:
        return False
    call = calls[0]
    second = response_payload(model, "")
    second["input"] = [
        {"role": "user", "content": prompt},
        call,
        {
            "type": "custom_tool_call_output",
            "call_id": call["call_id"],
            "output": "Patch applied successfully.",
        },
    ]
    second.update({"tools": [base.PATCH_TOOL], "tool_choice": "none"})
    response = base.post(base_url, "/v1/responses", second, api_key)
    return exact_text_without_any_tool(response, "CUSTOM_CONTINUATION_OK")


def invalid_request_cases(model: str):
    invalid_named = response_payload(model, "Call the missing tool.")
    invalid_named.update(
        {
            "tools": [WEATHER],
            "tool_choice": {"type": "function", "name": "missing"},
        }
    )

    undeclared_allowed = response_payload(model, "Call the missing tool.")
    undeclared_allowed.update(
        {
            "tools": [WEATHER],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "missing"}],
            },
        }
    )

    empty_required_allowed = response_payload(model, "Call a tool.")
    empty_required_allowed.update(
        {
            "tools": [WEATHER],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [],
            },
        }
    )

    return [
        ("responses_invalid_named_tool_http_400", invalid_named),
        ("responses_undeclared_allowed_tool_http_400", undeclared_allowed),
        ("responses_empty_required_allowed_set_http_400", empty_required_allowed),
    ]


def post_status(
    base_url: str, payload: dict[str, Any], api_key: str | None
) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/responses",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read().decode(errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GLM-5.3-Flash-FP8")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for name, path, payload, validate in nonstream_cases(args.model):
        started = time.perf_counter()
        try:
            response = base.post(args.base_url, path, payload, args.api_key)
            passed = validate(response)
            error = None if passed else "response failed semantic validation"
        except Exception as exception:
            response = None
            passed = False
            error = str(exception)
        results.append(
            {
                "name": name,
                "passed": passed,
                "seconds": round(time.perf_counter() - started, 4),
                "error": error,
                "request": payload,
                "response": response,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    for name, payload, required_event, validate in streaming_cases(args.model):
        started = time.perf_counter()
        try:
            events = stream_events(args.base_url, payload, args.api_key)
            response = completed_response(events)
            event_types = [event.get("type") for event in events]
            passed = required_event in event_types and validate(response)
            error = None if passed else "stream failed semantic validation"
        except Exception as exception:
            response = None
            event_types = []
            passed = False
            error = str(exception)
        results.append(
            {
                "name": name,
                "passed": passed,
                "seconds": round(time.perf_counter() - started, 4),
                "error": error,
                "request": payload,
                "event_types": event_types,
                "response": response,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    started = time.perf_counter()
    try:
        passed = continuation_case(args.model, args.base_url, args.api_key)
        error = None if passed else "continuation failed semantic validation"
    except Exception as exception:
        passed = False
        error = str(exception)
    results.append(
        {
            "name": "responses_function_continuation",
            "passed": passed,
            "seconds": round(time.perf_counter() - started, 4),
            "error": error,
        }
    )
    print(
        f"{'PASS' if passed else 'FAIL'} responses_function_continuation",
        flush=True,
    )

    started = time.perf_counter()
    try:
        passed = custom_continuation_case(args.model, args.base_url, args.api_key)
        error = None if passed else "custom continuation failed semantic validation"
    except Exception as exception:
        passed = False
        error = str(exception)
    results.append(
        {
            "name": "responses_custom_lark_continuation",
            "passed": passed,
            "seconds": round(time.perf_counter() - started, 4),
            "error": error,
        }
    )
    print(
        f"{'PASS' if passed else 'FAIL'} responses_custom_lark_continuation",
        flush=True,
    )

    for name, payload in invalid_request_cases(args.model):
        started = time.perf_counter()
        try:
            status, body = post_status(args.base_url, payload, args.api_key)
            passed = status == 400
            error = None if passed else f"expected HTTP 400, got HTTP {status}"
        except Exception as exception:
            status = None
            body = None
            passed = False
            error = str(exception)
        results.append(
            {
                "name": name,
                "passed": passed,
                "seconds": round(time.perf_counter() - started, 4),
                "error": error,
                "request": payload,
                "status": status,
                "body": body,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
