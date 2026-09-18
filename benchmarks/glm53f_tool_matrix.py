#!/usr/bin/env python3
"""Live OpenAI-compatible tool matrix for GLM-5.3-Flash.

Exercises Chat Completions function tools and Responses custom Lark tools.
The server under test may use speculative decoding; acceptance/throughput is
reported separately by vLLM's /metrics endpoint and benchmark harnesses.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


APPLY_PATCH_LARK = """start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?
hunk: update_hunk
update_hunk: "*** Update File: " filename LF change
filename: /(.+)/
change: change_context change_line+
change_context: "@@" LF
change_line: ("+" | "-" | " ") /(.*)/ LF
%import common.LF
"""


def function_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


WEATHER_TOOL = function_tool("get_weather", "Return the weather for a city.")
TIME_TOOL = function_tool("get_time", "Return the local time for a city.")
PATCH_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Apply one patch directly, without JSON, and then stop.",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": APPLY_PATCH_LARK,
    },
}


def post(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def chat_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices") or []
    return [] if not choices else choices[0].get("message", {}).get("tool_calls") or []


def chat_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    return "" if not choices else choices[0].get("message", {}).get("content") or ""


def response_custom_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in response.get("output", []) if item.get("type") == "custom_tool_call"]


def response_text(response: dict[str, Any]) -> str:
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def one_native_call(response: dict[str, Any]) -> bool:
    calls = chat_calls(response)
    if len(calls) != 1 or calls[0].get("function", {}).get("name") != "get_weather":
        return False
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return args == {"city": "Reykjavik"}


def two_native_calls(response: dict[str, Any]) -> bool:
    calls = chat_calls(response)
    if len(calls) != 2:
        return False
    names = [call.get("function", {}).get("name") for call in calls]
    return sorted(names) == ["get_time", "get_weather"]


def one_patch_call(response: dict[str, Any]) -> bool:
    calls = response_custom_calls(response)
    if len(calls) != 1 or response.get("status") != "completed":
        return False
    patch = calls[0].get("input", "")
    return (
        calls[0].get("name") == "apply_patch"
        and patch.startswith("*** Begin Patch\n")
        and "*** Update File: greeting.txt\n" in patch
        and "-Hello, world!\n" in patch
        and "+Hello, SlimServe!\n" in patch
        and patch.rstrip().endswith("*** End Patch")
    )


def native_payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [WEATHER_TOOL],
        "temperature": 0,
        "max_tokens": 256,
    }


def custom_payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "tools": [PATCH_TOOL],
        "temperature": 0,
        "max_output_tokens": 128,
    }


def cases(model: str) -> list[tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], bool]]]:
    call_weather = "Call get_weather exactly once with city Reykjavik, then stop."
    patch_file = (
        "greeting.txt contains exactly `Hello, world!\\n`. Call apply_patch exactly "
        "once to change it to exactly `Hello, SlimServe!\\n`, then stop."
    )

    native_required = native_payload(model, call_weather)
    native_required.update({"tool_choice": "required", "parallel_tool_calls": False})

    native_named_thinking = native_payload(model, call_weather)
    native_named_thinking.update(
        {
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "parallel_tool_calls": False,
            "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
        }
    )

    native_auto = native_payload(model, call_weather)
    native_auto.update({"tool_choice": "auto", "parallel_tool_calls": False})

    native_none = native_payload(model, "Do not call a tool. Reply exactly NATIVE_NONE_OK")
    native_none.update({"tool_choice": "none", "max_tokens": 32})

    native_parallel = native_payload(
        model,
        "Call get_weather for Reykjavik and get_time for Tokyo, exactly once each, then stop.",
    )
    native_parallel.update(
        {
            "tools": [WEATHER_TOOL, TIME_TOOL],
            "tool_choice": "required",
            "parallel_tool_calls": True,
        }
    )

    custom_required = custom_payload(model, patch_file)
    custom_required.update(
        {"tool_choice": "required", "parallel_tool_calls": False, "max_tool_calls": 1}
    )

    custom_named_thinking = custom_payload(model, patch_file)
    custom_named_thinking.update(
        {
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
            # GLM-5.3-Flash custom tools deliberately override this unsafe
            # combination to non-thinking in the compatibility policy.
            "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
        }
    )

    custom_auto = custom_payload(model, patch_file)
    custom_auto.update(
        {"tool_choice": "auto", "parallel_tool_calls": False, "max_tool_calls": 1}
    )

    custom_none = custom_payload(model, "Do not call a tool. Reply exactly CUSTOM_NONE_OK")
    custom_none.update({"tool_choice": "none", "max_output_tokens": 32})

    return [
        ("native_required", "/v1/chat/completions", native_required, one_native_call),
        (
            "native_named_thinking",
            "/v1/chat/completions",
            native_named_thinking,
            one_native_call,
        ),
        ("native_auto", "/v1/chat/completions", native_auto, one_native_call),
        (
            "native_none",
            "/v1/chat/completions",
            native_none,
            lambda response: not chat_calls(response)
            and chat_text(response).strip() == "NATIVE_NONE_OK",
        ),
        (
            "native_parallel",
            "/v1/chat/completions",
            native_parallel,
            two_native_calls,
        ),
        ("custom_required", "/v1/responses", custom_required, one_patch_call),
        (
            "custom_named_thinking_policy",
            "/v1/responses",
            custom_named_thinking,
            one_patch_call,
        ),
        ("custom_auto", "/v1/responses", custom_auto, one_patch_call),
        (
            "custom_none",
            "/v1/responses",
            custom_none,
            lambda response: not response_custom_calls(response)
            and response_text(response).strip() == "CUSTOM_NONE_OK",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GLM-5.3-Flash-FP8")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="Run only this named case; repeat to select multiple cases.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    selected_cases = cases(args.model)
    if args.case_names:
        known_names = {case[0] for case in selected_cases}
        unknown_names = sorted(set(args.case_names) - known_names)
        if unknown_names:
            parser.error(
                "unknown --case value(s): " + ", ".join(unknown_names)
            )
        requested_names = set(args.case_names)
        selected_cases = [
            case for case in selected_cases if case[0] in requested_names
        ]

    jobs = [
        (repetition, case_index, case)
        for repetition in range(args.repetitions)
        for case_index, case in enumerate(selected_cases)
    ]

    def run_case(job):
        repetition, case_index, case = job
        name, path, payload, validate = case
        started = time.perf_counter()
        try:
            response = post(
                args.base_url,
                path,
                copy.deepcopy(payload),
                api_key=args.api_key,
            )
            passed = validate(response)
            error = None if passed else "response failed semantic validation"
        except Exception as exception:  # Keep the full matrix running.
            response = None
            passed = False
            error = str(exception)
        result = {
            "name": name,
            "repetition": repetition,
            "case_index": case_index,
            "passed": passed,
            "seconds": round(time.perf_counter() - started, 4),
            "error": error,
            "request": payload,
            "response": response,
        }
        print(
            f"{'PASS' if passed else 'FAIL'} {name} "
            f"rep={repetition} seconds={result['seconds']}",
            flush=True,
        )
        return result

    if args.concurrency == 1:
        results = [run_case(job) for job in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            results = list(executor.map(run_case, jobs))

    results.sort(key=lambda result: (result["repetition"], result["case_index"]))

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "repetitions": args.repetitions,
        "cases": [case[0] for case in selected_cases],
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
