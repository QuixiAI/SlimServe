#!/usr/bin/env python3
"""Live text, image, tool, and small answer canaries for the Affine profile."""

import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

MODEL = "QuixiAI/affine-king-r21-grpo5-s75-vision-NVFP4"


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "model": MODEL,
        "max_tokens": 512,
        "temperature": 0.5,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    checks: dict[str, bool] = {}

    def run(name: str, payload: dict) -> dict:
        result = post(args.url, {**common, **payload})
        (args.output / f"{name}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result["choices"][0]["message"]

    text = run(
        "text",
        {"messages": [{"role": "user", "content": "Reply with exactly READY."}]},
    )
    checks["text"] = "READY" in (text.get("content") or "")

    picture = Image.new("RGB", (64, 64), "blue")
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    image = run(
        "image",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is the square?"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]
        },
    )
    checks["image"] = "blue" in (image.get("content") or "").lower()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    call = run(
        "tool",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use get_weather to check the weather in Paris.",
                }
            ],
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    tool_calls = call.get("tool_calls") or []
    checks["tool"] = bool(
        tool_calls
        and tool_calls[0]["function"]["name"] == "get_weather"
        and "Paris" in tool_calls[0]["function"]["arguments"]
    )
    if tool_calls:
        followup = run(
            "tool_followup",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use get_weather to check the weather in Paris.",
                    },
                    call,
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0]["id"],
                        "content": '{"city":"Paris","temperature_c":18,"condition":"sunny"}',
                    },
                ],
                "tools": tools,
            },
        )
        checks["tool_followup"] = "18" in (followup.get("content") or "")
    else:
        checks["tool_followup"] = False

    for name, question, answer in [
        ("math_1", "What is 7 times 8? Answer with the number.", "56"),
        ("math_2", "What is 13 plus 29? Answer with the number.", "42"),
        ("math_3", "What is 5 factorial? Answer with the number.", "120"),
        ("fact", "What is the capital of France? One word.", "Paris"),
    ]:
        message = run(name, {"messages": [{"role": "user", "content": question}]})
        checks[name] = answer.lower() in (message.get("content") or "").lower()

    (args.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
