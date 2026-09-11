#!/usr/bin/env python3
"""Canaries and exact-token measurements against a running GLM-5.3 profile.

Boot with ``slimserve glm53f-nvfp4-{4,8} --serve -y --port 8400`` first.
This runner never kills processes, and refuses to overwrite its artifacts.
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8400")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=300)
    parser.add_argument(
        "--repeat-source",
        action="store_true",
        help="repeat natural-text source for longer exact prompts",
    )
    parser.add_argument("--wait-seconds", type=int, default=1500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "canaries.json").exists():
        parser.error("canaries.json already exists; choose a new output directory")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            with urllib.request.urlopen(args.base_url + "/health", timeout=5):
                break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError("server did not reach health") from None
            time.sleep(5)

    responses = {}

    def chat(name, content, **kwargs):
        body = dict(
            model=args.model,
            messages=[dict(role="user", content=content)],
            max_tokens=400,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            seed=42,
        )
        body.update(kwargs)
        request = urllib.request.Request(
            args.base_url + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            responses[name] = json.load(response)
        # Preserve the failing response before asserting its contents.
        (args.out / "canaries.json").write_text(json.dumps(responses, indent=2))
        return responses[name]["choices"][0]["message"]

    message = chat(
        "text",
        "What is the capital of France? Answer in one short sentence.",
        max_tokens=300,
    )
    assert "paris" in (message.get("content") or "").lower(), message
    assert message.get("reasoning") or message.get("reasoning_content"), message
    picture = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(picture).rectangle([48, 48, 208, 208], fill="red")
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    message = chat(
        "image",
        [
            dict(
                type="image_url", image_url=dict(url=f"data:image/png;base64,{encoded}")
            ),
            dict(
                type="text", text="What shape and color is in this image? One sentence."
            ),
        ],
    )
    content = (message.get("content") or "").lower()
    assert "red" in content and "square" in content, message
    message = chat(
        "tool",
        "Use the weather tool to check the weather in Paris.",
        tools=[
            dict(
                type="function",
                function=dict(
                    name="get_weather",
                    description="Get current weather for a city",
                    parameters=dict(
                        type="object",
                        properties=dict(city=dict(type="string")),
                        required=["city"],
                    ),
                ),
            )
        ],
        tool_choice="auto",
    )
    calls = message.get("tool_calls") or []
    assert any(
        call["function"]["name"] == "get_weather"
        and "paris" in json.loads(call["function"]["arguments"]).get("city", "").lower()
        for call in calls
    ), message
    print("Text, reasoning, image and tool canaries PASS", flush=True)

    repo = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo / "benchmarks/benchmark_dsv4_exact.py"),
        "--model",
        args.tokenizer,
        "--served-model-name",
        args.model,
        "--source",
        str(repo / "benchmarks/sonnet.txt"),
        "--url",
        args.base_url + "/v1/completions",
        "--input-tokens",
        str(args.input_tokens),
        "--allow-no-spec",
        "--temperature",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
    ]
    if args.repeat_source:
        command.append("--repeat-source")
    for concurrency in args.concurrency:
        for repeat in range(args.repeats + 1):
            warmup = repeat == 0
            tag = (
                f"warmup_c{concurrency}"
                if warmup
                else f"exact_c{concurrency}_r{repeat}"
            )
            run_command = command + [
                "--concurrency",
                str(concurrency),
                "--output-tokens",
                "32" if warmup else str(args.output_tokens),
                "--seed",
                "1" if warmup else "42",
            ]
            with (
                (args.out / f"{tag}.json").open("x") as out,
                (args.out / f"{tag}.log").open("x") as err,
            ):
                subprocess.run(
                    run_command, stdout=out, stderr=err, check=True, timeout=1800
                )
            result = json.loads((args.out / f"{tag}.json").read_text())
            print(f"{tag}: {result['aggregate_output_tps']:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
