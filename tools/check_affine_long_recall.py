#!/usr/bin/env python3
"""Near-limit semantic recall canary against a live Affine chat endpoint."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from vllm.tokenizers.registry import get_tokenizer

MODEL = "QuixiAI/affine-king-r21-grpo5-s75-vision-NVFP4"
LOCAL = "/home/eric/models/affine-king-r21-grpo5-s75-vision-NVFP4"
SECRET = "QUARTZ-7319"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = get_tokenizer(LOCAL)
    head = f"Remember this secret passcode: {SECRET}.\n"
    filler = "The blue lake is quiet and the wind is gentle. "
    tail = "\nWhat was the secret passcode at the beginning? Reply with only the passcode."
    filler_tokens = len(tokenizer.encode(filler, add_special_tokens=False))
    # Adjacent copies share tokenizer boundaries, so the whole prompt is
    # about 8% shorter than a per-copy estimate.
    repeats = 278000 // filler_tokens
    prompt = head + filler * repeats + tail
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if not 250000 <= prompt_tokens <= 260000:
        raise SystemExit(f"Unexpected prompt token count: {prompt_tokens}")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.5,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    elapsed = time.monotonic() - start
    content = result["choices"][0]["message"].get("content") or ""
    summary = {
        "prompt_tokens": prompt_tokens,
        "usage": result.get("usage"),
        "elapsed_seconds": elapsed,
        "content": content,
        "pass": SECRET in content,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
