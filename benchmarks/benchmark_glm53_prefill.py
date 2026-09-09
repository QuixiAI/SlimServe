#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold-prefix exact-ID prefill measurements against an owned SlimServe server.

Start its real profile with --request-metrics. Every request has a unique cache
salt and must report zero cached prompt tokens. Client TTFT includes transport
and frontend work. Engine TTFT is scheduled-to-first-token, NOT pure GPU prefill.
All warmups, repetitions, streaming events and failures are retained.
"""

import argparse
import hashlib
import json
import math
import secrets
import statistics
import time
import urllib.request
from pathlib import Path


def summarize_response(row):
    body = row["request"]
    usage = row.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if type(cached) is not int or cached != 0:
        raise ValueError(f"cold-prefix proof requires cached_tokens=0, got {cached!r}")
    if usage.get("prompt_tokens") != len(body["prompt"]):
        raise ValueError("server changed exact prompt-token count")
    if (
        usage.get("completion_tokens") != body["max_tokens"]
        or len(row["token_ids"]) != body["max_tokens"]
    ):
        raise ValueError("incomplete or inconsistent generated-token count")
    if "\ufffd" in row["text"]:
        raise ValueError("replacement character in generated text")
    metrics = row.get("metrics") or {}
    engine_ms = metrics.get("time_to_first_token_ms")
    first = row.get("first_token_seconds")
    if any(
        not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
        for v in (engine_ms, first)
    ):
        raise ValueError("missing/invalid client or engine TTFT; use --request-metrics")
    queue_ms = metrics.get("queue_time_ms")
    if (
        not isinstance(queue_ms, (float, int))
        or not math.isfinite(queue_ms)
        or queue_ms < 0
    ):
        raise ValueError("missing/invalid engine queue time")
    return {
        "prompt_tokens": len(body["prompt"]),
        "cached_tokens": cached,
        "client_ttft_ms": first * 1000,
        "engine_scheduled_to_first_token_ms": engine_ms,
        "engine_queue_ms": queue_ms,
        "effective_input_tokens_per_client_ttft_second": len(body["prompt"]) / first,
        "effective_input_tokens_per_engine_ttft_second": len(body["prompt"])
        * 1000
        / engine_ms,
        "complete_request_ms": row["wall_seconds"] * 1000,
    }


def request(url, model, ids, output_tokens, output, salt):
    body = {
        "model": model,
        "prompt": ids,
        "max_tokens": output_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    row = {
        "status": "running",
        "request": body,
        "events": [],
        "token_ids": [],
        "text": "",
    }
    with output.open("x") as stream:
        json.dump(row, stream, indent=2)
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                seconds = time.perf_counter() - started
                if raw == b"[DONE]":
                    row["done"] = True
                    break
                event = json.loads(raw)
                row["events"].append({"seconds": seconds, "data": event})
                if event.get("usage"):
                    row["usage"] = event["usage"]
                if event.get("metrics"):
                    row["metrics"] = event["metrics"]
                for choice in event.get("choices", []):
                    tokens = choice.get("token_ids") or []
                    if tokens:
                        row.setdefault("first_token_seconds", seconds)
                        row["token_ids"].extend(tokens)
                    row["text"] += choice.get("text", "")
        row["wall_seconds"] = time.perf_counter() - started
        if not row.get("done"):
            raise ValueError("missing end-of-stream marker")
        row["summary"] = summarize_response(row)
        row["status"] = "complete"
    except Exception as error:
        row["status"] = "failed"
        row["error"] = repr(error)
        raise
    finally:
        output.write_text(json.dumps(row, indent=2) + "\n")
    return row["summary"]


def run(args, tokenizer):
    if min(args.repeats, args.warmups, args.output_tokens, *args.contexts) < 1:
        raise ValueError(
            "positive repeats, warmups, output tokens and contexts required"
        )
    source = args.source.read_bytes()
    ids = tokenizer.encode(source.decode(), add_special_tokens=False)
    if len(ids) < max(args.contexts):
        raise ValueError("source too short for the largest exact-ID prompt")
    args.output.mkdir(parents=True, exist_ok=False)
    record = args.output / "summary.json"
    result = {
        "status": "running",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_tokens": len(ids),
        "method": __doc__,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "requests": [],
    }
    used_salts = set()

    def save():
        record.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for context in args.contexts:
            # Same token IDs across every cold warmup/repetition. Salt alone
            # changes cache identity, not model input or position arithmetic.
            prompt = ids[:context]
            for warmup, count in ((True, args.warmups), (False, args.repeats)):
                for rep in range(count):
                    salt = secrets.token_hex(32)
                    if salt in used_salts:
                        raise ValueError("duplicate cache salt")
                    used_salts.add(salt)
                    kind = "warmup" if warmup else "repeat"
                    path = args.output / f"{kind}-{rep + 1}-ctx{context}.json"
                    row = {
                        "context": context,
                        "warmup": warmup,
                        "repeat": rep + 1,
                        "path": str(path),
                        "status": "running",
                    }
                    result["requests"].append(row)
                    save()
                    try:
                        row["summary"] = request(
                            args.url, args.model, prompt, args.output_tokens, path, salt
                        )
                        row["status"] = "complete"
                    except Exception as error:
                        row.update(status="failed", error=repr(error))
                        raise
                    save()
                    print(json.dumps(row), flush=True)
        result["aggregates"] = {}
        for context in args.contexts:
            rows = [
                r["summary"]
                for r in result["requests"]
                if r["context"] == context and not r["warmup"]
            ]
            result["aggregates"][str(context)] = {
                key: {
                    "median": statistics.median(r[key] for r in rows),
                    "min": min(r[key] for r in rows),
                    "max": max(r[key] for r in rows),
                }
                for key in rows[0]
            }
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    return result


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=8)
    args = parser.parse_args()
    run(args, AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True))


if __name__ == "__main__":
    main()
