"""Benign long-context, multi-turn tool protocol/throughput benchmark.

Run against the real SlimServe profile; records exact API usage and SSE events.
This is a local inventory workflow, not the external Evalhub benchmark.

Sampling follows serving policy: temperature 1.0 / top_p 0.95 / top_k 20,
seeded (request i uses seed + i), thinking left enabled. Raw output defaults
to <repo>/perf/results/<YYYY-MM-DD>/<run-id>/qwen_tool_protocol.json, with a
unique run id.

Timing per round: ``response_created_s`` is the time to the first SSE event
(``response.created``, emitted before any generation); ``first_output_delta_s``
is the time to the first generated delta (reasoning, text or call arguments).
A round that ends ``response.incomplete`` because it hit max_output_tokens is
counted only as ``incomplete_max_output_tokens``: its (possibly cut-off) call
arguments are not validated. ``completed_rounds`` counts rounds that were
neither truncated nor carried a protocol error.
"""

import argparse
import asyncio
import json
import secrets
import time
from datetime import datetime
from pathlib import Path

import httpx

# Served by the qwen38-27b profiles in slimserve/profiles.json.
MODEL = "Qwen3.8-27B"

OUTPUT_DELTA_TYPES = {
    "response.output_text.delta",
    "response.reasoning_text.delta",
    "response.function_call_arguments.delta",
}


def payload(index, args):
    catalog = "\n".join(
        f"Item {i:04d}: category={i % 13}, quantity={i % 29}, "
        f"warehouse=W{i % 4}, inspected=yes, packaging=reusable, "
        f"handling=standard, label=CAT-{i:04d}."
        for i in range(args.records)
    )
    return {
        "model": args.model,
        "instructions": "Review the inventory and use record_summary "
        "to record findings. "
        "Use the exact batch_id supplied. Keep observations factual and concise.",
        "input": [
            {
                "role": "user",
                "content": f"Batch BATCH-{index}.\n{catalog}\n"
                "Record a summary with the batch id, "
                "three observations and two recommendations.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "record_summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "observations": {"type": "array", "items": {"type": "string"}},
                        "recommendations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["batch_id", "observations", "recommendations"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": "required",
        "stream": True,
        "store": False,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed + index,
    }


def evaluate_round(events, index):
    """Classify one streamed round: (final, calls, truncated, errors).

    A round truncated by max_output_tokens is reported only as truncated: its
    call arguments may legitimately be cut off, so they are not validated and
    the round carries no protocol errors.
    """
    terminal = [
        e
        for e in events
        if e.get("type")
        in {"response.completed", "response.failed", "response.incomplete"}
    ]
    final = terminal[-1].get("response", {}) if terminal else {}
    calls = [o for o in final.get("output", []) if o.get("type") == "function_call"]
    truncated = (
        final.get("status") == "incomplete"
        and (final.get("incomplete_details") or {}).get("reason")
        == "max_output_tokens"
    )
    if truncated:
        # Not a protocol fault: the budget ran out. The round cannot be
        # replayed, so the conversation stops here.
        return final, calls, True, []
    errors = []
    if final.get("status") != "completed":
        errors.append("missing successful terminal response")
    elif not calls:
        errors.append("required request returned no call")
    for call in calls:
        try:
            obj = json.loads(call["arguments"])
            if not isinstance(obj, dict):
                errors.append("arguments not object")
            elif obj.get("batch_id") != f"BATCH-{index}":
                errors.append("batch_id missing or corrupted")
        except ValueError:
            errors.append("malformed arguments")
    for event in events:
        if event.get("type") == "response.function_call_arguments.done":
            try:
                json.loads(event["arguments"])
            except ValueError:
                errors.append("malformed done arguments")
    assembled = {}
    for event in events:
        if event.get("type") == "response.function_call_arguments.delta":
            item = event["item_id"]
            assembled[item] = assembled.get(item, "") + event["delta"]
        elif event.get("type") == "response.function_call_arguments.done":
            if assembled.get(event["item_id"], "") != event["arguments"]:
                errors.append("stream delta/done mismatch")
    return final, calls, False, errors


async def run_one(client, url, index, args):
    body = payload(index, args)
    rounds = []
    for turn in range(args.turns):
        started = time.perf_counter()
        events = []
        first = None
        first_delta = None
        async with client.stream("POST", url + "/v1/responses", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if first is None:
                    first = time.perf_counter() - started
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    if line[6:] != "[DONE]":
                        raise
                    continue
                if first_delta is None and event.get("type") in OUTPUT_DELTA_TYPES:
                    first_delta = time.perf_counter() - started
                events.append(event)
        elapsed = time.perf_counter() - started
        final, calls, truncated, errors = evaluate_round(events, index)
        rounds.append(
            {
                "turn": turn,
                "elapsed_s": elapsed,
                "response_created_s": first,
                "first_output_delta_s": first_delta,
                "incomplete_max_output_tokens": truncated,
                "usage": final.get("usage", {}),
                "errors": errors,
                "events": events,
            }
        )
        if errors or truncated:
            break
        body["input"].extend(final.get("output", []))
        body["input"].extend(
            {
                "type": "function_call_output",
                "call_id": c["call_id"],
                "output": "Recorded successfully.",
            }
            for c in calls
        )
        body["input"].append(
            {
                "role": "user",
                "content": "Review the same batch for storage "
                "and labeling concerns, then record the updated summary.",
            }
        )
    return {"index": index, "rounds": rounds}


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_output(now=None):
    """perf/results/<date>/<run-id>/ under the repo root, whatever the CWD.

    The run id carries microseconds and a random suffix so concurrent or
    back-to-back runs never share (and overwrite) a directory.
    """
    now = now or datetime.now()
    run_id = now.strftime("qwen-tool-protocol-%H%M%S-%f-") + secrets.token_hex(3)
    return (
        REPO_ROOT
        / "perf"
        / "results"
        / now.strftime("%Y-%m-%d")
        / run_id
        / "qwen_tool_protocol.json"
    )


def summarize(rows, elapsed):
    """Round counts: a round is completed only if it was neither truncated
    nor carried a protocol error; the three counts partition ``rounds``."""
    rounds = [r for row in rows for r in row.get("rounds", [])]
    tokens = sum(r["usage"].get("output_tokens", 0) for r in rounds)
    errored = sum(bool(r["errors"]) for r in rounds)
    truncated = sum(r["incomplete_max_output_tokens"] for r in rounds)
    return {
        "elapsed_s": elapsed,
        "output_tokens": tokens,
        "aggregate_output_tokens_per_s": tokens / elapsed if elapsed else 0.0,
        "rounds": len(rounds),
        "completed_rounds": sum(
            not r["errors"] and not r["incomplete_max_output_tokens"]
            for r in rounds
        ),
        "protocol_errors": errored + sum("error" in row for row in rows),
        "incomplete_max_output_tokens": truncated,
    }


async def main(args):
    started_at_unix = time.time()
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=240) as client:
        results = await asyncio.gather(
            *(run_one(client, args.url, i, args) for i in range(args.concurrency)),
            return_exceptions=True,
        )
    rows = [x if isinstance(x, dict) else {"error": str(x)} for x in results]
    elapsed = time.perf_counter() - started
    summary = summarize(rows, elapsed)
    summary.update(
        {
            "started_at_unix": started_at_unix,
            "model": args.model,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
            },
            "max_output_tokens": args.max_output_tokens,
            "concurrency": args.concurrency,
            "turns": args.turns,
            "records": args.records,
        }
    )
    output = Path(args.output) if args.output else default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"raw: {output}", flush=True)
    if (
        summary["protocol_errors"]
        or summary["incomplete_max_output_tokens"]
        or summary["completed_rounds"] != args.concurrency * args.turns
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--records", type=int, default=240)
    parser.add_argument("--max-output-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        default=None,
        help="raw JSON path (default: perf/results/<YYYY-MM-DD>/<run-id>/"
        "qwen_tool_protocol.json)",
    )
    asyncio.run(main(parser.parse_args()))
