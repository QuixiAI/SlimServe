"""Benign long-context, multi-turn tool protocol/throughput benchmark.

Run against the real SlimServe profile; records exact API usage and SSE events.
This is a local inventory workflow, not the external Evalhub benchmark.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

MODEL = "Qwen3.8-27B-Uncensored-FP8"


def payload(index, records):
    catalog = "\n".join(
        f"Item {i:04d}: category={i % 13}, quantity={i % 29}, "
        f"warehouse=W{i % 4}, inspected=yes, packaging=reusable, "
        f"handling=standard, label=CAT-{i:04d}."
        for i in range(records)
    )
    return {
        "model": MODEL,
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
        "max_output_tokens": 3072,
        "temperature": 0,
        "seed": 17,
    }


async def run_one(client, url, index, args):
    body = payload(index, args.records)
    rounds = []
    for turn in range(args.turns):
        started = time.perf_counter()
        events = []
        first = None
        async with client.stream("POST", url + "/v1/responses", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if first is None:
                    first = time.perf_counter() - started
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    if line[6:] != "[DONE]":
                        raise
        elapsed = time.perf_counter() - started
        terminal = [
            e
            for e in events
            if e.get("type")
            in {"response.completed", "response.failed", "response.incomplete"}
        ]
        final = terminal[-1].get("response", {}) if terminal else {}
        calls = [o for o in final.get("output", []) if o.get("type") == "function_call"]
        errors = []
        if final.get("status") != "completed":
            errors.append("missing successful terminal response")
        if not calls:
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
        rounds.append(
            {
                "turn": turn,
                "elapsed_s": elapsed,
                "first_event_s": first,
                "usage": final.get("usage", {}),
                "errors": errors,
                "events": events,
            }
        )
        if errors:
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
    rounds = [r for row in rows for r in row.get("rounds", [])]
    tokens = sum(r["usage"].get("output_tokens", 0) for r in rounds)
    errors = sum(bool(r["errors"]) for r in rounds) + sum(
        "error" in row for row in rows
    )
    summary = {
        "started_at_unix": started_at_unix,
        "elapsed_s": elapsed,
        "output_tokens": tokens,
        "aggregate_output_tokens_per_s": tokens / elapsed,
        "completed_rounds": len(rounds),
        "errors": errors,
        "concurrency": args.concurrency,
        "turns": args.turns,
        "records": args.records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    if errors or len(rounds) != args.concurrency * args.turns:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--records", type=int, default=240)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
