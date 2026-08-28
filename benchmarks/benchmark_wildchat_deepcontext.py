# SPDX-License-Identifier: Apache-2.0
"""Deep-context multi-turn stress test seeded from WildChat-1M.

Runs N concurrent sessions that grow toward the model's full context
window: every user turn combines a real WildChat user message with a long
"document paste" built from real long WildChat messages, and the model's
own streamed replies join the history. Each session plants a marker fact
in turn 1 and is quizzed on it at increasing depths (long-range recall
through the quantized KV). Sessions end when the context ceiling is
reached (a clean over-limit rejection followed by a final squeeze turn)
or the wall clock expires.

Usage:
  python benchmarks/benchmark_wildchat_deepcontext.py \
      --parquet <shard> --concurrency 32 --max-hours 2 \
      --api-key KEY --out deep.json
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import pyarrow.parquet as pq

BUCKETS = [
    (0, 16_000),
    (16_000, 65_536),
    (65_536, 131_072),
    (131_072, 200_000),
    (200_000, 262_144),
]


def load_pools(path: str, seed: int):
    """Return (user_turn_pool, paste_pool) of real WildChat user texts."""
    f = pq.ParquetFile(path)
    turns: list[str] = []
    pastes: list[str] = []
    for rg in range(min(4, f.metadata.num_row_groups)):
        t = f.read_row_group(rg, columns=["conversation", "toxic", "redacted"])
        for row in t.to_pylist():
            if row["toxic"] or row["redacted"]:
                continue
            for m in row["conversation"]:
                if m["role"] != "user" or not m["content"]:
                    continue
                n = len(m["content"])
                if 40 <= n < 2000 and len(turns) < 4000:
                    turns.append(m["content"])
                elif n >= 4000 and len(pastes) < 1500:
                    pastes.append(m["content"])
        if len(turns) >= 4000 and len(pastes) >= 1500:
            break
    rng = random.Random(seed)
    rng.shuffle(turns)
    rng.shuffle(pastes)
    return turns, pastes


def make_paste(pastes: list[str], rng: random.Random, target_chars: int) -> str:
    parts: list[str] = []
    total = 0
    while total < target_chars:
        p = rng.choice(pastes)
        parts.append(p)
        total += len(p)
    return "\n\n---\n\n".join(parts)[:target_chars]


async def run_session(client, model, sid, turns, pastes, records, stop_at,
                      args, sem):
    rng = random.Random(args.seed * 1000 + sid)
    marker = f"ZX{rng.randint(1000, 9999)}Q"
    history = [
        {
            "role": "user",
            "content": (
                f"Remember this for later: my project codename is {marker}. "
                "Confirm you have it, briefly."
            ),
        }
    ]
    depth = 0
    ctx_tokens = 0
    ended = "wall_clock"
    async with sem:
        # Turn 0: plant the marker.
        reply = await _turn(client, model, history, records, sid, 0, args, 128)
        if reply is None:
            ended = "error"
            depth = -1
        else:
            history.append({"role": "assistant", "content": reply})
        while depth >= 0 and time.time() < stop_at:
            depth += 1
            if depth % args.probe_every == 0:
                user_msg = (
                    "Quick check before we continue: what is my project "
                    "codename, exactly?"
                )
                probe = True
                max_tokens = 96
            else:
                user_msg = (
                    make_paste(pastes, rng, args.paste_chars)
                    + "\n\n"
                    + rng.choice(turns)
                )
                probe = False
                max_tokens = args.reply_tokens
            history.append({"role": "user", "content": user_msg})
            reply = await _turn(
                client, model, history, records, sid, depth, args, max_tokens,
                probe_marker=marker if probe else None,
            )
            if reply is None:
                # Most likely the context ceiling: try one squeeze turn with
                # a tiny budget; if that also fails, the session is done.
                history.pop()
                history.append(
                    {"role": "user", "content": "Summarize our conversation in one line."}
                )
                reply = await _turn(
                    client, model, history, records, sid, depth, args, 32
                )
                ended = "ceiling" if reply is not None else "error"
                break
            history.append({"role": "assistant", "content": reply})
            last = [r for r in records if r.get("session") == sid and "prompt_tokens" in r]
            if last:
                ctx_tokens = last[-1]["prompt_tokens"] + last[-1]["completion_tokens"]
            if ctx_tokens >= args.ctx_target:
                ended = "target"
                break
    records.append(
        {"session": sid, "final": True, "depth": depth, "ctx_tokens": ctx_tokens,
         "ended": ended}
    )


async def _turn(client, model, messages, records, sid, depth, args, max_tokens,
                probe_marker=None):
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    usage = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            extra_body={"top_k": 20},
            stream=True,
            stream_options={"include_usage": True},
            timeout=args.turn_timeout,
        )
        async for ev in stream:
            if ev.usage is not None:
                usage = ev.usage
            if ev.choices and ev.choices[0].delta and (
                ev.choices[0].delta.content
                or getattr(ev.choices[0].delta, "reasoning_content", None)
            ):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if ev.choices[0].delta.content:
                    chunks.append(ev.choices[0].delta.content)
    except Exception as e:  # noqa: BLE001
        records.append(
            {"session": sid, "depth": depth,
             "error": f"{type(e).__name__}: {e}"[:300], "t": time.time()}
        )
        return None
    reply = "".join(chunks)
    rec = {
        "session": sid,
        "depth": depth,
        "ttft_s": ttft,
        "e2e_s": time.perf_counter() - t0,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "t": time.time(),
    }
    if probe_marker is not None:
        rec["probe"] = True
        rec["recall_ok"] = probe_marker in reply
    records.append(rec)
    return reply


def bucket_of(tokens):
    for lo, hi in BUCKETS:
        if lo <= tokens < hi:
            return f"{lo//1000}K-{hi//1000}K"
    return ">=262K"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--ctx-target", type=int, default=255_000)
    ap.add_argument("--paste-chars", type=int, default=30_000)
    ap.add_argument("--reply-tokens", type=int, default=512)
    ap.add_argument("--probe-every", type=int, default=6)
    ap.add_argument("--turn-timeout", type=float, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import openai
    client = openai.AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key,
        timeout=args.turn_timeout, max_retries=0,
    )
    model = (await client.models.list()).data[0].id
    turns, pastes = load_pools(args.parquet, args.seed)
    print(f"pools: {len(turns)} user turns, {len(pastes)} pastes; "
          f"{args.concurrency} sessions -> {args.ctx_target} tokens, "
          f"{args.max_hours}h cap", flush=True)

    stop_at = time.time() + args.max_hours * 3600
    records: list = []
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    await asyncio.gather(*[
        run_session(client, model, sid, turns, pastes, records, stop_at, args, sem)
        for sid in range(args.concurrency)
    ])
    wall = time.time() - t0

    ok = [r for r in records if "prompt_tokens" in r and r.get("completion_tokens")]
    finals = [r for r in records if r.get("final")]
    errs = [r for r in records if "error" in r]
    probes = [r for r in records if r.get("probe")]
    by_bucket: dict = {}
    for r in ok:
        b = bucket_of(r["prompt_tokens"])
        by_bucket.setdefault(b, []).append(r)
    summary = {
        "wall_s": wall,
        "turns_ok": len(ok),
        "errors": len(errs),
        "total_prompt_tok": sum(r["prompt_tokens"] for r in ok),
        "total_completion_tok": sum(r["completion_tokens"] for r in ok),
        "sessions": {
            "ceiling": sum(1 for r in finals if r["ended"] == "ceiling"),
            "target": sum(1 for r in finals if r["ended"] == "target"),
            "wall_clock": sum(1 for r in finals if r["ended"] == "wall_clock"),
            "error": sum(1 for r in finals if r["ended"] == "error"),
            "max_ctx": max((r["ctx_tokens"] for r in finals), default=0),
            "median_ctx": statistics.median(
                [r["ctx_tokens"] for r in finals]) if finals else 0,
        },
        "recall": {
            "probes": len(probes),
            "ok": sum(1 for r in probes if r.get("recall_ok")),
            "failed_at": sorted(
                r["prompt_tokens"] for r in probes if not r.get("recall_ok")
            )[:20],
        },
        "by_context": {
            b: {
                "n": len(rs),
                "ttft_p50_s": statistics.median(
                    [r["ttft_s"] for r in rs if r["ttft_s"]]) if rs else None,
                "ttft_p90_s": sorted(
                    [r["ttft_s"] for r in rs if r["ttft_s"]]
                )[max(0, int(0.9 * len(rs)) - 1)] if rs else None,
                "e2e_p50_s": statistics.median([r["e2e_s"] for r in rs]),
            }
            for b, rs in sorted(by_bucket.items())
        },
        "error_kinds": {},
    }
    for r in errs:
        k = r["error"].split(":")[0]
        summary["error_kinds"][k] = summary["error_kinds"].get(k, 0) + 1
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
