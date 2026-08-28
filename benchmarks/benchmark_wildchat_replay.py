# SPDX-License-Identifier: Apache-2.0
"""Multi-turn conversation replay benchmark from WildChat-1M.

Simulates real-world chat traffic against an OpenAI-compatible endpoint:
each sampled WildChat conversation becomes a session whose user turns are
replayed one at a time; the model's own streamed replies are appended to the
history, so context grows turn over turn exactly as in production. A fixed
number of sessions run concurrently (closed loop): when one conversation
finishes, the next begins.

Usage:
  python benchmarks/benchmark_wildchat_replay.py \
      --parquet <wildchat shard> --concurrency 32 --sessions 96 \
      --base-url http://localhost:8000/v1 --api-key KEY --out results.json
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import pyarrow.parquet as pq


def load_sessions(path: str, n: int, min_turns: int, max_turns: int, seed: int):
    """Sample conversations and return their user-turn scripts."""
    f = pq.ParquetFile(path)
    convs = []
    seen = set()
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(
            rg, columns=["conversation_hash", "conversation", "turn", "toxic", "redacted"]
        )
        for row in t.to_pylist():
            if row["toxic"] or row["redacted"]:
                continue
            if row["turn"] < min_turns:
                continue
            if row["conversation_hash"] in seen:
                continue
            msgs = row["conversation"]
            if not msgs or msgs[0]["role"] != "user":
                continue
            users = [
                m["content"]
                for m in msgs
                if m["role"] == "user" and m["content"] and len(m["content"]) < 32768
            ]
            if len(users) < min_turns:
                continue
            seen.add(row["conversation_hash"])
            convs.append(users[:max_turns])
        if len(convs) >= n * 20:
            break
    random.Random(seed).shuffle(convs)
    return convs[:n]


async def run_session(client, model: str, users: list[str], sem, records: list,
                      session_id: int, max_tokens: int, ctx_stop: int):
    history = []
    async with sem:
        for depth, user_msg in enumerate(users, start=1):
            messages = history + [{"role": "user", "content": user_msg}]
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
                    timeout=600,
                )
                async for ev in stream:
                    if ev.usage is not None:
                        usage = ev.usage
                    if ev.choices and ev.choices[0].delta and (
                        ev.choices[0].delta.content or
                        getattr(ev.choices[0].delta, "reasoning_content", None)
                    ):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        if ev.choices[0].delta.content:
                            chunks.append(ev.choices[0].delta.content)
            except Exception as e:  # noqa: BLE001 - record and end session
                records.append({
                    "session": session_id, "depth": depth, "error": f"{type(e).__name__}: {e}"[:300],
                    "t_end": time.time(),
                })
                return
            t1 = time.perf_counter()
            reply = "".join(chunks)
            rec = {
                "session": session_id,
                "depth": depth,
                "ttft_s": ttft,
                "e2e_s": t1 - t0,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "t_end": time.time(),
            }
            records.append(rec)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": reply})
            if usage and usage.prompt_tokens + usage.completion_tokens > ctx_stop:
                return


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
    return vals[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--sessions", type=int, default=96)
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--ctx-stop", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import openai
    client = openai.AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    model = args.model or (await client.models.list()).data[0].id

    sessions = load_sessions(args.parquet, args.sessions, args.min_turns,
                             args.max_turns, args.seed)
    print(f"replaying {len(sessions)} conversations, "
          f"{sum(len(s) for s in sessions)} user turns, "
          f"concurrency {args.concurrency}, model {model}")

    sem = asyncio.Semaphore(args.concurrency)
    records: list = []
    t_start = time.time()
    await asyncio.gather(*[
        run_session(client, model, users, sem, records, i,
                    args.max_tokens, args.ctx_stop)
        for i, users in enumerate(sessions)
    ])
    wall = time.time() - t_start

    ok = [r for r in records if "error" not in r and r.get("completion_tokens")]
    errs = [r for r in records if "error" in r]
    out_tok = sum(r["completion_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    summary = {
        "config": vars(args),
        "model": model,
        "wall_s": wall,
        "turns_ok": len(ok),
        "turns_err": len(errs),
        "sessions": len(sessions),
        "output_tok": out_tok,
        "prompt_tok": in_tok,
        "output_tok_per_s": out_tok / wall,
        "total_tok_per_s": (out_tok + in_tok) / wall,
        "ttft_p50_s": pct(ttfts, 50),
        "ttft_p90_s": pct(ttfts, 90),
        "ttft_p99_s": pct(ttfts, 99),
        "e2e_p50_s": pct([r["e2e_s"] for r in ok], 50),
        "e2e_p90_s": pct([r["e2e_s"] for r in ok], 90),
        "mean_completion_tok": statistics.mean(r["completion_tokens"] for r in ok) if ok else None,
        "by_depth": {},
    }
    for d in sorted({r["depth"] for r in ok}):
        rows = [r for r in ok if r["depth"] == d]
        summary["by_depth"][d] = {
            "n": len(rows),
            "prompt_tok_mean": statistics.mean(r["prompt_tokens"] for r in rows),
            "ttft_p50_s": pct([r["ttft_s"] for r in rows if r["ttft_s"]], 50),
            "ttft_p90_s": pct([r["ttft_s"] for r in rows if r["ttft_s"]], 90),
            "out_tok_per_req_s_p50": pct(
                [r["completion_tokens"] / (r["e2e_s"] - r["ttft_s"])
                 for r in rows if r["ttft_s"] and r["e2e_s"] > r["ttft_s"]], 50),
        }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
