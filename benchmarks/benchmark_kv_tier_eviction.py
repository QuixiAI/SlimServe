#!/usr/bin/env python3
"""Host KV tier eviction-restore acceptance.

Proves a session's KV survives full GPU-pool eviction through the host
tier (HostTierConnector):

1. Build a target session that plants distinct marker facts at several
   context depths, growing it to --target-ctx tokens.
2. Churn the GPU pool with filler sessions of fresh (uncacheable) content
   until the cumulative filler KV exceeds --pool-tokens by --churn-factor,
   evicting the target session's blocks to the host tier.
3. Resume the target session and probe every marker; each answer must
   contain its marker verbatim.

Run the server with VLLM_KV_TIER_VERIFY=1 for the byte-level SHA
round-trip check on every offload/restore alongside this content-level
check. Sampling follows serving policy: temperature 1.0 / top_p 0.95,
seeded, thinking enabled.

The pool size is in the boot log ("GPU KV cache size: N tokens"); pass it
as --pool-tokens.
"""

import argparse
import asyncio
import json
import random
import string
import sys
import time

import aiohttp

SAMPLING = {"temperature": 1.0, "top_p": 0.95}


def _words(rng: random.Random, n: int) -> str:
    return " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9)))
        for _ in range(n)
    )


async def _turn(session, args, messages, max_tokens, seed):
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "seed": seed,
        **SAMPLING,
    }
    async with session.post(
        f"{args.base_url}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=args.turn_timeout),
    ) as resp:
        body = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {str(body)[:200]}")
    msg = body["choices"][0]["message"]
    usage = body.get("usage", {})
    return msg.get("content") or "", usage.get("prompt_tokens", 0)


async def build_target(session, args, rng):
    """Grow one session to target-ctx, planting markers along the way."""
    markers = []  # (depth_tokens, codeword)
    messages = []
    ctx = 0
    plant_every = max(args.target_ctx // (args.markers + 1), 1)
    next_plant = plant_every
    while ctx < args.target_ctx:
        if len(markers) < args.markers and ctx >= next_plant * (len(markers) + 1) - plant_every:
            code = f"KV{rng.randint(1000, 9999)}T{rng.randint(100, 999)}"
            ask = (
                f"Remember this for later: secret checkpoint "
                f"#{len(markers) + 1} is {code}. Just acknowledge briefly. "
                + _words(rng, args.filler_words)
            )
            markers.append({"index": len(markers) + 1, "code": code, "ctx_at_plant": ctx})
        else:
            ask = (
                "Please briefly summarize this note. " + _words(rng, args.filler_words)
            )
        messages.append({"role": "user", "content": ask})
        reply, ctx = await _turn(session, args, messages, args.reply_tokens, args.seed)
        messages.append({"role": "assistant", "content": reply})
        print(f"[target] ctx={ctx} markers={len(markers)}", flush=True)
    return messages, markers, ctx


async def churn_filler(session, args, rng, fid, tokens_goal, counter):
    """One filler stream: fresh random content, never reusing prefixes."""
    while counter["driven"] < tokens_goal:
        messages = [{
            "role": "user",
            "content": f"Note {fid}-{rng.randint(0, 10**9)}: "
            + _words(rng, args.filler_words * 4)
            + " Summarize in one sentence.",
        }]
        try:
            _, ptoks = await _turn(session, args, messages, 128, args.seed + fid)
        except Exception as e:  # noqa: BLE001 - keep churning through timeouts
            print(f"[filler {fid}] error: {e}", flush=True)
            await asyncio.sleep(2)
            continue
        counter["driven"] += ptoks
        if counter["driven"] // 50_000 != (counter["driven"] - ptoks) // 50_000:
            print(f"[churn] driven ~{counter['driven']} tokens", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool-tokens", type=int, required=True,
                    help="GPU KV pool size in tokens (from the boot log)")
    ap.add_argument("--target-ctx", type=int, default=40_000)
    ap.add_argument("--markers", type=int, default=6)
    ap.add_argument("--churn-factor", type=float, default=1.5,
                    help="filler tokens driven = pool_tokens * this")
    ap.add_argument("--rechurn-factor", type=float, default=0.6,
                    help="extra churn before each later probe so every "
                    "probe restores from the tier (0 disables)")
    ap.add_argument("--churn-streams", type=int, default=8)
    ap.add_argument("--filler-words", type=int, default=400)
    ap.add_argument("--reply-tokens", type=int, default=256)
    ap.add_argument("--probe-tokens", type=int, default=512)
    ap.add_argument("--turn-timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    result = {"args": vars(args), "phases": {}}
    async with aiohttp.ClientSession() as session:
        t0 = time.monotonic()
        messages, markers, ctx = await build_target(session, args, rng)
        result["phases"]["build"] = {
            "ctx": ctx, "markers": markers, "seconds": time.monotonic() - t0,
        }
        print(f"[build done] ctx={ctx} markers={len(markers)}", flush=True)

        t0 = time.monotonic()
        goal = int(args.pool_tokens * args.churn_factor)
        counter = {"driven": 0}
        await asyncio.gather(*[
            churn_filler(session, args, random.Random(args.seed + 100 + i),
                         i, goal, counter)
            for i in range(args.churn_streams)
        ])
        result["phases"]["churn"] = {
            "driven_tokens": counter["driven"], "goal": goal,
            "seconds": time.monotonic() - t0,
        }
        print(f"[churn done] drove {counter['driven']} tokens "
              f"(pool {args.pool_tokens})", flush=True)

        t0 = time.monotonic()
        probes = []
        for m in markers:
            if m["index"] > 1 and args.rechurn_factor > 0:
                # Re-evict between probes so every marker exercises a
                # genuine tier restore, not a GPU prefix-cache hit on the
                # blocks the previous probe just restored.
                goal2 = counter["driven"] + int(
                    args.pool_tokens * args.rechurn_factor
                )
                await asyncio.gather(*[
                    churn_filler(
                        session, args,
                        random.Random(args.seed + 1000 * m["index"] + i),
                        1000 * m["index"] + i, goal2, counter,
                    )
                    for i in range(args.churn_streams)
                ])
            q = (
                f"Without repeating anything else, what exactly is secret "
                f"checkpoint #{m['index']}? Reply with just the code."
            )
            trial = messages + [{"role": "user", "content": q}]
            try:
                reply, ptoks = await _turn(
                    session, args, trial, args.probe_tokens, args.seed
                )
                ok = m["code"] in reply
            except Exception as e:  # noqa: BLE001
                reply, ptoks, ok = f"ERROR: {e}", 0, False
            probes.append({
                "index": m["index"], "code": m["code"], "ok": ok,
                "ctx_at_plant": m["ctx_at_plant"], "reply": reply[:200],
            })
            print(f"[probe {m['index']}] ok={ok} code={m['code']}", flush=True)
        result["phases"]["probe"] = {
            "probes": probes,
            "ok": sum(p["ok"] for p in probes),
            "total": len(probes),
            "seconds": time.monotonic() - t0,
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    ok, total = result["phases"]["probe"]["ok"], result["phases"]["probe"]["total"]
    print(f"EVICTION-RESTORE ACCEPTANCE: {ok}/{total} markers recalled "
          f"after {result['phases']['churn']['driven_tokens']}-token churn "
          f"of a {args.pool_tokens}-token pool", flush=True)
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
