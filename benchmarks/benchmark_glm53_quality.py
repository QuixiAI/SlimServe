#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record exact-token continuation scores and fixed-length needle contrasts.

These are regression measurements, not a general model-quality certification.
Prompts are token IDs: continuation boundaries never rely on decoding slices
and separately retokenizing suffixes. Generation uses the recommended sampling;
only raw prompt log probabilities contribute to the scores.
These scores exercise prefill. Decode changes additionally need per-shape
kernel parity and generated-request checks; this tool does not replace them.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

# Fixed seed-42 random codes, all seven tokens with the pinned GLM tokenizer.
# Keep the runtime length check: a tokenizer change must not bias the contrast.
CODES = ("161559407816", "316475255341", "928327648350", "305641395376")


def continuation_windows(ids, pairs, prefix_tokens, score_tokens):
    width = prefix_tokens + score_tokens
    if min(pairs, prefix_tokens, score_tokens) < 1 or len(ids) < width:
        raise ValueError("positive sizes and enough source tokens are required")
    starts = [round(i * (len(ids) - width) / max(1, pairs - 1)) for i in range(pairs)]
    if len(set(starts)) != pairs:
        raise ValueError("source is too short for distinct continuation windows")
    return [(start, ids[start : start + width]) for start in starts]


def scored_tail(response, ids, prefix_tokens):
    if response["usage"]["prompt_tokens"] != len(ids):
        raise ValueError("server changed the explicit prompt-token count")
    scores = response["choices"][0]["prompt_logprobs"]
    if len(scores) != len(ids):
        raise ValueError("prompt log probabilities do not align with token IDs")
    tail = []
    for token, distribution in zip(ids[prefix_tokens:], scores[prefix_tokens:]):
        if not distribution or str(token) not in distribution:
            raise ValueError("missing log probability for the actual prompt token")
        value = distribution[str(token)]["logprob"]
        if not math.isfinite(value):
            raise ValueError("non-finite continuation log probability")
        tail.append(value)
    if not tail:
        raise ValueError("empty scored continuation")
    return tail


def request_score(url, model, ids):
    payload = {
        "model": model,
        "prompt": ids,
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "prompt_logprobs": 0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as stream:
        response = json.load(stream)
    return response


def needle_prefix(encode, source_ids, context_tokens, fraction):
    instruction = encode("Read this record and remember its secret code.\n")
    fact = encode(f"\nThe secret code is {CODES[0]}. Keep this exact code.\n")
    question = encode("\nQuestion: What is the secret code?\nAnswer: ")
    hay_tokens = context_tokens - len(instruction) - len(fact) - len(question)
    if hay_tokens < 1 or not 0 <= fraction <= 1:
        raise ValueError("needle context is too short or position is invalid")
    hay = (source_ids * ((hay_tokens + len(source_ids) - 1) // len(source_ids)))[
        :hay_tokens
    ]
    split = int(hay_tokens * fraction)
    return instruction + hay[:split] + fact + hay[split:] + question


def run(args, tokenizer):
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    source_bytes = args.source.read_bytes()
    ids = encode(source_bytes.decode())
    windows = continuation_windows(
        ids, args.pairs, args.prefix_tokens, args.score_tokens
    )
    suffixes = [encode(code) for code in CODES]
    if len({len(suffix) for suffix in suffixes}) != 1:
        raise ValueError("needle alternatives must have equal token lengths")
    result = {
        "model": args.model,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_tokens": len(ids),
        "method": "explicit token IDs; raw logprobs; equal-length code contrasts",
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text": [],
        "needles": [],
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        for offset, prompt in windows:
            row = {
                "offset": offset,
                "prompt_ids": prompt,
                "response": request_score(args.url, args.model, prompt),
            }
            result["text"].append(row)
            save()  # Retain malformed responses before checking them.
            row["per_token"] = scored_tail(row["response"], prompt, args.prefix_tokens)
            save()
        for context in args.needle_contexts:
            for position in args.needle_positions:
                prefix = needle_prefix(encode, ids, context, position)
                row = {
                    "context_tokens": context,
                    "position": position,
                    "prefix_ids": prefix,
                    "candidates": [],
                }
                result["needles"].append(row)
                for code, suffix in zip(CODES, suffixes):
                    candidate = {
                        "code": code,
                        "suffix_ids": suffix,
                        "response": request_score(
                            args.url, args.model, prefix + suffix
                        ),
                    }
                    row["candidates"].append(candidate)
                    save()
                    candidate["per_token"] = scored_tail(
                        candidate["response"], prefix + suffix, len(prefix)
                    )
                    save()
                totals = [sum(c["per_token"]) for c in row["candidates"]]
                row["margin"] = totals[0] - max(totals[1:])
                save()
        scores = [v for row in result["text"] for v in row["per_token"]]
        result["summary"] = {
            "scored_text_tokens": len(scores),
            "mean_text_logprob": statistics.mean(scores),
            "needle_margins": [row["margin"] for row in result["needles"]],
            "all_needles_rank_first": all(
                row["margin"] > 0 for row in result["needles"]
            ),
        }
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        save()
        raise
    save()
    print(json.dumps(result["summary"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--score-tokens", type=int, default=128)
    parser.add_argument(
        "--needle-contexts", type=int, nargs="+", default=[1024, 8192, 32768]
    )
    parser.add_argument(
        "--needle-positions", type=float, nargs="+", default=[0.25, 0.75]
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    from transformers import AutoTokenizer

    result = run(args, AutoTokenizer.from_pretrained(args.tokenizer))
    if not result["summary"]["all_needles_rank_first"]:
        raise SystemExit("needle contrast failed; all measurements retained in output")


if __name__ == "__main__":
    main()
