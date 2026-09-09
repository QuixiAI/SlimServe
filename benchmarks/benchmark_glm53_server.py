#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the fixed GLM campaign workload on an explicitly supplied local server.

This client does not start, stop, or reconfigure a server. Its owner must retain
the launch command, immutable server/model identity, startup failures, and a
predetermined number of starts separately. The decode, quality and cold-prefill
implementations are the SAME functions used by benchmark_glm53_campaign.py.
Competing checkpoints/activation/KV precisions must be labeled, not called a
matched-precision result. Chat canaries use each server's template defaults.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import urllib.parse
from pathlib import Path

from benchmark_glm53_campaign import (
    exact_prompts,
    get_tokenizer,
    gpu_snapshot,
    round_requests,
)
from benchmark_glm53_prefill import run as run_prefill
from benchmark_glm53_quality import run as run_quality

from slimserve.smoke import (
    _IMAGE_PROMPT,
    _TEXT_PROMPT,
    _red_image_data_url,
    _require_match,
)
from slimserve.stream import chat_completion, visible_text


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer_receipt(reference, candidate, source, load=get_tokenizer):
    """Check all semantic tokenizer components and the complete source IDs.

    Serialized padding/truncation defaults may differ; the actual loader must
    still produce identical full-source IDs. Do not accept vocabulary-only
    equality, which misses normalization, pre-tokenization and added tokens.
    """
    records = []
    semantics = []
    for directory in (reference, candidate):
        data = json.loads((directory / "tokenizer.json").read_text())
        defaults = {key: data.pop(key, None) for key in ("padding", "truncation")}
        semantics.append(data)
        records.append(
            {
                "directory": str(directory),
                "serialized_defaults": defaults,
                "file_sha256": {
                    name: file_sha256(directory / name)
                    for name in (
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                    )
                    if (directory / name).is_file()
                },
            }
        )
    if semantics[0] != semantics[1]:
        raise ValueError("tokenizer semantic components differ")
    tokenizers = [load(str(directory)) for directory in (reference, candidate)]
    ids = [tok.encode(source, add_special_tokens=False) for tok in tokenizers]
    if ids[0] != ids[1]:
        raise ValueError("loaded tokenizers produce different source IDs")
    if len(ids[0]) < 131072:
        raise ValueError("source must provide at least 131072 untruncated tokens")
    return tokenizers[0], {
        "reference": records[0],
        "candidate": records[1],
        "semantic_components_equal": True,
        "source_tokens": len(ids[0]),
        "source_token_ids_sha256": hashlib.sha256(
            json.dumps(ids[0], separators=(",", ":")).encode()
        ).hexdigest(),
    }


def check_round(result, concurrency, input_tokens=1000, output_tokens=300):
    if any(
        row["usage"].get("prompt_tokens") != input_tokens
        or row["usage"].get("completion_tokens") != output_tokens
        or len(row["token_ids"]) != output_tokens
        or row["replacement_characters"]
        for row in result["requests"]
    ):
        raise ValueError("exact-token or replacement-character gate failed")
    if len(result["requests"]) != concurrency:
        raise ValueError("request/concurrency count mismatch")


def run(args):
    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("this campaign client requires an explicit local HTTP server")
    if args.repeats < 1:
        raise ValueError("positive repeat count required")
    args.output.mkdir(parents=True, exist_ok=False)
    record = args.output / "summary.json"
    receipt = {
        "status": "preparing",
        "method": __doc__,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "implementation_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in (
                "benchmark_glm53_server.py",
                "benchmark_glm53_campaign.py",
                "benchmark_dsv4_exact.py",
                "benchmark_glm53_quality.py",
                "benchmark_glm53_prefill.py",
            )
        },
        "source_sha256": file_sha256(args.source),
        "canaries": {},
        "warmups": [],
        "measurements": [],
    }

    def save():
        record.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        source = args.source.read_text()
        tokenizer, receipt["tokenizers"] = tokenizer_receipt(
            args.reference_tokenizer, args.tokenizer, source
        )
        prompts = {
            c: exact_prompts(tokenizer, source, c, 1000, 0, False) for c in (1, 8, 16)
        }
        receipt["prompt_sha256"] = {
            str(c): [hashlib.sha256(p.encode()).hexdigest() for p in rows]
            for c, rows in prompts.items()
        }
        receipt["status"] = "running"
        save()
        for label, prompt, image, pattern in (
            ("text", _TEXT_PROMPT, None, r"\b4\b|\bfour\b"),
            ("image", _IMAGE_PROMPT, _red_image_data_url(), r"\bred\b"),
        ):
            content = (
                prompt
                if image is None
                else [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": prompt},
                ]
            )
            started = time.perf_counter()
            raw = "".join(
                chat_completion(
                    args.url,
                    args.model,
                    [{"role": "user", "content": content}],
                    max_tokens=256,
                    seed=42,
                    timeout=600,
                )
            )
            canary = {
                "answer": visible_text(raw)[:500],
                "seconds": time.perf_counter() - started,
            }
            receipt["canaries"][label] = canary
            save()
            _require_match(canary, pattern, label)
        for warmup, count in ((True, 1), (False, args.repeats)):
            for rep in range(1, count + 1):
                for c in (1, 8, 16):
                    kind = "warmup" if warmup else f"repeat-{rep}"
                    path = args.output / f"{kind}-c{c}.json"
                    row = {
                        "concurrency": c,
                        "repeat": rep,
                        "path": str(path),
                        "status": "running",
                    }
                    receipt["warmups" if warmup else "measurements"].append(row)
                    save()
                    result = round_requests(args.url, args.model, prompts[c], 300, 42)
                    result["gpu_after_round"] = gpu_snapshot()
                    path.write_text(json.dumps(result, indent=2) + "\n")
                    check_round(result, c)
                    row.update({k: v for k, v in result.items() if k != "requests"})
                    row["status"] = "complete"
                    save()
                    print(
                        f"{kind} c{c}: E2E {result['aggregate_output_tps']:.2f}, "
                        f"client decode {result['client_decode_tps']:.2f} tok/s",
                        flush=True,
                    )
        if args.quality:
            quality = run_quality(
                argparse.Namespace(
                    url=args.url,
                    model=args.model,
                    tokenizer=str(args.reference_tokenizer),
                    source=args.source,
                    output=args.output / "quality.json",
                    pairs=32,
                    prefix_tokens=512,
                    score_tokens=128,
                    needle_contexts=[1024, 8192, 32768],
                    needle_positions=[0.25, 0.75],
                ),
                tokenizer,
            )
            receipt["quality"] = quality["summary"]
            save()
            if not quality["summary"]["all_needles_rank_first"]:
                raise ValueError("needle contrast gate failed")
        if args.prefill:
            prefill = run_prefill(
                argparse.Namespace(
                    url=args.url,
                    model=args.model,
                    tokenizer=str(args.reference_tokenizer),
                    source=args.source,
                    output=args.output / "prefill",
                    contexts=[32768, 131072],
                    repeats=3,
                    warmups=1,
                    output_tokens=8,
                    traces=False,
                ),
                tokenizer,
            )
            receipt["prefill"] = prefill["aggregates"]
        receipt["aggregates"] = {}
        for c in (1, 8, 16):
            values = [
                r["aggregate_output_tps"]
                for r in receipt["measurements"]
                if r["concurrency"] == c
            ]
            receipt["aggregates"][str(c)] = {
                "count": len(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        receipt["status"] = "complete"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--reference-tokenizer", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps(result["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
