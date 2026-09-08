#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repeated profile starts with all runs retained and separate client timings.

Every start gets the same warmup and measured workload. No throughput-based
retry, exclusion, or early stopping is permitted. The JSON records every
request, startup failure, and the exact profile, source text, and git state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

os.environ["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"

from benchmark_dsv4_exact import exact_prompts, get_tokenizer

from slimserve import hardware, registry
from slimserve.server import free_port
from slimserve.smoke import (
    _IMAGE_PROMPT,
    _TEXT_PROMPT,
    _red_image_data_url,
    _request,
    _require_match,
    compatible_profile_ids,
)


def post(base: str, path: str, body: dict | None = None) -> bytes:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def request(base, model, prompt, output_tokens, seed, first_event=None):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    token_ids, chunks, text = [], [], []
    usage = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                text.append(choice.get("text", ""))
                if ids:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        if first_event:
                            first_event.set()
                    last = now
                    token_ids.extend(ids)
                    chunks.append({"seconds": now - start, "tokens": len(ids)})
    if first is None or not usage or len(token_ids) != output_tokens:
        raise ValueError(f"incomplete stream: {len(token_ids)} tokens, usage={usage}")
    if usage["completion_tokens"] != len(token_ids):
        raise ValueError(f"stream/usage token mismatch: {usage}")
    output = "".join(text)
    return {
        "start": start,
        "first": first,
        "last": last,
        "end": time.perf_counter(),
        "ttft_seconds": first - start,
        "decode_seconds": last - first,
        "tokens_after_first_chunk": len(token_ids) - chunks[0]["tokens"],
        "usage": usage,
        "chunks": chunks,
        "text": output,
        "token_ids": token_ids,
        "replacement_characters": output.count("\ufffd"),
        "seed": seed,
    }


def round_requests(base, model, prompts, tokens, seed, profile=False):
    events = [threading.Event() for _ in prompts]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as pool:
        futures = [
            pool.submit(request, base, model, p, tokens, seed + i, events[i])
            for i, p in enumerate(prompts)
        ]
        if profile:
            for event in events:
                if not event.wait(120):
                    raise RuntimeError("timed out waiting for prefill before profiling")
            time.sleep(0.5)
            post(base, "/start_profile")
        rows = [future.result() for future in futures]
    wall = time.perf_counter() - started
    if profile:
        post(base, "/stop_profile")
    decode_window = max(r["last"] for r in rows) - min(r["first"] for r in rows)
    return {
        "wall_seconds": wall,
        "aggregate_output_tps": sum(r["usage"]["completion_tokens"] for r in rows)
        / wall,
        "client_decode_tps": sum(r["tokens_after_first_chunk"] for r in rows)
        / decode_window,
        "ttft_mean_seconds": statistics.mean(r["ttft_seconds"] for r in rows),
        "requests": rows,
    }


def stop_owned(process):
    # Workers inherit this newly-created process group. Never match global
    # process names: another workload may start while this campaign is running.
    for sig, timeout in (
        (signal.SIGINT, 30),
        (signal.SIGTERM, 15),
        (signal.SIGKILL, 10),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # A parent can exit before its workers; finish only when the group is gone.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)
    raise RuntimeError(f"owned process group {process.pid} did not exit")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="glm53-nvfp4-4")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--boots", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--input-tokens", type=int, default=1000)
    ap.add_argument("--output-tokens", type=int, default=300)
    ap.add_argument("--traces", action="store_true")
    args = ap.parse_args()
    if min(args.boots, args.repeats, *args.concurrency) < 1 or args.output_tokens < 2:
        ap.error("boots, repeats and concurrency must be positive; output tokens >= 2")
    machine = hardware.detect()
    compatible = compatible_profile_ids(machine)
    if args.profile not in compatible:
        ap.error(f"profile not compatible; available: {compatible}")
    plan = registry.resolve(args.profile, machine.platform, machine.count, None)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.read_text()
    tokenizer = get_tokenizer(str(plan.model_dir))
    prompts = {
        c: exact_prompts(tokenizer, source, c, args.input_tokens, 0, False)
        for c in args.concurrency
    }
    command = lambda *cmd: subprocess.check_output(cmd, text=True).strip()
    receipt = {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "plan": dataclasses.asdict(plan),
        "compatible_profiles": compatible,
        "command": sys.argv,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_", "SLIMSERVE_", "NCCL_", "CUDA_", "OMP_"))
            and not any(s in k for s in ("TOKEN", "PASSWORD", "SECRET", "API_KEY"))
        },
        "runs": [],
    }
    record = args.output / "summary.json"
    for boot in range(1, args.boots + 1):
        active = command(
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
        )
        if active:
            raise RuntimeError(f"GPUs already have compute processes: {active}")
        folder = args.output / f"boot-{boot}"
        folder.mkdir()
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        argv = [
            sys.executable,
            "-m",
            "slimserve.cli",
            args.profile,
            "--serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-y",
        ]
        if args.traces:
            argv += ["--torch-profile-dir", str(folder / "traces")]
        run = {"boot": boot, "argv": argv, "status": "starting", "measurements": []}
        receipt["runs"].append(run)
        record.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"boot {boot}/{args.boots}: starting", flush=True)
        with (folder / "server.log").open("wb") as log:
            process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            started = time.monotonic()
            try:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"server exited {process.returncode}; "
                            f"see {folder / 'server.log'}"
                        )
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=3):
                            break
                    except OSError as error:
                        if time.monotonic() - started > 1200:
                            raise TimeoutError(
                                "profile startup exceeded 1200 seconds"
                            ) from error
                        time.sleep(2)
                run["startup_seconds"] = time.monotonic() - started
                model = plan.engine["served_model_name"]
                run["canaries"] = {}
                for label, prompt, image, pattern in (
                    ("text", _TEXT_PROMPT, None, r"\b4\b|\bfour\b"),
                    ("image", _IMAGE_PROMPT, _red_image_data_url(), r"\bred\b"),
                ):
                    if label == "image" and "image" not in plan.source["modalities"]:
                        continue
                    canary = _request(
                        plan, base, prompt, image_url=image, max_tokens=256, timeout=600
                    )
                    run["canaries"][label] = canary
                    _require_match(canary, pattern, label)
                run["warmups"] = []
                for c in args.concurrency:
                    warmup = round_requests(base, model, prompts[c], 32, 42)
                    warmup_path = folder / f"warmup-c{c}.json"
                    warmup_path.write_text(json.dumps(warmup, indent=2) + "\n")
                    run["warmups"].append(str(warmup_path))
                for rep in range(args.repeats):
                    for c in args.concurrency:
                        result = round_requests(
                            base, model, prompts[c], args.output_tokens, 42
                        )
                        result["exact"] = all(
                            r["usage"]["prompt_tokens"] == args.input_tokens
                            and r["usage"]["completion_tokens"] == args.output_tokens
                            for r in result["requests"]
                        )
                        result_path = folder / f"repeat-{rep + 1}-c{c}.json"
                        result_path.write_text(json.dumps(result, indent=2) + "\n")
                        run["measurements"].append(
                            {
                                "repeat": rep + 1,
                                "concurrency": c,
                                "path": str(result_path),
                                **{k: v for k, v in result.items() if k != "requests"},
                            }
                        )
                        record.write_text(json.dumps(receipt, indent=2) + "\n")
                        print(
                            f"boot {boot} repeat {rep + 1} c{c}: "
                            f"E2E {result['aggregate_output_tps']:.2f}, "
                            f"client decode {result['client_decode_tps']:.2f} tok/s, "
                            f"TTFT {1000 * result['ttft_mean_seconds']:.1f} ms",
                            flush=True,
                        )
                        if not result["exact"] or any(
                            r["replacement_characters"] for r in result["requests"]
                        ):
                            raise ValueError(
                                "token-count or replacement-character gate failed"
                            )
                if args.traces:
                    for c in (c for c in args.concurrency if c in (1, 8)):
                        result = round_requests(
                            base, model, prompts[c], 384, 42, profile=True
                        )
                        (folder / f"profile-c{c}.json").write_text(
                            json.dumps(result, indent=2) + "\n"
                        )
                run["status"] = "complete"
            except Exception as error:
                run["status"] = "failed"
                run["error"] = repr(error)
                print(f"boot {boot}: {error}", flush=True)
            finally:
                stop_owned(process)
                record.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["aggregates"] = {}
    for c in args.concurrency:
        values = [
            m["aggregate_output_tps"]
            for r in receipt["runs"]
            for m in r["measurements"]
            if m["concurrency"] == c
        ]
        if values:
            receipt["aggregates"][str(c)] = {
                "count": len(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["aggregates"], indent=2), flush=True)
    return int(any(r["status"] != "complete" for r in receipt["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
