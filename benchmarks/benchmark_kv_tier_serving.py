# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry-driven KV tier acceptance and exact-token serving measurements.

Run on/off in separate invocations on an idle machine. Off is an explicit
diagnostic: only kv_transfer_config is removed from the registered plan.
Reset acceptance forces GPU eviction; --pressure-tokens additionally exercises
natural eviction through concurrent, distinct filler requests. JSON checkpoints
survive any failed request. No empty response or missing restore counts as a pass.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import regex as re

from slimserve import fetch, hardware
from slimserve.server import Server
from slimserve.smoke import resolve_profiles, validate_acceleration


class Probe:
    def __init__(self, server, output, timeout):
        self.server, self.output, self.timeout = server, output, timeout
        self.model = server.plan.engine["served_model_name"]
        self.prompt_counts = {}
        self.logprobs = None
        self.chat_temperature = None

    def post(self, path, body):
        req = urllib.request.Request(
            self.server.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = response.read()
        return json.loads(data) if data else None

    def chat(self, messages, label):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "seed": 42,
        }
        if self.chat_temperature is not None:
            body["temperature"] = self.chat_temperature
        start = time.perf_counter()
        response = self.post("/v1/chat/completions", body)
        result = {
            "seconds": time.perf_counter() - start,
            "request": body,
            "response": response,
        }
        (self.output / f"{label}.json").write_text(json.dumps(result, indent=2))
        self.prompt_counts[label] = response.get("usage", {}).get("prompt_tokens", 0)
        choice = response["choices"][0]
        answer = choice["message"].get("content") or ""
        if not answer.strip() or choice["finish_reason"] == "length":
            raise AssertionError(f"{label}: empty or truncated answer")
        return choice["message"], result["seconds"]

    def tokens(self, text):
        return self.post(
            "/tokenize",
            {"model": self.model, "prompt": text, "add_special_tokens": False},
        )["tokens"]

    def complete(self, prompt, count):
        start = time.perf_counter()
        response = self.post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": count,
                "min_tokens": count,
                "ignore_eos": True,
                "seed": 42,
                "logprobs": self.logprobs,
            },
        )
        if response["usage"]["completion_tokens"] != count:
            self.reject_completion(
                response, "completion did not produce the exact token count"
            )
        text = response["choices"][0].get("text") or ""
        if count >= 32 and len(text) / count < 0.5:
            self.reject_completion(
                response, "degenerate completion: fewer than 0.5 chars/token"
            )
        if count >= 32:
            words = re.findall(r"\w+", text.lower())
            dominant = Counter(words).most_common(1)
            if (len(words) >= 128 and dominant[0][1] > len(words) / 2) or re.fullmatch(
                r"(.{1,16})\1{31,}", text.strip(), re.DOTALL
            ):
                self.reject_completion(
                    response, "degenerate completion: repetitive output"
                )
        return {"seconds": time.perf_counter() - start, "response": response}

    def reject_completion(self, response, reason):
        (self.output / f"failed-completion-{time.time_ns()}.json").write_text(
            json.dumps(response, indent=2)
        )
        raise AssertionError(reason)

    def draft_tokens(self):
        with urllib.request.urlopen(self.server.base_url + "/metrics", timeout=30) as r:
            metrics = r.read().decode()
        return sum(
            float(value)
            for value in re.findall(
                r"^vllm:spec_decode_num_draft_tokens_total(?:\{[^\n]*\})?\s+([\d.eE+-]+)",
                metrics,
                re.MULTILINE,
            )
        )


def document(seed, words):
    rng = random.Random(seed)
    vocabulary = [
        "harbor",
        "granite",
        "meridian",
        "lattice",
        "ember",
        "cascade",
        "orchid",
        "tundra",
    ]
    return " ".join(
        rng.choice(vocabulary) + str(rng.randrange(1000)) for _ in range(words)
    )


def acceptance(probe, log_path, args, checkpoint):
    # Exercise the minimum-token sampler before the expensive session setup.
    probe.complete(probe.tokens("Health check. The weather is calm."), 1)
    if not args.pressure_tokens:
        reset_prefix_cache(probe)
    histories, markers = [], []
    for i in range(args.sessions):
        marker = [
            "opal harbor lantern",
            "amber cedar meadow",
            "silver maple river",
            "violet stone garden",
        ][i % 4]
        instruction = (
            f"My launch phrase is {marker}. Remember it and repeat it exactly. "
        )
        if getattr(args, "marker_format", "plain") == "quoted":
            instruction = (
                "Your task is exact text copying. The launch phrase is quoted here: "
                f'"{marker}". Copy only the three words inside those quotes, '
                "with no changes. Remember the phrase for later turns. "
            )
        messages = [
            {
                "role": "user",
                "content": (
                    instruction + f"Background notes for session {i}: "
                    f"{document(100 + i, args.context_words)}"
                    + (
                        "\nWhat is my launch phrase? Reply with only its three "
                        "words, exactly as written."
                        if getattr(args, "marker_format", "plain") == "question-last"
                        else ""
                    )
                ),
            }
        ]
        reply, _ = probe.chat(messages, f"plant-{i}")
        if marker not in reply["content"].lower():
            raise AssertionError(f"plant {i}: marker not echoed exactly")
        messages.append(reply)
        histories.append(messages)
        markers.append(marker)
    for turn in range(args.rounds):
        # Flush pending finish-time snapshots with an unrelated engine step.
        probe.complete(probe.tokens(f"Checkpoint {turn}. The weather is calm."), 1)
        if args.pressure_tokens:
            chunk = min(16000, args.pressure_tokens)
            jobs = (args.pressure_tokens + chunk - 1) // chunk

            # Each filler has a distinct prefix; cycling one prompt would
            # measure cache hits instead of exerting eviction pressure.
            def filler(i, turn=turn, chunk=chunk):
                source = probe.tokens(document(4000 + turn * 100000 + i, chunk))
                return probe.complete(source[:chunk], 1)

            with ThreadPoolExecutor(max_workers=args.sessions) as pool:
                pressure_results = list(pool.map(filler, range(jobs)))
            (probe.output / f"pressure-{turn}.json").write_text(
                json.dumps(pressure_results, indent=2)
            )
            pressure_actual = sum(
                r["response"]["usage"]["prompt_tokens"] for r in pressure_results
            )
            if pressure_actual < args.pressure_tokens:
                raise AssertionError(
                    "filler requests did not reach the pressure budget"
                )
        else:
            reset_prefix_cache(probe)
        before = len(log_path.read_text())

        def recall(i, turn=turn):
            history = histories[i]
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"Check {turn + 1}: what is my launch phrase? "
                        "Reply with just its three words."
                    ),
                }
            )
            reply, seconds = probe.chat(history, f"recall-{turn}-{i}")
            if markers[i] not in reply["content"].lower():
                raise AssertionError(
                    f"turn {turn} session {i}: incorrect marker recall"
                )
            history.append(reply)
            return seconds

        with ThreadPoolExecutor(max_workers=args.sessions) as pool:
            latencies = list(pool.map(recall, range(args.sessions)))
        evidence = restore_evidence(
            log_path,
            before,
            args,
            min(probe.prompt_counts[f"plant-{i}"] for i in range(args.sessions)),
        )
        checkpoint.append(
            {
                "turn": turn,
                **evidence,
                "latency_seconds": latencies,
                "recall_correct": True,
            }
        )
        (probe.output / "acceptance.json").write_text(json.dumps(checkpoint, indent=2))


def restore_evidence(log_path, before, args, prompt_tokens):
    log = log_path.read_text()
    trail = log[before:]
    completed_ids = {
        request_id
        for line in trail.splitlines()
        if "worker done recv=" in line
        for request_id in re.findall(r"'([a-zA-Z0-9_-]+)'", line.split("recv=", 1)[1])
    }
    completed = len(completed_ids)
    if args.tier == "on" and completed < args.sessions:
        raise AssertionError(
            f"only {completed}/{args.sessions} completed tier restores"
        )
    restored = {
        request_id: int(tokens)
        for request_id, tokens in re.findall(
            r"host-tier: hit for (\S+): resume at block \d+ \((\d+) tokens\)", trail
        )
        if request_id in completed_ids
    }
    block = re.search(r"host-tier: \d+ slots, block (\d+) tokens", log)
    block_size = int(block[1]) if block else 1
    minimum = max(128, prompt_tokens - block_size)
    if block and re.search(r"host-tier: group \d+: MambaSpec\b", log):
        # Speculative align-mode prefill deliberately caches one block before
        # the final full block (the drafter's cache-hit guard). The partial
        # tail is recomputed. Require that real state boundary, not a snapshot
        # inside the final chunk that the worker never materialized.
        minimum = max(128, (prompt_tokens // block_size - 1) * block_size)
    if args.tier == "on" and (
        len(restored) < args.sessions or any(n < minimum for n in restored.values())
    ):
        raise AssertionError(
            f"restore depth {restored} did not preserve session history "
            f"(required >= {minimum} tokens for each session)"
        )
    if re.search(
        r"VERIFY MISMATCH|[1-9]\d*/\d+ restores mismatched|failed load|IO thread died",
        trail,
    ):
        raise AssertionError("tier verification or load failed")
    return {
        "restore_completions": completed,
        "restored_tokens": restored,
        "minimum_restore_tokens": minimum,
    }


def replay_acceptance(probe, log_path, args, checkpoint):
    """Require generation equality against primed GPU-cache controls."""
    probe.logprobs = getattr(args, "replay_logprobs", 0) or None
    probe.complete(probe.tokens("Health check. The weather is calm."), 1)
    reset_prefix_cache(probe)
    source_words = Path(args.source).read_text().split()
    body = " ".join(
        (source_words * (args.context_words // len(source_words) + 1))[
            : args.context_words
        ]
    )
    prompts = [
        probe.tokens(
            f"Session {i}.\n{body}\nExplain the main ideas in this document:\n"
        )
        for i in range(args.sessions)
    ]
    (probe.output / "replay-prompts.json").write_text(json.dumps(prompts))
    for prompt in prompts:
        probe.complete(prompt, 8)

    def generate(prompt):
        return probe.complete(prompt, args.replay_output_tokens)

    with ThreadPoolExecutor(max_workers=args.sessions) as pool:
        controls = list(pool.map(generate, prompts))
    (probe.output / "replay-controls.json").write_text(json.dumps(controls, indent=2))
    for repeat in range(1, getattr(args, "replay_control_repeats", 1)):
        with ThreadPoolExecutor(max_workers=args.sessions) as pool:
            repeated = list(pool.map(generate, prompts))
        (probe.output / f"replay-controls-{repeat}.json").write_text(
            json.dumps(repeated, indent=2)
        )
        if any(
            a["response"]["choices"][0]["text"] != b["response"]["choices"][0]["text"]
            for a, b in zip(controls, repeated)
        ):
            raise AssertionError("GPU-cache control changed before eviction")
    minimum_prompt = min(r["response"]["usage"]["prompt_tokens"] for r in controls)
    for turn in range(args.rounds):
        probe.complete(probe.tokens(f"Checkpoint {turn}. The weather is calm."), 1)
        reset_prefix_cache(probe)
        before = len(log_path.read_text())
        with ThreadPoolExecutor(max_workers=args.sessions) as pool:
            responses = list(pool.map(generate, prompts))
        (probe.output / f"replay-round-{turn}.json").write_text(
            json.dumps(responses, indent=2)
        )
        evidence = restore_evidence(log_path, before, args, minimum_prompt)
        for i, (control, response) in enumerate(zip(controls, responses)):
            if (
                control["response"]["choices"][0]["text"]
                != response["response"]["choices"][0]["text"]
            ):
                raise AssertionError(
                    f"turn {turn} session {i}: "
                    "generation differs from GPU-cache control"
                )
        checkpoint.append(
            {
                "turn": turn,
                **evidence,
                "generation_exact": True,
                "latency_seconds": [r["seconds"] for r in responses],
            }
        )
        (probe.output / "acceptance.json").write_text(json.dumps(checkpoint, indent=2))


def reset_prefix_cache(probe):
    for attempt in range(10):
        result = probe.post("/reset_prefix_cache", {})
        if result and result.get("success") is True:
            return
        # Drive pending finish-time snapshots through their release step.
        probe.complete(probe.tokens(f"Drain checkpoint {attempt}."), 1)
        time.sleep(0.05)
    raise AssertionError("local prefix cache reset did not succeed")


def benchmark(probe, args):
    # Replay diagnostics must not add top-k logprob work to timed samples.
    probe.logprobs = None
    source = Path(args.source).read_text()
    tokens = probe.tokens(source)
    if len(tokens) < args.input_tokens + max(args.concurrency):
        raise ValueError("benchmark source is too short")
    results = []
    for concurrency in args.concurrency:
        samples = []
        for repeat in range(args.repeats):
            prompts = [tokens[i : i + args.input_tokens] for i in range(concurrency)]
            # Some model renderers add a wrapper even to token-ID prompts.
            # Measure it, then adjust so API usage is exactly input_tokens.
            warm = probe.complete(prompts[0], 8)
            overhead = warm["response"]["usage"]["prompt_tokens"] - args.input_tokens
            if overhead < 0 or overhead >= args.input_tokens:
                raise AssertionError(f"unexpected prompt overhead {overhead}")
            if overhead:
                prompts = [p[:-overhead] for p in prompts]
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(lambda p: probe.complete(p, 8), prompts))
                drafts_before = probe.draft_tokens()
                start = time.perf_counter()
                started_at = time.time()
                futures = [
                    pool.submit(probe.complete, p, args.output_tokens) for p in prompts
                ]
                responses, errors = [], []
                for request_index, future in enumerate(futures):
                    try:
                        responses.append(future.result())
                    except Exception as error:
                        responses.append(None)
                        errors.append(
                            {
                                "request_index": request_index,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                elapsed = time.perf_counter() - start
                finished_at = time.time()
                if errors:
                    # Keep successful peers when another request times out.
                    # A partial batch never receives a throughput result.
                    (
                        probe.output / f"failed-bench-c{concurrency}-r{repeat}.json"
                    ).write_text(
                        json.dumps(
                            {
                                "started_at_unix": started_at,
                                "finished_at_unix": finished_at,
                                "wall_seconds": elapsed,
                                "responses": responses,
                                "errors": errors,
                            },
                            indent=2,
                        )
                    )
                    raise AssertionError(f"benchmark requests failed: {errors}")
                draft_tokens = probe.draft_tokens() - drafts_before
            if draft_tokens <= 0:
                raise AssertionError("registered drafter produced no draft tokens")
            for response in responses:
                if response["response"]["usage"]["prompt_tokens"] != args.input_tokens:
                    raise AssertionError("prompt token count does not match")
            sample = {
                "started_at_unix": started_at,
                "finished_at_unix": finished_at,
                "wall_seconds": elapsed,
                "clock_gap_seconds": finished_at - started_at - elapsed,
                "draft_tokens": draft_tokens,
                "tokens_per_second": concurrency * args.output_tokens / elapsed,
                "responses": responses,
            }
            (probe.output / f"bench-c{concurrency}-r{repeat}.json").write_text(
                json.dumps(sample, indent=2)
            )
            # On macOS the monotonic clock can exclude machine sleep. Keep
            # the raw sample, but never promote a suspended timing interval.
            if abs(sample["clock_gap_seconds"]) > 2:
                raise AssertionError("benchmark clock discontinuity or machine sleep")
            samples.append(sample["tokens_per_second"])
        results.append(
            {
                "concurrency": concurrency,
                "tps": samples,
                "median_tps": statistics.median(samples),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--tier", choices=["on", "off"], default="on")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument(
        "--acceptance-temperature",
        type=float,
        help="Marker-chat temperature; exact-token benchmarks keep API defaults",
    )
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument(
        "--acceptance-mode", choices=["marker", "replay"], default="marker"
    )
    parser.add_argument("--replay-output-tokens", type=int, default=128)
    parser.add_argument("--replay-control-repeats", type=int, default=2)
    parser.add_argument("--replay-logprobs", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--context-words", type=int, default=700)
    parser.add_argument(
        "--marker-format", choices=["plain", "quoted", "question-last"], default="plain"
    )
    parser.add_argument("--pressure-tokens", type=int, default=0)
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--source", default="perf/perf.md")
    args = parser.parse_args()
    if args.acceptance_only and args.benchmark_only:
        parser.error("choose acceptance-only or benchmark-only")
    for name in (
        "sessions",
        "rounds",
        "context_words",
        "input_tokens",
        "output_tokens",
        "replay_output_tokens",
        "replay_control_repeats",
        "repeats",
        "timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.pressure_tokens < 0 or any(c <= 0 for c in args.concurrency):
        parser.error("pressure-tokens must be nonnegative and concurrency positive")
    if args.acceptance_mode == "replay" and args.pressure_tokens:
        parser.error("replay currently requires reset eviction")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "harness.py").write_text(Path(__file__).read_text())
    machine = hardware.detect()
    summary = {
        "machine": asdict(machine),
        "arguments": vars(args).copy(),
        "profiles": [],
    }
    summary["arguments"]["output"] = str(args.output)
    failed = False
    for plan in resolve_profiles(machine, args.profile):
        item = {"profile": plan.profile_id, "passed": False, "acceptance": []}
        summary["profiles"].append(item)
        output = args.output / plan.profile_id
        output.mkdir(exist_ok=True)
        try:
            item["acceleration"] = validate_acceleration(plan)
            tiered = "kv_transfer_config" in plan.engine
            if args.acceptance_only and not tiered:
                item["skipped"] = "profile has no configured KV tier"
                continue
            if args.tier == "off":
                engine = dict(plan.engine)
                engine.pop("kv_transfer_config", None)
                plan = replace(plan, engine=engine)
            elif args.acceptance_only:
                plan = replace(plan, env={**plan.env, "VLLM_KV_TIER_VERIFY": "1"})
            if not args.benchmark_only and not args.pressure_tokens:
                # The local reset endpoint is registered only in dev mode.
                # Server binds loopback; this is acceptance instrumentation.
                plan = replace(plan, env={**plan.env, "VLLM_SERVER_DEV_MODE": "1"})
            fetch.ensure(plan, assume_yes=True)
            log_path = output / "server.log"
            with Server(plan) as server:
                server.start(str(log_path))
                server.wait_until_ready(timeout=args.timeout)
                probe = Probe(server, output, args.timeout)
                probe.chat_temperature = args.acceptance_temperature
                if tiered and not args.benchmark_only:
                    check = (
                        replay_acceptance
                        if args.acceptance_mode == "replay"
                        else acceptance
                    )
                    check(probe, log_path, args, item["acceptance"])
                if not args.acceptance_only:
                    item["benchmark"] = benchmark(probe, args)
            item["passed"] = True
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
            failed = True
        finally:
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(item), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
