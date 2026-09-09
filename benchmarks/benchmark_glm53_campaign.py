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
import importlib.metadata
import json
import os
import secrets
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


def diagnostic_only(args):
    return bool(
        args.routing
        or args.cuda_traces
        or args.quality_repeats > 1
        or os.environ.get("SLIMSERVE_GLM53_SCORE_JOURNAL")
        or os.environ.get("SLIMSERVE_GLM53_INDEX_JOURNAL")
        or any(
            os.environ.get(key) == "1"
            for key in (
                "SLIMSERVE_GLM53_MODEL_JOURNAL",
                "SLIMSERVE_GLM53_MOE_JOURNAL",
                "SLIMSERVE_GLM53_CANONICAL_MOE",
                "SLIMSERVE_GLM53_STABLE_ROUTE",
                "SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER",
                "SLIMSERVE_GLM53_CANONICAL_INDEX_TIES",
                "SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED",
            )
        )
    )


def benchmark_sources():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *(
            f"benchmarks/{name}.py"
            for name in (
                "benchmark_glm53_campaign",
                "benchmark_glm53_server",
                "benchmark_glm53_b12x",
                "benchmark_dsv4_exact",
                "benchmark_glm53_quality",
                "benchmark_glm53_prefill",
            )
        ),
        "slimserve/smoke.py",
        "slimserve/stream.py",
    ]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths
    }


# Freeze the client sources when Python imports the workload, not per boot.
# Editing a loaded module does not change the running function's bytecode.
LOADED_BENCHMARK_SOURCES = benchmark_sources()


def require_benchmark_sources():
    current = benchmark_sources()
    changed = sorted(
        name
        for name in current.keys() | LOADED_BENCHMARK_SOURCES.keys()
        if current.get(name) != LOADED_BENCHMARK_SOURCES.get(name)
    )
    if changed:
        raise RuntimeError(f"benchmark source changed after import: {changed}")
    return dict(LOADED_BENCHMARK_SOURCES)


def post(base: str, path: str, body: dict | None = None) -> bytes:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def request(
    base, model, prompt, output_tokens, seed, first_event=None, cache_salt=None
):
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
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
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
        "cache_salt": cache_salt,
    }


def cold_prefix_verified(result):
    cached = [
        (row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
        for row in result["requests"]
    ]
    return bool(cached) and all(type(value) is int and value == 0 for value in cached)


def round_requests(
    base, model, prompts, tokens, seed, profile=False, cold_prefix=False
):
    require_benchmark_sources()
    events = [threading.Event() for _ in prompts]
    salt = secrets.token_hex(16) if cold_prefix else None
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as pool:
        futures = [
            pool.submit(
                request,
                base,
                model,
                p,
                tokens,
                seed + i,
                events[i],
                f"{salt}:{i}" if salt is not None else None,
            )
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
        "cache_policy": "isolated-cold" if cold_prefix else "profile-default",
        "wall_seconds": wall,
        "aggregate_output_tps": sum(r["usage"]["completion_tokens"] for r in rows)
        / wall,
        "client_decode_tps": sum(r["tokens_after_first_chunk"] for r in rows)
        / decode_window,
        "ttft_mean_seconds": statistics.mean(r["ttft_seconds"] for r in rows),
        "requests": rows,
    }


def process_group_snapshot(pgid):
    """Record Linux process state without mistaking zombies for running workers."""
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm may contain spaces or parentheses; fields after its last ')'
            # begin with state, ppid and pgrp.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid:
            rows.append(
                {"pid": int(entry.name), "ppid": int(fields[1]), "state": fields[0]}
            )
    return sorted(rows, key=lambda row: row["pid"])


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
            return []
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # A parent can exit before its workers. Nsight may adopt terminated
        # workers without reaping them until this controller exits. Zombies
        # cannot execute; record them without waiting for their reaper. The
        # separate GPU-release gate must still wait for driver cleanup to
        # remove their compute-process entries. Live members need escalation.
        deadline = time.monotonic() + timeout
        while True:
            members = process_group_snapshot(process.pid)
            if all(row["state"] == "Z" for row in members):
                return members
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    raise RuntimeError(
        f"owned process group {process.pid} did not exit; "
        f"remaining processes: {process_group_snapshot(process.pid)}"
    )


def gpu_compute_pids():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    return sorted({int(line.strip()) for line in raw.splitlines() if line.strip()})


def wait_owned_gpu_release(owned_pids, evidence, timeout=30):
    """Bounded read-only wait for this run's driver entries, not foreign jobs."""
    owned_pids = set(owned_pids)
    evidence.update(status="waiting", owned_pids=sorted(owned_pids), samples=[])
    started = time.monotonic()
    while True:
        try:
            active = set(gpu_compute_pids())
        except BaseException as error:
            evidence.update(status="failed", error=repr(error))
            raise
        elapsed = time.monotonic() - started
        remaining = sorted(active & owned_pids)
        evidence["samples"].append(
            {
                "elapsed_seconds": elapsed,
                "owned_active_pids": remaining,
                "other_active_pids": sorted(active - owned_pids),
            }
        )
        if not remaining:
            evidence.update(status="complete", elapsed_seconds=elapsed)
            return
        if elapsed >= timeout:
            evidence.update(status="failed", elapsed_seconds=elapsed)
            raise RuntimeError(f"owned GPU processes did not release: {remaining}")
        time.sleep(0.25)


def require_gpus_free(receipt, record, boot):
    try:
        active = gpu_compute_pids()
        if active:
            raise RuntimeError(f"GPUs already have compute processes: {active}")
    except BaseException as error:
        receipt.update(status="failed", blocked_before_boot=boot, error=repr(error))
        record.write_text(json.dumps(receipt, indent=2) + "\n")
        raise


def finish_owned_run(process, run, receipt, record):
    # A teardown exception must not leave the on-disk run marked 'running',
    # lose completed quality/prefill results, or start another server.
    try:
        members = process_group_snapshot(process.pid)
        owned = {process.pid, *(row["pid"] for row in members)}
        run["teardown"] = {"status": "stopping", "initial_owned_processes": members}
        zombies = stop_owned(process)
        owned.update(row["pid"] for row in zombies)
        run["teardown"].update(
            returncode=process.returncode,
            remaining_zombies=zombies,
            gpu_release={},
        )
        wait_owned_gpu_release(owned, run["teardown"]["gpu_release"])
        run["teardown"]["status"] = "complete"
    except BaseException as error:
        run["status"] = "failed"
        receipt["status"] = "failed"
        run.setdefault("teardown", {}).update(status="failed", error=repr(error))
        raise
    finally:
        record.write_text(json.dumps(receipt, indent=2) + "\n")


def gpu_snapshot():
    fields = (
        "index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,"
        "utilization.gpu,clocks.current.graphics,clocks.current.memory,"
        "clocks.max.memory,pstate,power.limit,power.draw,temperature.gpu"
    )
    return subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"], text=True
    ).strip()


def runtime_identity():
    # Editable git identity alone does not identify rebuilt native extensions.
    # Include MoE, allocator and other root-level native extensions as well:
    # _C*.so alone does not identify the Marlin kernels in _moe_C*.so.
    native = sorted(Path("vllm").glob("*.so"))
    hashes = {}
    for path in native:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    versions = {}
    for package in (
        "torch",
        "triton",
        "transformers",
        "safetensors",
        "flashinfer-python",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "packages": versions,
        "native_sha256": hashes,
        "gpu_before_start": gpu_snapshot(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }


def observer_return_rounds(args, base, model, prompts, folder):
    """Repeat the exact pre-trace matrix; never merge these into the baseline."""
    rows = []
    for rep in range(args.repeats):
        for concurrency in args.concurrency:
            result = round_requests(
                base,
                model,
                prompts[concurrency],
                args.output_tokens,
                42,
                cold_prefix=getattr(args, "cold_prefix", False),
            )
            result["gpu_after_round"] = gpu_snapshot()
            result["exact"] = all(
                request["usage"]["prompt_tokens"] == args.input_tokens
                and request["usage"]["completion_tokens"] == args.output_tokens
                for request in result["requests"]
            ) and (
                not getattr(args, "cold_prefix", False) or cold_prefix_verified(result)
            )
            path = folder / f"observer-return-{rep + 1}-c{concurrency}.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            rows.append(
                {
                    "repeat": rep + 1,
                    "concurrency": concurrency,
                    "path": str(path),
                    **{
                        key: value for key, value in result.items() if key != "requests"
                    },
                }
            )
            if not result["exact"] or any(
                request["replacement_characters"] for request in result["requests"]
            ):
                raise ValueError(
                    "observer return token-count or replacement-character gate failed"
                )
    return rows


def quality_passes(args, base, model, model_dir, tokenizer, folder, run, save):
    """Repeat fixed scoring on one live model, preserving every partial pass.

    The first pass keeps the historical filename/summary fields. Additional
    passes are an explicit reproducibility diagnostic, not replacement scores.
    """
    from benchmark_glm53_quality import run as run_quality

    run["quality_passes"] = []
    for repeat in range(1, args.quality_repeats + 1):
        require_benchmark_sources()
        path = folder / (
            "quality.json" if repeat == 1 else f"quality-repeat-{repeat}.json"
        )
        row = {"repeat": repeat, "path": str(path), "status": "running"}
        run["quality_passes"].append(row)
        if repeat == 1:
            run["quality_path"] = str(path)
        save()
        try:
            quality = run_quality(
                argparse.Namespace(
                    url=base,
                    model=model,
                    tokenizer=str(model_dir),
                    source=args.source,
                    output=path,
                    pairs=32,
                    prefix_tokens=512,
                    score_tokens=128,
                    needle_contexts=[1024, 8192, 32768],
                    needle_positions=[0.25, 0.75],
                ),
                tokenizer,
            )
            row["summary"] = quality["summary"]
            if repeat == 1:
                run["quality"] = quality["summary"]
            if not quality["summary"]["all_needles_rank_first"]:
                raise ValueError("needle contrast gate failed")
            row["status"] = "complete"
        except BaseException as error:
            row.update(status="failed", error=repr(error))
            raise
        finally:
            save()


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
    ap.add_argument(
        "--cold-prefix",
        action="store_true",
        help="isolate every timing request's cache and require zero cached tokens",
    )
    profiler = ap.add_mutually_exclusive_group()
    profiler.add_argument("--traces", action="store_true")
    profiler.add_argument(
        "--cuda-traces",
        action="store_true",
        help="CUDA API ranges and return timings; run under Nsight, diagnostic only",
    )
    ap.add_argument(
        "--cuda-trace-concurrency",
        type=int,
        help="one CUDA range per model process (default 1); timing matrix is unchanged",
    )
    ap.add_argument(
        "--routing",
        action="store_true",
        help="diagnostic-only actual-step routing; timings are NOT a baseline",
    )
    ap.add_argument(
        "--quality",
        action="store_true",
        help="score exact continuations and needle contrasts after timed work",
    )
    ap.add_argument(
        "--quality-repeats",
        type=int,
        default=1,
        help="repeat quality without restarting; >1 is diagnostic only",
    )
    ap.add_argument(
        "--prefill",
        action="store_true",
        help="record cold-cache 32K/128K exact-ID TTFT after other workloads",
    )
    ap.add_argument(
        "--prefill-traces",
        action="store_true",
        help="with --prefill, trace additional cold requests after all TTFT timings",
    )
    args = ap.parse_args()
    if args.quality_repeats < 1 or (args.quality_repeats != 1 and not args.quality):
        ap.error("positive --quality-repeats requires --quality when greater than 1")
    if args.cuda_trace_concurrency is not None and not args.cuda_traces:
        ap.error("--cuda-trace-concurrency requires --cuda-traces")
    if args.cuda_traces:
        if args.cuda_trace_concurrency is None:
            args.cuda_trace_concurrency = 1
        if args.cuda_trace_concurrency not in args.concurrency:
            ap.error(
                "CUDA trace concurrency must be in the measured concurrency matrix"
            )
    if args.prefill_traces and not args.prefill:
        ap.error("--prefill-traces requires --prefill")
    if args.cuda_traces and (args.prefill_traces or args.routing):
        ap.error(
            "--cuda-traces cannot combine with Torch prefill traces or routing capture"
        )
    if min(args.boots, args.repeats, *args.concurrency) < 1 or args.output_tokens < 2:
        ap.error("boots, repeats and concurrency must be positive; output tokens >= 2")
    machine = hardware.detect()
    compatible = compatible_profile_ids(machine)
    if args.profile not in compatible:
        ap.error(f"profile not compatible; available: {compatible}")
    plan = registry.resolve(args.profile, machine.platform, machine.count, None)
    if args.prefill or args.cold_prefix:
        plan = dataclasses.replace(
            plan,
            engine={
                **plan.engine,
                "enable_prompt_tokens_details": True,
                "enable_per_request_metrics": True,
            },
        )
    if args.routing:
        from slimserve.routing_journal import diagnostic_plan

        diagnostic_plan(plan, args.output)  # Validate scope before any launch.
        print(
            "DIAGNOSTIC ROUTING: scheduling changes; throughput is NOT a baseline",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.read_text()
    tokenizer = get_tokenizer(str(plan.model_dir))
    prompts = {
        c: exact_prompts(tokenizer, source, c, args.input_tokens, 0, False)
        for c in args.concurrency
    }
    command = lambda *cmd: subprocess.check_output(cmd, text=True).strip()
    receipt = {
        "status": "running",
        "benchmark_implementation_sha256": require_benchmark_sources(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "plan": dataclasses.asdict(plan),
        "compatible_profiles": compatible,
        "command": sys.argv,
        "diagnostic_only": diagnostic_only(args),
        "throughput_is_baseline_eligible": not diagnostic_only(args),
        "cuda_profiler_ranges": args.cuda_traces,
        "runtime": runtime_identity(),
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
        require_benchmark_sources()
        require_gpus_free(receipt, record, boot)
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
        if args.traces or args.prefill_traces:
            argv += ["--torch-profile-dir", str(folder / "traces")]
        if args.cuda_traces:
            argv += ["--cuda-profile"]
        if args.routing:
            argv += ["--route-profile-dir", str(folder / "routing")]
        if args.prefill or args.cold_prefix:
            argv += ["--request-metrics"]
        run = {"boot": boot, "argv": argv, "status": "starting", "measurements": []}
        if args.routing:
            run["diagnostic_plan"] = dataclasses.asdict(
                diagnostic_plan(plan, folder / "routing")
            )
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
                    # Reach the same state-copy/cache/context boundaries as
                    # measured requests. A short decode warmup can miss JIT
                    # kernels needed only later in a request (GLM: 1088 tokens).
                    warmup = round_requests(
                        base,
                        model,
                        prompts[c],
                        args.output_tokens,
                        42,
                        cold_prefix=args.cold_prefix,
                    )
                    warmup_path = folder / f"warmup-c{c}.json"
                    warmup_path.write_text(json.dumps(warmup, indent=2) + "\n")
                    run["warmups"].append(str(warmup_path))
                    if args.cold_prefix and not cold_prefix_verified(warmup):
                        raise ValueError(
                            "warmup cold-prefix gate requires cached_tokens=0"
                        )
                for rep in range(args.repeats):
                    for c in args.concurrency:
                        result = round_requests(
                            base,
                            model,
                            prompts[c],
                            args.output_tokens,
                            42,
                            cold_prefix=args.cold_prefix,
                        )
                        result["gpu_after_round"] = gpu_snapshot()
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
                        if args.cold_prefix and not cold_prefix_verified(result):
                            raise ValueError(
                                "cold-prefix gate requires cached_tokens=0"
                            )
                        if not result["exact"] or any(
                            r["replacement_characters"] for r in result["requests"]
                        ):
                            raise ValueError(
                                "token-count or replacement-character gate failed"
                            )
                if args.traces or args.cuda_traces:
                    # Nsight repeated ranges can omit activities for graphs
                    # instantiated before the first range. Do not mutate the
                    # model's graphs to accommodate collection: one range per
                    # process, independent of the complete timing matrix.
                    profile_concurrencies = (
                        [args.cuda_trace_concurrency]
                        if args.cuda_traces
                        else [c for c in args.concurrency if c in (1, 8)]
                    )
                    for c in profile_concurrencies:
                        result = round_requests(
                            base,
                            model,
                            prompts[c],
                            384,
                            42,
                            profile=True,
                            cold_prefix=args.cold_prefix,
                        )
                        (folder / f"profile-c{c}.json").write_text(
                            json.dumps(result, indent=2) + "\n"
                        )
                        if args.cold_prefix and not cold_prefix_verified(result):
                            raise ValueError(
                                "profile cold-prefix gate requires cached_tokens=0"
                            )
                if args.cuda_traces:
                    run["observer_return"] = observer_return_rounds(
                        args, base, model, prompts, folder
                    )
                    record.write_text(json.dumps(receipt, indent=2) + "\n")
                if args.quality:
                    quality_passes(
                        args,
                        base,
                        model,
                        plan.model_dir,
                        tokenizer,
                        folder,
                        run,
                        lambda: record.write_text(json.dumps(receipt, indent=2) + "\n"),
                    )
                if args.prefill:
                    require_benchmark_sources()
                    from benchmark_glm53_prefill import run as run_prefill

                    prefill_path = folder / "prefill"
                    run["prefill_path"] = str(prefill_path)
                    prefill = run_prefill(
                        argparse.Namespace(
                            url=base,
                            model=model,
                            tokenizer=str(plan.model_dir),
                            source=args.source,
                            output=prefill_path,
                            contexts=[32768, 131072],
                            repeats=3,
                            warmups=1,
                            output_tokens=8,
                            traces=args.prefill_traces,
                        ),
                        tokenizer,
                    )
                    run["prefill"] = prefill["aggregates"]
                require_benchmark_sources()
                run["status"] = "complete"
            except Exception as error:
                run["status"] = "failed"
                run["error"] = repr(error)
                print(f"boot {boot}: {error}", flush=True)
            finally:
                finish_owned_run(process, run, receipt, record)
    receipt["aggregates"] = {}
    receipt["status"] = (
        "complete"
        if all(r["status"] == "complete" for r in receipt["runs"])
        else "failed"
    )
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
