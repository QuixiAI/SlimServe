#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed A/B/A test of node-profiler perturbation of an unchanged CUDA graph.

Diagnostic only: tiny one-GPU graphs cannot establish TP4 model behavior.
No per-replay synchronization; CUDA events bracket the whole replay batch.
All phase timings, correctness failures and node traces are retained.
Run under a memory cap on an exclusively owned CUDA device.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch


def run(args):
    if min(args.nodes, args.replays, args.rounds) < 1:
        raise ValueError("positive nodes, replays and rounds required")
    if args.nodes * args.replays >= 2**24:
        raise ValueError("FP32 integer correctness oracle would no longer be exact")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "running",
        "diagnostic_only": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,clocks.sm,clocks.mem",
                "--format=csv",
            ],
            text=True,
        ),
        "nodes_per_branch": args.nodes,
        "replays": args.replays,
        "rounds": args.rounds,
        "profiler": args.profiler,
        "phases": [],
    }
    path = args.output / "summary.json"
    profile_phase = (
        "torch-node-profile" if args.profiler == "torch" else "cuda-api-profile"
    )
    try:
        primary, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(primary):
            x = torch.zeros(1024, device="cuda")
            y = torch.zeros_like(x)
        primary.synchronize()
        for fork_join in (False, True):

            def body(fork_join=fork_join):
                for _ in range(args.nodes):
                    if fork_join:
                        auxiliary.wait_stream(primary)
                        with torch.cuda.stream(auxiliary):
                            y.add_(1)
                    x.add_(1)
                    if fork_join:
                        primary.wait_stream(auxiliary)

            with torch.cuda.stream(primary):
                body()
                primary.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=primary):
                    body()
                for _ in range(3):
                    graph.replay()
                primary.synchronize()
            for repeat in range(args.rounds):
                for phase in (
                    "unprofiled-before",
                    profile_phase,
                    "unprofiled-after",
                ):
                    row = {
                        "fork_join": fork_join,
                        "round": repeat + 1,
                        "phase": phase,
                        "status": "running",
                    }
                    receipt["phases"].append(row)
                    path.write_text(json.dumps(receipt, indent=2) + "\n")
                    profile = phase == profile_phase
                    profiler = (
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=False,
                            profile_memory=False,
                            with_stack=False,
                            with_flops=False,
                        )
                        if profile and args.profiler == "torch"
                        else torch.cuda.profiler.profile()
                        if profile
                        else nullcontext()
                    )
                    with torch.cuda.stream(primary):
                        x.zero_()
                        y.zero_()
                        primary.synchronize()
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        # Materialize events before the timed/profiler phase.
                        begin.record()
                        end.record()
                        end.synchronize()
                        with profiler:
                            wall_start = time.perf_counter()
                            begin.record()
                            for _ in range(args.replays):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            row["batch_wall_ms"] = (
                                time.perf_counter() - wall_start
                            ) * 1000
                            row["batch_cuda_ms"] = begin.elapsed_time(end)
                        row["cuda_us_per_replay"] = (
                            row["batch_cuda_ms"] * 1000 / args.replays
                        )
                        expected = args.nodes * args.replays
                        row["exact"] = bool(
                            torch.equal(x.cpu(), torch.full((1024,), float(expected)))
                            and torch.equal(
                                y.cpu(),
                                torch.full(
                                    (1024,), float(expected if fork_join else 0)
                                ),
                            )
                        )
                    if profile and args.profiler == "torch":
                        trace = (
                            args.output
                            / f"fork{int(fork_join)}-round{repeat + 1}.trace.json"
                        )
                        profiler.export_chrome_trace(str(trace))
                        row["trace"] = str(trace)
                    if not row["exact"]:
                        row["status"] = "failed"
                        raise AssertionError("changed-input replay correctness failed")
                    row["status"] = "complete"
                    path.write_text(json.dumps(receipt, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
        receipt["groups"] = [
            {
                "fork_join": fork_join,
                "phase": phase,
                "median_cuda_us_per_replay": statistics.median(
                    r["cuda_us_per_replay"]
                    for r in receipt["phases"]
                    if r["fork_join"] == fork_join and r["phase"] == phase
                ),
            }
            for fork_join in (False, True)
            for phase in ("unprofiled-before", profile_phase, "unprofiled-after")
        ]
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        raise
    finally:
        path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=1200)
    parser.add_argument("--replays", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--profiler", choices=("torch", "cuda"), default="torch")
    run(parser.parse_args())
