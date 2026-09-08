#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small-k sampler vs a stable-order FP32 PyTorch path; not serving TPS.

The unit suite additionally checks FP64 mass/analytic nucleus boundaries.
The timing reference here remains FP32, like the existing serving fallback;
different reduction orders need not produce bitwise-identical cutoffs.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import torch

from vllm.quixicore.ops import quixicore_ops


def reference(logits, k, p, noise):
    values, ids = logits.sort(dim=-1, stable=True)
    threshold = values.gather(1, (values.shape[1] - k.long()).unsqueeze(1))
    values = values.masked_fill(values < threshold, -float("inf"))
    discard = values.softmax(-1).cumsum(-1) <= 1 - p.unsqueeze(1)
    discard[:, -1] = False
    values = values.masked_fill(discard, -float("inf"))
    masked = torch.full_like(logits, -float("inf")).scatter(1, ids, values)
    return (masked.softmax(-1) / noise).argmax(-1)


def measure(fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    eager, replay = [], []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        eager.append((time.perf_counter() - start) * 1e6 / repeats)
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(repeats):
            graph.replay()
        b.record()
        b.synchronize()
        replay.append(a.elapsed_time(b) * 1000 / repeats)
    return {
        "eager_wall_us": eager,
        "graph_cuda_us": replay,
        "eager_median_us": statistics.median(eager),
        "graph_median_us": statistics.median(replay),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16])
    args = parser.parse_args()
    if min(args.repeats, *args.batch) < 1:
        parser.error("repeats and batch sizes must be positive")
    if args.output.exists():
        parser.error("output already exists")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs already have compute processes: {active}")
    native = next(Path("vllm").glob("_quixicore_C*.so"))
    with native.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "native_sha256": digest,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "sampling": {"top_k": 20, "top_p": 0.95, "noise": "fixed per vocabulary ID"},
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    for batch in args.batch:
        for case in ("normal", "bf16", "all_tied"):
            generator = torch.Generator(device="cuda").manual_seed(911)
            logits = torch.randn(batch, 154880, device="cuda", generator=generator) * 3
            if case == "bf16":
                logits = logits.bfloat16().float()
            if case == "all_tied":
                logits.zero_()
            k = torch.full((batch,), 20, device="cuda", dtype=torch.int32)
            p = torch.full((batch,), 0.95, device="cuda")
            noise = torch.empty_like(logits).exponential_(generator=generator)
            native_fn = partial(quixicore_ops.topk_sample, logits, k, p, noise)
            ref_fn = partial(reference, logits, k, p, noise)
            row = {
                "batch": batch,
                "case": case,
                "correct": torch.equal(native_fn(), ref_fn()),
            }
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            assert row["correct"], (batch, case)
            row["native"] = measure(native_fn, args.repeats)
            row["reference"] = measure(ref_fn, args.repeats)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
