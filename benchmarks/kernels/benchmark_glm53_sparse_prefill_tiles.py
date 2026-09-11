#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Screen existing sparse-MLA prefill tiles on SM120, not serving performance.

BF16 Q/KV and the existing FP16 probability product are unchanged. Synthetic
queries use 512 distinct pooled groups (four contiguous tokens per group) from
a 32K/128K cache. Compare N64/one-stage candidates with the installed N32/two-
stage kernel. The previously rejected N32/eight-warp variant is not repeated.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_marlin_schedule import measure
from vllm.v1.attention.backends.mla import quixicore_mla_sparse_prefill as pf


def capture(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[2048, 7616])
    parser.add_argument("--context", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.replays, *args.batch) < 1:
        parser.error("new output path and positive counts required")
    if any(c not in (32768, 131072) for c in args.context):
        parser.error("use the fixed 32K/128K cache cases")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs busy: {active}")
    if torch.cuda.get_device_capability() != (12, 0):
        parser.error("SM120 required")
    torch.manual_seed(91053)
    result = {
        "status": "running",
        "diagnostic_only": True,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        for batch in args.batch:
            for context in args.context:
                q = (torch.randn(batch, 32, 512, device="cuda") * 0.2).bfloat16()
                kv = (
                    torch.randn(context // 64, 64, 512, device="cuda") * 0.5
                ).bfloat16()
                bt = torch.arange(context // 64, device="cuda", dtype=torch.int32)
                bt = bt[None].repeat(batch, 1)
                # Odd strides permute a power-of-two pool domain, guaranteeing
                # distinct groups without a giant random ranking matrix.
                stride = torch.randint(context // 8, (batch, 1), device="cuda") * 2 + 1
                offset = torch.randint(context // 4, (batch, 1), device="cuda")
                group = (torch.arange(512, device="cuda")[None] * stride + offset) % (
                    context // 4
                )
                indices = (
                    (group[:, :, None] * 4 + torch.arange(4, device="cuda"))
                    .reshape(batch, 2048)
                    .int()
                )
                lengths = torch.full((batch,), 2048, device="cuda", dtype=torch.int32)
                reference = pf.sparse_mla_prefill_nope(
                    q, kv, bt, indices, lengths, 64, 1 / math.sqrt(512)
                )
                baseline_out = torch.empty_like(q)
                candidate_out = torch.empty_like(q)

                def call(
                    out,
                    n,
                    warps,
                    stages,
                    batch=batch,
                    q=q,
                    kv=kv,
                    bt=bt,
                    indices=indices,
                    lengths=lengths,
                ):
                    return pf._sparse_mla_prefill_kernel[(batch,)](
                        q,
                        kv,
                        bt,
                        indices,
                        lengths,
                        out,
                        2048,
                        bt.shape[1],
                        kv.stride(0),
                        64,
                        1 / math.sqrt(512),
                        H=32,
                        D=512,
                        BLOCK_N=n,
                        num_warps=warps,
                        num_stages=stages,
                    )

                baseline_kernel = call(baseline_out, 32, 4, 2)
                torch.testing.assert_close(baseline_out, reference, rtol=0, atol=0)
                baseline = capture(partial(call, baseline_out, 32, 4, 2))
                for warps in (4, 8):
                    row = {
                        "batch": batch,
                        "context": context,
                        "tile": [64, warps, 1],
                        "baseline_resources": {
                            "registers": baseline_kernel.n_regs,
                            "spills": baseline_kernel.n_spills,
                            "shared": baseline_kernel.metadata.shared,
                        },
                    }
                    result["cases"].append(row)
                    try:
                        kernel = call(candidate_out, 64, warps, 1)
                    except torch.OutOfMemoryError:
                        raise
                    except Exception as error:
                        if "out of resource" not in str(error).lower():
                            raise
                        row.update(status="unsupported", error=str(error))
                        save()
                        continue
                    torch.testing.assert_close(
                        candidate_out, reference, rtol=0.01, atol=0.016
                    )
                    delta = candidate_out.float() - reference.float()
                    row.update(
                        max_abs=delta.abs().max().item(),
                        normalized_rms=(
                            delta.square().mean().sqrt()
                            / reference.float().square().mean().sqrt()
                        ).item(),
                        registers=kernel.n_regs,
                        spills=kernel.n_spills,
                        shared=kernel.metadata.shared,
                    )
                    candidate = capture(partial(call, candidate_out, 64, warps, 1))
                    row["samples"] = [
                        {
                            name: measure(graph, 1, args.replays)
                            for name, graph in (
                                ("a_before_us", baseline),
                                ("b_us", candidate),
                                ("a_after_us", baseline),
                            )
                        }
                        for _ in range(args.rounds)
                    ]
                    row["paired_speedup"] = statistics.median(
                        (s["a_before_us"] + s["a_after_us"]) / (2 * s["b_us"])
                        for s in row["samples"]
                    )
                    row["status"] = "complete"
                    print(json.dumps(row), flush=True)
                    save()
                    del candidate
                del baseline
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
