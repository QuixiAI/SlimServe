#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Archived rejected SM120 sparse-prefill experiments; diagnostic only.

BF16 Q/KV and the existing FP16 probability product are unchanged. Synthetic
queries use 512 distinct pooled groups (four contiguous tokens per group) from
a 32K/128K cache. All five experiment families completed on 2026-09-10 without
a useful gain. Preserve this reproducer and its raw results; do not resume the
sweep. See perf/optimization_status.md, "Close sparse-prefill local variants".
No serving dispatcher imports the experimental kernels.
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
    parser.add_argument("--split-heads", action="store_true")
    parser.add_argument("--fused-accumulator", action="store_true")
    parser.add_argument("--reload-query", action="store_true")
    parser.add_argument("--split-values", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.replays, *args.batch) < 1:
        parser.error("new output path and positive counts required")
    if any(c not in (32768, 131072) for c in args.context):
        parser.error("use the fixed 32K/128K cache cases")
    if (
        sum(
            (
                args.split_heads,
                args.fused_accumulator,
                args.reload_query,
                args.split_values,
            )
        )
        > 1
    ):
        parser.error("test one kernel hypothesis at a time")
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
                    implementation=pf._sparse_mla_prefill_kernel,
                    kernel_kwargs=None,
                ):
                    value_tile = (kernel_kwargs or {}).get("VALUE_TILE", 512) or 512
                    return implementation[(batch, 512 // value_tile)](
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
                        **(kernel_kwargs or {}),
                    )

                baseline_kernel = call(baseline_out, 32, 4, 2)
                torch.testing.assert_close(baseline_out, reference, rtol=0, atol=0)
                baseline = capture(partial(call, baseline_out, 32, 4, 2))
                if args.split_heads:
                    # Each head is independent. This isolated view uses the
                    # existing H16 kernel with duplicated request metadata;
                    # a serving implementation would use a second grid axis.
                    split_q = q.reshape(batch * 2, 16, 512)
                    split_bt = bt.repeat_interleave(2, dim=0)
                    split_indices = indices.repeat_interleave(2, dim=0)
                    split_lengths = lengths.repeat_interleave(2, dim=0)

                    def candidate_call(
                        out,
                        n,
                        warps,
                        stages,
                        q=split_q,
                        bt=split_bt,
                        indices=split_indices,
                        lengths=split_lengths,
                        kv=kv,
                    ):
                        return pf._sparse_mla_prefill_kernel[(q.shape[0],)](
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
                            H=16,
                            D=512,
                            BLOCK_N=n,
                            num_warps=warps,
                            num_stages=stages,
                        )

                    configs = [(32, 4, 2), (32, 4, 1), (64, 4, 1)]
                elif args.fused_accumulator or args.reload_query or args.split_values:
                    from benchmarks.kernels.glm53_sparse_prefill_fused import (
                        kernel as fused_kernel,
                    )

                    candidate_call = partial(
                        call,
                        implementation=fused_kernel,
                        kernel_kwargs={
                            "RELOAD_Q": args.reload_query,
                            "FUSE_ACC": args.fused_accumulator,
                            "VALUE_TILE": 256 if args.split_values else 0,
                        },
                    )
                    configs = (
                        [(32, 4, 1), (32, 4, 2)]
                        if args.split_values
                        else [(32, 4, 1), (32, 4, 2), (32, 4, 3)]
                        if args.reload_query
                        else [(32, 4, 2), (32, 8, 2), (64, 4, 1)]
                    )
                else:
                    candidate_call = call
                    configs = [(64, 4, 1), (64, 8, 1)]
                for n, warps, stages in configs:
                    row = {
                        "batch": batch,
                        "context": context,
                        "tile": [n, warps, stages],
                        "heads_per_block": 16 if args.split_heads else 32,
                        "value_tile": 256 if args.split_values else 512,
                        "baseline_resources": {
                            "registers": baseline_kernel.n_regs,
                            "spills": baseline_kernel.n_spills,
                            "shared": baseline_kernel.metadata.shared,
                        },
                    }
                    result["cases"].append(row)
                    try:
                        compiled = candidate_call(candidate_out, n, warps, stages)
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
                        registers=compiled.n_regs,
                        spills=compiled.n_spills,
                        shared=compiled.metadata.shared,
                    )
                    candidate = capture(
                        partial(candidate_call, candidate_out, n, warps, stages)
                    )
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
