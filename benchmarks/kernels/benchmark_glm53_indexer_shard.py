#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase4.5 complete TP4 prefill indexer: row ownership + pool-ID exchange.

Precedent: vLLM #54951, b466281a9ab20e483444736f85d650418ca40d44.
Adaptation: exchange 512 pool IDs before 4x expansion, once per forward, with
equal contiguous row ownership and padded all-gather. Preserve every arithmetic
kernel. Fixed 7616-row/32K and /128K windows, three A/B/A rounds x five repeats.
Synthetic replicated Q/cache/weights; not serving TPS or a model-quality result.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers import glm5_next_indexer as gi


def sharded_prefill(q, weights, ape, cache, chunks, output, group):
    """Probe implementation; chunks use global query offsets and original req IDs."""
    rows = q.shape[0]
    owned = (rows + group.world_size - 1) // group.world_size
    start = group.rank_in_group * owned
    stop = min(start + owned, rows)
    local = torch.full((owned, 512), -1, device=q.device, dtype=torch.int32)
    for chunk in chunks:
        lo, hi = max(start, chunk.start), min(stop, chunk.stop)
        if lo >= hi:
            continue
        offset = slice(lo - chunk.start, hi - chunk.start)
        logits = torch.empty((hi - lo, chunk.pools), device=q.device)
        local[lo - start : hi - start] = gi._pooled_topk(
            q[lo:hi],
            weights[lo:hi],
            ape,
            cache,
            chunk.block_table,
            chunk.row_req[offset],
            chunk.visible[offset],
            logits,
            chunk.pools,
            64,
            128**-0.5,
            512,
            4,
            by_request=True,
        )
    gathered = group.all_gather(local, dim=0)
    # Each original chunk keeps its own visibility/tail, including request edges.
    for chunk in chunks:
        selected = gathered[chunk.start : chunk.stop]
        gi._expand_topk_kernel[(chunk.stop - chunk.start,)](
            selected,
            chunk.visible,
            output[chunk.start : chunk.stop],
            selected.stride(0),
            KP=4,
            KSEL=512,
            OUT_W=2051,
            BLOCK_S=64,
        )


def baseline(q, weights, ape, cache, chunks, output):
    for c in chunks:
        rows = c.stop - c.start
        gi._pooled_select(
            q[c.start : c.stop],
            weights[c.start : c.stop],
            ape,
            cache,
            c.block_table,
            c.row_req,
            c.visible,
            torch.empty((rows, c.pools), device=q.device),
            c.pools,
            64,
            128**-0.5,
            512,
            output[c.start : c.stop],
            4,
            by_request=True,
        )


def make_case(rows, context, ragged=False):
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    q = torch.randn(rows, 32, 128, device=dev, dtype=torch.bfloat16)
    weights = torch.rand(rows, 32, device=dev) * (32**-0.5)
    ape = torch.randn(4, 128, device=dev)
    requests = 2 if ragged else 1
    blocks = (context + 63) // 64
    # A noncontiguous page view catches accidental flat-cache assumptions.
    cache = torch.randn(
        requests * blocks, 2, 64, 256, device=dev, dtype=torch.bfloat16
    )[:, 0]
    block_table = torch.arange(requests * blocks, device=dev, dtype=torch.int32).view(
        requests, blocks
    )
    visible = torch.arange(
        context - rows + 1, context + 1, device=dev, dtype=torch.int32
    )
    row_req = torch.zeros(rows, device=dev, dtype=torch.int32)
    if ragged:
        boundary = rows // 3
        row_req[boundary:] = 1
        visible[:boundary] = torch.arange(1, boundary + 1, device=dev)
    # The metadata's 512 MiB limit uses unpooled token extents; our pooled
    # score allocation is 4x narrower (128 MiB), matching recorded subchunks.
    chunk_rows = max(1, (128 * 1024 * 1024) // (4 * (context // 4)))
    chunks = [
        SimpleNamespace(
            start=i,
            stop=min(rows, i + chunk_rows),
            pools=context // 4,
            block_table=block_table,
            row_req=row_req[i : i + chunk_rows],
            visible=visible[i : i + chunk_rows],
        )
        for i in range(0, rows, chunk_rows)
    ]
    return q, weights, ape, cache, chunks


def measure(call, group):
    # Eager prefill: wall time includes launches, allocation and the real TP
    # collective. Synchronize each whole window, not its individual kernels.
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    dist.barrier(group=group.cpu_group)
    start = time.perf_counter()
    for _ in range(5):
        call()
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / 5
    values = [None] * group.world_size
    dist.all_gather_object(values, elapsed, group=group.cpu_group)
    return values


def compare_outputs(a, b, group):
    """Collect every rank's independent pool-set, tail and ordering checks."""
    local = {
        "pool_sets_equal": torch.equal(
            a[:, :2048].sort(1).values, b[:, :2048].sort(1).values
        ),
        "tail_equal": torch.equal(a[:, 2048:], b[:, 2048:]),
        "order_equal": torch.equal(a, b),
    }
    ranks = [None] * group.world_size
    dist.all_gather_object(ranks, local, group=group.cpu_group)
    return {
        "per_rank": ranks,
        "rank0_order_equal": ranks[0]["order_equal"],
        "sets_and_tail_exact_all_ranks": all(
            row["pool_sets_equal"] and row["tail_equal"] for row in ranks
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    if args.output.exists():
        parser.error("NEW output required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(2)
    torch.manual_seed(4501)
    assert torch.cuda.get_device_capability() == (12, 0)
    assert (gi._ROW_TILE, gi._POOL_TILE) == (2, 128)
    init_distributed_environment()
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=4)
    group = get_tp_group()
    assert group.world_size == 4
    result = {
        "status": "running",
        "roadmap": "4.5",
        "method": __doc__,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "indexer_sha256": hashlib.sha256(Path(gi.__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cases": [],
    }

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for rows, context, ragged in (
            (2177, 32768, True),
            (7616, 32768, False),
            (7616, 131072, False),
        ):
            data = make_case(rows, context, ragged)
            a = torch.empty(rows, 2051, device="cuda", dtype=torch.int32)
            b = torch.empty_like(a)

            def control(data=data, a=a):
                baseline(*data, a)

            def candidate(data=data, b=b):
                sharded_prefill(*data, b, group)

            control()
            candidate()
            # Existing selector has unordered atomic publication; require exact
            # selected sets AND exact tail, report ordering separately.
            row = {
                "rows": rows,
                "context": context,
                "ragged": ragged,
                **compare_outputs(a, b, group),
                "samples": [],
            }
            result["cases"].append(row)
            save()
            assert row["sets_and_tail_exact_all_ranks"], row["per_rank"]
            if ragged:
                data[0].mul_(0.5)
                control()
                candidate()
                row["changed_input"] = compare_outputs(a, b, group)
                save()
                assert row["changed_input"]["sets_and_tail_exact_all_ranks"], (
                    row["changed_input"]["per_rank"]
                )
            samples = row["samples"]
            if not ragged:
                for _ in range(3):
                    samples.append(
                        {
                            "before_ms": measure(control, group),
                            "candidate_ms": measure(candidate, group),
                            "after_ms": measure(control, group),
                        }
                    )
            if samples:
                row["median_slowest_rank_ms"] = {
                    name: statistics.median(max(s[name]) for s in samples)
                    for name in samples[0]
                }
            if rank == 0:
                print(json.dumps(row), flush=True)
            save()
            del data, a, b, control, candidate
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        save()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
